from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import ZeroCone
from trajopt.constraints.constraint_list import BuiltConstraintList
from trajopt.costs.objective import Objective
from trajopt.dynamics.base import AbstractModel
from trajopt.expansions import Expansion
from trajopt.problem import MPCState, Problem
from trajopt.solvers.ilqr import build_warm_start, ilqr_solve
from trajopt.solvers.options import SolverOptions, TerminationStatus, to_solver_status
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z
from trajopt.transcription.result import SolverStatus

_EMPTY = np.zeros(0, dtype=np.float64)


class ALConstraints(eqx.Module):
    """Padded per-knot, per-row augmented-Lagrangian duals and penalties.

    Stores lambda using Altro's **non-conic** sign convention (reference Finding E):
    `lambda_bar = lambda + mu * c`, with penalty cost `+ lambda' c`. This is neither Altro's
    conic convention (`lambda_bar = lambda - mu * c`) nor the convention the code this module
    replaces used (shifting by `c + lambda / mu` and projecting onto the dual cone). Ticket 31's
    conic path must convert explicitly at its boundary rather than reinterpret this field.

    Rows are laid out per knot as `[constraint rows (0 .. p_cons_max) | x_upper(n) | x_lower(n) |
    u_upper(m) | u_lower(m)]`, padded to a fixed `p_max = p_cons_max + 2n + 2m` across every knot.
    `row_mask` marks which rows are structurally real at that knot (fewer constraint rows than
    `p_cons_max`, an infinite bound, or a terminal knot's absent control rows); masked rows carry
    `lam = 0` and are excluded from every reduction (cost, gradient, Hessian, violation, updates).

    Parameters
    ----------
    lam : jax.Array
        Padded dual multipliers of shape (N, p_max).
    mu : jax.Array
        Padded penalty parameters of shape (N, p_max).
    row_mask : jax.Array
        Boolean mask of shape (N, p_max), True where the row is a real constraint/bound row.
    is_equality : jax.Array
        Boolean mask of shape (N, p_max), True where the row maps to Altro's `Equality`
        (`ZeroCone`); False maps to `Inequality` (`NegativeOrthant`, including every box bound).
    p_cons_max : int
        Padded width of the constraint-only block (excludes the box-bound rows).
    """

    lam: jax.Array
    mu: jax.Array
    row_mask: jax.Array
    is_equality: jax.Array
    p_cons_max: int = eqx.field(static=True)

    @classmethod
    def build(cls, constraints: BuiltConstraintList, penalty_initial: float = 1.0) -> "ALConstraints":
        """Allocate a fresh padded AL layout for `constraints`, lambda=0 and mu=penalty_initial on real rows.

        Structural (row_mask, is_equality) computation is eager Python/NumPy over `constraints`,
        run once when a Problem is set up rather than under trace, since it depends only on
        constraint structure, not on any trajectory.
        """
        N = constraints.N
        n = constraints.n
        m = constraints.m
        p_cons_max = max(constraints.p) if constraints.p else 0

        row_mask_cons = np.zeros((N, p_cons_max), dtype=bool)
        is_eq_cons = np.zeros((N, p_cons_max), dtype=bool)
        for k, ev in enumerate(constraints.knot_evaluators):
            off = 0
            for c in ev.constraints:
                p_c = c.p
                is_eq_cons[k, off : off + p_c] = isinstance(c.cone, ZeroCone)
                off += p_c
            row_mask_cons[k, : ev.p] = True

        x_upper = np.asarray(constraints.x_upper)
        x_lower = np.asarray(constraints.x_lower)
        u_upper_pad = np.concatenate([np.asarray(constraints.u_upper), np.full((1, m), np.inf)], axis=0)
        u_lower_pad = np.concatenate([np.asarray(constraints.u_lower), np.full((1, m), -np.inf)], axis=0)

        row_mask_bound = np.concatenate(
            [np.isfinite(x_upper), np.isfinite(x_lower), np.isfinite(u_upper_pad), np.isfinite(u_lower_pad)],
            axis=-1,
        )
        is_eq_bound = np.zeros((N, 2 * n + 2 * m), dtype=bool)

        row_mask = jnp.asarray(np.concatenate([row_mask_cons, row_mask_bound], axis=-1))
        is_equality = jnp.asarray(np.concatenate([is_eq_cons, is_eq_bound], axis=-1))

        p_max = p_cons_max + 2 * n + 2 * m
        lam = jnp.zeros((N, p_max), dtype=jnp.float64)
        mu = jnp.where(row_mask, jnp.asarray(penalty_initial, dtype=jnp.float64), 0.0)

        return cls(lam=lam, mu=mu, row_mask=row_mask, is_equality=is_equality, p_cons_max=p_cons_max)


def _evaluate_constraint_block(
    constraints: BuiltConstraintList,
    X: jax.Array,
    U: jax.Array,
    T: jax.Array,
    p_cons_max: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Scatter each structural constraint group's evaluate/jacobian into a padded (N, p_cons_max, ...) block.

    Iterates `constraints.groups`, a tuple bounded by the number of structurally distinct knots
    (typically O(1): stage vs. terminal), not by N -- each group is filled with a single `vmap`
    over its knot indices, so this loop does not unroll per knot under trace.
    """
    N, n = X.shape
    m = U.shape[1]
    dtype = X.dtype

    C = jnp.zeros((N, p_cons_max), dtype=dtype)
    Jx = jnp.zeros((N, p_cons_max, n), dtype=dtype)
    Ju = jnp.zeros((N, p_cons_max, m), dtype=dtype)

    for g in constraints.groups:
        p_g = g.evaluator.p
        if p_g == 0:
            continue
        knots = jnp.asarray(g.knots)
        c_g = g.evaluate(X, U, T)
        jx_g, ju_g = g.jacobian(X, U, T)
        C = C.at[knots, :p_g].set(c_g)
        Jx = Jx.at[knots, :p_g, :].set(jx_g)
        if not g.evaluator.is_terminal:
            Ju = Ju.at[knots, :p_g, :].set(ju_g)

    return C, Jx, Ju


def _evaluate_bound_block(
    constraints: BuiltConstraintList,
    X: jax.Array,
    U: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate box-bound residual rows [x_upper | x_lower | u_upper | u_lower] and constant Jacobians.

    Pure vectorized array ops over all N knots at once (no python loop, no vmap needed); infinite
    bounds are replaced with a safe finite stand-in since `ALConstraints.row_mask` excludes those
    rows from every reduction regardless of the (otherwise meaningless) residual value there.
    """
    N, n = X.shape
    m = U.shape[1]
    dtype = X.dtype

    x_upper, x_lower = constraints.x_upper, constraints.x_lower
    U_pad = jnp.concatenate([U, jnp.zeros((1, m), dtype=dtype)], axis=0)
    u_upper_pad = jnp.concatenate([constraints.u_upper, jnp.full((1, m), jnp.inf, dtype=dtype)], axis=0)
    u_lower_pad = jnp.concatenate([constraints.u_lower, jnp.full((1, m), -jnp.inf, dtype=dtype)], axis=0)

    x_upper_safe = jnp.where(jnp.isfinite(x_upper), x_upper, 0.0)
    x_lower_safe = jnp.where(jnp.isfinite(x_lower), x_lower, 0.0)
    u_upper_safe = jnp.where(jnp.isfinite(u_upper_pad), u_upper_pad, 0.0)
    u_lower_safe = jnp.where(jnp.isfinite(u_lower_pad), u_lower_pad, 0.0)

    C_bound = jnp.concatenate(
        [X - x_upper_safe, x_lower_safe - X, U_pad - u_upper_safe, u_lower_safe - U_pad],
        axis=-1,
    )

    I_n = jnp.eye(n, dtype=dtype)
    I_m = jnp.eye(m, dtype=dtype)
    Jx_row = jnp.concatenate([I_n, -I_n, jnp.zeros((2 * m, n), dtype=dtype)], axis=0)
    Ju_row = jnp.concatenate([jnp.zeros((2 * n, m), dtype=dtype), I_m, -I_m], axis=0)
    Jx_bound = jnp.broadcast_to(Jx_row, (N, 2 * n + 2 * m, n))
    Ju_bound = jnp.broadcast_to(Ju_row, (N, 2 * n + 2 * m, m))

    return C_bound, Jx_bound, Ju_bound


def evaluate_al_constraints(
    al: ALConstraints,
    constraints: BuiltConstraintList,
    model: AbstractModel | None,
    traj: Trajectory,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate padded per-knot constraint and box-bound residuals and Jacobians in error coordinates.

    Parameters
    ----------
    al : ALConstraints
        AL layout providing `p_cons_max`.
    constraints : BuiltConstraintList
        Built constraint list holding constraint groups and box-bound limits.
    model : AbstractModel | None
        Model defining error-state coordinates. None means Euclidean (G = I).
    traj : Trajectory
        Trajectory holding stacked states X, controls U, and times t.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        C of shape (N, p_max), dc/dx in error coordinates of shape (N, p_max, ne), and dc/du of
        shape (N, p_max, m).
    """
    X, U, T = traj.X, traj.U, traj.t
    C_cons, Jx_cons, Ju_cons = _evaluate_constraint_block(constraints, X, U, T, al.p_cons_max)
    C_bound, Jx_bound, Ju_bound = _evaluate_bound_block(constraints, X, U)

    C = jnp.concatenate([C_cons, C_bound], axis=-1)
    Jx = jnp.concatenate([Jx_cons, Jx_bound], axis=1)
    Ju = jnp.concatenate([Ju_cons, Ju_bound], axis=1)

    n = X.shape[1]
    dtype = X.dtype
    if model is not None:
        G_all = jax.vmap(model.errstate_jacobian)(X)
    else:
        G_all = jnp.broadcast_to(jnp.eye(n, dtype=dtype), (X.shape[0], n, n))
    Jx_err = jnp.einsum("kpn,kne->kpe", Jx, G_all)

    return C, Jx_err, Ju


def _active_penalty(al: ALConstraints, C: jax.Array) -> jax.Array:
    """Row-wise active penalty `a`: `mu` where `(equality | c >= 0 | lam > 0) & row_mask`, else 0.

    Reduces to Altro's `Equality` branch (`a = mu` unconditionally) since `is_equality` forces the
    active test true, and to its `Inequality` branch (`a = ((c>=0)|(lam>0)) * mu`) otherwise.
    """
    active = (al.is_equality | (C >= 0.0) | (al.lam > 0.0)) & al.row_mask
    return jnp.where(active, al.mu, 0.0)


def _lam_bar(al: ALConstraints, C: jax.Array, a: jax.Array) -> jax.Array:
    """Row-wise `lambda_bar = lambda + a * c`, masked rows contributing lambda = 0."""
    lam_masked = jnp.where(al.row_mask, al.lam, 0.0)
    return jnp.where(al.row_mask, lam_masked + a * C, 0.0)


def al_cost(al: ALConstraints, C: jax.Array) -> jax.Array:
    """Augmented Lagrangian penalty cost, Altro's `alcost`: sum of `lambda' c + 0.5 * a * c^2`.

    Parameters
    ----------
    al : ALConstraints
        Current duals and penalties.
    C : jax.Array
        Padded constraint residuals of shape (N, p_max), from `evaluate_al_constraints`.

    Returns
    -------
    jax.Array
        Scalar penalty cost, masked rows contributing 0.
    """
    a = _active_penalty(al, C)
    lam_masked = jnp.where(al.row_mask, al.lam, 0.0)
    per_row = jnp.where(al.row_mask, lam_masked * C + 0.5 * a * C**2, 0.0)
    return jnp.sum(per_row)


def al_grad_hess(
    al: ALConstraints,
    C: jax.Array,
    Jx_err: jax.Array,
    Ju: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Augmented Lagrangian gradient and Gauss-Newton Hessian blocks, Altro's `algrad!` / `alhess!`.

    The Hessian is `J' diag(a) J` (Gauss-Newton), not `jax.hessian` of the penalty: exact for
    affine constraints, PSD by construction for nonlinear ones (reference §7.1).

    Parameters
    ----------
    al : ALConstraints
        Current duals and penalties.
    C : jax.Array
        Padded constraint residuals of shape (N, p_max).
    Jx_err : jax.Array
        Padded state Jacobian in error coordinates, shape (N, p_max, ne).
    Ju : jax.Array
        Padded control Jacobian, shape (N, p_max, m).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
        `(grad_x, grad_u, Hxx, Huu, Hux)` of shapes (N, ne), (N, m), (N, ne, ne), (N, m, m),
        (N, m, ne). `grad_u`, `Huu`, `Hux` are structurally zero at the terminal knot.
    """
    a = _active_penalty(al, C)
    lam_bar = _lam_bar(al, C, a)

    grad_x = jnp.einsum("kpe,kp->ke", Jx_err, lam_bar)
    grad_u = jnp.einsum("kpm,kp->km", Ju, lam_bar)
    Hxx = jnp.einsum("kpe,kp,kpf->kef", Jx_err, a, Jx_err)
    Huu = jnp.einsum("kpm,kp,kpn->kmn", Ju, a, Ju)
    Hux = jnp.einsum("kpm,kp,kpe->kme", Ju, a, Jx_err)
    return grad_x, grad_u, Hxx, Huu, Hux


def add_al_expansion(
    expansion: Expansion,
    al: ALConstraints,
    C: jax.Array,
    Jx_err: jax.Array,
    Ju: jax.Array,
) -> Expansion:
    """Add augmented Lagrangian gradient and Hessian contributions into an existing Expansion."""
    grad_x, grad_u, Hxx, Huu, Hux = al_grad_hess(al, C, Jx_err, Ju)
    return Expansion(
        A=expansion.A,
        B=expansion.B,
        q=expansion.q + grad_x,
        r=expansion.r + grad_u[:-1],
        Q=expansion.Q + Hxx,
        R=expansion.R + Huu[:-1],
        H=expansion.H + Hux[:-1],
    )


def dual_update(al: ALConstraints, C: jax.Array, options: SolverOptions) -> ALConstraints:
    """Update lambda, Altro's `dualupdate!`: equality `lam <- lam + mu*c`, inequality `lam <- max(0, lam + mu*c)`.

    Uses the full `mu`, not the active-gated `a` (Altro's dual update always applies the full
    penalty). Both branches are then clamped to `+-dual_max`, matching the unconditional clamp
    Altro applies after the cone-specific branch. Masked rows stay at lambda = 0.
    """
    raw = al.lam + al.mu * C
    updated = jnp.where(al.is_equality, raw, jnp.maximum(0.0, raw))
    clamped = jnp.clip(updated, -options.dual_max, options.dual_max)
    new_lam = jnp.where(al.row_mask, clamped, 0.0)
    return eqx.tree_at(lambda a: a.lam, al, new_lam)


def penalty_update(al: ALConstraints, options: SolverOptions) -> ALConstraints:
    """Update mu, Altro's `penaltyupdate!`: `mu <- clamp(mu * penalty_scaling, 0, penalty_max)`.

    Applied unconditionally to every real row (no active-set gating); masked rows are left frozen,
    keeping them inert rather than accumulating scaling that is never read.
    """
    scaled = jnp.clip(al.mu * options.penalty_scaling, 0.0, options.penalty_max)
    new_mu = jnp.where(al.row_mask, scaled, al.mu)
    return eqx.tree_at(lambda a: a.mu, al, new_mu)


def max_violation(al: ALConstraints, C: jax.Array) -> jax.Array:
    """Max constraint violation, Altro's `max_violation`: infinity-norm of `Pi_K(c) - c` over real rows.

    Equality rows project to 0 (`viol = -c`); inequality rows project onto the negative orthant
    (`viol = min(0, c) - c = -max(0, c)`).
    """
    viol = jnp.where(al.is_equality, -C, -jnp.maximum(0.0, C))
    viol_masked = jnp.where(al.row_mask, jnp.abs(viol), 0.0)
    return jnp.max(viol_masked)


def max_penalty(al: ALConstraints) -> jax.Array:
    """Max penalty parameter over real rows, Altro's `max_penalty`; 0 if there are no real rows."""
    mu_masked = jnp.where(al.row_mask, al.mu, -jnp.inf)
    has_rows = jnp.any(al.row_mask)
    return jnp.where(has_rows, jnp.max(mu_masked), 0.0)


class _ALObjective(eqx.Module):
    """Adapts (base objective, constraints, model, duals) into iLQR's `.cost(traj)` duck type.

    `ilqr_solve`'s line search calls `obj.cost` and nothing else (ticket 27), so adding the AL
    penalty cost here is the entire "different cost function" half of the composition seam
    (reference §5.1, ALObjective): the inner solver never learns a constraint exists.
    """

    obj: Objective
    constraints: BuiltConstraintList
    model: AbstractModel
    al: ALConstraints

    def cost(self, traj: Trajectory) -> jax.Array:
        """Add the augmented-Lagrangian penalty cost `al_cost` onto the base unconstrained cost."""
        base = self.obj.cost(traj)
        C, _Jx, _Ju = evaluate_al_constraints(self.al, self.constraints, self.model, traj)
        return base + al_cost(self.al, C)


class _ALProblem(eqx.Module):
    """Adapts `(Problem, duals)` into `ilqr_solve`'s duck-typed input -- the whole composition seam.

    `dynamics_expansion` passes through `problem` unchanged; `obj` and `cost_expansion` add the
    AL penalty cost and its Gauss-Newton gradient/Hessian (ticket 28) on top of the base
    objective. This is the only new code `ilqr_solve` sees: nothing in `ilqr.py` changes
    (reference §5.1, ticket 29's central constraint).
    """

    problem: Problem
    al: ALConstraints

    @property
    def model(self) -> AbstractModel:
        """Delegate to the wrapped problem's model; unconstrained dynamics are untouched."""
        return self.problem.model

    @property
    def obj(self) -> _ALObjective:
        """Build the AL-augmented cost-function adapter fresh from the current duals."""
        return _ALObjective(self.problem.obj, self.problem.constraints, self.problem.model, self.al)

    def dynamics_expansion(self, traj: Trajectory) -> Expansion:
        """Pass through the wrapped problem's unconstrained dynamics expansion."""
        return self.problem.dynamics_expansion(traj)

    def cost_expansion(self, traj: Trajectory) -> Expansion:
        """Add the AL penalty's gradient/Hessian onto the base cost expansion, via `add_al_expansion`."""
        base = self.problem.cost_expansion(traj)
        C, Jx_err, Ju = evaluate_al_constraints(self.al, self.problem.constraints, self.problem.model, traj)
        return add_al_expansion(base, self.al, C, Jx_err, Ju)


class ALStats(eqx.Module):
    """Fixed-size pytree of per-outer-iteration AL solve statistics, traceable end to end.

    Mirrors `SolverStats` (ticket 24): preallocated at `options.iterations_outer`, written with
    `.at[i].set(...)` under trace, trimmed only at the eager boundary. Reference §8.2 row 15
    requires the whole `(cost, c_max, penalty_max)` history, not just the final values.

    Parameters
    ----------
    iterations : jax.Array
        Number of completed outer iterations so far, as an int32 scalar.
    cost : jax.Array
        AL-augmented cost history of shape `(options.iterations_outer,)`.
    c_max : jax.Array
        Max constraint violation history of shape `(options.iterations_outer,)`.
    penalty_max : jax.Array
        Max AL penalty history of shape `(options.iterations_outer,)`.
    """

    iterations: jax.Array
    cost: jax.Array
    c_max: jax.Array
    penalty_max: jax.Array

    @classmethod
    def create(cls, options: SolverOptions) -> "ALStats":
        """Allocate zeroed history buffers of length `options.iterations_outer`."""
        history = jnp.zeros(options.iterations_outer, dtype=jnp.float64)
        return cls(iterations=jnp.asarray(0, dtype=jnp.int32), cost=history, c_max=history, penalty_max=history)


def _trim_al_stats(stats: ALStats, n_iter: int) -> ALStats:
    """Slice a finished AL solve's fixed-size stats buffers down to the completed iteration count."""
    return ALStats(
        iterations=stats.iterations,
        cost=stats.cost[:n_iter],
        c_max=stats.c_max[:n_iter],
        penalty_max=stats.penalty_max[:n_iter],
    )


class ALCarry(NamedTuple):
    """Traced `lax.while_loop` state for one AL solve; see `al_solve` for field meaning."""

    i: jax.Array
    trajectory: Trajectory
    al: ALConstraints
    stats: ALStats
    done: jax.Array
    status: jax.Array


def _evaluate_al_convergence(
    c_max: jax.Array,
    mu_max: jax.Array,
    inner_iterations: jax.Array,
    iter_num: jax.Array,
    options: SolverOptions,
) -> tuple[jax.Array, jax.Array]:
    """Decide the outer-loop status and whether to stop, matching `al_solve.jl`'s `evaluate_convergence`.

    Reproduces the four independent `if`s from reference §5.5 as a **last-match-wins** sequence
    of overwrites (finding A), not an ordered first-match list: a later check silently overwrites
    an earlier one's `status`, so converging on the same outer iteration that also exhausts
    `iterations_outer` reports `MAX_ITERATIONS_OUTER`, not `SOLVE_SUCCEEDED`.

    `kickout_max_penalty` (finding B) is Altro's own broken branch, ported as clearly intended:
    `mu_max >= options.penalty_max` ends the loop ("converged") without writing `status` at all,
    so `status` can still be `UNSOLVED` when kickout is the only thing that fired. That is a
    deliberate divergence from a clean "converged implies SOLVE_SUCCEEDED" API -- Altro's own
    branch throws (`solver.stats.penalty_max[i]` references an undefined `i`) and so cannot be
    parity-tested against Julia; ticket 33's ALTRO polish phase is what turns this early exit
    into `SOLVE_SUCCEEDED` once projected Newton drives the violation under tolerance.

    Parameters
    ----------
    c_max : jax.Array
        Max constraint violation for this outer iteration.
    mu_max : jax.Array
        Max AL penalty for this outer iteration.
    inner_iterations : jax.Array
        Completed iLQR iteration count from this outer iteration's inner solve.
    iter_num : jax.Array
        Completed outer iteration count (1-indexed) including this iteration.
    options : SolverOptions
        Supplies `constraint_tolerance`, `kickout_max_penalty`, `penalty_max`, `iterations`,
        `iterations_outer`.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        `(status, done)`: the `TerminationStatus` ordinal (possibly still `UNSOLVED`) as an int32
        scalar, and whether the outer loop should stop this iteration, as a bool scalar.
    """
    status = jnp.int32(TerminationStatus.UNSOLVED)

    converged_violation = c_max < options.constraint_tolerance
    status = jnp.where(converged_violation, jnp.int32(TerminationStatus.SOLVE_SUCCEEDED), status)

    kickout = jnp.asarray(options.kickout_max_penalty) & (mu_max >= options.penalty_max)

    max_iters_hit = inner_iterations >= options.iterations
    status = jnp.where(max_iters_hit, jnp.int32(TerminationStatus.MAX_ITERATIONS), status)

    max_outer_hit = iter_num >= options.iterations_outer
    status = jnp.where(max_outer_hit, jnp.int32(TerminationStatus.MAX_ITERATIONS_OUTER), status)

    done = converged_violation | kickout | max_iters_hit | max_outer_hit
    return status, done


def _al_step(carry: ALCarry, problem: Problem, options: SolverOptions) -> ALCarry:
    """One augmented-Lagrangian outer iteration, matching `al_solve.jl`'s per-iteration body.

    Follows reference §5.5's order exactly: effective (possibly intermediate) tolerances, solve
    the inner iLQR on the AL-augmented objective (`_ALProblem`), break on an ordinal inner-status
    failure (finding C) before anything else runs, then cost/violation/penalty maxima, record,
    check outer convergence (`_evaluate_al_convergence`), and only when neither break fired: dual
    update, penalty update. `ilqr_solve`'s own unconditional open-loop re-rollout each call *is*
    Altro's `initialize!` inside every `solve!(ilqr)` (reference §4.1) -- carrying the previous
    outer iteration's accepted trajectory in as the next call's warm start reproduces Altro's
    per-outer-iteration reset without any change to `ilqr.py`.

    The effective `(cost_tolerance, gradient_tolerance)` pair is computed here as traced scalars
    and passed straight to `ilqr_solve`'s override kwargs, not `dataclasses.replace`-d into a new
    `SolverOptions` (ticket 29: `options` stays frozen and untraced end to end).
    """
    i = carry.i
    is_last = i == jnp.int32(options.iterations_outer - 1)
    cost_tol = jnp.where(is_last, options.cost_tolerance, options.cost_tolerance_intermediate)
    grad_tol = jnp.where(is_last, options.gradient_tolerance, options.gradient_tolerance_intermediate)

    wrapped = _ALProblem(problem=problem, al=carry.al)
    # _ALProblem duck-types Problem's (model, obj, dynamics_expansion, cost_expansion) surface --
    # the whole composition seam -- without being one; ilqr_solve only ever calls through that
    # surface, so the cast is a type-checker formality, not a behavioral claim.
    new_traj, inner_stats, inner_status = ilqr_solve(
        cast("Problem", wrapped), carry.trajectory, options, cost_tolerance=cost_tol, gradient_tolerance=grad_tol
    )
    inner_failed = inner_status > jnp.int32(TerminationStatus.SOLVE_SUCCEEDED)

    C, _Jx_err, _Ju = evaluate_al_constraints(carry.al, problem.constraints, problem.model, new_traj)
    J = _ALObjective(problem.obj, problem.constraints, problem.model, carry.al).cost(new_traj)
    c_max = max_violation(carry.al, C)
    mu_max = max_penalty(carry.al)

    iter_num = i + 1
    idx = i
    stats = carry.stats
    recorded_stats = ALStats(
        iterations=iter_num,
        cost=stats.cost.at[idx].set(J),
        c_max=stats.c_max.at[idx].set(c_max),
        penalty_max=stats.penalty_max.at[idx].set(mu_max),
    )
    new_stats = jax.tree.map(lambda new, old: jnp.where(inner_failed, old, new), recorded_stats, stats)

    conv_status, conv_done = _evaluate_al_convergence(c_max, mu_max, inner_stats.iterations, iter_num, options)

    skip_dual_update = inner_failed | conv_done
    al_updated = penalty_update(dual_update(carry.al, C, options), options)
    new_al = jax.tree.map(lambda new, old: jnp.where(skip_dual_update, old, new), al_updated, carry.al)

    final_status = jnp.where(inner_failed, inner_status, conv_status)
    final_done = inner_failed | conv_done

    return ALCarry(
        i=iter_num,
        trajectory=new_traj,
        al=new_al,
        stats=new_stats,
        done=final_done,
        status=final_status,
    )


def al_solve(
    problem: Problem,
    trajectory: Trajectory,
    al0: ALConstraints,
    options: SolverOptions,
) -> tuple[Trajectory, ALConstraints, ALStats, jax.Array]:
    """Traced augmented-Lagrangian outer loop, matching `Altro.ALSolver`'s `solve!` (`al_solve.jl`).

    A pure `(problem, trajectory, al0, options) -> (trajectory, al, stats, status)` function
    built from one `lax.while_loop` around `ilqr_solve` (ticket 27), jittable and vmappable end
    to end with `options` static. `al0` is caller-supplied rather than built here: allocating its
    padded row layout (`ALConstraints.build`) is eager Python/NumPy over `problem.constraints`
    (ticket 28), not traceable, and is also where `reset_duals`/`reset_penalties`-gated
    warm-starting from a prior `MPCState.al` belongs (reference §5.5's note that the outer loop
    body itself never resets duals/penalties -- only a whole solve's start does).

    Parameters
    ----------
    problem : Problem
        Supplies the model, unconstrained objective, constraints, and the `cost_expansion` /
        `dynamics_expansion` methods `_ALProblem` wraps.
    trajectory : Trajectory
        Warm-start guess passed straight through to the first `ilqr_solve` call.
    al0 : ALConstraints
        Initial duals and penalties, already reset or warm-started by the caller.
    options : SolverOptions
        Static solve configuration; must not be traced.

    Returns
    -------
    tuple[Trajectory, ALConstraints, ALStats, jax.Array]
        The accepted trajectory, final duals/penalties, the outer stats history (buffers sized
        `options.iterations_outer`, untrimmed), and the exit `TerminationStatus` ordinal as an
        int32 scalar.
    """
    init_carry = ALCarry(
        i=jnp.int32(0),
        trajectory=trajectory,
        al=al0,
        stats=ALStats.create(options),
        done=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
        status=jnp.int32(TerminationStatus.UNSOLVED),
    )

    def cond(carry: ALCarry) -> jax.Array:
        return (~carry.done) & (carry.i < options.iterations_outer)

    def body(carry: ALCarry) -> ALCarry:
        return _al_step(carry, problem, options)

    final = jax.lax.while_loop(cond, body, init_carry)
    return final.trajectory, final.al, final.stats, final.status


class ALResult(NamedTuple):
    """Result of a native augmented-Lagrangian solve, satisfying the `SolverResult` protocol.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the core exited with `TerminationStatus.SOLVE_SUCCEEDED`.
    status : int
        `TerminationStatus` ordinal the traced core exited with.
    message : str
        `TerminationStatus` member name -- the precise Altro exit reason, kept for diagnostics.
    solver_status : SolverStatus
        `status` mapped through `to_solver_status`'s table; the authoritative public status
        `Problem.solve` uses for `MPCState.status`, rather than guessing from `message`.
    cost : float
        Final AL-augmented objective value.
    Z : jax.Array
        Optimal flat primal vector.
    info : dict[str, Any]
        Holds the trimmed outer `ALStats` history under `"stats"`.
    iterations : int, optional
        Number of completed outer iterations. Defaults to 0.
    constraint_violation : float, optional
        Final `max_violation` over the returned duals/trajectory. Defaults to 0.0.
    lam : np.ndarray, optional
        Always empty: AL duals live in `al`, in its own padded per-knot-per-row layout, not the
        canonical transcription row order this field promises. Defaults to empty.
    mu : np.ndarray, optional
        Always empty, for the same reason as `lam`. Defaults to empty.
    al : ALConstraints | None, optional
        Final padded duals and penalties, threaded into `MPCState.al` for warm-starting the next
        solve. Defaults to None.
    """

    trajectory: Trajectory
    success: bool
    status: int
    message: str
    solver_status: SolverStatus
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    iterations: int = 0
    constraint_violation: float = 0.0
    lam: np.ndarray = _EMPTY
    mu: np.ndarray = _EMPTY
    al: ALConstraints | None = None


@dataclass(frozen=True)
class AL:
    """Native augmented-Lagrangian solver backend, satisfying the `Solver` protocol.

    A thin eager wrapper over the traced `al_solve` core (ticket 29): `.solve()` builds the
    warm-start trajectory from `state`, builds or warm-starts the initial padded duals/penalties
    from `state.al` (subject to `options.reset_duals` / `reset_penalties`), calls the jitted
    core, then converts the traced status int and stats buffers into `success` / `message` /
    `info` at the boundary -- work that cannot happen inside a trace. Bound constraints
    (`ControlBound`, state bounds) are ordinary inequality constraints here, handled by the outer
    loop like any other; ticket 30 adds the alternative that treats them specially in the
    backward pass.

    Parameters
    ----------
    options : SolverOptions, optional
        Static solve configuration. Defaults to `SolverOptions()`.
    """

    options: SolverOptions = field(default_factory=SolverOptions)

    def solve(self, problem: Problem, state: MPCState) -> ALResult:
        """Run the traced AL outer loop from `state`'s warm-start trajectory/duals and boundary-convert the result."""
        options = self.options
        problem_eff, init_traj = build_warm_start(problem, state)

        fresh_al = ALConstraints.build(problem_eff.constraints, penalty_initial=options.penalty_initial)
        if state.al is not None:
            lam = fresh_al.lam if options.reset_duals else state.al.lam
            mu = fresh_al.mu if options.reset_penalties else state.al.mu
            init_al = eqx.tree_at(lambda a: (a.lam, a.mu), fresh_al, (lam, mu))
        else:
            init_al = fresh_al

        final_traj, final_al, stats, status_int = al_solve(problem_eff, init_traj, init_al, options)

        status = TerminationStatus(int(status_int))
        n_iter = int(stats.iterations)
        C, _Jx, _Ju = evaluate_al_constraints(final_al, problem_eff.constraints, problem_eff.model, final_traj)
        final_cost = _ALObjective(problem_eff.obj, problem_eff.constraints, problem_eff.model, final_al).cost(
            final_traj
        )

        return ALResult(
            trajectory=final_traj,
            success=status == TerminationStatus.SOLVE_SUCCEEDED,
            status=int(status_int),
            message=status.name,
            solver_status=to_solver_status(status),
            cost=float(final_cost),
            Z=_trajectory_to_z(final_traj.X, final_traj.U),
            info={"stats": _trim_al_stats(stats, n_iter)},
            iterations=n_iter,
            constraint_violation=float(max_violation(final_al, C)),
            al=final_al,
        )
