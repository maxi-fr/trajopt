"""Projected Newton polish phase (Altro's `ProjectedNewtonSolver`, `src/direct/`).

**Finding L -- a different formulation, not "one more phase".** The AL-iLQR phase (`al.py`,
`ilqr.py`) is single shooting: controls are the only decision variables and the dynamics hold by
construction because the trajectory is rolled out. Projected Newton is multiple shooting: states
and controls are stacked into one primal vector, the dynamics become explicit equality
constraints, and one KKT system covers the whole horizon at once. So this module assembles its
own primal and dual layout (`PNLayout` below), a second row-ordering convention entirely
independent of `transcription/layout.py`'s canonical NLP row order -- nothing here is called `Z`
or `lam` bare; the primal is always `z_pn` and the dual/residual is always `d_pn`.

**Dense KKT, a declared divergence from Altro.** Altro builds a sparse upper-triangular KKT
matrix and factors it with QDLDL; the active set changes shape between iterations, and a sparse
pattern that changes shape cannot be a static `jax` shape. This port assembles the KKT system
dense at the full `(Np + Nd, Np + Nd)` size every time, with inactive dual rows/columns masked to
an identity block (zero row, 1 on the diagonal) so inactive multipliers solve to zero and the
factorization stays well posed regardless of which rows are active. For the small, fixed `n`/`m`
this port targets, a dense `jnp.linalg.solve` is numerically equivalent to Altro's sparse QDLDL
solve and drastically simpler to trace.

**`multiplier_projection` is dead code upstream** (`pn_solve.jl`, Altro issue #35): the
implementation is commented out and the call site hardcodes `res = Inf`. This port implements it
for real, gated behind `options.multiplier_projection` (default `False`),
which makes this a superset of upstream. There is nothing on the Julia side to compare the
projection's numerical output against, so any cross-parity test must run with the option off on
both sides.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.expansions import _stage_cost_expansion, _terminal_cost_expansion
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.solvers.al import ALConstraints, _evaluate_bound_block, _evaluate_constraint_block
from trajopt.solvers.ilqr import build_warm_start
from trajopt.solvers.options import SolverOptions, TerminationStatus, to_solver_status
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z, parse_solver_initial_state
from trajopt.transcription.result import SolverStatus

_MAX_REFINEMENTS = 10  # hard-coded upstream (finding M), not an option
_INNER_LINESEARCH_STEPS = 10  # hard-coded upstream (finding M), not an option
_EMPTY = np.zeros(0, dtype=np.float64)


class PNLayout(NamedTuple):
    """Static structural layout for Projected Newton's own primal/dual row ordering (finding L).

    The primal `z_pn` stacks `[x_0, u_0, x_1, u_1, ..., x_{N-2}, u_{N-2}, x_{N-1}]`: state `x_k`
    always starts at flat column `k * (n + m)` for every k (this holds even at the terminal knot,
    since `(N-1) * (n + m)` is exactly where the trailing `x_{N-1}` block begins), and control
    `u_k` starts at `k * (n + m) + n` for k = 0 .. N-2 (no terminal control).

    The dual / residual `d_pn` stacks, in this fixed order: the initial-condition equality
    residual `x0 - x_0` (n rows), the dynamics equality residual `f(x_k, u_k) - x_{k+1}` for every
    k = 0 .. N-2 (n rows each), then one padded per-knot stage/bound constraint block (`p_max`
    rows per knot, `row_mask` marking real rows and `is_equality` marking `ZeroCone` rows, exactly
    `ALConstraints`'s padding convention) for every k = 0 .. N-1.

    Parameters
    ----------
    n, m, N : int
        State dimension, control dimension, horizon length.
    Np, Nd : int
        Primal and dual/residual vector lengths.
    p_cons_max, p_max : int
        Padded per-knot stage-constraint-only width, and total (stage + box-bound) width.
    row_mask, is_equality : jax.Array
        Boolean (N, p_max) masks for the padded constraint block only -- the initial-condition
        and dynamics blocks are always real and always equality rows, so they carry no mask here.
    """

    n: int
    m: int
    N: int
    Np: int
    Nd: int
    p_cons_max: int
    p_max: int
    row_mask: jax.Array
    is_equality: jax.Array

    @classmethod
    def build(cls, problem: Problem) -> "PNLayout":
        """Derive PN's static layout from `problem`'s structure (eager, not traced)."""
        n = int(problem.model.n)
        m = int(problem.model.m)
        N = int(problem.N)
        al0 = ALConstraints.build(problem.constraints)
        p_cons_max = int(al0.p_cons_max)
        p_max = p_cons_max + 2 * n + 2 * m
        Np = N * n + (N - 1) * m
        Nd = n + (N - 1) * n + N * p_max
        return cls(
            n=n,
            m=m,
            N=N,
            Np=Np,
            Nd=Nd,
            p_cons_max=p_cons_max,
            p_max=p_max,
            row_mask=al0.row_mask,
            is_equality=al0.is_equality,
        )


def _pack_z_pn(X: jax.Array, U: jax.Array) -> jax.Array:
    """Pack a Trajectory's (X, U) into PN's own flat primal `z_pn` of shape (Np,)."""
    body = jnp.concatenate([X[:-1], U], axis=1).reshape(-1)
    return jnp.concatenate([body, X[-1]])


def _unpack_z_pn(z_pn: jax.Array, layout: PNLayout) -> tuple[jax.Array, jax.Array]:
    """Unpack PN's flat primal `z_pn` back into (X, U) of shapes (N, n) and (N-1, m)."""
    n, m, N = layout.n, layout.m, layout.N
    body = (N - 1) * (n + m)
    zb = z_pn[:body].reshape(N - 1, n + m)
    X = jnp.concatenate([zb[:, :n], z_pn[body:].reshape(1, n)], axis=0)
    U = zb[:, n:]
    return X, U


def _scatter_block_diag(blocks: jax.Array, col_starts: jax.Array, size: int) -> jax.Array:
    """Place a stack of (K, d, d) blocks on the diagonal of a (size, size) zero matrix."""
    k, d, _ = blocks.shape
    local = jnp.arange(d)
    rows = jnp.broadcast_to(col_starts[:, None, None] + local[None, :, None], (k, d, d))
    cols = jnp.broadcast_to(col_starts[:, None, None] + local[None, None, :], (k, d, d))
    return jnp.zeros((size, size), dtype=blocks.dtype).at[rows, cols].set(blocks)


def _scatter_rect(
    blocks: jax.Array, row_starts: jax.Array, col_starts: jax.Array, rows_size: int, cols_size: int
) -> jax.Array:
    """Additively scatter a stack of (K, rdim, cdim) blocks into a (rows_size, cols_size) zero matrix."""
    k, rdim, cdim = blocks.shape
    rloc = jnp.arange(rdim)
    cloc = jnp.arange(cdim)
    rows = jnp.broadcast_to(row_starts[:, None, None] + rloc[None, :, None], (k, rdim, cdim))
    cols = jnp.broadcast_to(col_starts[:, None, None] + cloc[None, None, :], (k, rdim, cdim))
    return jnp.zeros((rows_size, cols_size), dtype=blocks.dtype).at[rows, cols].add(blocks)


class PNEval(NamedTuple):
    """One evaluation of PN's residual, active set, cost Hessian/gradient, and constraint Jacobian.

    Parameters
    ----------
    d_pn : jax.Array
        Stacked residual of shape (Nd,), PN's dual row order (`PNLayout`).
    active : jax.Array
        Boolean (Nd,) active-set mask in the same row order: equality rows (initial condition,
        dynamics) are always active; padded rows are never active; other inequality rows are
        active when `row_mask & (c > -active_set_tolerance_pn)` (Altro's `Inequality` branch).
    H : jax.Array
        Raw (non-error-coordinate) cost Hessian of shape (Np, Np), block diagonal per knot.
    g : jax.Array
        Raw cost gradient of shape (Np,).
    D : jax.Array
        Full constraint Jacobian of shape (Nd, Np), `d(d_pn)/d(z_pn)`.
    """

    d_pn: jax.Array
    active: jax.Array
    H: jax.Array
    g: jax.Array
    D: jax.Array


def _pn_evaluate(  # noqa: PLR0913, PLR0917 -- one argument per PNEval input; a bundle would just rename them
    problem: Problem,
    layout: PNLayout,
    options: SolverOptions,
    x0: jax.Array,
    traj: Trajectory,
    X: jax.Array,
    U: jax.Array,
) -> PNEval:
    """Evaluate residual, active set, raw cost Hessian/gradient, and constraint Jacobian at (X, U).

    Cost derivatives reuse `expansions._stage_cost_expansion` / `_terminal_cost_expansion` with
    `model=None`, which forces their error-state map `G` to the identity -- exactly the raw
    (non-error) gradient/Hessian PN's own primal coordinates need, without re-deriving the
    per-cost-type (diagonal / dense-quadratic / generic autodiff) dispatch those functions already
    implement. Dynamics Jacobians come directly from `model.state_jacobian` / `control_jacobian`.
    Stage/bound constraint Jacobians reuse `al._evaluate_constraint_block` / `_evaluate_bound_block`,
    which already return raw, pre-error-projection Jacobians (only `al.evaluate_al_constraints`'s
    public wrapper applies the error-state `G`).
    """
    n, m, N = layout.n, layout.m, layout.N
    Np, Nd, p_max = layout.Np, layout.Nd, layout.p_max
    model = problem.model
    T, dt = traj.t, traj.dt
    dtype = X.dtype

    # -- Residuals --------------------------------------------------------------------------
    d_init = x0 - X[0]
    f_next = jax.vmap(model.evaluate)(X[:-1], U, T[:-1], dt)
    d_dyn = f_next - X[1:]

    C_cons, Jx_cons, Ju_cons = _evaluate_constraint_block(problem.constraints, X, U, T, layout.p_cons_max)
    C_bound, Jx_bound, Ju_bound = _evaluate_bound_block(problem.constraints, X, U)
    C = jnp.concatenate([C_cons, C_bound], axis=-1)
    Jx_cn = jnp.concatenate([Jx_cons, Jx_bound], axis=1)
    Ju_cn = jnp.concatenate([Ju_cons, Ju_bound], axis=1)

    d_pn = jnp.concatenate([d_init, d_dyn.reshape(-1), C.reshape(-1)])

    # -- Active set ---------------------------------------------------------------------------
    tol = options.active_set_tolerance_pn
    active_cons = layout.row_mask & (layout.is_equality | (-tol < C))
    active = jnp.concatenate(
        [
            jnp.ones(n, dtype=bool),
            jnp.ones((N - 1) * n, dtype=bool),
            active_cons.reshape(-1),
        ]
    )

    # -- Raw cost Hessian / gradient, block diagonal per knot ----------------------------------
    q_st, r_st, Q_st, R_st, H_st = _stage_cost_expansion(problem.obj, X[:-1], U, T[:-1], None)
    q_term, Q_term = _terminal_cost_expansion(problem.obj, X[-1], T[-1], None)

    top = jnp.concatenate([Q_st, jnp.swapaxes(H_st, 1, 2)], axis=2)
    bot = jnp.concatenate([H_st, R_st], axis=2)
    stage_H = jnp.concatenate([top, bot], axis=1)  # (N-1, n+m, n+m)
    stage_g = jnp.concatenate([q_st, r_st], axis=1)  # (N-1, n+m)

    col_x_all = jnp.arange(N) * (n + m)
    body = (N - 1) * (n + m)
    H = jnp.zeros((Np, Np), dtype=dtype)
    H = H.at[:body, :body].set(_scatter_block_diag(stage_H, col_x_all[:-1], body))
    H = H.at[body:, body:].set(Q_term)
    g = jnp.zeros((Np,), dtype=dtype).at[:body].set(stage_g.reshape(-1)).at[body:].set(q_term)

    # -- Constraint Jacobian D: d(d_pn)/d(z_pn) ------------------------------------------------
    col_u_all = col_x_all + n
    D = jnp.zeros((Nd, Np), dtype=dtype)
    D = D.at[:n, :n].add(-jnp.eye(n, dtype=dtype))

    A_dyn = jax.vmap(model.state_jacobian)(X[:-1], U, T[:-1], dt)
    B_dyn = jax.vmap(model.control_jacobian)(X[:-1], U, T[:-1], dt)
    row_dyn = n + jnp.arange(N - 1) * n
    D = D + _scatter_rect(A_dyn, row_dyn, col_x_all[:-1], Nd, Np)
    D = D + _scatter_rect(B_dyn, row_dyn, col_u_all[:-1], Nd, Np)
    D = D + _scatter_rect(-jnp.broadcast_to(jnp.eye(n, dtype=dtype), (N - 1, n, n)), row_dyn, col_x_all[1:], Nd, Np)

    row_cons = n + (N - 1) * n + jnp.arange(N) * p_max
    D = D + _scatter_rect(Jx_cn, row_cons, col_x_all, Nd, Np)
    D = D + _scatter_rect(Ju_cn[:-1], row_cons[:-1], col_u_all[:-1], Nd, Np)

    return PNEval(d_pn=d_pn, active=active, H=H, g=g, D=D)


def _violation(d_pn: jax.Array, active: jax.Array) -> jax.Array:
    """Max absolute residual over active rows only, Altro's `max_violation`."""
    return jnp.max(jnp.where(active, jnp.abs(d_pn), 0.0))


def _residual_only(  # noqa: PLR0913, PLR0917 -- one argument per residual input; matches _pn_evaluate
    problem: Problem,
    layout: PNLayout,
    x0: jax.Array,
    traj: Trajectory,
    X: jax.Array,
    U: jax.Array,
) -> jax.Array:
    """Residual `d_pn` only, for line-search trial points (no active-set update, no Jacobians)."""
    T, dt = traj.t, traj.dt
    d_init = x0 - X[0]
    f_next = jax.vmap(problem.model.evaluate)(X[:-1], U, T[:-1], dt)
    d_dyn = f_next - X[1:]
    C_cons, _, _ = _evaluate_constraint_block(problem.constraints, X, U, T, layout.p_cons_max)
    C_bound, _, _ = _evaluate_bound_block(problem.constraints, X, U)
    C = jnp.concatenate([C_cons, C_bound], axis=-1)
    return jnp.concatenate([d_init, d_dyn.reshape(-1), C.reshape(-1)])


def _solve_kkt_step(ev: PNEval, layout: PNLayout, options: SolverOptions) -> jax.Array:
    """Solve the dense masked KKT system for the primal Newton step `p`, Altro's `_qdldl_solve!` assembly.

    Minimizes `0.5 dz' (H + rho_primal I) dz` subject to the *active* rows of `D dz = -d_pn`
    (inactive rows masked to an identity block so their multiplier solves to zero) -- a
    Hessian-weighted projection onto the active constraint manifold, not a full-gradient Newton
    step: Altro's own RHS zeroes the primal block (`update_b!` clears `b` before filling only the
    dual block with `-d`), so the cost gradient never enters this system.
    """
    Np = layout.Np
    H_reg = ev.H + options.rho_primal * jnp.eye(Np, dtype=ev.H.dtype)
    D_masked = ev.D * ev.active[:, None]
    inactive_diag = jnp.where(ev.active, 0.0, 1.0)

    top = jnp.concatenate([H_reg, D_masked.T], axis=1)
    bottom = jnp.concatenate([D_masked, jnp.diag(inactive_diag)], axis=1)
    kkt = jnp.concatenate([top, bottom], axis=0)
    rhs = jnp.concatenate([jnp.zeros(Np, dtype=ev.H.dtype), jnp.where(ev.active, -ev.d_pn, 0.0)])

    sol = jnp.linalg.solve(kkt, rhs)
    return sol[:Np]


def multiplier_projection(ev: PNEval, layout: PNLayout) -> jax.Array:
    """Project multipliers onto the active constraint manifold via the KKT stationarity normal equations.

    Solves `(D_active D_active^T) y = -D_active g` for the dual `y` that best satisfies the
    stationarity condition `g + D^T y = 0` over the currently active rows, with inactive rows
    masked to an identity block exactly like `_solve_kkt_step` so the result keeps PN's full
    static `(Nd,)` shape regardless of which rows are structurally active. Gated by
    `options.multiplier_projection` at the call site (default `True`, matching Altro's own
    default): Altro's own `multiplier_projection!` is commented-out dead code and its call site
    hardcodes `res = Inf` (issue #35), so this is a genuine port, not a translation of working
    upstream code, and there is nothing on the Julia side to numerically compare it against.
    """
    Nd = layout.Nd
    D_masked = ev.D * ev.active[:, None]
    inactive_eye = jnp.where(ev.active, 0.0, 1.0)[:, None] * jnp.eye(Nd, dtype=ev.D.dtype)
    gram = D_masked @ D_masked.T + inactive_eye
    rhs = jnp.where(ev.active, -(D_masked @ ev.g), 0.0)
    return jnp.linalg.solve(gram, rhs)


class _LSCarry(NamedTuple):
    i: jax.Array
    alpha: jax.Array
    z: jax.Array
    viol: jax.Array
    accepted: jax.Array


def _pn_linesearch(  # noqa: PLR0913, PLR0917 -- mirrors _qdldl_linesearch's fixed argument set
    problem: Problem,
    layout: PNLayout,
    x0: jax.Array,
    traj: Trajectory,
    z0: jax.Array,
    p: jax.Array,
    active: jax.Array,
    viol0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Halve alpha up to 10 times, accepting the first step that reduces violation (Altro's `_qdldl_linesearch`).

    Does not update the active set: violation at every trial point is measured against the fixed
    `active` mask the caller already computed, on a fresh residual only (no Jacobians/Hessian).
    Returns `(z, viol, accepted)`; on exhaustion `z`/`viol` are unchanged from `z0`/`viol0` and
    `accepted` is False.
    """

    def viol_at(z: jax.Array) -> jax.Array:
        x, u = _unpack_z_pn(z, layout)
        d = _residual_only(problem, layout, x0, traj, x, u)
        return _violation(d, active)

    def cond(c: _LSCarry) -> jax.Array:
        return (~c.accepted) & (c.i < _INNER_LINESEARCH_STEPS)

    def body(c: _LSCarry) -> _LSCarry:
        z_trial = z0 + c.alpha * p
        v_trial = viol_at(z_trial)
        accept = v_trial < viol0
        new_z = jnp.where(accept, z_trial, c.z)
        new_viol = jnp.where(accept, v_trial, c.viol)
        return _LSCarry(i=c.i + 1, alpha=c.alpha / 2.0, z=new_z, viol=new_viol, accepted=accept)

    init = _LSCarry(
        i=jnp.int32(0),
        alpha=jnp.asarray(1.0, dtype=z0.dtype),
        z=z0,
        viol=viol0,
        accepted=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
    )
    final = jax.lax.while_loop(cond, body, init)
    return final.z, final.viol, final.accepted


def _refine_converged(viol: jax.Array, viol_prev: jax.Array, tolerance: float, r_threshold: float) -> jax.Array:
    """Return True once refinement should stop: violation clears `tolerance`, or its convergence rate falls below `r_threshold`."""
    log_prev = jnp.log10(jnp.maximum(viol_prev, 1e-300))
    log_new = jnp.log10(jnp.maximum(viol, 1e-300))
    rate = log_new / jnp.where(log_prev == 0.0, 1.0, log_prev)
    return (viol < tolerance) | (rate < r_threshold)


class _RefineCarry(NamedTuple):
    i: jax.Array
    z: jax.Array
    viol: jax.Array
    done: jax.Array


def _pn_refine(  # noqa: PLR0913, PLR0917 -- mirrors projection_solve!'s fixed argument set
    problem: Problem,
    layout: PNLayout,
    options: SolverOptions,
    x0: jax.Array,
    traj: Trajectory,
    z0: jax.Array,
    p: jax.Array,
    active: jax.Array,
    viol0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reapply the fixed Newton step `p` with a fresh line search up to 10 times (Altro's `projection_solve!` inner loop).

    The step `p` is solved once per outer iteration (chord method) and reused unmodified across
    every refinement round; only the line search's `alpha` restarts at 1.0 each round. Exits when
    the violation clears `options.projected_newton_tolerance`, when the convergence rate
    `log10(viol) / log10(viol_prev)` falls below `options.r_threshold`, or when a round's line
    search fails to find any alpha that reduces the violation.
    """

    def cond(c: _RefineCarry) -> jax.Array:
        return (~c.done) & (c.i < _MAX_REFINEMENTS)

    def body(c: _RefineCarry) -> _RefineCarry:
        z_new, viol_new, accepted = _pn_linesearch(problem, layout, x0, traj, c.z, p, active, c.viol)
        converged = _refine_converged(viol_new, c.viol, options.projected_newton_tolerance, options.r_threshold)
        done = converged | (~accepted)
        return _RefineCarry(i=c.i + 1, z=z_new, viol=viol_new, done=done)

    init = _RefineCarry(i=jnp.int32(0), z=z0, viol=viol0, done=jnp.asarray(False))  # noqa: FBT003
    final = jax.lax.while_loop(cond, body, init)
    return final.z, final.viol


class PNStats(eqx.Module):
    """Fixed-size pytree of per-outer-iteration Projected Newton solve statistics, traceable end to end.

    Parameters
    ----------
    iterations : jax.Array
        Number of completed outer projection solves, as an int32 scalar.
    c_max : jax.Array
        Max violation after refinement, history of shape `(options.n_steps + 1,)`.
    active : jax.Array
        Active-set mask before the projection solve of each outer iteration, shape
        `(options.n_steps + 1, Nd)`.
    """

    iterations: jax.Array
    c_max: jax.Array
    active: jax.Array

    @classmethod
    def create(cls, options: SolverOptions, nd: int) -> "PNStats":
        """Allocate zeroed history buffers sized `options.n_steps + 1`."""
        k = options.n_steps + 1
        return cls(
            iterations=jnp.asarray(0, dtype=jnp.int32),
            c_max=jnp.zeros(k, dtype=jnp.float64),
            active=jnp.zeros((k, nd), dtype=bool),
        )


def _trim_pn_stats(stats: PNStats, n_iter: int) -> PNStats:
    """Slice a finished PN solve's fixed-size stats buffers down to the completed iteration count."""
    return PNStats(iterations=stats.iterations, c_max=stats.c_max[:n_iter], active=stats.active[:n_iter])


class _PNCarry(NamedTuple):
    count: jax.Array
    z: jax.Array
    viol: jax.Array
    duals: jax.Array
    stats: PNStats


def pn_solve(
    problem: Problem,
    trajectory: Trajectory,
    x0: jax.Array,
    options: SolverOptions,
    bc: BoundaryConditions | None = None,
) -> tuple[Trajectory, PNStats, jax.Array, jax.Array]:
    """Traced Projected Newton polish-phase outer loop, matching `pn_solve.jl`'s `solve!`.

    `x0` is threaded explicitly rather than read off `trajectory.X[0]`: PN is multiple shooting
    (finding L), so `X[0]` is a free primal variable pinned back to `x0` by the initial-condition
    equality row, not a fixed rollout anchor the way single-shooting AL/iLQR treat it. Altro pins
    `pn.x0` once from `prob.x0` at solver construction; this mirrors that.

    Parameters
    ----------
    problem : Problem
        Supplies the model, objective, and constraints `_pn_evaluate` assembles the KKT system
        from.
    trajectory : Trajectory
        Warm-start guess, typically the AL-iLQR phase's output.
    x0 : jax.Array
        Fixed initial condition of shape (n,), held constant across every outer/middle/inner
        iteration.
    options : SolverOptions
        Static solve configuration; must not be traced.
    bc : BoundaryConditions | None, optional
        Traced boundary conditions; their reference window retargets `problem`'s objective here,
        inside the trace, so a moving target costs no recompile. Defaults to None, meaning the
        objective keeps the target it was built with.

    Returns
    -------
    tuple[Trajectory, PNStats, jax.Array, jax.Array]
        `(trajectory, stats, duals, status)`: the projected trajectory, the outer stats history
        (buffers sized `options.n_steps + 1`, untrimmed), the final projected dual vector in PN's
        own `Nd`-row order (zeros when `options.multiplier_projection` is False), and the exit
        `TerminationStatus` ordinal as an int32 scalar.
    """
    layout = PNLayout.build(problem)
    problem = retarget_problem(problem, bc)
    z_init = _pack_z_pn(trajectory.X, trajectory.U)
    ev0 = _pn_evaluate(problem, layout, options, x0, trajectory, trajectory.X, trajectory.U)
    viol0 = _violation(ev0.d_pn, ev0.active)
    eps_feas = options.constraint_tolerance

    init_carry = _PNCarry(
        count=jnp.int32(0),
        z=z_init,
        viol=viol0,
        duals=jnp.zeros(layout.Nd, dtype=z_init.dtype),
        stats=PNStats.create(options, layout.Nd),
    )

    def cond(c: _PNCarry) -> jax.Array:
        return (c.count <= options.n_steps) & (c.viol > eps_feas)

    def body(c: _PNCarry) -> _PNCarry:
        X, U = _unpack_z_pn(c.z, layout)
        ev = _pn_evaluate(problem, layout, options, x0, trajectory, X, U)
        viol_before = _violation(ev.d_pn, ev.active)
        p = _solve_kkt_step(ev, layout, options)
        z_refined, viol_after = _pn_refine(problem, layout, options, x0, trajectory, c.z, p, ev.active, viol_before)

        duals = c.duals
        if options.multiplier_projection:
            X_r, U_r = _unpack_z_pn(z_refined, layout)
            ev_r = _pn_evaluate(problem, layout, options, x0, trajectory, X_r, U_r)
            duals = multiplier_projection(ev_r, layout)

        idx = c.count
        stats = PNStats(
            iterations=c.stats.iterations + 1,
            c_max=c.stats.c_max.at[idx].set(viol_after),
            active=c.stats.active.at[idx].set(ev.active),
        )
        return _PNCarry(count=c.count + 1, z=z_refined, viol=viol_after, duals=duals, stats=stats)

    final = jax.lax.while_loop(cond, body, init_carry)

    X_final, U_final = _unpack_z_pn(final.z, layout)
    final_traj = eqx.tree_at(lambda t: (t.X, t.U), trajectory, (X_final, U_final))

    status = jnp.where(
        final.viol <= eps_feas,
        jnp.int32(TerminationStatus.SOLVE_SUCCEEDED),
        jnp.int32(TerminationStatus.MAX_ITERATIONS_OUTER),
    )
    return final_traj, final.stats, final.duals, status


def _jit_pn_solve(
    program: Program,
    trajectory: Trajectory,
    x0: jax.Array,
    options: SolverOptions,
    bc: BoundaryConditions | None = None,
) -> tuple[Trajectory, PNStats, jax.Array, jax.Array]:
    """Run `program`'s `pn_solve` core, called from `PN.solve()`.

    `problem` is closed over by the core rather than passed as a jit argument (`Program.core`'s
    docstring has the reason: `PNLayout.build(problem)` reads its constraint bounds with eager
    `np.asarray`, which breaks under trace). The program builds the core once and reuses it across
    steps. `altro_solve`'s own internal call to `pn_solve` is left un-wrapped: it already runs
    inside `ALTRO.solve()`'s own jitted core (`altro.py`'s `_jit_altro_solve`), so wrapping it
    again would just nest one jit inside another for no benefit. `bc` is a traced argument, so a
    run-time target that moves between calls does not disturb the reuse.
    """
    core = program.core(pn_solve, key=options, options=options)
    return core(trajectory=trajectory, x0=x0, bc=bc)


class PNResult(NamedTuple):
    """Result of a native Projected Newton polish solve, satisfying the `SolverResult` protocol.

    Parameters
    ----------
    trajectory : Trajectory
        Projected state and control trajectory.
    success : bool
        Whether the core exited with `TerminationStatus.SOLVE_SUCCEEDED`.
    status : int
        `TerminationStatus` ordinal the traced core exited with.
    message : str
        `TerminationStatus` member name.
    solver_status : SolverStatus
        `status` mapped through `to_solver_status`'s table.
    cost : float
        Final objective value at the projected trajectory.
    Z : jax.Array
        Optimal flat primal vector, in `transcription/layout.py`'s canonical order (not PN's own
        `z_pn`, which is internal).
    info : dict[str, Any]
        Holds the trimmed outer `PNStats` history under `"stats"`.
    constraint_violation : float
        Final `max_violation` over PN's own residual/active-set.
    iterations : int, optional
        Number of completed outer projection solves. Defaults to 0.
    lam : np.ndarray, optional
        Always empty: PN's duals live in `duals`, in its own row layout, not the canonical
        transcription row order this field promises. Defaults to empty.
    mu : np.ndarray, optional
        Always empty, for the same reason as `lam`. Defaults to empty.
    duals : jax.Array | None, optional
        Final `multiplier_projection` output in PN's own `Nd`-row order, or None when
        `options.multiplier_projection` is False. Defaults to None.
    """

    trajectory: Trajectory
    success: bool
    status: int
    message: str
    solver_status: SolverStatus
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    constraint_violation: float
    iterations: int = 0
    lam: np.ndarray = _EMPTY
    mu: np.ndarray = _EMPTY
    duals: jax.Array | None = None


@dataclass(frozen=True)
class PN:
    """Native Projected Newton polish-phase solver backend, satisfying the `Solver` protocol.

    A thin eager wrapper over the traced `pn_solve` core: `.solve()` builds the warm-start
    trajectory from `ws` (typically the AL phase's output), pins `x0` from `bc.x0`, calls
    the jitted core, then converts the traced status int and stats buffers into `success` /
    `message` / `info` at the boundary -- work that cannot happen inside a trace.

    Parameters
    ----------
    options : SolverOptions, optional
        Static solve configuration. Defaults to `SolverOptions()`.
    """

    options: SolverOptions = field(default_factory=SolverOptions)

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> PNResult:
        """Run the traced PN outer loop from `ws`'s warm-start trajectory and boundary-convert the result."""
        problem = program.problem
        options = self.options
        init_traj = build_warm_start(problem, bc, ws)
        x0_arr, _t0_arr, _dt_arr, _xf_val, _z0 = parse_solver_initial_state(problem, bc, ws)
        problem_eff = retarget_problem(problem, bc)

        final_traj, stats, duals, status_int = _jit_pn_solve(program, init_traj, x0_arr, options, bc)

        status = TerminationStatus(int(status_int))
        n_iter = int(stats.iterations)
        layout = PNLayout.build(problem_eff)
        ev_final = _pn_evaluate(problem_eff, layout, options, x0_arr, final_traj, final_traj.X, final_traj.U)

        return PNResult(
            trajectory=final_traj,
            success=status == TerminationStatus.SOLVE_SUCCEEDED,
            status=int(status_int),
            message=status.name,
            solver_status=to_solver_status(status),
            cost=float(problem_eff.obj.cost(final_traj)),
            Z=_trajectory_to_z(final_traj.X, final_traj.U),
            info={"stats": _trim_pn_stats(stats, n_iter)},
            iterations=n_iter,
            constraint_violation=float(_violation(ev_final.d_pn, ev_final.active)),
            duals=duals if options.multiplier_projection else None,
        )
