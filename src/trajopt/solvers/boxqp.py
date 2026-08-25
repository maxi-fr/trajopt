import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from trajopt.constraints.constraint_list import BuiltConstraintList
from trajopt.problem import MPCState, Problem
from trajopt.solvers.al import ALConstraints, ALStats, _jit_al_solve, al_cost, evaluate_al_constraints, max_violation
from trajopt.solvers.ilqr import SolveKD, build_warm_start
from trajopt.solvers.options import SolverOptions, TerminationStatus, to_solver_status
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z
from trajopt.transcription.result import SolverStatus

_EMPTY = np.zeros(0, dtype=np.float64)

# Projected-Newton box-QP internals (Tassa, Erez & Todorov's `boxQP.m`); hard-coded rather than
# `SolverOptions` fields since there is no oracle asking for them to be tunable (ticket 30 has no
# Altro reference, and these mirror `boxQP.m`'s own hard-coded defaults).
_MAX_ITERS = 100
_MIN_GRAD = 1e-8
_MIN_REL_IMPROVE = 1e-8
_ARMIJO = 0.1
_STEP_DEC = 0.6
_MIN_STEP = 1e-22


def _qp_value(Quu: jax.Array, Qu: jax.Array, x: jax.Array) -> jax.Array:
    """Evaluate the box-QP objective `x'Qu + 0.5*x'Quu*x` at `x`."""
    return jnp.dot(x, Qu) + 0.5 * jnp.dot(x, Quu @ x)


def _masked_solve(H: jax.Array, rhs: jax.Array, free: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve `H_free @ x_free = rhs_free`, decoupled from the clamped block by a diagonal projector.

    `free` is a float mask (1.0 free, 0.0 clamped) of shape (m,). Building
    `H_mod = diag(free) @ H @ diag(free) + diag(1 - free)` makes the clamped block the identity
    and zeroes every free/clamped cross term, so `H_mod` factors whenever `H`'s free-free block
    is PD, and `cho_solve` against a `free`-masked right-hand side returns exactly zero on every
    clamped row -- no dynamically-shaped submatrix extraction under trace.

    Parameters
    ----------
    H : jax.Array
        Symmetric matrix of shape (m, m).
    rhs : jax.Array
        Right-hand side of shape (m,) or (m, k).
    free : jax.Array
        Free-variable mask of shape (m,), 1.0 free / 0.0 clamped.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        `(x, failed)`: the solution (zero on clamped rows), and whether the free-free block's
        Cholesky factorization was not positive definite, as a bool scalar.
    """
    D = free[:, None] * free[None, :]
    H_mod = D * H + jnp.diag(1.0 - free)
    L = jnp.linalg.cholesky(H_mod)
    failed = jnp.any(jnp.isnan(L))
    masked_rhs = rhs * (free[:, None] if rhs.ndim == 2 else free)  # noqa: PLR2004 -- rhs.ndim is 1 or 2, no other case
    x = jax.scipy.linalg.cho_solve((L, True), masked_rhs)
    return x, failed


class BoxQPResult(eqx.Module):
    """Minimizer and free/clamped classification of one box-constrained QP.

    Parameters
    ----------
    x : jax.Array
        Minimizer of shape (m,).
    free : jax.Array
        Boolean free-variable mask of shape (m,); False marks a clamped (bound-active) row.
    failed : jax.Array
        Whether a free-subspace Cholesky factorization hit a non-positive-definite matrix at any
        iteration, as a bool scalar.
    iterations : jax.Array
        Number of outer projected-Newton iterations run, as an int32 scalar.
    """

    x: jax.Array
    free: jax.Array
    failed: jax.Array
    iterations: jax.Array


class _LineSearchCarry(NamedTuple):
    step: jax.Array
    done: jax.Array
    ok: jax.Array


def _box_qp_line_search(  # noqa: PLR0913, PLR0917 -- one Armijo backtrack needs the full local QP context
    Quu: jax.Array,
    Qu: jax.Array,
    x: jax.Array,
    search: jax.Array,
    sdotg: jax.Array,
    old_value: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Armijo backtrack from step=1 along `search`, halved by `_STEP_DEC`, per `boxQP.m`'s inner loop.

    Returns the *unmoved* `(x, old_value)` and `failed=True` if `step` underflowed `_MIN_STEP`
    before any step satisfied the Armijo condition, matching `boxQP.m`'s `result=2` exit (which
    leaves the outer iterate at its pre-step value).
    """

    def cond(c: _LineSearchCarry) -> jax.Array:
        return ~c.done

    def body(c: _LineSearchCarry) -> _LineSearchCarry:
        xc = jnp.clip(x + c.step * search, lo, hi)
        vc = _qp_value(Quu, Qu, xc)
        armijo_ok = (vc - old_value) <= _ARMIJO * c.step * sdotg
        too_small = c.step < _MIN_STEP
        done = armijo_ok | too_small
        return _LineSearchCarry(step=jnp.where(done, c.step, c.step * _STEP_DEC), done=done, ok=armijo_ok)

    init = _LineSearchCarry(
        step=jnp.asarray(1.0, dtype=x.dtype),
        done=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
        ok=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
    )
    final = jax.lax.while_loop(cond, body, init)

    xc = jnp.clip(x + final.step * search, lo, hi)
    vc = _qp_value(Quu, Qu, xc)
    new_x = jnp.where(final.ok, xc, x)
    new_value = jnp.where(final.ok, vc, old_value)
    return new_x, new_value, ~final.ok


class _BoxQPCarry(NamedTuple):
    i: jax.Array
    x: jax.Array
    value: jax.Array
    done: jax.Array
    failed: jax.Array


def box_qp_solve(
    Quu: jax.Array, Qu: jax.Array, lo: jax.Array, hi: jax.Array, x0: jax.Array | None = None
) -> BoxQPResult:
    """Solve `min 0.5*x'Quu*x + Qu'x  s.t.  lo <= x <= hi` by projected-Newton box-QP.

    Ports Tassa, Erez & Todorov's `boxQP.m` (control-limited DDP): each outer iteration
    classifies the clamped/free variables from the current gradient, takes a Newton step on the
    free subspace with clamped variables fixed at their current (bound-active) value -- solved
    with the masked Cholesky `_masked_solve` rather than a dynamically-shaped submatrix, since
    shapes must stay static under trace -- then backtracks an Armijo line search from step=1.

    There is no Altro oracle for this ticket (docs/altro-port/30), so the deliberate
    simplifications from the published `boxQP.m` are recorded here rather than checked against a
    reference: the free-free Cholesky is refactored every outer iteration rather than only when
    the clamped set changes (`boxQP.m`'s factorization-reuse optimization, immaterial to the
    result, just to its cost), and the outer loop is `lax.while_loop`-bounded by `_MAX_ITERS`
    rather than `boxQP.m`'s unbounded `for iter = 1:maxIter`. The free-set update rule
    (reclassify from the current gradient every iteration, `boxQP.m`'s default) and the Armijo
    line search (backtrack by `_STEP_DEC` from step=1, accept at `_ARMIJO`, fail below
    `_MIN_STEP`) are otherwise a direct port.

    Parameters
    ----------
    Quu : jax.Array
        Symmetric (possibly regularized) Hessian of shape (m, m).
    Qu : jax.Array
        Linear term of shape (m,).
    lo : jax.Array
        Lower bound of shape (m,).
    hi : jax.Array
        Upper bound of shape (m,).
    x0 : jax.Array | None, optional
        Initial guess of shape (m,), clamped into bounds. Defaults to zeros.

    Returns
    -------
    BoxQPResult
        The minimizer, free/clamped mask, failure flag, and outer iteration count.
    """
    m = Qu.shape[0]
    dtype = Qu.dtype
    x_raw = jnp.zeros(m, dtype=dtype) if x0 is None else x0
    x_init = jnp.clip(x_raw, lo, hi)
    v_init = _qp_value(Quu, Qu, x_init)

    init = _BoxQPCarry(
        i=jnp.int32(0),
        x=x_init,
        value=v_init,
        done=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
        failed=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
    )

    def cond(c: _BoxQPCarry) -> jax.Array:
        return (~c.done) & (c.i < _MAX_ITERS)

    def body(c: _BoxQPCarry) -> _BoxQPCarry:
        grad = Qu + Quu @ c.x
        at_lo = c.x <= lo
        at_hi = c.x >= hi
        free = jnp.where(at_lo & (grad > 0.0), 0.0, jnp.where(at_hi & (grad < 0.0), 0.0, 1.0)).astype(dtype)
        all_clamped = jnp.all(free == 0.0)
        gnorm = jnp.linalg.norm(jnp.where(free > 0.0, grad, 0.0))
        grad_converged = gnorm < _MIN_GRAD

        grad_clamped = Qu + Quu @ (c.x * (1.0 - free))
        delta, chol_failed = _masked_solve(Quu, -grad_clamped, free)
        search = delta - free * c.x
        sdotg = jnp.dot(search, grad)
        no_descent = sdotg >= 0.0

        stop_before_step = all_clamped | grad_converged | chol_failed | no_descent

        ls_x, ls_value, ls_failed = _box_qp_line_search(Quu, Qu, c.x, search, sdotg, c.value, lo, hi)
        new_x = jnp.where(stop_before_step, c.x, ls_x)
        new_value = jnp.where(stop_before_step, c.value, ls_value)

        rel_improve = c.value - new_value
        no_improve = (c.i > 0) & ~stop_before_step & (rel_improve < _MIN_REL_IMPROVE * jnp.abs(c.value))

        done = stop_before_step | (ls_failed & ~stop_before_step) | no_improve
        failed = c.failed | chol_failed

        return _BoxQPCarry(i=c.i + 1, x=new_x, value=new_value, done=done, failed=failed)

    final = jax.lax.while_loop(cond, body, init)

    grad_final = Qu + Quu @ final.x
    at_lo = final.x <= lo
    at_hi = final.x >= hi
    free_final = ~(at_lo & (grad_final > 0.0)) & ~(at_hi & (grad_final < 0.0))

    return BoxQPResult(x=final.x, free=free_final, failed=final.failed, iterations=final.i)


def _control_bound_solve_kd(  # noqa: PLR0913, PLR0917 -- one knot's box-QP (K, d) solve needs its full context
    lo: jax.Array,
    hi: jax.Array,
    U: jax.Array,
    k: jax.Array,
    Quu_reg: jax.Array,
    Qux: jax.Array,
    Qu: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`SolveKD` for one knot: box-QP feedforward, masked-Cholesky feedback with clamped rows zeroed.

    The feedback gain is not the unconstrained gain: `_masked_solve` zeroes `K`'s rows for every
    control the box-QP solve clamped, since a clamped control does not respond to state
    deviation (a control pinned at a bound has no room left to react).
    """
    u_bar = U[k]
    qp = box_qp_solve(Quu_reg, Qu, lo - u_bar, hi - u_bar)
    free = qp.free.astype(Quu_reg.dtype)
    neg_K, k_failed = _masked_solve(Quu_reg, Qux, free)
    K_k = -neg_K
    return K_k, qp.x, qp.failed | k_failed


def make_control_bound_solve_kd(lo: jax.Array, hi: jax.Array) -> Callable[[Trajectory], SolveKD]:
    """Build ticket 30's `SolveKD` factory for `ilqr_solve`/`al_solve`: box-QP on control bounds.

    Routes only `ControlBound` rows to the box-QP backward pass; `lo`/`hi` are a single pair
    closed over for every knot in the horizon (see `extract_uniform_control_bounds` for why
    per-knot-varying bounds are rejected at build time instead).

    Parameters
    ----------
    lo : jax.Array
        Control lower bound of shape (m,).
    hi : jax.Array
        Control upper bound of shape (m,).

    Returns
    -------
    Callable[[Trajectory], SolveKD]
        Given the trajectory carried into one iLQR iteration, returns that iteration's
        `SolveKD`, closing over the nominal controls `traj.U` the box-QP bounds are offset
        against (rebuilt every iteration since `U` changes iteration to iteration).
    """

    def build(traj: Trajectory) -> SolveKD:
        U = traj.U

        def solve_kd(
            k: jax.Array, Quu_reg: jax.Array, Qux: jax.Array, Qu: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _control_bound_solve_kd(lo, hi, U, k, Quu_reg, Qux, Qu)

        return solve_kd

    return build


@functools.lru_cache(maxsize=32)
def _cached_control_bound_solve_kd_builder(
    lo_key: tuple[float, ...], hi_key: tuple[float, ...]
) -> Callable[[Trajectory], SolveKD]:
    """Memoized `make_control_bound_solve_kd`, keyed on `lo`/`hi`'s concrete values.

    `BoxQP.solve()` passes its `solve_kd_builder` to `jax.jit`'s `_jit_al_solve` as a static
    argument (Python callables cannot be traced), so `jax.jit`'s compilation cache -- which hashes
    static arguments by identity for anything without value-based `__eq__`/`__hash__` -- would miss
    on every call if a fresh closure were built each time, even when the bounds themselves (the
    common MPC case: fixed control limits, only the state changing) have not changed. Returning
    the same closure object for the same concrete bounds restores the cache hit.
    """
    return make_control_bound_solve_kd(jnp.asarray(lo_key), jnp.asarray(hi_key))


def extract_uniform_control_bounds(constraints: BuiltConstraintList) -> tuple[jax.Array, jax.Array]:
    """Extract one `(lo, hi)` control-bound pair for box-QP routing, eager over `constraints`.

    Box-QP here supports only a control bound uniform across the whole horizon (Tassa's classic
    formulation): `backward_pass`'s box-QP variant closes over a single `(lo, hi)` pair used at
    every knot in its `scan`, not a per-knot gather. Run once at Problem-build time, not under
    trace, so a non-uniform `ControlBound` -- one whose registered limits differ from knot to
    knot -- is a build-time error naming the problem rather than a silently wrong per-knot solve.

    Parameters
    ----------
    constraints : BuiltConstraintList
        Built constraint list; `constraints.u_lower`/`u_upper` already fold every registered
        `ControlBound`/`BoundConstraint` control row into one `(N-1, m)` array per bound.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        `(lo, hi)`, each of shape `(m,)`.

    Raises
    ------
    ValueError
        If the built control bounds differ across knots.
    """
    u_lower = np.asarray(constraints.u_lower)
    u_upper = np.asarray(constraints.u_upper)
    uniform = bool(np.all(u_lower == u_lower[0])) and bool(np.all(u_upper == u_upper[0]))
    if not uniform:
        msg = (
            "BoxQP requires ControlBound limits uniform across the whole horizon, but this "
            "problem's control bounds vary per knot -- a bound structure the box-QP backward "
            "pass does not support (it closes over a single (lo, hi) pair for every knot)."
        )
        raise ValueError(msg)
    return jnp.asarray(u_lower[0]), jnp.asarray(u_upper[0])


class BoxQPSolveResult(NamedTuple):
    """Result of a native control-limited solve, satisfying the `SolverResult` protocol.

    Combines ticket 30's box-QP backward pass (control bounds, enforced exactly by construction)
    with ticket 29's augmented-Lagrangian outer loop (every other constraint, state bounds
    included). A problem carrying only control bounds degenerates cleanly: the AL layout built
    from it has no real rows, so `al_solve`'s first outer iteration finds `c_max = 0 <
    constraint_tolerance` and exits immediately -- one inner iLQR solve, exactly Tassa's plain
    control-limited DDP.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the core exited with `TerminationStatus.SOLVE_SUCCEEDED`.
    status : int
        `TerminationStatus` ordinal the traced core exited with.
    message : str
        `TerminationStatus` member name.
    solver_status : SolverStatus
        `status` mapped through `to_solver_status`'s table.
    cost : float
        Final AL-augmented objective value (control-bound rows never contribute, since box-QP
        enforces them directly rather than through a penalty).
    Z : jax.Array
        Optimal flat primal vector.
    info : dict[str, Any]
        Holds the trimmed outer `ALStats` history under `"stats"`.
    iterations : int, optional
        Number of completed outer iterations. Defaults to 0.
    constraint_violation : float, optional
        Final `max_violation` over the non-control-bound AL rows. Defaults to 0.0.
    lam : np.ndarray, optional
        Always empty, for the same reason as `ALResult.lam`. Defaults to empty.
    mu : np.ndarray, optional
        Always empty, for the same reason as `ALResult.mu`. Defaults to empty.
    al : ALConstraints | None, optional
        Final padded duals/penalties (control-bound rows always masked out), for MPCState
        warm-starting. Defaults to None.
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
class BoxQP:
    """Native control-limited solver: box-QP for control bounds, AL for everything else.

    A thin eager wrapper, mirroring `AL` and `ILQR`: `.solve()` builds the warm-start trajectory,
    extracts the problem's uniform control bounds (raising at build time if they are not
    uniform, per `extract_uniform_control_bounds`), builds an AL layout with control-bound rows
    neutralized (`u_lower`/`u_upper` set to +-inf so `ALConstraints.build`'s `isfinite` check
    masks them out -- `problem.constraints` itself is untouched, so `evaluate_al_constraints`
    still evaluates real bound residuals that the mask simply never lets contribute), then calls
    `al_solve` with a `solve_kd_builder` that routes every knot's `(K, d)` through
    `box_qp_solve`.

    Parameters
    ----------
    options : SolverOptions, optional
        Static solve configuration. Defaults to `SolverOptions()`.
    """

    options: SolverOptions = field(default_factory=SolverOptions)

    def solve(self, problem: Problem, state: MPCState) -> BoxQPSolveResult:
        """Run the traced AL outer loop with a box-QP inner backward pass, boundary-converting the result."""
        options = self.options
        problem_eff, init_traj = build_warm_start(problem, state)
        constraints = problem_eff.constraints

        lo, hi = extract_uniform_control_bounds(constraints)
        solve_kd_builder = _cached_control_bound_solve_kd_builder(
            tuple(np.asarray(lo).tolist()), tuple(np.asarray(hi).tolist())
        )

        neutral_u_lower = jnp.full_like(constraints.u_lower, -jnp.inf)
        neutral_u_upper = jnp.full_like(constraints.u_upper, jnp.inf)
        constraints_for_al = eqx.tree_at(
            lambda c: (c.u_lower, c.u_upper), constraints, (neutral_u_lower, neutral_u_upper)
        )

        fresh_al = ALConstraints.build(constraints_for_al, penalty_initial=options.penalty_initial)
        if state.al is not None:
            lam = fresh_al.lam if options.reset_duals else state.al.lam
            mu = fresh_al.mu if options.reset_penalties else state.al.mu
            init_al = eqx.tree_at(lambda a: (a.lam, a.mu), fresh_al, (lam, mu))
        else:
            init_al = fresh_al

        final_traj, final_al, stats, status_int = _jit_al_solve(
            problem_eff, init_traj, init_al, options, solve_kd_builder=solve_kd_builder, u_bounds=(lo, hi)
        )

        status = TerminationStatus(int(status_int))
        n_iter = int(stats.iterations)
        C, _Jx, _Ju = evaluate_al_constraints(final_al, problem_eff.constraints, problem_eff.model, final_traj)
        final_cost = problem_eff.obj.cost(final_traj) + al_cost(final_al, C)

        trimmed_stats = ALStats(
            iterations=stats.iterations,
            cost=stats.cost[:n_iter],
            c_max=stats.c_max[:n_iter],
            penalty_max=stats.penalty_max[:n_iter],
        )

        return BoxQPSolveResult(
            trajectory=final_traj,
            success=status == TerminationStatus.SOLVE_SUCCEEDED,
            status=int(status_int),
            message=status.name,
            solver_status=to_solver_status(status),
            cost=float(final_cost),
            Z=_trajectory_to_z(final_traj.X, final_traj.U),
            info={"stats": trimmed_stats},
            iterations=n_iter,
            constraint_violation=float(max_violation(final_al, C)),
            al=final_al,
        )
