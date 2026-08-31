import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.problem import MPCState, Problem
from trajopt.solvers._jit_cache import JitCacheSlot
from trajopt.solvers.al import (
    ALConstraints,
    ALStats,
    al_solve,
    evaluate_al_constraints,
    max_violation,
)
from trajopt.solvers.al import _trim_al_stats as trim_al_stats
from trajopt.solvers.ilqr import _jit_ilqr_solve, build_warm_start
from trajopt.solvers.ilqr import _trim_stats as trim_ilqr_stats
from trajopt.solvers.options import SolverOptions, TerminationStatus, to_solver_status
from trajopt.solvers.pn import PNStats, pn_solve
from trajopt.solvers.pn import _trim_pn_stats as trim_pn_stats
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z, parse_solver_initial_state
from trajopt.transcription.result import SolverStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from trajopt.solvers.ilqr import SolveKD

_EMPTY = np.zeros(0, dtype=np.float64)


def _al_phase_tolerance(options: SolverOptions) -> tuple[float, bool]:
    """Compute the AL phase's effective constraint tolerance and kickout flag, `altro_solve.jl`'s pre-solve block.

    Both are plain Python values derived only from `options` (static config), computed once up
    front rather than by mutating `opts.constraint_tolerance` and restoring it afterward (ticket
    33: "tolerance mutation becomes a threaded value"). When `options.projected_newton` is off,
    AL keeps the real `constraint_tolerance` and `kickout_max_penalty` as the caller set it. When
    it is on, a non-negative `projected_newton_tolerance` loosens AL's tolerance to it (so AL
    stops early and leaves polish to PN); a negative one sets AL's tolerance to zero and forces
    `kickout_max_penalty` on (ticket 29's `kickout_max_penalty` branch, exercised for real here).
    """
    if not options.projected_newton:
        return options.constraint_tolerance, options.kickout_max_penalty
    if options.projected_newton_tolerance >= 0:
        return options.projected_newton_tolerance, options.kickout_max_penalty
    return 0.0, True


class ALTROSolveResult(NamedTuple):
    """Traced output of `altro_solve`, one Python-level intermediate the eager `ALTRO.solve()` boundary-converts.

    Parameters
    ----------
    trajectory : Trajectory
        Final trajectory: PN's projected trajectory when the polish phase ran, else AL's own.
    status : jax.Array
        Exit `TerminationStatus` ordinal as an int32 scalar, possibly upgraded to
        `SOLVE_SUCCEEDED` by the backup check.
    al : ALConstraints
        AL's final duals and penalties, unmodified by PN, for `MPCState.al` warm-starting.
    al_stats : ALStats
        AL phase's outer stats history, buffers sized `options.iterations_outer`, untrimmed.
    pn_stats : PNStats
        PN phase's outer stats history, buffers sized `options.n_steps + 1`, untrimmed --
        meaningless (all zero) when `ran_pn` is False, since PN still runs under trace but its
        result is discarded.
    pn_duals : jax.Array
        PN's final multiplier-projection dual vector in its own row order, meaningless (zero) when
        `ran_pn` is False or `options.multiplier_projection` is False.
    ran_pn : jax.Array
        Whether the polish phase's result was actually selected, as a bool scalar.
    c_max : jax.Array
        Backup check's constraint violation on the final trajectory, using AL's own duals'
        structural row layout (independent of lambda/mu).
    """

    trajectory: Trajectory
    status: jax.Array
    al: ALConstraints
    al_stats: ALStats
    pn_stats: PNStats
    pn_duals: jax.Array
    ran_pn: jax.Array
    c_max: jax.Array


def altro_solve(  # noqa: PLR0913, PLR0917 -- ticket 30's solve_kd_builder/u_bounds are load-bearing, forwarded to the AL phase
    problem: Problem,
    trajectory: Trajectory,
    al0: ALConstraints,
    x0: jax.Array,
    options: SolverOptions,
    solve_kd_builder: "Callable[[Trajectory], SolveKD] | None" = None,
    u_bounds: "tuple[jax.Array, jax.Array] | None" = None,
) -> ALTROSolveResult:
    """Traced two-phase ALTRO driver, matching `altro_solve.jl`'s `solve!` past its unconstrained shortcut.

    Assumes `problem` is constrained -- `ALTRO.solve()` takes the unconstrained iLQR shortcut
    itself, before ever building `al0` (acceptance: no AL/PN state is constructed for an
    unconstrained problem), so this function never sees that branch.

    Runs the AL phase at a possibly loosened tolerance (`_al_phase_tolerance`), then always
    evaluates PN's projection (its shapes are static regardless of whether the result is needed)
    and selects between AL's and PN's trajectories with `jnp.where(run_pn, ...)` rather than
    `lax.cond`, matching this codebase's established style for traced branches (e.g. `al.py`'s
    `inner_failed` handling) and keeping the whole driver `jax.jit`/`jax.vmap`-able end to end.

    Reproduces finding I's two corrections exactly:

    - The backup check's status upgrade requires `al_status <= SOLVE_SUCCEEDED`, dropped nowhere:
      a `MAX_ITERATIONS_OUTER` (or worse) exit is never upgraded to `SOLVE_SUCCEEDED`, even when
      PN drives the violation under tolerance. Reproduced here, not fixed, because parity tests
      hold this port to Altro's own (arguably wrong) behaviour.
    - `c_max`, the value that decides whether PN needs to run at all, is read from the AL stats
      cache (`al_stats.c_max[al_stats.iterations - 1]`) whenever `al_stats.iterations > 1`, and
      recomputed from the constraints directly only otherwise -- the two can differ, since the
      cached value predates the last inner iLQR solve of that outer iteration.

    Julia's `opts.force_pn` and the `status <= SOLVE_SUCCEEDED || force_pn` gate wrap the *whole*
    polish block, including the inner `status ∈ {≤SUCCEEDED, MAX_ITERATIONS_OUTER}` check. Because
    `MAX_ITERATIONS_OUTER`'s ordinal (4) is not `<= SOLVE_SUCCEEDED` (2), that outer gate excludes
    a `MAX_ITERATIONS_OUTER` exit unless `force_pn` is set -- making the inner check's own
    `MAX_ITERATIONS_OUTER` branch dead upstream except when `force_pn` already forces `run_pn`
    regardless of status. `run_pn` below reproduces both levels explicitly (`outer_gate` then
    `inner_condition`) rather than only the inner one, so PN does not run on a `MAX_ITERATIONS_OUTER`
    exit unless `force_pn` is set.

    Parameters
    ----------
    problem : Problem
        Supplies the model, unconstrained objective, and constraints both phases assemble from.
    trajectory : Trajectory
        Warm-start guess passed to the AL phase's first inner iLQR solve.
    al0 : ALConstraints
        Initial duals and penalties, already reset or warm-started by the caller.
    x0 : jax.Array
        Fixed initial condition of shape (n,), pinned by PN's initial-condition equality row.
    options : SolverOptions
        Static solve configuration; must not be traced. The *real* `constraint_tolerance` --
        `_al_phase_tolerance` derives the AL phase's own loosened value from a copy of this.
    solve_kd_builder : Callable[[Trajectory], SolveKD] | None, optional
        Forwarded to the AL phase's inner `ilqr_solve` calls (ticket 30's box-QP hook). Defaults
        to None.
    u_bounds : tuple[jax.Array, jax.Array] | None, optional
        Forwarded to the AL phase to clip its closed-loop rollout (ticket 30). Defaults to None.

    Returns
    -------
    ALTROSolveResult
        See field docstrings.
    """
    al_tol, kickout = _al_phase_tolerance(options)
    al_options = dataclasses.replace(options, constraint_tolerance=al_tol, kickout_max_penalty=kickout)

    al_traj, al_final, al_stats, al_status = al_solve(problem, trajectory, al0, al_options, solve_kd_builder, u_bounds)

    success_ord = jnp.int32(TerminationStatus.SOLVE_SUCCEEDED)
    al_status_ok = al_status <= success_ord

    n_iter = al_stats.iterations
    cache_idx = jnp.clip(n_iter - 1, 0, al_stats.c_max.shape[0] - 1)
    C_al, _Jx_al, _Ju_al = evaluate_al_constraints(al_final, problem.constraints, problem.model, al_traj)
    recomputed_c_max = max_violation(al_final, C_al)
    c_max = jnp.where(n_iter > 1, al_stats.c_max[cache_idx], recomputed_c_max)

    force_pn = jnp.asarray(options.force_pn)
    outer_gate = al_status_ok | force_pn
    inner_condition = (
        jnp.asarray(options.projected_newton)
        & (c_max > options.constraint_tolerance)
        & (al_status_ok | (al_status == jnp.int32(TerminationStatus.MAX_ITERATIONS_OUTER)))
    ) | force_pn
    run_pn = outer_gate & inner_condition

    pn_traj, pn_stats, pn_duals, _pn_status = pn_solve(problem, al_traj, x0, options)

    final_traj = jax.tree.map(lambda p, a: jnp.where(run_pn, p, a), pn_traj, al_traj)

    C_backup, _Jx_bk, _Ju_bk = evaluate_al_constraints(al_final, problem.constraints, problem.model, final_traj)
    c_max2 = max_violation(al_final, C_backup)
    upgrade = al_status_ok & (c_max2 < options.constraint_tolerance)
    final_status = jnp.where(upgrade, success_ord, al_status)

    return ALTROSolveResult(
        trajectory=final_traj,
        status=final_status,
        al=al_final,
        al_stats=al_stats,
        pn_stats=pn_stats,
        pn_duals=pn_duals,
        ran_pn=run_pn,
        c_max=c_max2,
    )


_altro_solve_jit_slot = JitCacheSlot()


def _jit_altro_solve(
    problem: Problem, trajectory: Trajectory, al0: ALConstraints, x0: jax.Array, options: SolverOptions
) -> ALTROSolveResult:
    """`altro_solve`, jit-compiled and cached per `(problem identity, options)`, called from `ALTRO.solve()`'s constrained branch.

    `problem` is closed over via `functools.partial` rather than passed as a jit argument
    (`JitCacheSlot`'s docstring has the reason). `ALTRO.solve()` never forwards a
    `solve_kd_builder`/`u_bounds` (ticket 30's box-QP hook is not wired into the ALTRO driver),
    so those stay at `altro_solve`'s own `None` defaults here too. The returned closure is reused
    across calls with the same `problem` object and `options`, so repeated same-shape calls (e.g.
    MPC) hit XLA's compilation cache instead of recompiling.
    """
    jitted = _altro_solve_jit_slot.get_or_build(
        altro_solve, problem, key=options, options=options, solve_kd_builder=None
    )
    return jitted(trajectory=trajectory, al0=al0, x0=x0)


class ALTROResult(NamedTuple):
    """Result of a native ALTRO solve, satisfying the `SolverResult` protocol.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the driver exited with `TerminationStatus.SOLVE_SUCCEEDED`.
    status : int
        `TerminationStatus` ordinal the traced driver exited with.
    message : str
        `TerminationStatus` member name.
    solver_status : SolverStatus
        `status` mapped through `to_solver_status`'s table.
    cost : float
        Final base-objective value at the returned trajectory (no AL penalty term, matching
        `PNResult.cost` rather than `ALResult.cost`: the trajectory a user reads off ALTRO is
        whichever phase produced it last, and the base cost is what's comparable across phases).
    Z : jax.Array
        Optimal flat primal vector.
    info : dict[str, Any]
        Holds `"al_stats"` (trimmed `ALStats`), `"pn_stats"` (trimmed `PNStats`, or None when the
        polish phase's result was not selected), and `"ran_pn"` (bool).
    constraint_violation : float
        Final `max_violation` over the returned trajectory (post-PN when PN ran).
    iterations : int, optional
        Number of completed AL outer iterations. Defaults to 0.
    lam : np.ndarray, optional
        Always empty, for the same reason as `ALResult.lam`. Defaults to empty.
    mu : np.ndarray, optional
        Always empty, for the same reason as `ALResult.mu`. Defaults to empty.
    al : ALConstraints | None, optional
        Final AL duals and penalties (unmodified by PN), threaded into `MPCState.al` for
        warm-starting the next solve. None for an unconstrained problem's iLQR-shortcut solve.
        Defaults to None.
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
    al: ALConstraints | None = None


@dataclass(frozen=True)
class ALTRO:
    """Native ALTRO solver backend, satisfying the `Solver` protocol -- a driver, not an algorithm.

    A thin eager wrapper composing the traced `ilqr_solve` (ticket 27), `al_solve` (ticket 29),
    and `pn_solve` (ticket 32) cores through the `altro_solve` driver above: `.solve()` builds the
    warm-start trajectory from `state`, takes the unconstrained shortcut (bare `ilqr_solve`, no AL
    or PN state constructed at all) when `problem`'s constraints are structurally empty, else
    builds or warm-starts the initial padded duals/penalties exactly as `AL.solve` does and calls
    the jitted `altro_solve` core, then converts the traced status int and stats buffers into
    `success` / `message` / `info` at the boundary -- work that cannot happen inside a trace.

    Parameters
    ----------
    options : SolverOptions, optional
        Static solve configuration. Defaults to `SolverOptions()`.
    """

    options: SolverOptions = field(default_factory=SolverOptions)

    def solve(self, problem: Problem, state: MPCState) -> ALTROResult:
        """Run the traced ALTRO driver from `state`'s warm-start trajectory/duals and boundary-convert the result.

        Raises
        ------
        ValueError
            `state.al` carries duals built under the opposite `use_conic_cost` convention and
            `options.reset_duals` is False -- identical guard to `AL.solve` (finding E).
        """
        options = self.options
        problem_eff, init_traj = build_warm_start(problem, state)

        if problem_eff.constraints.is_unconstrained():
            final_traj, stats, status_int = _jit_ilqr_solve(problem_eff, init_traj, options)
            status = TerminationStatus(int(status_int))
            n_iter = int(stats.iterations)
            return ALTROResult(
                trajectory=final_traj,
                success=status == TerminationStatus.SOLVE_SUCCEEDED,
                status=int(status_int),
                message=status.name,
                solver_status=to_solver_status(status),
                cost=float(problem_eff.obj.cost(final_traj)),
                Z=_trajectory_to_z(final_traj.X, final_traj.U),
                info={"stats": trim_ilqr_stats(stats, n_iter), "ran_pn": False},
                constraint_violation=0.0,  # no rows and no bounds: nothing here can be violated
                iterations=n_iter,
                al=None,
            )

        fresh_al = ALConstraints.build(
            problem_eff.constraints, penalty_initial=options.penalty_initial, use_conic_cost=options.use_conic_cost
        )
        if state.al is not None:
            if not options.reset_duals and bool(state.al.is_conic) != options.use_conic_cost:
                msg = (
                    f"state.al was built with use_conic_cost={bool(state.al.is_conic)}, but "
                    f"options.use_conic_cost={options.use_conic_cost}. The two conventions store "
                    "lambda with opposite signs (finding E), so warm-starting across the switch "
                    "would silently reinterpret it. Set options.reset_duals=True to discard the "
                    "old duals, or keep use_conic_cost consistent with the state that produced them."
                )
                raise ValueError(msg)
            lam = fresh_al.lam if options.reset_duals else state.al.lam
            mu = fresh_al.mu if options.reset_penalties else state.al.mu
            init_al = eqx.tree_at(lambda a: (a.lam, a.mu), fresh_al, (lam, mu))
        else:
            init_al = fresh_al

        x0_arr, _t0_arr, _dt_arr, _xf_val, _z0 = parse_solver_initial_state(state)

        result = _jit_altro_solve(problem_eff, init_traj, init_al, x0_arr, options)

        status = TerminationStatus(int(result.status))
        n_iter_al = int(result.al_stats.iterations)
        n_iter_pn = int(result.pn_stats.iterations)
        ran_pn = bool(result.ran_pn)

        return ALTROResult(
            trajectory=result.trajectory,
            success=status == TerminationStatus.SOLVE_SUCCEEDED,
            status=int(result.status),
            message=status.name,
            solver_status=to_solver_status(status),
            cost=float(problem_eff.obj.cost(result.trajectory)),
            Z=_trajectory_to_z(result.trajectory.X, result.trajectory.U),
            info={
                "al_stats": trim_al_stats(result.al_stats, n_iter_al),
                "pn_stats": trim_pn_stats(result.pn_stats, n_iter_pn) if ran_pn else None,
                "ran_pn": ran_pn,
            },
            iterations=n_iter_al,
            constraint_violation=float(result.c_max),
            al=result.al,
        )
