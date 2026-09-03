from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.problem import MPCState, Problem
from trajopt.solvers.al import AL, ALConstraints, ALResult, _evaluate_al_convergence
from trajopt.solvers.options import SolverOptions, SolverStats, TerminationStatus
from trajopt.trajectory import Trajectory
from trajopt.transcription.result import Solver, SolverResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from trajopt.solvers.ilqr import SolveKD


XF = jnp.array([0.0, np.pi, 0.0, 0.0])


def _cartpole_problem(u_bnd: float = 3.0) -> tuple[Problem, jnp.ndarray, float]:
    """Cartpole swing-up with a symmetric control bound and a terminal goal constraint."""
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    x0 = jnp.zeros(n)
    model = Cartpole()
    obj = LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=XF.tolist()), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())
    return prob, x0, dt


def test_al_satisfies_solver_protocol() -> None:
    """AL structurally satisfies the Solver protocol (has a matching .solve)."""
    assert isinstance(AL(), Solver)


def test_al_result_satisfies_solver_result_protocol() -> None:
    """ALResult structurally satisfies the SolverResult protocol."""
    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    result = AL(options=SolverOptions(iterations=2, iterations_outer=1)).solve(prob, state)
    assert isinstance(result, ALResult)
    assert isinstance(result, SolverResult)


@pytest.mark.slow
def test_al_cartpole_converges_under_constraint_tolerance() -> None:
    """A bounded, goal-constrained cartpole swing-up drives max_violation under tolerance."""
    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=30)

    result = AL(options=options).solve(prob, state)

    assert result.success
    assert result.status == int(TerminationStatus.SOLVE_SUCCEEDED)
    assert result.constraint_violation < options.constraint_tolerance
    np.testing.assert_allclose(np.asarray(result.trajectory.X[-1]), np.asarray(XF), atol=1e-4)
    assert bool(jnp.all(jnp.abs(result.trajectory.U) <= 3.0 + 1e-4))


def test_al_solve_options_stay_untraced_and_hashable(monkeypatch: pytest.MonkeyPatch) -> None:
    """SolverOptions never carries a tracer: ilqr_solve's `options` argument stays hashable under jit.

    Ticket 29 computes the effective (possibly intermediate) cost/gradient tolerance pair as
    traced scalars threaded through the loop carry, not by `dataclasses.replace`-ing them into a
    new `SolverOptions` (which would make that object's fields tracers and break its "static,
    never traced" contract). Patching `ilqr_solve` to assert `hash(options)` succeeds on every
    call -- which fails immediately if any field is a `jax.Array` tracer, since tracers are not
    hashable -- exercises that invariant directly, rather than only checking the caller's own
    `options` object was left untouched (which `dataclasses.replace` guarantees trivially).
    """
    import trajopt.solvers.al as al_module

    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=30)

    real_ilqr_solve = al_module.ilqr_solve
    call_count = 0

    def checked_ilqr_solve(
        problem: Problem,
        trajectory: Trajectory,
        options: SolverOptions,
        *,
        cost_tolerance: jax.Array | float | None = None,
        gradient_tolerance: jax.Array | float | None = None,
        solve_kd_builder: "Callable[[Trajectory], SolveKD] | None" = None,
        u_bounds: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[Trajectory, SolverStats, jax.Array]:
        nonlocal call_count
        call_count += 1
        hash(options)  # raises TypeError if any field is an unhashable jax tracer
        return real_ilqr_solve(
            problem,
            trajectory,
            options,
            cost_tolerance=cost_tolerance,
            gradient_tolerance=gradient_tolerance,
            solve_kd_builder=solve_kd_builder,
            u_bounds=u_bounds,
        )

    monkeypatch.setattr(al_module, "ilqr_solve", checked_ilqr_solve)
    result = AL(options=options).solve(prob, state)

    assert result.success
    assert call_count > 0


def test_al_max_iterations_outer_maps_to_infeasible_status() -> None:
    """The real `.solve()` -> `MPCState.status` path reports "infeasible" on MAX_ITERATIONS_OUTER.

    Regression test for a defect found in mid-point review: `AL.solve()` used to hand
    `TerminationStatus.MAX_ITERATIONS_OUTER.name` as `message` and let `Problem.solve`'s
    `normalize_status` substring-match it, which incorrectly produced "iteration_limit" (the
    substring "iter" matches, but ticket 24's table maps MAX_ITERATIONS_OUTER to "infeasible").
    With `iterations_outer=1`, the cartpole swing-up cannot drive its violation under
    `constraint_tolerance` in a single outer iteration but its inner iLQR solve still succeeds,
    so the outer loop exhausts `MAX_ITERATIONS_OUTER` -- the most common non-convergence outcome
    for a genuinely constrained problem.
    """
    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=1)

    result = AL(options=options).solve(prob, state)
    assert result.status == int(TerminationStatus.MAX_ITERATIONS_OUTER)
    assert result.solver_status == "infeasible"

    new_state = prob.solve(
        MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None), solver=AL(options=options)
    )
    assert new_state.status == "infeasible"


def test_al_populates_mpc_state_al_for_warm_starting() -> None:
    """prob.solve(state, solver=AL()) returns an MPCState whose `al` field carries the final duals."""
    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=30)

    new_state = prob.solve(state, solver=AL(options=options))

    assert new_state.status == "converged"
    assert new_state.al is not None
    assert isinstance(new_state.al, ALConstraints)
    assert bool(jnp.any(new_state.al.lam != 0.0))


@pytest.mark.slow
def test_al_warm_start_converges_in_fewer_outer_iterations_than_cold() -> None:
    """Reusing a prior solve's duals/penalties (reset_duals=False) converges in strictly fewer outer iterations."""
    prob, x0, dt = _cartpole_problem()
    options = SolverOptions(iterations=300, iterations_outer=30, reset_duals=False, reset_penalties=False)

    cold_state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    cold_result = AL(options=options).solve(prob, cold_state)
    assert cold_result.success

    warm_state = prob.solve(
        MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None), solver=AL(options=options)
    )
    warm_result = AL(options=options).solve(prob, warm_state)

    assert warm_result.success
    assert warm_result.iterations < cold_result.iterations


def test_evaluate_al_convergence_last_match_wins_max_outer_beats_solve_succeeded() -> None:
    """Finding A: converging on the same outer iteration that exhausts iterations_outer reports MAX_ITERATIONS_OUTER.

    Both the violation-converged check and the outer-iteration-exhausted check fire on this call;
    since `_evaluate_al_convergence` overwrites `status` sequentially rather than short-circuiting
    on the first match, the later `max_outer_hit` branch silently wins over the earlier
    `converged_violation` branch.
    """
    options = SolverOptions(constraint_tolerance=1e-6, iterations=1000, iterations_outer=5)

    status, done = _evaluate_al_convergence(
        c_max=jnp.asarray(1e-8),
        mu_max=jnp.asarray(10.0),
        inner_iterations=jnp.asarray(3),
        iter_num=jnp.asarray(5),
        options=options,
    )

    assert int(status) == int(TerminationStatus.MAX_ITERATIONS_OUTER)
    assert bool(done)


def test_evaluate_al_convergence_solve_succeeded_when_not_last_outer_iteration() -> None:
    """The same violation convergence, one outer iteration earlier, reports SOLVE_SUCCEEDED."""
    options = SolverOptions(constraint_tolerance=1e-6, iterations=1000, iterations_outer=5)

    status, done = _evaluate_al_convergence(
        c_max=jnp.asarray(1e-8),
        mu_max=jnp.asarray(10.0),
        inner_iterations=jnp.asarray(3),
        iter_num=jnp.asarray(4),
        options=options,
    )

    assert int(status) == int(TerminationStatus.SOLVE_SUCCEEDED)
    assert bool(done)


def test_evaluate_al_convergence_max_iterations_beats_solve_succeeded() -> None:
    """Last-match-wins also lets MAX_ITERATIONS overwrite an earlier SOLVE_SUCCEEDED match."""
    options = SolverOptions(constraint_tolerance=1e-6, iterations=3, iterations_outer=30)

    status, done = _evaluate_al_convergence(
        c_max=jnp.asarray(1e-8),
        mu_max=jnp.asarray(10.0),
        inner_iterations=jnp.asarray(3),
        iter_num=jnp.asarray(2),
        options=options,
    )

    assert int(status) == int(TerminationStatus.MAX_ITERATIONS)
    assert bool(done)


def test_evaluate_al_convergence_kickout_max_penalty_stops_without_setting_status() -> None:
    """Finding B: kickout_max_penalty ends the loop when mu_max saturates, without writing status.

    Hand-written expectation, not a Julia cross test: Altro's own `kickout_max_penalty` branch
    references an undefined loop variable and throws, so it cannot be exercised against Julia.
    Here `c_max` is deliberately left unconverged and `iter_num`/`inner_iterations` well under
    their limits, isolating the kickout branch as the only thing that can set `done`.
    """
    options = SolverOptions(
        constraint_tolerance=1e-6,
        iterations=1000,
        iterations_outer=30,
        kickout_max_penalty=True,
        penalty_max=1e6,
    )

    status, done = _evaluate_al_convergence(
        c_max=jnp.asarray(1.0),
        mu_max=jnp.asarray(1e6),
        inner_iterations=jnp.asarray(5),
        iter_num=jnp.asarray(3),
        options=options,
    )

    assert bool(done)
    assert int(status) == int(TerminationStatus.UNSOLVED)


def test_evaluate_al_convergence_kickout_disabled_does_not_stop() -> None:
    """The same saturated-penalty scenario does not stop the loop when kickout_max_penalty is off."""
    options = SolverOptions(
        constraint_tolerance=1e-6,
        iterations=1000,
        iterations_outer=30,
        kickout_max_penalty=False,
        penalty_max=1e6,
    )

    status, done = _evaluate_al_convergence(
        c_max=jnp.asarray(1.0),
        mu_max=jnp.asarray(1e6),
        inner_iterations=jnp.asarray(5),
        iter_num=jnp.asarray(3),
        options=options,
    )

    assert not bool(done)
    assert int(status) == int(TerminationStatus.UNSOLVED)


def test_al_solve_breaks_on_ordinal_inner_status_without_updating_duals() -> None:
    """finding C: an inner solve that exhausts options.iterations (status > SOLVE_SUCCEEDED) breaks the outer loop.

    With options.iterations=1, the nonlinear cartpole swing-up cannot converge on its first iLQR
    iteration, so the inner solve exits MAX_ITERATIONS (ordinal 3, greater than SOLVE_SUCCEEDED's
    2). The outer loop must propagate that status directly and must not have recorded any stats
    or run a dual/penalty update for the (failed) first outer iteration.
    """
    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=1, iterations_outer=30)

    result = AL(options=options).solve(prob, state)

    assert not result.success
    assert result.status == int(TerminationStatus.MAX_ITERATIONS)
    assert result.iterations == 0
    assert result.al is not None
    assert bool(jnp.all(result.al.lam == 0.0))


def test_evaluate_al_convergence_max_iters_uses_this_iterations_inner_count() -> None:
    """max_iters_hit reads this outer iteration's own (reset) inner iteration count, not a cumulative total."""
    options = SolverOptions(constraint_tolerance=1e-6, iterations=10, iterations_outer=30)

    status, done = _evaluate_al_convergence(
        c_max=jnp.asarray(1.0),
        mu_max=jnp.asarray(1.0),
        inner_iterations=jnp.asarray(10),
        iter_num=jnp.asarray(2),
        options=options,
    )
    assert int(status) == int(TerminationStatus.MAX_ITERATIONS)
    assert bool(done)

    status2, done2 = _evaluate_al_convergence(
        c_max=jnp.asarray(1.0),
        mu_max=jnp.asarray(1.0),
        inner_iterations=jnp.asarray(9),
        iter_num=jnp.asarray(2),
        options=options,
    )
    assert int(status2) == int(TerminationStatus.UNSOLVED)
    assert not bool(done2)


@pytest.mark.parametrize("u_bnd", [3.0])
def test_al_stats_history_length_matches_iterations(u_bnd: float) -> None:
    """The trimmed ALStats history in .info["stats"] has exactly `iterations` entries, no trailing zeros."""
    prob, x0, dt = _cartpole_problem(u_bnd=u_bnd)
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=30)

    result = AL(options=options).solve(prob, state)
    stats = result.info["stats"]

    assert stats.cost.shape[0] == result.iterations
    assert stats.c_max.shape[0] == result.iterations
    assert stats.penalty_max.shape[0] == result.iterations
    assert float(stats.c_max[-1]) == pytest.approx(result.constraint_violation, abs=1e-8)
