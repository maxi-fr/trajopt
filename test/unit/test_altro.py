import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.solvers.al import ALConstraints, ALStats, evaluate_al_constraints, max_violation
from trajopt.solvers.altro import (
    ALTRO,
    ALTROResult,
    _al_phase_tolerance,
    altro_solve,
)
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory
from trajopt.transcription.result import Solver, SolverResult


def _cartpole_problem(u_bnd: float = 3.0, N: int = 101) -> tuple[Problem, jnp.ndarray, float, jnp.ndarray]:
    """Cartpole swing-up with a symmetric control bound and a terminal goal constraint, plus its goal."""
    n, m = 4, 1
    tf = 5.0 * (N - 1) / 100.0 if N > 1 else 5.0
    dt = tf / (N - 1) if N > 1 else 0.05
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    x0 = jnp.zeros(n)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    model = Cartpole()
    obj = LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    if N > 1:
        clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=xf.tolist()), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, dt=dt, integrator=RK4())
    return prob, x0, dt, xf


def _lq_problem() -> tuple[Problem, jnp.ndarray, float, jnp.ndarray]:
    """A small unconstrained pendulum swing-up problem, for the iLQR-shortcut path, plus its goal."""
    model = Pendulum()
    N = 21
    Q = jnp.diag(jnp.array([1.0, 0.1]))
    R = jnp.eye(1) * 0.01
    Qf = jnp.diag(jnp.array([10.0, 1.0]))
    xf = jnp.array([jnp.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, N=N)
    prob = Problem(model=model, obj=obj, N=N, dt=0.05, integrator=RK4())
    x0 = jnp.array([0.0, 0.0])
    return prob, x0, 0.05, xf


def test_altro_satisfies_solver_protocol() -> None:
    """ALTRO structurally satisfies the Solver protocol (has a matching .solve)."""
    assert isinstance(ALTRO(), Solver)


def test_altro_result_satisfies_solver_result_protocol() -> None:
    """ALTROResult structurally satisfies the SolverResult protocol."""
    prob, x0, _dt, xf = _lq_problem()
    result = MPC(prob, ALTRO(), x0=x0, xf=xf).solve()
    assert isinstance(result, ALTROResult)
    assert isinstance(result, SolverResult)


def test_altro_unconstrained_takes_ilqr_shortcut_without_al_or_pn_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconstrained problem never calls al_solve/pn_solve at all -- no AL or PN state is built."""
    import trajopt.solvers.altro as altro_module

    def fail_al_solve(*_args: object, **_kwargs: object) -> None:
        msg = "al_solve must not be called for an unconstrained problem"
        raise AssertionError(msg)

    def fail_pn_solve(*_args: object, **_kwargs: object) -> None:
        msg = "pn_solve must not be called for an unconstrained problem"
        raise AssertionError(msg)

    monkeypatch.setattr(altro_module, "al_solve", fail_al_solve)
    monkeypatch.setattr(altro_module, "pn_solve", fail_pn_solve)

    prob, x0, _dt, xf = _lq_problem()
    result = MPC(prob, ALTRO(), x0=x0, xf=xf).solve()

    assert isinstance(result, ALTROResult)
    assert result.success
    assert result.al is None
    assert result.info["ran_pn"] is False
    assert "stats" in result.info


def test_is_unconstrained_true_for_bare_problem() -> None:
    """A Problem built with no constraints and no box bounds is structurally unconstrained."""
    prob, _x0, _dt, _xf = _lq_problem()
    assert prob.constraints.is_unconstrained()


def test_is_unconstrained_false_with_control_bound() -> None:
    """A Problem with a ControlBound is not structurally unconstrained."""
    prob, _x0, _dt, _xf = _cartpole_problem(N=5)
    assert not prob.constraints.is_unconstrained()


def test_al_phase_tolerance_loosened_to_projected_newton_tolerance() -> None:
    """With projected_newton on and its tolerance >= 0, AL gets that tolerance, kickout untouched."""
    options = SolverOptions(projected_newton=True, projected_newton_tolerance=1e-3, kickout_max_penalty=False)
    tol, kickout = _al_phase_tolerance(options)
    assert tol == pytest.approx(1e-3)
    assert kickout is False


def test_al_phase_tolerance_negative_projected_newton_tolerance_turns_on_kickout() -> None:
    """With projected_newton on and its tolerance < 0, AL tolerance drops to 0, kickout forced True."""
    options = SolverOptions(projected_newton=True, projected_newton_tolerance=-1.0, kickout_max_penalty=False)
    tol, kickout = _al_phase_tolerance(options)
    assert tol == 0.0
    assert kickout is True


def test_al_phase_tolerance_projected_newton_off_leaves_tolerance_and_kickout_untouched() -> None:
    """With projected_newton off, AL gets the real constraint_tolerance, kickout untouched."""
    options = SolverOptions(
        projected_newton=False,
        constraint_tolerance=1e-5,
        projected_newton_tolerance=-1.0,
        kickout_max_penalty=True,
    )
    tol, kickout = _al_phase_tolerance(options)
    assert tol == pytest.approx(1e-5)
    assert kickout is True


@pytest.mark.slow
def test_altro_cartpole_reaches_tight_violation_via_pn() -> None:
    """A bound + goal constrained cartpole solve reaches max_violation < constraint_tolerance, with PN polishing.

    AL alone (ticket 29's own test) only drives the violation to roughly `projected_newton_tolerance`
    (1e-3); the real `constraint_tolerance` default (1e-6) is tighter, so PN must actually run and
    do the polishing for this to pass.
    """
    prob, x0, _dt, xf = _cartpole_problem()
    options = SolverOptions(iterations=300, iterations_outer=30)

    result = MPC(prob, ALTRO(options=options), x0=x0, xf=xf).solve()

    assert result.success
    assert result.status == int(TerminationStatus.SOLVE_SUCCEEDED)
    assert result.constraint_violation < options.constraint_tolerance
    assert result.info["ran_pn"] is True
    np.testing.assert_allclose(np.asarray(result.trajectory.X[-1]), np.asarray(xf), atol=1e-6)
    assert bool(jnp.all(jnp.abs(result.trajectory.U) <= 3.0 + 1e-6))


@pytest.mark.slow
def test_altro_negative_projected_newton_tolerance_works_end_to_end_via_kickout() -> None:
    """`projected_newton_tolerance < 0` drives AL's own tolerance to 0 and turns on kickout_max_penalty.

    AL's loosened tolerance can never be satisfied by `c_max < 0`, so the only way the AL phase
    ends before exhausting `iterations_outer` is `kickout_max_penalty` firing once mu saturates at
    penalty_max -- ticket 29's branch, exercised here for real for the first time (ticket 29's own
    test is necessarily hand-written since Altro's own branch throws). In this scenario AL's own
    rollout is already accurate enough that the resulting violation lands under the real
    constraint_tolerance without PN's help, so it is `al_iterations < iterations_outer` -- not
    `ran_pn` -- that proves the kickout path actually fired.
    """
    prob, x0, _dt, xf = _cartpole_problem()
    options = SolverOptions(iterations=300, iterations_outer=30, projected_newton_tolerance=-1.0)

    result = MPC(prob, ALTRO(options=options), x0=x0, xf=xf).solve()

    assert result.iterations < options.iterations_outer  # exited via kickout, not iteration exhaustion
    assert result.success
    assert result.constraint_violation < options.constraint_tolerance


def test_altro_backup_check_does_not_upgrade_max_iterations_outer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding I's guard: a MAX_ITERATIONS_OUTER exit is never upgraded even when the final violation is converged.

    `al_solve` is replaced with a stub that returns an already-feasible trajectory (goal hit
    exactly, control within bound) but reports `TerminationStatus.MAX_ITERATIONS_OUTER` -- the
    same shape a real AL solve that ran out of outer iterations while genuinely still improving
    would have. The backup check's own recomputed violation is then far under tolerance, but the
    ordinal guard `al_status <= SOLVE_SUCCEEDED` must still block the upgrade.
    """
    import trajopt.solvers.altro as altro_module

    prob, x0, dt, xf = _cartpole_problem(N=5)
    N, m = prob.N, prob.model.m
    t = jnp.arange(N) * dt
    dt_arr = jnp.full(N - 1, dt)
    X = jnp.linspace(x0, xf, N)
    U = jnp.zeros((N - 1, m))
    feasible_traj = Trajectory(X=X, U=U, t=t, dt=dt_arr)

    al0 = ALConstraints.build(prob.constraints, penalty_initial=SolverOptions().penalty_initial)
    options = SolverOptions(iterations=2, iterations_outer=2, n_steps=1)

    def fake_al_solve(
        _problem: Problem, _trajectory: Trajectory, al: ALConstraints, opts: SolverOptions, *_extra: object
    ) -> tuple[Trajectory, ALConstraints, ALStats, jax.Array]:
        stats = ALStats.create(opts)
        stats = ALStats(iterations=jnp.int32(1), cost=stats.cost, c_max=stats.c_max, penalty_max=stats.penalty_max)
        return feasible_traj, al, stats, jnp.int32(TerminationStatus.MAX_ITERATIONS_OUTER)

    monkeypatch.setattr(altro_module, "al_solve", fake_al_solve)

    C, _Jx, _Ju = evaluate_al_constraints(al0, prob.constraints, prob.model, feasible_traj)
    assert float(max_violation(al0, C)) < options.constraint_tolerance  # sanity: genuinely feasible

    x0_arr = jnp.asarray(x0)
    result = altro_solve(prob, feasible_traj, al0, x0_arr, options)

    assert int(result.status) == int(TerminationStatus.MAX_ITERATIONS_OUTER)
    assert int(result.status) != int(TerminationStatus.SOLVE_SUCCEEDED)


def test_altro_pn_does_not_run_on_max_iterations_outer_without_force_pn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The outer gate (`al_status_ok | force_pn`) must exclude a `MAX_ITERATIONS_OUTER` exit from running PN.

    `al_solve` is replaced with a stub that reports `MAX_ITERATIONS_OUTER` on a genuinely
    infeasible trajectory (control far outside the bound, so `c_max > constraint_tolerance`).
    Altro's own two-level gate (`docs/altro-jl-reference.md` §6: `if status <= SOLVE_SUCCEEDED or
    force_pn: ... if (... status in {<=SUCCEEDED, MAX_ITERATIONS_OUTER}) or force_pn: run PN`)
    never reaches the inner PN-triggering check at all when the outer gate is closed, since
    `MAX_ITERATIONS_OUTER`'s ordinal is not `<= SOLVE_SUCCEEDED`. With `force_pn` at its default
    `False`, PN must not run and the returned trajectory must be AL's own, not PN's projection --
    the case the earlier `test_altro_backup_check_does_not_upgrade_max_iterations_outer` stub
    (already feasible, so `c_max > constraint_tolerance` was never true) could not exercise.
    """
    import trajopt.solvers.altro as altro_module

    prob, x0, dt, _xf = _cartpole_problem(N=5)
    N, m = prob.N, prob.model.m
    t = jnp.arange(N) * dt
    dt_arr = jnp.full(N - 1, dt)
    X = jnp.repeat(x0[None, :], N, axis=0)
    U = jnp.full((N - 1, m), 100.0)  # far outside the +-3.0 control bound
    infeasible_traj = Trajectory(X=X, U=U, t=t, dt=dt_arr)

    al0 = ALConstraints.build(prob.constraints, penalty_initial=SolverOptions().penalty_initial)
    options = SolverOptions(iterations=2, iterations_outer=2, n_steps=1)

    def fake_al_solve(
        _problem: Problem, _trajectory: Trajectory, al: ALConstraints, opts: SolverOptions, *_extra: object
    ) -> tuple[Trajectory, ALConstraints, ALStats, jax.Array]:
        stats = ALStats.create(opts)
        stats = ALStats(iterations=jnp.int32(1), cost=stats.cost, c_max=stats.c_max, penalty_max=stats.penalty_max)
        return infeasible_traj, al, stats, jnp.int32(TerminationStatus.MAX_ITERATIONS_OUTER)

    monkeypatch.setattr(altro_module, "al_solve", fake_al_solve)

    C, _Jx, _Ju = evaluate_al_constraints(al0, prob.constraints, prob.model, infeasible_traj)
    assert float(max_violation(al0, C)) > options.constraint_tolerance  # sanity: genuinely infeasible

    x0_arr = jnp.asarray(x0)
    result = altro_solve(prob, infeasible_traj, al0, x0_arr, options)

    assert not bool(result.ran_pn)
    np.testing.assert_array_equal(np.asarray(result.trajectory.U), np.asarray(infeasible_traj.U))
    assert int(result.status) == int(TerminationStatus.MAX_ITERATIONS_OUTER)


def test_altro_c_max_uses_stats_cache_when_iterations_gt_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_pn`'s decision reads the cached `al_stats.c_max` entry once `iterations > 1`, not a fresh recompute.

    A genuinely infeasible trajectory (control far outside the bound) is paired with a stub
    `al_solve` that reports `iterations=1` in one case (forcing a fresh recompute, which sees the
    real large violation and asks for PN) and `iterations=2` with a stale near-zero cached entry
    in the other (which must be believed instead, so PN does not get requested).
    """
    import trajopt.solvers.altro as altro_module

    prob, x0, dt, _xf = _cartpole_problem(N=5)
    N, m = prob.N, prob.model.m
    t = jnp.arange(N) * dt
    dt_arr = jnp.full(N - 1, dt)
    X = jnp.repeat(x0[None, :], N, axis=0)
    U = jnp.full((N - 1, m), 100.0)  # far outside the +-3.0 control bound
    infeasible_traj = Trajectory(X=X, U=U, t=t, dt=dt_arr)

    al0 = ALConstraints.build(prob.constraints, penalty_initial=SolverOptions().penalty_initial)
    options = SolverOptions(iterations=2, iterations_outer=2, n_steps=1)
    x0_arr = jnp.asarray(x0)

    def make_fake_al_solve(n_iter: int, cached_c_max: float) -> object:
        def fake_al_solve(
            _problem: Problem, _trajectory: Trajectory, al: ALConstraints, opts: SolverOptions, *_extra: object
        ) -> tuple[Trajectory, ALConstraints, ALStats, jax.Array]:
            stats = ALStats.create(opts)
            c_max_hist = stats.c_max.at[n_iter - 1].set(cached_c_max)
            stats = ALStats(
                iterations=jnp.int32(n_iter), cost=stats.cost, c_max=c_max_hist, penalty_max=stats.penalty_max
            )
            return infeasible_traj, al, stats, jnp.int32(TerminationStatus.SOLVE_SUCCEEDED)

        return fake_al_solve

    monkeypatch.setattr(altro_module, "al_solve", make_fake_al_solve(1, cached_c_max=1e-9))
    result_recompute = altro_solve(prob, infeasible_traj, al0, x0_arr, options)
    assert bool(result_recompute.ran_pn)  # recompute sees the real (large) violation

    monkeypatch.setattr(altro_module, "al_solve", make_fake_al_solve(2, cached_c_max=1e-9))
    result_cached = altro_solve(prob, infeasible_traj, al0, x0_arr, options)
    assert not bool(result_cached.ran_pn)  # cached (stale, near-zero) value is believed instead


def test_altro_solve_reuses_jitted_closure_across_repeated_calls_on_same_problem() -> None:
    """Two `.solve()` calls on the same driver reuse the same compiled `jax.jit` closure.

    Verifies fix 2 (ticket 34): the solver's `Program` hands back the core it already built on a
    repeat call (same `problem` identity, same `options`) instead of compiling a fresh `jax.jit`
    wrapper every `.solve()` call -- the MPC-loop regime the Program exists for.
    """
    prob, x0, _dt, xf = _cartpole_problem(N=5)
    altro = ALTRO(options=SolverOptions(iterations=2, iterations_outer=2, n_steps=1))

    mpc = MPC(prob, altro, x0=x0, xf=xf)
    _ = mpc.solve()
    program = mpc.program
    cores_after_first = dict(program._cores)  # noqa: SLF001 -- white-box core-reuse check
    assert len(cores_after_first) == 1

    _ = mpc.solve()

    assert mpc.program is program
    assert program._cores == cores_after_first  # noqa: SLF001 -- white-box core-reuse check


def test_altro_solve_is_jittable_and_vmappable_with_static_options() -> None:
    """altro_solve runs unchanged under jax.jit, and vmaps over a batch of initial states."""
    prob, x0, dt, _xf = _cartpole_problem(N=5)
    N, m = prob.N, prob.model.m
    options = SolverOptions(iterations=2, iterations_outer=2, n_steps=1)
    al0 = ALConstraints.build(prob.constraints, penalty_initial=options.penalty_initial)

    def make_traj(x0_: jax.Array) -> Trajectory:
        X0 = jnp.repeat(x0_[None, :], N, axis=0)
        U0 = jnp.zeros((N - 1, m))
        t = jnp.arange(N) * dt
        dt_arr = jnp.full(N - 1, dt)
        return Trajectory(X=X0, U=U0, t=t, dt=dt_arr)

    traj = make_traj(x0)
    eager = altro_solve(prob, traj, al0, x0, options)

    # `problem` carries structural constraint bounds that PN's layout builder converts with eager
    # numpy (never traced, by design -- see `ALConstraints.build`'s docstring), so it must be
    # closed over rather than passed as a jitted positional argument; only the trajectory-shaped
    # values are meant to be traced.
    jitted = jax.jit(functools.partial(altro_solve, problem=prob, options=options))
    jit_result = jitted(trajectory=traj, al0=al0, x0=x0)
    np.testing.assert_allclose(np.asarray(jit_result.trajectory.X), np.asarray(eager.trajectory.X), atol=1e-6)
    assert int(jit_result.status) == int(eager.status)

    batch_x0 = jnp.stack([x0, x0])

    def solve_one(x0_: jax.Array) -> jax.Array:
        return altro_solve(prob, make_traj(x0_), al0, x0_, options).status

    statuses = jax.vmap(solve_one)(batch_x0)
    assert statuses.shape == (2,)
    assert int(statuses[0]) == int(eager.status)
    assert int(statuses[1]) == int(eager.status)


def test_altro_bypasses_pn_when_projected_newton_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """When projected_newton=False, pn_solve is never called on a constrained problem."""
    import trajopt.solvers.altro as altro_module

    def fail_pn_solve(*_args: object, **_kwargs: object) -> None:
        msg = "pn_solve must not be called when projected_newton=False"
        raise AssertionError(msg)

    monkeypatch.setattr(altro_module, "pn_solve", fail_pn_solve)

    prob, x0, dt, xf = _cartpole_problem(N=5)
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=xf, initial_trajectory=None)
    options = SolverOptions(iterations=2, iterations_outer=2, projected_newton=False)
    result = ALTRO(options=options).solve(prob, state)

    assert result.info["ran_pn"] is False
    assert result.info["pn_stats"] is None
