import jax.numpy as jnp
import numpy as np

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models import Cartpole
from trajopt.problem import Problem
from trajopt.solvers.al import (
    ALCarry,
    ALConstraints,
    ALStats,
    _al_transition,
    evaluate_al_constraints,
    evaluate_al_residuals,
    penalty_update,
)
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory


def _cartpole_problem(
    N: int = 11,
    dt: float = 0.05,
    u_bnd: float = 20.0,
) -> tuple[Problem, Trajectory]:
    model = Cartpole()
    n, m = model.n, model.m
    Q = jnp.array([1.0, 10.0, 1.0, 1.0])
    R = jnp.array([0.1])
    Qf = jnp.array([100.0, 100.0, 10.0, 10.0])
    xf = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(n=n, m=m, u_min=[-u_bnd], u_max=[u_bnd]), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())

    X = jnp.zeros((N, n))
    U = jnp.zeros((N - 1, m))
    t = jnp.linspace(0.0, dt * (N - 1), N)
    dt_arr = jnp.full((N - 1,), dt)
    traj = Trajectory(X=X, U=U, t=t, dt=dt_arr)
    return prob, traj


def _make_initial_carry(prob: Problem, traj: Trajectory, options: SolverOptions) -> ALCarry:
    al = ALConstraints.build(prob.constraints, penalty_initial=options.penalty_initial)
    stats = ALStats.create(options)
    return ALCarry(
        i=jnp.int32(0),
        trajectory=traj,
        al=al,
        stats=stats,
        done=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar
        status=jnp.int32(TerminationStatus.UNSOLVED),
    )


def test_al_transition_soft_stalls_continue_outer_loop() -> None:
    """Soft stalls (NO_PROGRESS, MAX_ITERATIONS) update duals/penalties and continue the outer loop."""
    prob, traj = _cartpole_problem()
    options = SolverOptions(iterations=100, iterations_outer=10)
    carry = _make_initial_carry(prob, traj, options)

    for stalled_status in (TerminationStatus.NO_PROGRESS, TerminationStatus.MAX_ITERATIONS):
        next_carry = _al_transition(
            carry=carry,
            problem=prob,
            options=options,
            new_traj=traj,
            inner_iterations=jnp.int32(100),
            inner_status=jnp.int32(stalled_status),
        )

        assert not bool(next_carry.done), f"Soft stall {stalled_status} prematurely marked done"
        assert int(next_carry.stats.iterations) == 1
        assert bool(jnp.any(next_carry.al.mu > carry.al.mu))


def test_al_transition_fatal_statuses_abort_immediately() -> None:
    """Fatal numerical failures abort immediately without updating duals or recording stats."""
    prob, traj = _cartpole_problem()
    options = SolverOptions(iterations=100, iterations_outer=10)
    carry = _make_initial_carry(prob, traj, options)

    for fatal_status in (
        TerminationStatus.COST_INCREASE,
        TerminationStatus.STATE_LIMIT,
        TerminationStatus.CONTROL_LIMIT,
    ):
        next_carry = _al_transition(
            carry=carry,
            problem=prob,
            options=options,
            new_traj=traj,
            inner_iterations=jnp.int32(5),
            inner_status=jnp.int32(fatal_status),
        )

        assert bool(next_carry.done), f"Fatal status {fatal_status} failed to abort"
        assert int(next_carry.status) == int(fatal_status)
        np.testing.assert_allclose(np.asarray(next_carry.al.mu), np.asarray(carry.al.mu))
        np.testing.assert_allclose(np.asarray(next_carry.al.lam), np.asarray(carry.al.lam))
        assert int(next_carry.stats.iterations) == 0


def test_al_transition_penalty_overflow_stops_outer_loop() -> None:
    """When quadratic penalties would overflow max_cost_value, AL halts with MAX_ITERATIONS_OUTER."""
    prob, traj = _cartpole_problem()
    options = SolverOptions(
        iterations=100,
        iterations_outer=10,
        max_cost_value=1.0,
        penalty_initial=100.0,
        penalty_scaling=10.0,
    )
    carry = _make_initial_carry(prob, traj, options)

    next_carry = _al_transition(
        carry=carry,
        problem=prob,
        options=options,
        new_traj=traj,
        inner_iterations=jnp.int32(5),
        inner_status=jnp.int32(TerminationStatus.SOLVE_SUCCEEDED),
    )

    assert bool(next_carry.done)
    assert int(next_carry.status) == int(TerminationStatus.MAX_ITERATIONS_OUTER)


def test_evaluate_al_residuals_bitwise_matches_evaluate_al_constraints() -> None:
    """evaluate_al_residuals returns values bitwise identical to evaluate_al_constraints[0]."""
    prob, traj = _cartpole_problem()
    al = ALConstraints.build(prob.constraints, penalty_initial=1.0)

    residuals = evaluate_al_residuals(al, prob.constraints, traj)
    c_full, _, _ = evaluate_al_constraints(al, prob.constraints, prob.model, traj)

    np.testing.assert_array_equal(np.asarray(residuals), np.asarray(c_full))


def test_penalty_clamping_invariant() -> None:
    """Penalty update clamps mu at penalty_max and leaves masked rows unchanged."""
    prob, _ = _cartpole_problem()
    options = SolverOptions(penalty_initial=1e7, penalty_scaling=100.0, penalty_max=1e8)
    al = ALConstraints.build(prob.constraints, penalty_initial=options.penalty_initial)

    updated = penalty_update(al, options)

    assert bool(jnp.all(updated.mu <= options.penalty_max))
    unmasked = al.row_mask
    np.testing.assert_allclose(np.asarray(updated.mu[unmasked]), options.penalty_max)
