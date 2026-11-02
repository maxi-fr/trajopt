import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.dubins import DubinsCar
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.program import WarmStart
from trajopt.solvers.al import ALConstraints
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt


def _pendulum_problem(n_knots: int = 10, dt: float = 0.05) -> Problem:
    """Build a standard pendulum swing-up Problem."""
    model = Pendulum()
    obj = LQRObjective(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), N=n_knots)
    cl = ConstraintList(n=2, m=1, N=n_knots)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[-5.0], u_max=[5.0]), range(n_knots - 1))
    return Problem(model=model, obj=obj, constraints=cl, N=n_knots, dt=dt, integrator=RK4())


def _tracking_problem(n_knots: int = 10, dt: float = 0.05) -> tuple[Problem, Trajectory]:
    """Build a Dubins car tracking Problem with an explicit reference Trajectory."""
    model = DubinsCar()
    n, m = model.n, model.m
    xf = jnp.array([2.0, 0.0, 0.0])
    t_arr = jnp.linspace(0.0, (n_knots - 1) * dt, n_knots)
    dt_arr = jnp.full((n_knots - 1,), dt)
    x_ref = jnp.zeros((n_knots, n)).at[:, 0].set(jnp.linspace(0.0, float(xf[0]), n_knots))
    u_ref = jnp.ones((n_knots - 1, m)) * jnp.array([float(xf[0]) / ((n_knots - 1) * dt), 0.0])
    ref = Trajectory(X=x_ref, U=u_ref, t=t_arr, dt=dt_arr)

    cl = ConstraintList(n=n, m=m, N=n_knots)
    cl.add_constraint(GoalConstraint(n=n, xf=xf), n_knots - 1)
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=n_knots)
    prob = Problem(model=model, obj=obj, constraints=cl, N=n_knots, dt=dt, integrator=RK4())
    return prob, ref


def test_shift_removed_no_alias() -> None:
    """Assert MPC.shift is removed with no compatibility alias."""
    prob = _pendulum_problem()
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(2))
    assert not hasattr(mpc, "shift")
    with pytest.raises(AttributeError):
        getattr(mpc, "shift")()  # noqa: B009 -- verify removed attribute raises AttributeError


def test_advance_validates_steps_type_and_range() -> None:
    """Reject booleans, floats, negatives, and steps > H before mutating driver state."""
    prob = _pendulum_problem(n_knots=8)
    h = prob.N - 1
    mpc = MPC(prob, Ipopt(), x0=jnp.array([0.1, 0.2]), t0=0.5)

    z_before = mpc.Z
    t0_before = mpc.t0
    x0_before = mpc.x0

    for invalid_val in (True, False, np.True_, np.False_):
        with pytest.raises(TypeError, match="integer"):
            mpc.advance(invalid_val)  # ty: ignore[invalid-argument-type]

    for invalid_type in (1.5, "1", [1], None):
        with pytest.raises(TypeError, match="integer"):
            mpc.advance(invalid_type)  # ty: ignore[invalid-argument-type]

    for invalid_range in (-1, -5, h + 1, h + 10):
        with pytest.raises(ValueError, match="between 0 and"):
            mpc.advance(invalid_range)

    np.testing.assert_allclose(mpc.Z, z_before)
    np.testing.assert_allclose(mpc.t0, t0_before)
    np.testing.assert_allclose(mpc.x0, x0_before)

    # Valid boundaries: 0 and H must succeed
    mpc.advance(0)
    mpc.advance(h)


def test_advance_zero_is_noop() -> None:
    """Assert MPC.advance(0) leaves warm start, clock, reference, and pending push unchanged."""
    prob, ref = _tracking_problem(n_knots=8)
    goal = jnp.array([3.0, 0.5, 0.1])
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(3), t0=1.0, xf=goal, reference=ref)
    mpc.push_reference(jnp.array([9.0, 9.0, 9.0]), jnp.array([1.0, 2.0]))

    z_before = mpc.Z
    t0_before = mpc.t0
    x0_before = mpc.x0
    xf_before = mpc.xf
    assert mpc.bc.X_ref is not None
    x_ref_before = mpc.bc.X_ref
    u_ref_before = mpc.bc.U_ref
    pending_before = mpc._pending_ref  # noqa: SLF001 -- verifying pending reference state

    mpc.advance(0)

    np.testing.assert_allclose(mpc.Z, z_before)
    np.testing.assert_allclose(mpc.t0, t0_before)
    np.testing.assert_allclose(mpc.x0, x0_before)
    assert mpc.xf is not None
    assert xf_before is not None
    np.testing.assert_allclose(mpc.xf, xf_before)
    assert mpc.bc.X_ref is not None
    assert x_ref_before is not None
    assert mpc.bc.U_ref is not None
    assert u_ref_before is not None
    np.testing.assert_allclose(mpc.bc.X_ref, x_ref_before)
    np.testing.assert_allclose(mpc.bc.U_ref, u_ref_before)
    assert mpc._pending_ref == pending_before  # noqa: SLF001 -- pending push must not be consumed
    assert mpc.result is None


def test_advance_warm_start_states_and_controls() -> None:
    """Assert warm-start states and controls advance by steps Knot Points holding final entries."""
    prob = _pendulum_problem(n_knots=6)
    n, m, n_knots = prob.model.n, prob.model.m, prob.N
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(2))

    x_init = jnp.arange(n_knots * n, dtype=jnp.float64).reshape((n_knots, n))
    u_init = jnp.arange((n_knots - 1) * m, dtype=jnp.float64).reshape((n_knots - 1, m))
    mpc._ws = mpc.warm_start.with_primal(prob, X=x_init, U=u_init)  # noqa: SLF001 -- seeding known primal

    mpc.advance(2)

    expected_x = jnp.concatenate([x_init[2:], jnp.repeat(x_init[-1:], 2, axis=0)], axis=0)
    expected_u = jnp.concatenate([u_init[2:], jnp.repeat(u_init[-1:], 2, axis=0)], axis=0)
    np.testing.assert_allclose(mpc.states, expected_x)
    np.testing.assert_allclose(mpc.controls, expected_u)
    np.testing.assert_allclose(mpc.x0, expected_x[0])


def test_advance_nonuniform_grid_t0() -> None:
    """Assert advance on a nonuniform grid advances t0 by the prefix sum of Problem.dt."""
    model = Pendulum()
    dt_steps = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])
    n_knots = len(dt_steps) + 1
    obj = LQRObjective(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), N=n_knots)
    prob = Problem(model=model, obj=obj, N=n_knots, dt=dt_steps, integrator=RK4())

    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(2), t0=1.0)
    dt_orig = jnp.array(prob.dt)

    mpc.advance(3)

    np.testing.assert_allclose(mpc.t0, 1.0 + float(jnp.sum(dt_steps[:3])))
    np.testing.assert_allclose(prob.dt, dt_orig)


def test_advance_reference_pending_push_consumed_once() -> None:
    """Assert pending reference push is consumed on step 1 and held on subsequent steps."""
    prob, ref = _tracking_problem(n_knots=6)
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(3), reference=ref)

    pushed_x = jnp.array([10.0, 11.0, 12.0])
    pushed_u = jnp.array([1.5, 2.5])
    mpc.push_reference(pushed_x, pushed_u)

    # Advance 3 steps in one call
    mpc.advance(3)

    assert mpc.bc.X_ref is not None
    assert mpc.bc.U_ref is not None
    assert mpc._pending_ref is None  # noqa: SLF001 -- pending push consumed once

    # The pushed point should enter at the end from step 1, and be held for steps 2 and 3
    # Step 1: ref.X[1:], pushed_x
    # Step 2: ref.X[2:], pushed_x, pushed_x
    # Step 3: ref.X[3:], pushed_x, pushed_x, pushed_x
    np.testing.assert_allclose(mpc.bc.X_ref[-1], pushed_x)
    np.testing.assert_allclose(mpc.bc.X_ref[-2], pushed_x)
    np.testing.assert_allclose(mpc.bc.X_ref[-3], pushed_x)
    np.testing.assert_allclose(mpc.bc.X_ref[:-3], ref.X[3:])

    np.testing.assert_allclose(mpc.bc.U_ref[-1], pushed_u)
    np.testing.assert_allclose(mpc.bc.U_ref[-2], pushed_u)
    np.testing.assert_allclose(mpc.bc.U_ref[-3], pushed_u)


def test_measure_after_advance_updates_x0_and_warmstart() -> None:
    """Assert MPC.measure after MPC.advance replaces Boundary Conditions and warm start X[0]."""
    prob = _pendulum_problem(n_knots=6)
    mpc = MPC(prob, Ipopt(), x0=jnp.array([1.0, 1.0]), t0=0.0)

    # Initial warm start
    x_init = jnp.ones((6, 2)) * 2.0
    mpc._ws = mpc.warm_start.with_primal(prob, X=x_init)  # noqa: SLF001 -- seeding known primal

    mpc.advance(1)
    np.testing.assert_allclose(mpc.x0, jnp.array([2.0, 2.0]))
    np.testing.assert_allclose(mpc.states[0], jnp.array([2.0, 2.0]))

    # Now inject real measurement
    x_meas = jnp.array([0.5, -0.5])
    mpc.measure(x_meas, t=0.1)

    np.testing.assert_allclose(mpc.x0, x_meas)
    np.testing.assert_allclose(mpc.t0, 0.1)
    np.testing.assert_allclose(mpc.states[0], x_meas)


def test_advance_multi_step_matches_consecutive_single_steps() -> None:
    """Assert advance(6) matches 6 x advance(1) on uniform grid across multiple configurations."""
    prob, ref = _tracking_problem(n_knots=10, dt=0.05)
    goal = jnp.array([5.0, 1.0, -1.0])

    pushed_x = jnp.array([9.0, 8.0, 7.0])
    pushed_u = jnp.array([0.4, 0.6])

    # Case 1: with pending push, moving reference, independent goal
    mpc_multi = MPC(prob, Ipopt(), x0=jnp.zeros(3), t0=0.2, xf=goal, reference=ref)
    mpc_seq = MPC(prob, Ipopt(), x0=jnp.zeros(3), t0=0.2, xf=goal, reference=ref)

    mpc_multi.push_reference(pushed_x, pushed_u)
    mpc_seq.push_reference(pushed_x, pushed_u)

    mpc_multi.advance(6)
    for _ in range(6):
        mpc_seq.advance(1)

    np.testing.assert_allclose(mpc_multi.Z, mpc_seq.Z)
    np.testing.assert_allclose(mpc_multi.warm_start.lam, mpc_seq.warm_start.lam)
    np.testing.assert_allclose(mpc_multi.warm_start.mu, mpc_seq.warm_start.mu)
    np.testing.assert_allclose(mpc_multi.t0, mpc_seq.t0)
    assert mpc_multi.bc.X_ref is not None
    assert mpc_seq.bc.X_ref is not None
    assert mpc_multi.bc.U_ref is not None
    assert mpc_seq.bc.U_ref is not None
    np.testing.assert_allclose(mpc_multi.bc.X_ref, mpc_seq.bc.X_ref)
    np.testing.assert_allclose(mpc_multi.bc.U_ref, mpc_seq.bc.U_ref)
    assert mpc_multi.xf is not None
    assert mpc_seq.xf is not None
    np.testing.assert_allclose(mpc_multi.xf, mpc_seq.xf)

    # Case 2: AL multipliers and penalties carry across multiple steps
    n_al_knots = 10
    clist = ConstraintList(n=2, m=1, N=n_al_knots)
    clist.add_constraint(ControlBound(m=1, u_min=[-1.0], u_max=[1.0], n=2), range(n_al_knots - 1))
    al_prob = Problem(
        model=Pendulum(),
        obj=LQRObjective(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), N=n_al_knots),
        constraints=clist,
        N=n_al_knots,
        dt=0.05,
        integrator=RK4(),
    )
    al = ALConstraints.build(al_prob.constraints, penalty_initial=1.0)
    u_start = al.p_cons_max + 2 * al_prob.model.n
    lam = al.lam.at[n_al_knots - 2, u_start:].set(7.5)
    ws_init = WarmStart.cold(al_prob, x0=jnp.zeros(2))
    ws_init = WarmStart(
        Z=ws_init.Z,
        lam=ws_init.lam,
        mu=ws_init.mu,
        al=ALConstraints(lam=lam, mu=al.mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max),
    )

    mpc_al_multi = MPC(al_prob, Ipopt(), x0=jnp.zeros(2))
    mpc_al_seq = MPC(al_prob, Ipopt(), x0=jnp.zeros(2))
    mpc_al_multi._ws = ws_init  # noqa: SLF001 -- injecting test AL state
    mpc_al_seq._ws = ws_init  # noqa: SLF001 -- injecting test AL state

    mpc_al_multi.advance(6)
    for _ in range(6):
        mpc_al_seq.advance(1)

    assert mpc_al_multi.warm_start.al is not None
    assert mpc_al_seq.warm_start.al is not None
    np.testing.assert_allclose(mpc_al_multi.warm_start.al.lam, mpc_al_seq.warm_start.al.lam)
    np.testing.assert_allclose(mpc_al_multi.warm_start.al.mu, mpc_al_seq.warm_start.al.mu)
