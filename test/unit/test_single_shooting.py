import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import (
    SingleShooting,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    single_shooting_dimensions,
)


def test_single_shooting_dimensions() -> None:
    """Solver sees controls only, with exactly the registered user constraints."""
    model = Pendulum()
    n, m, N = model.n, model.m, 21
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(
        Q=jnp.diag(jnp.array([10.0, 1.0])),
        R=jnp.diag(jnp.array([0.1])),
        Qf=jnp.diag(jnp.array([100.0, 10.0])),
        xf=xf,
        N=N,
    )
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    constraints.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, integrator=RK4())

    n_u, p_user = single_shooting_dimensions(prob)
    assert n_u == (N - 1) * m
    assert p_user == n

    x0 = jnp.array([0.0, 0.0])
    u = jnp.zeros(n_u)
    assert eval_g(prob, u, x0, t0=0.0, dt=dt, xf=xf).shape == (p_user,)
    assert eval_grad_f(prob, u, x0, t0=0.0, dt=dt, xf=xf).shape == (n_u,)
    assert eval_jac_g(prob, u, x0, t0=0.0, dt=dt, xf=xf).shape == (p_user * n_u,)


def test_single_shooting_rejects_state_bound() -> None:
    """State bounds have no primal variable in single shooting and must be refused."""
    model = Pendulum()
    n, m, N = model.n, model.m, 10
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), xf=xf, N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(StateBound(n=n, x_min=[-1.0, -1.0], x_max=[1.0, 1.0]), range(N))
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, integrator=RK4())
    state = MPCState.initial(prob, x0=jnp.zeros(n), xf=xf, dt=0.05)

    with pytest.raises(ValueError, match="state bound"):
        SingleShooting(Ipopt()).solve(prob, state)


def test_single_shooting_rejects_stage_constraint() -> None:
    """Non-goal stage constraints are refused until they are supported."""
    model = Pendulum()
    n, m, N = model.n, model.m, 10
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), xf=xf, N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([0.0])),
        range(N - 1),
    )
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, integrator=RK4())
    state = MPCState.initial(prob, x0=jnp.zeros(n), xf=xf, dt=0.05)

    with pytest.raises(ValueError, match="LinearConstraint"):
        SingleShooting(Ipopt()).solve(prob, state)


def test_single_shooting_matches_multiple_shooting() -> None:
    """Both transcriptions solve the same NLP, so their optimal controls agree."""
    pytest.importorskip("cyipopt")

    model = Pendulum()
    n, m, N = model.n, model.m, 21
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(
        Q=jnp.diag(jnp.array([10.0, 1.0])),
        R=jnp.diag(jnp.array([0.1])),
        Qf=jnp.diag(jnp.array([100.0, 10.0])),
        xf=xf,
        N=N,
    )
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    constraints.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, integrator=RK4())
    x0 = jnp.array([0.0, 0.0])
    state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf, dt=dt)

    options = {"max_iter": 200, "tol": 1e-8, "print_level": 0}
    ms = prob.solve(state, solver=Ipopt(options=options))
    ss = prob.solve(state, solver=SingleShooting(Ipopt(options=options), hessian="dense"))

    np.testing.assert_allclose(ss.controls, ms.controls, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ss.states, ms.states, atol=1e-5, rtol=1e-5)

    # Single shooting satisfies dynamics by construction, not merely to solver tolerance.
    rolled = prob.model.rollout(ss.to_trajectory())
    np.testing.assert_allclose(rolled.X, ss.states, atol=1e-12)


def test_closed_loop_cartpole_mpc_single_shooting() -> None:
    """Closed-loop cartpole MPC stabilizes through the single-shooting transcription."""
    pytest.importorskip("cyipopt")

    N = 20
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m

    x_curr = jnp.array([0.0, np.pi - 0.25, 0.1, -0.2])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])

    Q = jnp.diag(jnp.array([5.0, 20.0, 1.0, 2.0]))
    R = jnp.diag(jnp.array([0.05]))
    Qf = jnp.diag(jnp.array([50.0, 200.0, 10.0, 20.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    state = MPCState.initial(prob, x0=x_curr, t0=0.0, xf=xf, dt=dt)
    dmodel = RK4(model)
    solver = SingleShooting(Ipopt(options={"max_iter": 200, "tol": 1e-4, "print_level": 0}))

    sim_steps = 20
    t_curr = 0.0
    for _ in range(sim_steps):
        state = state.with_measurement(x_curr, t_curr)
        state = prob.solve(state, solver=solver)
        u_cmd = state.controls[0]
        x_curr = dmodel.discrete_dynamics(x_curr, u_cmd, t_curr, dt)
        t_curr += dt
        state = state.shift(dt)

    np.testing.assert_allclose(x_curr[1], np.pi, atol=0.1)
    np.testing.assert_allclose(x_curr[0], 0.0, atol=0.5)
    np.testing.assert_allclose(x_curr[2:], 0.0, atol=0.5)
