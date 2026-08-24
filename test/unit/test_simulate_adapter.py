import jax.numpy as jnp
import numpy as np
import pytest
from simulate.dynamics import StateLog
from simulate.estimator import IdentityEstimator
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor, LinearMeasurement
from simulate.simulation import Simulation

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.base import DiscretizedDynamics
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.simulate import (
    TrajOptDynamics,
    TrajOptMPC,
    TrajOptMPCLog,
)
from trajopt.transcription.ipopt import Ipopt


def _make_pendulum_problem(N: int = 15, dt: float = 0.05) -> tuple[Problem, MPCState]:
    """Helper to build a small pendulum optimal control problem."""
    model = Pendulum()
    xf = jnp.array([np.pi, 0.0])
    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.diag(jnp.array([0.1]))
    Qf = jnp.diag(jnp.array([50.0, 5.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    constraints = ConstraintList(n=2, m=1, N=N)
    constraints.add_constraint(ControlBound(n=2, m=1, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    constraints.add_constraint(GoalConstraint(n=2, xf=xf), N - 1)

    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, integrator=RK4())
    state = MPCState.initial(prob, x0=jnp.array([0.1, 0.0]), t0=0.0, xf=xf, dt=dt)
    return prob, state


def test_mpc_update_step_and_warmstart_shift() -> None:
    """Verify one update step runs the solve, returns control, and shifts the internal state."""
    prob, initial_state = _make_pendulum_problem(N=10, dt=0.05)
    controller = TrajOptMPC(
        dt=0.05,
        problem=prob,
        initial_state=initial_state,
        solver=Ipopt(options={"print_level": 0, "max_iter": 50}),
    )

    ref = np.array([np.pi, 0.0])
    x_hat = np.array([0.2, 0.0])

    u, log = controller.update(t=0.0, ref=ref, x_hat=x_hat)

    assert isinstance(u, np.ndarray)
    assert u.shape == (1,)
    assert isinstance(log, TrajOptMPCLog)
    assert log.solve_success is True
    assert log.fallback_used is False
    assert not np.isnan(log.cost)
    assert log.cost > 0.0
    assert log.solve_time_s > 0.0
    np.testing.assert_allclose(log.u, u)

    assert float(controller.state.t0) == pytest.approx(0.05)


def test_mpc_from_config_direct_problem() -> None:
    """Verify instantiating MPC from config dict with an existing Problem."""
    prob, initial_state = _make_pendulum_problem()
    cfg = {
        "dt": 0.05,
        "problem": prob,
        "initial_state": initial_state,
        "solver": Ipopt(options={"print_level": 0}),
    }

    controller = TrajOptMPC.from_config(cfg)
    assert isinstance(controller, TrajOptMPC)
    assert controller.dt == 0.05
    assert isinstance(controller.solver, Ipopt)


def test_mpc_from_config_class_path() -> None:
    """Verify instantiating MPC from config dict with problem class_path dictionary."""
    cfg = {
        "dt": 0.05,
        "problem": {
            "class_path": "trajopt.benchmarks.cartpole_swingup_benchmark",
            "N": 12,
            "dt": 0.05,
        },
    }

    controller = TrajOptMPC.from_config(cfg)
    assert isinstance(controller, TrajOptMPC)
    assert controller.dt == 0.05
    assert controller.problem.N == 12


def test_mpc_fallback_on_solver_failure() -> None:
    """Verify solver failure triggers warm-start fallback without raising."""
    prob, initial_state = _make_pendulum_problem()
    controller = TrajOptMPC(
        dt=0.05,
        problem=prob,
        initial_state=initial_state,
        # An option cyipopt rejects at add_option time is what breaks the solve; the fallback
        # path doesn't care why the solver failed, only that it did.
        solver=Ipopt(options={"max_iter": "not_a_number"}),
    )

    u, log = controller.update(t=0.0, ref=np.array([np.pi, 0.0]), x_hat=np.array([0.1, 0.0]))
    assert isinstance(u, np.ndarray)
    assert log.solve_success is False
    assert log.fallback_used is True
    assert np.isnan(log.cost)


def test_trajopt_dynamics_continuous() -> None:
    """Verify continuous trajopt model wrapped into simulate Dynamics."""
    model = Pendulum()
    dt = 0.05
    x0 = np.array([0.2, 0.0])
    plant = TrajOptDynamics(dt=dt, model=model, x0=x0)

    assert plant.dt == dt
    assert plant.n_inputs == 1
    np.testing.assert_allclose(plant.x, x0)
    assert isinstance(plant.model, DiscretizedDynamics)

    u = np.array([1.0])
    x_next, log = plant.update(t=0.0, u=u)

    assert isinstance(x_next, np.ndarray)
    assert isinstance(log, StateLog)
    np.testing.assert_allclose(log.x, x0)
    assert not np.allclose(x_next, x0)


def test_trajopt_dynamics_discrete() -> None:
    """Verify discrete trajopt model wrapped into simulate Dynamics without integrator."""
    continuous_model = Pendulum()
    discrete_model = DiscretizedDynamics(continuous_dynamics=continuous_model, integrator=RK4())
    dt = 0.05
    x0 = np.array([0.2, 0.0])
    plant = TrajOptDynamics(dt=dt, model=discrete_model, x0=x0)

    assert plant.dt == dt
    assert plant.n_inputs == 1
    assert plant.model is discrete_model

    u = np.array([1.0])
    x_next, log = plant.update(t=0.0, u=u)

    assert isinstance(x_next, np.ndarray)
    assert isinstance(log, StateLog)
    np.testing.assert_allclose(log.x, x0)


def test_trajopt_dynamics_from_config_class_path() -> None:
    """Verify instantiating TrajOptDynamics from config dict with {class_path, ...}."""
    plant_dict = TrajOptDynamics.from_config(
        {
            "dt": 0.05,
            "model": {
                "class_path": "trajopt.models.cartpole.Cartpole",
                "mc": 1.5,
                "mp": 0.3,
                "l": 0.6,
            },
            "x0": [0.0, 0.0, 0.0, 0.0],
        }
    )
    assert isinstance(plant_dict.model, DiscretizedDynamics)
    assert isinstance(plant_dict.model.continuous_dynamics, Cartpole)
    assert plant_dict.model.continuous_dynamics.mc == 1.5
    assert plant_dict.model.n == 4
    assert plant_dict.model.m == 1


def test_unified_closed_loop_plant_and_mpc() -> None:
    """Verify end-to-end simulation sharing the same trajopt model for plant and MPC."""
    dt = 0.05
    model = Pendulum()

    plant = TrajOptDynamics(dt=dt, model=model, x0=np.array([0.1, 0.0]))

    prob, initial_state = _make_pendulum_problem(N=10, dt=dt)
    controller = TrajOptMPC(
        dt=dt,
        problem=prob,
        initial_state=initial_state,
        solver=Ipopt(options={"print_level": 0, "max_iter": 30}),
    )

    reference = StepReference(dt=dt, step_value=np.array([np.pi, 0.0]))
    sensor = GaussianSensor(
        dt=dt,
        std_dev=0.001,
        measurement=LinearMeasurement(C=np.eye(2), D=np.zeros((2, 1))),
    )
    estimator = IdentityEstimator(dt=dt)

    sim = Simulation(
        t_end=0.2,
        dynamics=plant,
        reference=reference,
        sensors=[sensor],
        estimator=estimator,
        controller=controller,
    )

    sim.run()

    assert sim.logger is not None
    logged_x = sim.logger.signal("dynamics", "x")
    logged_u = sim.logger.signal("controller", "u")
    logged_cost = sim.logger.signal("controller", "cost")

    assert len(logged_x) == 5
    assert len(logged_u) == 5
    assert not np.isnan(logged_cost[0])
