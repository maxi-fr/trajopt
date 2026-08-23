import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, SecondOrderCone
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import ContinuousDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import Euler
from trajopt.models.cartpole import Cartpole
from trajopt.problem import Problem
from trajopt.transcription.osqp import OSQPResult, solve_osqp


class DoubleIntegrator(ContinuousDynamics):
    """Double integrator: pos, vel, u is force."""

    def __init__(self) -> None:
        super().__init__(n=2, m=1, ne=2)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate continuous dynamics."""
        del t
        return jnp.array([x[1], u[0]])


def test_osqp_basic_solve() -> None:
    """Test OSQP solve on unconstrained double integrator."""
    model = DiscretizedDynamics(DoubleIntegrator(), Euler())
    n, m, N = 2, 1, 11
    dt = 0.1

    Q = jnp.eye(n) * 1.0
    R = jnp.eye(m) * 0.1
    Qf = jnp.eye(n) * 10.0
    cost = QuadraticCost(Q=Q, R=R, r=jnp.zeros(m), c=0.0)
    term_cost = QuadraticCost(Q=Qf, R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=term_cost, N=N)

    problem = Problem(model=model, obj=obj, constraints=ConstraintList(n, m, N), N=N)
    x0 = jnp.array([2.0, 0.0])

    res = solve_osqp(problem, x0, dt=dt)

    assert isinstance(res, OSQPResult)
    assert res.success is True
    assert res.status in {1, 2}
    assert res.iterations > 0
    assert res.cost > 0.0
    assert res.constraint_violation < 1e-4
    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (N - 1, m)
    assert np.allclose(res.trajectory.X[0], [2.0, 0.0], atol=1e-5)


def test_osqp_with_bounds_and_linear_constraints() -> None:
    """Test OSQP solve with state bounds, control bounds, and linear inequality constraints."""
    model = DiscretizedDynamics(DoubleIntegrator(), Euler())
    n, m, N = 2, 1, 11
    dt = 0.1

    Q = jnp.eye(n)
    R = jnp.eye(m) * 0.1
    obj = Objective(
        stage_cost=QuadraticCost(Q=Q, R=R, r=jnp.zeros(m), c=0.0),
        terminal_cost=QuadraticCost(Q=10.0 * Q, R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0),
        N=N,
    )

    clist = ConstraintList(n, m, N)
    clist.add_constraint(ControlBound(m=m, u_min=jnp.array([-5.0]), u_max=jnp.array([5.0]), n=n), range(N - 1))
    clist.add_constraint(StateBound(n=n, x_min=jnp.array([-5.0, -5.0]), x_max=jnp.array([5.0, 5.0]), m=m), range(N))
    # Linear constraint: x[0] <= 1.5 for k >= 5
    A_lin = jnp.array([[1.0, 0.0]])
    b_lin = jnp.array([1.5])
    clist.add_constraint(
        LinearConstraint(n=n, m=m, A=A_lin, b=b_lin, sense=NegativeOrthant(), inds=[0, 1]),
        range(5, N - 1),
    )

    problem = Problem(model=model, obj=obj, constraints=clist, N=N)
    x0 = jnp.array([2.0, 0.0])

    res = solve_osqp(problem, x0, dt=dt, options={"eps_abs": 1e-6, "eps_rel": 1e-6})

    assert res.success is True
    assert res.constraint_violation < 1e-4
    assert np.all(res.trajectory.U >= -5.0 - 1e-4)
    assert np.all(res.trajectory.U <= 5.0 + 1e-4)
    assert np.all(res.trajectory.X[:, 1] >= -5.0 - 1e-4)
    assert np.all(res.trajectory.X[:, 1] <= 5.0 + 1e-4)
    assert np.all(res.trajectory.X[5:-1, 0] <= 1.5 + 1e-4)


def test_osqp_rejects_second_order_cone() -> None:
    """Test that OSQP rejects problems containing SecondOrderCone constraints."""
    model = DiscretizedDynamics(Cartpole(), Euler())
    n, m, N = 4, 1, 6

    cost = QuadraticCost(Q=jnp.eye(n), R=jnp.eye(m), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=cost, N=N)

    clist = ConstraintList(n, m, N)
    clist.add_constraint(NormConstraint(n=n, m=m, val=1.0, sense=SecondOrderCone(), inds=[0, 1]), range(N - 1))

    problem = Problem(model=model, obj=obj, constraints=clist, N=N)
    x0 = jnp.zeros(n)

    with pytest.raises(TypeError, match="OSQP does not support SecondOrderCone constraints"):
        solve_osqp(problem, x0)
