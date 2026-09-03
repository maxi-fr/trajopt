import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import ContinuousDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import Euler
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.transcription.clarabel import Clarabel, ClarabelResult


class DoubleIntegrator(ContinuousDynamics):
    """Double integrator: pos, vel, u is force."""

    def __init__(self) -> None:
        super().__init__(n=2, m=1, ne=2)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate continuous dynamics."""
        del t
        return jnp.array([x[1], u[0]])


def test_clarabel_basic_solve() -> None:
    """Test Clarabel solve on unconstrained double integrator."""
    model = DiscretizedDynamics(DoubleIntegrator(), Euler())
    n, m, N = 2, 1, 11
    dt = 0.1

    Q = jnp.eye(n) * 1.0
    R = jnp.eye(m) * 0.1
    Qf = jnp.eye(n) * 10.0
    cost = QuadraticCost(Q=Q, R=R, r=jnp.zeros(m), c=0.0)
    term_cost = QuadraticCost(Q=Qf, R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=term_cost, N=N)

    problem = Problem(model=model, obj=obj, constraints=ConstraintList(n, m, N), N=N, dt=dt)
    x0 = jnp.array([2.0, 0.0])

    res = MPC(problem, Clarabel(), x0=x0).solve()

    assert isinstance(res, ClarabelResult)
    assert res.success is True
    assert "Solved" in res.status
    assert res.iterations >= 0
    assert res.cost > 0.0
    assert res.constraint_violation < 1e-4
    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (N - 1, m)
    assert np.allclose(res.trajectory.X[0], [2.0, 0.0], atol=1e-5)


def test_clarabel_with_bounds_and_orthants() -> None:
    """Test Clarabel with state bounds, control bounds, ZeroCone, and Orthant constraints."""
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
    # Equality constraint at stage: x[1] == 0.0 (ZeroCone)
    clist.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[0.0, 1.0]]), b=jnp.array([0.0]), sense=ZeroCone(), inds=[0, 1]),
        N - 2,
    )
    # Inequality constraint: x[0] <= 1.5 (NegativeOrthant) for k >= 5
    clist.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0]]), b=jnp.array([1.5]), sense=NegativeOrthant(), inds=[0, 1]),
        range(5, N - 1),
    )
    # Inequality constraint: x[0] >= -2.0 (PositiveOrthant) for k >= 5
    clist.add_constraint(
        LinearConstraint(
            n=n, m=m, A=jnp.array([[1.0, 0.0]]), b=jnp.array([-2.0]), sense=PositiveOrthant(), inds=[0, 1]
        ),
        range(5, N - 1),
    )

    problem = Problem(model=model, obj=obj, constraints=clist, N=N, dt=dt)
    x0 = jnp.array([2.0, 0.0])

    res = MPC(problem, Clarabel(), x0=x0).solve()

    assert res.success is True
    assert res.constraint_violation < 1e-4
    assert np.all(res.trajectory.U >= -5.0 - 1e-4)
    assert np.all(res.trajectory.U <= 5.0 + 1e-4)
    assert np.all(res.trajectory.X[5:-1, 0] <= 1.5 + 1e-4)
    assert np.all(res.trajectory.X[5:-1, 0] >= -2.0 - 1e-4)
    assert np.isclose(res.trajectory.X[N - 2, 1], 0.0, atol=1e-4)


def test_clarabel_with_second_order_cone() -> None:
    """Test Clarabel solve with SecondOrderCone constraint."""
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
    # Norm constraint on control u <= 0.4
    clist.add_constraint(NormConstraint(n=n, m=m, val=0.4, sense=SecondOrderCone(), inds="control"), range(N - 1))

    problem = Problem(model=model, obj=obj, constraints=clist, N=N, dt=dt)
    x0 = jnp.array([2.0, 0.0])

    res = MPC(problem, Clarabel(), x0=x0).solve()

    assert res.success is True
    assert res.constraint_violation < 1e-4
    for k in range(N - 1):
        u_norm = float(np.linalg.norm(res.trajectory.U[k]))
        assert u_norm <= 0.4 + 1e-4
