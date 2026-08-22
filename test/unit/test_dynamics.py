import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.dynamics import (
    ContinuousDynamics,
    DiscreteDynamics,
    DiscretizedDynamics,
    EuclideanModel,
)
from trajopt.models import Cartpole


class LinearContinuousModel(ContinuousDynamics):
    """Linear continuous-time model xdot = A x + B u."""

    A: jax.Array
    B: jax.Array

    def __init__(self, A: jax.Array, B: jax.Array):
        super().__init__(n=A.shape[0], m=B.shape[1])
        self.A = A
        self.B = B

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del t
        return self.A @ x + self.B @ u


class LinearDiscreteModel(DiscreteDynamics):
    """Linear discrete-time model x_{k+1} = Ad x + Bd u."""

    Ad: jax.Array
    Bd: jax.Array

    def __init__(self, Ad: jax.Array, Bd: jax.Array):
        super().__init__(n=Ad.shape[0], m=Bd.shape[1])
        self.Ad = Ad
        self.Bd = Bd

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        del t, dt
        return self.Ad @ x + self.Bd @ u


def test_continuous_dynamics_euclidean_defaults() -> None:
    A = jnp.array([[0.0, 1.0], [-2.0, -3.0]])
    B = jnp.array([[0.0], [1.0]])
    model = LinearContinuousModel(A, B)

    # Static properties
    assert model.n == 2
    assert model.m == 1
    assert model.ne == 2  # ne defaults to n for Euclidean

    # Evaluation
    x = jnp.array([1.0, 2.0])
    u = jnp.array([0.5])
    xdot = model.dynamics(x, u)
    expected_xdot = A @ x + B @ u
    np.testing.assert_allclose(xdot, expected_xdot)

    # Jacobian via AD: shape (n, n + m)
    J = model.jacobian(x, u)
    assert J.shape == (2, 3)
    np.testing.assert_allclose(J[:, :2], A)
    np.testing.assert_allclose(J[:, 2:], B)

    # State difference defaults to subtraction
    x0 = jnp.array([0.5, 1.0])
    dx = model.state_diff(x, x0)
    assert dx.shape == (2,)
    np.testing.assert_allclose(dx, x - x0)

    # Error-state Jacobian defaults to identity
    G = model.errstate_jacobian(x)
    assert G.shape == (2, 2)
    np.testing.assert_allclose(G, jnp.eye(2))


def test_euclidean_model_declaration_without_manifold() -> None:
    """Confirms a user-defined Euclidean model can be declared without referencing any manifold concept."""

    class SimpleHarmonicOscillator(EuclideanModel):
        omega: float = 1.0

        def __init__(self, omega: float = 1.0):
            super().__init__(n=2, m=1)
            self.omega = omega

        def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
            del t
            return jnp.array([x[1], -(self.omega**2) * x[0] + u[0]])

    sho = SimpleHarmonicOscillator(omega=2.0)
    assert sho.n == 2
    assert sho.m == 1
    assert sho.ne == 2

    x = jnp.array([1.0, 0.5])
    u = jnp.array([0.1])
    xdot = sho.dynamics(x, u)
    np.testing.assert_allclose(xdot, jnp.array([0.5, -4.0 * 1.0 + 0.1]))

    # Error-state dimension and map work automatically
    np.testing.assert_allclose(sho.state_diff(x, jnp.zeros(2)), x)
    np.testing.assert_allclose(sho.errstate_jacobian(x), jnp.eye(2))


def test_discrete_dynamics_base() -> None:
    Ad = jnp.array([[1.0, 0.1], [0.0, 0.9]])
    Bd = jnp.array([[0.0], [0.1]])
    model = LinearDiscreteModel(Ad, Bd)

    assert model.n == 2
    assert model.m == 1
    assert model.ne == 2

    x = jnp.array([1.0, 2.0])
    u = jnp.array([0.5])
    xnext = model.discrete_dynamics(x, u, 0.0, 0.1)
    np.testing.assert_allclose(xnext, Ad @ x + Bd @ u)

    J = model.jacobian(x, u, 0.0, 0.1)
    assert J.shape == (2, 3)
    np.testing.assert_allclose(J[:, :2], Ad)
    np.testing.assert_allclose(J[:, 2:], Bd)


def test_discretized_dynamics_with_euler_integrator() -> None:
    A = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    cont_model = LinearContinuousModel(A, B)

    # Simple Euler step function: x_{k+1} = x + dt * f(x, u, t)
    def euler_step(
        c_model: ContinuousDynamics, x: jax.Array, u: jax.Array, t: float | jax.Array, dt: float | jax.Array
    ) -> jax.Array:
        return x + dt * c_model.dynamics(x, u, t)

    disc_model = DiscretizedDynamics(cont_model, euler_step)
    assert disc_model.n == 2
    assert disc_model.m == 1
    assert disc_model.ne == 2

    x = jnp.array([1.0, 0.0])
    u = jnp.array([2.0])
    dt = 0.05
    xnext = disc_model.discrete_dynamics(x, u, 0.0, dt)
    expected_xnext = x + dt * (A @ x + B @ u)
    np.testing.assert_allclose(xnext, expected_xnext)

    # Jacobian of discretized dynamics
    J = disc_model.jacobian(x, u, 0.0, dt)
    expected_J = jnp.hstack([jnp.eye(2) + dt * A, dt * B])
    np.testing.assert_allclose(J, expected_J)


def test_cartpole_pytree_structure() -> None:
    cartpole = Cartpole(mc=1.0, mp=0.2, l=0.5, g=9.81)

    assert cartpole.n == 4
    assert cartpole.m == 1
    assert cartpole.ne == 4

    # Parameters are leaves
    leaves = jax.tree_util.tree_leaves(cartpole)
    # The leaves should be the 4 floating/array parameters: mc, mp, l, g
    assert len(leaves) == 4
    leaf_vals = [float(leaf) for leaf in leaves]
    assert 1.0 in leaf_vals
    assert 0.2 in leaf_vals
    assert 0.5 in leaf_vals
    assert 9.81 in leaf_vals

    # JIT compilation works seamlessly
    @eqx.filter_jit
    def eval_dyn(model: Cartpole, x: jax.Array, u: jax.Array) -> jax.Array:
        return model.dynamics(x, u)

    x = jnp.array([0.0, 0.0, 0.0, 0.0])
    u = jnp.array([0.0])
    res = eval_dyn(cartpole, x, u)
    assert res.shape == (4,)
    np.testing.assert_allclose(res, jnp.zeros(4))
