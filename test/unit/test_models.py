"""Unit tests for benchmark models: Cartpole, Pendulum, and DubinsCar."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt import models
from trajopt.models import Cartpole, DubinsCar, Pendulum


def test_cartpole_instantiation_and_properties() -> None:
    model = Cartpole()
    assert model.n == 4
    assert model.m == 1
    assert model.ne == 4

    assert float(model.mc) == 1.0
    assert float(model.mp) == 0.2
    assert float(model.l) == 0.5
    assert float(model.g) == 9.81

    # Custom parameters
    custom = Cartpole(mc=2.0, mp=0.5, l=1.0, g=9.8)
    assert float(custom.mc) == 2.0
    assert float(custom.mp) == 0.5
    assert float(custom.l) == 1.0
    assert float(custom.g) == 9.8


def test_pendulum_instantiation_and_properties() -> None:
    model = Pendulum()
    assert model.n == 2
    assert model.m == 1
    assert model.ne == 2

    # Default parameters matching RobotZoo
    assert float(model.mass) == 1.0
    assert float(model.len) == 0.5
    assert float(model.b) == 0.1
    assert float(model.lc) == 0.5
    assert float(model.I) == 0.25
    assert float(model.g) == 9.81

    # Custom parameters
    custom = Pendulum(mass=2.0, len=1.2, b=0.05, lc=0.6, I=0.5, g=9.80665)
    assert float(custom.mass) == 2.0
    assert float(custom.len) == 1.2
    assert float(custom.b) == 0.05
    assert float(custom.lc) == 0.6
    assert float(custom.I) == 0.5
    assert float(custom.g) == 9.80665


def test_dubins_car_instantiation_and_properties() -> None:
    model = DubinsCar()
    assert model.n == 3
    assert model.m == 2
    assert model.ne == 3

    # Default parameters matching RobotZoo
    assert float(model.radius) == 0.175

    # Custom parameters
    custom = DubinsCar(radius=0.25)
    assert float(custom.radius) == 0.25


def test_pendulum_dynamics_and_jacobians() -> None:
    model = Pendulum()
    x = jnp.array([0.0, 0.0])
    u = jnp.array([0.0])

    # Equilibrium at downward vertical: theta=0, omega=0, tau=0 => xdot=0
    xdot = model.dynamics(x, u)
    assert xdot.shape == (2,)
    np.testing.assert_allclose(xdot, jnp.zeros(2), atol=1e-15)

    # Hand-calculated test at non-zero state: theta=pi/2, omega=2.0, tau=1.0
    x_test = jnp.array([np.pi / 2, 2.0])
    u_test = jnp.array([1.0])
    xdot_test = model.dynamics(x_test, u_test)
    np.testing.assert_allclose(xdot_test, jnp.array([2.0, -16.42]), rtol=1e-12)

    # Jacobians
    fx = model.state_jacobian(x_test, u_test)
    fu = model.control_jacobian(x_test, u_test)
    J = model.jacobian(x_test, u_test)

    assert fx.shape == (2, 2)
    assert fu.shape == (2, 1)
    assert J.shape == (2, 3)
    np.testing.assert_allclose(J, jnp.hstack([fx, fu]))

    # Analytical state Jacobian check
    expected_fx = jnp.array(
        [
            [0.0, 1.0],
            [-9.81 / 0.5 * np.cos(np.pi / 2), -0.1 / 0.25],
        ]
    )
    expected_fu = jnp.array(
        [
            [0.0],
            [1.0 / 0.25],
        ]
    )
    np.testing.assert_allclose(fx, expected_fx, atol=1e-14)
    np.testing.assert_allclose(fu, expected_fu, atol=1e-14)


def test_dubins_car_dynamics_and_jacobians() -> None:
    model = DubinsCar()
    x = jnp.array([1.0, 2.0, np.pi / 3])
    u = jnp.array([2.0, -0.5])

    # Dynamics verification: xdot = [v*cos(theta), v*sin(theta), omega]
    xdot = model.dynamics(x, u)
    assert xdot.shape == (3,)
    expected_xdot = jnp.array([2.0 * np.cos(np.pi / 3), 2.0 * np.sin(np.pi / 3), -0.5])
    np.testing.assert_allclose(xdot, expected_xdot, rtol=1e-14)

    # Jacobians
    fx = model.state_jacobian(x, u)
    fu = model.control_jacobian(x, u)
    J = model.jacobian(x, u)

    assert fx.shape == (3, 3)
    assert fu.shape == (3, 2)
    assert J.shape == (3, 5)
    np.testing.assert_allclose(J, jnp.hstack([fx, fu]))

    # Analytical check
    expected_fx = jnp.array(
        [
            [0.0, 0.0, -2.0 * np.sin(np.pi / 3)],
            [0.0, 0.0, 2.0 * np.cos(np.pi / 3)],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_fu = jnp.array(
        [
            [np.cos(np.pi / 3), 0.0],
            [np.sin(np.pi / 3), 0.0],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(fx, expected_fx, atol=1e-14)
    np.testing.assert_allclose(fu, expected_fu, atol=1e-14)


def test_equinox_leaf_static_split() -> None:
    # 1. Pendulum
    pendulum = Pendulum()
    leaves_p = jax.tree_util.tree_leaves(pendulum)
    # Leaves: mass, len, b, lc, I, g (6 parameters)
    assert len(leaves_p) == 6
    leaf_vals_p = [float(leaf) for leaf in leaves_p]
    assert 1.0 in leaf_vals_p
    assert 0.5 in leaf_vals_p
    assert 0.1 in leaf_vals_p
    assert 0.25 in leaf_vals_p
    assert 9.81 in leaf_vals_p

    # 2. DubinsCar
    car = DubinsCar()
    leaves_c = jax.tree_util.tree_leaves(car)
    # Leaves: radius (1 parameter)
    assert len(leaves_c) == 1
    assert float(leaves_c[0]) == 0.175

    # 3. JIT compilation works seamlessly
    @eqx.filter_jit
    def eval_pendulum(m: Pendulum, x: jax.Array, u: jax.Array) -> jax.Array:
        return m.dynamics(x, u)

    @eqx.filter_jit
    def eval_car(m: DubinsCar, x: jax.Array, u: jax.Array) -> jax.Array:
        return m.dynamics(x, u)

    xp = jnp.array([0.1, 0.2])
    up = jnp.array([0.5])
    res_p = eval_pendulum(pendulum, xp, up)
    assert res_p.shape == (2,)

    xc = jnp.array([0.1, 0.2, 0.3])
    uc = jnp.array([1.0, 0.1])
    res_c = eval_car(car, xc, uc)
    assert res_c.shape == (3,)


def test_quadrotor_scope_boundary() -> None:
    """Explicitly verify and document that Quadrotor is out of scope for Ticket 05.

    Quadrotor is a RigidBody system with manifold state [r, q, v, omega] (n=13, ne=12)
    and belongs to the rotations strand (Ticket 13: rigidbody-quadrotor-error-expansions.md).
    """
    # Quadrotor must not be exported until Ticket 13
    assert not hasattr(models, "Quadrotor"), "Quadrotor belongs to Ticket 13 (rotations / rigid body strand)"
