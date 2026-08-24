import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt import models
from trajopt.dynamics.base import RigidBody
from trajopt.models import Cartpole, DubinsCar, Pendulum, Quadrotor
from trajopt.rotations.quaternion import Quaternion, attitude_jacobian, error_map


class _ConcreteRigidBody(RigidBody):
    def __init__(self, m: int = 4) -> None:
        super().__init__(m=m)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del u, t
        return jnp.zeros_like(x)


def test_rigid_body_properties_and_error_interface() -> None:
    """Assert RigidBody base layout [r, q, v, omega] with n=13 and ne=12."""
    model = _ConcreteRigidBody(m=4)
    assert model.n == 13
    assert model.m == 4
    assert model.ne == 12

    # State layout: r(3), q(4), v(3), omega(3)
    q1 = Quaternion.from_array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])
    q0 = Quaternion.from_array([0.0, np.sin(-0.2), 0.0, np.cos(-0.2)])
    x = jnp.array([1.0, 2.0, 3.0, *q1.to_array(), 0.5, -0.5, 1.0, 0.1, 0.2, -0.3])
    x0 = jnp.array([0.5, 1.5, 2.0, *q0.to_array(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # state_diff check
    dx = model.state_diff(x, x0)
    assert dx.shape == (12,)
    np.testing.assert_allclose(dx[:3], x[:3] - x0[:3])
    np.testing.assert_allclose(dx[3:6], error_map(q1, q0))
    np.testing.assert_allclose(dx[6:9], x[7:10] - x0[7:10])
    np.testing.assert_allclose(dx[9:12], x[10:13] - x0[10:13])

    # errstate_jacobian shape and block diagonal structure
    G = model.errstate_jacobian(x)
    assert G.shape == (13, 12), "Attitude Jacobian G must be (n, ne) = (13, 12)"

    # Shape assertion pinning the direction: G maps error variations (12,) into state variations (13,)
    dx_error = jnp.ones(12)
    dx_state = G @ dx_error
    assert dx_state.shape == (13,), "G @ dx_error must produce state variation of shape (13,)"

    # Block diagonal structure: [I3, 0, 0, 0; 0, 0.5*Xi(q), 0, 0; 0, 0, I3, 0; 0, 0, 0, I3]
    eye3 = np.eye(3)
    np.testing.assert_allclose(G[:3, :3], eye3)
    np.testing.assert_allclose(G[:3, 3:], np.zeros((3, 9)))
    np.testing.assert_allclose(G[3:7, :3], np.zeros((4, 3)))
    np.testing.assert_allclose(G[3:7, 3:6], attitude_jacobian(q1))
    np.testing.assert_allclose(G[3:7, 6:], np.zeros((4, 6)))
    np.testing.assert_allclose(G[7:10, :6], np.zeros((3, 6)))
    np.testing.assert_allclose(G[7:10, 6:9], eye3)
    np.testing.assert_allclose(G[7:10, 9:], np.zeros((3, 3)))
    np.testing.assert_allclose(G[10:13, :9], np.zeros((3, 9)))
    np.testing.assert_allclose(G[10:13, 9:12], eye3)


def test_quadrotor_instantiation_and_parameters() -> None:
    """Assert Quadrotor default parameters match RobotZoo bit-for-bit."""
    model = Quadrotor()
    assert model.n == 13
    assert model.m == 4
    assert model.ne == 12

    assert float(model.mass) == 0.5
    np.testing.assert_allclose(model.J, np.array([0.0023, 0.0023, 0.004]))
    np.testing.assert_allclose(model.gravity, np.array([0.0, 0.0, -9.81]))
    assert float(model.motor_dist) == 0.1750
    assert float(model.kf) == 1.0
    assert float(model.km) == 0.0245

    # Custom parameters
    custom = Quadrotor(
        mass=1.2,
        J=(0.005, 0.005, 0.009),
        gravity=(0.0, 0.0, -9.80665),
        motor_dist=0.25,
        kf=1.5,
        km=0.03,
    )
    assert float(custom.mass) == 1.2
    assert float(custom.motor_dist) == 0.25
    assert float(custom.kf) == 1.5
    assert float(custom.km) == 0.03


def test_quadrotor_motor_mixing_matrix() -> None:
    """Assert Quadrotor motor mixing matrix maps controls to [Fz, tau_x, tau_y, tau_z]."""
    model = Quadrotor()
    M = model.motor_mixing_matrix
    assert M.shape == (4, 4)

    kf = 1.0
    km = 0.0245
    L = 0.1750
    expected_M = np.array(
        [
            [kf, kf, kf, kf],
            [0.0, L * kf, 0.0, -L * kf],
            [-L * kf, 0.0, L * kf, 0.0],
            [km, -km, km, -km],
        ]
    )
    np.testing.assert_allclose(M, expected_M)

    u = jnp.array([1.2, 0.8, 1.5, 0.9])
    forces_and_torques = M @ u
    assert forces_and_torques.shape == (4,)
    np.testing.assert_allclose(forces_and_torques[0], kf * jnp.sum(u))
    np.testing.assert_allclose(forces_and_torques[1], L * kf * (u[1] - u[3]))
    np.testing.assert_allclose(forces_and_torques[2], L * kf * (u[2] - u[0]))
    np.testing.assert_allclose(forces_and_torques[3], km * (u[0] - u[1] + u[2] - u[3]))


def test_quadrotor_dynamics_hover_equilibrium() -> None:
    """Assert Quadrotor in level attitude with hover thrust produces xdot = 0."""
    model = Quadrotor()
    # Level attitude in JPL: [0, 0, 0, 1]
    x_hover = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Hover thrust per motor: F_total = m * g => 4 * kf * u_i = 0.5 * 9.81 => u_i = 0.5 * 9.81 / 4
    u_hover = jnp.full(4, float(model.mass * 9.81 / 4.0))

    xdot = model.dynamics(x_hover, u_hover)
    assert xdot.shape == (13,)
    np.testing.assert_allclose(xdot, np.zeros(13), atol=1e-14)


def test_quadrotor_dynamics_and_jacobians() -> None:
    """Assert Quadrotor continuous dynamics, state Jacobian, and control Jacobian."""
    model = Quadrotor()
    q = Quaternion.from_array([0.1, -0.2, 0.3, 0.9])
    q_norm = float(np.linalg.norm(q.to_array()))
    q_unit = Quaternion.from_array(q.to_array() / q_norm)

    x = jnp.array([1.0, 2.0, 3.0, *q_unit.to_array(), 0.5, -0.2, 0.8, 0.1, -0.3, 0.2])
    u = jnp.array([1.5, 1.2, 1.8, 1.0])

    xdot = model.dynamics(x, u)
    assert xdot.shape == (13,)

    # Position dot = velocity
    np.testing.assert_allclose(xdot[:3], x[7:10])

    # Attitude dot = 0.5 * Xi(q) @ omega
    np.testing.assert_allclose(xdot[3:7], q_unit.kinematics(x[10:13]))

    # Jacobians
    fx = model.state_jacobian(x, u)
    fu = model.control_jacobian(x, u)
    J = model.jacobian(x, u)

    assert fx.shape == (13, 13)
    assert fu.shape == (13, 4)
    assert J.shape == (13, 17)
    np.testing.assert_allclose(J, jnp.hstack([fx, fu]))

    # Equinox JIT test
    @eqx.filter_jit
    def eval_quad(m: Quadrotor, x_: jax.Array, u_: jax.Array) -> jax.Array:
        return m.dynamics(x_, u_)

    xdot_jit = eval_quad(model, x, u)
    np.testing.assert_allclose(xdot_jit, xdot, atol=1e-15)


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


def test_quadrotor_rotor_forces_clamp_at_zero() -> None:
    """Assert a negative rotor command produces no thrust and no roll/pitch torque, as in RobotZoo."""
    model = Quadrotor()
    u = jnp.array([-1.0, 2.0, 1.0, 0.5])

    np.testing.assert_allclose(model.rotor_forces(u), [0.0, 2.0, 1.0, 0.5])

    x = jnp.zeros(13).at[6].set(1.0)  # identity JPL quaternion, scalar-last
    xdot = np.asarray(model.dynamics(x, u))

    # Values read off a live RobotZoo.jl Quadrotor at the same state and control. Without the
    # clamp the unclamped sum gives -4.81 and 152.17 instead.
    np.testing.assert_allclose(xdot[9], -2.81, atol=1e-12)
    np.testing.assert_allclose(xdot[11], 0.175 * (1.0 - 0.0) / 0.0023, rtol=1e-12)
    np.testing.assert_allclose(xdot[11], 76.08695652173913, rtol=1e-12)

    # Yaw is a reaction torque taken from the commands themselves, so it stays unclamped.
    np.testing.assert_allclose(xdot[12], 0.0245 * (-1.0 - 2.0 + 1.0 - 0.5) / 0.004, rtol=1e-12)


def test_quadrotor_clamp_is_inactive_for_nonnegative_controls() -> None:
    """Assert the clamp leaves the model unchanged wherever the control bounds keep u at or above zero."""
    model = Quadrotor()
    M = model.motor_mixing_matrix
    rng = np.random.default_rng(17)

    for _ in range(20):
        u = jnp.asarray(rng.uniform(0.0, 3.0, size=4))
        x = jnp.asarray(np.concatenate([rng.standard_normal(3), [0.0, 0.0, 0.0, 1.0], rng.standard_normal(6)]))
        wrench = M @ u
        np.testing.assert_allclose(jnp.sum(model.rotor_forces(u)), wrench[0], rtol=1e-14)
        np.testing.assert_allclose(model.moments(x, u), wrench[1:], rtol=1e-14, atol=1e-15)
