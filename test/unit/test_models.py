import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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
