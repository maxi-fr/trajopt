from collections.abc import Sequence

import jax
import jax.numpy as jnp

from trajopt.dynamics.base import RigidBody
from trajopt.rotations.quaternion import Quaternion

_TENSOR_NDIM = 2


class Quadrotor(RigidBody):
    """Quadrotor benchmark model with parameters matched bit-for-bit to RobotZoo.jl.

    State:
        x = [r, q, v, omega]
        r : position in world frame [px, py, pz] of shape (3,)
        q : attitude JPL quaternion [qx, qy, qz, qw] in body frame of shape (4,)
        v : linear velocity in world frame [vx, vy, vz] of shape (3,)
        omega : angular velocity in body frame [wx, wy, wz] of shape (3,)

    Control:
        u = [w1, w2, w3, w4]
        Rotor thrust forces / squared motor speeds of shape (4,). Each rotor force is clamped at
        max(0, kf * w_i), as in RobotZoo, so the model is non-differentiable at w_i = 0.

    Parameters
    ----------
    mass : float | jax.Array, optional
        Mass of the quadrotor in kg. Defaults to 0.5.
    J : Sequence[float] | jax.Array, optional
        Principal moments of inertia [Jx, Jy, Jz] in kg*m^2 of shape (3,), or a diagonal
        (3, 3) tensor. Defaults to (0.0023, 0.0023, 0.004).
    gravity : Sequence[float] | jax.Array, optional
        Gravitational acceleration vector in world frame of shape (3,).
        Defaults to (0.0, 0.0, -9.81).
    motor_dist : float | jax.Array, optional
        Distance from center of mass to motor axis in m. Defaults to 0.1750.
    kf : float | jax.Array, optional
        Motor thrust coefficient. Defaults to 1.0.
    km : float | jax.Array, optional
        Motor torque coefficient. Defaults to 0.0245.
    """

    mass: jax.Array
    J: jax.Array
    J_inv: jax.Array
    gravity: jax.Array
    motor_dist: jax.Array
    kf: jax.Array
    km: jax.Array

    def __init__(  # noqa: PLR0913, PLR0917 -- Quadrotor takes 6 physical parameters
        self,
        mass: float | jax.Array = 0.5,
        J: Sequence[float] | jax.Array = (0.0023, 0.0023, 0.004),
        gravity: Sequence[float] | jax.Array = (0.0, 0.0, -9.81),
        motor_dist: float | jax.Array = 0.1750,
        kf: float | jax.Array = 1.0,
        km: float | jax.Array = 0.0245,
    ) -> None:
        super().__init__(m=4)
        self.mass = jnp.asarray(mass, dtype=float)
        J_arr = jnp.asarray(J, dtype=float)
        if J_arr.ndim == _TENSOR_NDIM:
            # Only a diagonal tensor survives the elementwise inverse below, so reject the rest
            # rather than silently discarding the off-diagonal terms.
            if J_arr.shape != (3, 3) or not bool(jnp.all(J_arr == jnp.diag(jnp.diag(J_arr)))):
                msg = (
                    f"J must be a (3,) vector of principal moments or a diagonal (3, 3) tensor, got shape {J_arr.shape}"
                )
                raise ValueError(msg)
            J_arr = jnp.diag(J_arr)
        self.J = J_arr
        self.J_inv = 1.0 / J_arr
        self.gravity = jnp.asarray(gravity, dtype=float)
        self.motor_dist = jnp.asarray(motor_dist, dtype=float)
        self.kf = jnp.asarray(kf, dtype=float)
        self.km = jnp.asarray(km, dtype=float)

    def rotor_forces(self, u: jax.Array) -> jax.Array:
        """Compute per-rotor thrust forces max(0, kf * u) of shape (4,).

        A rotor cannot pull, so RobotZoo clamps each force at zero and this port follows it. The
        clamp is inactive wherever u >= 0, which the benchmark's ControlBound enforces; below
        zero it is what separates the two implementations, and it puts a kink in the dynamics.
        """
        return jnp.maximum(0.0, self.kf * u)

    @property
    def motor_mixing_matrix(self) -> jax.Array:
        """Motor mixing matrix mapping controls u to [Fz, tau_x, tau_y, tau_z] of shape (4, 4).

        The unclamped map, so it agrees with `forces` and `moments` only for u >= 0.
        """
        kf = self.kf
        km = self.km
        L = self.motor_dist
        return jnp.array(
            [
                [kf, kf, kf, kf],
                [0.0, L * kf, 0.0, -L * kf],
                [-L * kf, 0.0, L * kf, 0.0],
                [km, -km, km, -km],
            ]
        )

    def forces(self, x: jax.Array, u: jax.Array) -> jax.Array:
        """Compute total forces in world frame of shape (3,).

        Parameters
        ----------
        x : jax.Array
            State vector [r, q, v, omega] of shape (13,).
        u : jax.Array
            Control vector [u1, u2, u3, u4] of shape (4,).

        Returns
        -------
        jax.Array
            Total force in world frame of shape (3,).
        """
        f_thrust = jnp.sum(self.rotor_forces(u))
        f_body = jnp.array([0.0, 0.0, f_thrust], dtype=u.dtype)
        q = Quaternion.from_array(x[3:7])
        r_mat = q.to_rot_mat()
        f_world = r_mat.T @ f_body
        return self.mass * self.gravity + f_world

    def moments(self, x: jax.Array, u: jax.Array) -> jax.Array:
        """Compute total moments in body frame of shape (3,).

        Parameters
        ----------
        x : jax.Array
            State vector [r, q, v, omega] of shape (13,).
        u : jax.Array
            Control vector [u1, u2, u3, u4] of shape (4,).

        Returns
        -------
        jax.Array
            Total torque in body frame of shape (3,).
        """
        del x
        L = self.motor_dist
        # Roll and pitch come from the clamped rotor forces; yaw is a reaction torque taken
        # straight from the motor commands, unclamped. RobotZoo splits them the same way.
        f = self.rotor_forces(u)
        tau_x = L * (f[1] - f[3])
        tau_y = L * (f[2] - f[0])
        tau_z = self.km * (u[0] - u[1] + u[2] - u[3])
        return jnp.array([tau_x, tau_y, tau_z], dtype=u.dtype)

    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate continuous-time quadrotor dynamics xdot = f(x, u, t) of shape (13,).

        Parameters
        ----------
        x : jax.Array
            State vector [r, q, v, omega] of shape (13,).
        u : jax.Array
            Control vector [u1, u2, u3, u4] of shape (4,).
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.

        Returns
        -------
        jax.Array
            State derivative [v, qdot, vdot, omegadot] of shape (13,).
        """
        del t
        q = Quaternion.from_array(x[3:7])
        v = x[7:10]
        omega = x[10:13]

        r_dot = v
        q_dot = q.kinematics(omega)
        f_world = self.forces(x, u)
        v_dot = f_world / self.mass
        tau = self.moments(x, u)
        j_omega = self.J * omega
        gyro = jnp.cross(omega, j_omega)
        omega_dot = self.J_inv * (tau - gyro)

        return jnp.concatenate([r_dot, q_dot, v_dot, omega_dot])
