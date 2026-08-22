import jax
import jax.numpy as jnp

from trajopt.dynamics.base import ContinuousDynamics


class Pendulum(ContinuousDynamics):
    """Pendulum benchmark model: a simple pendulum controlled by torque at the base.

    State:
        x = [theta, omega]
        theta : angle from downward vertical (rad)
        omega : angular velocity (rad/s)

    Control:
        u = [tau]
        tau : torque applied at the base (N*m)

    Parameters
    ----------
    mass : float | jax.Array, optional
        Mass of the pendulum in kg. Defaults to 1.0.
    len : float | jax.Array, optional
        Total length of the pendulum in m. Defaults to 0.5.
    b : float | jax.Array, optional
        Viscous friction coefficient in N*m*s/rad. Defaults to 0.1.
    lc : float | jax.Array, optional
        Distance to center of mass in m. Defaults to 0.5.
    I : float | jax.Array, optional
        Rotational inertia about center of mass in kg*m^2. Defaults to 0.25.
    g : float | jax.Array, optional
        Acceleration due to gravity in m/s^2. Defaults to 9.81.
    """

    mass: jax.Array
    len: jax.Array
    b: jax.Array
    lc: jax.Array
    I: jax.Array  # noqa: E741
    g: jax.Array

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        mass: float | jax.Array = 1.0,
        len: float | jax.Array = 0.5,  # noqa: A002
        b: float | jax.Array = 0.1,
        lc: float | jax.Array = 0.5,
        I: float | jax.Array = 0.25,  # noqa: E741
        g: float | jax.Array = 9.81,
    ) -> None:
        super().__init__(n=2, m=1, ne=2)
        self.mass = jnp.asarray(mass, dtype=jnp.float64)
        self.len = jnp.asarray(len, dtype=jnp.float64)
        self.b = jnp.asarray(b, dtype=jnp.float64)
        self.lc = jnp.asarray(lc, dtype=jnp.float64)
        self.I = jnp.asarray(I, dtype=jnp.float64)
        self.g = jnp.asarray(g, dtype=jnp.float64)

    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate continuous-time pendulum dynamics xdot = f(x, u, t).

        Parameters
        ----------
        x : jax.Array
            State vector [theta, omega] of shape (2,).
        u : jax.Array
            Control vector [tau] of shape (1,).
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.

        Returns
        -------
        jax.Array
            State derivative [omega, theta_ddot] of shape (2,).
        """
        del t
        x_arr = jnp.asarray(x)
        u_arr = jnp.asarray(u)

        theta = x_arr[0]
        omega = x_arr[1]
        tau = u_arr[0] if u_arr.ndim > 0 else u_arr

        m_eff = self.mass * (self.lc**2)
        theta_ddot = tau / m_eff - self.g * jnp.sin(theta) / self.lc - self.b * omega / m_eff

        return jnp.array([omega, theta_ddot])
