import jax
import jax.numpy as jnp

from trajopt.dynamics.base import ContinuousDynamics


class Cartpole(ContinuousDynamics):
    """Cartpole benchmark model: a cart with a pendulum attached to it.

    State:
        x = [p, theta, p_dot, theta_dot]
        p : cart position (m)
        theta : pole angle from downward vertical (rad)
        p_dot : cart velocity (m/s)
        theta_dot : pole angular velocity (rad/s)

    Control:
        u = [F]
        F : horizontal force applied to the cart (N)

    Parameters
    ----------
    mc : float | jax.Array, optional
        Mass of the cart in kg. Defaults to 1.0.
    mp : float | jax.Array, optional
        Mass of the pendulum pole in kg. Defaults to 0.2.
    l : float | jax.Array, optional
        Length of the pendulum pole in m. Defaults to 0.5.
    g : float | jax.Array, optional
        Acceleration due to gravity in m/s^2. Defaults to 9.81.
    """

    mc: jax.Array
    mp: jax.Array
    l: jax.Array  # noqa: E741
    g: jax.Array

    def __init__(
        self,
        mc: float | jax.Array = 1.0,
        mp: float | jax.Array = 0.2,
        l: float | jax.Array = 0.5,  # noqa: E741
        g: float | jax.Array = 9.81,
    ) -> None:
        super().__init__(n=4, m=1, ne=4)
        self.mc = jnp.asarray(mc, dtype=jnp.float64)
        self.mp = jnp.asarray(mp, dtype=jnp.float64)
        self.l = jnp.asarray(l, dtype=jnp.float64)
        self.g = jnp.asarray(g, dtype=jnp.float64)

    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate continuous-time cartpole dynamics xdot = f(x, u, t).

        Parameters
        ----------
        x : jax.Array
            State vector [p, theta, p_dot, theta_dot] of shape (4,).
        u : jax.Array
            Control vector [F] of shape (1,).
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.

        Returns
        -------
        jax.Array
            State derivative [p_dot, theta_dot, p_ddot, theta_ddot] of shape (4,).
        """
        del t
        x_arr = jnp.asarray(x)
        u_arr = jnp.asarray(u)

        p_dot = x_arr[2]
        theta = x_arr[1]
        theta_dot = x_arr[3]

        s = jnp.sin(theta)
        c = jnp.cos(theta)

        H = jnp.array(
            [
                [self.mc + self.mp, self.mp * self.l * c],
                [self.mp * self.l * c, self.mp * (self.l**2)],
            ]
        )
        C = jnp.array(
            [
                [0.0, -self.mp * theta_dot * self.l * s],
                [0.0, 0.0],
            ]
        )
        G = jnp.array([0.0, self.mp * self.g * self.l * s])
        B = jnp.array([1.0, 0.0])

        u_val = u_arr[0] if u_arr.ndim > 0 else u_arr
        qd = jnp.array([p_dot, theta_dot])
        rhs = C @ qd + G - B * u_val
        qdd = -jnp.linalg.solve(H, rhs)

        return jnp.concatenate([qd, qdd])
