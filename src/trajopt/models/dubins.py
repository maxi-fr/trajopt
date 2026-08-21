"""Dubins car (unicycle) benchmark model matching RobotZoo.jl parameters and dynamics."""

import jax
import jax.numpy as jnp

from trajopt.dynamics.base import ContinuousDynamics


class DubinsCar(ContinuousDynamics):
    """Dubins car (unicycle) benchmark model.

    State:
        x = [x, y, theta]
        x : x position in global frame (m)
        y : y position in global frame (m)
        theta : heading / orientation angle (rad)

    Control:
        u = [v, omega]
        v : forward linear velocity (m/s)
        omega : angular velocity / turning rate (rad/s)

    Parameters
    ----------
    radius : float | jax.Array, optional
        Collision / body radius of the car in m. Defaults to 0.175.
    """

    radius: jax.Array

    def __init__(
        self,
        radius: float | jax.Array = 0.175,
    ) -> None:
        super().__init__(n=3, m=2, ne=3)
        self.radius = jnp.asarray(radius, dtype=jnp.float64)

    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate continuous-time Dubins car dynamics xdot = f(x, u, t).

        Parameters
        ----------
        x : jax.Array
            State vector [x, y, theta] of shape (3,).
        u : jax.Array
            Control vector [v, omega] of shape (2,).
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.

        Returns
        -------
        jax.Array
            State derivative [x_dot, y_dot, theta_dot] of shape (3,).
        """
        del t
        x_arr = jnp.asarray(x)
        u_arr = jnp.asarray(u)

        theta = x_arr[2]
        v = u_arr[0]
        omega = u_arr[1]

        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = omega

        return jnp.array([x_dot, y_dot, theta_dot])
