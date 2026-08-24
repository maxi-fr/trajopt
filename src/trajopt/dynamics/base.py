from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.rotations.quaternion import (
    Quaternion,
)
from trajopt.trajectory import Trajectory

IntegratorCallable = Callable[
    ["ContinuousDynamics", jax.Array, jax.Array, float | jax.Array, float | jax.Array], jax.Array
]


class AbstractModel(eqx.Module):
    """Dimensions, AD-derived Jacobians, and the error-state interface shared by all models.

    Parameters
    ----------
    n : int
        State dimension. Compile-time static metadata.
    m : int
        Control dimension. Compile-time static metadata.
    ne : int | None, optional
        Error-state dimension. Defaults to n for Euclidean state vectors.
    """

    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    ne: int = eqx.field(static=True)

    def __init__(self, n: int, m: int, ne: int | None = None) -> None:
        self.n = n
        self.m = m
        self.ne = n if ne is None else ne

    @abstractmethod
    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the model at state x of shape (n,) and control u of shape (m,).

        Continuous models return xdot of shape (n,) and take no extra arguments; discrete
        models return the next state of shape (n,) and take the step duration dt.
        """

    @abstractmethod
    def discretize(self, integrator: "IntegratorCallable | None" = None) -> "DiscreteDynamics":
        """Return this model in discrete form, wrapping a continuous model with integrator (default RK4)."""

    def rollout(self, trajectory: Trajectory, x0: jax.Array | None = None) -> Trajectory:
        """Forward simulate this model over trajectory's controls, timestamps, and step durations.

        Sets x_0 to x0 (defaulting to trajectory's first state) and propagates
        x_{k+1} = f_d(x_k, u_k, t_k, dt_k) using the controls, timestamps, and step durations
        stored in trajectory. A continuous model is discretized with RK4.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory supplying the controls, timestamps, and step durations.
        x0 : jax.Array | None, optional
            Initial state of shape (n,). Defaults to the trajectory's first state.

        Returns
        -------
        Trajectory
            New Trajectory holding the simulated states X of shape (N, n) and the inputs it was given.
        """
        from trajopt.dynamics.rollout import _rollout_scan  # noqa: PLC0415 -- avoid circular import

        x0_val = trajectory.X[0] if x0 is None else jnp.asarray(x0, dtype=trajectory.X.dtype)
        X_sim = _rollout_scan(self.discretize(), x0_val, trajectory.U, trajectory.t, trajectory.dt)
        return Trajectory(X=X_sim, U=trajectory.U, t=trajectory.t, dt=trajectory.dt)

    def state_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the state Jacobian of shape (n, n) via automatic differentiation."""
        return jax.jacobian(lambda x_: self.evaluate(x_, u, t, *args))(x)

    def control_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the control Jacobian of shape (n, m) via automatic differentiation."""
        return jax.jacobian(lambda u_: self.evaluate(x, u_, t, *args))(u)

    def jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the joint Jacobian [df/dx, df/du] of shape (n, n + m)."""
        fx = self.state_jacobian(x, u, t, *args)
        fu = self.control_jacobian(x, u, t, *args)
        return jnp.hstack([fx, fu])

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute the error state dx = x (-) x0 of shape (ne,) from states of shape (n,).

        Defaults to Euclidean subtraction x - x0.
        """
        return x - x0

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the error-state Jacobian G = dx / d(delta x) of shape (n, ne).

        Defaults to the identity matrix for Euclidean systems.
        """
        return jnp.eye(self.n, dtype=x.dtype)


class ContinuousDynamics(AbstractModel):
    """Abstract base class for continuous-time dynamics models: xdot = f(x, u, t)."""

    @abstractmethod
    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate xdot = f(x, u, t) of shape (n,) from x of shape (n,) and u of shape (m,)."""

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Evaluate xdot of shape (n,); extra arguments are not used by continuous models."""
        del args
        return self.dynamics(x, u, t)

    def discretize(self, integrator: "IntegratorCallable | None" = None) -> "DiscreteDynamics":
        """Wrap this model with integrator, defaulting to RK4."""
        from trajopt.dynamics.integrators import RK4  # noqa: PLC0415 -- avoid circular import

        integ = integrator if integrator is not None else RK4()
        return DiscretizedDynamics(self, integ)


class EuclideanModel(ContinuousDynamics):
    """Continuous-time model on a Euclidean state space: ne == n with an identity error map."""

    def __init__(self, n: int, m: int) -> None:
        super().__init__(n=n, m=m, ne=n)


class RigidBody(ContinuousDynamics):
    """Continuous-time rigid-body dynamics model with SO(3) attitude state.

    State layout (n = 13):
        x = [r, q, v, omega]
        r : position in world frame of shape (3,)
        q : JPL unit quaternion [qx, qy, qz, qw] in body frame of shape (4,)
        v : linear velocity in world frame of shape (3,)
        omega : angular velocity in body frame of shape (3,)

    Error state layout (ne = 12):
        delta_x = [delta_r, delta_theta, delta_v, delta_omega]
        delta_r : position error of shape (3,)
        delta_theta : small-angle attitude error 2 * vec(q (x) q0^-1) of shape (3,)
        delta_v : linear velocity error of shape (3,)
        delta_omega : angular velocity error of shape (3,)

    Parameters
    ----------
    m : int
        Control dimension. Compile-time static metadata.
    """

    def __init__(self, m: int) -> None:
        super().__init__(n=13, m=m, ne=12)

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute error state dx = x (-) x0 of shape (12,) from states of shape (13,).

        Parameters
        ----------
        x : jax.Array
            Current state of shape (13,).
        x0 : jax.Array
            Reference state of shape (13,).

        Returns
        -------
        jax.Array
            Error state vector of shape (12,).
        """
        dr = x[:3] - x0[:3]
        q = Quaternion.from_array(x[3:7])
        q0 = Quaternion.from_array(x0[3:7])
        dtheta = q.error_map(q0)
        dv = x[7:10] - x0[7:10]
        domega = x[10:13] - x0[10:13]
        return jnp.concatenate([dr, dtheta, dv, domega])

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate error-state Jacobian G = dx / d(delta x) of shape (13, 12).

        Parameters
        ----------
        x : jax.Array
            State vector of shape (13,).

        Returns
        -------
        jax.Array
            Error-state Jacobian blockdiag(I_3, 0.5 * Xi(q), I_3, I_3) of shape (13, 12).
        """
        q = Quaternion.from_array(x[3:7])
        g_rot = q.attitude_jacobian()
        eye3 = jnp.eye(3, dtype=x.dtype)
        z33 = jnp.zeros((3, 3), dtype=x.dtype)
        z43 = jnp.zeros((4, 3), dtype=x.dtype)

        row0 = jnp.hstack([eye3, z33, z33, z33])
        row1 = jnp.hstack([z43, g_rot, z43, z43])
        row2 = jnp.hstack([z33, z33, eye3, z33])
        row3 = jnp.hstack([z33, z33, z33, eye3])
        return jnp.vstack([row0, row1, row2, row3])


class DiscreteDynamics(AbstractModel):
    """Abstract base class for discrete-time dynamics: x_{k+1} = f_d(x_k, u_k, t_k, dt_k)."""

    @abstractmethod
    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the next state of shape (n,) from x of shape (n,) and u of shape (m,)."""

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the next state of shape (n,); the single extra argument is the step duration dt."""
        (dt,) = args
        return self.discrete_dynamics(x, u, t, dt)

    def discretize(self, integrator: "IntegratorCallable | None" = None) -> "DiscreteDynamics":
        """Return self; already discrete."""
        del integrator
        return self


class DiscretizedDynamics(DiscreteDynamics):
    """Discrete-time dynamics pairing a continuous-time model with an integrator."""

    continuous_dynamics: ContinuousDynamics
    integrator: IntegratorCallable = eqx.field(static=True)

    def __init__(
        self,
        continuous_dynamics: ContinuousDynamics,
        integrator: IntegratorCallable,
    ) -> None:
        super().__init__(
            n=continuous_dynamics.n,
            m=continuous_dynamics.m,
            ne=continuous_dynamics.ne,
        )
        self.continuous_dynamics = continuous_dynamics
        self.integrator = integrator

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the next state of shape (n,) via the configured integrator."""
        return self.integrator(self.continuous_dynamics, x, u, t, dt)

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute the error state of shape (ne,), delegating to the continuous-time model."""
        return self.continuous_dynamics.state_diff(x, x0)

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the error-state Jacobian of shape (n, ne), delegating to the continuous model."""
        return self.continuous_dynamics.errstate_jacobian(x)
