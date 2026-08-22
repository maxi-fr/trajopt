from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

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


class EuclideanModel(ContinuousDynamics):
    """Continuous-time model on a Euclidean state space: ne == n with an identity error map."""

    def __init__(self, n: int, m: int) -> None:
        super().__init__(n=n, m=m, ne=n)


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
