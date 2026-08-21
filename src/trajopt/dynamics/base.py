"""Base classes and types for continuous, discrete, and discretized dynamical systems."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp


class IntegratorProtocol(Protocol):
    """Protocol for callables that step a continuous dynamics model forward by dt."""

    def __call__(
        self,
        continuous_dynamics: "ContinuousDynamics",
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Step continuous_dynamics from state x with control u at time t for duration dt."""
        ...


IntegratorCallable = Callable[
    ["ContinuousDynamics", jax.Array, jax.Array, float | jax.Array, float | jax.Array], jax.Array
]


class ContinuousDynamics(eqx.Module):
    """Abstract base class for continuous-time dynamics models: xdot = f(x, u, t).

    Parameters
    ----------
    n : int
        State dimension. Declared as compile-time static metadata.
    m : int
        Control dimension. Declared as compile-time static metadata.
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
    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the continuous-time dynamics vector xdot = f(x, u, t).

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.

        Returns
        -------
        jax.Array
            Time derivative of state xdot of shape (n,).
        """

    def state_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the state Jacobian df/dx at (x, u, t) via automatic differentiation.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Current time.

        Returns
        -------
        jax.Array
            State Jacobian matrix of shape (n, n).
        """
        return jax.jacobian(lambda x_: self.dynamics(x_, u, t))(x)

    def control_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the control Jacobian df/du at (x, u, t) via automatic differentiation.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Current time.

        Returns
        -------
        jax.Array
            Control Jacobian matrix of shape (n, m).
        """
        return jax.jacobian(lambda u_: self.dynamics(x, u_, t))(u)

    def jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the joint Jacobian [df/dx, df/du] of shape (n, n + m).

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Current time.

        Returns
        -------
        jax.Array
            Jacobian matrix of shape (n, n + m).
        """
        fx = self.state_jacobian(x, u, t)
        fu = self.control_jacobian(x, u, t)
        return jnp.hstack([fx, fu])

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute the error state dx = x (-) x0.

        Defaults to standard Euclidean subtraction x - x0.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        x0 : jax.Array
            Nominal state vector of shape (n,).

        Returns
        -------
        jax.Array
            Error state vector of shape (ne,).
        """
        return x - x0

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the attitude / error-state Jacobian G = dx / d(delta x).

        Defaults to the identity matrix of shape (n, n) for Euclidean systems.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).

        Returns
        -------
        jax.Array
            Error-state Jacobian matrix of shape (n, ne).
        """
        return jnp.eye(self.n, dtype=x.dtype)


class EuclideanModel(ContinuousDynamics):
    """Convenience base class for continuous-time Euclidean dynamical models."""

    def __init__(self, n: int, m: int, ne: int | None = None) -> None:
        super().__init__(n=n, m=m, ne=n if ne is None else ne)


class DiscreteDynamics(eqx.Module):
    """Abstract base class for discrete-time dynamics models: x_{k+1} = f_d(x_k, u_k, t_k, dt_k).

    Parameters
    ----------
    n : int
        State dimension. Declared as compile-time static metadata.
    m : int
        Control dimension. Declared as compile-time static metadata.
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
    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the discrete step x_{k+1} = f_d(x_k, u_k, t_k, dt_k).

        Parameters
        ----------
        x : jax.Array
            Current state of shape (n,).
        u : jax.Array
            Current control of shape (m,).
        t : float | jax.Array
            Current timestamp.
        dt : float | jax.Array
            Time step duration.

        Returns
        -------
        jax.Array
            Next state of shape (n,).
        """

    def state_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the discrete state Jacobian df_d/dx at (x, u, t, dt) via AD.

        Parameters
        ----------
        x : jax.Array
            Current state of shape (n,).
        u : jax.Array
            Current control of shape (m,).
        t : float | jax.Array
            Current timestamp.
        dt : float | jax.Array
            Time step duration.

        Returns
        -------
        jax.Array
            State transition Jacobian A_k of shape (n, n).
        """
        return jax.jacobian(lambda x_: self.discrete_dynamics(x_, u, t, dt))(x)

    def control_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the discrete control Jacobian df_d/du at (x, u, t, dt) via AD.

        Parameters
        ----------
        x : jax.Array
            Current state of shape (n,).
        u : jax.Array
            Current control of shape (m,).
        t : float | jax.Array
            Current timestamp.
        dt : float | jax.Array
            Time step duration.

        Returns
        -------
        jax.Array
            Control transition Jacobian B_k of shape (n, m).
        """
        return jax.jacobian(lambda u_: self.discrete_dynamics(x, u_, t, dt))(u)

    def jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the joint discrete Jacobian [A_k, B_k] of shape (n, n + m).

        Parameters
        ----------
        x : jax.Array
            Current state of shape (n,).
        u : jax.Array
            Current control of shape (m,).
        t : float | jax.Array
            Current timestamp.
        dt : float | jax.Array
            Time step duration.

        Returns
        -------
        jax.Array
            Jacobian matrix of shape (n, n + m).
        """
        fx = self.state_jacobian(x, u, t, dt)
        fu = self.control_jacobian(x, u, t, dt)
        return jnp.hstack([fx, fu])

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute the error state dx = x (-) x0.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        x0 : jax.Array
            Nominal state vector of shape (n,).

        Returns
        -------
        jax.Array
            Error state vector of shape (ne,).
        """
        return x - x0

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the error-state Jacobian G = dx / d(delta x).

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).

        Returns
        -------
        jax.Array
            Error-state Jacobian matrix of shape (n, ne).
        """
        return jnp.eye(self.n, dtype=x.dtype)


class DiscretizedDynamics(DiscreteDynamics):
    """Discrete-time dynamics obtained by pairing a continuous-time model with an integrator.

    Parameters
    ----------
    continuous_dynamics : ContinuousDynamics
        Underlying continuous-time model.
    integrator : Any
        Integrator function or object implementing step / integrate.
    """

    continuous_dynamics: ContinuousDynamics
    integrator: Any = eqx.field(static=True)

    def __init__(
        self,
        continuous_dynamics: ContinuousDynamics,
        integrator: IntegratorCallable | object,
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
        """Evaluate the discrete step via the configured integrator."""
        integ: Any = self.integrator
        if callable(integ):
            res: jax.Array = integ(self.continuous_dynamics, x, u, t, dt)
            return res
        if hasattr(integ, "step"):
            res_step: jax.Array = integ.step(self.continuous_dynamics, x, u, t, dt)
            return res_step
        if hasattr(integ, "integrate"):
            res_int: jax.Array = integ.integrate(self.continuous_dynamics, x, u, t, dt)
            return res_int
        msg = f"Unsupported integrator type: {type(integ).__name__}"
        raise TypeError(msg)

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute the error state, delegating to the continuous-time model."""
        return self.continuous_dynamics.state_diff(x, x0)

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the error-state Jacobian, delegating to the continuous-time model."""
        return self.continuous_dynamics.errstate_jacobian(x)
