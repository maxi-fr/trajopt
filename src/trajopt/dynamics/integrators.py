"""Numerical integrators for continuous-time dynamics models."""

from abc import ABC, abstractmethod
from typing import Any, overload

import jax
import jax.numpy as jnp

from trajopt.dynamics.base import ContinuousDynamics, DiscretizedDynamics


def euler_step(
    model: ContinuousDynamics,
    x: jax.Array,
    u: jax.Array,
    t: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.0,
) -> jax.Array:
    """Explicit Euler step: x_{k+1} = x_k + dt * f(x_k, u_k, t_k).

    Parameters
    ----------
    model : ContinuousDynamics
        Continuous-time dynamics model.
    x : jax.Array
        Current state vector of shape (n,).
    u : jax.Array
        Control input vector of shape (m,).
    t : float | jax.Array, optional
        Current timestamp. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.0.

    Returns
    -------
    jax.Array
        Next state vector x_{k+1} of shape (n,).
    """
    dt_arr = jnp.asarray(dt)
    return x + dt_arr * model.dynamics(x, u, t)


def rk4_step(
    model: ContinuousDynamics,
    x: jax.Array,
    u: jax.Array,
    t: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.0,
) -> jax.Array:
    """Explicit 4th-order Runge-Kutta step.

    Parameters
    ----------
    model : ContinuousDynamics
        Continuous-time dynamics model.
    x : jax.Array
        Current state vector of shape (n,).
    u : jax.Array
        Control input vector of shape (m,).
    t : float | jax.Array, optional
        Current timestamp. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.0.

    Returns
    -------
    jax.Array
        Next state vector x_{k+1} of shape (n,).
    """
    dt_arr = jnp.asarray(dt)
    half_dt = 0.5 * dt_arr

    k1 = model.dynamics(x, u, t)
    k2 = model.dynamics(x + half_dt * k1, u, t + half_dt)
    k3 = model.dynamics(x + half_dt * k2, u, t + half_dt)
    k4 = model.dynamics(x + dt_arr * k3, u, t + dt_arr)

    return x + (dt_arr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def implicit_midpoint_step(  # noqa: PLR0913, PLR0917
    model: ContinuousDynamics,
    x: jax.Array,
    u: jax.Array,
    t: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.0,
    iters: int = 10,
) -> jax.Array:
    """Implicit midpoint step solved via fixed-iteration Newton's method.

    Solves the collocation root equation:
        r(x_{k+1}) = x_k + dt * f((x_k + x_{k+1}) / 2, u_k, t_k + dt / 2) - x_{k+1} = 0

    Parameters
    ----------
    model : ContinuousDynamics
        Continuous-time dynamics model.
    x : jax.Array
        Current state vector of shape (n,).
    u : jax.Array
        Control input vector of shape (m,).
    t : float | jax.Array, optional
        Current timestamp. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.0.
    iters : int, optional
        Number of fixed Newton iterations. Defaults to 10.

    Returns
    -------
    jax.Array
        Next state vector x_{k+1} of shape (n,).
    """
    dt_arr = jnp.asarray(dt)
    half_dt = 0.5 * dt_arr
    t_mid = t + half_dt
    n = model.n
    eye_n = jnp.eye(n, dtype=x.dtype)

    def newton_body(i: int, x_curr: jax.Array) -> jax.Array:
        del i
        x_mid = 0.5 * (x + x_curr)
        f_mid = model.dynamics(x_mid, u, t_mid)
        r = x + dt_arr * f_mid - x_curr
        df_dx = model.state_jacobian(x_mid, u, t_mid)
        # J_residual = 0.5 * dt * df_dx - I  => (I - 0.5 * dt * df_dx) delta = r
        a_mat = eye_n - half_dt * df_dx
        delta = jnp.linalg.solve(a_mat, r)
        return x_curr + delta

    return jax.lax.fori_loop(0, iters, newton_body, x)


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    @abstractmethod
    def __call__(
        self,
        continuous_dynamics: ContinuousDynamics,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Step continuous_dynamics forward by dt from state x with control u at time t.

        Parameters
        ----------
        continuous_dynamics : ContinuousDynamics
            Continuous-time dynamical model.
        x : jax.Array
            Current state of shape (n,).
        u : jax.Array
            Current control of shape (m,).
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.
        dt : float | jax.Array, optional
            Time step duration. Defaults to 0.0.

        Returns
        -------
        jax.Array
            Next state of shape (n,).
        """

    def step(
        self,
        continuous_dynamics: ContinuousDynamics,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Step continuous_dynamics forward by dt."""
        return self(continuous_dynamics, x, u, t, dt)


class Euler(Integrator):
    """Explicit Euler numerical integrator: x_{k+1} = x_k + dt * f(x_k, u_k, t_k)."""

    @overload
    def __new__(cls, model: ContinuousDynamics) -> DiscretizedDynamics: ...

    @overload
    def __new__(cls, model: None = None) -> "Euler": ...

    def __new__(
        cls,
        model: ContinuousDynamics | None = None,
    ) -> Any:
        """Create an Euler integrator or return a DiscretizedDynamics if passed a ContinuousDynamics model."""
        if model is not None and isinstance(model, ContinuousDynamics):
            integrator = super().__new__(cls)
            return DiscretizedDynamics(model, integrator)
        return super().__new__(cls)

    def __call__(
        self,
        continuous_dynamics: ContinuousDynamics,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Step continuous_dynamics using explicit Euler integration."""
        return euler_step(continuous_dynamics, x, u, t, dt)


class RK4(Integrator):
    """Explicit 4th-order Runge-Kutta numerical integrator."""

    @overload
    def __new__(cls, model: ContinuousDynamics) -> DiscretizedDynamics: ...

    @overload
    def __new__(cls, model: None = None) -> "RK4": ...

    def __new__(
        cls,
        model: ContinuousDynamics | None = None,
    ) -> Any:
        """Create an RK4 integrator or return a DiscretizedDynamics if passed a ContinuousDynamics model."""
        if model is not None and isinstance(model, ContinuousDynamics):
            integrator = super().__new__(cls)
            return DiscretizedDynamics(model, integrator)
        return super().__new__(cls)

    def __call__(
        self,
        continuous_dynamics: ContinuousDynamics,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Step continuous_dynamics using explicit RK4 integration."""
        return rk4_step(continuous_dynamics, x, u, t, dt)


class ImplicitMidpoint(Integrator):
    """Implicit midpoint numerical integrator solved via fixed-iteration Newton's method.

    Equation:
        x_{k+1} - x_k - dt * f((x_k + x_{k+1}) / 2, u_k, t_k + dt / 2) = 0

    Parameters
    ----------
    model : ContinuousDynamics | None, optional
        If provided, returns a DiscretizedDynamics wrapping this model.
    iters : int, optional
        Number of fixed Newton iterations. Defaults to 10.
    """

    iters: int

    @overload
    def __new__(cls, model: ContinuousDynamics, *, iters: int = 10) -> DiscretizedDynamics: ...

    @overload
    def __new__(cls, model: None = None, *, iters: int = 10) -> "ImplicitMidpoint": ...

    def __new__(
        cls,
        model: ContinuousDynamics | None = None,
        *,
        iters: int = 10,
    ) -> Any:
        """Create an ImplicitMidpoint integrator or return DiscretizedDynamics if passed a model."""
        if model is not None and isinstance(model, ContinuousDynamics):
            integrator = super().__new__(cls)
            integrator.iters = iters
            return DiscretizedDynamics(model, integrator)
        instance = super().__new__(cls)
        instance.iters = iters
        return instance

    def __init__(
        self,
        model: ContinuousDynamics | None = None,
        *,
        iters: int = 10,
    ) -> None:
        del model
        self.iters = iters

    def __call__(
        self,
        continuous_dynamics: ContinuousDynamics,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Step continuous_dynamics using implicit midpoint integration."""
        return implicit_midpoint_step(continuous_dynamics, x, u, t, dt, iters=self.iters)
