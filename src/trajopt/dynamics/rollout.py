"""Forward simulation and trajectory rollout using jax.lax.scan."""

import jax
import jax.numpy as jnp

from trajopt.dynamics.base import ContinuousDynamics, DiscreteDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import RK4
from trajopt.trajectory import Trajectory

_EXPECTED_NDIM_2D = 2


def _rollout_scan(
    model: DiscreteDynamics,
    x0: jax.Array,
    U: jax.Array,
    t: jax.Array,
    dt: jax.Array,
) -> jax.Array:
    """Core forward simulation scan kernel.

    Parameters
    ----------
    model : DiscreteDynamics
        Discrete-time dynamics model.
    x0 : jax.Array
        Initial state of shape (n,).
    U : jax.Array
        Control trajectory of shape (N-1, m).
    t : jax.Array
        Timestamps of shape (N,).
    dt : jax.Array
        Step durations of shape (N-1,).

    Returns
    -------
    jax.Array
        Simulated states X of shape (N, n).
    """

    def _step(carry: jax.Array, inputs: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_k = carry
        u_k, t_k, dt_k = inputs
        x_next = model.discrete_dynamics(x_k, u_k, t_k, dt_k)
        return x_next, x_next

    n_steps = U.shape[0]
    t_inputs = t[:n_steps]
    inputs = (U, t_inputs, dt)
    _, X_rest = jax.lax.scan(_step, x0, inputs)
    return jnp.concatenate([jnp.expand_dims(x0, axis=0), X_rest], axis=0)


def rollout_states(
    model: DiscreteDynamics | ContinuousDynamics,
    x0: jax.Array,
    U: jax.Array,
    t: jax.Array | float | None = None,
    dt: jax.Array | float | None = None,
) -> jax.Array:
    """Simulate states forward over time given initial condition and controls.

    Parameters
    ----------
    model : DiscreteDynamics | ContinuousDynamics
        Dynamics model. If continuous, defaults to RK4 integration.
    x0 : jax.Array
        Initial state vector of shape (n,).
    U : jax.Array
        Control trajectory of shape (N-1, m).
    t : jax.Array | float | None, optional
        Timestamps of shape (N,) or initial timestamp float. Defaults to 0.0 cumulative time.
    dt : jax.Array | float | None, optional
        Step durations of shape (N-1,) or scalar step duration. Defaults to 0.01.

    Returns
    -------
    jax.Array
        Stacked simulated states X of shape (N, n).
    """
    if isinstance(model, ContinuousDynamics) and not isinstance(model, DiscreteDynamics):
        discrete_model: DiscreteDynamics = DiscretizedDynamics(model, RK4())
    else:
        discrete_model = model

    x0_arr = jnp.asarray(x0)
    U_arr = jnp.asarray(U)
    if U_arr.ndim != _EXPECTED_NDIM_2D:
        msg = f"Control trajectory U must have 2 dimensions (N-1, m), got shape {U_arr.shape}"
        raise ValueError(msg)

    n_steps = U_arr.shape[0]
    if dt is None:
        dt_arr = jnp.full((n_steps,), 0.01, dtype=x0_arr.dtype)
    elif jnp.ndim(dt) == 0:
        dt_arr = jnp.full((n_steps,), dt, dtype=x0_arr.dtype)
    else:
        dt_arr = jnp.asarray(dt, dtype=x0_arr.dtype)

    if t is None:
        t_arr = jnp.concatenate([jnp.zeros((1,), dtype=x0_arr.dtype), jnp.cumsum(dt_arr)], axis=0)
    elif jnp.ndim(t) == 0:
        t_arr = t + jnp.concatenate([jnp.zeros((1,), dtype=x0_arr.dtype), jnp.cumsum(dt_arr)], axis=0)
    else:
        t_arr = jnp.asarray(t, dtype=x0_arr.dtype)

    return _rollout_scan(discrete_model, x0_arr, U_arr, t_arr, dt_arr)


def rollout(  # noqa: PLR0913
    model: DiscreteDynamics | ContinuousDynamics,
    trajectory_or_x0: Trajectory | jax.Array,
    U: jax.Array | None = None,
    t: jax.Array | float | None = None,
    dt: jax.Array | float | None = None,
    *,
    x0: jax.Array | None = None,
) -> Trajectory:
    """Forward simulate a dynamical system over a horizon using jax.lax.scan.

    Sets x_1 = x_0 and propagates x_{k+1} = f_d(x_k, u_k, t_k, dt_k).

    Parameters
    ----------
    model : DiscreteDynamics | ContinuousDynamics
        Dynamics model. If continuous, defaults to RK4 integration.
    trajectory_or_x0 : Trajectory | jax.Array
        Either a Trajectory containing controls, times, and step durations,
        or an initial state vector x0 of shape (n,).
    U : jax.Array | None, optional
        Control trajectory of shape (N-1, m). Required if trajectory_or_x0 is an array.
    t : jax.Array | float | None, optional
        Timestamps of shape (N,) or initial timestamp float. Defaults to 0.0 cumulative time.
    dt : jax.Array | float | None, optional
        Step durations of shape (N-1,) or scalar step duration. Defaults to 0.01.
    x0 : jax.Array | None, optional
        Override for the initial condition if trajectory_or_x0 is a Trajectory.

    Returns
    -------
    Trajectory
        New Trajectory instance containing simulated states X and the controls, times, and dt.
    """
    if isinstance(model, ContinuousDynamics) and not isinstance(model, DiscreteDynamics):
        discrete_model: DiscreteDynamics = DiscretizedDynamics(model, RK4())
    else:
        discrete_model = model

    if isinstance(trajectory_or_x0, Trajectory):
        traj = trajectory_or_x0
        x0_val = traj.X[0] if x0 is None else jnp.asarray(x0, dtype=traj.X.dtype)
        X_sim = _rollout_scan(discrete_model, x0_val, traj.U, traj.t, traj.dt)
        return Trajectory(X=X_sim, U=traj.U, t=traj.t, dt=traj.dt)

    # trajectory_or_x0 is initial state x0
    x0_arr = jnp.asarray(trajectory_or_x0)
    if U is None:
        msg = "Control trajectory U must be provided when passing initial state x0"
        raise ValueError(msg)
    U_arr = jnp.asarray(U)
    if U_arr.ndim != _EXPECTED_NDIM_2D:
        msg = f"Control trajectory U must have 2 dimensions (N-1, m), got shape {U_arr.shape}"
        raise ValueError(msg)

    n_steps = U_arr.shape[0]
    if dt is None:
        dt_arr = jnp.full((n_steps,), 0.01, dtype=x0_arr.dtype)
    elif jnp.ndim(dt) == 0:
        dt_arr = jnp.full((n_steps,), dt, dtype=x0_arr.dtype)
    else:
        dt_arr = jnp.asarray(dt, dtype=x0_arr.dtype)

    if t is None:
        t_arr = jnp.concatenate([jnp.zeros((1,), dtype=x0_arr.dtype), jnp.cumsum(dt_arr)], axis=0)
    elif jnp.ndim(t) == 0:
        t_arr = t + jnp.concatenate([jnp.zeros((1,), dtype=x0_arr.dtype), jnp.cumsum(dt_arr)], axis=0)
    else:
        t_arr = jnp.asarray(t, dtype=x0_arr.dtype)

    X_sim = _rollout_scan(discrete_model, x0_arr, U_arr, t_arr, dt_arr)
    return Trajectory(X=X_sim, U=U_arr, t=t_arr, dt=dt_arr)
