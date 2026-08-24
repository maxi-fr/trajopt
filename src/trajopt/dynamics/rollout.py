import jax
import jax.numpy as jnp

from trajopt.dynamics.base import ContinuousDynamics, DiscreteDynamics

_EXPECTED_NDIM_2D = 2


def _step_durations_and_times(
    t: jax.Array | float | None,
    dt: jax.Array | float | None,
    n_steps: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Normalize t and dt into arrays of shape (n_steps + 1,) and (n_steps,)."""
    if dt is None:
        dt_arr = jnp.full((n_steps,), 0.01, dtype=dtype)
    elif jnp.ndim(dt) == 0:
        dt_arr = jnp.full((n_steps,), dt, dtype=dtype)
    else:
        dt_arr = jnp.asarray(dt, dtype=dtype)

    if t is None or jnp.ndim(t) == 0:
        t0 = 0.0 if t is None else t
        t_arr = t0 + jnp.concatenate([jnp.zeros((1,), dtype=dtype), jnp.cumsum(dt_arr)], axis=0)
    else:
        t_arr = jnp.asarray(t, dtype=dtype)

    return dt_arr, t_arr


def _rollout_scan(
    model: DiscreteDynamics,
    x0: jax.Array,
    U: jax.Array,
    t: jax.Array,
    dt: jax.Array,
) -> jax.Array:
    """Simulate states X of shape (N, n) from x0 of shape (n,), U (N-1, m), t (N,), dt (N-1,)."""

    def _step(carry: jax.Array, inputs: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_k = carry
        u_k, t_k, dt_k = inputs
        x_next = model.discrete_dynamics(x_k, u_k, t_k, dt_k)
        return x_next, x_next

    n_steps = U.shape[0]
    inputs = (U, t[:n_steps], dt)
    _, X_rest = jax.lax.scan(_step, x0, inputs)
    return jnp.concatenate([jnp.expand_dims(x0, axis=0), X_rest], axis=0)


def rollout_states(
    model: DiscreteDynamics | ContinuousDynamics,
    x0: jax.Array,
    U: jax.Array,
    t: jax.Array | float | None = None,
    dt: jax.Array | float | None = None,
) -> jax.Array:
    """Simulate states forward over time given an initial condition and controls.

    Parameters
    ----------
    model
        Dynamics model. A continuous model is discretized with RK4.
    x0
        Initial state of shape (n,).
    U
        Control trajectory of shape (N-1, m).
    t
        Timestamps of shape (N,), or an initial timestamp. Defaults to cumulative time from 0.0.
    dt
        Step durations of shape (N-1,), or a scalar step duration. Defaults to 0.01.

    Returns
    -------
    Simulated states X of shape (N, n).
    """
    x0_arr = jnp.asarray(x0)
    U_arr = jnp.asarray(U)
    if U_arr.ndim != _EXPECTED_NDIM_2D:
        msg = f"Control trajectory U must have 2 dimensions (N-1, m), got shape {U_arr.shape}"
        raise ValueError(msg)

    dt_arr, t_arr = _step_durations_and_times(t, dt, n_steps=U_arr.shape[0], dtype=x0_arr.dtype)
    return _rollout_scan(model.discretize(), x0_arr, U_arr, t_arr, dt_arr)

