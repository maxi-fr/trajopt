from collections.abc import Iterator
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

_MIN_HORIZON = 2
_EXPECTED_NDIM_2D = 2
_EXPECTED_NDIM_1D = 1


@dataclass(frozen=True)
class KnotPoint:
    """Read-only view of a single knot point along a trajectory.

    Parameters
    ----------
    x : jax.Array
        State vector of shape (n,).
    u : jax.Array | None
        Control vector of shape (m,), or None for terminal knot points.
    t : jax.Array | float
        Timestamp of the knot point.
    dt : jax.Array | float
        Time step duration to the next knot point (0.0 for terminal knot points).
    is_terminal : bool
        Whether this is the final knot point of the trajectory.
    """

    x: jax.Array
    u: jax.Array | None
    t: jax.Array | float
    dt: jax.Array | float
    is_terminal: bool


class Trajectory(eqx.Module):
    """Struct-of-arrays trajectory storage holding states, controls, times, and step durations.

    Holds states `X` (N, n), controls `U` (N-1, m), timestamps `t` (N,), and step durations `dt` (N-1,)
    as stacked arrays. All operations return new instances to preserve immutability.

    Parameters
    ----------
    X : jax.Array
        Stacked state trajectory of shape (N, n).
    U : jax.Array
        Stacked control trajectory of shape (N-1, m).
    t : jax.Array
        Stacked timestamp array of shape (N,).
    dt : jax.Array
        Stacked step durations array of shape (N-1,).
    """

    X: jax.Array
    U: jax.Array
    t: jax.Array
    dt: jax.Array

    def __init__(
        self,
        X: jax.Array,
        U: jax.Array,
        t: jax.Array,
        dt: jax.Array,
    ) -> None:
        X_arr = jnp.asarray(X)
        U_arr = jnp.asarray(U)
        t_arr = jnp.asarray(t)
        dt_arr = jnp.asarray(dt)

        if X_arr.ndim != _EXPECTED_NDIM_2D:
            msg = f"State array X must have 2 dimensions (N, n), got shape {X_arr.shape}"
            raise ValueError(msg)
        if U_arr.ndim != _EXPECTED_NDIM_2D:
            msg = f"Control array U must have 2 dimensions (N-1, m), got shape {U_arr.shape}"
            raise ValueError(msg)
        if t_arr.ndim != _EXPECTED_NDIM_1D:
            msg = f"Time array t must have 1 dimension (N,), got shape {t_arr.shape}"
            raise ValueError(msg)
        if dt_arr.ndim != _EXPECTED_NDIM_1D:
            msg = f"Step duration array dt must have 1 dimension (N-1,), got shape {dt_arr.shape}"
            raise ValueError(msg)

        n_knots = X_arr.shape[0]
        if n_knots < _MIN_HORIZON:
            msg = f"Horizon N must be at least 2, got {n_knots}"
            raise ValueError(msg)
        if U_arr.shape[0] != n_knots - 1:
            msg = f"Controls U length ({U_arr.shape[0]}) must equal N - 1 ({n_knots - 1})"
            raise ValueError(msg)
        if t_arr.shape[0] != n_knots:
            msg = f"Times t length ({t_arr.shape[0]}) must equal N ({n_knots})"
            raise ValueError(msg)
        if dt_arr.shape[0] != n_knots - 1:
            msg = f"Step durations dt length ({dt_arr.shape[0]}) must equal N - 1 ({n_knots - 1})"
            raise ValueError(msg)

        self.X = X_arr
        self.U = U_arr
        self.t = t_arr
        self.dt = dt_arr

    @property
    def N(self) -> int:  # noqa: N802
        """Number of knot points in the trajectory."""
        return int(self.X.shape[0])

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.X.shape[1])

    @property
    def m(self) -> int:
        """Control dimension."""
        return int(self.U.shape[1])

    def states(self) -> jax.Array:
        """Return the stacked state array X of shape (N, n)."""
        return self.X

    def controls(self) -> jax.Array:
        """Return the stacked control array U of shape (N-1, m)."""
        return self.U

    def times(self) -> jax.Array:
        """Return the stacked timestamp array t of shape (N,)."""
        return self.t

    def set_states(self, X: jax.Array) -> "Trajectory":
        """Return a new Trajectory with updated states.

        Parameters
        ----------
        X : jax.Array
            New states of shape (N, n).

        Returns
        -------
        Trajectory
            New trajectory instance.
        """
        return Trajectory(X=X, U=self.U, t=self.t, dt=self.dt)

    def set_controls(self, U: jax.Array) -> "Trajectory":
        """Return a new Trajectory with updated controls.

        Parameters
        ----------
        U : jax.Array
            New controls of shape (N-1, m).

        Returns
        -------
        Trajectory
            New trajectory instance.
        """
        return Trajectory(X=self.X, U=U, t=self.t, dt=self.dt)

    def with_initial_time(self, t0: float | jax.Array) -> "Trajectory":
        """Return a new Trajectory shifted so that the initial time is t0.

        Parameters
        ----------
        t0 : float | jax.Array
            New initial timestamp.

        Returns
        -------
        Trajectory
            New trajectory instance with shifted timestamps.
        """
        t0_val = jnp.asarray(t0, dtype=self.t.dtype)
        shift_amount = t0_val - self.t[0]
        new_t = self.t + shift_amount
        return Trajectory(X=self.X, U=self.U, t=new_t, dt=self.dt)

    def shift(self, dt: float | jax.Array | None = None) -> "Trajectory":
        """Shift the trajectory forward one step for MPC warm-starting.

        Discards the first state and control, shifts all elements forward by one index,
        and repeats the final state and control at the end of the horizon.

        Parameters
        ----------
        dt : float | jax.Array | None, optional
            Duration for the appended final step. If None, the previous final step duration is used.

        Returns
        -------
        Trajectory
            New trajectory instance advanced by one step.
        """
        new_X = jnp.concatenate([self.X[1:], self.X[-1:]], axis=0)
        new_U = jnp.concatenate([self.U[1:], self.U[-1:]], axis=0)

        if dt is None:
            new_dt = jnp.concatenate([self.dt[1:], self.dt[-1:]], axis=0)
            final_dt = self.dt[-1]
        else:
            final_dt = jnp.asarray(dt, dtype=self.dt.dtype)
            new_dt = jnp.concatenate([self.dt[1:], jnp.atleast_1d(final_dt)], axis=0)

        t_final = self.t[-1] + jnp.squeeze(final_dt)
        new_t = jnp.concatenate([self.t[1:], jnp.atleast_1d(t_final)], axis=0)

        return Trajectory(X=new_X, U=new_U, t=new_t, dt=new_dt)

    def __len__(self) -> int:
        """Return the number of knot points N."""
        return self.N

    def __getitem__(self, idx: int) -> KnotPoint:
        """Return a KnotPoint view at the specified index.

        Parameters
        ----------
        idx : int
            Index of the knot point (supports negative indexing).

        Returns
        -------
        KnotPoint
            View of the knot point.

        Raises
        ------
        IndexError
            If idx is out of range [ -N, N - 1 ].
        """
        if not isinstance(idx, int):
            msg = f"Trajectory index must be an integer, got {type(idx).__name__}"
            raise TypeError(msg)

        n_pts = self.N
        k = n_pts + idx if idx < 0 else idx
        if k < 0 or k >= n_pts:
            msg = f"Trajectory index {idx} out of range for trajectory of length {n_pts}"
            raise IndexError(msg)

        is_term = k == n_pts - 1
        return KnotPoint(
            x=self.X[k],
            u=self.U[k] if not is_term else None,
            t=self.t[k],
            dt=self.dt[k] if not is_term else 0.0,
            is_terminal=is_term,
        )

    def __iter__(self) -> Iterator[KnotPoint]:
        """Iterate over knot-point views for all knot points in the trajectory."""
        for k in range(self.N):
            yield self[k]
