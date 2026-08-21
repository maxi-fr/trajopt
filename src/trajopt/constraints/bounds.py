"""Box bound constraints for states, controls, and combined stages."""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import NegativeOrthant
from trajopt.constraints.base import ControlConstraint, StageConstraint, StateConstraint


class StateBound(StateConstraint):
    """Box bound constraint on states: x_min <= x <= x_max.

    Maps to NegativeOrthant: [x[i_max] - x_max; x_min - x[i_min]] <= 0.
    """

    x_min: jax.Array
    x_max: jax.Array
    x_min_finite: jax.Array
    x_max_finite: jax.Array
    i_min_arr: jax.Array
    i_max_arr: jax.Array
    i_min: tuple[int, ...] = eqx.field(static=True)
    i_max: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        x_min: jax.Array | Sequence[float] | None = None,
        x_max: jax.Array | Sequence[float] | None = None,
        m: int = 0,
    ) -> None:
        n_int = int(n)
        m_int = int(m)

        x_min_arr = np.full(n_int, -np.inf) if x_min is None else np.asarray(x_min, dtype=float)
        x_max_arr = np.full(n_int, np.inf) if x_max is None else np.asarray(x_max, dtype=float)

        if len(x_min_arr) == 1 and n_int > 1:
            x_min_arr = np.full(n_int, x_min_arr[0])
        if len(x_max_arr) == 1 and n_int > 1:
            x_max_arr = np.full(n_int, x_max_arr[0])

        if np.any(x_max_arr < x_min_arr):
            msg = "Upper bounds must be greater than or equal to lower bounds."
            raise ValueError(msg)

        i_max = tuple(int(i) for i, v in enumerate(x_max_arr) if np.isfinite(v))
        i_min = tuple(int(i) for i, v in enumerate(x_min_arr) if np.isfinite(v))
        p = len(i_max) + len(i_min)

        super().__init__(n=n_int, m=m_int, p=p, cone=NegativeOrthant())

        self.x_min = jnp.asarray(x_min_arr)
        self.x_max = jnp.asarray(x_max_arr)
        self.i_max = i_max
        self.i_min = i_min
        self.i_max_arr = jnp.asarray(i_max, dtype=int)
        self.i_min_arr = jnp.asarray(i_min, dtype=int)
        self.x_max_finite = jnp.asarray(x_max_arr[list(i_max)])
        self.x_min_finite = jnp.asarray(x_min_arr[list(i_min)])

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state bounds inequality vector."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate StateBound."
            raise ValueError(msg)
        if self.p == 0:
            return jnp.zeros(0, dtype=x.dtype)

        parts = []
        if len(self.i_max) > 0:
            parts.append(x[self.i_max_arr] - self.x_max_finite)
        if len(self.i_min) > 0:
            parts.append(self.x_min_finite - x[self.i_min_arr])
        return jnp.concatenate(parts)

    def primal_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return solver variable limits (x_min, x_max) for the state vector."""
        return self.x_min, self.x_max


class ControlBound(ControlConstraint):
    """Box bound constraint on controls: u_min <= u <= u_max.

    Maps to NegativeOrthant: [u[i_max] - u_max; u_min - u[i_min]] <= 0.
    """

    u_min: jax.Array
    u_max: jax.Array
    u_min_finite: jax.Array
    u_max_finite: jax.Array
    i_min_arr: jax.Array
    i_max_arr: jax.Array
    i_min: tuple[int, ...] = eqx.field(static=True)
    i_max: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        m: int,
        u_min: jax.Array | Sequence[float] | None = None,
        u_max: jax.Array | Sequence[float] | None = None,
        n: int = 0,
    ) -> None:
        m_int = int(m)
        n_int = int(n)

        u_min_arr = np.full(m_int, -np.inf) if u_min is None else np.asarray(u_min, dtype=float)
        u_max_arr = np.full(m_int, np.inf) if u_max is None else np.asarray(u_max, dtype=float)

        if len(u_min_arr) == 1 and m_int > 1:
            u_min_arr = np.full(m_int, u_min_arr[0])
        if len(u_max_arr) == 1 and m_int > 1:
            u_max_arr = np.full(m_int, u_max_arr[0])

        if np.any(u_max_arr < u_min_arr):
            msg = "Upper bounds must be greater than or equal to lower bounds."
            raise ValueError(msg)

        i_max = tuple(int(i) for i, v in enumerate(u_max_arr) if np.isfinite(v))
        i_min = tuple(int(i) for i, v in enumerate(u_min_arr) if np.isfinite(v))
        p = len(i_max) + len(i_min)

        super().__init__(n=n_int, m=m_int, p=p, cone=NegativeOrthant())

        self.u_min = jnp.asarray(u_min_arr)
        self.u_max = jnp.asarray(u_max_arr)
        self.i_max = i_max
        self.i_min = i_min
        self.i_max_arr = jnp.asarray(i_max, dtype=int)
        self.i_min_arr = jnp.asarray(i_min, dtype=int)
        self.u_max_finite = jnp.asarray(u_max_arr[list(i_max)])
        self.u_min_finite = jnp.asarray(u_min_arr[list(i_min)])

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control bounds inequality vector."""
        del x, t
        if u is None:
            msg = "Control vector u is required to evaluate ControlBound."
            raise ValueError(msg)
        if self.p == 0:
            return jnp.zeros(0, dtype=u.dtype)

        parts = []
        if len(self.i_max) > 0:
            parts.append(u[self.i_max_arr] - self.u_max_finite)
        if len(self.i_min) > 0:
            parts.append(self.u_min_finite - u[self.i_min_arr])
        return jnp.concatenate(parts)

    def primal_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return solver variable limits (u_min, u_max) for the control vector."""
        return self.u_min, self.u_max


class BoundConstraint(StageConstraint):
    """Combined box bound constraint on state and control: z_min <= z <= z_max.

    Maps to NegativeOrthant: [z[i_max] - z_max; z_min - z[i_min]] <= 0 where z = [x; u].
    """

    z_min: jax.Array
    z_max: jax.Array
    z_min_finite: jax.Array
    z_max_finite: jax.Array
    i_min_arr: jax.Array
    i_max_arr: jax.Array
    i_min: tuple[int, ...] = eqx.field(static=True)
    i_max: tuple[int, ...] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        n: int,
        m: int,
        *,
        x_min: jax.Array | Sequence[float] | None = None,
        x_max: jax.Array | Sequence[float] | None = None,
        u_min: jax.Array | Sequence[float] | None = None,
        u_max: jax.Array | Sequence[float] | None = None,
        z_min: jax.Array | Sequence[float] | None = None,
        z_max: jax.Array | Sequence[float] | None = None,
    ) -> None:
        n_int = int(n)
        m_int = int(m)
        nm = n_int + m_int

        if z_min is not None or z_max is not None:
            z_min_arr = np.full(nm, -np.inf) if z_min is None else np.asarray(z_min, dtype=float)
            z_max_arr = np.full(nm, np.inf) if z_max is None else np.asarray(z_max, dtype=float)
        else:
            x_min_arr = np.full(n_int, -np.inf) if x_min is None else np.asarray(x_min, dtype=float)
            x_max_arr = np.full(n_int, np.inf) if x_max is None else np.asarray(x_max, dtype=float)
            u_min_arr = np.full(m_int, -np.inf) if u_min is None else np.asarray(u_min, dtype=float)
            u_max_arr = np.full(m_int, np.inf) if u_max is None else np.asarray(u_max, dtype=float)

            if len(x_min_arr) == 1 and n_int > 1:
                x_min_arr = np.full(n_int, x_min_arr[0])
            if len(x_max_arr) == 1 and n_int > 1:
                x_max_arr = np.full(n_int, x_max_arr[0])
            if len(u_min_arr) == 1 and m_int > 1:
                u_min_arr = np.full(m_int, u_min_arr[0])
            if len(u_max_arr) == 1 and m_int > 1:
                u_max_arr = np.full(m_int, u_max_arr[0])

            z_min_arr = np.concatenate([x_min_arr, u_min_arr])
            z_max_arr = np.concatenate([x_max_arr, u_max_arr])

        if np.any(z_max_arr < z_min_arr):
            msg = "Upper bounds must be greater than or equal to lower bounds."
            raise ValueError(msg)

        i_max = tuple(int(i) for i, v in enumerate(z_max_arr) if np.isfinite(v))
        i_min = tuple(int(i) for i, v in enumerate(z_min_arr) if np.isfinite(v))
        p = len(i_max) + len(i_min)

        super().__init__(n=n_int, m=m_int, p=p, cone=NegativeOrthant())

        self.z_min = jnp.asarray(z_min_arr)
        self.z_max = jnp.asarray(z_max_arr)
        self.i_max = i_max
        self.i_min = i_min
        self.i_max_arr = jnp.asarray(i_max, dtype=int)
        self.i_min_arr = jnp.asarray(i_min, dtype=int)
        self.z_max_finite = jnp.asarray(z_max_arr[list(i_max)])
        self.z_min_finite = jnp.asarray(z_min_arr[list(i_min)])

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate combined box bounds inequality vector."""
        del t
        if x is None or u is None:
            msg = "Both x and u are required to evaluate BoundConstraint."
            raise ValueError(msg)
        if self.p == 0:
            return jnp.zeros(0, dtype=x.dtype)

        z = jnp.concatenate([x, u])
        parts = []
        if len(self.i_max) > 0:
            parts.append(z[self.i_max_arr] - self.z_max_finite)
        if len(self.i_min) > 0:
            parts.append(self.z_min_finite - z[self.i_min_arr])
        return jnp.concatenate(parts)

    def primal_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return solver variable limits (z_min, z_max) for the combined [x; u] vector."""
        return self.z_min, self.z_max
