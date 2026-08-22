from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import NegativeOrthant
from trajopt.constraints.base import ControlConstraint, StageConstraint, StateConstraint

BoundSpec = jax.Array | np.ndarray | Sequence[float] | None


class Box(eqx.Module):
    """Box limits lo <= z <= hi, reduced to gathers over the finitely bounded entries.

    Parameters
    ----------
    lo : jax.Array | Sequence[float] | None
        Lower limits of shape (dim,) or (1,) to broadcast; None means -inf everywhere.
    hi : jax.Array | Sequence[float] | None
        Upper limits of shape (dim,) or (1,) to broadcast; None means +inf everywhere.
    dim : int
        Length of the bounded vector z.
    """

    lo: jax.Array
    hi: jax.Array
    lo_finite: jax.Array
    hi_finite: jax.Array
    i_min_arr: jax.Array
    i_max_arr: jax.Array
    i_min: tuple[int, ...] = eqx.field(static=True)
    i_max: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, lo: BoundSpec, hi: BoundSpec, dim: int) -> None:
        lo_arr = np.full(dim, -np.inf) if lo is None else np.asarray(lo, dtype=float)
        hi_arr = np.full(dim, np.inf) if hi is None else np.asarray(hi, dtype=float)

        if len(lo_arr) == 1 and dim > 1:
            lo_arr = np.full(dim, lo_arr[0])
        if len(hi_arr) == 1 and dim > 1:
            hi_arr = np.full(dim, hi_arr[0])

        if np.any(hi_arr < lo_arr):
            msg = "Upper bounds must be greater than or equal to lower bounds."
            raise ValueError(msg)

        i_max = tuple(int(i) for i, v in enumerate(hi_arr) if np.isfinite(v))
        i_min = tuple(int(i) for i, v in enumerate(lo_arr) if np.isfinite(v))

        self.lo = jnp.asarray(lo_arr)
        self.hi = jnp.asarray(hi_arr)
        self.i_max = i_max
        self.i_min = i_min
        self.i_max_arr = jnp.asarray(i_max, dtype=int)
        self.i_min_arr = jnp.asarray(i_min, dtype=int)
        self.hi_finite = jnp.asarray(hi_arr[list(i_max)])
        self.lo_finite = jnp.asarray(lo_arr[list(i_min)])

    @property
    def p(self) -> int:
        """Number of finite bound rows, len(i_max) + len(i_min)."""
        return len(self.i_max) + len(self.i_min)

    def residual(self, z: jax.Array) -> jax.Array:
        """Evaluate [z[i_max] - hi; lo - z[i_min]] of shape (p,) from z of shape (dim,)."""
        if self.p == 0:
            return jnp.zeros(0, dtype=z.dtype)

        parts = []
        if len(self.i_max) > 0:
            parts.append(z[self.i_max_arr] - self.hi_finite)
        if len(self.i_min) > 0:
            parts.append(self.lo_finite - z[self.i_min_arr])
        return jnp.concatenate(parts)


class StateBound(StateConstraint):
    """Box bound constraint on states: x_min <= x <= x_max.

    Maps to NegativeOrthant: [x[i_max] - x_max; x_min - x[i_min]] <= 0.
    """

    bounds_box: Box

    def __init__(
        self,
        n: int,
        x_min: BoundSpec = None,
        x_max: BoundSpec = None,
        m: int = 0,
    ) -> None:
        box = Box(x_min, x_max, int(n))
        super().__init__(n=int(n), m=int(m), p=box.p, cone=NegativeOrthant())
        self.bounds_box = box

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state bound residual of shape (p,) from x of shape (n,)."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate StateBound."
            raise ValueError(msg)
        return self.bounds_box.residual(x)

    def primal_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return solver variable limits (x_min, x_max), each of shape (n,)."""
        return self.bounds_box.lo, self.bounds_box.hi


class ControlBound(ControlConstraint):
    """Box bound constraint on controls: u_min <= u <= u_max.

    Maps to NegativeOrthant: [u[i_max] - u_max; u_min - u[i_min]] <= 0.
    """

    bounds_box: Box

    def __init__(
        self,
        m: int,
        u_min: BoundSpec = None,
        u_max: BoundSpec = None,
        n: int = 0,
    ) -> None:
        box = Box(u_min, u_max, int(m))
        super().__init__(n=int(n), m=int(m), p=box.p, cone=NegativeOrthant())
        self.bounds_box = box

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control bound residual of shape (p,) from u of shape (m,)."""
        del x, t
        if u is None:
            msg = "Control vector u is required to evaluate ControlBound."
            raise ValueError(msg)
        return self.bounds_box.residual(u)

    def primal_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return solver variable limits (u_min, u_max), each of shape (m,)."""
        return self.bounds_box.lo, self.bounds_box.hi


class BoundConstraint(StageConstraint):
    """Combined box bound constraint on state and control: z_min <= z <= z_max.

    Maps to NegativeOrthant: [z[i_max] - z_max; z_min - z[i_min]] <= 0 where z = [x; u].
    """

    bounds_box: Box

    def __init__(  # noqa: PLR0913
        self,
        n: int,
        m: int,
        *,
        x_min: BoundSpec = None,
        x_max: BoundSpec = None,
        u_min: BoundSpec = None,
        u_max: BoundSpec = None,
        z_min: BoundSpec = None,
        z_max: BoundSpec = None,
    ) -> None:
        n_int = int(n)
        m_int = int(m)

        if z_min is not None or z_max is not None:
            box = Box(z_min, z_max, n_int + m_int)
        else:
            x_box = Box(x_min, x_max, n_int)
            u_box = Box(u_min, u_max, m_int)
            box = Box(
                np.concatenate([x_box.lo, u_box.lo]),
                np.concatenate([x_box.hi, u_box.hi]),
                n_int + m_int,
            )

        super().__init__(n=n_int, m=m_int, p=box.p, cone=NegativeOrthant())
        self.bounds_box = box

    def uses_control(self) -> bool:
        """Whether any finite bound falls in the control block of z = [x; u]."""
        return any(i >= self.n for i in self.bounds_box.i_min + self.bounds_box.i_max)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate combined bound residual of shape (p,) from x of shape (n,) and u of shape (m,)."""
        del t
        return self.bounds_box.residual(self.stage_vector(x, u))

    def primal_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return solver variable limits (z_min, z_max), each of shape (n + m,)."""
        return self.bounds_box.lo, self.bounds_box.hi
