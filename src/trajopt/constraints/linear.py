from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import AbstractCone, NegativeOrthant, ZeroCone
from trajopt.constraints.base import StageConstraint, StateConstraint

EXPECTED_A_NDIM = 2
EXPECTED_B_NDIM = 1


class GoalConstraint(StateConstraint):
    """Goal state constraint: x[inds] - xf = 0.

    Parameters
    ----------
    n : int
        State dimension.
    xf : jax.Array | Sequence[float]
        Goal state vector. Can be length p = len(inds) or full state length n.
    inds : Sequence[int] | None, optional
        State indices constrained to goal values. Defaults to all indices (0..n-1).
    m : int, optional
        Control dimension. Defaults to 0.
    """

    xf: jax.Array
    inds_arr: jax.Array
    inds: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        xf: jax.Array | Sequence[float],
        inds: Sequence[int] | None = None,
        m: int = 0,
    ) -> None:
        n_int = int(n)
        m_int = int(m)

        inds_tuple = tuple(range(n_int)) if inds is None else tuple(int(i) for i in inds)

        p = len(inds_tuple)
        xf_arr = np.asarray(xf, dtype=float)
        if len(xf_arr) == n_int and p != n_int:
            xf_arr = xf_arr[list(inds_tuple)]
        elif len(xf_arr) != p:
            msg = f"Goal state xf length ({len(xf_arr)}) does not match indices length ({p})."
            raise ValueError(msg)

        super().__init__(n=n_int, m=m_int, p=p, cone=ZeroCone())
        self.xf = jnp.asarray(xf_arr)
        self.inds = inds_tuple
        self.inds_arr = jnp.asarray(inds_tuple, dtype=int)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate goal defect x[inds] - xf of shape (p,) from x of shape (n,)."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate GoalConstraint."
            raise ValueError(msg)
        return x[self.inds_arr] - self.xf


class LinearConstraint(StageConstraint):
    """Linear constraint: A * z[inds] - b {<=, =} 0 where z = [x; u].

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Control dimension.
    A : jax.Array
        Constraint matrix of shape (p, len(inds)).
    b : jax.Array
        Constraint vector of shape (p,).
    sense : AbstractCone | None, optional
        Conic set for constraint residual (default NegativeOrthant() for A z - b <= 0).
    inds : Sequence[int] | None, optional
        Indices into concatenated vector z = [x; u]. Defaults to all indices (0..n+m-1).
    """

    A: jax.Array
    b: jax.Array
    inds_arr: jax.Array
    inds: tuple[int, ...] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        n: int,
        m: int,
        A: jax.Array,
        b: jax.Array,
        *,
        sense: AbstractCone | None = None,
        inds: Sequence[int] | None = None,
    ) -> None:
        n_int = int(n)
        m_int = int(m)
        nm = n_int + m_int

        inds_tuple = tuple(range(nm)) if inds is None else tuple(int(i) for i in inds)

        A_arr = jnp.asarray(A)
        b_arr = jnp.asarray(b)

        if A_arr.ndim != EXPECTED_A_NDIM:
            msg = f"A must be a 2D matrix, got shape {A_arr.shape}."
            raise ValueError(msg)
        if b_arr.ndim != EXPECTED_B_NDIM:
            msg = f"b must be a 1D vector, got shape {b_arr.shape}."
            raise ValueError(msg)
        if A_arr.shape[0] != b_arr.shape[0]:
            msg = f"Row dimension of A ({A_arr.shape[0]}) does not match length of b ({b_arr.shape[0]})."
            raise ValueError(msg)
        if A_arr.shape[1] != len(inds_tuple):
            msg = f"Column dimension of A ({A_arr.shape[1]}) does not match inds length ({len(inds_tuple)})."
            raise ValueError(msg)

        p = int(A_arr.shape[0])
        cone = NegativeOrthant() if sense is None else sense
        super().__init__(n=n_int, m=m_int, p=p, cone=cone)
        self.A = A_arr
        self.b = b_arr
        self.inds = inds_tuple
        self.inds_arr = jnp.asarray(inds_tuple, dtype=int)

    def uses_control(self) -> bool:
        """Whether any gathered index falls in the control block of z = [x; u]."""
        return any(i >= self.n for i in self.inds)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate residual A * z[inds] - b of shape (p,) from z = [x; u] of shape (n + m,)."""
        del t
        return self.A @ self.stage_vector(x, u)[self.inds_arr] - self.b
