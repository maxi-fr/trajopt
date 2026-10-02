from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.cones import AbstractCone, NegativeOrthant

_MATRIX_NDIM = 2


class LinearHorizonConstraint(eqx.Module):
    """Linear Horizon residual ``A @ Z - b`` over the complete Primal Vector."""

    A: jax.Array
    b: jax.Array
    cone: AbstractCone
    inds: tuple[int, ...] = eqx.field(static=True)
    full_vector: bool = eqx.field(static=True)

    def __init__(
        self,
        A: jax.Array,
        b: jax.Array,
        *,
        sense: AbstractCone | None = None,
        inds: Sequence[int] | None = None,
    ) -> None:
        """Store rows whose columns address ``inds`` in the full Primal Vector."""
        A_arr = jnp.asarray(A)
        b_arr = jnp.asarray(b)
        if A_arr.ndim != _MATRIX_NDIM or b_arr.ndim != 1 or A_arr.shape[0] != b_arr.shape[0]:
            msg = "A must have shape (p, q) and b must have shape (p,)."
            raise ValueError(msg)
        columns = tuple(range(A_arr.shape[1])) if inds is None else tuple(int(i) for i in inds)
        if A_arr.shape[1] != len(columns) or len(set(columns)) != len(columns):
            msg = "A columns must match unique Primal Vector indices."
            raise ValueError(msg)
        self.A = A_arr
        self.b = b_arr
        self.cone = NegativeOrthant() if sense is None else sense
        self.inds = columns
        self.full_vector = inds is None

    @property
    def p(self) -> int:
        """Number of Horizon rows."""
        return int(self.A.shape[0])

    def evaluate(self, Z: jax.Array) -> jax.Array:
        """Evaluate residuals of shape (p,) from the flat Primal Vector."""
        return self.A @ Z[jnp.asarray(self.inds)] - self.b
