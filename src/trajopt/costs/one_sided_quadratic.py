from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.base import CostFunction
from trajopt.costs.quadratic import DiagonalCost
from trajopt.costs.reformulation import CostReformulation

_MATRIX_NDIM = 2


class OneSidedQuadratic(CostFunction):
    """Rows of a squared positive-part penalty, with residual `A @ z - b`.

    At a control Knot Point, z is [x, u]; at the terminal Knot Point, z is x.
    A has shape (p, len(z)), b has shape (p,), and weight is scalar or shape (p,).
    """

    A: jax.Array
    b: jax.Array
    weight: jax.Array

    formulation: Literal["epigraph", "direct"] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913 -- affine rows, physical dimensions and formulation are specified together
        self,
        A: jax.Array,
        b: jax.Array,
        weight: float | Sequence[float] | jax.Array,
        *,
        n: int,
        terminal: bool = False,
        formulation: Literal["epigraph", "direct"] = "epigraph",
    ) -> None:
        """Store finite affine rows and strictly positive weights."""
        A_arr, b_arr, w_arr = (np.asarray(value, dtype=float) for value in (A, b, weight))
        if A_arr.ndim != _MATRIX_NDIM or A_arr.shape[0] == 0 or b_arr.shape != (A_arr.shape[0],):
            msg = "A and b must have shapes (p, width) and (p,) with p > 0."
            raise ValueError(msg)
        if w_arr.shape not in ((), b_arr.shape) or not np.all(np.isfinite(w_arr) & (w_arr > 0)):
            msg = "weight must be a finite positive scalar or one positive weight per row."
            raise ValueError(msg)
        if not np.all(np.isfinite(A_arr)) or not np.all(np.isfinite(b_arr)):
            msg = "A and b must be finite."
            raise ValueError(msg)
        if A_arr.shape[1] < n or (terminal and A_arr.shape[1] != n):
            msg = "A must act on [x, u] for a Stage Cost or x for a Terminal Cost."
            raise ValueError(msg)
        if formulation not in ("epigraph", "direct"):
            msg = "formulation must be 'epigraph' or 'direct'."
            raise ValueError(msg)
        super().__init__(n, A_arr.shape[1] - n, terminal=terminal)
        self.formulation = formulation
        self.A = jnp.asarray(A_arr)
        self.b = jnp.asarray(b_arr)
        self.weight = jnp.asarray(np.broadcast_to(w_arr, b_arr.shape))

    def violation(self, x: jax.Array, u: jax.Array | None) -> jax.Array:
        """Return positive affine residuals over any leading batch dimensions."""
        if self.m and u is None:
            msg = "The Stage Cost requires a control vector."
            raise ValueError(msg)
        z = x if self.terminal or u is None or not self.m else jnp.concatenate([x, u], axis=-1)
        return jax.nn.relu(z @ self.A.T - self.b)

    def evaluate(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate the weighted squared positive part at one Knot Point."""
        del t
        return jnp.sum(self.weight * self.violation(x, u) ** 2)

    def as_terminal(self) -> "OneSidedQuadratic":
        """Use the state columns of the affine rows for the derived Terminal Cost."""
        if self.terminal:
            return self
        return OneSidedQuadratic(
            self.A[:, : self.n], self.b, self.weight, n=self.n, terminal=True, formulation=self.formulation
        )

    def reformulate(self) -> CostReformulation:
        """Supply affine slack constraints and a quadratic replacement cost."""
        if self.formulation == "direct":
            return CostReformulation(violation=self.violation, requires_iterative=True)
        p, width = self.A.shape
        eye = -jnp.eye(p)
        A = jnp.block([[self.A, eye], [jnp.zeros((p, width)), eye]])
        b = jnp.pad(self.b, (0, p))
        if self.terminal:
            cost = DiagonalCost(Q=jnp.pad(2 * self.weight, (self.n, 0)), terminal=True)
            constraint = LinearConstraint(self.n + p, 0, A, b)
        else:
            cost = DiagonalCost(Q=jnp.zeros(self.n), R=jnp.pad(2 * self.weight, (self.m, 0)))
            constraint = LinearConstraint(self.n, self.m + p, A, b)
        return CostReformulation(cost, (constraint,), p, self.violation, self.violation)
