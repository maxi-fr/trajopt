import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.base import CostFunction
from trajopt.costs.quadratic import DiagonalCost
from trajopt.costs.reformulation import CostReformulation


class L1ControlCost(CostFunction):
    """Exact weighted sum of absolute control values with an epigraph representation."""

    weight: jax.Array

    def __init__(self, n: int, m: int, weight: float) -> None:
        """Store a finite positive scalar weight for controls of shape (m,)."""
        if not isinstance(weight, (int, float, np.floating)) or not np.isfinite(weight) or weight <= 0:
            msg = "weight must be a finite positive scalar."
            raise ValueError(msg)
        super().__init__(n, m)
        self.weight = jnp.asarray(weight)

    def evaluate(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate the weighted L1 norm at one Knot Point."""
        del t
        return self.weight * jnp.sum(self._absolute_controls(x, u))

    def as_terminal(self) -> CostFunction:
        """Return a zero Terminal Cost on states of shape (n,)."""
        return DiagonalCost(Q=jnp.zeros(self.n), terminal=True, m=self.m)

    def reformulate(self) -> CostReformulation:
        """Price slacks linearly and bound them below by both signs of the controls."""
        eye = jnp.eye(self.m)
        A = jnp.block([[eye, -eye], [-eye, -eye]])
        rows = LinearConstraint(self.n, 2 * self.m, A, jnp.zeros(2 * self.m), inds=range(self.n, self.n + 2 * self.m))
        cost = DiagonalCost(
            Q=jnp.zeros(self.n), R=jnp.zeros(2 * self.m), r=jnp.pad(jnp.full(self.m, self.weight), (self.m, 0))
        )
        return CostReformulation(
            cost, (rows,), self.m, self._absolute_controls, self._absolute_controls, supports_control_blocking=True
        )

    def _absolute_controls(self, x: jax.Array, u: jax.Array | None) -> jax.Array:
        """Compute absolute controls at one Knot Point or across the Horizon."""
        del x
        if u is None:
            msg = "A control vector is required for L1ControlCost."
            raise ValueError(msg)
        return jnp.abs(u)
