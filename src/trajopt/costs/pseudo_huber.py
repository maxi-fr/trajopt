import jax
import jax.numpy as jnp
import numpy as np

from trajopt.costs.base import CostFunction
from trajopt.costs.quadratic import DiagonalCost


class PseudoHuberControlCost(CostFunction):
    """Smooth L1 stage cost on controls with the same outer slope as an L1 penalty.

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Control dimension.
    weight : float
        Nonnegative multiplier of the summed control penalty.
    delta : float
        Positive smoothing radius; the penalty approaches L1 as delta tends to zero.
    """

    weight: jax.Array
    delta: jax.Array

    def __init__(self, n: int, m: int, weight: float, delta: float) -> None:
        """Store a nonnegative weight and positive smoothing radius."""
        if not np.isfinite(weight) or weight < 0 or not np.isfinite(delta) or delta <= 0:
            msg = "weight must be finite and nonnegative, and delta must be finite and positive."
            raise ValueError(msg)
        super().__init__(n=n, m=m)
        self.weight = jnp.asarray(weight)
        self.delta = jnp.asarray(delta)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate weight times the smooth absolute values of control u of shape (m,)."""
        del x, t
        if u is None:
            msg = "A control vector is required for PseudoHuberControlCost."
            raise ValueError(msg)
        return self.weight * jnp.sum(jnp.hypot(u, self.delta) - self.delta)

    def as_terminal(self) -> CostFunction:
        """Return a zero terminal cost on the state of shape (n,)."""
        return DiagonalCost(Q=jnp.zeros(self.n), terminal=True, m=self.m)
