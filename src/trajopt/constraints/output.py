import equinox as eqx
import jax

from trajopt.constraints.base import Constraint
from trajopt.dynamics.base import AbstractModel


class OutputConstraint(Constraint):
    """Constraint on output space y = g(x, u, t) adapted to state-control space."""

    model: AbstractModel
    constraint: Constraint
    _feedthrough: bool = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractModel,
        constraint: Constraint,
        *,
        has_feedthrough: bool | None = None,
    ) -> None:
        if model.p is None:
            msg = "Model must define an output dimension p."
            raise ValueError(msg)
        if constraint.n != model.p:
            msg = f"Inner constraint n ({constraint.n}) does not match model output dimension p ({model.p})."
            raise ValueError(msg)
        if constraint.uses_control() and constraint.m != model.m:
            msg = (
                f"Inner constraint m ({constraint.m}) must match model control dimension "
                f"m ({model.m}) when reading control."
            )
            raise ValueError(msg)
        super().__init__(
            n=model.n,
            m=model.m,
            p=constraint.p,
            cone=constraint.cone,
        )
        self.model = model
        self.constraint = constraint
        self._feedthrough = has_feedthrough if has_feedthrough is not None else model.has_control_feedthrough()

    def uses_control(self) -> bool:
        """Whether constraint reads control directly or through model output feedthrough."""
        return self.constraint.uses_control() or self._feedthrough

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the wrapped constraint of shape (p,) on output y = g(x, u, t)."""
        if x is None:
            msg = f"State vector x is required to evaluate {type(self).__name__}."
            raise ValueError(msg)
        y = self.model.output(x, u, t)
        u_inner = None if not self.constraint.uses_control() else u
        return self.constraint.evaluate(y, u_inner, t)
