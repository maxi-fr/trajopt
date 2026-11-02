import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.base import Constraint
from trajopt.dynamics.base import AbstractModel


class OutputConstraint(Constraint):
    """Constraint on output space y = g(x, u, t) adapted to state-control space."""

    model: AbstractModel
    constraint: Constraint
    _feedthrough: bool = eqx.field(static=True)
    _uses_state: bool = eqx.field(static=True)
    _structural_key: tuple[object, ...] = eqx.field(static=True)

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
        self._uses_state = constraint.uses_state() and model.output_uses_state()

        from trajopt.models.affine import AffineModel  # noqa: PLC0415 -- local import
        inner_key = constraint.structural_key()
        if isinstance(model, AffineModel):
            import numpy as np  # noqa: PLC0415 -- local import
            d_bytes = np.asarray(model.D).tobytes() if model.D is not None else b""
            dy_bytes = np.asarray(model.dy).tobytes() if model.dy is not None else b""
            self._structural_key = (
                "OutputConstraint",
                model.n,
                model.m,
                constraint.p,
                inner_key,
                d_bytes,
                dy_bytes,
            )
        else:
            self._structural_key = ("OutputConstraint", id(self))

    def structural_key(self) -> tuple[object, ...]:
        """Return static structural identity key."""
        return self._structural_key

    def uses_state(self) -> bool:
        """Whether constraint reads state directly or through model output."""
        return self._uses_state

    def uses_control(self) -> bool:
        """Whether constraint reads control directly or through model output feedthrough."""
        return self.constraint.uses_control() or self._feedthrough

    def is_time_independent(self) -> bool:
        """Whether output constraint is invariant to timestamp t."""
        model_time_indep = getattr(self.model, "output_is_time_independent", lambda: False)()
        return self.constraint.is_time_independent() and model_time_indep

    def is_time_independent_input_constraint(self) -> bool:
        """Whether constraint is a time-independent pure input constraint."""
        return not self.uses_state() and self.uses_control() and self.is_time_independent()

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate wrapped constraint of shape (p,) on model output y = g(x, u, t)."""
        if x is None:
            if not self.uses_state():
                x = jnp.zeros(self.n, dtype=u.dtype if u is not None else jnp.float64)
            else:
                msg = f"State vector x is required to evaluate {type(self).__name__}."
                raise ValueError(msg)
        y = self.model.output(x, u, t)
        u_inner = None if not self.constraint.uses_control() else u
        x_inner = y if self.constraint.uses_state() else None
        return self.constraint.evaluate(x_inner, u_inner, t)
