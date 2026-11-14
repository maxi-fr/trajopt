from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax

if TYPE_CHECKING:
    from collections.abc import Callable

    from trajopt.constraints.base import Constraint
    from trajopt.costs.base import CostFunction


@dataclass(frozen=True)
class CostReformulation:
    """Solver representation of a Knot Point cost.

    The replacement cost and constraints use x and [u, auxiliary] at control
    Knot Points, and [x, auxiliary] with u=None at the terminal Knot Point.
    An absent replacement keeps the original cost and requires zero auxiliaries.
    The Program applies one representation per term across its selected Knot
    Points. Return a replacement ready for evaluation by the selected solver.
    Optional callbacks accept batched or unbatched physical x and u (None at the
    terminal Knot Point) and return an auxiliary seed or diagnostic residual.
    """

    cost: "CostFunction | None" = None
    constraints: "tuple[Constraint, ...]" = ()
    auxiliary_dim: int = 0
    initial_auxiliary: "Callable[[jax.Array, jax.Array | None], jax.Array] | None" = None
    violation: "Callable[[jax.Array, jax.Array | None], jax.Array] | None" = None
    requires_iterative: bool = False

    def __post_init__(self) -> None:
        """Validate the auxiliary layout described by the cost."""
        if self.auxiliary_dim < 0 or (self.auxiliary_dim and self.cost is None):
            msg = "Auxiliary dimensions must be nonnegative and require a replacement cost."
            raise ValueError(msg)


class PenaltyDiagnostics(NamedTuple):
    """Evaluated violations and solved slacks for one reformulated cost term."""

    stage_violation: jax.Array | None
    terminal_violation: jax.Array | None
    stage_slack: jax.Array | None
    terminal_slack: jax.Array | None
