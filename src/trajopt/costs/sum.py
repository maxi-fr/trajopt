import jax

from trajopt.costs.base import CostFunction, QuadraticCostFunction


class SumCost(CostFunction):
    """Add cost functions evaluated at the same Knot Point."""

    terms: tuple[CostFunction, ...]

    def __init__(self, first: CostFunction, *others: CostFunction) -> None:
        """Flatten sums and validate their state and control dimensions."""
        self.terms = tuple(
            term for cost in (first, *others) for term in (cost.terms if isinstance(cost, SumCost) else (cost,))
        )
        terminal = all(term.terminal for term in self.terms)
        m = max(term.m for term in self.terms)
        if any(term.n != first.n or (not term.terminal and term.m != m) for term in self.terms):
            msg = "Costs in a sum must share state and control dimensions."
            raise ValueError(msg)
        super().__init__(first.n, m, terminal=terminal)

    def evaluate(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Sum the scalar costs at one Knot Point."""
        return sum((term.evaluate(x, u, t) for term in self.terms[1:]), self.terms[0].evaluate(x, u, t))

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Sum batched contributions, including pre-stacked cost parameters."""
        return sum((term.stage_costs(X, U, t) for term in self.terms[1:]), self.terms[0].stage_costs(X, U, t))

    def stacked(self, N: int) -> "SumCost":
        """Stack each term once and check existing Horizon dimensions."""
        terms = []
        for term in self.terms:
            if isinstance(term, QuadraticCostFunction) and term.is_stacked and len(term.Q) != N - 1:
                msg = "Stacked cost length does not match the Objective Horizon."
                raise ValueError(msg)
            terms.append(term if term.is_stacked else term.stacked(N))
        return SumCost(*terms)

    def unstacked(self, k: int | jax.Array) -> "SumCost":
        """Select the parameters at Knot Point k from every term."""
        return SumCost(*(term.unstacked(k) for term in self.terms))

    def as_terminal(self) -> "SumCost":
        """Derive and sum the terminal contributions from each term."""
        return SumCost(*(term.as_terminal() for term in self.terms))


def cost_terms(cost: CostFunction) -> tuple[CostFunction, ...]:
    """Return the terms of a sum or the single supplied cost."""
    return cost.terms if isinstance(cost, SumCost) else (cost,)
