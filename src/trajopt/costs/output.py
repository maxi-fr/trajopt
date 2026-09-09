import jax
import jax.numpy as jnp

from trajopt.costs.base import CostFunction, QuadraticCostFunction
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import AbstractModel
from trajopt.models.affine import AffineModel


class OutputCost(CostFunction):
    """CostFunction adapter evaluating a cost on output y = g(x, u, t)."""

    model: AbstractModel
    cost: CostFunction

    def __init__(self, model: AbstractModel, cost: CostFunction) -> None:
        if model.p is None:
            msg = "Model must define an output dimension p."
            raise ValueError(msg)
        if cost.n != model.p:
            msg = f"Cost state dimension ({cost.n}) must match model output dimension p ({model.p})."
            raise ValueError(msg)
        if cost.m not in (0, model.m):
            msg = f"Cost control dimension ({cost.m}) must be 0 or match model control dimension ({model.m})."
            raise ValueError(msg)
        super().__init__(n=model.n, m=model.m, terminal=cost.terminal)
        self.model = model
        self.cost = cost

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost composed with output function."""
        y = self.model.output(x, u, t)
        u_cost = None if (self.terminal or self.cost.m == 0) else u
        return self.cost.evaluate(y, u_cost, t)

    def as_terminal(self) -> "OutputCost":
        """Derive a terminal cost from this cost."""
        return OutputCost(self.model, self.cost.as_terminal())


def pullback_output_cost(
    model: AffineModel,
    cost: QuadraticCostFunction,
) -> QuadraticCost:
    """Analytically transform quadratic cost on output y = C x + D u + dy into QuadraticCost on x and u."""
    if model.p is None or model.C is None or model.D is None or model.dy is None:
        msg = "AffineModel must define output matrices C, D, and dy."
        raise ValueError(msg)
    if cost.n != model.p:
        msg = f"Cost state dimension ({cost.n}) must match model output dimension p ({model.p})."
        raise ValueError(msg)
    if cost.m not in (0, model.m):
        msg = f"Cost control dimension ({cost.m}) must be 0 or match model control dimension ({model.m})."
        raise ValueError(msg)

    quad = cost.to_quadratic()
    C = model.C
    D = model.D
    dy = model.dy
    Q = quad.Q
    q = quad.q
    c = quad.c

    Q_new = jnp.matmul(C.T, jnp.matmul(Q, C))
    Q_dy = jnp.matmul(Q, dy)
    v = Q_dy + q
    q_new = jnp.matmul(v, C)
    c_new = c + jnp.sum(q * dy, axis=-1) + 0.5 * jnp.sum(dy * Q_dy, axis=-1)

    if quad.terminal:
        return QuadraticCost(
            Q=Q_new,
            q=q_new,
            c=c_new,
            terminal=True,
            m=model.m,
        )

    if quad.m == model.m:
        R_y = quad.R
        H_y = jnp.zeros((model.m, model.p), dtype=Q.dtype) if quad.H is None else quad.H
        r_y = quad.r
    else:
        R_y = jnp.zeros((model.m, model.m), dtype=Q.dtype)
        H_y = jnp.zeros((model.m, model.p), dtype=Q.dtype)
        r_y = jnp.zeros(model.m, dtype=Q.dtype)

    M = jnp.matmul(H_y, D)
    M_sym = M + jnp.swapaxes(M, -1, -2)
    R_new = jnp.matmul(D.T, jnp.matmul(Q, D)) + R_y + M_sym

    H_new = jnp.matmul(D.T, jnp.matmul(Q, C)) + jnp.matmul(H_y, C)
    r_new = jnp.matmul(v, D) + r_y + jnp.matmul(H_y, dy)

    return QuadraticCost(
        Q=Q_new,
        R=R_new,
        H=H_new,
        q=q_new,
        r=r_new,
        c=c_new,
        terminal=False,
    )
