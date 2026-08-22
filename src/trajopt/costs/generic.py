import inspect
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.costs.base import CostFunction


class GenericCost(CostFunction):
    """User-supplied generic cost function differentiated automatically via JAX.

    Parameters
    ----------
    cost_fn : Callable
        Callable evaluating the scalar cost. For stage costs: (x, u, t) or (x, u).
        For terminal costs: (x, t) or (x).
    n : int
        State dimension.
    m : int, optional
        Control dimension. Default is 0.
    terminal : bool, optional
        Whether this is a terminal cost. Default is False.
    """

    cost_fn: Callable = eqx.field(static=True)
    has_t: bool = eqx.field(static=True)

    def __init__(
        self,
        cost_fn: Callable,
        n: int,
        m: int = 0,
        *,
        terminal: bool = False,
    ) -> None:
        super().__init__(n=n, m=m, terminal=terminal)
        self.cost_fn = cost_fn

        # Inspect callable arity at initialization time
        try:
            sig = inspect.signature(cost_fn)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            expected_args_with_t = 2 if terminal else 3
            self.has_t = len(params) >= expected_args_with_t or any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            self.has_t = True

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the user-supplied cost function.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal cost.
        t : float | jax.Array, optional
            Timestamp. Default is 0.0.

        Returns
        -------
        jax.Array
            Scalar cost value.
        """
        x_arr = jnp.asarray(x)
        if self.terminal or u is None:
            res = self.cost_fn(x_arr, t) if self.has_t else self.cost_fn(x_arr)
            return jnp.asarray(res, dtype=x_arr.dtype)

        u_arr = jnp.asarray(u)
        res = self.cost_fn(x_arr, u_arr, t) if self.has_t else self.cost_fn(x_arr, u_arr)
        return jnp.asarray(res, dtype=x_arr.dtype)
