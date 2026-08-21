"""Base classes for cost functions in trajectory optimization."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp


class CostFunction(eqx.Module):
    """Abstract base class for cost functions.

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Control dimension.
    terminal : bool, optional
        Whether this is a terminal cost (depends only on state x, no control u). Default is False.
    """

    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    terminal: bool = eqx.field(static=True)

    def __init__(self, n: int, m: int = 0, *, terminal: bool = False) -> None:
        self.n = int(n)
        self.m = int(m)
        self.terminal = bool(terminal)

    @abstractmethod
    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate scalar cost l(x, u, t) or l(x, t) for terminal cost.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Current time. Default is 0.0.

        Returns
        -------
        jax.Array
            Scalar cost value.
        """

    def __call__(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost as a callable.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Current time. Default is 0.0.

        Returns
        -------
        jax.Array
            Scalar cost value.
        """
        return self.evaluate(x, u, t)

    def gradient(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost gradient with respect to inputs.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Current time.

        Returns
        -------
        jax.Array
            Gradient vector of shape (n + m,) for stage costs or (n,) for terminal costs.
        """
        if self.terminal or u is None:
            return jax.grad(lambda x_: self.evaluate(x_, None, t))(x)
        return jax.grad(lambda z_: self.evaluate(z_[: self.n], z_[self.n : self.n + self.m], t))(
            jnp.concatenate([x, u])
        )

    def hessian(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost Hessian with respect to inputs.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Current time.

        Returns
        -------
        jax.Array
            Hessian matrix of shape (n + m, n + m) for stage costs or (n, n) for terminal costs.
        """
        if self.terminal or u is None:
            return jax.hessian(lambda x_: self.evaluate(x_, None, t))(x)
        return jax.hessian(lambda z_: self.evaluate(z_[: self.n], z_[self.n : self.n + self.m], t))(
            jnp.concatenate([x, u])
        )

    def invert(self) -> "CostFunction":
        """Analytic inverse of the cost function parameters.

        Returns
        -------
        CostFunction
            New cost function with inverted parameters.
        """
        msg = f"{type(self).__name__} does not implement analytic parameter inversion."
        raise NotImplementedError(msg)

    def hessian_inverse(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the inverse of the Hessian matrix.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Current time.

        Returns
        -------
        jax.Array
            Inverted Hessian matrix of shape (n + m, n + m) or (n, n).
        """
        x_val = jnp.zeros(self.n) if x is None else x
        u_val = jnp.zeros(self.m) if (u is None and not self.terminal) else u
        h = self.hessian(x_val, u_val, t)
        return jnp.linalg.inv(h)


class QuadraticCostFunction(CostFunction):
    """Abstract base class for quadratic cost functions.

    Represents cost functions of the form:
    0.5 * x^T Q x + 0.5 * u^T R u + u^T H x + q^T x + r^T u + c
    """

    @property
    def is_diag(self) -> bool:
        """Whether the Hessian is strictly diagonal."""
        return False

    @property
    def is_blockdiag(self) -> bool:
        """Whether the Hessian is block diagonal (H = 0)."""
        return False
