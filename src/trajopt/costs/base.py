from abc import abstractmethod
from typing import TYPE_CHECKING

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

    @property
    def is_diag(self) -> bool:
        """Whether the Hessian is strictly diagonal."""
        return False

    @property
    def is_blockdiag(self) -> bool:
        """Whether the Hessian is block diagonal (H = 0)."""
        return False

    @property
    def is_stacked(self) -> bool:
        """Whether the parameters carry a leading horizon axis."""
        return False

    def stacked(self, N: int) -> "CostFunction":
        """Stage cost with parameters repeated over N - 1 stages; a no-op without parameters."""
        del N
        return self

    def as_terminal(self) -> "CostFunction":
        """Terminal cost derived from this cost; a parameterless cost is its own terminal cost."""
        return self

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the cost at each stage of X (N-1, n), U (N-1, m), t (N-1,), giving (N-1,)."""
        return jax.vmap(self.evaluate)(X, U, t)

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

    Weight parameters are either single-knot or stacked over the horizon along a leading axis.
    """

    if TYPE_CHECKING:
        # Declared for the type checker only: at runtime these are dataclass fields on the
        # concrete subclasses, which a base class attribute would shadow. The constructor covers
        # both what the subclasses accept and what super().__init__ is given.
        Q: jax.Array
        R: jax.Array
        H: jax.Array | None
        q: jax.Array
        r: jax.Array
        c: jax.Array

        def __init__(  # noqa: PLR0913
            self,
            Q: jax.Array | None = None,
            R: jax.Array | None = None,
            q: jax.Array | None = None,
            r: jax.Array | None = None,
            c: float | jax.Array = 0.0,
            *,
            n: int | None = None,
            m: int | None = None,
            terminal: bool = False,
        ) -> None: ...

    @abstractmethod
    def to_quadratic(self) -> "QuadraticCostFunction":
        """Re-express this cost with dense weight matrices."""

    @staticmethod
    @abstractmethod
    def matvec(W: jax.Array, v: jax.Array) -> jax.Array:
        """Apply weights W to vectors v, broadcasting over any leading stacked axes.

        Parameters
        ----------
        W : jax.Array
            Weights of shape (..., n) if diagonal or (..., m, n) if dense.
        v : jax.Array
            Vectors of shape (..., n).

        Returns
        -------
        jax.Array
            Products of shape (..., n) if diagonal or (..., m) if dense.
        """

    @classmethod
    def quad_form(cls, W: jax.Array, v: jax.Array) -> jax.Array:
        """Quadratic form v^T W v for weights W and vectors v of shape (..., n), giving (...,)."""
        return jnp.sum(v * cls.matvec(W, v), axis=-1)

    @classmethod
    def tracking(cls, Q: jax.Array, R: jax.Array, X: jax.Array, U: jax.Array) -> "QuadraticCostFunction":
        """Build a stage cost penalizing deviation from a reference trajectory.

        Parameters
        ----------
        Q : jax.Array
            Stacked state weights of shape (N-1, n) if diagonal or (N-1, n, n) if dense.
        R : jax.Array
            Stacked control weights of shape (N-1, m) if diagonal or (N-1, m, m) if dense.
        X : jax.Array
            Reference states of shape (N-1, n), or (n,) to hold one state over the horizon.
        U : jax.Array
            Reference controls of shape (N-1, m), or (m,) to hold one control over the horizon.

        Returns
        -------
        QuadraticCostFunction
            Stage cost with stacked linear terms q, r of shapes (N-1, n), (N-1, m) and constant
            terms c of shape (N-1,).
        """
        return cls(
            Q=Q,
            R=R,
            q=-cls.matvec(Q, X),
            r=-cls.matvec(R, U),
            c=0.5 * cls.quad_form(Q, X) + 0.5 * cls.quad_form(R, U),
        )

    @classmethod
    def terminal_tracking(cls, Q_f: jax.Array, x: jax.Array, m: int) -> "QuadraticCostFunction":
        """Build a terminal cost penalizing deviation from reference state x of shape (n,).

        Parameters
        ----------
        Q_f : jax.Array
            Terminal state weights of shape (n,) if diagonal or (n, n) if dense.
        x : jax.Array
            Reference terminal state of shape (n,).
        m : int
            Control dimension carried by the terminal cost.
        """
        return cls(
            Q=Q_f,
            q=-cls.matvec(Q_f, x),
            c=0.5 * cls.quad_form(Q_f, x),
            terminal=True,
            m=m,
        )

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the stacked parameters against X (N-1, n), U (N-1, m), t (N-1,), giving (N-1,)."""
        del t
        return (
            0.5 * self.quad_form(self.Q, X)
            + 0.5 * self.quad_form(self.R, U)
            + jnp.sum(self.q * X, axis=-1)
            + jnp.sum(self.r * U, axis=-1)
            + self.c
        )

    def stacked(self, N: int) -> "QuadraticCostFunction":
        """Repeat every parameter over N - 1 stages, adding a leading axis of that length."""
        return jax.tree.map(lambda leaf: jnp.repeat(leaf[None], N - 1, axis=0), self)

    def as_terminal(self) -> "QuadraticCostFunction":
        """Terminal cost built from this cost's state parameters Q, q, c."""
        return type(self)(Q=self.Q, q=self.q, c=self.c, terminal=True, m=self.m)
