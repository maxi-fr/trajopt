"""Base classes for constraints in trajectory optimization."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.cones import AbstractCone, ZeroCone


class Constraint(eqx.Module):
    """Abstract base class for trajectory optimization constraints.

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Control dimension.
    p : int
        Constraint vector output dimension.
    cone : AbstractCone | None, optional
        Conic set into which the constraint maps: c(x, u) in K (default ZeroCone()).
    """

    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    cone: AbstractCone = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        m: int = 0,
        p: int = 0,
        cone: AbstractCone | None = None,
    ) -> None:
        self.n = int(n)
        self.m = int(m)
        self.p = int(p)
        self.cone = ZeroCone() if cone is None else cone

    @abstractmethod
    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate constraint vector c(x, u, t) of shape (p,).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Constraint vector of shape (p,).
        """

    def __call__(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate constraint as a callable.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Constraint vector of shape (p,).
        """
        return self.evaluate(x, u, t)

    def jacobian_x(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state Jacobian dc/dx of shape (p, n).

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            State Jacobian of shape (p, n).
        """
        return jax.jacobian(lambda x_: self.evaluate(x_, u, t))(x)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control Jacobian dc/du of shape (p, m).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Control Jacobian of shape (p, m).
        """
        if u is None:
            dtype = x.dtype if x is not None else jnp.float64
            return jnp.zeros((self.p, self.m), dtype=dtype)
        return jax.jacobian(lambda u_: self.evaluate(x, u_, t))(u)

    def jacobian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate both Jacobian blocks (dc/dx, dc/du) of shapes (p, n) and (p, m).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,).
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            (dc/dx, dc/du) of shapes (p, n) and (p, m).
        """
        if x is None:
            dtype = u.dtype if u is not None else jnp.float64
            jx = jnp.zeros((self.p, self.n), dtype=dtype)
        else:
            jx = self.jacobian_x(x, u, t)
        ju = self.jacobian_u(x, u, t)
        return jx, ju

    def joint_jacobian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate concatenated Jacobian [dc/dx, dc/du] of shape (p, n + m).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Concatenated Jacobian matrix of shape (p, n + m).
        """
        jx, ju = self.jacobian(x, u, t)
        return jnp.hstack([jx, ju])


class StateConstraint(Constraint):
    """Constraint that depends only on state x: c(x, t) in K.

    Control Jacobian block is implied zeros rather than stored or computed via AD.
    """

    def jacobian_x(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state Jacobian dc/dx of shape (p, n) via AD.

        Parameters
        ----------
        x : jax.Array
            State vector.
        u : jax.Array | None, optional
            Ignored for StateConstraint.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            State Jacobian of shape (p, n).
        """
        del u
        return jax.jacobian(lambda x_: self.evaluate(x_, None, t))(x)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Implied zero block of shape (p, m).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Zero matrix of shape (p, m).
        """
        del u, t
        dtype = x.dtype if x is not None else jnp.float64
        return jnp.zeros((self.p, self.m), dtype=dtype)

    def jacobian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate Jacobian blocks: (dc/dx, 0).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            (dc/dx, 0) of shapes (p, n) and (p, m).
        """
        if x is None:
            dtype = u.dtype if u is not None else jnp.float64
            jx = jnp.zeros((self.p, self.n), dtype=dtype)
        else:
            jx = self.jacobian_x(x, None, t)
        ju = self.jacobian_u(x, u, t)
        return jx, ju


class ControlConstraint(Constraint):
    """Constraint that depends only on control u: c(u, t) in K.

    State Jacobian block is implied zeros rather than stored or computed via AD.
    """

    def jacobian_x(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Implied zero block of shape (p, n).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Zero matrix of shape (p, n).
        """
        del t
        dtype = x.dtype if x is not None else (u.dtype if u is not None else jnp.float64)
        return jnp.zeros((self.p, self.n), dtype=dtype)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control Jacobian dc/du of shape (p, m) via AD.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Control Jacobian of shape (p, m).
        """
        del x
        if u is None:
            return jnp.zeros((self.p, self.m))
        return jax.jacobian(lambda u_: self.evaluate(None, u_, t))(u)

    def jacobian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate Jacobian blocks: (0, dc/du).

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            (0, dc/du) of shapes (p, n) and (p, m).
        """
        jx = self.jacobian_x(x, u, t)
        ju = self.jacobian_u(None, u, t)
        return jx, ju


class StageConstraint(Constraint):
    """Constraint that depends on both state x and control u: c(x, u, t) in K."""

    def jacobian_x(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state Jacobian dc/dx via AD.

        Parameters
        ----------
        x : jax.Array
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            State Jacobian matrix of shape (p, n).
        """
        return jax.jacobian(lambda x_: self.evaluate(x_, u, t))(x)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control Jacobian dc/du via AD.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Control Jacobian matrix of shape (p, m).
        """
        if u is None:
            dtype = x.dtype if x is not None else jnp.float64
            return jnp.zeros((self.p, self.m), dtype=dtype)
        return jax.jacobian(lambda u_: self.evaluate(x, u_, t))(u)
