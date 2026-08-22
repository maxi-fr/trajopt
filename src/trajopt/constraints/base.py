from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.cones import AbstractCone, ZeroCone


class ConstraintShape(eqx.Module):
    """Dimensions and target cone shared by every constraint kind.

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


class Constraint(ConstraintShape):
    """Abstract base class for stage constraints c(x, u, t) in K."""

    def uses_state(self) -> bool:
        """Whether c reads x; a False state Jacobian block is implied zeros."""
        return True

    def uses_control(self) -> bool:
        """Whether c reads u; a False control Jacobian block is implied zeros.

        Also decides whether the constraint may be registered at the terminal knot
        point, where no control exists.
        """
        return True

    def stage_vector(self, x: jax.Array | None, u: jax.Array | None) -> jax.Array:
        """Assemble the gather source z = [x; u] of shape (n + m,), or x of shape (n,) alone.

        Constraints that index into z must go through this so that a constraint reaching
        into the control block can never silently gather out of range when u is absent.
        """
        if x is None:
            msg = f"State vector x is required to evaluate {type(self).__name__}."
            raise ValueError(msg)
        if u is not None:
            return jnp.concatenate([x, u])
        if self.uses_control():
            msg = f"{type(self).__name__} indexes the control block of z = [x; u], so u is required."
            raise ValueError(msg)
        return x

    @abstractmethod
    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate constraint vector c(x, u, t) of shape (p,) from x of shape (n,) and u of shape (m,)."""

    def __call__(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate constraint vector of shape (p,) as a callable."""
        return self.evaluate(x, u, t)

    def jacobian_x(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state Jacobian dc/dx of shape (p, n) via AD."""
        return jax.jacobian(lambda x_: self.evaluate(x_, u, t))(x)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control Jacobian dc/du of shape (p, m) via AD, or implied zeros if u is None."""
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
        """Evaluate both Jacobian blocks (dc/dx, dc/du) of shapes (p, n) and (p, m)."""
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
        """Evaluate concatenated Jacobian [dc/dx, dc/du] of shape (p, n + m)."""
        jx, ju = self.jacobian(x, u, t)
        return jnp.hstack([jx, ju])


class StateConstraint(Constraint):
    """Constraint that depends only on state x: c(x, t) in K.

    Control Jacobian block is implied zeros rather than stored or computed via AD.
    """

    def uses_control(self) -> bool:
        """Never reads u."""
        return False

    def jacobian_x(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate state Jacobian dc/dx of shape (p, n) via AD."""
        del u
        return jax.jacobian(lambda x_: self.evaluate(x_, None, t))(x)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Implied zero block of shape (p, m)."""
        del u, t
        dtype = x.dtype if x is not None else jnp.float64
        return jnp.zeros((self.p, self.m), dtype=dtype)

    def jacobian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate Jacobian blocks (dc/dx, 0) of shapes (p, n) and (p, m)."""
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

    def uses_state(self) -> bool:
        """Never reads x."""
        return False

    def jacobian_x(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Implied zero block of shape (p, n)."""
        del t
        dtype = x.dtype if x is not None else (u.dtype if u is not None else jnp.float64)
        return jnp.zeros((self.p, self.n), dtype=dtype)

    def jacobian_u(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate control Jacobian dc/du of shape (p, m) via AD."""
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
        """Evaluate Jacobian blocks (0, dc/du) of shapes (p, n) and (p, m)."""
        jx = self.jacobian_x(x, u, t)
        ju = self.jacobian_u(None, u, t)
        return jx, ju


class StageConstraint(Constraint):
    """Constraint that depends on both state x and control u: c(x, u, t) in K."""
