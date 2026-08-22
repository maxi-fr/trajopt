from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import AbstractCone, NegativeOrthant, SecondOrderCone
from trajopt.constraints.base import StageConstraint, StateConstraint


class CircleConstraint(StateConstraint):
    """Circular obstacle keep-out constraint in 2D plane.

    Defect formula: -(x[xi] - xc)^2 - (x[yi] - yc)^2 + r^2 <= 0.

    Parameters
    ----------
    n : int
        State dimension.
    xc : jax.Array | Sequence[float]
        X-coordinates of obstacle centers (length P).
    yc : jax.Array | Sequence[float]
        Y-coordinates of obstacle centers (length P).
    radius : jax.Array | Sequence[float]
        Obstacle radii (length P).
    xi : int, optional
        State index for x-coordinate (default 0).
    yi : int, optional
        State index for y-coordinate (default 1).
    m : int, optional
        Control dimension (default 0).
    """

    xc: jax.Array
    yc: jax.Array
    radius: jax.Array
    xi: int = eqx.field(static=True)
    yi: int = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        n: int,
        xc: jax.Array | Sequence[float],
        yc: jax.Array | Sequence[float],
        radius: jax.Array | Sequence[float],
        *,
        xi: int = 0,
        yi: int = 1,
        m: int = 0,
    ) -> None:
        n_int = int(n)
        m_int = int(m)

        xc_arr = np.atleast_1d(np.asarray(xc, dtype=float))
        yc_arr = np.atleast_1d(np.asarray(yc, dtype=float))
        r_arr = np.atleast_1d(np.asarray(radius, dtype=float))

        if not (len(xc_arr) == len(yc_arr) == len(r_arr)):
            msg = f"Lengths of xc ({len(xc_arr)}), yc ({len(yc_arr)}), and radius ({len(r_arr)}) must match."
            raise ValueError(msg)

        p = len(xc_arr)
        super().__init__(n=n_int, m=m_int, p=p, cone=NegativeOrthant())
        self.xc = jnp.asarray(xc_arr)
        self.yc = jnp.asarray(yc_arr)
        self.radius = jnp.asarray(r_arr)
        self.xi = int(xi)
        self.yi = int(yi)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate circular obstacle avoidance defect."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate CircleConstraint."
            raise ValueError(msg)
        px = x[self.xi]
        py = x[self.yi]
        return -((px - self.xc) ** 2) - (py - self.yc) ** 2 + self.radius**2


class SphereConstraint(StateConstraint):
    """Spherical obstacle keep-out constraint in 3D.

    Defect formula: -(x[xi] - xc)^2 - (x[yi] - yc)^2 - (x[zi] - zc)^2 + r^2 <= 0.

    Parameters
    ----------
    n : int
        State dimension.
    xc : jax.Array | Sequence[float]
        X-coordinates of obstacle centers (length P).
    yc : jax.Array | Sequence[float]
        Y-coordinates of obstacle centers (length P).
    zc : jax.Array | Sequence[float]
        Z-coordinates of obstacle centers (length P).
    radius : jax.Array | Sequence[float]
        Obstacle radii (length P).
    xi : int, optional
        State index for x-coordinate (default 0).
    yi : int, optional
        State index for y-coordinate (default 1).
    zi : int, optional
        State index for z-coordinate (default 2).
    m : int, optional
        Control dimension (default 0).
    """

    xc: jax.Array
    yc: jax.Array
    zc: jax.Array
    radius: jax.Array
    xi: int = eqx.field(static=True)
    yi: int = eqx.field(static=True)
    zi: int = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        n: int,
        xc: jax.Array | Sequence[float],
        yc: jax.Array | Sequence[float],
        zc: jax.Array | Sequence[float],
        radius: jax.Array | Sequence[float],
        *,
        xi: int = 0,
        yi: int = 1,
        zi: int = 2,
        m: int = 0,
    ) -> None:
        n_int = int(n)
        m_int = int(m)

        xc_arr = np.atleast_1d(np.asarray(xc, dtype=float))
        yc_arr = np.atleast_1d(np.asarray(yc, dtype=float))
        zc_arr = np.atleast_1d(np.asarray(zc, dtype=float))
        r_arr = np.atleast_1d(np.asarray(radius, dtype=float))

        if not (len(xc_arr) == len(yc_arr) == len(zc_arr) == len(r_arr)):
            msg = f"Lengths of xc, yc, zc, and radius must match, got ({len(xc_arr)}, {len(yc_arr)}, {len(zc_arr)}, {len(r_arr)})."
            raise ValueError(msg)

        p = len(xc_arr)
        super().__init__(n=n_int, m=m_int, p=p, cone=NegativeOrthant())
        self.xc = jnp.asarray(xc_arr)
        self.yc = jnp.asarray(yc_arr)
        self.zc = jnp.asarray(zc_arr)
        self.radius = jnp.asarray(r_arr)
        self.xi = int(xi)
        self.yi = int(yi)
        self.zi = int(zi)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate spherical obstacle avoidance defect."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate SphereConstraint."
            raise ValueError(msg)
        px = x[self.xi]
        py = x[self.yi]
        pz = x[self.zi]
        return -((px - self.xc) ** 2) - (py - self.yc) ** 2 - (pz - self.zc) ** 2 + self.radius**2


class CollisionConstraint(StateConstraint):
    """Pairwise non-self-collision constraint between two bodies.

    Defect formula: r^2 - ||x[x1] - x[x2]||_2^2 <= 0.

    Parameters
    ----------
    n : int
        State dimension.
    x1 : Sequence[int]
        Position coordinate indices for body 1.
    x2 : Sequence[int]
        Position coordinate indices for body 2.
    radius : float
        Collision radius.
    m : int, optional
        Control dimension (default 0).
    """

    radius: float = eqx.field(static=True)
    x1_arr: jax.Array
    x2_arr: jax.Array
    x1: tuple[int, ...] = eqx.field(static=True)
    x2: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        x1: Sequence[int],
        x2: Sequence[int],
        radius: float,
        *,
        m: int = 0,
    ) -> None:
        n_int = int(n)
        m_int = int(m)
        x1_tuple = tuple(int(i) for i in x1)
        x2_tuple = tuple(int(i) for i in x2)

        if len(x1_tuple) != len(x2_tuple):
            msg = f"Position index dimensions must match, got {len(x1_tuple)} and {len(x2_tuple)}."
            raise ValueError(msg)

        super().__init__(n=n_int, m=m_int, p=1, cone=NegativeOrthant())
        self.radius = float(radius)
        self.x1 = x1_tuple
        self.x2 = x2_tuple
        self.x1_arr = jnp.asarray(x1_tuple, dtype=int)
        self.x2_arr = jnp.asarray(x2_tuple, dtype=int)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate collision defect r^2 - ||x[x1] - x[x2]||^2."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate CollisionConstraint."
            raise ValueError(msg)
        pos1 = x[self.x1_arr]
        pos2 = x[self.x2_arr]
        d = pos1 - pos2
        return jnp.array([self.radius**2 - jnp.dot(d, d)])


class NormConstraint(StageConstraint):
    """Norm constraint ||y||_2 <= a, in quadratic inequality/equality or second-order cone forms.

    Formulas:
    - Quadratic inequality: ||y||_2^2 - a^2 <= 0 (NegativeOrthant, p=1)
    - Quadratic equality:   ||y||_2^2 - a^2 == 0 (ZeroCone, p=1)
    - Second-order cone:    [y; a] in K_soc (SecondOrderCone, p=D+1)

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Control dimension.
    val : float
        Bound constant a >= 0.
    sense : AbstractCone | None, optional
        Conic sense: NegativeOrthant(), ZeroCone(), or SecondOrderCone(). Defaults to NegativeOrthant().
    inds : Sequence[int] | str | None, optional
        Indices of y in z = [x; u]. Can be "state", "control", or integer sequence. Defaults to all (0..n+m-1).
    """

    val: float = eqx.field(static=True)
    inds_arr: jax.Array
    inds: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        m: int,
        val: float,
        *,
        sense: AbstractCone | None = None,
        inds: Sequence[int] | str | None = None,
    ) -> None:
        n_int = int(n)
        m_int = int(m)
        nm = n_int + m_int

        if val < 0.0:
            msg = f"Norm bound value must be non-negative, got {val}."
            raise ValueError(msg)

        if inds == "state":
            inds_tuple = tuple(range(n_int))
        elif inds == "control":
            inds_tuple = tuple(range(n_int, nm))
        elif inds is None:
            inds_tuple = tuple(range(nm))
        else:
            inds_tuple = tuple(int(i) for i in inds)

        cone = NegativeOrthant() if sense is None else sense
        p = len(inds_tuple) + 1 if isinstance(cone, SecondOrderCone) else 1

        super().__init__(n=n_int, m=m_int, p=p, cone=cone)
        self.val = float(val)
        self.inds = inds_tuple
        self.inds_arr = jnp.asarray(inds_tuple, dtype=int)

    def uses_control(self) -> bool:
        """Whether any gathered index falls in the control block of z = [x; u]."""
        return any(i >= self.n for i in self.inds)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate norm constraint of shape (p,) from z = [x; u] of shape (n + m,)."""
        del t
        y = self.stage_vector(x, u)[self.inds_arr]

        if isinstance(self.cone, SecondOrderCone):
            return jnp.concatenate([y, jnp.array([self.val], dtype=y.dtype)])
        return jnp.array([jnp.dot(y, y) - self.val**2], dtype=y.dtype)
