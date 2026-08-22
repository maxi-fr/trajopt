from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.base import Constraint, StageConstraint


class IndexedConstraint(StageConstraint):
    """Wraps a subsystem constraint onto specified indices of a larger state/control space.

    Parameters
    ----------
    n : int
        Full state dimension.
    m : int
        Full control dimension.
    constraint : Constraint
        Inner subsystem constraint.
    ix : Sequence[int] | None, optional
        State indices mapping to inner constraint state. Defaults to (0..constraint.n-1).
    iu : Sequence[int] | None, optional
        Control indices mapping to inner constraint control. Defaults to (0..constraint.m-1).
    """

    constraint: Constraint
    ix_arr: jax.Array
    iu_arr: jax.Array
    ix: tuple[int, ...] = eqx.field(static=True)
    iu: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        m: int,
        constraint: Constraint,
        ix: Sequence[int] | None = None,
        iu: Sequence[int] | None = None,
    ) -> None:
        n_int = int(n)
        m_int = int(m)

        ix_tuple = tuple(range(constraint.n)) if ix is None else tuple(int(i) for i in ix)
        iu_tuple = tuple(range(constraint.m)) if iu is None else tuple(int(i) for i in iu)

        if len(ix_tuple) != constraint.n:
            msg = f"State index length ({len(ix_tuple)}) does not match inner constraint n ({constraint.n})."
            raise ValueError(msg)
        if len(iu_tuple) != constraint.m:
            msg = f"Control index length ({len(iu_tuple)}) does not match inner constraint m ({constraint.m})."
            raise ValueError(msg)

        super().__init__(n=n_int, m=m_int, p=constraint.p, cone=constraint.cone)
        self.constraint = constraint
        self.ix = ix_tuple
        self.iu = iu_tuple
        self.ix_arr = jnp.asarray(ix_tuple, dtype=int)
        self.iu_arr = jnp.asarray(iu_tuple, dtype=int)

    def uses_control(self) -> bool:
        """Whether the wrapped constraint reads any of the mapped control indices."""
        return len(self.iu) > 0 and self.constraint.uses_control()

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the wrapped constraint of shape (p,) on x[ix] and u[iu]."""
        x_sub = x[self.ix_arr] if (x is not None and len(self.ix) > 0) else None
        u_sub = u[self.iu_arr] if (u is not None and len(self.iu) > 0) else None
        return self.constraint.evaluate(x_sub, u_sub, t)
