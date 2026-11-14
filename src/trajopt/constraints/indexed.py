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
    _state_identity: bool = eqx.field(static=True)
    _control_identity: bool = eqx.field(static=True)
    _structural_key: tuple[object, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        m: int,
        constraint: Constraint,
        ix: Sequence[int] | None = None,
        iu: Sequence[int] | None = None,
    ) -> None:
        """Bind the wrapped constraint and identify full coordinate maps."""
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

        inner_key = constraint.structural_key()
        super().__init__(n=n_int, m=m_int, p=constraint.p, cone=constraint.cone)
        self.constraint = constraint
        self.ix = ix_tuple
        self.iu = iu_tuple
        self._state_identity = ix_tuple == tuple(range(n_int))
        self._control_identity = iu_tuple == tuple(range(m_int))
        self.ix_arr = jnp.asarray(ix_tuple, dtype=int)
        self.iu_arr = jnp.asarray(iu_tuple, dtype=int)
        self._structural_key = (
            "IndexedConstraint",
            n_int,
            m_int,
            constraint.p,
            iu_tuple,
            inner_key,
        )

    def structural_key(self) -> tuple[object, ...]:
        """Return static structural identity key."""
        return self._structural_key

    def uses_state(self) -> bool:
        """Whether the wrapped constraint reads any of the mapped state indices."""
        return len(self.ix) > 0 and self.constraint.uses_state()

    def uses_control(self) -> bool:
        """Whether the wrapped constraint reads any of the mapped control indices."""
        return len(self.iu) > 0 and self.constraint.uses_control()

    def is_time_independent(self) -> bool:
        """Whether the wrapped constraint is invariant to timestamp t."""
        return self.constraint.is_time_independent()

    def is_time_independent_input_constraint(self) -> bool:
        """Whether the wrapped constraint is a time-independent input-only constraint."""
        return not self.uses_state() and self.constraint.is_time_independent_input_constraint()

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the wrapped constraint on mapped coordinates, passing full identity maps through."""
        x_sub = None
        if x is not None and self.ix:
            x_sub = x if self._state_identity and x.shape[0] == self.n else x[self.ix_arr]
        u_sub = None
        if u is not None and self.iu:
            u_sub = u if self._control_identity and u.shape[0] == self.m else u[self.iu_arr]
        return self.constraint.evaluate(x_sub, u_sub, t)
