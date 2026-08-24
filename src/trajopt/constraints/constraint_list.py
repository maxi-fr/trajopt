from collections.abc import Iterator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.base import Constraint
from trajopt.constraints.bounds import BoundConstraint, ControlBound, StateBound
from trajopt.constraints.linear import GoalConstraint

BoxBound = (StateBound, ControlBound, BoundConstraint)


class BuiltKnotConstraint(eqx.Module):
    """Fused constraint evaluator for a single knot point."""

    constraints: tuple[Constraint, ...]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    is_terminal: bool = eqx.field(static=True)

    def __init__(
        self,
        constraints: Sequence[Constraint],
        n: int,
        m: int,
        *,
        is_terminal: bool = False,
    ) -> None:
        self.constraints = tuple(constraints)
        self.n = int(n)
        self.m = int(m)
        self.is_terminal = bool(is_terminal)
        self.p = sum(c.p for c in self.constraints)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
        *,
        xf: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate the concatenated constraint vector of shape (p,) from x (n,) and u (m,)."""
        if self.p == 0:
            return jnp.zeros(0, dtype=x.dtype)
        uk = None if self.is_terminal else u
        c_vals = []
        for c in self.constraints:
            if isinstance(c, GoalConstraint) and xf is not None:
                c_vals.append(c.evaluate(x, uk, t, xf=xf))
            else:
                c_vals.append(c.evaluate(x, uk, t))
        return jnp.concatenate(c_vals)

    def jacobian(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the concatenated Jacobian blocks from x (n,) and u (m,).

        Returns
        -------
        tuple[jax.Array, jax.Array]
            (dc/dx, dc/du) of shapes (p, n) and (p, m); the control block has width 0
            at the terminal knot point, where no control exists.
        """
        dtype = x.dtype
        if self.p == 0:
            m_out = 0 if self.is_terminal else self.m
            return jnp.zeros((0, self.n), dtype=dtype), jnp.zeros((0, m_out), dtype=dtype)

        if self.is_terminal:
            jx = jnp.vstack([c.jacobian_x(x, None, t) for c in self.constraints])
            return jx, jnp.zeros((self.p, 0), dtype=dtype)

        blocks = [c.jacobian(x, u, t) for c in self.constraints]
        return jnp.vstack([jx for jx, _ in blocks]), jnp.vstack([ju for _, ju in blocks])


class ConstraintGroup(eqx.Module):
    """Knot points sharing one fused constraint structure, evaluated in a single vmap."""

    evaluator: BuiltKnotConstraint
    knots: tuple[int, ...] = eqx.field(static=True)

    def evaluate(self, X: jax.Array, U: jax.Array, T: jax.Array) -> jax.Array:
        """Evaluate this group's knot points in one batched pass.

        Parameters
        ----------
        X : jax.Array
            State trajectory of shape (N, n).
        U : jax.Array
            Control trajectory of shape (N-1, m).
        T : jax.Array
            Knot times of shape (N,).

        Returns
        -------
        jax.Array
            Constraint values of shape (len(knots), p).
        """
        ks = np.asarray(self.knots)
        evaluator = self.evaluator
        if evaluator.is_terminal:
            return jax.vmap(lambda x, t: evaluator.evaluate(x, None, t))(X[ks], T[ks])
        return jax.vmap(evaluator.evaluate)(X[ks], U[ks], T[ks])

    def jacobian(self, X: jax.Array, U: jax.Array, T: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Evaluate this group's Jacobian blocks in one batched pass.

        Parameters
        ----------
        X : jax.Array
            State trajectory of shape (N, n).
        U : jax.Array
            Control trajectory of shape (N-1, m).
        T : jax.Array
            Knot times of shape (N,).

        Returns
        -------
        tuple[jax.Array, jax.Array]
            (dc/dx, dc/du) of shapes (len(knots), p, n) and (len(knots), p, m), the control
            block having width 0 for the terminal group.
        """
        ks = np.asarray(self.knots)
        evaluator = self.evaluator
        if evaluator.is_terminal:
            return jax.vmap(lambda x, t: evaluator.jacobian(x, None, t))(X[ks], T[ks])
        return jax.vmap(evaluator.jacobian)(X[ks], U[ks], T[ks])


class BuiltConstraintList(eqx.Module):
    """Fused constraint set across all horizon knot points.

    `knot_evaluators` holds the transcribed constraint rows; box bounds are absent from it
    and carried as primal limits instead. `bound_evaluators` reconstitutes those limits as
    constraints for consumers with no native bound handling -- the augmented Lagrangian,
    where a box is enforced through the penalty rather than by the solver.
    """

    knot_evaluators: tuple[BuiltKnotConstraint, ...]
    bound_evaluators: tuple[BuiltKnotConstraint, ...]
    groups: tuple[ConstraintGroup, ...]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    p: tuple[int, ...] = eqx.field(static=True)
    x_lower: jax.Array
    x_upper: jax.Array
    u_lower: jax.Array
    u_upper: jax.Array

    def __init__(
        self,
        knot_evaluators: Sequence[BuiltKnotConstraint],
        n: int,
        m: int,
        N: int,
        bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.knot_evaluators = tuple(knot_evaluators)
        self.n = int(n)
        self.m = int(m)
        self.N = int(N)
        self.p = tuple(k.p for k in self.knot_evaluators)

        if bounds is not None:
            self.x_lower = jnp.asarray(bounds[0])
            self.x_upper = jnp.asarray(bounds[1])
            self.u_lower = jnp.asarray(bounds[2])
            self.u_upper = jnp.asarray(bounds[3])
        else:
            self.x_lower = jnp.full((self.N, self.n), -np.inf)
            self.x_upper = jnp.full((self.N, self.n), np.inf)
            self.u_lower = jnp.full((self.N - 1, self.m), -np.inf)
            self.u_upper = jnp.full((self.N - 1, self.m), np.inf)

        self.bound_evaluators = tuple(
            BuiltKnotConstraint(
                constraints=self._knot_bounds(k),
                n=self.n,
                m=self.m,
                is_terminal=(k == self.N - 1),
            )
            for k in range(self.N)
        )

        by_structure: dict[tuple, list[int]] = {}
        for k, evaluator in enumerate(self.knot_evaluators):
            key = (tuple(id(c) for c in evaluator.constraints), evaluator.is_terminal)
            by_structure.setdefault(key, []).append(k)
        self.groups = tuple(
            ConstraintGroup(evaluator=self.knot_evaluators[ks[0]], knots=tuple(ks)) for ks in by_structure.values()
        )

    def _knot_bounds(self, k: int) -> list[Constraint]:
        """Rebuild knot k's box limits as constraints, for consumers without native bounds."""
        cons: list[Constraint] = []
        x_lo = np.asarray(self.x_lower[k], dtype=float)
        x_hi = np.asarray(self.x_upper[k], dtype=float)
        if np.any(np.isfinite(x_lo)) or np.any(np.isfinite(x_hi)):
            cons.append(StateBound(n=self.n, x_min=x_lo, x_max=x_hi, m=self.m))

        if k < self.N - 1:
            u_lo = np.asarray(self.u_lower[k], dtype=float)
            u_hi = np.asarray(self.u_upper[k], dtype=float)
            if np.any(np.isfinite(u_lo)) or np.any(np.isfinite(u_hi)):
                cons.append(ControlBound(m=self.m, u_min=u_lo, u_max=u_hi, n=self.n))
        return cons

    def has_goal_constraint(self) -> bool:
        """Whether any knot carries a GoalConstraint, the one constraint that reads a run-time xf."""
        return any(isinstance(c, GoalConstraint) for ev in self.knot_evaluators for c in ev.constraints)

    def primal_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the collected primal variable box bounds (xL, xU, uL, uU)."""
        return (
            np.asarray(self.x_lower, dtype=np.float64),
            np.asarray(self.x_upper, dtype=np.float64),
            np.asarray(self.u_lower, dtype=np.float64),
            np.asarray(self.u_upper, dtype=np.float64),
        )

    def evaluate_knot(
        self,
        k: int,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
        *,
        xf: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate the fused constraint vector of shape (p_k,) at knot point k."""
        return self.knot_evaluators[k].evaluate(x, u, t, xf=xf)

    def jacobian_knot(
        self,
        k: int,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the fused Jacobians (dc/dx, dc/du) of shapes (p_k, n) and (p_k, m) at knot point k."""
        return self.knot_evaluators[k].jacobian(x, u, t)

    def evaluate(
        self,
        X: jax.Array,
        U: jax.Array,
        T: jax.Array | None = None,
    ) -> tuple[jax.Array, ...]:
        """Evaluate constraints over the horizon, one batched pass per structural group.

        Parameters
        ----------
        X : jax.Array
            State trajectory of shape (N, n).
        U : jax.Array
            Control trajectory of shape (N-1, m).
        T : jax.Array | None, optional
            Knot times of shape (N,). Defaults to zeros.

        Returns
        -------
        tuple[jax.Array, ...]
            One array of shape (len(g.knots), g.evaluator.p) per group g in `groups`.
        """
        T_arr = jnp.zeros(self.N, dtype=X.dtype) if T is None else T
        return tuple(g.evaluate(X, U, T_arr) for g in self.groups)

    def jacobian(
        self,
        X: jax.Array,
        U: jax.Array,
        T: jax.Array | None = None,
    ) -> tuple[tuple[jax.Array, jax.Array], ...]:
        """Evaluate Jacobians over the horizon, one batched pass per structural group.

        Parameters
        ----------
        X : jax.Array
            State trajectory of shape (N, n).
        U : jax.Array
            Control trajectory of shape (N-1, m).
        T : jax.Array | None, optional
            Knot times of shape (N,). Defaults to zeros.

        Returns
        -------
        tuple[tuple[jax.Array, jax.Array], ...]
            One (dc/dx, dc/du) pair per group g in `groups`, of shapes
            (len(g.knots), g.evaluator.p, n) and (len(g.knots), g.evaluator.p, m).
        """
        T_arr = jnp.zeros(self.N, dtype=X.dtype) if T is None else T
        return tuple(g.jacobian(X, U, T_arr) for g in self.groups)


class ConstraintList:
    """Stores the set of constraints and active knot-point ranges for an optimization problem.

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Control dimension.
    N : int
        Number of knot points in the trajectory horizon.
    """

    n: int
    m: int
    N: int
    constraints: list[Constraint]
    inds: list[tuple[int, ...]]
    p: np.ndarray

    def __init__(self, n: int, m: int, N: int) -> None:
        self.n = int(n)
        self.m = int(m)
        self.N = int(N)
        self.constraints = []
        self.inds = []
        self.p = np.zeros(self.N, dtype=int)

    def add_constraint(
        self,
        con: Constraint,
        inds: int | Sequence[int] | range | slice,
    ) -> None:
        """Register a constraint against an active knot-point index range.

        Parameters
        ----------
        con : Constraint
            Constraint object to add.
        inds : int | Sequence[int] | range | slice
            Knot-point indices at which this constraint applies (0-indexed).
        """
        if isinstance(inds, int):
            inds_tuple = (inds,)
        elif isinstance(inds, range):
            inds_tuple = tuple(inds)
        elif isinstance(inds, slice):
            inds_tuple = tuple(range(self.N)[inds])
        else:
            inds_tuple = tuple(int(i) for i in inds)

        if con.n != self.n:
            msg = f"State dimension mismatch: constraint n={con.n}, problem n={self.n}."
            raise ValueError(msg)
        if con.uses_control() and con.m != self.m:
            msg = f"Control dimension mismatch: constraint m={con.m}, problem m={self.m}."
            raise ValueError(msg)

        for k in inds_tuple:
            if not (0 <= k < self.N):
                msg = f"Index out of horizon [0, {self.N - 1}]: {k}."
                raise ValueError(msg)
            if k == self.N - 1 and con.uses_control():
                msg = (
                    "Control constraint cannot be applied at terminal knot point N-1: "
                    f"{type(con).__name__} reads the control block of z = [x; u]."
                )
                raise ValueError(msg)

        self.constraints.append(con)
        self.inds.append(inds_tuple)
        self._recompute_p()

    def _recompute_p(self) -> None:
        """Recompute total constraint dimension p across the horizon."""
        self.p.fill(0)
        for con, k_inds in zip(self.constraints, self.inds, strict=True):
            for k in k_inds:
                self.p[k] += con.p

    def num_constraints(self) -> np.ndarray:
        """Return registered constraint dimension at each knot point, of shape (N,).

        Counts box bounds, matching the Julia reference. `build` hoists those into primal
        limits, so `BuiltConstraintList.p` -- the transcribed row count -- is smaller.
        """
        return self.p.copy()

    def _apply_state_bound(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        k_inds: tuple[int, ...],
        xL: np.ndarray,
        xU: np.ndarray,
    ) -> None:
        """Tighten the state limits at knot points k_inds, writing into xL and xU in place.

        Parameters
        ----------
        lo, hi : np.ndarray
            State limits of shape (n,).
        k_inds : tuple[int, ...]
            Knot points at which the bound is active.
        xL, xU : np.ndarray
            Output arrays of shape (N, n), mutated in place.
        """
        for k in k_inds:
            xL[k] = np.maximum(xL[k], lo)
            xU[k] = np.minimum(xU[k], hi)

    def _apply_control_bound(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        k_inds: tuple[int, ...],
        uL: np.ndarray,
        uU: np.ndarray,
    ) -> None:
        """Tighten the control limits at knot points k_inds, writing into uL and uU in place.

        Parameters
        ----------
        lo, hi : np.ndarray
            Control limits of shape (m,).
        k_inds : tuple[int, ...]
            Knot points at which the bound is active; the terminal knot carries no control.
        uL, uU : np.ndarray
            Output arrays of shape (N-1, m), mutated in place.
        """
        for k in k_inds:
            if k < self.N - 1:
                uL[k] = np.maximum(uL[k], lo)
                uU[k] = np.minimum(uU[k], hi)

    def primal_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect the registered box bounds into per-knot solver variable limits.

        Interleaving these into the flat NLP vector Z is owned by `transcription/`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (xL, xU) of shape (N, n) and (uL, uU) of shape (N-1, m).
        """
        xL = np.full((self.N, self.n), -np.inf)
        xU = np.full((self.N, self.n), np.inf)
        uL = np.full((self.N - 1, self.m), -np.inf)
        uU = np.full((self.N - 1, self.m), np.inf)

        for con, k_inds in zip(self.constraints, self.inds, strict=True):
            if not isinstance(con, BoxBound):
                continue
            lo, hi = (np.asarray(b) for b in con.primal_bounds())
            if con.uses_state():
                self._apply_state_bound(lo[: self.n], hi[: self.n], k_inds, xL, xU)
            if con.uses_control():
                offset = lo.shape[0] - self.m
                self._apply_control_bound(lo[offset:], hi[offset:], k_inds, uL, uU)

        return xL, xU, uL, uU

    def build(self) -> BuiltConstraintList:
        """Trace and fuse all registered constraints into a single BuiltConstraintList.

        Box bounds are hoisted out of the knot evaluators and carried only as primal variable
        limits. `Box.residual` reproduces exactly the limits `primal_bounds` collects, so
        emitting both would duplicate every bound as a constraint row whose gradient is the
        unit vector of an already-active variable bound -- degenerating the active set.
        """
        knot_evaluators = [
            BuiltKnotConstraint(
                constraints=[
                    con
                    for con, k_inds in zip(self.constraints, self.inds, strict=True)
                    if k in k_inds and not isinstance(con, BoxBound)
                ],
                n=self.n,
                m=self.m,
                is_terminal=(k == self.N - 1),
            )
            for k in range(self.N)
        ]

        return BuiltConstraintList(
            knot_evaluators=knot_evaluators,
            n=self.n,
            m=self.m,
            N=self.N,
            bounds=self.primal_bounds(),
        )

    def __len__(self) -> int:
        """Return number of registered constraint groups."""
        return len(self.constraints)

    def __getitem__(self, idx: int | slice) -> Constraint | list[Constraint]:
        """Get registered constraint by index or slice."""
        return self.constraints[idx]

    def __iter__(self) -> Iterator[Constraint]:
        """Iterate over registered constraints."""
        return iter(self.constraints)
