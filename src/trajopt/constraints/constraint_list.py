"""ConstraintList for registering and fusing constraints across the horizon."""

from collections.abc import Iterator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.base import Constraint, ControlConstraint, StateConstraint
from trajopt.constraints.bounds import BoundConstraint, ControlBound, StateBound


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
    ) -> jax.Array:
        """Evaluate concatenated constraint vector at this knot point."""
        if len(self.constraints) == 0 or self.p == 0:
            return jnp.zeros(0, dtype=x.dtype)

        evals = [c.evaluate(x, u if not self.is_terminal else None, t) for c in self.constraints]
        return jnp.concatenate(evals)

    def jacobian(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate concatenated Jacobians (dc/dx, dc/du) at this knot point."""
        dtype = x.dtype
        if len(self.constraints) == 0 or self.p == 0:
            jx = jnp.zeros((0, self.n), dtype=dtype)
            ju = jnp.zeros((0, 0 if self.is_terminal else self.m), dtype=dtype)
            return jx, ju

        jx_list: list[jax.Array] = []
        ju_list: list[jax.Array] = []

        for c in self.constraints:
            if isinstance(c, StateConstraint):
                jx_c = c.jacobian_x(x, None, t)
                jx_list.append(jx_c)
                if not self.is_terminal:
                    ju_list.append(jnp.zeros((c.p, self.m), dtype=dtype))
            elif isinstance(c, ControlConstraint):
                jx_list.append(jnp.zeros((c.p, self.n), dtype=dtype))
                if not self.is_terminal:
                    ju_c = c.jacobian_u(None, u, t)
                    ju_list.append(ju_c)
            else:
                jx_c, ju_c = c.jacobian(x, u if not self.is_terminal else None, t)
                jx_list.append(jx_c)
                if not self.is_terminal:
                    ju_list.append(ju_c)

        jx = jnp.vstack(jx_list)
        ju = jnp.zeros((self.p, 0), dtype=dtype) if self.is_terminal else jnp.vstack(ju_list)

        return jx, ju


class BuiltConstraintList(eqx.Module):
    """Fused constraint set across all horizon knot points."""

    knot_evaluators: tuple[BuiltKnotConstraint, ...]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    p: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        knot_evaluators: Sequence[BuiltKnotConstraint],
        n: int,
        m: int,
        N: int,
    ) -> None:
        self.knot_evaluators = tuple(knot_evaluators)
        self.n = int(n)
        self.m = int(m)
        self.N = int(N)
        self.p = tuple(k.p for k in self.knot_evaluators)

    def evaluate_knot(
        self,
        k: int,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate fused constraint vector at knot point k."""
        return self.knot_evaluators[k].evaluate(x, u, t)

    def jacobian_knot(
        self,
        k: int,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate fused Jacobians (dc/dx, dc/du) at knot point k."""
        return self.knot_evaluators[k].jacobian(x, u, t)

    def evaluate(
        self,
        X: jax.Array,
        U: jax.Array,
        T: jax.Array | None = None,
    ) -> list[jax.Array]:
        """Evaluate constraints across all horizon knot points.

        Parameters
        ----------
        X : jax.Array
            State trajectory of shape (N, n).
        U : jax.Array
            Control trajectory of shape (N-1, m).
        T : jax.Array | None, optional
            Time array of shape (N,). Defaults to zeros.

        Returns
        -------
        list[jax.Array]
            List of N constraint vectors c_k of shape (p_k,).
        """
        results: list[jax.Array] = []
        for k in range(self.N):
            xk = X[k]
            uk = U[k] if k < self.N - 1 else None
            tk = T[k] if T is not None else 0.0
            results.append(self.knot_evaluators[k].evaluate(xk, uk, tk))
        return results

    def jacobian(
        self,
        X: jax.Array,
        U: jax.Array,
        T: jax.Array | None = None,
    ) -> list[tuple[jax.Array, jax.Array]]:
        """Evaluate Jacobians across all horizon knot points.

        Parameters
        ----------
        X : jax.Array
            State trajectory of shape (N, n).
        U : jax.Array
            Control trajectory of shape (N-1, m).
        T : jax.Array | None, optional
            Time array of shape (N,). Defaults to zeros.

        Returns
        -------
        list[tuple[jax.Array, jax.Array]]
            List of N tuples (dc_k/dx_k, dc_k/du_k) of shapes (p_k, n) and (p_k, m).
        """
        results: list[tuple[jax.Array, jax.Array]] = []
        for k in range(self.N):
            xk = X[k]
            uk = U[k] if k < self.N - 1 else None
            tk = T[k] if T is not None else 0.0
            results.append(self.knot_evaluators[k].jacobian(xk, uk, tk))
        return results


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

        # Dimension checks
        if con.n != self.n:
            msg = f"State dimension mismatch: constraint n={con.n}, problem n={self.n}."
            raise ValueError(msg)
        if con.m != self.m and not isinstance(con, StateConstraint):
            msg = f"Control dimension mismatch: constraint m={con.m}, problem m={self.m}."
            raise ValueError(msg)

        # Horizon index checks
        for k in inds_tuple:
            if not (0 <= k < self.N):
                msg = f"Index out of horizon [0, {self.N - 1}]: {k}."
                raise ValueError(msg)
            if k == self.N - 1 and (
                isinstance(con, ControlConstraint)
                or (
                    isinstance(con, BoundConstraint)
                    and len(con.i_max) + len(con.i_min) > 0
                    and any(c_idx >= self.n for c_idx in list(con.i_max) + list(con.i_min))
                )
            ):
                msg = "Control constraint cannot be applied at terminal knot point N-1."
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
        """Return total constraint dimension at each knot point across the horizon."""
        return self.p.copy()

    def _apply_state_bound(
        self,
        con: StateBound,
        k_inds: tuple[int, ...],
        zL: np.ndarray,
        zU: np.ndarray,
        nm: int,
    ) -> None:
        xL, xU = con.primal_bounds()
        xmin_np = np.asarray(xL)
        xmax_np = np.asarray(xU)
        for k in k_inds:
            offset = k * nm
            zL[offset : offset + self.n] = np.maximum(zL[offset : offset + self.n], xmin_np)
            zU[offset : offset + self.n] = np.minimum(zU[offset : offset + self.n], xmax_np)

    def _apply_control_bound(
        self,
        con: ControlBound,
        k_inds: tuple[int, ...],
        zL: np.ndarray,
        zU: np.ndarray,
        nm: int,
    ) -> None:
        uL, uU = con.primal_bounds()
        umin_np = np.asarray(uL)
        umax_np = np.asarray(uU)
        for k in k_inds:
            if k < self.N - 1:
                offset = k * nm + self.n
                zL[offset : offset + self.m] = np.maximum(zL[offset : offset + self.m], umin_np)
                zU[offset : offset + self.m] = np.minimum(zU[offset : offset + self.m], umax_np)

    def _apply_bound_con(
        self,
        con: BoundConstraint,
        k_inds: tuple[int, ...],
        zL: np.ndarray,
        zU: np.ndarray,
        nm: int,
    ) -> None:
        zL_con, zU_con = con.primal_bounds()
        zmin_np = np.asarray(zL_con)
        zmax_np = np.asarray(zU_con)
        for k in k_inds:
            offset = k * nm
            if k < self.N - 1:
                zL[offset : offset + nm] = np.maximum(zL[offset : offset + nm], zmin_np)
                zU[offset : offset + nm] = np.minimum(zU[offset : offset + nm], zmax_np)
            else:
                zL[offset : offset + self.n] = np.maximum(zL[offset : offset + self.n], zmin_np[: self.n])
                zU[offset : offset + self.n] = np.minimum(zU[offset : offset + self.n], zmax_np[: self.n])

    def primal_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute solver primal variable bounds (zL, zU) across the trajectory horizon.

        Primal layout: Z = [x_0, u_0, x_1, u_1, ..., x_{N-2}, u_{N-2}, x_{N-1}].
        Length: N * n + (N - 1) * m.
        """
        nm = self.n + self.m
        total_dim = (self.N - 1) * nm + self.n
        zL = np.full(total_dim, -np.inf)
        zU = np.full(total_dim, np.inf)

        for con, k_inds in zip(self.constraints, self.inds, strict=True):
            if isinstance(con, StateBound):
                self._apply_state_bound(con, k_inds, zL, zU, nm)
            elif isinstance(con, ControlBound):
                self._apply_control_bound(con, k_inds, zL, zU, nm)
            elif isinstance(con, BoundConstraint):
                self._apply_bound_con(con, k_inds, zL, zU, nm)

        return zL, zU

    def build(self) -> BuiltConstraintList:
        """Trace and fuse all registered constraints into a single BuiltConstraintList."""
        knot_evaluators: list[BuiltKnotConstraint] = []

        for k in range(self.N):
            active_cons: list[Constraint] = []
            for con, k_inds in zip(self.constraints, self.inds, strict=True):
                if k in k_inds:
                    active_cons.append(con)

            evaluator = BuiltKnotConstraint(
                constraints=active_cons,
                n=self.n,
                m=self.m,
                is_terminal=(k == self.N - 1),
            )
            knot_evaluators.append(evaluator)

        return BuiltConstraintList(
            knot_evaluators=knot_evaluators,
            n=self.n,
            m=self.m,
            N=self.N,
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
