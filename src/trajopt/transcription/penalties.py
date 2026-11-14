from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import ZeroCone
from trajopt.constraints.constraint_list import BuiltConstraintList, BuiltKnotConstraint
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.indexed import IndexedConstraint
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.base import CostFunction
from trajopt.costs.objective import Objective
from trajopt.costs.reformulation import CostReformulation, PenaltyDiagnostics
from trajopt.costs.sum import cost_terms
from trajopt.problem import Problem, _EpigraphModel
from trajopt.transcription.layout import _trajectory_to_z, _z_to_trajectory, compute_constraint_violation

if TYPE_CHECKING:
    from trajopt.problem import BoundaryConditions
    from trajopt.program import WarmStart
    from trajopt.trajectory import Trajectory
    from trajopt.transcription.result import SolverResult


class _KnotCost(CostFunction):
    """Evaluate a physical cost sum with selected terms replaced in solver coordinates."""

    terms: tuple[CostFunction, ...]
    indices: tuple[tuple[jax.Array, jax.Array], ...]

    def __init__(
        self, original: CostFunction, entries: tuple[_Entry, ...], dimensions: tuple[int, int], *, terminal: bool
    ) -> None:
        """Bind each replacement to its own physical and auxiliary coordinates."""
        super().__init__(*dimensions, terminal=terminal)
        self.terms = tuple(
            term
            if entry.form.cost is None
            else entry.form.cost.stacked(entry.horizon)
            if not terminal
            else entry.form.cost
            for term, entry in zip(cost_terms(original), entries, strict=True)
        )
        self.indices = tuple(entry.indices for entry in entries)

    def evaluate(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate replacement or unchanged terms at one Knot Point."""
        return jnp.sum(
            jnp.stack(
                [
                    cost.evaluate(x[ix], None if u is None else u[iu], t)
                    for cost, (ix, iu) in zip(self.terms, self.indices, strict=True)
                ]
            )
        )

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate all terms over the control Knot Points."""
        return jnp.sum(
            jnp.stack(
                [
                    cost.stage_costs(X[:, ix], U[:, iu], t)
                    for cost, (ix, iu) in zip(self.terms, self.indices, strict=True)
                ]
            ),
            axis=0,
        )

    def unstacked(self, k: int | jax.Array) -> CostFunction:
        """Select one Knot Point's individual cost parameters."""
        return eqx.tree_at(lambda cost: cost.terms, self, tuple(term.unstacked(k) for term in self.terms))


class _PenaltyObjective(Objective):
    """Retarget physical terms while retaining their prepared solver replacements."""

    physical: Objective
    stage_cost: _KnotCost
    terminal_cost: _KnotCost
    stage_kept: tuple[bool, ...] = eqx.field(static=True)
    terminal_kept: tuple[bool, ...] = eqx.field(static=True)

    def __init__(self, physical: Objective, layout: PenaltyLayout) -> None:
        """Build the cost functions consumed by the transformed Problem."""
        self.physical = physical
        dimensions = (layout.n + layout.ns, layout.m + layout.ms)
        stage = _KnotCost(physical.stage_cost, layout.stage, dimensions, terminal=False)
        terminal = _KnotCost(physical.terminal_cost, layout.terminal, dimensions, terminal=True)
        self.stage_kept = tuple(entry.form.cost is None for entry in layout.stage)
        self.terminal_kept = tuple(entry.form.cost is None for entry in layout.terminal)
        super().__init__(stage, terminal, physical.N)

    @property
    def can_retarget(self) -> bool:
        """Report whether the physical Objective accepts references."""
        return self.physical.can_retarget

    def with_reference(self, X_ref: jax.Array, U_ref: jax.Array) -> Objective:
        """Retarget unchanged terms and preserve solver replacement costs."""
        physical = self.physical.with_reference(X_ref[..., : self.physical.n], U_ref[..., : self.physical.m])
        stage = tuple(
            new if kept else old
            for new, old, kept in zip(
                cost_terms(physical.stage_cost), self.stage_cost.terms, self.stage_kept, strict=True
            )
        )
        terminal = tuple(
            new if kept else old
            for new, old, kept in zip(
                cost_terms(physical.terminal_cost), self.terminal_cost.terms, self.terminal_kept, strict=True
            )
        )
        return eqx.tree_at(
            lambda obj: (obj.physical, obj.stage_cost.terms, obj.terminal_cost.terms), self, (physical, stage, terminal)
        )


class _Entry(NamedTuple):
    """A cost's solver representation and its assigned auxiliary coordinates."""

    form: CostReformulation
    slot: slice
    indices: tuple[jax.Array, jax.Array]
    horizon: int


class PenaltyLayout:
    """Apply cost reformulations and map between physical and solver coordinates."""

    def __init__(self, problem: Problem, forms: tuple[tuple[CostReformulation | None, ...], ...]) -> None:
        """Allocate auxiliary coordinates for stage and terminal reformulations."""
        self.physical = problem
        self.n, self.m = problem.model.n, problem.model.m
        self.stage, self.terminal = (
            self._entries(parts, terminal=terminal) for parts, terminal in zip(forms, (False, True), strict=True)
        )
        self.ns = self.terminal[-1].slot.stop - self.n
        self.ms = self.stage[-1].slot.stop - self.m + self.ns
        self.has_new_rows = any(entry.form.constraints for entry in (*self.stage, *self.terminal))
        self.lowered = self._lower()
        self.lam_indices = self._physical_rows()

    def _entries(self, forms: tuple[CostReformulation | None, ...], *, terminal: bool) -> tuple[_Entry, ...]:
        """Assign contiguous auxiliary slices after the physical coordinates."""
        cursor = self.n if terminal else self.m
        items = []
        for definition in forms:
            form = CostReformulation() if definition is None else definition
            end = cursor + form.auxiliary_dim
            x = jnp.arange(self.n) if not terminal else jnp.r_[jnp.arange(self.n), jnp.arange(cursor, end)]
            u = jnp.r_[jnp.arange(self.m), jnp.arange(cursor, end)] if not terminal else jnp.arange(0)
            items.append(_Entry(form, slice(cursor, end), (x, u), self.physical.N))
            cursor = end
        return tuple(items)

    def _lower(self) -> Problem:
        """Build a standard Problem with ordinary costs, bounds, and constraints."""
        problem = self.physical
        obj = _PenaltyObjective(problem.obj, self)
        if not self.ms and not self.has_new_rows:
            return eqx.tree_at(lambda p: p.obj, problem, obj)
        model = _EpigraphModel(problem.model, self.ms, self.ns)
        return Problem(
            model,
            obj,
            self._constraints(),
            dt=problem.dt,
            binary_control_indices=problem.binary_control_indices,
            control_block_steps=problem.control_block_steps,
        )

    def _constraints(self) -> BuiltConstraintList:
        """Lift physical rows and attach the affine epigraph inequalities at their Knot Points."""
        problem, n, m = self.physical, self.n, self.m
        nn, mm = n + self.ns, m + self.ms
        stage_rows, terminal_rows = (
            [
                IndexedConstraint(nn, mm, con, ix=entry.indices[0].tolist(), iu=entry.indices[1].tolist()[: con.m])
                for entry in entries
                for con in entry.form.constraints
            ]
            for entries in (self.stage, self.terminal)
        )

        transport = (
            [
                LinearConstraint(
                    nn,
                    mm,
                    jnp.eye(self.ns),
                    jnp.zeros(self.ns),
                    sense=ZeroCone(),
                    inds=range(nn + mm - self.ns, nn + mm),
                )
            ]
            if self.ns
            else []
        )
        lifted = []
        original = problem.constraints
        for k, ev in enumerate(original.knot_evaluators):
            constraints = [
                GoalConstraint(nn, con.xf, con.inds, mm)
                if isinstance(con, GoalConstraint)
                else IndexedConstraint(nn, mm, con, ix=range(con.n), iu=range(con.m))
                for con in ev.constraints
            ]
            constraints.extend(terminal_rows if ev.is_terminal else stage_rows)
            if k < problem.N - 2:
                constraints.extend(transport)
            lifted.append(BuiltKnotConstraint(constraints, nn, mm, is_terminal=ev.is_terminal))

        N = problem.N
        stages = np.arange(N - 1)[:, None] * (nn + mm)
        stage_indices = stages + np.r_[np.arange(n), nn + np.arange(m)]
        physical_indices = np.r_[stage_indices.ravel(), (N - 1) * (nn + mm) + np.arange(n)]
        horizon = [
            LinearHorizonConstraint(con.A, con.b, sense=con.cone, inds=tuple(physical_indices[list(con.inds)]))
            for con in original.horizon_constraints
        ]
        xL, xU, uL, uU = original.primal_bounds()
        xL = np.pad(xL, ((0, 0), (0, self.ns)), constant_values=-np.inf)
        xU = np.pad(xU, ((0, 0), (0, self.ns)), constant_values=np.inf)
        uL = np.pad(uL, ((0, 0), (0, self.ms)), constant_values=-np.inf)
        uU = np.pad(uU, ((0, 0), (0, self.ms)), constant_values=np.inf)
        return BuiltConstraintList(lifted, nn, mm, N, bounds=(xL, xU, uL, uU), horizon_constraints=horizon)

    def _physical_rows(self) -> np.ndarray:
        """Index original pin, Defect, Knot Point, and Horizon rows in the lifted layout."""
        n, nn = self.n, self.n + self.ns
        indices = list(range(n))
        offset = nn
        for k, (old_p, new_p) in enumerate(zip(self.physical.constraints.p, self.lowered.constraints.p, strict=True)):
            if k < self.physical.N - 1:
                indices.extend(range(offset, offset + n))
                offset += nn
            indices.extend(range(offset, offset + old_p))
            offset += new_p
        horizon = sum(con.p for con in self.physical.constraints.horizon_constraints)
        indices.extend(range(offset, offset + horizon))
        return np.array(indices, dtype=int)

    def lift_primal(self, Z: jax.Array) -> jax.Array:
        """Seed slacks from positive residuals at the physical Primal Vector."""
        X, U = _z_to_trajectory(Z, self.physical.N, self.n, self.m)
        X_lift = jnp.pad(X, ((0, 0), (0, self.ns)))
        U_lift = jnp.pad(U, ((0, 0), (0, self.ms)))
        for entry in self.stage:
            if entry.form.initial_auxiliary is not None:
                U_lift = U_lift.at[:, entry.slot].set(entry.form.initial_auxiliary(X[:-1], U))
        for entry in self.terminal:
            if entry.form.initial_auxiliary is not None:
                end = entry.form.initial_auxiliary(X[-1], None)
                X_lift = X_lift.at[-1, entry.slot].set(end)
                transport = self.m + self.ms - self.ns - self.n
                U_lift = U_lift.at[-1, transport + entry.slot.start : transport + entry.slot.stop].set(end)
        return _trajectory_to_z(X_lift, U_lift)

    def lift_trajectory(self, traj: Trajectory) -> Trajectory:
        """Lift a physical Operating Point into the transformed trajectory layout."""
        Z = self.lift_primal(_trajectory_to_z(traj.X, traj.U))
        X, U = _z_to_trajectory(Z, traj.N, self.n + self.ns, self.m + self.ms)
        return eqx.tree_at(lambda t: (t.X, t.U), traj, (X, U))

    def lift_boundary(self, bc: BoundaryConditions) -> BoundaryConditions:
        """Pad Boundary Conditions for private coordinates while preserving absent targets."""
        xf = jnp.pad(bc.xf, (0, self.ns)) if bc.xf is not None and len(bc.xf) == self.n else bc.xf
        xr = None if bc.X_ref is None else jnp.pad(bc.X_ref, ((0, 0), (0, self.ns)))
        ur = None if bc.U_ref is None else jnp.pad(bc.U_ref, ((0, 0), (0, self.ms)))
        return replace(bc, x0=jnp.pad(bc.x0, (0, self.ns)), X_ref=xr, U_ref=ur, xf=xf)

    def lift_warm_start(self, ws: WarmStart) -> WarmStart:
        """Lift the primal seed and scatter physical Multipliers into the transformed layout."""
        mx, mu = _z_to_trajectory(ws.mu, self.physical.N, self.n, self.m)
        lifted_mu = _trajectory_to_z(jnp.pad(mx, ((0, 0), (0, self.ns))), jnp.pad(mu, ((0, 0), (0, self.ms))))
        size = (
            (self.n + self.ns) * self.lowered.N
            + sum(self.lowered.constraints.p)
            + sum(con.p for con in self.lowered.constraints.horizon_constraints)
        )
        lifted_lam = jnp.zeros(size, dtype=ws.lam.dtype).at[self.lam_indices].set(ws.lam)
        return eqx.tree_at(
            lambda w: (w.Z, w.lam, w.mu, w.al), ws, (self.lift_primal(ws.Z), lifted_lam, lifted_mu, None)
        )

    def diagnostics(self, traj: Trajectory) -> tuple[PenaltyDiagnostics, ...]:
        """Report positive residuals and raw solved slacks in declared term order."""
        X, U = traj.X[:, : self.n], traj.U[:, : self.m]
        stage = tuple(
            PenaltyDiagnostics(
                None if entry.form.violation is None else entry.form.violation(X[:-1], U),
                None,
                traj.U[:, entry.slot] if entry.form.auxiliary_dim else None,
                None,
            )
            for entry in self.stage
            if entry.form.violation is not None or entry.form.auxiliary_dim
        )
        terminal = tuple(
            PenaltyDiagnostics(
                None,
                None if entry.form.violation is None else entry.form.violation(X[-1], None),
                None,
                traj.X[-1, entry.slot] if entry.form.auxiliary_dim else None,
            )
            for entry in self.terminal
            if entry.form.violation is not None or entry.form.auxiliary_dim
        )
        return stage + terminal

    def project(self, result: SolverResult, bc: BoundaryConditions, *, diagnostics: bool) -> SolverResult:
        """Recover physical variables, Multipliers and cost from the transformed solve."""
        traj = result.trajectory
        X, U = traj.X[:, : self.n], traj.U[:, : self.m]
        Z = _trajectory_to_z(X, U)
        physical_traj = eqx.tree_at(lambda t: (t.X, t.U), traj, (X, U))
        lam, mu = result.lam, result.mu
        if len(lam):
            lam = np.asarray(lam)[self.lam_indices]
        if self.ms and len(mu):
            mx, mv = _z_to_trajectory(jnp.asarray(mu), self.physical.N, self.n + self.ns, self.m + self.ms)
            mu = np.asarray(_trajectory_to_z(mx[:, : self.n], mv[:, : self.m]))
        info = dict(result.info)
        if (self.ms or self.has_new_rows) and info.get("y") is not None:
            info["y"] = np.concatenate([lam, mu])
        if self.ms or self.has_new_rows:
            # Native solver vectors use the lifted layout. Canonical Multipliers are on the result.
            for key in ("x", "g", "z", "s", "mult_g", "mult_x_L", "mult_x_U"):
                info.pop(key, None)
        if diagnostics:
            info["penalties"] = self.diagnostics(traj)
        physical_violation = compute_constraint_violation(
            self.physical, Z, bc.x0, t0=bc.t0, dt=self.physical.dt, xf=bc.xf
        )
        return _PhysicalResult(
            result,
            physical_traj,
            Z,
            lam,
            mu,
            info,
            float(bc.retarget(self.physical.obj).cost(physical_traj)),
            max(physical_violation, result.constraint_violation),
            getattr(result, "al", None) if not (self.ms or self.has_new_rows) else None,
            getattr(result, "duals", None) if not (self.ms or self.has_new_rows) else None,
            result.success,
            result.status,
            result.message,
            result.iterations,
        )


@dataclass(frozen=True)
class _PhysicalResult:
    """A solver result with physical primal and dual coordinates."""

    _result: SolverResult
    trajectory: Trajectory
    Z: jax.Array
    lam: np.ndarray
    mu: np.ndarray
    info: dict[str, object]
    cost: float
    constraint_violation: float
    al: object
    duals: object
    success: bool
    status: object
    message: str
    iterations: int

    def __getattr__(self, name: str) -> object:
        """Forward solver-specific status and statistics."""
        return getattr(self._result, name)
