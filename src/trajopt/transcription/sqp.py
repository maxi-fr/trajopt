# ruff: noqa: PLR0915 -- the SQP solve coordinates evaluation, QP, and line search

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import SecondOrderCone
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _z_to_trajectory, constraint_bounds, parse_solver_initial_state, primal_bounds
from trajopt.transcription.result import warm_start_duals
from trajopt.transcription.sparsity import hessian_sparsity_pattern, jacobian_sparsity_pattern
from trajopt.transcription.subproblem import constraint_blocks
from trajopt.transcription.transcription import (
    constraints_and_jac,
    cost_and_grad,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_h,
    eval_jac_g,
)


class SQPResult(NamedTuple):
    """Result of a full-space sequential quadratic programming solve."""

    trajectory: Trajectory
    success: bool
    status: str
    message: str
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    constraint_violation: float
    iterations: int
    lam: np.ndarray
    mu: np.ndarray


def _violation(
    c: np.ndarray,
    z: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: tuple,
) -> np.ndarray:
    """Return nonnegative row and bound violations, replacing cone rows by their projection distance."""
    gl, gu, zl, zu = bounds
    rows = np.maximum(gl - c, 0.0) + np.maximum(c - gu, 0.0)
    for block in blocks:
        if isinstance(block.cone, SecondOrderCone):
            v = c[block.start : block.stop]
            rows[block.start : block.stop] = v - np.asarray(block.cone.project(jnp.asarray(v)))
    return np.concatenate([np.abs(rows), np.maximum(zl - z, 0.0), np.maximum(z - zu, 0.0)])


def _bfgs(h: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply a Powell-damped BFGS update to a positive definite matrix."""
    bs = h @ s
    sbs = float(s @ bs)
    if sbs <= 0.0:
        return h
    sy = float(s @ y)
    if sy < 0.2 * sbs:
        theta = 0.8 * sbs / (sbs - sy)
        y = theta * y + (1.0 - theta) * bs
        sy = float(s @ y)
    return h - np.outer(bs, bs) / sbs + np.outer(y, y) / sy


def _linearized_rows(
    c: np.ndarray, jac: sp.csr_matrix, bounds: tuple[np.ndarray, np.ndarray], blocks: tuple
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Build affine row bounds, replacing each cone by a supporting plane."""
    gl, gu = bounds
    lower, upper = gl - c, gu - c
    if not any(isinstance(block.cone, SecondOrderCone) for block in blocks):
        return jac.tocsc(), lower, upper
    a = jac.tolil(copy=True)
    for block in blocks:
        if isinstance(block.cone, SecondOrderCone):
            v = c[block.start : block.stop]
            jblock = jac[block.start : block.stop]
            norm = np.linalg.norm(v[:-1])
            direction = v[:-1] / norm if norm > 0.0 else np.zeros(len(v) - 1)
            plane = sp.csr_matrix(direction.reshape(1, -1)) @ jblock[:-1] - jblock[-1]
            a[block.start, :] = plane
            lower[block.start] = -np.inf
            upper[block.start] = v[-1] - norm
            lower[block.start + 1 : block.stop] = -np.inf
            upper[block.start + 1 : block.stop] = np.inf
    return a.tocsc(), lower, upper


def _canonical_cone_duals(duals: np.ndarray, c: np.ndarray, blocks: tuple) -> np.ndarray:
    """Map each cone supporting-plane Multiplier into its canonical vector rows."""
    lam = duals.copy()
    for block in blocks:
        if isinstance(block.cone, SecondOrderCone):
            v = c[block.start : block.stop]
            norm = np.linalg.norm(v[:-1])
            multiplier = lam[block.start]
            lam[block.start : block.stop - 1] = multiplier * (v[:-1] / norm if norm > 0.0 else 0.0)
            lam[block.stop - 1] = -multiplier
    return lam


def _qp_hessian(problem: Problem, z: np.ndarray, t0: jax.Array, dt: jax.Array) -> sp.csc_matrix:
    """Evaluate the positive semidefinite objective Hessian for Gauss-Newton mode."""
    n, m, horizon = int(problem.model.n), int(problem.model.m), int(problem.N)
    rows, cols = hessian_sparsity_pattern(horizon, n, m)
    vals = np.asarray(eval_h(problem, jnp.asarray(z), t0=t0, dt=dt))
    h = sp.coo_matrix((vals, (rows, cols)), shape=(len(z), len(z))).tocsc()
    return (h + sp.tril(h, k=-1).T + sp.eye(len(z), format="csc") * 1e-8).tocsc()


def _knot_slices(problem: Problem) -> tuple[slice, ...]:
    """Partition the Primal Vector into stage and terminal Knot Point blocks."""
    width = int(problem.model.n) + int(problem.model.m)
    horizon = int(problem.N)
    return tuple(slice(k * width, (k + 1) * width if k < horizon - 1 else None) for k in range(horizon))


def _block_hessian(
    blocks: list[np.ndarray], pattern: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]
) -> sp.csc_matrix:
    """Fill the fixed CSC upper-block pattern with knot-local BFGS values."""
    indices, indptr, positions = pattern
    values = np.concatenate([block.ravel()[part] for block, part in zip(blocks, positions, strict=True)])
    return sp.csc_matrix((values, indices, indptr), shape=(indptr.size - 1, indptr.size - 1))


@eqx.filter_jit
def _trial_values(  # noqa: PLR0913 -- trial evaluation needs the state and time context
    problem: Problem, z: jax.Array, x0: jax.Array, t0: jax.Array, dt: jax.Array, *, xf: jax.Array | None
) -> tuple[jax.Array, jax.Array]:
    """Evaluate cost and constraints together for a line-search trial."""
    return eval_f(problem, z, t0, dt), eval_g(problem, z, x0, t0, dt, xf=xf)


@eqx.filter_jit
def _trial_derivatives(  # noqa: PLR0913 -- derivatives need the state and time context
    problem: Problem, z: jax.Array, x0: jax.Array, t0: jax.Array, dt: jax.Array, *, xf: jax.Array | None
) -> tuple[jax.Array, jax.Array]:
    """Evaluate derivatives after accepting a trial without returning its values again."""
    return eval_grad_f(problem, z, t0, dt), eval_jac_g(problem, z, x0, t0, dt, xf=xf)


@dataclass(frozen=True)
class SQP:
    """SQP Backend with sparse knot-local damped BFGS or Gauss-Newton Hessians.

    SecondOrderCone blocks use a local supporting plane, which can be a poor model near the
    cone apex. The BFGS mode updates each Knot Point block separately, so Hessian storage is
    linear in Horizon length for fixed state and control dimensions. The direct OSQP Backend
    instead solves one Quadratic Subproblem.
    """

    hessian: str = "bfgs"
    max_iter: int = 100
    primal_tol: float = 1e-7
    dual_tol: float = 1e-7
    min_step: float = 1e-8
    armijo: float = 1e-4
    backtrack: float = 0.5
    memory: int = 5
    penalty_max: float = 1e8
    options: Mapping[str, Any] = field(default_factory=dict)

    def solve(  # noqa: C901 -- the SQP loop has distinct termination paths
        self,
        program: Program,
        bc: BoundaryConditions,
        ws: WarmStart,
    ) -> SQPResult:
        """Solve a Program by repeated OSQP steps and an L1 Merit Function line search."""
        import osqp  # noqa: PLC0415 -- OSQP is an optional solver dependency

        problem = retarget_problem(program.problem, bc)
        x0, t0, dt, xf, initial = parse_solver_initial_state(problem, bc, ws)
        dt = jnp.broadcast_to(dt, (int(problem.N) - 1,))
        z = np.asarray(initial, dtype=np.float64).copy()
        nz = len(z)
        gl, gu = constraint_bounds(problem)
        zl, zu = primal_bounds(problem)
        blocks = constraint_blocks(problem)
        jr, jc = jacobian_sparsity_pattern(
            int(problem.N), int(problem.model.n), int(problem.model.m), problem.constraints.p
        )
        knot_slices = _knot_slices(problem)
        bfgs_blocks = [np.eye(z[part].size) for part in knot_slices]
        positions = tuple(
            np.concatenate([np.arange(col + 1) * len(block) + col for col in range(len(block))])
            for block in bfgs_blocks
        )
        indices = np.concatenate(
            [
                np.concatenate([np.arange(col + 1) + part.start for col in range(len(block))])
                for block, part in zip(bfgs_blocks, knot_slices, strict=True)
            ]
        )
        indptr = np.r_[0, np.cumsum([col + 1 for block in bfgs_blocks for col in range(len(block))])]
        h_pattern = (indices, indptr, positions)
        h = _block_hessian(bfgs_blocks, h_pattern)
        lam0, mu0 = warm_start_duals(problem, ws)
        lam = np.zeros(len(gl)) if lam0 is None else lam0.copy()
        mu = np.zeros(nz) if mu0 is None else mu0.copy()
        penalty = 1.0
        history: list[float] = []
        status = "iteration_limit"
        iterations = 0
        qp_info: dict[str, Any] = {}
        solver: osqp.OSQP | None = None
        qp_p_pattern: tuple[np.ndarray, np.ndarray] | None = None
        qp_a_pattern: tuple[np.ndarray, np.ndarray] | None = None

        def evaluate(
            point: np.ndarray, values: tuple[float, np.ndarray] | None = None
        ) -> tuple[float, np.ndarray, np.ndarray, sp.csr_matrix]:
            """Evaluate derivatives and reuse cost and constraints from an accepted trial."""
            p = jnp.asarray(point)
            if values is None:
                f, grad = cost_and_grad(problem, p, t0, dt)
                c, jvals = constraints_and_jac(problem, p, x0, t0, dt, xf=xf)
            else:
                f, c = values
                grad, jvals = _trial_derivatives(problem, p, x0, t0, dt, xf=xf)
            jac = sp.coo_matrix((np.asarray(jvals), (jr, jc)), shape=(len(gl), nz)).tocsr()
            return float(f), np.asarray(grad), np.asarray(c), jac

        def solve_qp() -> Any:  # noqa: ANN401 -- OSQP's C-extension result has no type
            """Solve the current convex Quadratic Subproblem in the step variable."""
            nonlocal solver, qp_p_pattern, qp_a_pattern
            a, lower, upper = _linearized_rows(c, jac, (gl, gu), blocks)
            matrix = sp.vstack([a, sp.eye(nz, format="csc")], format="csc")
            p = h if self.hessian == "bfgs" else sp.triu(h, format="csc")
            lower = np.r_[lower, zl - z]
            upper = np.r_[upper, zu - z]
            p_pattern = (p.indptr, p.indices)
            a_pattern = (matrix.indptr, matrix.indices)
            same_pattern = (
                qp_p_pattern is not None
                and qp_a_pattern is not None
                and all(
                    np.array_equal(old, new)
                    for old, new in zip((*qp_p_pattern, *qp_a_pattern), (*p_pattern, *a_pattern), strict=True)
                )
            )
            if same_pattern and solver is not None:
                solver.update(Px=p.data, Ax=matrix.data, q=grad, l=lower, u=upper)
            else:
                solver = osqp.OSQP()
                qp_options = {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 20000, **self.options}
                solver.setup(P=p, q=grad, A=matrix, l=lower, u=upper, **qp_options)
                qp_p_pattern = p_pattern
                qp_a_pattern = a_pattern
                if iteration == 0 and lam0 is not None and mu0 is not None:
                    solver.warm_start(x=np.zeros(nz), y=np.r_[lam, mu])
            return solver.solve()

        def backtrack() -> tuple[np.ndarray, float, float, np.ndarray] | None:
            """Find an Armijo step against the recent Merit Function memory."""
            alpha = 1.0
            while alpha >= self.min_step:
                trial = z + alpha * step
                trial_f_jax, trial_c_jax = _trial_values(problem, jnp.asarray(trial), x0, t0, dt, xf=xf)
                trial_f = float(trial_f_jax)
                trial_c = np.asarray(trial_c_jax)
                trial_merit = trial_f + penalty * float(np.sum(_violation(trial_c, trial, (gl, gu, zl, zu), blocks)))
                if trial_merit <= max(history) + self.armijo * alpha * min(predicted, 0.0):
                    return trial, alpha, trial_f, trial_c
                alpha *= self.backtrack
            return None

        f, grad, c, jac = evaluate(z)
        for iteration in range(self.max_iter + 1):
            violation = _violation(c, z, (gl, gu, zl, zu), blocks)
            primal = float(np.max(violation))
            dual = float(np.max(np.abs(grad + jac.T @ lam + mu)))
            if primal <= self.primal_tol and dual <= self.dual_tol:
                status = "converged"
                break
            if iteration == self.max_iter:
                break
            if self.hessian == "gauss_newton":
                h = _qp_hessian(problem, z, t0, dt)
            elif self.hessian != "bfgs":
                msg = f"Unknown SQP Hessian mode: {self.hessian}"
                raise ValueError(msg)
            result = solve_qp()
            qp_info = {"status": result.info.status, "status_val": result.info.status_val}
            if result.info.status_val not in (1, 2):
                status = "infeasible" if "infeasible" in result.info.status.lower() else "error"
                break
            step = np.asarray(result.x)
            raw_lam, next_mu = np.split(np.asarray(result.y), [len(gl)])
            next_lam = _canonical_cone_duals(raw_lam, c, blocks)
            penalty = min(self.penalty_max, max(penalty, 1.1 * float(np.max(np.abs(result.y)))))
            merit = f + penalty * float(np.sum(violation))
            history.append(merit)
            history = history[-self.memory :]
            predicted = float(grad @ step - penalty * np.sum(violation))
            accepted = backtrack()
            if accepted is None:
                status = "line search failure"
                break
            trial, alpha, trial_f, trial_c = accepted
            old_grad, old_jac = grad, jac
            z = trial
            f, grad, c, jac = evaluate(z, (trial_f, trial_c))
            if self.hessian == "bfgs":
                s = alpha * step
                y = grad + jac.T @ next_lam - old_grad - old_jac.T @ next_lam
                bfgs_blocks = [
                    _bfgs(block, s[part], y[part]) for block, part in zip(bfgs_blocks, knot_slices, strict=True)
                ]
                h = _block_hessian(bfgs_blocks, h_pattern)
            lam, mu = next_lam, next_mu
            iterations += 1

        Z = jnp.asarray(z)
        X, U = _z_to_trajectory(Z, int(problem.N), int(problem.model.n), int(problem.model.m))
        trajectory = Trajectory(X=X, U=U, t=t0 + jnp.r_[jnp.zeros(1), jnp.cumsum(dt)], dt=dt)
        return SQPResult(
            trajectory,
            status == "converged",
            status,
            status,
            f,
            Z,
            qp_info,
            float(np.max(_violation(c, z, (gl, gu, zl, zu), blocks))),
            iterations,
            lam,
            mu,
        )
