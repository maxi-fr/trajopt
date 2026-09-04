import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import (
    _z_to_trajectory,
    compute_constraint_violation,
    operating_point_z,
    parse_solver_initial_state,
    primal_bounds,
)
from trajopt.transcription.subproblem import QuadraticSubproblem, quadratic_subproblem
from trajopt.transcription.transcription import (
    eval_f,
)

_EMPTY = np.zeros(0, dtype=np.float64)


class ClarabelResult(NamedTuple):
    """Result of a Clarabel trajectory optimization solve.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the solver converged to optimality or within tolerance.
    status : str
        Clarabel solver status string.
    message : str
        Solver return status message.
    cost : float
        Final objective value.
    Z : jax.Array
        Optimal flat primal vector of shape ``(N * n + (N - 1) * m,)``.
    info : dict[str, Any]
        Raw Clarabel return info dictionary.
    constraint_violation : float
        Maximum constraint violation across all constraints.
    iterations : int, optional
        Number of solver iterations. Defaults to 0.
    lam : np.ndarray, optional
        Constraint duals in canonical row order, of shape ``(P,)``. Defaults to empty.
    mu : np.ndarray, optional
        Signed bound duals of shape ``(N * n + (N - 1) * m,)``. Defaults to empty.
    """

    trajectory: Trajectory
    success: bool
    status: str
    message: str
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    constraint_violation: float
    iterations: int = 0
    lam: np.ndarray = _EMPTY
    mu: np.ndarray = _EMPTY


def _make_cone(name: str, dim: int) -> object:
    """Instantiate a Clarabel cone constraint object."""
    import clarabel  # noqa: PLC0415 -- clarabel is an optional solver dependency

    cone_cls: Any = getattr(clarabel, name)
    return cone_cls(dim)


def _conic_rows(qp: QuadraticSubproblem) -> tuple[list[sp.spmatrix], list[np.ndarray], list[object]]:
    """Re-express the subproblem's linearized rows as Clarabel's ``A z + s = b, s in K``.

    Every Cone but the second-order one is a box on ``A z``, so its row block goes over with at
    most a sign flip. A `SecondOrderCone` block is the exception: Clarabel wants the scalar bound
    first and the vector after, while the transcription emits the vector first, so that block is
    permuted as well as negated.
    """
    A_rows: list[sp.spmatrix] = []
    b_vals: list[np.ndarray] = []
    cones: list[object] = []

    for block in qp.blocks:
        rows = qp.A[block.start : block.stop]
        affine = qp.affine[block.start : block.stop]
        dim_c = block.stop - block.start

        if isinstance(block.cone, SecondOrderCone):
            order = np.concatenate([[dim_c - 1], np.arange(dim_c - 1)])
            A_rows.append(-rows[order])
            b_vals.append(affine[order])
            cones.append(_make_cone("SecondOrderConeT", dim_c))
        elif isinstance(block.cone, ZeroCone):
            A_rows.append(rows)
            b_vals.append(-affine)
            cones.append(_make_cone("ZeroConeT", dim_c))
        elif isinstance(block.cone, PositiveOrthant):
            A_rows.append(-rows)
            b_vals.append(affine)
            cones.append(_make_cone("NonnegativeConeT", dim_c))
        elif isinstance(block.cone, NegativeOrthant):
            A_rows.append(rows)
            b_vals.append(-affine)
            cones.append(_make_cone("NonnegativeConeT", dim_c))
        else:
            msg = f"Clarabel adapter does not support cone {type(block.cone).__name__}."
            raise TypeError(msg)

    return A_rows, b_vals, cones


def _extract_conic_bounds(
    problem: Problem,
    nz: int,
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[object]]:
    """Extract primal variable bounds as NonnegativeCone constraints for Clarabel."""
    A_rows: list[sp.spmatrix] = []
    b_vals: list[np.ndarray] = []
    cones: list[object] = []

    zL, zU = primal_bounds(problem)
    ub_indices = np.where(np.isfinite(zU))[0]
    lb_indices = np.where(np.isfinite(zL))[0]

    if len(ub_indices) > 0:
        A_ub = sp.lil_matrix((len(ub_indices), nz), dtype=np.float64)
        for row_idx, var_idx in enumerate(ub_indices):
            A_ub[row_idx, var_idx] = 1.0
        A_rows.append(A_ub.tocsr())
        b_vals.append(zU[ub_indices])
        cones.append(_make_cone("NonnegativeConeT", len(ub_indices)))

    if len(lb_indices) > 0:
        A_lb = sp.lil_matrix((len(lb_indices), nz), dtype=np.float64)
        for row_idx, var_idx in enumerate(lb_indices):
            A_lb[row_idx, var_idx] = -1.0
        A_rows.append(A_lb.tocsr())
        b_vals.append(-zL[lb_indices])
        cones.append(_make_cone("NonnegativeConeT", len(lb_indices)))

    return A_rows, b_vals, cones


def _normalise_duals(
    problem: Problem,
    z_dual: np.ndarray,
    nz: int,
    n_con_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split Clarabel's cone duals into canonical constraint duals and signed bound duals.

    The subproblem's rows already come in canonical order, so the leading `n_con_rows` duals are
    `lam` as it stands. Clarabel keeps every cone dual non-negative and posts the two sides of a
    box as separate NonnegativeCone blocks, upper first, so recombining those as ``upper - lower``
    recovers the signed convention the other backends report.
    """
    lam = z_dual[:n_con_rows]

    zL, zU = primal_bounds(problem)
    ub_indices = np.where(np.isfinite(zU))[0]
    lb_indices = np.where(np.isfinite(zL))[0]

    mu = np.zeros(nz, dtype=np.float64)
    bound_base = n_con_rows
    mu[ub_indices] += z_dual[bound_base : bound_base + len(ub_indices)]
    bound_base += len(ub_indices)
    mu[lb_indices] -= z_dual[bound_base : bound_base + len(lb_indices)]
    return lam, mu


@dataclass(frozen=True)
class Clarabel:
    """Clarabel conic interior-point solver backend, expanding the problem about `operating_point`.

    Clarabel is a conic interior-point solver, so this adapter hands it one convex approximation
    of the problem, expanded about ``operating_point`` (the origin by default): the dynamics
    linearized to :math:`x_{k+1} = A_k x_k + B_k u_k + d_k`, the constraints linearized, and the
    cost taken to second order. The approximation is exact — and the result is the true optimum —
    when the dynamics are affine, the cost quadratic and the constraints linear or second-order
    conic, whatever the operating point.

    Otherwise this is linearized MPC: the returned trajectory optimizes the approximation, not
    the nonlinear problem, and ``success`` reports the conic solver's convergence.
    ``constraint_violation`` is measured against the true nonlinear dynamics and is the number
    that says how far the two have drifted apart; drive it down by re-solving with the operating
    point set to the previous solution. Use Ipopt to solve the nonlinear problem directly.

    Parameters
    ----------
    operating_point : Trajectory | jax.Array | None, optional
        Point about which the dynamics, constraints and cost are expanded, as a trajectory or a
        flat vector of shape ``(N * n + (N - 1) * m,)``. Defaults to None, meaning the origin.
    options : Mapping[str, Any], optional
        Native Clarabel DefaultSettings options (e.g. {"tol_gap_abs": 1e-8, "max_iter": 200}).
        Defaults to empty.
    """

    operating_point: Trajectory | jax.Array | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> ClarabelResult:
        """Solve the transcribed optimal control problem using Clarabel, warning that it is one linearization.

        The warning is unconditional rather than gated on the problem being nonlinear: every
        caller handing this Backend a `Problem` gets one convex solve about the Operating
        Point, and whether that answers their problem is theirs to judge.
        """
        problem = program.problem
        warnings.warn(
            "Clarabel solves a single convex subproblem built about the Operating Point: the "
            "dynamics are linearized and the cost taken to second order once, so the result "
            "optimizes an approximation of a nonlinear problem, not the problem itself.",
            stacklevel=2,
        )

        try:
            import clarabel  # noqa: PLC0415 -- clarabel is an optional solver dependency
        except ImportError as e:
            msg = (
                "clarabel is not installed. It is part of the `solvers` extra: install with "
                '`pip install "trajopt[solvers]"` or `uv add "trajopt[solvers]"`.'
            )
            raise ImportError(msg) from e

        settings_cls: Any = getattr(clarabel, "DefaultSettings")  # noqa: B009 -- clarabel is an untyped C-extension
        solver_cls: Any = getattr(clarabel, "DefaultSolver")  # noqa: B009 -- clarabel is an untyped C-extension
        solver_status_cls: Any = getattr(clarabel, "SolverStatus")  # noqa: B009 -- clarabel is an untyped C-extension

        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)
        nz = N * n + (N - 1) * m

        x0_arr, t0_arr, dt_arr, xf_val, z0 = parse_solver_initial_state(problem, bc, ws)
        problem = retarget_problem(problem, bc)

        z_op = operating_point_z(problem, self.operating_point)
        qp = quadratic_subproblem(problem, z_op, bc)
        P_triu, q_vec = qp.P, qp.q

        A_con, b_con, cones_con = _conic_rows(qp)
        A_bounds, b_bounds, cones_bounds = _extract_conic_bounds(problem, nz)

        n_con_rows = qp.A.shape[0]
        A_mat = sp.vstack([*A_con, *A_bounds]).tocsc()
        b_vec = np.concatenate([*b_con, *b_bounds])
        cones = [*cones_con, *cones_bounds]

        settings = settings_cls()
        settings.verbose = False
        if self.options:
            for k, v in self.options.items():
                if hasattr(settings, k):
                    setattr(settings, k, v)

        solver = solver_cls(P_triu, q_vec, A_mat, b_vec, cones, settings)
        res = solver.solve()

        status_str = str(res.status)
        success = res.status in {solver_status_cls.Solved, solver_status_cls.AlmostSolved}
        iter_count = int(getattr(res, "iterations", 0))

        Z_opt_np = (
            np.asarray(res.x, dtype=np.float64)
            if (res.x is not None and len(res.x) == nz)
            else np.asarray(z0, dtype=np.float64)
        )
        Z_opt_jax = jnp.asarray(Z_opt_np, dtype=jnp.float64)
        cost_val = float(eval_f(problem, Z_opt_jax, t0=t0_arr, dt=dt_arr))
        viol = compute_constraint_violation(problem, Z_opt_jax, x0_arr, t0=t0_arr, dt=dt_arr, xf=xf_val)

        X_opt, U_opt = _z_to_trajectory(Z_opt_jax, N, n, m)
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])

        opt_traj = Trajectory(
            X=X_opt,
            U=U_opt,
            t=t_opt,
            dt=dt_arr,
        )

        lam_out, mu_out = _normalise_duals(problem, np.asarray(res.z, dtype=np.float64), nz, n_con_rows)

        info_dict = {
            "status": status_str,
            "iterations": iter_count,
            "solve_time": getattr(res, "solve_time", 0.0),
            "r_prim": getattr(res, "r_prim", 0.0),
            "r_dual": getattr(res, "r_dual", 0.0),
            "z": res.z,
            "s": res.s,
        }

        return ClarabelResult(
            trajectory=opt_traj,
            success=success,
            status=status_str,
            message=status_str,
            cost=cost_val,
            Z=Z_opt_jax,
            info=info_dict,
            iterations=iter_count,
            constraint_violation=viol,
            lam=lam_out,
            mu=mu_out,
        )
