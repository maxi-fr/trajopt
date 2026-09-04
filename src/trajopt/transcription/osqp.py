import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import NegativeOrthant, SecondOrderCone, ZeroCone
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import (
    _z_to_trajectory,
    compute_constraint_violation,
    operating_point_z,
    parse_solver_initial_state,
)
from trajopt.transcription.result import warm_start_duals
from trajopt.transcription.subproblem import ConstraintBlock, quadratic_subproblem
from trajopt.transcription.transcription import (
    eval_f,
)

_SUCCESS_STATUS_VALS = {1, 2}  # 1: solved, 2: solved inaccurate
_EMPTY = np.zeros(0, dtype=np.float64)


class OSQPResult(NamedTuple):
    """Result of an OSQP trajectory optimization solve.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the solver converged to optimality or within tolerance.
    status : int
        OSQP integer status value code.
    message : str
        Solver return status message.
    cost : float
        Final objective value.
    Z : jax.Array
        Optimal flat primal vector of shape ``(N * n + (N - 1) * m,)``.
    info : dict[str, Any]
        Raw OSQP return info dictionary.
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
    status: int
    message: str
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    constraint_violation: float
    iterations: int = 0
    lam: np.ndarray = _EMPTY
    mu: np.ndarray = _EMPTY


def _reject_unsupported_cones(blocks: Sequence[ConstraintBlock]) -> None:
    """Reject the Cones OSQP's box form cannot express, before the QP is handed over."""
    for block in blocks:
        if isinstance(block.cone, SecondOrderCone):
            msg = (
                "OSQP does not support SecondOrderCone constraints. "
                "Use Clarabel for second-order cone constraints or Ipopt for nonlinear formulations."
            )
            raise TypeError(msg)
        if not isinstance(block.cone, (ZeroCone, NegativeOrthant)):
            msg = f"OSQP adapter does not support cone {type(block.cone).__name__}."
            raise TypeError(msg)


class BoxQP(NamedTuple):
    """One step's QP in OSQP's box form, with both matrices already canonicalized to csc.

    Parameters
    ----------
    P : sp.csc_matrix
        Upper triangle of the objective Hessian, of shape ``(nz, nz)``.
    q : np.ndarray
        Linear objective term of shape ``(nz,)``.
    A : sp.csc_matrix
        Constraint rows stacked over the identity bound rows, of shape ``(n_rows, nz)``.
    lower, upper : np.ndarray
        Box on ``A z`` of shape ``(n_rows,)``.
    """

    P: sp.csc_matrix
    q: np.ndarray
    A: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray


def _warm_start(
    solver: Any,  # noqa: ANN401 -- osqp.OSQP is an untyped C-extension
    problem: Problem,
    ws: WarmStart,
    qp: BoxQP,
    *,
    z0: jax.Array | None,
) -> None:
    """Seed OSQP with the previous primal and, when the WarmStart carries them, dual iterates.

    Both iterates are always written, falling back to zero, because a handle reused across
    receding-horizon steps still holds the previous solve's iterates where a fresh `setup` would
    have started from zero. Writing both puts a reused handle on the same iterates a rebuilt one
    would start from. OSQP's adapted penalty `rho` still carries over, so a reused handle reaches
    a different point inside the requested tolerance rather than the identical one.
    """
    solver.warm_start(x=np.zeros(qp.A.shape[1]) if z0 is None else np.asarray(z0, dtype=np.float64))

    lam0, mu0 = warm_start_duals(problem, ws)
    y0 = np.zeros(qp.A.shape[0]) if lam0 is None or mu0 is None else np.concatenate([lam0, mu0])
    solver.warm_start(y=y0)


@dataclass
class OSQPHandle:
    """A live OSQP solver plus the data its factorization was built from, cached on a `Program`.

    Parameters
    ----------
    solver : Any
        The C solver handle, already set up.
    options : dict[str, Any]
        Options the handle was set up with; a change to them invalidates it.
    shape : tuple[int, int]
        ``(n_rows, nz)`` of the constraint matrix the handle was set up with.
    P_indptr, P_indices : np.ndarray
        Canonical csc sparsity of the objective Hessian's upper triangle.
    A_indptr, A_indices : np.ndarray
        Canonical csc sparsity of the stacked constraint matrix.
    P_data, A_data : np.ndarray
        Numeric values last pushed into the handle, in that canonical order.
    regime : str
        Which of "setup", "matrix" or "vector" the last solve took. Diagnostic only.
    """

    solver: Any
    options: dict[str, Any]
    shape: tuple[int, int]
    P_indptr: np.ndarray
    P_indices: np.ndarray
    A_indptr: np.ndarray
    A_indices: np.ndarray
    P_data: np.ndarray
    A_data: np.ndarray
    regime: str

    def _same_pattern(self, P: sp.csc_matrix, A: sp.csc_matrix) -> bool:
        """Whether `P` and `A` have exactly this handle's dimensions and canonical sparsity."""
        return (
            A.shape == self.shape
            and np.array_equal(P.indptr, self.P_indptr)
            and np.array_equal(P.indices, self.P_indices)
            and np.array_equal(A.indptr, self.A_indptr)
            and np.array_equal(A.indices, self.A_indices)
        )

    def regime_for(self, P: sp.csc_matrix, A: sp.csc_matrix, options: dict[str, Any]) -> str:
        """Cheapest regime this handle can serve `P`, `A` and `options` in.

        "vector" reuses the factorization outright, "matrix" refactorizes in place, and "setup"
        means the handle cannot be reused at all. The choice is read off the matrices themselves,
        never off what the caller claims to have changed, so a caller that quietly perturbs a
        Hessian cannot get a stale factorization.
        """
        if options != self.options or not self._same_pattern(P, A):
            return "setup"
        if np.array_equal(P.data, self.P_data) and np.array_equal(A.data, self.A_data):
            return "vector"
        return "matrix"


def _canonical_csc(matrix: sp.csc_matrix | sp.csr_matrix | sp.coo_matrix) -> sp.csc_matrix:
    """Return `matrix` as csc with summed duplicates and sorted indices, so two forms compare structurally.

    Explicit zeros are deliberately kept: the QP's matrices are assembled from a fixed structural
    sparsity pattern, so keeping the stored zeros is what makes the pattern value-independent and
    the handle reusable across steps.
    """
    out = matrix.tocsc(copy=True)
    out.sum_duplicates()
    out.sort_indices()
    return out


def _apply(
    program: Program,
    qp: BoxQP,
    options: dict[str, Any],
    osqp_module: Any,  # noqa: ANN401 -- osqp is an untyped C-extension
) -> OSQPHandle:
    """Push this step's QP into the program's cached OSQP handle, rebuilding only as much as changed."""
    P, A = qp.P, qp.A
    handle: OSQPHandle | None = program.handles.get("osqp")
    regime = "setup" if handle is None else handle.regime_for(P, A, options)

    if handle is None or regime == "setup":
        solver = osqp_module.OSQP()
        solver.setup(P=P, q=qp.q, A=A, l=qp.lower, u=qp.upper, **options)
        handle = OSQPHandle(
            solver=solver,
            options=options,
            shape=A.shape,
            P_indptr=P.indptr.copy(),
            P_indices=P.indices.copy(),
            A_indptr=A.indptr.copy(),
            A_indices=A.indices.copy(),
            P_data=P.data.copy(),
            A_data=A.data.copy(),
            regime="setup",
        )
        program.handles["osqp"] = handle
        return handle

    if regime == "matrix":
        handle.solver.update(Px=P.data, Ax=A.data)
        handle.P_data = P.data.copy()
        handle.A_data = A.data.copy()
    handle.solver.update(q=qp.q, l=qp.lower, u=qp.upper)
    handle.regime = regime
    return handle


@dataclass(frozen=True)
class OSQP:
    """OSQP convex QP solver backend, expanding the problem about `operating_point`.

    OSQP is a convex quadratic programming solver, so this adapter hands it one convex
    approximation of the problem, expanded about ``operating_point`` (the origin by default):
    the dynamics linearized to :math:`x_{k+1} = A_k x_k + B_k u_k + d_k`, the constraints
    linearized, and the cost taken to second order. The approximation is exact — and the result
    is the true optimum — when the dynamics are affine, the cost quadratic and the constraints
    linear, whatever the operating point.

    Otherwise this is linearized MPC: the returned trajectory optimizes the approximation, not
    the nonlinear problem, and ``success`` reports the QP's convergence. ``constraint_violation``
    is measured against the true nonlinear dynamics and is the number that says how far the two
    have drifted apart; drive it down by re-solving with the operating point set to the previous
    solution.

    Second-order cone constraints (e.g. SecondOrderCone) are rejected — use Clarabel for those,
    or Ipopt to solve the nonlinear problem directly.

    Parameters
    ----------
    operating_point : Trajectory | jax.Array | None, optional
        Point about which the dynamics, constraints and cost are expanded, as a trajectory or a
        flat vector of shape ``(N * n + (N - 1) * m,)``. Defaults to None, meaning the origin.
    options : Mapping[str, Any], optional
        Native OSQP options (e.g. {"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 4000}).
        Defaults to empty.
    """

    operating_point: Trajectory | jax.Array | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> OSQPResult:
        """Solve the transcribed optimal control problem using OSQP, warning that it is one linearization.

        The warning is unconditional rather than gated on the problem being nonlinear: every
        caller handing this Backend a `Problem` gets one convex solve about the Operating
        Point, and whether that answers their problem is theirs to judge.

        The C solver lives on the `Program`, so a receding-horizon loop whose QP keeps its
        sparsity reuses the factorization instead of setting one up again -- see `_apply`.
        """
        problem = program.problem
        warnings.warn(
            "OSQP solves a single convex subproblem built about the Operating Point: the "
            "dynamics are linearized and the cost taken to second order once, so the result "
            "optimizes an approximation of a nonlinear problem, not the problem itself.",
            stacklevel=2,
        )

        try:
            import osqp  # noqa: PLC0415 -- osqp is an optional solver dependency
        except ImportError as e:
            msg = (
                "osqp is not installed. It is part of the `solvers` extra: install with "
                '`pip install "trajopt[solvers]"` or `uv add "trajopt[solvers]"`.'
            )
            raise ImportError(msg) from e

        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)
        nz = N * n + (N - 1) * m

        x0_arr, t0_arr, dt_arr, xf_val, z0 = parse_solver_initial_state(problem, bc, ws)
        problem = retarget_problem(problem, bc)

        z_op = operating_point_z(problem, self.operating_point)
        qp = quadratic_subproblem(problem, z_op, bc)
        _reject_unsupported_cones(qp.blocks)

        # Bounds ride as identity rows, so every dual OSQP returns is either a lam or a mu.
        n_con_rows = qp.A.shape[0]
        A_mat = _canonical_csc(sp.vstack([qp.A, sp.eye(nz, format="csr", dtype=np.float64)]))
        P_triu, q_vec = _canonical_csc(qp.P), qp.q
        l_vec = np.concatenate([qp.row_lower, qp.z_lower])
        u_vec = np.concatenate([qp.row_upper, qp.z_upper])

        solver_opts: dict[str, Any] = {"verbose": False}
        if self.options:
            solver_opts.update(self.options)

        box = BoxQP(P=P_triu, q=q_vec, A=A_mat, lower=l_vec, upper=u_vec)
        solver = _apply(program, box, solver_opts, osqp).solver
        _warm_start(solver, problem, ws, box, z0=z0)

        res = solver.solve()

        status_val = int(getattr(res.info, "status_val", -1))
        success = status_val in _SUCCESS_STATUS_VALS
        status_msg = str(getattr(res.info, "status", "unknown"))
        iter_count = int(getattr(res.info, "iter", 0))

        Z_opt_np = np.asarray(res.x, dtype=np.float64) if res.x is not None else np.asarray(z0, dtype=np.float64)
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

        y_out = np.asarray(res.y, dtype=np.float64) if res.y is not None else np.zeros(A_mat.shape[0])
        lam_out = y_out[:n_con_rows]
        mu_out = y_out[n_con_rows:]

        info_dict = {
            "status": status_msg,
            "status_val": status_val,
            "iter": iter_count,
            "obj_val": getattr(res.info, "obj_val", cost_val),
            "prim_res": getattr(res.info, "prim_res", 0.0),
            "dual_res": getattr(res.info, "dual_res", 0.0),
            "y": res.y,
        }

        return OSQPResult(
            trajectory=opt_traj,
            success=success,
            status=status_val,
            message=status_msg,
            cost=cost_val,
            Z=Z_opt_jax,
            info=info_dict,
            iterations=iter_count,
            constraint_violation=viol,
            lam=lam_out,
            mu=mu_out,
        )
