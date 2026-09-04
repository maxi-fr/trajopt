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


def _warm_start(
    solver: Any,  # noqa: ANN401 -- osqp.OSQP is an untyped C-extension
    problem: Problem,
    ws: WarmStart,
    *,
    z0: jax.Array | None,
) -> None:
    """Seed OSQP with the previous primal and, when the WarmStart carries them, dual iterates."""
    if z0 is not None:
        solver.warm_start(x=z0)

    lam0, mu0 = warm_start_duals(problem, ws)
    if lam0 is None or mu0 is None:
        return

    solver.warm_start(y=np.concatenate([lam0, mu0]))


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
        A_mat = sp.vstack([qp.A, sp.eye(nz, format="csr", dtype=np.float64)]).tocsc()
        P_triu, q_vec = qp.P, qp.q
        l_vec = np.concatenate([qp.row_lower, qp.z_lower])
        u_vec = np.concatenate([qp.row_upper, qp.z_upper])

        solver = osqp.OSQP()
        solver_opts: dict[str, Any] = {"verbose": False}
        if self.options:
            solver_opts.update(self.options)

        solver.setup(P=P_triu, q=q_vec, A=A_mat, l=l_vec, u=u_vec, **solver_opts)
        _warm_start(solver, problem, ws, z0=z0)

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
