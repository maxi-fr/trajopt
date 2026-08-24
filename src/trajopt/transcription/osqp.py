from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import NegativeOrthant, SecondOrderCone, ZeroCone
from trajopt.constraints.bounds import BoundConstraint, ControlBound, StateBound
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import MPCState, Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import (
    _z_to_trajectory,
    build_linear_constraint_block,
    compute_constraint_violation,
    extract_quadratic_cost,
    operating_point_z,
    parse_solver_initial_state,
    primal_bounds,
)
from trajopt.transcription.result import blocked_to_canonical, warm_start_duals
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
    iterations : int, optional
        Number of solver iterations. Defaults to 0.
    constraint_violation : float, optional
        Maximum constraint violation across all constraints. Defaults to 0.0.
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
    iterations: int = 0
    constraint_violation: float = 0.0
    lam: np.ndarray = _EMPTY
    mu: np.ndarray = _EMPTY


def _extract_qp_dynamics(  # noqa: PLR0913 -- Dynamics extraction helper takes 8 parameters
    discrete_model: DiscreteDynamics,
    N: int,
    n: int,
    m: int,
    nz: int,
    *,
    x0_arr: jax.Array,
    t_stage: jax.Array,
    dt_arr: jax.Array,
    X_op: jax.Array,
    U_op: jax.Array,
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[np.ndarray]]:
    """Assemble initial condition and linear dynamics defect rows for OSQP."""
    A_rows: list[sp.spmatrix] = []
    l_vals: list[np.ndarray] = []
    u_vals: list[np.ndarray] = []

    # Initial condition: x_0 = x0_arr
    A_init = sp.lil_matrix((n, nz), dtype=np.float64)
    A_init[:, :n] = np.eye(n)
    A_rows.append(A_init.tocsr())
    x0_np = np.asarray(x0_arr, dtype=np.float64)
    l_vals.append(x0_np)
    u_vals.append(x0_np)

    for k in range(N - 1):
        tk = t_stage[k]
        dtk = dt_arr[k]
        x_op = X_op[k]
        u_op = U_op[k]
        Ak = np.asarray(discrete_model.state_jacobian(x_op, u_op, tk, dtk), dtype=np.float64)
        Bk = np.asarray(discrete_model.control_jacobian(x_op, u_op, tk, dtk), dtype=np.float64)
        f_op = np.asarray(discrete_model.discrete_dynamics(x_op, u_op, tk, dtk), dtype=np.float64)
        # f(x, u) ~ f(x_op, u_op) + A (x - x_op) + B (u - u_op), collected into the constant
        dk = f_op - Ak @ np.asarray(x_op, dtype=np.float64) - Bk @ np.asarray(u_op, dtype=np.float64)

        A_dyn_k = sp.lil_matrix((n, nz), dtype=np.float64)
        col_x_k = k * (n + m)
        col_u_k = col_x_k + n
        col_x_next = (k + 1) * (n + m)
        A_dyn_k[:, col_x_k : col_x_k + n] = -Ak
        A_dyn_k[:, col_u_k : col_u_k + m] = -Bk
        A_dyn_k[:, col_x_next : col_x_next + n] = np.eye(n)
        A_rows.append(A_dyn_k.tocsr())
        l_vals.append(dk)
        u_vals.append(dk)

    return A_rows, l_vals, u_vals


def _extract_qp_stage_constraints(  # noqa: PLR0913 -- Stage constraint extraction helper takes 8 arguments
    knot_evaluators: Sequence[Any],
    N: int,
    n: int,
    m: int,
    nz: int,
    *,
    t_stage: jax.Array,
    t_term: jax.Array,
    xf_val: jax.Array | None,
    z_op: jax.Array,
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[np.ndarray]]:
    """Assemble stage and terminal linear constraint rows for OSQP."""
    A_rows: list[sp.spmatrix] = []
    l_vals: list[np.ndarray] = []
    u_vals: list[np.ndarray] = []

    for k in range(N):
        if k >= len(knot_evaluators):
            continue
        ev = knot_evaluators[k]
        tk = t_stage[k] if k < N - 1 else t_term
        col_k = k * (n + m)
        is_term = k == N - 1
        z_op_k = z_op[col_k : col_k + (n if is_term else n + m)]

        for con in ev.constraints:
            if isinstance(con, (BoundConstraint, ControlBound, StateBound)):
                continue

            if isinstance(con.cone, SecondOrderCone):
                msg = (
                    "OSQP does not support SecondOrderCone constraints. "
                    "Use Clarabel for second-order cone constraints or Ipopt for nonlinear formulations."
                )
                raise TypeError(msg)

            dim_c = int(con.p)
            if dim_c == 0:
                continue

            A_c_block, val0_np = build_linear_constraint_block(
                con,
                n,
                m,
                tk=tk,
                is_term=is_term,
                xf_val=xf_val,
                z_op_k=z_op_k,
            )
            A_con = sp.lil_matrix((dim_c, nz), dtype=np.float64)
            A_con[:, col_k : col_k + A_c_block.shape[1]] = A_c_block
            A_rows.append(A_con.tocsr())

            if isinstance(con.cone, ZeroCone):
                l_vals.append(-val0_np)
                u_vals.append(-val0_np)
            elif isinstance(con.cone, NegativeOrthant):
                l_vals.append(np.full(dim_c, -np.inf, dtype=np.float64))
                u_vals.append(-val0_np)
            else:
                msg = (
                    f"OSQP adapter does not support cone {type(con.cone).__name__} on constraint {type(con).__name__}."
                )
                raise TypeError(msg)

    return A_rows, l_vals, u_vals


def _warm_start(  # noqa: PLR0913 -- seeding takes the solver, the problem, the state, and the row map
    solver: Any,  # noqa: ANN401 -- osqp.OSQP is an untyped C-extension
    problem: Problem,
    state: MPCState,
    *,
    z0: jax.Array | None,
    canonical_rows: np.ndarray,
    n_rows: int,
) -> None:
    """Seed OSQP with the previous primal and, when the MPCState carries them, dual iterates."""
    if z0 is not None:
        solver.warm_start(x=z0)

    lam0, mu0 = warm_start_duals(problem, state)
    if lam0 is None or mu0 is None:
        return

    y0 = np.empty(n_rows, dtype=np.float64)
    y0[canonical_rows] = lam0
    y0[len(canonical_rows) :] = mu0
    solver.warm_start(y=y0)


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

    def solve(self, problem: Problem, state: MPCState) -> OSQPResult:
        """Solve the transcribed optimal control problem using OSQP."""
        import osqp  # noqa: PLC0415 -- osqp is an optional solver dependency

        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)
        nz = N * n + (N - 1) * m

        x0_arr, t0_arr, dt_arr, xf_val, z0 = parse_solver_initial_state(state)

        t_stage = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr[:-1])])
        t_term = t0_arr + jnp.sum(dt_arr)

        z_op = operating_point_z(problem, self.operating_point)
        X_op, U_op = _z_to_trajectory(z_op, N, n, m)

        P_triu, q_vec = extract_quadratic_cost(
            problem,
            N,
            n,
            m,
            nz,
            t0_arr=t0_arr,
            dt_arr=dt_arr,
            xf_val=xf_val,
            z_op=z_op,
        )

        discrete_model = problem.model
        A_dyn, l_dyn, u_dyn = _extract_qp_dynamics(
            discrete_model,
            N,
            n,
            m,
            nz,
            x0_arr=x0_arr,
            t_stage=t_stage,
            dt_arr=dt_arr,
            X_op=X_op,
            U_op=U_op,
        )

        knot_evaluators = problem.constraints.knot_evaluators if problem.constraints is not None else ()
        A_con, l_con, u_con = _extract_qp_stage_constraints(
            knot_evaluators,
            N,
            n,
            m,
            nz,
            t_stage=t_stage,
            t_term=t_term,
            xf_val=xf_val,
            z_op=z_op,
        )

        zL, zU = primal_bounds(problem)
        A_bounds = sp.eye(nz, format="csr", dtype=np.float64)

        A_mat = sp.vstack([*A_dyn, *A_con, A_bounds]).tocsc()
        canonical_rows = blocked_to_canonical(problem)
        l_vec = np.concatenate([*l_dyn, *l_con, zL])
        u_vec = np.concatenate([*u_dyn, *u_con, zU])

        solver = osqp.OSQP()
        solver_opts: dict[str, Any] = {"verbose": False}
        if self.options:
            solver_opts.update(self.options)

        solver.setup(P=P_triu, q=q_vec, A=A_mat, l=l_vec, u=u_vec, **solver_opts)
        _warm_start(solver, problem, state, z0=z0, canonical_rows=canonical_rows, n_rows=A_mat.shape[0])

        res = solver.solve()

        status_val = int(getattr(res.info, "status_val", -1))
        success = status_val in _SUCCESS_STATUS_VALS
        status_msg = str(getattr(res.info, "status", "unknown"))
        iter_count = int(getattr(res.info, "iter", 0))

        Z_opt_np = np.asarray(res.x, dtype=np.float64) if res.x is not None else np.asarray(z0, dtype=np.float64)
        Z_opt_jax = jnp.asarray(Z_opt_np, dtype=jnp.float64)
        cost_val = float(eval_f(problem, Z_opt_jax, t0=t0_arr, dt=dt_arr, xf=xf_val))
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
        lam_out = y_out[canonical_rows]
        mu_out = y_out[len(canonical_rows) :]

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
