import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone
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
from trajopt.transcription.result import blocked_to_canonical
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


def _extract_conic_dynamics(  # noqa: PLR0913 -- Conic dynamics extraction helper takes 8 parameters
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
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[object]]:
    """Assemble initial condition and dynamics defect rows with ZeroCones for Clarabel."""
    A_rows: list[sp.spmatrix] = []
    b_vals: list[np.ndarray] = []
    cones: list[object] = []

    # Initial condition: x_0 = x0_arr => I x_0 + s = x0_arr, s in ZeroConeT(n)
    A_init = sp.lil_matrix((n, nz), dtype=np.float64)
    A_init[:, :n] = np.eye(n)
    A_rows.append(A_init.tocsr())
    x0_np = np.asarray(x0_arr, dtype=np.float64)
    b_vals.append(x0_np)
    cones.append(_make_cone("ZeroConeT", n))

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
        b_vals.append(dk)
        cones.append(_make_cone("ZeroConeT", n))

    return A_rows, b_vals, cones


def _extract_conic_stage_constraints(  # noqa: PLR0913 -- Stage constraint extraction helper takes 8 arguments
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
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[object]]:
    """Assemble stage, terminal, and second-order cone constraints for Clarabel."""
    A_rows: list[sp.spmatrix] = []
    b_vals: list[np.ndarray] = []
    cones: list[object] = []

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

            if isinstance(con.cone, SecondOrderCone):
                dim_y = dim_c - 1
                A_soc = sp.lil_matrix((dim_c, nz), dtype=np.float64)
                # s_0 = b_0 - A_soc[0] * z = s(z)
                A_soc[0, col_k : col_k + A_c_block.shape[1]] = -A_c_block[-1]
                # s_{1:D} = b_{1:D} - A_soc[1:D] * z = v(z)
                A_soc[1:dim_c, col_k : col_k + A_c_block.shape[1]] = -A_c_block[:dim_y]

                b_soc = np.zeros(dim_c, dtype=np.float64)
                b_soc[0] = float(val0_np[-1])
                b_soc[1:dim_c] = val0_np[:dim_y]

                A_rows.append(A_soc.tocsr())
                b_vals.append(b_soc)
                cones.append(_make_cone("SecondOrderConeT", dim_c))
            elif isinstance(con.cone, ZeroCone):
                A_con = sp.lil_matrix((dim_c, nz), dtype=np.float64)
                A_con[:, col_k : col_k + A_c_block.shape[1]] = A_c_block
                A_rows.append(A_con.tocsr())
                b_vals.append(-val0_np)
                cones.append(_make_cone("ZeroConeT", dim_c))
            elif isinstance(con.cone, PositiveOrthant):
                A_con = sp.lil_matrix((dim_c, nz), dtype=np.float64)
                A_con[:, col_k : col_k + A_c_block.shape[1]] = -A_c_block
                A_rows.append(A_con.tocsr())
                b_vals.append(val0_np)
                cones.append(_make_cone("NonnegativeConeT", dim_c))
            elif isinstance(con.cone, NegativeOrthant):
                A_con = sp.lil_matrix((dim_c, nz), dtype=np.float64)
                A_con[:, col_k : col_k + A_c_block.shape[1]] = A_c_block
                A_rows.append(A_con.tocsr())
                b_vals.append(-val0_np)
                cones.append(_make_cone("NonnegativeConeT", dim_c))
            else:
                msg = (
                    f"Clarabel adapter does not support cone {type(con.cone).__name__} "
                    f"on constraint {type(con).__name__}."
                )
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


def _normalise_duals(problem: Problem, z_dual: np.ndarray, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Split Clarabel's cone duals into canonical constraint duals and signed bound duals.

    Clarabel keeps every cone dual non-negative and posts the two sides of a box as separate
    NonnegativeCone blocks, upper first. Recombining them as ``upper - lower`` recovers the
    signed convention the other backends report.
    """
    canonical_rows = blocked_to_canonical(problem)
    lam = z_dual[canonical_rows]

    zL, zU = primal_bounds(problem)
    ub_indices = np.where(np.isfinite(zU))[0]
    lb_indices = np.where(np.isfinite(zL))[0]

    mu = np.zeros(nz, dtype=np.float64)
    bound_base = len(canonical_rows)
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

    def solve(self, problem: Problem, state: MPCState) -> ClarabelResult:
        """Solve the transcribed optimal control problem using Clarabel, warning that it is one linearization.

        The warning is unconditional rather than gated on the problem being nonlinear: every
        caller handing this Backend a `Problem` gets one convex solve about the Operating
        Point, and whether that answers their problem is theirs to judge.
        """
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
        A_dyn, b_dyn, cones_dyn = _extract_conic_dynamics(
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
        A_con, b_con, cones_con = _extract_conic_stage_constraints(
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

        A_bounds, b_bounds, cones_bounds = _extract_conic_bounds(problem, nz)

        A_mat = sp.vstack([*A_dyn, *A_con, *A_bounds]).tocsc()
        b_vec = np.concatenate([*b_dyn, *b_con, *b_bounds])
        cones = [*cones_dyn, *cones_con, *cones_bounds]

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

        lam_out, mu_out = _normalise_duals(problem, np.asarray(res.z, dtype=np.float64), nz)

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
