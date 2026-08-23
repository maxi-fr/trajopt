from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import PositiveOrthant, SecondOrderCone, ZeroCone
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import MPCState, Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import (
    build_linear_constraint_block,
    compute_constraint_violation,
    extract_quadratic_cost,
    parse_solver_initial_state,
    primal_bounds,
    z_to_trajectory,
)
from trajopt.transcription.transcription import (
    _extract_discrete_model,
    eval_f,
)


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
    iterations : int, optional
        Number of solver iterations. Defaults to 0.
    constraint_violation : float, optional
        Maximum constraint violation across all constraints. Defaults to 0.0.
    """

    trajectory: Trajectory
    success: bool
    status: str
    message: str
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    iterations: int = 0
    constraint_violation: float = 0.0


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
        Ak = np.asarray(
            discrete_model.state_jacobian(
                jnp.zeros(n, dtype=jnp.float64),
                jnp.zeros(m, dtype=jnp.float64),
                tk,
                dtk,
            ),
            dtype=np.float64,
        )
        Bk = np.asarray(
            discrete_model.control_jacobian(
                jnp.zeros(n, dtype=jnp.float64),
                jnp.zeros(m, dtype=jnp.float64),
                tk,
                dtk,
            ),
            dtype=np.float64,
        )
        dk = np.asarray(
            discrete_model.discrete_dynamics(
                jnp.zeros(n, dtype=jnp.float64),
                jnp.zeros(m, dtype=jnp.float64),
                tk,
                dtk,
            ),
            dtype=np.float64,
        )

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

        for con in ev.constraints:
            from trajopt.constraints.bounds import (  # noqa: PLC0415 -- type check for box bounds
                BoundConstraint,
                ControlBound,
                StateBound,
            )

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
            else:
                A_con = sp.lil_matrix((dim_c, nz), dtype=np.float64)
                A_con[:, col_k : col_k + A_c_block.shape[1]] = A_c_block
                A_rows.append(A_con.tocsr())
                b_vals.append(-val0_np)
                cones.append(_make_cone("NonnegativeConeT", dim_c))

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


def solve_clarabel(  # noqa: PLR0913 -- solver configuration takes 8 arguments
    problem: Problem,
    x0: jax.Array | MPCState,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    initial_trajectory: Trajectory | None = None,
    initial_z: jax.Array | None = None,
    xf: jax.Array | None = None,
    options: Mapping[str, Any] | None = None,
) -> ClarabelResult:
    """Solve the transcribed optimal control problem using Clarabel.

    Clarabel is a conic interior-point solver. It natively accepts convex quadratic objectives,
    affine dynamics, linear equality/inequality constraints, and second-order cone constraints.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, objective, constraints, and horizon.
    x0 : jax.Array | MPCState
        Initial state condition of shape ``(n,)`` or an MPCState instance.
    t0 : float | jax.Array, optional
        Initial timestamp scalar. Defaults to 0.0.
    dt : float | jax.Array, optional
        Step duration (scalar or array of length N-1). Defaults to 0.05.
    initial_trajectory : Trajectory | None, optional
        Initial trajectory guess. Defaults to repeating x0 with zero controls.
    initial_z : jax.Array | None, optional
        Flat initial guess vector of shape ``(N * n + (N - 1) * m,)``.
    xf : jax.Array | None, optional
        Goal state vector of shape ``(n,)``. Defaults to None.
    options : Mapping[str, Any] | None, optional
        Solver options passed to Clarabel DefaultSettings (e.g. ``{"tol_gap_abs": 1e-8, "max_iter": 200}``).

    Returns
    -------
    ClarabelResult
        Optimization result including optimal trajectory, convergence flag, status, and cost.
    """
    import clarabel  # noqa: PLC0415 -- clarabel is an optional solver dependency

    settings_cls: Any = getattr(clarabel, "DefaultSettings")  # noqa: B009 -- clarabel is an untyped C-extension
    solver_cls: Any = getattr(clarabel, "DefaultSolver")  # noqa: B009 -- clarabel is an untyped C-extension
    solver_status_cls: Any = getattr(clarabel, "SolverStatus")  # noqa: B009 -- clarabel is an untyped C-extension

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    nz = N * n + (N - 1) * m

    x0_arr, t0_arr, dt_arr, xf_val, z0 = parse_solver_initial_state(
        problem,
        x0,
        t0=t0,
        dt=dt,
        initial_trajectory=initial_trajectory,
        initial_z=initial_z,
        xf=xf,
    )

    t_stage = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr[:-1])])
    t_term = t0_arr + jnp.sum(dt_arr)

    P_triu, q_vec = extract_quadratic_cost(
        problem,
        N,
        n,
        m,
        nz,
        t0_arr=t0_arr,
        dt_arr=dt_arr,
        xf_val=xf_val,
    )

    discrete_model = _extract_discrete_model(problem)
    A_dyn, b_dyn, cones_dyn = _extract_conic_dynamics(
        discrete_model,
        N,
        n,
        m,
        nz,
        x0_arr=x0_arr,
        t_stage=t_stage,
        dt_arr=dt_arr,
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
    )

    A_bounds, b_bounds, cones_bounds = _extract_conic_bounds(problem, nz)

    A_mat = sp.vstack([*A_dyn, *A_con, *A_bounds]).tocsc()
    b_vec = np.concatenate([*b_dyn, *b_con, *b_bounds])
    cones = [*cones_dyn, *cones_con, *cones_bounds]

    settings = settings_cls()
    settings.verbose = False
    if options:
        for k, v in options.items():
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

    X_opt, U_opt = z_to_trajectory(Z_opt_jax, N, n, m)
    t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])

    opt_traj = Trajectory(
        X=X_opt,
        U=U_opt,
        t=t_opt,
        dt=dt_arr,
    )

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
    )
