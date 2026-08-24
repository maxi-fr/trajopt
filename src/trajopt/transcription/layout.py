from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import IdentityCone, NegativeOrthant, PositiveOrthant

if TYPE_CHECKING:
    from trajopt.constraints.base import Constraint
    from trajopt.constraints.constraint_list import BuiltKnotConstraint
    from trajopt.problem import MPCState, Problem
    from trajopt.trajectory import Trajectory


def _trajectory_to_z(X: jax.Array, U: jax.Array) -> jax.Array:
    """Interleave states and controls into the flat NLP primal vector Z.

    Parameters
    ----------
    X : jax.Array
        Stacked state trajectory of shape (N, n).
    U : jax.Array
        Stacked control trajectory of shape (N-1, m).

    Returns
    -------
    jax.Array
        Flat primal vector Z of shape (N * n + (N - 1) * m,).
    """
    Z_stages = jnp.concatenate([X[:-1], U], axis=1).reshape(-1)
    return jnp.concatenate([Z_stages, X[-1]])


def _z_to_trajectory(
    Z: jax.Array,
    N: int,
    n: int,
    m: int,
) -> tuple[jax.Array, jax.Array]:
    """Recover state and control trajectories from the flat NLP primal vector Z.

    Parameters
    ----------
    Z : jax.Array
        Flat primal vector of shape (N * n + (N - 1) * m,).
    N : int
        Horizon length in knot points.
    n : int
        State dimension.
    m : int
        Control dimension.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        State trajectory X of shape (N, n) and control trajectory U of shape (N-1, m).
    """
    stage_len = (N - 1) * (n + m)
    stage_part = Z[:stage_len].reshape((N - 1, n + m))
    X_stage = stage_part[:, :n]
    U = stage_part[:, n:]
    X_term = Z[stage_len : stage_len + n]
    X = jnp.vstack([X_stage, X_term])
    return X, U


def primal_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Extract solver variable limits zL <= Z <= zU from the problem's box bounds.

    Parameters
    ----------
    problem : Problem
        Problem instance containing constraints and horizon dimensions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower bounds zL and upper bounds zU of shape (N * n + (N - 1) * m,).
    """
    xL, xU, uL, uU = problem.constraints.primal_bounds()

    zL_stages = np.concatenate([xL[:-1], uL], axis=1).reshape(-1)
    zL = np.concatenate([zL_stages, xL[-1]])

    zU_stages = np.concatenate([xU[:-1], uU], axis=1).reshape(-1)
    zU = np.concatenate([zU_stages, xU[-1]])

    return zL.astype(np.float64), zU.astype(np.float64)


def _cone_bounds(cone: object, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute (gL, gU) bounds corresponding to a cone constraint."""
    if isinstance(cone, NegativeOrthant):
        return np.full(p, -np.inf, dtype=np.float64), np.zeros(p, dtype=np.float64)
    if isinstance(cone, PositiveOrthant):
        return np.zeros(p, dtype=np.float64), np.full(p, np.inf, dtype=np.float64)
    if isinstance(cone, IdentityCone):
        return np.full(p, -np.inf, dtype=np.float64), np.full(p, np.inf, dtype=np.float64)
    # Default equality (ZeroCone or general equality)
    return np.zeros(p, dtype=np.float64), np.zeros(p, dtype=np.float64)


def _evaluator_bounds(evaluator: BuiltKnotConstraint) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute (gL, gU) arrays for all constraints in a knot evaluator."""
    gL_list: list[np.ndarray] = []
    gU_list: list[np.ndarray] = []
    for c in evaluator.constraints:
        lo, hi = _cone_bounds(c.cone, int(c.p))
        gL_list.append(lo)
        gU_list.append(hi)
    return gL_list, gU_list


def constraint_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper bounds gL <= c(Z) <= gU for the transcribed constraint vector.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, constraints, and horizon.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Constraint lower and upper bounds of shape (P,).
    """
    N = int(problem.N)
    n = int(problem.model.n)

    gL_list: list[np.ndarray] = []
    gU_list: list[np.ndarray] = []

    # 1. Initial state condition: x0 - x_init = 0
    gL_list.append(np.zeros(n, dtype=np.float64))
    gU_list.append(np.zeros(n, dtype=np.float64))

    knot_evaluators = problem.constraints.knot_evaluators if problem.constraints is not None else ()

    for k in range(N - 1):
        # 2a. Dynamics defect k: x_{k+1} - f_d(x_k, u_k) = 0
        gL_list.append(np.zeros(n, dtype=np.float64))
        gU_list.append(np.zeros(n, dtype=np.float64))

        # 2b. Stage constraints at knot k
        if k < len(knot_evaluators):
            lo_k, hi_k = _evaluator_bounds(knot_evaluators[k])
            gL_list.extend(lo_k)
            gU_list.extend(hi_k)

    # 3. Terminal constraints at knot N - 1
    if len(knot_evaluators) > N - 1:
        lo_term, hi_term = _evaluator_bounds(knot_evaluators[N - 1])
        gL_list.extend(lo_term)
        gU_list.extend(hi_term)

    gL = np.concatenate(gL_list) if gL_list else np.empty(0, dtype=np.float64)
    gU = np.concatenate(gU_list) if gU_list else np.empty(0, dtype=np.float64)
    return gL, gU


def _eval_stage_violations(  # noqa: PLR0913 -- Stage evaluation helper takes 6 parameters
    knot_evaluators: tuple[BuiltKnotConstraint, ...],
    X: jax.Array,
    U: jax.Array,
    *,
    t_stage: jax.Array,
    t_term: jax.Array,
    xf_jax: jax.Array | None,
) -> float:
    """Compute maximum violation across all stage and terminal constraint objects."""
    from trajopt.constraints.linear import GoalConstraint  # noqa: PLC0415 -- type check for goal constraint

    N = len(X)
    max_viol = 0.0

    for k in range(N - 1):
        if k < len(knot_evaluators):
            ev = knot_evaluators[k]
            for con in ev.constraints:
                val = (
                    con.evaluate(X[k], U[k], t_stage[k], xf=xf_jax)
                    if isinstance(con, GoalConstraint) and xf_jax is not None
                    else con.evaluate(X[k], U[k], t_stage[k])
                )
                proj = con.cone.project(val)
                diff = float(np.max(np.abs(np.asarray(val, dtype=np.float64) - np.asarray(proj, dtype=np.float64))))
                max_viol = max(max_viol, diff)

    if len(knot_evaluators) > N - 1:
        ev = knot_evaluators[N - 1]
        for con in ev.constraints:
            val = (
                con.evaluate(X[-1], None, t_term, xf=xf_jax)
                if isinstance(con, GoalConstraint) and xf_jax is not None
                else con.evaluate(X[-1], None, t_term)
            )
            proj = con.cone.project(val)
            diff = float(np.max(np.abs(np.asarray(val, dtype=np.float64) - np.asarray(proj, dtype=np.float64))))
            max_viol = max(max_viol, diff)

    return max_viol


def compute_constraint_violation(  # noqa: PLR0913 -- Metric calculation takes 6 arguments
    problem: Problem,
    Z: jax.Array | np.ndarray,
    x0: jax.Array | np.ndarray,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | np.ndarray | None = None,
) -> float:
    """Compute maximum constraint violation across all transcribed constraints and bounds.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, constraints, and horizon dimensions.
    Z : jax.Array | np.ndarray
        Flat primal trajectory vector of shape ``(N * n + (N - 1) * m,)``.
    x0 : jax.Array | np.ndarray
        Initial state condition of shape ``(n,)``.
    t0 : float | jax.Array, optional
        Initial timestamp scalar. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration (scalar or array of length N-1). Defaults to 0.05.
    xf : jax.Array | np.ndarray | None, optional
        Goal state vector of shape ``(n,)``. Defaults to None.

    Returns
    -------
    float
        Maximum constraint violation scalar.
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    Z_arr = jnp.asarray(Z, dtype=jnp.float64)
    X, U = _z_to_trajectory(Z_arr, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    discrete_model = problem.model
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    # 1. Primal variable bounds
    zL, zU = primal_bounds(problem)
    Z_np = np.asarray(Z_arr, dtype=np.float64)
    viol_lb = np.maximum(0.0, zL - Z_np)
    viol_ub = np.maximum(0.0, Z_np - zU)
    max_primal = max(float(np.max(viol_lb)), float(np.max(viol_ub))) if len(viol_lb) > 0 else 0.0

    # 2. Initial state condition
    viol_init = float(np.max(np.abs(np.asarray(X[0], dtype=np.float64) - np.asarray(x0, dtype=np.float64))))

    # 3. Dynamics defects
    def step_defect(
        xk: jax.Array,
        uk: jax.Array,
        x_next: jax.Array,
        tk: jax.Array,
        dtk: jax.Array,
    ) -> jax.Array:
        return x_next - discrete_model.discrete_dynamics(xk, uk, tk, dtk)

    dyn_defects = jax.vmap(step_defect)(X[:-1], U, X[1:], t_stage, dt_arr)
    viol_dyn = float(np.max(np.abs(np.asarray(dyn_defects, dtype=np.float64))))

    # 4. Stage and terminal constraints
    xf_jax = None if xf is None else jnp.asarray(xf, dtype=jnp.float64)
    viol_stage = _eval_stage_violations(
        knot_evaluators,
        X,
        U,
        t_stage=t_stage,
        t_term=t_term,
        xf_jax=xf_jax,
    )

    return max(max_primal, viol_init, viol_dyn, viol_stage)


def parse_solver_initial_state(
    state: MPCState,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    """Extract standard JAX array representations of `state`'s boundary and warm-start data."""
    x0_arr = jnp.asarray(state.x0, dtype=jnp.float64)
    t0_arr = jnp.asarray(state.t0, dtype=jnp.float64)
    dt_arr = jnp.asarray(state.dt, dtype=jnp.float64)
    xf_val = jnp.asarray(state.xf, dtype=jnp.float64) if state.xf is not None else None
    z0 = state.Z

    return x0_arr, t0_arr, dt_arr, xf_val, z0


def operating_point_z(problem: Problem, operating_point: Trajectory | jax.Array | None) -> jax.Array:
    """Flat expansion point z_op of shape (N * n + (N - 1) * m,), the origin when None.

    Parameters
    ----------
    problem : Problem
        Problem instance supplying the horizon and dimensions.
    operating_point : Trajectory | jax.Array | None
        Point about which the QP adapters expand the problem, as a trajectory or a flat vector.
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    nz = N * n + (N - 1) * m

    from trajopt.trajectory import Trajectory as _Trajectory  # noqa: PLC0415 -- avoid circular import

    if operating_point is None:
        return jnp.zeros(nz, dtype=jnp.float64)
    if isinstance(operating_point, _Trajectory):
        return _trajectory_to_z(operating_point.X, operating_point.U).astype(jnp.float64)

    z_op = jnp.asarray(operating_point, dtype=jnp.float64)
    if z_op.shape != (nz,):
        msg = f"Operating point has shape {z_op.shape}, expected ({nz},) for N={N}, n={n}, m={m}."
        raise ValueError(msg)
    return z_op


def extract_quadratic_cost(  # noqa: PLR0913 -- Cost extraction helper takes 8 parameters
    problem: Problem,
    N: int,
    n: int,
    m: int,
    nz: int,
    *,
    t0_arr: jax.Array,
    dt_arr: jax.Array,
    xf_val: jax.Array | None,
    z_op: jax.Array,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Extract upper-triangular Hessian P_triu and linear term q of the expansion about z_op."""
    from trajopt.transcription.sparsity import hessian_sparsity_pattern  # noqa: PLC0415 -- avoid circular import
    from trajopt.transcription.transcription import (  # noqa: PLC0415 -- avoid circular import
        eval_grad_f,
        eval_h,
    )

    g_jax = eval_grad_f(problem, z_op, t0=t0_arr, dt=dt_arr, xf=xf_val)

    h_rows, h_cols = hessian_sparsity_pattern(N, n, m)
    h_vals_jax = eval_h(problem, z_op, t0=t0_arr, dt=dt_arr, xf=xf_val)
    h_vals = np.asarray(h_vals_jax, dtype=np.float64)

    H_full = sp.coo_matrix((h_vals, (h_rows, h_cols)), shape=(nz, nz), dtype=np.float64).tocsc()
    P_triu = sp.triu(H_full, format="csc")

    # 0.5 (z - z_op)' H (z - z_op) + g'(z - z_op) in z, dropping the constant
    q_vec = np.asarray(g_jax, dtype=np.float64) - H_full @ np.asarray(z_op, dtype=np.float64)
    return P_triu, q_vec


def build_linear_constraint_block(  # noqa: PLR0913 -- Constraint block builder takes 7 arguments
    con: Constraint,
    n: int,
    m: int,
    *,
    tk: jax.Array,
    is_term: bool,
    xf_val: jax.Array | None,
    z_op_k: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Jacobian block and affine offset of a constraint linearized about z_op_k.

    Parameters
    ----------
    z_op_k : jax.Array
        Knot slice of the operating point, of shape (n + m,) at a stage knot and (n,) at the
        terminal knot, about which the constraint is expanded.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Jacobian block J of shape (p, n + m) or (p, n), and the affine offset
        c(z_op) - J z_op of shape (p,), so that c(z) is approximated by J z + offset.
    """
    from trajopt.constraints.linear import GoalConstraint  # noqa: PLC0415 -- type check for goal constraint

    x_op = z_op_k[:n]
    if is_term:
        jx, _ = con.jacobian(x_op, None, tk)
        val_op = (
            con.evaluate(x_op, None, tk, xf=xf_val)
            if isinstance(con, GoalConstraint) and xf_val is not None
            else con.evaluate(x_op, None, tk)
        )
        A_c_block = np.asarray(jx, dtype=np.float64)
    else:
        u_op = z_op_k[n : n + m]
        jx, ju = con.jacobian(x_op, u_op, tk)
        val_op = (
            con.evaluate(x_op, u_op, tk, xf=xf_val)
            if isinstance(con, GoalConstraint) and xf_val is not None
            else con.evaluate(x_op, u_op, tk)
        )
        A_c_block = np.hstack([np.asarray(jx, dtype=np.float64), np.asarray(ju, dtype=np.float64)])

    offset = np.asarray(val_op, dtype=np.float64) - A_c_block @ np.asarray(
        z_op_k[: A_c_block.shape[1]], dtype=np.float64
    )
    return A_c_block, offset
