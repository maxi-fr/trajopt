from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import IdentityCone, NegativeOrthant, PositiveOrthant, ZeroCone

if TYPE_CHECKING:
    from trajopt.constraints.constraint_list import BuiltKnotConstraint
    from trajopt.problem import BoundaryConditions, Problem
    from trajopt.program import WarmStart
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


def _trajectory_to_nlp_z(X: jax.Array, U_blk: jax.Array, b: int) -> jax.Array:
    """Interleave states and block controls into the flat NLP primal vector z_nlp.

    Parameters
    ----------
    X : jax.Array
        Stacked state trajectory of shape (N, n).
    U_blk : jax.Array
        Stacked block control trajectory of shape (B, m) where B = (N - 1) // b.
    b : int
        Block size in native steps.

    Returns
    -------
    jax.Array
        Flat NLP primal vector z_nlp of shape (N * n + B * m,).
    """
    if b == 1:
        return _trajectory_to_z(X, U_blk)
    n = X.shape[1]
    B = U_blk.shape[0]
    X_blocks = X[:-1].reshape((B, b * n))
    stages = jnp.concatenate([X_blocks, U_blk], axis=1).reshape(-1)
    return jnp.concatenate([stages, X[-1]])


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
        Horizon length in Knot Points.
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


def _nlp_z_to_trajectory(
    z: jax.Array,
    N: int,
    n: int,
    m: int,
    b: int,
) -> tuple[jax.Array, jax.Array]:
    """Recover state trajectory and block controls from the flat NLP primal vector z_nlp.

    Parameters
    ----------
    z : jax.Array
        Flat NLP primal vector of shape (N * n + B * m,).
    N : int
        Horizon length in Knot Points.
    n : int
        State dimension.
    m : int
        Control dimension.
    b : int
        Block size in native steps.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        State trajectory X of shape (N, n) and block controls U_blk of shape (B, m).
    """
    if b == 1:
        return _z_to_trajectory(z, N, n, m)
    H = N - 1
    B = H // b
    d_blk = b * n + m
    stage_len = B * d_blk
    stage_part = z[:stage_len].reshape((B, d_blk))
    X_stage = stage_part[:, : b * n].reshape((H, n))
    U_blk = stage_part[:, b * n :]
    X_term = z[stage_len : stage_len + n]
    X = jnp.vstack([X_stage, X_term])
    return X, U_blk


def reduce_controls(U: jax.Array | np.ndarray, b: int) -> jax.Array | np.ndarray:
    """Reduce canonical controls of shape (H, m) to block controls of shape (B, m) via first-step sampling."""
    return U[::b]


def expand_controls(U_blk: jax.Array, b: int) -> jax.Array:
    """Expand block controls of shape (B, m) to native controls of shape (B * b, m)."""
    return jnp.repeat(U_blk, b, axis=0)


def canonical_to_nlp_col(c: int, N: int, n: int, m: int, b: int) -> int:
    """Map canonical Primal Vector column index c to reduced NLP decision vector column index."""
    if b == 1:
        return c
    stage_part_len = (N - 1) * (n + m)
    d_blk = b * n + m
    B = (N - 1) // b
    if c >= stage_part_len:
        return B * d_blk + (c - stage_part_len)
    stage_idx = c // (n + m)
    rem = c % (n + m)
    if rem < n:
        return (stage_idx // b) * d_blk + (stage_idx % b) * n + rem
    return (stage_idx // b) * d_blk + b * n + (rem - n)


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
    """Compute (gL, gU) bounds corresponding to an interval Cone constraint."""
    if isinstance(cone, NegativeOrthant):
        return np.full(p, -np.inf, dtype=np.float64), np.zeros(p, dtype=np.float64)
    if isinstance(cone, PositiveOrthant):
        return np.zeros(p, dtype=np.float64), np.full(p, np.inf, dtype=np.float64)
    if isinstance(cone, IdentityCone):
        return np.full(p, -np.inf, dtype=np.float64), np.full(p, np.inf, dtype=np.float64)
    if isinstance(cone, ZeroCone):
        return np.zeros(p, dtype=np.float64), np.zeros(p, dtype=np.float64)
    msg = f"Cannot express non-interval cone {type(cone).__name__} as bounds."
    raise ValueError(msg)


def blocked_control_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Compute and validate intersected per-block control bounds of shape (B, m).

    Parameters
    ----------
    problem : Problem
        Problem instance specifying Horizon, controls, block size, and binary coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Block lower bounds and upper bounds of shape (B, m) where B = (N - 1) // b.

    Raises
    ------
    ValueError
        If any block lower bound exceeds its upper bound, or if a declared binary control
        coordinate's intersected block interval contains neither 0.0 nor 1.0.
    """
    N = int(problem.N)
    m = int(problem.model.m)
    H = N - 1
    b = int(problem.control_block_steps)
    B = H // b

    _, _, uL_orig, uU_orig = problem.constraints.primal_bounds()
    uL = np.array(uL_orig, dtype=np.float64, copy=True)
    uU = np.array(uU_orig, dtype=np.float64, copy=True)

    if problem.binary_control_indices:
        binary = list(problem.binary_control_indices)
        uL[:, binary] = np.maximum(uL[:, binary], 0.0)
        uU[:, binary] = np.minimum(uU[:, binary], 1.0)
        if np.any(uL[:, binary] > uU[:, binary]):
            msg = "Binary control bounds contradict existing control bounds."
            raise ValueError(msg)

    uL_blocks = uL.reshape(B, b, m)
    uU_blocks = uU.reshape(B, b, m)
    block_uL = np.max(uL_blocks, axis=1)
    block_uU = np.min(uU_blocks, axis=1)

    if np.any(block_uL > block_uU):
        msg = "Block control lower bound exceeds upper bound."
        raise ValueError(msg)

    if problem.binary_control_indices:
        for j in problem.binary_control_indices:
            for k in range(B):
                lo = block_uL[k, j]
                hi = block_uU[k, j]
                has_zero = lo <= 0.0 <= hi
                has_one = lo <= 1.0 <= hi
                if not (has_zero or has_one):
                    msg = (
                        f"Binary control coordinate {j} in block {k} has intersected bounds "
                        f"[{lo}, {hi}] which contains neither 0.0 nor 1.0."
                    )
                    raise ValueError(msg)

    return block_uL, block_uU


def nlp_primal_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Extract solver variable limits zL <= z_nlp <= zU for the reduced NLP vector.

    Parameters
    ----------
    problem : Problem
        Problem instance containing constraints, horizon dimensions, and control_block_steps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower bounds zL and upper bounds zU of shape (N * n + B * m,).
    """
    b = int(problem.control_block_steps)
    if b == 1:
        return primal_bounds(problem)

    N = int(problem.N)
    n = int(problem.model.n)
    H = N - 1
    B = H // b

    xL, xU, _, _ = problem.constraints.primal_bounds()
    block_uL, block_uU = blocked_control_bounds(problem)

    xL_blocks = xL[:-1].reshape((B, b * n))
    zL_stages = np.concatenate([xL_blocks, block_uL], axis=1).reshape(-1)
    zL = np.concatenate([zL_stages, xL[-1]])

    xU_blocks = xU[:-1].reshape((B, b * n))
    zU_stages = np.concatenate([xU_blocks, block_uU], axis=1).reshape(-1)
    zU = np.concatenate([zU_stages, xU[-1]])

    return zL.astype(np.float64), zU.astype(np.float64)


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
    """Compute bounds for canonical constraints, ending with linear Horizon rows.

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
        # 2a. Dynamics Defect k: x_{k+1} - f_d(x_k, u_k) = 0
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

    for con in problem.constraints.horizon_constraints:
        lo, hi = _cone_bounds(con.cone, con.p)
        gL_list.append(lo)
        gU_list.append(hi)

    gL = np.concatenate(gL_list) if gL_list else np.empty(0, dtype=np.float64)
    gU = np.concatenate(gU_list) if gU_list else np.empty(0, dtype=np.float64)
    return gL, gU


def constraint_structural_key(con: object) -> tuple[object, ...]:
    """Compute structural identity key for deduplicating time-independent input constraints."""
    from trajopt.constraints.base import Constraint  # noqa: PLC0415 -- local import

    if isinstance(con, Constraint):
        return con.structural_key()
    return ("Constraint", id(con))


def retained_knot_evaluators(problem: Problem) -> tuple[BuiltKnotConstraint, ...]:
    """Return Knot Point evaluators with duplicate time-independent input constraints collapsed per block."""
    from trajopt.constraints.constraint_list import BuiltKnotConstraint  # noqa: PLC0415 -- avoid circular import

    if problem.constraints is None:
        return ()

    b = int(problem.control_block_steps)
    knot_evaluators = problem.constraints.knot_evaluators
    if b == 1:
        return knot_evaluators

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    H = N - 1
    B = H // b

    retained: list[BuiltKnotConstraint] = []
    for blk in range(B):
        seen_keys: set[tuple[object, ...]] = set()
        for s in range(b):
            j = blk * b + s
            ev = knot_evaluators[j] if j < len(knot_evaluators) else None
            if ev is None or ev.p == 0:
                retained.append(BuiltKnotConstraint([], n=n, m=m, is_terminal=False))
                continue
            retained_cons = []
            for con in ev.constraints:
                if con.is_time_independent_input_constraint():
                    key = constraint_structural_key(con)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                retained_cons.append(con)
            retained.append(BuiltKnotConstraint(retained_cons, n=n, m=m, is_terminal=False))

    if len(knot_evaluators) > N - 1:
        retained.append(knot_evaluators[N - 1])
    else:
        retained.append(BuiltKnotConstraint([], n=n, m=m, is_terminal=True))

    return tuple(retained)


def nlp_constraint_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Compute bounds for the reduced NLP constraint vector, ending with linear Horizon rows."""
    b = int(problem.control_block_steps)
    if b == 1:
        return constraint_bounds(problem)

    N = int(problem.N)
    n = int(problem.model.n)
    retained = retained_knot_evaluators(problem)

    gL_list: list[np.ndarray] = []
    gU_list: list[np.ndarray] = []

    # 1. Initial state condition: x0 - x_init = 0
    gL_list.append(np.zeros(n, dtype=np.float64))
    gU_list.append(np.zeros(n, dtype=np.float64))

    for k in range(N - 1):
        # 2a. Dynamics Defect k: x_{k+1} - f_d(x_k, u_k) = 0
        gL_list.append(np.zeros(n, dtype=np.float64))
        gU_list.append(np.zeros(n, dtype=np.float64))

        # 2b. Retained stage constraints at knot k
        if k < len(retained) and retained[k].p > 0:
            lo_k, hi_k = _evaluator_bounds(retained[k])
            gL_list.extend(lo_k)
            gU_list.extend(hi_k)

    # 3. Terminal constraints at knot N - 1
    if len(retained) > N - 1 and retained[N - 1].p > 0:
        lo_term, hi_term = _evaluator_bounds(retained[N - 1])
        gL_list.extend(lo_term)
        gU_list.extend(hi_term)

    # 4. Trailing linear Horizon rows
    for con in problem.constraints.horizon_constraints:
        lo, hi = _cone_bounds(con.cone, con.p)
        gL_list.append(lo)
        gU_list.append(hi)

    gL = np.concatenate(gL_list) if gL_list else np.empty(0, dtype=np.float64)
    gU = np.concatenate(gU_list) if gU_list else np.empty(0, dtype=np.float64)
    return gL, gU


def nlp_constraint_row_count(problem: Problem) -> int:
    """Return retained NLP constraint row count, including trailing linear Horizon rows."""
    b = int(problem.control_block_steps)
    if b == 1:
        from trajopt.transcription.result import constraint_row_count  # noqa: PLC0415 -- local import

        return constraint_row_count(problem)

    n = int(problem.model.n)
    N = int(problem.N)
    H = N - 1
    retained = retained_knot_evaluators(problem)
    return n + H * n + sum(k.p for k in retained) + sum(c.p for c in problem.constraints.horizon_constraints)


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
    """Compute maximum violation across dynamics, bounds, Knot Point, Horizon, and block constancy constraints.

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

    viol_horizon = 0.0
    for con in problem.constraints.horizon_constraints:
        val = con.evaluate(Z_arr)
        viol_horizon = max(viol_horizon, float(jnp.max(jnp.abs(val - con.cone.project(val)))))

    viol_block = 0.0
    b = int(problem.control_block_steps)
    if b > 1:
        H = N - 1
        B = H // b
        U_blocks = np.asarray(U, dtype=np.float64).reshape((B, b, m))
        viol_block = float(np.max(np.abs(U_blocks - U_blocks[:, :1, :])))

    return max(max_primal, viol_init, viol_dyn, viol_stage, viol_horizon, viol_block)


def parse_solver_initial_state(
    problem: Problem,
    bc: BoundaryConditions,
    ws: WarmStart,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array]:
    """Extract standard JAX array representations of one solve's boundary, time-grid and warm-start data."""
    x0_arr = jnp.asarray(bc.x0, dtype=jnp.float64)
    t0_arr = jnp.asarray(bc.t0, dtype=jnp.float64)
    dt_arr = jnp.asarray(problem.dt, dtype=jnp.float64)
    xf_val = jnp.asarray(bc.xf, dtype=jnp.float64) if bc.xf is not None else None

    return x0_arr, t0_arr, dt_arr, xf_val, ws.Z


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
