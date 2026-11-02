from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.constraint_list import BuiltKnotConstraint
from trajopt.costs.base import CostFunction
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.transcription.layout import (
    _nlp_z_to_trajectory,
    _trajectory_to_z,
    _z_to_trajectory,
    canonical_to_nlp_col,
    expand_controls,
    retained_knot_evaluators,
)


def _cost_fn(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate total trajectory cost scalar J(Z), aiming the objective at `bc`'s reference window."""
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    X, U = _z_to_trajectory(Z, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=Z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    obj = retarget_problem(problem, bc).obj

    stage_costs = obj.stage_cost.stage_costs(X[:-1], U, t_stage)
    term_cost = obj.terminal_cost.evaluate(X[-1], None, t_term)
    return jnp.sum(stage_costs) + term_cost


@eqx.filter_jit
def cost_and_grad(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    bc: BoundaryConditions | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate objective value J(Z) and its gradient nabla J(Z).

    Parameters
    ----------
    problem : Problem
        Problem instance containing objective and horizon.
    Z : jax.Array
        Flat primal vector of shape (N * n + (N - 1) * m,).
    t0 : float | jax.Array, optional
        Initial time. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.05.
    bc : BoundaryConditions | None, optional
        Traced boundary conditions whose reference window aims the objective. Defaults to None,
        meaning the objective keeps the target it was built with.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Scalar cost J(Z) and gradient nabla J(Z) of shape (N * n + (N - 1) * m,).
    """
    return jax.value_and_grad(lambda z: _cost_fn(problem, z, t0, dt, bc))(Z)


@eqx.filter_jit
def eval_f(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate objective value J(Z) as a scalar, aimed at `bc`'s reference window."""
    return _cost_fn(problem, Z, t0, dt, bc)


@eqx.filter_jit
def eval_grad_f(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate objective gradient nabla J(Z) of shape (N * n + (N - 1) * m,), aimed at `bc`."""
    return jax.grad(lambda z: _cost_fn(problem, z, t0, dt, bc))(Z)


@eqx.filter_jit
def constraints_and_jac(  # noqa: PLR0913 -- Constraint evaluation takes 6 arguments
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate canonical constraints and Jacobian, appending linear Horizon rows.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, constraints, and Horizon.
    Z : jax.Array
        Flat primal vector of shape (N * n + (N - 1) * m,).
    x0 : jax.Array
        Initial state condition of shape (n,).
    t0 : float | jax.Array, optional
        Initial time. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.05.
    xf : jax.Array | None, optional
        Goal state vector. Defaults to None.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Constraint vector c(Z) of shape (P,) and sparse Jacobian nonzeros of shape (nnz_jac,).
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = problem.model
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    X, U = _z_to_trajectory(Z, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=Z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    # 1. Initial state defect and jacobian
    c_init = X[0] - x0
    jac_init = jnp.eye(n, dtype=Z.dtype).reshape(-1)

    # 2. Dynamics defects
    def step_dyn(
        xk: jax.Array,
        uk: jax.Array,
        x_next: jax.Array,
        tk: jax.Array,
        dtk: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        defect = x_next - discrete_model.discrete_dynamics(xk, uk, tk, dtk)
        Ak = discrete_model.state_jacobian(xk, uk, tk, dtk)
        Bk = discrete_model.control_jacobian(xk, uk, tk, dtk)
        J_ku = -jnp.hstack([Ak, Bk])
        return defect, J_ku

    dyn_defects, dyn_jacs_ku = jax.vmap(step_dyn)(X[:-1], U, X[1:], t_stage, dt_arr)
    eye_n = jnp.eye(n, dtype=Z.dtype).reshape(-1)

    c_list: list[jax.Array] = [c_init]
    jac_list: list[jax.Array] = [jac_init]

    for k in range(N - 1):
        c_list.append(dyn_defects[k])
        jac_list.append(dyn_jacs_ku[k].reshape(-1))
        jac_list.append(eye_n)

        if k < len(knot_evaluators) and knot_evaluators[k].p > 0:
            evaluator = knot_evaluators[k]
            val_k = evaluator.evaluate(X[k], U[k], t_stage[k], xf=xf)
            jx_k, ju_k = evaluator.jacobian(X[k], U[k], t_stage[k])
            j_k = jnp.hstack([jx_k, ju_k])
            c_list.append(val_k)
            jac_list.append(j_k.reshape(-1))

    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        evaluator = knot_evaluators[N - 1]
        val_term = evaluator.evaluate(X[-1], None, t_term, xf=xf)
        jx_term, _ = evaluator.jacobian(X[-1], None, t_term)
        c_list.append(val_term)
        jac_list.append(jx_term.reshape(-1))

    for con in built_constraints.horizon_constraints:
        c_list.append(con.evaluate(Z))
        jac_list.append(con.A.reshape(-1))

    c_all = jnp.concatenate(c_list)
    jac_all = jnp.concatenate(jac_list)
    return c_all, jac_all


def _constraints_fn(  # noqa: PLR0913 -- Constraint evaluation takes 6 arguments
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate constraint vector c(Z), including trailing Horizon rows, without Jacobians."""
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = problem.model
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    X, U = _z_to_trajectory(Z, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=Z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    # 1. Initial condition
    c_init = X[0] - x0

    # 2. Dynamics defects
    def step_dyn_defect(
        xk: jax.Array,
        uk: jax.Array,
        x_next: jax.Array,
        tk: jax.Array,
        dtk: jax.Array,
    ) -> jax.Array:
        return x_next - discrete_model.discrete_dynamics(xk, uk, tk, dtk)

    dyn_defects = jax.vmap(step_dyn_defect)(X[:-1], U, X[1:], t_stage, dt_arr)

    c_list: list[jax.Array] = [c_init]
    for k in range(N - 1):
        c_list.append(dyn_defects[k])
        if k < len(knot_evaluators) and knot_evaluators[k].p > 0:
            c_list.append(knot_evaluators[k].evaluate(X[k], U[k], t_stage[k], xf=xf))

    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        c_list.append(knot_evaluators[N - 1].evaluate(X[-1], None, t_term, xf=xf))

    c_list.extend(con.evaluate(Z) for con in built_constraints.horizon_constraints)

    return jnp.concatenate(c_list)


@eqx.filter_jit
def eval_g(  # noqa: PLR0913 -- Constraint callback takes 6 arguments
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate constraint vector c(Z) of shape (P,)."""
    return _constraints_fn(problem, Z, x0, t0, dt, xf=xf)


@eqx.filter_jit
def eval_jac_g(  # noqa: PLR0913 -- Constraint Jacobian callback takes 6 arguments
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate sparse constraint Jacobian values of shape (nnz_jac,)."""
    _, jac_val = constraints_and_jac(problem, Z, x0, t0, dt, xf=xf)
    return jac_val


def _knot_lagrangian_hessian(  # noqa: PLR0913 -- Knot Lagrangian takes 10 structural arguments
    stage_cost_k: CostFunction,
    discrete_model: DiscreteDynamics,
    evaluator_k: BuiltKnotConstraint | None,
    *,
    lam_dyn_k: jax.Array,
    lam_con_k: jax.Array | None,
    zk: jax.Array,
    tk: jax.Array,
    dtk: jax.Array,
    obj_factor: float | jax.Array,
    n: int,
) -> jax.Array:
    """Evaluate lower-triangular Hessian of a single stage knot Lagrangian."""

    def knot_lagrangian(z_in: jax.Array) -> jax.Array:
        x_in = z_in[:n]
        u_in = z_in[n:]
        cost_val = stage_cost_k.evaluate(x_in, u_in, tk)
        dyn_val = -jnp.dot(lam_dyn_k, discrete_model.discrete_dynamics(x_in, u_in, tk, dtk))
        if evaluator_k is not None and evaluator_k.p > 0 and lam_con_k is not None:
            con_val = jnp.dot(lam_con_k, evaluator_k.evaluate(x_in, u_in, tk))
        else:
            con_val = jnp.zeros((), dtype=z_in.dtype)
        return obj_factor * cost_val + dyn_val + con_val

    return jax.hessian(knot_lagrangian)(zk)


def _term_lagrangian_hessian(  # noqa: PLR0913 -- Terminal knot Lagrangian takes 6 structural arguments
    term_cost: CostFunction,
    evaluator_term: BuiltKnotConstraint | None,
    lam_con_term: jax.Array | None,
    x_term: jax.Array,
    t_term: jax.Array,
    *,
    obj_factor: float | jax.Array,
) -> jax.Array:
    """Evaluate lower-triangular Hessian of the terminal knot Lagrangian."""

    def term_lagrangian(x_in: jax.Array) -> jax.Array:
        cost_val = term_cost.evaluate(x_in, None, t_term)
        if evaluator_term is not None and evaluator_term.p > 0 and lam_con_term is not None:
            con_val = jnp.dot(lam_con_term, evaluator_term.evaluate(x_in, None, t_term))
        else:
            con_val = jnp.zeros((), dtype=x_in.dtype)
        return obj_factor * cost_val + con_val

    return jax.hessian(term_lagrangian)(x_term)


@eqx.filter_jit
def hessian(  # noqa: PLR0913 -- Lagrangian Hessian requires 7 arguments
    problem: Problem,
    Z: jax.Array,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate lower-triangular nonzeros of the block-diagonal Lagrangian Hessian.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, objective, constraints, and horizon.
    Z : jax.Array
        Flat primal vector of shape (N * n + (N - 1) * m,).
    t0 : float | jax.Array, optional
        Initial time. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.05.
    obj_factor : float | jax.Array, optional
        Objective scale factor sigma_f. Defaults to 1.0.
    lam : jax.Array | None, optional
        Constraint multiplier vector of shape (P,). Defaults to zeros.
    bc : BoundaryConditions | None, optional
        Traced boundary conditions, accepted for signature parity. Defaults to None.

    Returns
    -------
    jax.Array
        Lower-triangular nonzeros of the Lagrangian Hessian of shape (nnz_hess,).
    """
    del bc  # retargeting rewrites only the linear terms, which the Hessian does not see
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = problem.model
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    X, U = _z_to_trajectory(Z, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=Z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    P_total = n + (N - 1) * n + sum(k.p for k in knot_evaluators)
    lam_vec = jnp.zeros(P_total, dtype=Z.dtype) if lam is None else jnp.asarray(lam, dtype=Z.dtype)

    # Slicing lam
    offset = n
    lam_dyn_list = []
    lam_con_list = []
    for k in range(N - 1):
        lam_dyn_list.append(lam_vec[offset : offset + n])
        offset += n
        pk = knot_evaluators[k].p if k < len(knot_evaluators) else 0
        if pk > 0:
            lam_con_list.append(lam_vec[offset : offset + pk])
            offset += pk
        else:
            lam_con_list.append(None)

    p_term = knot_evaluators[N - 1].p if len(knot_evaluators) > N - 1 else 0
    lam_con_term = lam_vec[offset : offset + p_term] if p_term > 0 else None

    d_stage = n + m
    tril_r_stage, tril_c_stage = np.tril_indices(d_stage)
    tril_r_term, tril_c_term = np.tril_indices(n)

    hess_blocks: list[jax.Array] = []

    # 1. Stage knots k = 0, ..., N - 2
    for k in range(N - 1):
        zk = jnp.concatenate([X[k], U[k]])
        tk = t_stage[k]
        dtk = dt_arr[k]
        lam_dyn_k = lam_dyn_list[k]
        lam_con_k = lam_con_list[k]
        evaluator_k = knot_evaluators[k] if k < len(knot_evaluators) else None
        stage_cost_k = problem.obj[k]

        Hk = _knot_lagrangian_hessian(
            stage_cost_k,
            discrete_model,
            evaluator_k,
            lam_dyn_k=lam_dyn_k,
            lam_con_k=lam_con_k,
            zk=zk,
            tk=tk,
            dtk=dtk,
            obj_factor=obj_factor,
            n=n,
        )
        hess_blocks.append(Hk[tril_r_stage, tril_c_stage])

    # 2. Terminal knot N - 1
    x_term = X[-1]
    evaluator_term = knot_evaluators[N - 1] if len(knot_evaluators) > N - 1 else None

    H_term = _term_lagrangian_hessian(
        problem.obj.terminal_cost,
        evaluator_term,
        lam_con_term,
        x_term,
        t_term,
        obj_factor=obj_factor,
    )
    hess_blocks.append(H_term[tril_r_term, tril_c_term])

    return jnp.concatenate(hess_blocks)


@eqx.filter_jit
def eval_h(  # noqa: PLR0913 -- Lagrangian Hessian callback requires 7 arguments
    problem: Problem,
    Z: jax.Array,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate lower-triangular nonzeros of the Lagrangian Hessian."""
    return hessian(problem, Z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam, bc=bc)


def _nlp_cost_fn(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate total Trajectory cost scalar J(z_nlp) on expanded controls."""
    b = int(problem.control_block_steps)
    if b == 1:
        return _cost_fn(problem, z, t0, dt, bc)

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    X, U_blk = _nlp_z_to_trajectory(z, N, n, m, b)
    U = expand_controls(U_blk, b)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    obj = retarget_problem(problem, bc).obj

    stage_costs = obj.stage_cost.stage_costs(X[:-1], U, t_stage)
    term_cost = obj.terminal_cost.evaluate(X[-1], None, t_term)
    return jnp.sum(stage_costs) + term_cost


@eqx.filter_jit
def nlp_cost_and_grad(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    bc: BoundaryConditions | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate objective value J(z_nlp) and its gradient nabla J(z_nlp) of shape (N * n + B * m,)."""
    return jax.value_and_grad(lambda z_: _nlp_cost_fn(problem, z_, t0, dt, bc))(z)


@eqx.filter_jit
def nlp_eval_f(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate objective value J(z_nlp) as a scalar on expanded controls."""
    return _nlp_cost_fn(problem, z, t0, dt, bc)


@eqx.filter_jit
def nlp_eval_grad_f(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate objective gradient nabla J(z_nlp) of shape (N * n + B * m,)."""
    return jax.grad(lambda z_: _nlp_cost_fn(problem, z_, t0, dt, bc))(z)


def _nlp_constraints_fn(  # noqa: PLR0913 -- Constraint evaluation takes 6 arguments
    problem: Problem,
    z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate constraint vector c(z_nlp) of shape (P,) without Jacobians."""
    b = int(problem.control_block_steps)
    if b == 1:
        return _constraints_fn(problem, z, x0, t0, dt, xf=xf)
    c_val, _ = nlp_constraints_and_jac(problem, z, x0, t0, dt, xf=xf)
    return c_val


def _nlp_horizon_jac(  # noqa: PLR0913 -- Horizon Jacobian helper takes 7 arguments
    horizon_constraints: tuple[Any, ...],
    Z_canon: jax.Array,
    *,
    N: int,
    n: int,
    m: int,
    b: int,
    B: int,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Evaluate Horizon constraint residuals and sparse Jacobian rows for z_nlp."""
    del B
    c_list: list[jax.Array] = []
    jac_list: list[jax.Array] = []

    for con in horizon_constraints:
        c_list.append(con.evaluate(Z_canon))
        mapped = [canonical_to_nlp_col(int(c), N, n, m, b) for c in con.inds]
        unique_cols = np.unique(mapped)
        cols_arr = jnp.asarray(mapped)
        u_cols_arr = jnp.asarray(unique_cols)

        def acc_col(u_col: jax.Array, cols: jax.Array = cols_arr, A: jax.Array = con.A) -> jax.Array:
            mask = cols == u_col
            return jnp.sum(jnp.where(mask[None, :], A, 0.0), axis=1)

        A_nlp = jax.vmap(acc_col, in_axes=0, out_axes=1)(u_cols_arr)
        jac_list.append(A_nlp.reshape(-1))

    return c_list, jac_list


def _nlp_stage_constraints_and_jac(  # noqa: PLR0913 -- Stage helper takes 8 arguments
    discrete_model: DiscreteDynamics,
    knot_evaluators: tuple[BuiltKnotConstraint, ...],
    X: jax.Array,
    U_blk: jax.Array,
    *,
    b: int,
    t_stage: jax.Array,
    dt_arr: jax.Array,
    xf: jax.Array | None,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Evaluate native Defect equations, stage Knot Point constraints, and their Jacobians for z_nlp."""
    H = len(X) - 1
    n = X.shape[1]
    eye_n = jnp.eye(n, dtype=X.dtype).reshape(-1)
    c_list: list[jax.Array] = []
    jac_list: list[jax.Array] = []

    for j in range(H):
        k = j // b
        u_j = U_blk[k]
        xk = X[j]
        x_next = X[j + 1]
        tk = t_stage[j]
        dtk = dt_arr[j]

        defect = x_next - discrete_model.discrete_dynamics(xk, u_j, tk, dtk)
        Ak = discrete_model.state_jacobian(xk, u_j, tk, dtk)
        Bk = discrete_model.control_jacobian(xk, u_j, tk, dtk)

        c_list.append(defect)
        jac_list.append(-Ak.reshape(-1))
        jac_list.append(-Bk.reshape(-1))
        jac_list.append(eye_n)

        if j < len(knot_evaluators) and knot_evaluators[j].p > 0:
            evaluator = knot_evaluators[j]
            c_list.append(evaluator.evaluate(xk, u_j, tk, xf=xf))
            jx_j, ju_j = evaluator.jacobian(xk, u_j, tk)
            jac_list.append(jx_j.reshape(-1))
            jac_list.append(ju_j.reshape(-1))

    return c_list, jac_list


@eqx.filter_jit
def nlp_constraints_and_jac(  # noqa: PLR0913 -- Constraint evaluation takes 6 arguments
    problem: Problem,
    z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate canonical constraints and Jacobian with respect to z_nlp.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, constraints, and Horizon.
    z : jax.Array
        Flat NLP primal vector of shape (N * n + B * m,).
    x0 : jax.Array
        Initial state condition of shape (n,).
    t0 : float | jax.Array, optional
        Initial time. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.05.
    xf : jax.Array | None, optional
        Goal state vector. Defaults to None.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Constraint vector c(z) of shape (P,) and sparse Jacobian nonzeros of shape (nnz_jac,).
    """
    b = int(problem.control_block_steps)
    if b == 1:
        return constraints_and_jac(problem, z, x0, t0, dt, xf=xf)

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    H = N - 1
    B = H // b
    discrete_model = problem.model
    retained_evaluators = retained_knot_evaluators(problem)

    X, U_blk = _nlp_z_to_trajectory(z, N, n, m, b)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (H,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    # 1. Initial state defect and jacobian
    c_init = X[0] - x0
    jac_init = jnp.eye(n, dtype=z.dtype).reshape(-1)

    c_list: list[jax.Array] = [c_init]
    jac_list: list[jax.Array] = [jac_init]

    c_stages, jac_stages = _nlp_stage_constraints_and_jac(
        discrete_model,
        retained_evaluators,
        X,
        U_blk,
        b=b,
        t_stage=t_stage,
        dt_arr=dt_arr,
        xf=xf,
    )
    c_list.extend(c_stages)
    jac_list.extend(jac_stages)

    if len(retained_evaluators) > N - 1 and retained_evaluators[N - 1].p > 0:
        evaluator = retained_evaluators[N - 1]
        val_term = evaluator.evaluate(X[-1], None, t_term, xf=xf)
        jx_term, _ = evaluator.jacobian(X[-1], None, t_term)
        c_list.append(val_term)
        jac_list.append(jx_term.reshape(-1))

    U_expanded = expand_controls(U_blk, b)
    Z_canon = _trajectory_to_z(X, U_expanded)

    c_hz, jac_hz = _nlp_horizon_jac(
        problem.constraints.horizon_constraints,
        Z_canon,
        N=N,
        n=n,
        m=m,
        b=b,
        B=B,
    )
    c_list.extend(c_hz)
    jac_list.extend(jac_hz)

    c_all = jnp.concatenate(c_list)
    jac_all = jnp.concatenate(jac_list)
    return c_all, jac_all


@eqx.filter_jit
def nlp_eval_g(  # noqa: PLR0913 -- Constraint callback takes 6 arguments
    problem: Problem,
    z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate constraint vector c(z_nlp) of shape (P,)."""
    return _nlp_constraints_fn(problem, z, x0, t0, dt, xf=xf)


@eqx.filter_jit
def nlp_eval_jac_g(  # noqa: PLR0913 -- Constraint Jacobian callback takes 6 arguments
    problem: Problem,
    z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    *,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate sparse constraint Jacobian values of shape (nnz_jac,)."""
    _, jac_val = nlp_constraints_and_jac(problem, z, x0, t0, dt, xf=xf)
    return jac_val


def _block_hessian_entries(  # noqa: PLR0913 -- Block Hessian helper takes 14 arguments
    problem: Problem,
    X: jax.Array,
    u_blk_k: jax.Array,
    k: int,
    *,
    b: int,
    n: int,
    m: int,
    t_stage: jax.Array,
    dt_arr: jax.Array,
    lam_dyn_list: list[jax.Array],
    lam_con_list: list[jax.Array | None],
    knot_evaluators: tuple[BuiltKnotConstraint, ...],
    obj_factor: float | jax.Array,
    discrete_model: DiscreteDynamics,
) -> list[jax.Array]:
    """Evaluate lower-triangular nonzeros for one control block."""
    H_xx_list = []
    H_ux_list = []
    H_uu_list = []

    for s in range(b):
        j = k * b + s
        zk = jnp.concatenate([X[j], u_blk_k])
        tk = t_stage[j]
        dtk = dt_arr[j]
        lam_dyn_j = lam_dyn_list[j]
        lam_con_j = lam_con_list[j]
        evaluator_j = knot_evaluators[j] if j < len(knot_evaluators) else None
        stage_cost_j = problem.obj[j]

        Hj = _knot_lagrangian_hessian(
            stage_cost_j,
            discrete_model,
            evaluator_j,
            lam_dyn_k=lam_dyn_j,
            lam_con_k=lam_con_j,
            zk=zk,
            tk=tk,
            dtk=dtk,
            obj_factor=obj_factor,
            n=n,
        )
        H_xx_list.append(Hj[:n, :n])
        H_ux_list.append(Hj[n:, :n])
        H_uu_list.append(Hj[n:, n:])

    tril_r_n, tril_c_n = np.tril_indices(n)
    tril_r_m, tril_c_m = np.tril_indices(m)

    block_entries: list[jax.Array] = []
    block_entries.extend(H_xx_list[s][tril_r_n, tril_c_n] for s in range(b))
    block_entries.extend(H_ux_list[s].reshape(-1) for s in range(b))
    H_uu_blk = jnp.sum(jnp.stack(H_uu_list), axis=0)
    block_entries.append(H_uu_blk[tril_r_m, tril_c_m])
    return block_entries


@eqx.filter_jit
def nlp_hessian(  # noqa: PLR0913 -- Lagrangian Hessian requires 7 arguments
    problem: Problem,
    z: jax.Array,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate lower-triangular nonzeros of the Lagrangian Hessian for z_nlp.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, objective, constraints, and Horizon.
    z : jax.Array
        Flat NLP primal vector of shape (N * n + B * m,).
    t0 : float | jax.Array, optional
        Initial time. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.05.
    obj_factor : float | jax.Array, optional
        Objective scale factor sigma_f. Defaults to 1.0.
    lam : jax.Array | None, optional
        Constraint Multiplier vector of shape (P,). Defaults to zeros.
    bc : BoundaryConditions | None, optional
        Traced boundary conditions, accepted for signature parity. Defaults to None.

    Returns
    -------
    jax.Array
        Lower-triangular nonzeros of the Lagrangian Hessian of shape (nnz_hess,).
    """
    del bc
    b = int(problem.control_block_steps)
    if b == 1:
        return hessian(problem, z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam)

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = problem.model
    retained_evaluators = retained_knot_evaluators(problem)

    H = N - 1
    B = H // b
    X, U_blk = _nlp_z_to_trajectory(z, N, n, m, b)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (H,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    P_total = n + H * n + sum(k.p for k in retained_evaluators)
    lam_vec = jnp.zeros(P_total, dtype=z.dtype) if lam is None else jnp.asarray(lam, dtype=z.dtype)

    offset = n
    lam_dyn_list = []
    lam_con_list = []
    for k in range(H):
        lam_dyn_list.append(lam_vec[offset : offset + n])
        offset += n
        pk = retained_evaluators[k].p if k < len(retained_evaluators) else 0
        if pk > 0:
            lam_con_list.append(lam_vec[offset : offset + pk])
            offset += pk
        else:
            lam_con_list.append(None)

    p_term = retained_evaluators[N - 1].p if len(retained_evaluators) > N - 1 else 0
    lam_con_term = lam_vec[offset : offset + p_term] if p_term > 0 else None

    tril_r_term, tril_c_term = np.tril_indices(n)
    hess_blocks: list[jax.Array] = []

    for k in range(B):
        entries_k = _block_hessian_entries(
            problem,
            X,
            U_blk[k],
            k,
            b=b,
            n=n,
            m=m,
            t_stage=t_stage,
            dt_arr=dt_arr,
            lam_dyn_list=lam_dyn_list,
            lam_con_list=lam_con_list,
            knot_evaluators=retained_evaluators,
            obj_factor=obj_factor,
            discrete_model=discrete_model,
        )
        hess_blocks.extend(entries_k)

    # 4. Terminal knot
    x_term = X[-1]
    evaluator_term = retained_evaluators[N - 1] if len(retained_evaluators) > N - 1 else None
    H_term = _term_lagrangian_hessian(
        problem.obj.terminal_cost,
        evaluator_term,
        lam_con_term,
        x_term,
        t_term,
        obj_factor=obj_factor,
    )
    hess_blocks.append(H_term[tril_r_term, tril_c_term])

    return jnp.concatenate(hess_blocks)


@eqx.filter_jit
def nlp_eval_h(  # noqa: PLR0913 -- Lagrangian Hessian callback requires 7 arguments
    problem: Problem,
    z: jax.Array,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
    bc: BoundaryConditions | None = None,
) -> jax.Array:
    """Evaluate lower-triangular nonzeros of the Lagrangian Hessian for z_nlp."""
    return nlp_hessian(problem, z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam, bc=bc)
