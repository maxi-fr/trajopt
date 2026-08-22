import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.constraint_list import BuiltKnotConstraint
from trajopt.costs.base import CostFunction
from trajopt.dynamics.base import AbstractModel, ContinuousDynamics, DiscreteDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import RK4
from trajopt.problem import Problem
from trajopt.transcription.layout import z_to_trajectory


def _extract_discrete_model(problem: Problem | AbstractModel) -> DiscreteDynamics:
    """Extract or construct a DiscreteDynamics model from problem."""
    if isinstance(problem, Problem):
        model = problem.model
        integrator = problem.integrator
    elif isinstance(problem, AbstractModel):
        model = problem
        integrator = None
    else:
        msg = f"Cannot extract dynamics model from {type(problem).__name__}"
        raise TypeError(msg)

    if isinstance(model, DiscreteDynamics):
        return model
    if isinstance(model, ContinuousDynamics):
        integ = integrator if integrator is not None else RK4()
        return DiscretizedDynamics(continuous_dynamics=model, integrator=integ)
    msg = f"Model {type(model).__name__} is neither DiscreteDynamics nor ContinuousDynamics"
    raise TypeError(msg)


def _cost_fn(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
) -> jax.Array:
    """Evaluate total trajectory cost scalar J(Z)."""
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    X, U = z_to_trajectory(Z, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=Z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    stage_costs = problem.obj.stage_cost.stage_costs(X[:-1], U, t_stage)
    term_cost = problem.obj.terminal_cost.evaluate(X[-1], None, t_term)
    return jnp.sum(stage_costs) + term_cost


@eqx.filter_jit
def cost_and_grad(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
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

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Scalar cost J(Z) and gradient nabla J(Z) of shape (N * n + (N - 1) * m,).
    """
    return jax.value_and_grad(lambda z: _cost_fn(problem, z, t0, dt))(Z)


@eqx.filter_jit
def eval_f(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate objective value J(Z) as a scalar."""
    return _cost_fn(problem, Z, t0, dt)


@eqx.filter_jit
def eval_grad_f(
    problem: Problem,
    Z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate objective gradient nabla J(Z) of shape (N * n + (N - 1) * m,)."""
    return jax.grad(lambda z: _cost_fn(problem, z, t0, dt))(Z)


@eqx.filter_jit
def constraints_and_jac(
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate constraint vector c(Z) and sparse constraint Jacobian values.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, constraints, and horizon.
    Z : jax.Array
        Flat primal vector of shape (N * n + (N - 1) * m,).
    x0 : jax.Array
        Initial state condition of shape (n,).
    t0 : float | jax.Array, optional
        Initial time. Defaults to 0.0.
    dt : float | jax.Array, optional
        Time step duration. Defaults to 0.05.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Constraint vector c(Z) of shape (P,) and Jacobian nonzeros of shape (nnz_jac,).
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = _extract_discrete_model(problem)
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    X, U = z_to_trajectory(Z, N, n, m)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_stage = t0 + jnp.concatenate([jnp.zeros(1, dtype=Z.dtype), jnp.cumsum(dt_arr[:-1])])
    t_term = t0 + jnp.sum(dt_arr)

    # 1. Initial state defect and jacobian
    c_init = X[0] - x0
    jac_init = jnp.eye(n, dtype=Z.dtype).reshape(-1)

    # 2. Dynamics defects and stage jacobians
    def step_dyn(
        xk: jax.Array, uk: jax.Array, x_next: jax.Array, tk: jax.Array, dtk: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        fd = discrete_model.discrete_dynamics(xk, uk, tk, dtk)
        defect = x_next - fd
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
            val_k = evaluator.evaluate(X[k], U[k], t_stage[k])
            jx_k, ju_k = evaluator.jacobian(X[k], U[k], t_stage[k])
            j_k = jnp.hstack([jx_k, ju_k])
            c_list.append(val_k)
            jac_list.append(j_k.reshape(-1))

    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        evaluator = knot_evaluators[N - 1]
        val_term = evaluator.evaluate(X[-1], None, t_term)
        jx_term, _ = evaluator.jacobian(X[-1], None, t_term)
        c_list.append(val_term)
        jac_list.append(jx_term.reshape(-1))

    c_all = jnp.concatenate(c_list)
    jac_all = jnp.concatenate(jac_list)
    return c_all, jac_all


def _constraints_fn(
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate constraint vector c(Z) of shape (P,) without computing Jacobians."""
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = _extract_discrete_model(problem)
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    X, U = z_to_trajectory(Z, N, n, m)
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
            c_list.append(knot_evaluators[k].evaluate(X[k], U[k], t_stage[k]))

    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        c_list.append(knot_evaluators[N - 1].evaluate(X[-1], None, t_term))

    return jnp.concatenate(c_list)


@eqx.filter_jit
def eval_g(
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate constraint vector c(Z) of shape (P,)."""
    return _constraints_fn(problem, Z, x0, t0, dt)


@eqx.filter_jit
def eval_jac_g(
    problem: Problem,
    Z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate sparse constraint Jacobian values of shape (nnz_jac,)."""
    _, jac_val = constraints_and_jac(problem, Z, x0, t0, dt)
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
def hessian(  # noqa: PLR0913 -- Lagrangian Hessian requires 6 arguments
    problem: Problem,
    Z: jax.Array,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
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

    Returns
    -------
    jax.Array
        Lower-triangular nonzeros of the Lagrangian Hessian of shape (nnz_hess,).
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    discrete_model = _extract_discrete_model(problem)
    built_constraints = problem.constraints
    knot_evaluators = built_constraints.knot_evaluators if built_constraints is not None else ()

    X, U = z_to_trajectory(Z, N, n, m)
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
def eval_h(  # noqa: PLR0913 -- Lagrangian Hessian callback requires 6 arguments
    problem: Problem,
    Z: jax.Array,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
) -> jax.Array:
    """Evaluate lower-triangular nonzeros of the Lagrangian Hessian."""
    return hessian(problem, Z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam)
