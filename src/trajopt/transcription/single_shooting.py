from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.dynamics.rollout import rollout_states
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.homotopy import (
    _SUCCESS_STATUSES,
    solve_continuation_passes,
    validate_homotopy_parameters,
)
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.layout import (
    _cone_bounds,
    _evaluator_bounds,
    _trajectory_to_z,
    _z_to_trajectory,
    blocked_control_bounds,
    compute_constraint_violation,
    expand_controls,
    parse_solver_initial_state,
    reduce_controls,
    retained_knot_evaluators,
)
from trajopt.transcription.result import (
    constraint_row_count,
    warm_start_duals,
)

_EMPTY = np.zeros(0, dtype=np.float64)


def _shooting_rollout(
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Roll out states from x0 and a flat block control vector, returning (X, U, t)."""
    N = int(problem.N)
    m = int(problem.model.m)
    b = int(problem.control_block_steps)
    B = (N - 1) // b
    U_blk = u.reshape((B, m))
    U = expand_controls(U_blk, b)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=u.dtype), jnp.cumsum(dt_arr)])
    X = rollout_states(problem.model, x0, U, t=t_arr, dt=dt_arr)
    return X, U, t_arr


def _cost_fn(
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
) -> jax.Array:
    """Evaluate total cost as a function of the flat control vector alone.

    `problem`'s objective is used as given, so a caller aiming it at a run-time reference window
    retargets the problem first.
    """
    X, U, t_arr = _shooting_rollout(problem, u, x0, t0, dt)
    obj = problem.obj
    stage_costs = obj.stage_cost.stage_costs(X[:-1], U, t_arr[:-1])
    term_cost = obj.terminal_cost.evaluate(X[-1], None, t_arr[-1])
    return jnp.sum(stage_costs) + term_cost


def _constraints_fn(  # noqa: PLR0913, PLR0917 -- Constraint evaluation takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the user constraint vector c(u) after rolling out the states."""
    N = int(problem.N)
    X, U, t_arr = _shooting_rollout(problem, u, x0, t0, dt)
    retained = retained_knot_evaluators(problem)

    c_list: list[jax.Array] = []
    for k in range(N - 1):
        evaluator = retained[k]
        if evaluator.p > 0:
            c_list.append(evaluator.evaluate(X[k], U[k], t_arr[k], xf=xf))
    if len(retained) > N - 1 and retained[N - 1].p > 0:
        c_list.append(retained[N - 1].evaluate(X[-1], None, t_arr[-1], xf=xf))

    Z_canon = _trajectory_to_z(X, U)
    c_list.extend(con.evaluate(Z_canon) for con in problem.constraints.horizon_constraints)

    if c_list:
        return jnp.concatenate(c_list)
    return jnp.zeros(0, dtype=u.dtype)


@eqx.filter_jit
def eval_f(
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate objective value J(u) as a scalar."""
    return _cost_fn(problem, u, x0, t0, dt)


@eqx.filter_jit
def eval_grad_f(
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate Objective gradient nabla J(u) of shape (B * m,)."""
    return jax.grad(lambda u_: _cost_fn(problem, u_, x0, t0, dt))(u)


@eqx.filter_jit
def eval_g(  # noqa: PLR0913, PLR0917 -- Constraint callback takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate constraint vector c(u) of shape (P_user,)."""
    return _constraints_fn(problem, u, x0, t0, dt, xf=xf)


@eqx.filter_jit
def eval_jac_g(  # noqa: PLR0913, PLR0917 -- Jacobian callback takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate dense constraint Jacobian values of shape (P_user * B * m,)."""
    return jax.jacrev(lambda u_: _constraints_fn(problem, u_, x0, t0, dt, xf=xf))(u).reshape(-1)


@eqx.filter_jit
def eval_h(  # noqa: PLR0913, PLR0917 -- Hessian callback takes 8 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    obj_factor: float | jax.Array = 1.0,
    lam: jax.Array | None = None,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate lower-triangular nonzeros of the dense Lagrangian Hessian w.r.t. u."""
    n_u, p_user = single_shooting_dimensions(problem)
    lam_vec = jnp.zeros(p_user, dtype=u.dtype) if lam is None else jnp.asarray(lam, dtype=u.dtype)

    def lagrangian(u_: jax.Array) -> jax.Array:
        cost = _cost_fn(problem, u_, x0, t0, dt)
        cons = _constraints_fn(problem, u_, x0, t0, dt, xf=xf)
        return obj_factor * cost + jnp.dot(lam_vec, cons)

    hess = jax.hessian(lagrangian)(u)
    tril_r, tril_c = np.tril_indices(n_u)
    return hess[tril_r, tril_c]


def single_shooting_dimensions(problem: Problem) -> tuple[int, int]:
    """Return the single-shooting NLP's Primal Vector and constraint row counts (n_u, p_user)."""
    b = int(problem.control_block_steps)
    B = (int(problem.N) - 1) // b
    n_u = B * int(problem.model.m)
    retained = retained_knot_evaluators(problem)
    p_user = int(sum(k.p for k in retained)) + sum(con.p for con in problem.constraints.horizon_constraints)
    return n_u, p_user


def _validate_supported_constraints(problem: Problem) -> None:
    """Refuse primal state bounds: single shooting has no state decision variable to bound."""
    x_lower, x_upper, _, _ = problem.constraints.primal_bounds()
    if np.any(np.isfinite(x_lower)) or np.any(np.isfinite(x_upper)):
        msg = (
            "Single shooting cannot express state bounds: with no state decision variables a "
            "StateBound has no primal variable to bound. Remove the state bound or use the "
            "multiple-shooting transcription."
        )
        raise ValueError(msg)


def _constraint_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper bounds gL <= c(u) <= gU for the user constraint rows."""
    N = int(problem.N)
    retained = retained_knot_evaluators(problem)

    g_lower: list[np.ndarray] = []
    g_upper: list[np.ndarray] = []
    for k in range(N - 1):
        lo_k, hi_k = _evaluator_bounds(retained[k])
        g_lower.extend(lo_k)
        g_upper.extend(hi_k)
    if len(retained) > N - 1:
        lo_term, hi_term = _evaluator_bounds(retained[N - 1])
        g_lower.extend(lo_term)
        g_upper.extend(hi_term)

    for con in problem.constraints.horizon_constraints:
        lo, hi = _cone_bounds(con.cone, con.p)
        g_lower.append(lo)
        g_upper.append(hi)

    g_l = np.concatenate(g_lower) if g_lower else np.empty(0, dtype=np.float64)
    g_u = np.concatenate(g_upper) if g_upper else np.empty(0, dtype=np.float64)
    return g_l, g_u


def _primal_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Return control-only primal bounds zL <= u <= zU of shape (B * m,)."""
    b = int(problem.control_block_steps)
    if b > 1:
        block_u_l, block_u_u = blocked_control_bounds(problem)
        return block_u_l.reshape(-1), block_u_u.reshape(-1)
    _, _, u_lower, u_upper = problem.constraints.primal_bounds()
    return u_lower.reshape(-1), u_upper.reshape(-1)


def _dense_jacobian_pattern(p: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense COO sparsity pattern for a (p, n) Jacobian, in row-major order."""
    if p == 0 or n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    rows = np.repeat(np.arange(p, dtype=np.int32), n)
    cols = np.tile(np.arange(n, dtype=np.int32), p)
    return rows, cols


def _extract_retained_warm_start_duals(
    problem: Problem,
    ws: WarmStart,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract retained user and Horizon constraint Multipliers and control-bound Multipliers from WarmStart."""
    lam_canon, mu_canon = warm_start_duals(problem, ws)
    if lam_canon is None or mu_canon is None:
        return None, None

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    knot_evaluators = problem.constraints.knot_evaluators

    offset = n  # initial-state pin
    retained_lam: list[np.ndarray] = []
    for k in range(N - 1):
        offset += n  # dynamics Defect k
        pk = knot_evaluators[k].p if k < len(knot_evaluators) else 0
        if pk > 0:
            retained_lam.append(lam_canon[offset : offset + pk])
            offset += pk

    p_term = knot_evaluators[N - 1].p if len(knot_evaluators) > N - 1 else 0
    if p_term > 0:
        retained_lam.append(lam_canon[offset : offset + p_term])
        offset += p_term

    for con in problem.constraints.horizon_constraints:
        if con.p > 0:
            retained_lam.append(lam_canon[offset : offset + con.p])
            offset += con.p

    lam_user = np.concatenate(retained_lam) if retained_lam else np.zeros(0, dtype=np.float64)

    _, U_mu = _z_to_trajectory(jnp.asarray(mu_canon), N, n, m)
    mu_u = np.asarray(U_mu, dtype=np.float64).reshape(-1)

    return lam_user, mu_u


def _unpack_retained_multipliers(
    problem: Problem,
    mult_g: np.ndarray | None,
) -> tuple[list[jax.Array], jax.Array]:
    """Unpack raw solver constraint Multipliers into per-Knot Point and Horizon arrays."""
    knot_evaluators = problem.constraints.knot_evaluators
    horizon_constraints = problem.constraints.horizon_constraints
    N = int(problem.N)

    mult_g_arr = np.asarray(mult_g, dtype=np.float64) if mult_g is not None else _EMPTY
    offset = 0
    lam_c: list[jax.Array] = []
    for k in range(N - 1):
        pk = knot_evaluators[k].p if k < len(knot_evaluators) else 0
        if pk > 0 and offset + pk <= len(mult_g_arr):
            lam_c.append(jnp.asarray(mult_g_arr[offset : offset + pk], dtype=jnp.float64))
            offset += pk
        else:
            lam_c.append(jnp.zeros(pk, dtype=jnp.float64))

    p_term = knot_evaluators[N - 1].p if len(knot_evaluators) > N - 1 else 0
    if p_term > 0 and offset + p_term <= len(mult_g_arr):
        lam_c.append(jnp.asarray(mult_g_arr[offset : offset + p_term], dtype=jnp.float64))
        offset += p_term
    else:
        lam_c.append(jnp.zeros(p_term, dtype=jnp.float64))

    p_hz = sum(con.p for con in horizon_constraints)
    if p_hz > 0 and offset + p_hz <= len(mult_g_arr):
        lam_hz = jnp.asarray(mult_g_arr[offset : offset + p_hz], dtype=jnp.float64)
    else:
        lam_hz = jnp.zeros(0, dtype=jnp.float64)

    return lam_c, lam_hz


def _state_stationarity_rhs(
    problem: Problem,
    traj: Trajectory,
    lam_c: list[jax.Array],
    lam_hz: jax.Array,
) -> jax.Array:
    """Evaluate sum of Objective and constraint gradients with respect to states."""
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    knot_evaluators = problem.constraints.knot_evaluators
    X_opt, U_opt, t_opt = traj.X, traj.U, traj.t

    def cost_of_x(X_var: jax.Array) -> jax.Array:
        stage_costs = problem.obj.stage_cost.stage_costs(X_var[:-1], U_opt, t_opt[:-1])
        term_cost = problem.obj.terminal_cost.evaluate(X_var[-1], None, t_opt[-1])
        return jnp.sum(stage_costs) + term_cost

    grad_cost_x = jax.grad(cost_of_x)(X_opt)

    grad_con_x_list: list[jax.Array] = []
    for k in range(N - 1):
        pk = knot_evaluators[k].p if k < len(knot_evaluators) else 0
        if pk > 0:
            jx_k, _ = knot_evaluators[k].jacobian(X_opt[k], U_opt[k], t_opt[k])
            grad_con_x_list.append(jx_k.T @ lam_c[k])
        else:
            grad_con_x_list.append(jnp.zeros(n, dtype=jnp.float64))

    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        jx_term, _ = knot_evaluators[N - 1].jacobian(X_opt[-1], None, t_opt[-1])
        grad_con_x_list.append(jx_term.T @ lam_c[N - 1])
    else:
        grad_con_x_list.append(jnp.zeros(n, dtype=jnp.float64))

    grad_con_x = jnp.stack(grad_con_x_list, axis=0)

    p_hz = sum(con.p for con in problem.constraints.horizon_constraints)
    if p_hz > 0:
        z_opt = _trajectory_to_z(X_opt, U_opt)
        grad_hz_z = jnp.zeros(len(z_opt), dtype=jnp.float64)
        hz_offset = 0
        for con in problem.constraints.horizon_constraints:
            con_lam = lam_hz[hz_offset : hz_offset + con.p]
            hz_offset += con.p
            grad_hz_z = grad_hz_z.at[jnp.asarray(con.inds)].add(con.A.T @ con_lam)
        grad_hz_x, _ = _z_to_trajectory(grad_hz_z, N, n, m)
    else:
        grad_hz_x = jnp.zeros((N, n), dtype=jnp.float64)

    return grad_cost_x + grad_con_x + grad_hz_x


def _backward_adjoint_recursion(
    problem: Problem,
    traj: Trajectory,
    rhs: jax.Array,
) -> tuple[jax.Array, list[jax.Array]]:
    """Recover initial pin and Defect Multipliers via backward adjoint recursion."""
    N = int(problem.N)
    n = int(problem.model.n)
    X_opt, U_opt, t_opt, dt_arr = traj.X, traj.U, traj.t, traj.dt
    lam_d: list[jax.Array] = [jnp.zeros(n, dtype=jnp.float64) for _ in range(N - 1)]
    lam_d[N - 2] = -rhs[N - 1]
    for k in range(N - 2, 0, -1):
        ak = problem.model.state_jacobian(X_opt[k], U_opt[k], t_opt[k], dt_arr[k])
        lam_d[k - 1] = ak.T @ lam_d[k] - rhs[k]

    a0 = problem.model.state_jacobian(X_opt[0], U_opt[0], t_opt[0], dt_arr[0])
    lam_0 = a0.T @ lam_d[0] - rhs[0]
    return lam_0, lam_d


def _assemble_canonical_lam(
    problem: Problem,
    lam_0: jax.Array,
    lam_d: list[jax.Array],
    lam_c: list[jax.Array],
    lam_hz: jax.Array,
) -> np.ndarray:
    """Assemble recovered and retained Multipliers into canonical constraint order."""
    N = int(problem.N)
    knot_evaluators = problem.constraints.knot_evaluators
    lam_parts: list[np.ndarray] = [np.asarray(lam_0, dtype=np.float64)]
    for k in range(N - 1):
        lam_parts.append(np.asarray(lam_d[k], dtype=np.float64))
        pk = knot_evaluators[k].p if k < len(knot_evaluators) else 0
        if pk > 0:
            lam_parts.append(np.asarray(lam_c[k], dtype=np.float64))
    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        lam_parts.append(np.asarray(lam_c[N - 1], dtype=np.float64))
    if len(lam_hz) > 0:
        lam_parts.append(np.asarray(lam_hz, dtype=np.float64))
    return np.concatenate(lam_parts)


def _assemble_canonical_mu(
    mult_x_l_raw: object,
    mult_x_u_raw: object,
    N: int,
    n: int,
    m: int,
) -> np.ndarray:
    """Assemble control-bound Multipliers into canonical Primal Vector bound layout."""
    mult_x_l_out = np.asarray(mult_x_l_raw, dtype=np.float64) if mult_x_l_raw is not None else _EMPTY
    mult_x_u_out = np.asarray(mult_x_u_raw, dtype=np.float64) if mult_x_u_raw is not None else _EMPTY
    if len(mult_x_l_out) == (N - 1) * m and len(mult_x_u_out) == (N - 1) * m:
        mu_u = mult_x_u_out - mult_x_l_out
        u_mu = jnp.asarray(mu_u.reshape(N - 1, m), dtype=jnp.float64)
        x_mu = jnp.zeros((N, n), dtype=jnp.float64)
        return np.asarray(_trajectory_to_z(x_mu, u_mu), dtype=np.float64)
    return _EMPTY


def _recover_canonical_multipliers(
    problem: Problem,
    traj: Trajectory,
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Recover canonical constraint and bound Multipliers from state stationarity."""
    mult_g = info.get("mult_g")
    mult_x_l_raw = info.get("mult_x_L")
    mult_x_u_raw = info.get("mult_x_U")
    if mult_g is None and mult_x_l_raw is None:
        return _EMPTY, _EMPTY

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    lam_c, lam_hz = _unpack_retained_multipliers(problem, mult_g)
    rhs = _state_stationarity_rhs(problem, traj, lam_c, lam_hz)
    lam_0, lam_d = _backward_adjoint_recursion(problem, traj, rhs)
    lam_canon = _assemble_canonical_lam(problem, lam_0, lam_d, lam_c, lam_hz)
    mu_canon = _assemble_canonical_mu(mult_x_l_raw, mult_x_u_raw, N, n, m)
    return lam_canon, mu_canon


class _SingleShootingCallback:
    """Callback wrapper connecting the single-shooting JAX kernels to cyipopt."""

    def __init__(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: jax.Array,
        dt: jax.Array,
        xf: jax.Array | None = None,
    ) -> None:
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf

        n_u, p_user = single_shooting_dimensions(problem)
        self.n_u = n_u
        self.p_user = p_user
        self.jac_rows, self.jac_cols = _dense_jacobian_pattern(p_user, n_u)
        hess_rows, hess_cols = np.tril_indices(n_u)
        self.hess_rows = hess_rows.astype(np.int32)
        self.hess_cols = hess_cols.astype(np.int32)
        self.iteration_count = 0

        N = int(problem.N)
        b = int(problem.control_block_steps)
        m = int(problem.model.m)
        B = (N - 1) // b
        self.binary_u_indices = np.asarray(
            [k * m + i for k in range(B) for i in problem.binary_control_indices],
            dtype=np.int32,
        )
        self.beta = 0.0

    def update(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: jax.Array,
        dt: jax.Array,
        xf: jax.Array | None = None,
    ) -> None:
        """Update dynamic problem and Boundary Conditions and reset per-solve state."""
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf
        self.beta = 0.0
        self.iteration_count = 0

    @property
    def binary_indices(self) -> np.ndarray:
        """Return indices of binary decision variables in the reduced control vector."""
        return self.binary_u_indices

    def intermediate(self, *args: object) -> bool:
        """Track solver iteration counter from Ipopt."""
        if len(args) > 1 and isinstance(args[1], (int, float, str)):
            self.iteration_count = int(args[1])
        return True

    def objective(self, u: np.ndarray) -> float:
        """Evaluate scalar objective value J(u) plus binary homotopy penalty."""
        b = int(self.problem.control_block_steps)
        a = u[self.binary_u_indices]
        cost = float(eval_f(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt))
        if len(a) and self.beta != 0.0:
            return cost + b * self.beta * float(np.sum(a * (1 - a)))
        return cost

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient nabla J(u) plus binary penalty gradient."""
        b = int(self.problem.control_block_steps)
        grad = np.asarray(
            eval_grad_f(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt),
            dtype=np.float64,
        ).copy()
        if len(self.binary_u_indices) and self.beta != 0.0:
            grad[self.binary_u_indices] += b * self.beta * (1 - 2 * u[self.binary_u_indices])
        return grad

    def constraints(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraint vector c(u)."""
        return np.asarray(
            eval_g(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt, xf=self.xf),
            dtype=np.float64,
        )

    def jacobian(self, u: np.ndarray) -> np.ndarray:
        """Evaluate dense constraint Jacobian values."""
        return np.asarray(
            eval_jac_g(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt, xf=self.xf),
            dtype=np.float64,
        )

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time constraint Jacobian sparsity pattern (rows, cols)."""
        return self.jac_rows, self.jac_cols

    def hessian(self, u: np.ndarray, lagrange: np.ndarray, obj_factor: float = 1.0) -> np.ndarray:
        """Evaluate lower-triangular nonzeros of the dense Lagrangian Hessian including binary penalty."""
        b = int(self.problem.control_block_steps)
        values = np.asarray(
            eval_h(
                self.problem,
                jnp.asarray(u),
                self.x0,
                t0=self.t0,
                dt=self.dt,
                obj_factor=jnp.asarray(obj_factor, dtype=jnp.float64),
                lam=jnp.asarray(lagrange),
                xf=self.xf,
            ),
            dtype=np.float64,
        ).copy()
        if len(self.binary_u_indices) and self.beta != 0.0:
            diagonal = self.hess_rows == self.hess_cols
            binary = np.isin(self.hess_rows, self.binary_u_indices)
            values[diagonal & binary] -= 2 * b * self.beta * obj_factor
        return values

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time Lagrangian Hessian sparsity pattern (rows, cols)."""
        return self.hess_rows, self.hess_cols


@dataclass
class SingleShootingHandle:
    """Retained cyipopt Problem handle and callback wrapper for a SingleShooting Program."""

    nlp: Any
    cb: _SingleShootingCallback

    @property
    def solver(self) -> Any:  # noqa: ANN401 -- cyipopt's C-extension handle is untyped
        """Alias for the underlying cyipopt Problem handle."""
        return self.nlp


@dataclass(frozen=True)
class SingleShooting:
    """Single-shooting transcription: solve over controls only, with states recovered by rollout.

    Parameters
    ----------
    solver : Ipopt
        NLP solver whose options are forwarded; only Ipopt is supported.
    hessian : Literal["lbfgs", "dense"], optional
        ``"lbfgs"`` (default) sets Ipopt's limited-memory approximation; ``"dense"`` supplies the
        exact dense Lagrangian Hessian.
    """

    solver: Ipopt
    hessian: Literal["lbfgs", "dense"] = "lbfgs"

    def __post_init__(self) -> None:
        """Refuse non-Ipopt solvers and unknown Hessian modes at construction."""
        if not isinstance(self.solver, Ipopt):
            msg = f"SingleShooting supports only Ipopt, got {type(self.solver).__name__}."
            raise TypeError(msg)
        if self.hessian not in ("lbfgs", "dense"):
            msg = f"hessian must be 'lbfgs' or 'dense', got {self.hessian!r}."
            raise ValueError(msg)

    @property
    def beta_initial(self) -> float:
        """Initial binary continuation penalty coefficient."""
        return float(self.solver.beta_initial)

    @property
    def beta_growth(self) -> float:
        """Growth factor for binary continuation penalty."""
        return float(self.solver.beta_growth)

    @property
    def max_passes(self) -> int:
        """Maximum number of binary continuation passes."""
        return int(self.solver.max_passes)

    @property
    def binary_tolerance(self) -> float:
        """Integrality tolerance for binary continuation."""
        return float(self.solver.binary_tolerance)

    def transcription_callback(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: jax.Array,
        dt: jax.Array,
        xf: jax.Array | None = None,
    ) -> _SingleShootingCallback:
        """Assemble callback wrapper connecting single-shooting Problem to cyipopt."""
        return _SingleShootingCallback(problem=problem, x0=x0, t0=t0, dt=dt, xf=xf)

    def _solve_passes(
        self,
        nlp: Any,  # noqa: ANN401 -- cyipopt's C-extension handle is untyped
        cb: _SingleShootingCallback,
        u0: np.ndarray,
        lam0: np.ndarray | None,
        mu0: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any], int, float]:
        """Continue from relaxation through penalty passes, carrying Primal Vector and Multipliers."""
        return solve_continuation_passes(
            nlp,
            cb,
            u0,
            lam0,
            mu0,
            binary_indices=cb.binary_u_indices,
            beta_initial=self.solver.beta_initial,
            beta_growth=self.solver.beta_growth,
            max_passes=self.solver.max_passes,
            binary_tolerance=self.solver.binary_tolerance,
        )

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
        """Solve the single-shooting Transcription of `program`'s Problem from `bc` and `ws` using Ipopt."""
        problem = program.problem
        if problem.binary_control_indices:
            validate_homotopy_parameters(
                self.solver.beta_initial,
                self.solver.beta_growth,
                self.solver.max_passes,
                self.solver.binary_tolerance,
            )
        _validate_supported_constraints(problem)

        try:
            import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency
        except ImportError as e:
            msg = (
                "cyipopt is not installed. It is part of the `solvers` extra: install with "
                '`pip install "trajopt[solvers]"` or `uv add "trajopt[solvers]"`, which compiles '
                "cyipopt against a system Ipopt (see the README's Installation section). "
                "Alternatively, drive the problem with a different solver, e.g. OSQP() or ALTRO()."
            )
            raise ImportError(msg) from e

        N = int(problem.N)
        b = int(problem.control_block_steps)

        x0_arr, t0_arr, dt_arr, xf_val, _ = parse_solver_initial_state(problem, bc, ws)
        problem = retarget_problem(problem, bc)
        dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))

        _, U_guess = ws.unpack(problem)
        u0 = np.asarray(reduce_controls(U_guess, b), dtype=np.float64).reshape(-1)
        handle: SingleShootingHandle | None = program.handles.get("single_shooting")
        if handle is None:
            z_lower, z_upper = _primal_bounds(problem)
            g_lower, g_upper = _constraint_bounds(problem)
            cb = self.transcription_callback(problem, x0_arr, t0_arr, dt_arr, xf_val)
            problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt is an untyped C-extension
            nlp = problem_cls(
                n=len(u0),
                m=len(g_lower),
                problem_obj=cb,
                lb=z_lower,
                ub=z_upper,
                cl=g_lower,
                cu=g_upper,
            )
            options: dict[str, Any] = dict(self.solver.options)
            if self.hessian == "lbfgs":
                options.setdefault("hessian_approximation", "limited-memory")
            for key, value in options.items():
                nlp.add_option(key, value)
            handle = SingleShootingHandle(nlp=nlp, cb=cb)
            program.handles["single_shooting"] = handle
            program.handles.setdefault("ipopt", handle)
        else:
            cb = handle.cb
            nlp = handle.nlp
            cb.update(problem, x0_arr, t0_arr, dt_arr, xf_val)

        if b == 1:
            lam0, mu0 = _extract_retained_warm_start_duals(problem, ws)
        else:
            lam0, mu0 = None, None

        u_opt, info, pass_count, binary_distance = self._solve_passes(nlp, cb, u0, lam0, mu0)
        return self._build_result(
            problem,
            cb,
            u_opt,
            info,
            pass_count,
            binary_distance,
            x0_arr=x0_arr,
            t0_arr=t0_arr,
            dt_arr=dt_arr,
            xf_val=xf_val,
        )

    def _build_result(  # noqa: PLR0913, PLR0917 -- packaging result needs solve metadata
        self,
        problem: Problem,
        cb: _SingleShootingCallback,
        u_opt: np.ndarray,
        info: dict[str, Any],
        pass_count: int,
        binary_distance: float,
        *,
        x0_arr: jax.Array,
        t0_arr: jax.Array,
        dt_arr: jax.Array,
        xf_val: jax.Array | None,
    ) -> IpoptResult:
        """Expand optimal block controls, roll out native states, and package canonical IpoptResult."""
        N = int(problem.N)
        b = int(problem.control_block_steps)
        n = int(problem.model.n)
        m = int(problem.model.m)
        B = (N - 1) // b

        status = int(info.get("status", -1))
        message = str(info.get("status_msg", ""))

        U_blk_opt = jnp.asarray(u_opt, dtype=jnp.float64).reshape((B, m))
        U_opt = expand_controls(U_blk_opt, b)
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])
        X_opt = rollout_states(problem.model, x0_arr, U_opt, t=t_opt, dt=dt_arr)

        opt_traj = Trajectory(X=X_opt, U=U_opt, t=t_opt, dt=dt_arr)
        Z_opt_jax = _trajectory_to_z(X_opt, U_opt)

        cost_val = float(eval_f(problem, jnp.asarray(u_opt), x0_arr, t0=t0_arr, dt=dt_arr))

        a = u_opt[cb.binary_u_indices] if len(cb.binary_u_indices) else _EMPTY
        info.update(
            homotopy_beta=cb.beta,
            homotopy_penalty=b * cb.beta * float(np.sum(a * (1 - a))) if len(a) else 0.0,
            binary_distance=binary_distance,
            homotopy_passes=pass_count,
        )

        iter_count = int(cb.iteration_count)
        viol = compute_constraint_violation(
            problem,
            Z_opt_jax,
            x0_arr,
            t0=t0_arr,
            dt=dt_arr,
            xf=xf_val,
        )

        success = status in _SUCCESS_STATUSES and (
            not problem.binary_control_indices
            or (pass_count > 1 and binary_distance <= self.solver.binary_tolerance)
        )
        if problem.binary_control_indices:
            feasibility_tol = float(self.solver.options.get("constr_viol_tol", 1e-4))
            success = success and viol <= feasibility_tol

        if b > 1:
            P_total = constraint_row_count(problem)
            nz_primal = N * n + (N - 1) * m
            lam_out = np.zeros(P_total, dtype=np.float64)
            mu_out = np.zeros(nz_primal, dtype=np.float64)
        else:
            lam_out, mu_out = _recover_canonical_multipliers(
                problem,
                opt_traj,
                info,
            )

        return IpoptResult(
            trajectory=opt_traj,
            success=success,
            status=status,
            message=message,
            cost=cost_val,
            Z=Z_opt_jax,
            info=info,
            iterations=iter_count,
            constraint_violation=viol,
            lam=lam_out,
            mu=mu_out,
        )
