"""Single-shooting transcription: the NLP primal is the control trajectory alone."""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.dynamics.rollout import rollout_states
from trajopt.problem import MPCState, Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.layout import (
    _evaluator_bounds,
    _trajectory_to_z,
    parse_solver_initial_state,
)

_SUCCESS_STATUSES = {0, 1}  # Ipopt Solve_Succeeded and Solved_To_Acceptable_Level
_EMPTY = np.zeros(0, dtype=np.float64)


def _shooting_rollout(
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Roll out states from x0 and a flat control vector, returning (X, U, t)."""
    N = int(problem.N)
    m = int(problem.model.m)
    U = u.reshape((N - 1, m))
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=u.dtype), jnp.cumsum(dt_arr)])
    X = rollout_states(problem.model, x0, U, t=t_arr, dt=dt_arr)
    return X, U, t_arr


def _cost_fn(  # noqa: PLR0913, PLR0917 -- Cost evaluation takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate total cost as a function of the flat control vector alone."""
    X, U, t_arr = _shooting_rollout(problem, u, x0, t0, dt)
    obj = problem.obj.with_goal(xf) if (xf is not None and problem.obj.regulates_to_goal) else problem.obj
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
    knot_evaluators = problem.constraints.knot_evaluators

    c_list: list[jax.Array] = []
    for k in range(N - 1):
        evaluator = knot_evaluators[k]
        if evaluator.p > 0:
            c_list.append(evaluator.evaluate(X[k], U[k], t_arr[k], xf=xf))
    if len(knot_evaluators) > N - 1 and knot_evaluators[N - 1].p > 0:
        c_list.append(knot_evaluators[N - 1].evaluate(X[-1], None, t_arr[-1], xf=xf))

    if c_list:
        return jnp.concatenate(c_list)
    return jnp.zeros(0, dtype=u.dtype)


@eqx.filter_jit
def eval_f(  # noqa: PLR0913, PLR0917 -- Objective callback takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate objective value J(u) as a scalar."""
    return _cost_fn(problem, u, x0, t0, dt, xf)


@eqx.filter_jit
def eval_grad_f(  # noqa: PLR0913, PLR0917 -- Gradient callback takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate objective gradient nabla J(u) of shape ((N-1)*m,)."""
    return jax.grad(lambda u_: _cost_fn(problem, u_, x0, t0, dt, xf))(u)


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
    """Evaluate dense constraint Jacobian values of shape (P_user * (N-1) * m,)."""
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
    n_u = (int(problem.N) - 1) * int(problem.model.m)
    p_user = int(sum(problem.constraints.p))
    lam_vec = jnp.zeros(p_user, dtype=u.dtype) if lam is None else jnp.asarray(lam, dtype=u.dtype)

    def lagrangian(u_: jax.Array) -> jax.Array:
        cost = _cost_fn(problem, u_, x0, t0, dt, xf)
        cons = _constraints_fn(problem, u_, x0, t0, dt, xf=xf)
        return obj_factor * cost + jnp.dot(lam_vec, cons)

    hess = jax.hessian(lagrangian)(u)
    tril_r, tril_c = np.tril_indices(n_u)
    return hess[tril_r, tril_c]


def single_shooting_dimensions(problem: Problem) -> tuple[int, int]:
    """Return the single-shooting NLP's primal and constraint row counts (n_u, p_user)."""
    n_u = (int(problem.N) - 1) * int(problem.model.m)
    p_user = int(sum(problem.constraints.p))
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
    knot_evaluators = problem.constraints.knot_evaluators

    g_lower: list[np.ndarray] = []
    g_upper: list[np.ndarray] = []
    for k in range(N - 1):
        lo_k, hi_k = _evaluator_bounds(knot_evaluators[k])
        g_lower.extend(lo_k)
        g_upper.extend(hi_k)
    if len(knot_evaluators) > N - 1:
        lo_term, hi_term = _evaluator_bounds(knot_evaluators[N - 1])
        g_lower.extend(lo_term)
        g_upper.extend(hi_term)

    g_l = np.concatenate(g_lower) if g_lower else np.empty(0, dtype=np.float64)
    g_u = np.concatenate(g_upper) if g_upper else np.empty(0, dtype=np.float64)
    return g_l, g_u


def _primal_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Return control-only primal bounds zL <= u <= zU of shape ((N-1)*m,)."""
    _, _, u_lower, u_upper = problem.constraints.primal_bounds()
    return u_lower.reshape(-1), u_upper.reshape(-1)


def _dense_jacobian_pattern(p: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense COO sparsity pattern for a (p, n) Jacobian, in row-major order."""
    if p == 0 or n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    rows = np.repeat(np.arange(p, dtype=np.int32), n)
    cols = np.tile(np.arange(n, dtype=np.int32), p)
    return rows, cols


def _compute_constraint_violation(  # noqa: PLR0913, PLR0917 -- Violation metric takes 6 arguments
    problem: Problem,
    u: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None,
) -> float:
    """Maximum violation over the control bounds and the user constraint rows."""
    c = np.asarray(eval_g(problem, jnp.asarray(u), x0, t0=t0, dt=dt, xf=xf), dtype=np.float64)
    max_con = float(np.max(np.abs(c))) if c.size else 0.0

    u_lower, u_upper = _primal_bounds(problem)
    u_np = np.asarray(u, dtype=np.float64)
    viol_lo = np.maximum(0.0, u_lower - u_np)
    viol_hi = np.maximum(0.0, u_np - u_upper)
    max_bound = max(float(np.max(viol_lo)), float(np.max(viol_hi))) if u_np.size else 0.0
    return max(max_con, max_bound)


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

    def intermediate(self, *args: object) -> bool:
        """Track solver iteration counter from Ipopt."""
        if len(args) > 1 and isinstance(args[1], (int, float, str)):
            self.iteration_count = int(args[1])
        return True

    def objective(self, u: np.ndarray) -> float:
        """Evaluate scalar objective value J(u)."""
        return float(eval_f(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt, xf=self.xf))

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient nabla J(u)."""
        return np.asarray(
            eval_grad_f(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt, xf=self.xf),
            dtype=np.float64,
        )

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
        """Evaluate lower-triangular nonzeros of the dense Lagrangian Hessian."""
        return np.asarray(
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
        )

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time Lagrangian Hessian sparsity pattern (rows, cols)."""
        return self.hess_rows, self.hess_cols


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
    single_shooting: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Refuse non-Ipopt solvers and unknown Hessian modes at construction."""
        if not isinstance(self.solver, Ipopt):
            msg = f"SingleShooting supports only Ipopt, got {type(self.solver).__name__}."
            raise TypeError(msg)
        if self.hessian not in ("lbfgs", "dense"):
            msg = f"hessian must be 'lbfgs' or 'dense', got {self.hessian!r}."
            raise ValueError(msg)

    def transcription_callback(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: jax.Array,
        dt: jax.Array,
        xf: jax.Array | None = None,
    ) -> _SingleShootingCallback:
        """Assemble the callback wrapper connecting the single-shooting problem to cyipopt."""
        return _SingleShootingCallback(problem=problem, x0=x0, t0=t0, dt=dt, xf=xf)

    def solve(self, problem: Problem, state: MPCState) -> IpoptResult:
        """Solve the single-shooting transcription of `problem` from `state` using Ipopt."""
        import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency

        _validate_supported_constraints(problem)

        N = int(problem.N)
        m = int(problem.model.m)

        x0_arr, t0_arr, dt_arr, xf_val, _ = parse_solver_initial_state(state)
        dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))

        u0 = np.asarray(state.controls, dtype=np.float64).reshape(-1)
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

        u_opt, info = nlp.solve(u0)
        status = int(info.get("status", -1))
        success = status in _SUCCESS_STATUSES
        message = str(info.get("status_msg", ""))
        cost_val = float(info.get("obj_val", cb.objective(u_opt)))

        U_opt = jnp.asarray(u_opt, dtype=jnp.float64).reshape((N - 1, m))
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])
        X_opt = rollout_states(problem.model, x0_arr, U_opt, t=t_opt, dt=dt_arr)

        opt_traj = Trajectory(X=X_opt, U=U_opt, t=t_opt, dt=dt_arr)
        Z_opt_jax = _trajectory_to_z(X_opt, U_opt)

        iter_count = int(cb.iteration_count)
        viol = _compute_constraint_violation(problem, u_opt, x0_arr, t0_arr, dt_arr, xf_val)

        lam_out = np.asarray(info.get("mult_g", _EMPTY), dtype=np.float64)
        mult_x_l_out = np.asarray(info.get("mult_x_L", _EMPTY), dtype=np.float64)
        mult_x_u_out = np.asarray(info.get("mult_x_U", _EMPTY), dtype=np.float64)
        mu_out = mult_x_u_out - mult_x_l_out

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
