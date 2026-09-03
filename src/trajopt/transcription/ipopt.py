from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.problem import MPCState, Problem, retarget_problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import (
    _z_to_trajectory,
    compute_constraint_violation,
    constraint_bounds,
    parse_solver_initial_state,
    primal_bounds,
)
from trajopt.transcription.result import split_bound_duals, warm_start_duals
from trajopt.transcription.sparsity import (
    hessian_sparsity_pattern,
    jacobian_sparsity_pattern,
)
from trajopt.transcription.transcription import (
    eval_f,
    eval_g,
    eval_grad_f,
    eval_h,
    eval_jac_g,
)

_SUCCESS_STATUSES = {0, 1}  # 0: Solve_Succeeded, 1: Solved_To_Acceptable_Level
_EMPTY = np.zeros(0, dtype=np.float64)


class IpoptResult(NamedTuple):
    """Result of an Ipopt trajectory optimization solve.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the solver converged to optimality or acceptable level.
    status : int
        Ipopt integer status code.
    message : str
        Solver return message.
    cost : float
        Final objective value.
    Z : jax.Array
        Optimal flat primal vector.
    info : dict[str, Any]
        Raw Ipopt return info dictionary.
    constraint_violation : float
        Maximum constraint violation across all constraints.
    iterations : int, optional
        Number of solver iterations. Defaults to 0.
    lam : np.ndarray, optional
        Constraint duals in canonical row order, of shape ``(P,)``. Defaults to empty.
    mu : np.ndarray, optional
        Signed bound duals ``mult_x_U - mult_x_L`` of shape ``(N * n + (N - 1) * m,)``.
        Defaults to empty.
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


class _IpoptCallback:
    """Callback wrapper connecting JAX transcription kernels to cyipopt."""

    def __init__(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: float | jax.Array,
        dt: float | jax.Array,
        xf: jax.Array | None = None,
    ) -> None:
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf

        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)
        p_seq = tuple(int(pk) for pk in problem.constraints.p)

        self.jac_rows, self.jac_cols = jacobian_sparsity_pattern(N, n, m, p_seq)
        self.hess_rows, self.hess_cols = hessian_sparsity_pattern(N, n, m)
        self.iteration_count = 0

    def intermediate(self, *args: object) -> bool:
        """Track solver iteration counter from Ipopt."""
        if len(args) > 1 and isinstance(args[1], (int, float, str)):
            self.iteration_count = int(args[1])
        return True

    def objective(self, z: np.ndarray) -> float:
        """Evaluate scalar objective value J(z)."""
        return float(eval_f(self.problem, jnp.asarray(z), self.t0, self.dt))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient nabla J(z)."""
        return np.asarray(eval_grad_f(self.problem, jnp.asarray(z), self.t0, self.dt), dtype=np.float64)

    def constraints(self, z: np.ndarray) -> np.ndarray:
        """Evaluate constraint vector c(z)."""
        return np.asarray(eval_g(self.problem, jnp.asarray(z), self.x0, self.t0, self.dt, xf=self.xf), dtype=np.float64)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Evaluate sparse constraint Jacobian values."""
        return np.asarray(
            eval_jac_g(self.problem, jnp.asarray(z), self.x0, self.t0, self.dt, xf=self.xf), dtype=np.float64
        )

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time constraint Jacobian sparsity pattern (rows, cols)."""
        return self.jac_rows, self.jac_cols

    def hessian(self, z: np.ndarray, lagrange: np.ndarray, obj_factor: float = 1.0) -> np.ndarray:
        """Evaluate lower-triangular nonzeros of the Lagrangian Hessian."""
        return np.asarray(
            eval_h(
                self.problem,
                jnp.asarray(z),
                t0=self.t0,
                dt=self.dt,
                # As a Python float this is static under filter_jit, so each objective scaling
                # factor Ipopt picks would compile the Hessian afresh.
                obj_factor=jnp.asarray(obj_factor, dtype=jnp.float64),
                lam=jnp.asarray(lagrange),
            ),
            dtype=np.float64,
        )

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time Lagrangian Hessian sparsity pattern (rows, cols)."""
        return self.hess_rows, self.hess_cols


@dataclass(frozen=True)
class Ipopt:
    """Ipopt nonlinear interior-point solver backend, solving the nonlinear problem directly.

    Parameters
    ----------
    options : Mapping[str, Any], optional
        Native cyipopt options (e.g. {"max_iter": 200, "tol": 1e-4, "print_level": 0}).
        Defaults to empty.
    """

    options: Mapping[str, Any] = field(default_factory=dict)

    def transcription_callback(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: float | jax.Array,
        dt: float | jax.Array,
        xf: jax.Array | None = None,
    ) -> _IpoptCallback:
        """Assemble the callback wrapper connecting the transcribed problem to cyipopt.

        Exposed so callers (e.g. benchmarks) can measure transcription setup cost without
        reaching into solver internals.
        """
        return _IpoptCallback(problem=problem, x0=x0, t0=t0, dt=dt, xf=xf)

    def solve(self, problem: Problem, state: MPCState) -> IpoptResult:
        """Solve the transcribed optimal control problem using Ipopt via cyipopt."""
        try:
            import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency
        except ImportError as e:
            msg = (
                "cyipopt is not installed. It is part of the `solvers` extra: install with "
                '`pip install "trajopt[solvers]"` or `uv add "trajopt[solvers]"`, which compiles '
                "cyipopt against a system Ipopt (see the README's Installation section). "
                "Alternatively, pass a different solver to Problem.solve, e.g. OSQP() or ALTRO()."
            )
            raise ImportError(msg) from e

        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)

        x0_arr, t0_arr, dt_arr, xf_val, z0_jax = parse_solver_initial_state(state)
        problem = retarget_problem(problem, state.bc)
        dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))
        z0 = np.asarray(z0_jax, dtype=np.float64)
        lam0, mu0 = warm_start_duals(problem, state)

        # Bounds
        zL, zU = primal_bounds(problem)
        gL, gU = constraint_bounds(problem)

        cb = self.transcription_callback(problem, x0_arr, t0_arr, dt_arr, xf_val)

        problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt is an untyped C-extension
        nlp = problem_cls(
            n=len(z0),
            m=len(gL),
            problem_obj=cb,
            lb=zL,
            ub=zU,
            cl=gL,
            cu=gU,
        )

        if self.options:
            for k, v in self.options.items():
                nlp.add_option(k, v)

        if lam0 is not None and mu0 is not None:
            # Ipopt only honours the supplied multipliers with the warm-start option set; the
            # push factors keep it from shoving them back off the bounds it just accepted.
            nlp.add_option("warm_start_init_point", "yes")
            nlp.add_option("warm_start_bound_push", 1e-9)
            nlp.add_option("warm_start_mult_bound_push", 1e-9)
            mult_x_L, mult_x_U = split_bound_duals(mu0)
            z_opt, info = nlp.solve(z0, lagrange=lam0, zl=mult_x_L, zu=mult_x_U)
        else:
            z_opt, info = nlp.solve(z0)
        status = int(info.get("status", -1))
        success = status in _SUCCESS_STATUSES
        message = str(info.get("status_msg", ""))
        cost_val = float(info.get("obj_val", cb.objective(z_opt)))

        Z_opt_jax = jnp.asarray(z_opt, dtype=jnp.float64)
        X_opt, U_opt = _z_to_trajectory(Z_opt_jax, N, n, m)
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])

        opt_traj = Trajectory(
            X=X_opt,
            U=U_opt,
            t=t_opt,
            dt=dt_arr,
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

        lam_out = np.asarray(info.get("mult_g", _EMPTY), dtype=np.float64)
        mult_x_L_out = np.asarray(info.get("mult_x_L", _EMPTY), dtype=np.float64)
        mult_x_U_out = np.asarray(info.get("mult_x_U", _EMPTY), dtype=np.float64)
        mu_out = mult_x_U_out - mult_x_L_out

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
