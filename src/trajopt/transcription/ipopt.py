from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.problem import MPCState, Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import (
    constraint_bounds,
    primal_bounds,
    trajectory_to_z,
    z_to_trajectory,
)
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
    """

    trajectory: Trajectory
    success: bool
    status: int
    message: str
    cost: float
    Z: jax.Array
    info: dict[str, Any]


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

    def objective(self, z: np.ndarray) -> float:
        """Evaluate scalar objective value J(z)."""
        return float(eval_f(self.problem, jnp.asarray(z), self.t0, self.dt, self.xf))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient nabla J(z)."""
        return np.asarray(eval_grad_f(self.problem, jnp.asarray(z), self.t0, self.dt, self.xf), dtype=np.float64)

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
                obj_factor=obj_factor,
                lam=jnp.asarray(lagrange),
                xf=self.xf,
            ),
            dtype=np.float64,
        )

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time Lagrangian Hessian sparsity pattern (rows, cols)."""
        return self.hess_rows, self.hess_cols


def solve_ipopt(  # noqa: PLR0913 -- solver configuration takes 8 arguments
    problem: Problem,
    x0: jax.Array | MPCState,
    *,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    initial_trajectory: Trajectory | None = None,
    initial_z: jax.Array | None = None,
    xf: jax.Array | None = None,
    options: Mapping[str, Any] | None = None,
) -> IpoptResult:
    """Solve the transcribed optimal control problem using Ipopt via cyipopt.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, objective, constraints, and horizon.
    x0 : jax.Array | MPCState
        Initial state condition of shape (n,) or an MPCState instance.
    t0 : float | jax.Array, optional
        Initial timestamp. Defaults to 0.0.
    dt : float | jax.Array, optional
        Step duration (scalar or array of length N-1). Defaults to 0.05.
    initial_trajectory : Trajectory | None, optional
        Initial trajectory guess. Defaults to repeating x0 with zero controls.
    initial_z : jax.Array | None, optional
        Flat initial guess vector of shape (N * n + (N - 1) * m,).
    xf : jax.Array | None, optional
        Goal state vector. Defaults to None.
    options : Mapping[str, Any] | None, optional
        Solver options passed to cyipopt (e.g. {"max_iter": 200, "tol": 1e-4, "print_level": 0}).

    Returns
    -------
    IpoptResult
        Optimization result including optimal trajectory, convergence flag, status, and cost.
    """
    import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency

    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    if isinstance(x0, MPCState):
        # x0 is an MPCState instance
        x0_arr = jnp.asarray(x0.x0, dtype=jnp.float64)
        t0_arr = jnp.asarray(x0.t0, dtype=jnp.float64)
        dt_arr = jnp.broadcast_to(jnp.asarray(x0.dt, dtype=jnp.float64), (N - 1,))
        xf_val = x0.xf
        z0 = np.asarray(x0.Z, dtype=np.float64)
    else:
        x0_arr = jnp.asarray(x0, dtype=jnp.float64)
        t0_arr = jnp.asarray(t0, dtype=jnp.float64)
        dt_arr = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N - 1,))
        xf_val = None if xf is None else jnp.asarray(xf, dtype=jnp.float64)
        if initial_z is not None:
            z0 = np.asarray(initial_z, dtype=np.float64)
        elif initial_trajectory is not None:
            z0 = np.asarray(trajectory_to_z(initial_trajectory.X, initial_trajectory.U), dtype=np.float64)
        else:
            X0 = jnp.repeat(x0_arr[None, :], N, axis=0)
            U0 = jnp.zeros((N - 1, m), dtype=jnp.float64)
            z0 = np.asarray(trajectory_to_z(X0, U0), dtype=np.float64)

    # Bounds
    zL, zU = primal_bounds(problem)
    gL, gU = constraint_bounds(problem)

    cb = _IpoptCallback(problem=problem, x0=x0_arr, t0=t0_arr, dt=dt_arr, xf=xf_val)

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

    if options:
        for k, v in options.items():
            nlp.add_option(k, v)

    z_opt, info = nlp.solve(z0)
    status = int(info.get("status", -1))
    success = status in _SUCCESS_STATUSES
    message = str(info.get("status_msg", ""))
    cost_val = float(info.get("obj_val", cb.objective(z_opt)))

    Z_opt_jax = jnp.asarray(z_opt, dtype=jnp.float64)
    X_opt, U_opt = z_to_trajectory(Z_opt_jax, N, n, m)
    t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])

    opt_traj = Trajectory(
        X=X_opt,
        U=U_opt,
        t=t_opt,
        dt=dt_arr,
    )

    return IpoptResult(
        trajectory=opt_traj,
        success=success,
        status=status,
        message=message,
        cost=cost_val,
        Z=Z_opt_jax,
        info=info,
    )
