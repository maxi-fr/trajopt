from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

ArrayLike = Sequence[float] | np.ndarray | jax.Array

from trajopt.constraints.geometric import CircleConstraint, SphereConstraint
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.constraints.rotations import QuatVecEq
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost, QuadraticCost
from trajopt.costs.rotations import QuatGeodesicCost
from trajopt.dynamics.base import DiscretizedDynamics
from trajopt.models.cartpole import Cartpole
from trajopt.models.dubins import DubinsCar
from trajopt.models.pendulum import Pendulum
from trajopt.models.quadrotor import Quadrotor
from trajopt.problem import Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.result import SolverResult


class CasadiResult(NamedTuple):
    """Result of a CasADi trajectory optimization solve.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    cost : float
        Final objective value.
    success : bool
        Whether Ipopt converged successfully.
    status : int
        Solver exit status code.
    mult_g : np.ndarray
        Lagrange multipliers for equality and inequality constraints, shape (P,).
    max_constraint_residual : float
        Maximum constraint violation across all transcribed constraints.
    opti_sol : Any
        Raw CasADi OptiSol object.
    info : dict[str, Any]
        Solver information dictionary.
    """

    trajectory: Trajectory
    cost: float
    success: bool
    status: int
    mult_g: np.ndarray
    max_constraint_residual: float
    opti_sol: Any
    info: dict[str, Any]


def cartpole_dynamics(
    x: ca.MX,
    u: ca.MX,
    mc: float = 1.0,
    mp: float = 0.2,
    l: float = 0.5,  # noqa: E741 -- length parameter matching spec
    g: float = 9.81,
) -> ca.MX:
    """Evaluate continuous cartpole dynamics in CasADi."""
    p = x[0]  # noqa: F841 -- unused position state
    theta = x[1]
    p_dot = x[2]
    theta_dot = x[3]

    s = ca.sin(theta)
    c = ca.cos(theta)

    H = ca.vertcat(
        ca.horzcat(mc + mp, mp * l * c),
        ca.horzcat(mp * l * c, mp * (l**2)),
    )
    C = ca.vertcat(
        ca.horzcat(0.0, -mp * theta_dot * l * s),
        ca.horzcat(0.0, 0.0),
    )
    G = ca.vertcat(0.0, mp * g * l * s)
    B = ca.vertcat(1.0, 0.0)

    qd = ca.vertcat(p_dot, theta_dot)
    u_val = u[0] if u.numel() > 0 else u
    rhs = ca.mtimes(C, qd) + G - B * u_val
    qdd = -ca.solve(H, rhs)

    return ca.vertcat(p_dot, theta_dot, qdd[0], qdd[1])


def dubins_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
    """Evaluate continuous Dubins car dynamics in CasADi."""
    theta = x[2]
    v = u[0]
    omega = u[1]

    x_dot = v * ca.cos(theta)
    y_dot = v * ca.sin(theta)
    theta_dot = omega

    return ca.vertcat(x_dot, y_dot, theta_dot)


def pendulum_dynamics(
    x: ca.MX,
    u: ca.MX,
    mass: float = 1.0,
    lc: float = 0.5,
    b: float = 0.1,
    g: float = 9.81,
) -> ca.MX:
    """Evaluate continuous pendulum dynamics in CasADi."""
    theta = x[0]
    omega = x[1]
    tau = u[0] if u.numel() > 0 else u

    m_eff = mass * (lc**2)
    theta_ddot = tau / m_eff - g * ca.sin(theta) / lc - b * omega / m_eff
    return ca.vertcat(omega, theta_ddot)


def quadrotor_dynamics(
    x: ca.MX,
    u: ca.MX,
    mass: float = 0.5,
    J: Sequence[float] = (0.0023, 0.0023, 0.004),
    gravity: Sequence[float] = (0.0, 0.0, -9.81),
    motor_dist: float = 0.1750,
    kf: float = 1.0,
    km: float = 0.0245,
) -> ca.MX:
    """Evaluate continuous quadrotor dynamics in CasADi matching RobotZoo.jl bit-for-bit."""
    r = x[0:3]  # noqa: F841 -- unused position state
    q = x[3:7]  # [qx, qy, qz, qw] in JPL convention
    v = x[7:10]
    omega = x[10:13]

    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    # Passive rotation matrix R(q)
    R = ca.vertcat(
        ca.horzcat(qw * qw + qx * qx - qy * qy - qz * qz, 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)),
        ca.horzcat(2.0 * (qx * qy - qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2.0 * (qy * qz + qw * qx)),
        ca.horzcat(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), qw * qw - qx * qx - qy * qy + qz * qz),
    )

    # Kinematics matrix Xi(q)
    Xi = ca.vertcat(
        ca.horzcat(qw, -qz, qy),
        ca.horzcat(qz, qw, -qx),
        ca.horzcat(-qy, qx, qw),
        ca.horzcat(-qx, -qy, -qz),
    )

    r_dot = v
    q_dot = 0.5 * ca.mtimes(Xi, omega)

    # Forces
    f_thrust = kf * ca.sum1(u)
    f_body = ca.vertcat(0.0, 0.0, f_thrust)
    f_world = ca.mtimes(R.T, f_body)
    grav_vec = ca.vertcat(gravity[0], gravity[1], gravity[2])
    v_dot = grav_vec + f_world / mass

    # Moments
    tau_x = motor_dist * kf * (u[1] - u[3])
    tau_y = motor_dist * kf * (u[2] - u[0])
    tau_z = km * (u[0] - u[1] + u[2] - u[3])
    tau = ca.vertcat(tau_x, tau_y, tau_z)

    # Gyro
    J_diag = ca.vertcat(J[0], J[1], J[2])
    J_omega = J_diag * omega
    gyro = ca.vertcat(
        omega[1] * J_omega[2] - omega[2] * J_omega[1],
        omega[2] * J_omega[0] - omega[0] * J_omega[2],
        omega[0] * J_omega[1] - omega[1] * J_omega[0],
    )
    omega_dot = (tau - gyro) / J_diag

    return ca.vertcat(r_dot, q_dot, v_dot, omega_dot)


def rk4_step(
    dyn_fn: Callable[[ca.MX, ca.MX], ca.MX],
    x: ca.MX,
    u: ca.MX,
    dt: float | ca.MX,
) -> ca.MX:
    """Explicit 4th-order Runge-Kutta step in CasADi."""
    half_dt = 0.5 * dt
    k1 = dyn_fn(x, u)
    k2 = dyn_fn(x + half_dt * k1, u)
    k3 = dyn_fn(x + half_dt * k2, u)
    k4 = dyn_fn(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(
    dyn_fn: Callable[[ca.MX, ca.MX], ca.MX],
    x: ca.MX,
    u: ca.MX,
    dt: float | ca.MX,
) -> ca.MX:
    """Explicit Euler step in CasADi."""
    return x + dt * dyn_fn(x, u)


def _flatten_bounds(
    xL: np.ndarray,
    xU: np.ndarray,
    uL: np.ndarray,
    uU: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten stage-wise state and control bounds into NLP layout."""
    zL_stages = np.concatenate([xL[:-1], uL], axis=1).reshape(-1)
    zL = np.concatenate([zL_stages, xL[-1]])
    zU_stages = np.concatenate([xU[:-1], uU], axis=1).reshape(-1)
    zU = np.concatenate([zU_stages, xU[-1]])
    return zL, zU


class CasadiProblem:
    """Direct transcription optimal control problem formulated in CasADi Opti.

    Parameters
    ----------
    opti : ca.Opti
        Configured CasADi Opti stack.
    X : ca.MX
        Symbolic state variable matrix of shape (n, N).
    U : ca.MX
        Symbolic control variable matrix of shape (m, N-1).
    cost : ca.MX
        Symbolic objective expression scalar.
    N : int
        Horizon length in knot points.
    n : int
        State dimension.
    m : int
        Control dimension.
    dt : float | ArrayLike
        Step duration (scalar or 1D array of length N-1).
    x0 : ArrayLike
        Initial state condition of shape (n,).
    xf : ArrayLike | None, optional
        Goal state of shape (n,). Defaults to None.
    z_min : np.ndarray | None, optional
        Primal lower bounds of shape (N * n + (N - 1) * m,).
    z_max : np.ndarray | None, optional
        Primal upper bounds of shape (N * n + (N - 1) * m,).
    raw_constraints : list[ca.MX] | None, optional
        List of raw symbolic constraint residual expressions.
    """

    def __init__(
        self,
        opti: ca.Opti,
        X: ca.MX,
        U: ca.MX,
        cost: ca.MX,
        N: int,
        n: int,
        m: int,
        dt: float | ArrayLike,
        x0: ArrayLike,
        xf: ArrayLike | None = None,
        z_min: np.ndarray | None = None,
        z_max: np.ndarray | None = None,
        raw_constraints: list[ca.MX] | None = None,
    ) -> None:
        self.opti = opti
        self.X = X
        self.U = U
        self.cost = cost
        self.N = int(N)
        self.n = int(n)
        self.m = int(m)
        dt_np = np.asarray(dt, dtype=np.float64)
        self.dt = dt_np if dt_np.ndim > 0 else float(dt_np)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.xf = None if xf is None else np.asarray(xf, dtype=np.float64)
        self.z_min = z_min
        self.z_max = z_max
        self.raw_constraints = raw_constraints or []

    def solve(
        self,
        options: Mapping[str, Any] | None = None,
        initial_X: np.ndarray | None = None,
        initial_U: np.ndarray | None = None,
    ) -> CasadiResult:
        """Solve the transcribed NLP using CasADi with Ipopt.

        Parameters
        ----------
        options : Mapping[str, Any] | None, optional
            Solver options passed to Ipopt (e.g. {"max_iter": 500, "tol": 1e-8, "print_level": 0}).
        initial_X : np.ndarray | None, optional
            Initial state guess of shape (N, n) or (n, N).
        initial_U : np.ndarray | None, optional
            Initial control guess of shape (N-1, m) or (m, N-1).

        Returns
        -------
        CasadiResult
            Solved optimal trajectory, cost, dual multipliers, and status.
        """
        # Initial guess
        if initial_X is not None:
            X0 = np.asarray(initial_X, dtype=np.float64)
            if X0.shape == (self.N, self.n):
                X0 = X0.T
            self.opti.set_initial(self.X, X0)
        else:
            self.opti.set_initial(self.X, np.repeat(self.x0[:, None], self.N, axis=1))

        if initial_U is not None:
            U0 = np.asarray(initial_U, dtype=np.float64)
            if U0.shape == (self.N - 1, self.m):
                U0 = U0.T
            self.opti.set_initial(self.U, U0)
        else:
            self.opti.set_initial(self.U, np.zeros((self.m, self.N - 1)))

        ipopt_opts: dict[str, Any] = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
        }
        if options:
            for k, v in options.items():
                if k in {"print_level", "max_iter", "tol", "acceptable_tol", "constr_viol_tol"}:
                    ipopt_opts[f"ipopt.{k}"] = v
                else:
                    ipopt_opts[k] = v

        self.opti.solver("ipopt", ipopt_opts)

        sol = self.opti.solve()

        X_opt = np.asarray(sol.value(self.X), dtype=np.float64)
        if X_opt.ndim == 1:
            X_opt = X_opt.reshape((self.n, self.N))
        X_traj = X_opt.T  # (N, n)

        U_opt = np.asarray(sol.value(self.U), dtype=np.float64)
        if U_opt.ndim == 1:
            U_opt = U_opt.reshape((self.m, self.N - 1))
        U_traj = U_opt.T  # (N-1, m)

        cost_val = float(sol.value(self.cost))

        dt_arr = (
            np.full(self.N - 1, self.dt, dtype=np.float64)
            if isinstance(self.dt, float)
            else np.broadcast_to(self.dt, (self.N - 1,)).astype(np.float64)
        )
        t_opt = np.concatenate([[0.0], np.cumsum(dt_arr)])

        opt_traj = Trajectory(
            X=jnp.asarray(X_traj),
            U=jnp.asarray(U_traj),
            t=jnp.asarray(t_opt),
            dt=jnp.asarray(dt_arr),
        )

        mult_g = np.asarray(sol.value(self.opti.lam_g), dtype=np.float64).flatten()
        max_residual = self.evaluate_max_constraint_residual(X_traj, U_traj)

        stats = self.opti.stats()
        success = bool(stats.get("success", True))
        status = int(stats.get("return_status_code", 0))

        return CasadiResult(
            trajectory=opt_traj,
            cost=cost_val,
            success=success,
            status=status,
            mult_g=mult_g,
            max_constraint_residual=max_residual,
            opti_sol=sol,
            info=stats,
        )

    def evaluate_max_constraint_residual(self, X_val: np.ndarray, U_val: np.ndarray) -> float:
        """Evaluate maximum constraint residual on trajectory."""
        max_res = 0.0
        # 1. Initial condition
        max_res = max(max_res, float(np.max(np.abs(X_val[0] - self.x0))))

        # 2. Primal bounds
        if self.z_min is not None or self.z_max is not None:
            z_stages = np.concatenate([X_val[:-1], U_val], axis=1).reshape(-1)
            Z = np.concatenate([z_stages, X_val[-1]])
            if self.z_min is not None:
                viol_min = np.maximum(0.0, self.z_min - Z)
                max_res = max(max_res, float(np.max(viol_min)))
            if self.z_max is not None:
                viol_max = np.maximum(0.0, Z - self.z_max)
                max_res = max(max_res, float(np.max(viol_max)))

        # 3. Goal condition
        if self.xf is not None:
            max_res = max(max_res, float(np.max(np.abs(X_val[-1] - self.xf))))

        return float(max_res)


def build_cartpole_casadi(
    N: int = 25,
    dt: float = 0.05,
    x0: ArrayLike = (0.0, 0.01, 0.0, 0.0),
    xf: ArrayLike = (0.0, np.pi, 0.0, 0.0),
    Q: ArrayLike = (1.0, 10.0, 0.1, 0.1),
    R: ArrayLike = (0.01,),
    Qf: ArrayLike = (100.0, 1000.0, 10.0, 10.0),
    u_min: float | ArrayLike = -20.0,
    u_max: float | ArrayLike = 20.0,
    x_min: ArrayLike | None = None,
    x_max: ArrayLike | None = None,
    mc: float = 1.0,
    mp: float = 0.2,
    l: float = 0.5,  # noqa: E741 -- length parameter matching spec
    g: float = 9.81,
) -> CasadiProblem:
    """Build standalone pure-CasADi Cartpole swing-up problem."""
    n = 4
    m = 1
    x0_arr = np.asarray(x0, dtype=np.float64)
    xf_arr = np.asarray(xf, dtype=np.float64)
    Q_mat = np.diag(Q) if np.ndim(Q) == 1 else np.asarray(Q, dtype=np.float64)
    R_mat = np.diag(R) if np.ndim(R) == 1 else np.asarray(R, dtype=np.float64)
    Qf_mat = np.diag(Qf) if np.ndim(Qf) == 1 else np.asarray(Qf, dtype=np.float64)

    opti = ca.Opti()
    X = opti.variable(n, N)
    U = opti.variable(m, N - 1)

    # Cost
    cost = 0
    for k in range(N - 1):
        dx = X[:, k] - xf_arr
        du = U[:, k]
        cost += 0.5 * ca.mtimes([dx.T, Q_mat, dx]) + 0.5 * ca.mtimes([du.T, R_mat, du])
    dx_term = X[:, N - 1] - xf_arr
    cost += 0.5 * ca.mtimes([dx_term.T, Qf_mat, dx_term])
    opti.minimize(cost)

    def dyn_fn(x_var: ca.MX, u_var: ca.MX) -> ca.MX:
        return cartpole_dynamics(x_var, u_var, mc=mc, mp=mp, l=l, g=g)

    # 1. Initial condition: X0 - x0 == 0
    opti.subject_to(X[:, 0] == x0_arr)

    # 2. Dynamics defects
    for k in range(N - 1):
        opti.subject_to(X[:, k + 1] == rk4_step(dyn_fn, X[:, k], U[:, k], dt))

    # 3. Terminal goal constraint
    opti.subject_to(X[:, N - 1] == xf_arr)

    # 4. Control and state bounds
    u_min_val = float(np.asarray(u_min).reshape(-1)[0])
    u_max_val = float(np.asarray(u_max).reshape(-1)[0])
    opti.subject_to(opti.bounded(u_min_val, U, u_max_val))

    if x_min is not None and x_max is not None:
        x_min_arr = np.asarray(x_min, dtype=np.float64)
        x_max_arr = np.asarray(x_max, dtype=np.float64)
        for i in range(n):
            if np.isfinite(x_min_arr[i]) or np.isfinite(x_max_arr[i]):
                opti.subject_to(opti.bounded(x_min_arr[i], X[i, 1:-1], x_max_arr[i]))
    else:
        x_min_arr = np.full(n, -np.inf)
        x_max_arr = np.full(n, np.inf)

    xL = np.repeat(x_min_arr[None, :], N, axis=0)
    xU = np.repeat(x_max_arr[None, :], N, axis=0)
    uL = np.full((N - 1, m), u_min_val)
    uU = np.full((N - 1, m), u_max_val)
    zL, zU = _flatten_bounds(xL, xU, uL, uU)

    return CasadiProblem(
        opti=opti,
        X=X,
        U=U,
        cost=cost,
        N=N,
        n=n,
        m=m,
        dt=dt,
        x0=x0_arr,
        xf=xf_arr,
        z_min=zL,
        z_max=zU,
    )


def build_dubins_casadi(
    N: int = 25,
    dt: float = 0.1,
    x0: ArrayLike = (0.0, 0.0, 0.0),
    xf: ArrayLike = (2.0, 1.0, 0.0),
    Q: ArrayLike = (1.0, 1.0, 0.1),
    R: ArrayLike = (0.1, 0.1),
    Qf: ArrayLike = (100.0, 100.0, 10.0),
    u_min: ArrayLike = (0.0, -1.5),
    u_max: ArrayLike = (2.0, 1.5),
    x_min: ArrayLike | None = None,
    x_max: ArrayLike | None = None,
    obstacles: Sequence[tuple[float, float, float]] | None = None,  # (xc, yc, radius)
) -> CasadiProblem:
    """Build standalone pure-CasADi Dubins car navigation problem."""
    n = 3
    m = 2
    x0_arr = np.asarray(x0, dtype=np.float64)
    xf_arr = np.asarray(xf, dtype=np.float64)
    Q_mat = np.diag(Q) if np.ndim(Q) == 1 else np.asarray(Q, dtype=np.float64)
    R_mat = np.diag(R) if np.ndim(R) == 1 else np.asarray(R, dtype=np.float64)
    Qf_mat = np.diag(Qf) if np.ndim(Qf) == 1 else np.asarray(Qf, dtype=np.float64)
    u_min_arr = np.asarray(u_min, dtype=np.float64)
    u_max_arr = np.asarray(u_max, dtype=np.float64)

    opti = ca.Opti()
    X = opti.variable(n, N)
    U = opti.variable(m, N - 1)

    cost = 0
    for k in range(N - 1):
        dx = X[:, k] - xf_arr
        du = U[:, k]
        cost += 0.5 * ca.mtimes([dx.T, Q_mat, dx]) + 0.5 * ca.mtimes([du.T, R_mat, du])
    dx_term = X[:, N - 1] - xf_arr
    cost += 0.5 * ca.mtimes([dx_term.T, Qf_mat, dx_term])
    opti.minimize(cost)

    # 1. Initial condition
    opti.subject_to(X[:, 0] == x0_arr)

    # 2. Dynamics + Obstacle keep-out
    for k in range(N - 1):
        opti.subject_to(X[:, k + 1] == rk4_step(dubins_dynamics, X[:, k], U[:, k], dt))
        if obstacles:
            for xc, yc, r in obstacles:
                opti.subject_to(-((X[0, k] - xc) ** 2) - (X[1, k] - yc) ** 2 + r**2 <= 0)

    # 3. Terminal goal constraint
    opti.subject_to(X[:, N - 1] == xf_arr)

    # 4. Bounds
    for i in range(m):
        opti.subject_to(opti.bounded(u_min_arr[i], U[i, :], u_max_arr[i]))

    if x_min is not None and x_max is not None:
        x_min_arr = np.asarray(x_min, dtype=np.float64)
        x_max_arr = np.asarray(x_max, dtype=np.float64)
        for i in range(n):
            if np.isfinite(x_min_arr[i]) or np.isfinite(x_max_arr[i]):
                opti.subject_to(opti.bounded(x_min_arr[i], X[i, 1:-1], x_max_arr[i]))
    else:
        x_min_arr = np.full(n, -np.inf)
        x_max_arr = np.full(n, np.inf)

    xL = np.repeat(x_min_arr[None, :], N, axis=0)
    xU = np.repeat(x_max_arr[None, :], N, axis=0)
    uL = np.repeat(u_min_arr[None, :], N - 1, axis=0)
    uU = np.repeat(u_max_arr[None, :], N - 1, axis=0)
    zL, zU = _flatten_bounds(xL, xU, uL, uU)

    return CasadiProblem(
        opti=opti,
        X=X,
        U=U,
        cost=cost,
        N=N,
        n=n,
        m=m,
        dt=dt,
        x0=x0_arr,
        xf=xf_arr,
        z_min=zL,
        z_max=zU,
    )


def build_quadrotor_casadi(
    N: int = 25,
    dt: float = 0.05,
    x0: ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    xf: ArrayLike = (3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    Q: ArrayLike = (1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    R: ArrayLike = (0.01, 0.01, 0.01, 0.01),
    Qf: ArrayLike = (100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    u_min: float | ArrayLike = 0.0,
    u_max: float | ArrayLike = 10.0,
    x_min: ArrayLike | None = None,
    x_max: ArrayLike | None = None,
    obstacles: Sequence[tuple[float, float, float, float]] | None = None,  # (xc, yc, zc, radius)
    mass: float = 0.5,
    J: Sequence[float] = (0.0023, 0.0023, 0.004),
    gravity: Sequence[float] = (0.0, 0.0, -9.81),
    motor_dist: float = 0.1750,
    kf: float = 1.0,
    km: float = 0.0245,
) -> CasadiProblem:
    """Build standalone pure-CasADi Quadrotor obstacle avoidance problem."""
    n = 13
    m = 4
    x0_arr = np.asarray(x0, dtype=np.float64)
    xf_arr = np.asarray(xf, dtype=np.float64)
    Q_mat = np.diag(Q) if np.ndim(Q) == 1 else np.asarray(Q, dtype=np.float64)
    R_mat = np.diag(R) if np.ndim(R) == 1 else np.asarray(R, dtype=np.float64)
    Qf_mat = np.diag(Qf) if np.ndim(Qf) == 1 else np.asarray(Qf, dtype=np.float64)
    u_min_arr = np.broadcast_to(np.asarray(u_min, dtype=np.float64), (m,))
    u_max_arr = np.broadcast_to(np.asarray(u_max, dtype=np.float64), (m,))

    opti = ca.Opti()
    X = opti.variable(n, N)
    U = opti.variable(m, N - 1)

    cost = 0
    for k in range(N - 1):
        dx = X[:, k] - xf_arr
        du = U[:, k]
        cost += 0.5 * ca.mtimes([dx.T, Q_mat, dx]) + 0.5 * ca.mtimes([du.T, R_mat, du])
    dx_term = X[:, N - 1] - xf_arr
    cost += 0.5 * ca.mtimes([dx_term.T, Qf_mat, dx_term])
    opti.minimize(cost)

    def dyn_fn(x_var: ca.MX, u_var: ca.MX) -> ca.MX:
        return quadrotor_dynamics(
            x_var,
            u_var,
            mass=mass,
            J=J,
            gravity=gravity,
            motor_dist=motor_dist,
            kf=kf,
            km=km,
        )

    # 1. Initial condition
    opti.subject_to(X[:, 0] == x0_arr)

    # 2. Dynamics + Obstacle keep-out
    for k in range(N - 1):
        opti.subject_to(X[:, k + 1] == rk4_step(dyn_fn, X[:, k], U[:, k], dt))
        if obstacles:
            for xc, yc, zc, r in obstacles:
                opti.subject_to(-((X[0, k] - xc) ** 2) - (X[1, k] - yc) ** 2 - (X[2, k] - zc) ** 2 + r**2 <= 0)

    # 3. Terminal goal constraint
    opti.subject_to(X[:, N - 1] == xf_arr)

    # 4. Bounds
    for i in range(m):
        opti.subject_to(opti.bounded(u_min_arr[i], U[i, :], u_max_arr[i]))

    if x_min is not None and x_max is not None:
        x_min_arr = np.asarray(x_min, dtype=np.float64)
        x_max_arr = np.asarray(x_max, dtype=np.float64)
        for i in range(n):
            if np.isfinite(x_min_arr[i]) or np.isfinite(x_max_arr[i]):
                opti.subject_to(opti.bounded(x_min_arr[i], X[i, 1:-1], x_max_arr[i]))
    else:
        x_min_arr = np.full(n, -np.inf)
        x_max_arr = np.full(n, np.inf)

    xL = np.repeat(x_min_arr[None, :], N, axis=0)
    xU = np.repeat(x_max_arr[None, :], N, axis=0)
    uL = np.repeat(u_min_arr[None, :], N - 1, axis=0)
    uU = np.repeat(u_max_arr[None, :], N - 1, axis=0)
    zL, zU = _flatten_bounds(xL, xU, uL, uU)

    return CasadiProblem(
        opti=opti,
        X=X,
        U=U,
        cost=cost,
        N=N,
        n=n,
        m=m,
        dt=dt,
        x0=x0_arr,
        xf=xf_arr,
        z_min=zL,
        z_max=zU,
    )


def _get_casadi_dyn_fn(model: Any) -> Callable[[ca.MX, ca.MX], ca.MX]:
    """Extract continuous dynamics callable for a model, reaching through a discretized wrapper."""
    if isinstance(model, DiscretizedDynamics):
        model = model.continuous_dynamics

    if isinstance(model, Cartpole):
        mc = float(np.asarray(model.mc))
        mp = float(np.asarray(model.mp))
        l_val = float(np.asarray(model.l))
        g_val = float(np.asarray(model.g))

        def cp_dyn(x_var: ca.MX, u_var: ca.MX) -> ca.MX:
            return cartpole_dynamics(x_var, u_var, mc=mc, mp=mp, l=l_val, g=g_val)

        return cp_dyn

    if isinstance(model, DubinsCar):
        return dubins_dynamics

    if isinstance(model, Pendulum):
        mass = float(np.asarray(model.mass))
        lc = float(np.asarray(model.lc))
        b = float(np.asarray(model.b))
        g = float(np.asarray(model.g))

        def pend_dyn(x_var: ca.MX, u_var: ca.MX) -> ca.MX:
            return pendulum_dynamics(x_var, u_var, mass=mass, lc=lc, b=b, g=g)

        return pend_dyn

    if isinstance(model, Quadrotor):
        mass = float(np.asarray(model.mass))
        J = tuple(float(v) for v in np.asarray(model.J))
        gravity = tuple(float(v) for v in np.asarray(model.gravity))
        motor_dist = float(np.asarray(model.motor_dist))
        kf = float(np.asarray(model.kf))
        km = float(np.asarray(model.km))

        def quad_dyn(x_var: ca.MX, u_var: ca.MX) -> ca.MX:
            return quadrotor_dynamics(
                x_var,
                u_var,
                mass=mass,
                J=J,
                gravity=gravity,
                motor_dist=motor_dist,
                kf=kf,
                km=km,
            )

        return quad_dyn

    msg = f"Model {type(model).__name__} is not yet supported in CasADi translation."
    raise NotImplementedError(msg)


def _build_stage_cost_term(st_cost: Any, obj: Objective, xk: ca.MX, uk: ca.MX, k: int, N: int) -> ca.MX | float | int:
    """Build single stage cost expression."""
    if isinstance(st_cost, (DiagonalCost, QuadraticCost)):
        Q_arr = np.asarray(obj.Q)
        R_arr = np.asarray(obj.R)
        q_arr = np.asarray(obj.q)
        r_arr = np.asarray(obj.r)
        c_arr = np.asarray(obj.c)

        Q_k = Q_arr[k] if Q_arr.shape[0] == N - 1 else Q_arr
        R_k = R_arr[k] if R_arr.shape[0] == N - 1 else R_arr
        q_k = q_arr[k] if q_arr.shape[0] == N - 1 else q_arr
        r_k = r_arr[k] if r_arr.shape[0] == N - 1 else r_arr
        c_k = float(c_arr[k]) if c_arr.shape[0] == N - 1 else float(c_arr)

        Q_mat = np.diag(Q_k) if Q_k.ndim == 1 else Q_k
        R_mat = np.diag(R_k) if R_k.ndim == 1 else R_k

        return (
            0.5 * ca.mtimes([xk.T, Q_mat, xk])
            + 0.5 * ca.mtimes([uk.T, R_mat, uk])
            + ca.dot(q_k, xk)
            + ca.dot(r_k, uk)
            + c_k
        )
    if isinstance(st_cost, QuatGeodesicCost):
        Q_k = np.asarray(st_cost.Q)
        R_k = np.asarray(st_cost.R)
        q_lin_k = np.asarray(st_cost.q_lin)
        r_lin_k = np.asarray(st_cost.r_lin)
        c_k = float(np.asarray(st_cost.c))
        w_k = float(np.asarray(st_cost.w))
        q_ref_k = np.asarray(st_cost.q_ref)
        qind = list(st_cost.qind)

        Q_mat = np.diag(Q_k) if Q_k.ndim == 1 else Q_k
        R_mat = np.diag(R_k) if R_k.ndim == 1 else R_k

        dq = ca.dot(q_ref_k, xk[qind])
        geodesic_val = w_k * ca.fmin(1.0 + dq, 1.0 - dq)

        return (
            0.5 * ca.mtimes([xk.T, Q_mat, xk])
            + 0.5 * ca.mtimes([uk.T, R_mat, uk])
            + ca.dot(q_lin_k, xk)
            + ca.dot(r_lin_k, uk)
            + c_k
            + geodesic_val
        )
    return 0


def _build_terminal_cost_term(term_cost: Any, obj: Objective, x_term: ca.MX) -> ca.MX | float | int:
    """Build terminal cost expression."""
    if isinstance(term_cost, (DiagonalCost, QuadraticCost)):
        Q_f = np.asarray(obj.Q_f)
        q_f = np.asarray(obj.q_f)
        c_f = float(np.asarray(obj.c_f))
        Qf_mat = np.diag(Q_f) if Q_f.ndim == 1 else Q_f
        return 0.5 * ca.mtimes([x_term.T, Qf_mat, x_term]) + ca.dot(q_f, x_term) + c_f
    if isinstance(term_cost, QuatGeodesicCost):
        Q_f = np.asarray(term_cost.Q)
        q_lin_f = np.asarray(term_cost.q_lin)
        c_f = float(np.asarray(term_cost.c))
        w_f = float(np.asarray(term_cost.w))
        q_ref_f = np.asarray(term_cost.q_ref)
        qind = list(term_cost.qind)

        Qf_mat = np.diag(Q_f) if Q_f.ndim == 1 else Q_f
        dq = ca.dot(q_ref_f, x_term[qind])
        geodesic_val = w_f * ca.fmin(1.0 + dq, 1.0 - dq)

        return 0.5 * ca.mtimes([x_term.T, Qf_mat, x_term]) + ca.dot(q_lin_f, x_term) + c_f + geodesic_val
    return 0


def _build_objective_expression(obj: Any, X: ca.MX, U: ca.MX, N: int) -> Any:
    """Build CasADi objective expression scalar from Problem Objective."""
    cost = 0
    if isinstance(obj, Objective):
        st_cost = obj.stage_cost
        term_cost = obj.terminal_cost
        for k in range(N - 1):
            cost += _build_stage_cost_term(st_cost, obj, X[:, k], U[:, k], k, N)
        cost += _build_terminal_cost_term(term_cost, obj, X[:, N - 1])
    return cost


def _add_stage_constraints(
    opti: ca.Opti,
    knot_evaluators: Sequence[Any],
    X: ca.MX,
    U: ca.MX,
    step_fn: Callable,
    dyn_fn: Callable[[ca.MX, ca.MX], ca.MX],
    dt_arr: np.ndarray,
    N: int,
) -> None:
    """Add dynamics defects and stage path constraints to Opti."""
    for k in range(N - 1):
        opti.subject_to(X[:, k + 1] == step_fn(dyn_fn, X[:, k], U[:, k], dt_arr[k]))

        if k < len(knot_evaluators):
            for con in knot_evaluators[k].constraints:
                if isinstance(con, CircleConstraint):
                    for xc, yc, rad in zip(con.xc, con.yc, con.radius, strict=False):
                        opti.subject_to(
                            -((X[con.xi, k] - float(xc)) ** 2) - (X[con.yi, k] - float(yc)) ** 2 + float(rad) ** 2 <= 0
                        )
                elif isinstance(con, SphereConstraint):
                    for xc, yc, zc, rad in zip(con.xc, con.yc, con.zc, con.radius, strict=False):
                        opti.subject_to(
                            -((X[con.xi, k] - float(xc)) ** 2)
                            - ((X[con.yi, k] - float(yc)) ** 2)
                            - ((X[con.zi, k] - float(zc)) ** 2)
                            + float(rad) ** 2
                            <= 0
                        )
                elif isinstance(con, LinearConstraint):
                    z_k = ca.vertcat(X[:, k], U[:, k])
                    z_sub = z_k[list(con.inds)]
                    opti.subject_to(np.asarray(con.A) @ z_sub - np.asarray(con.b) <= 0)


def _add_terminal_constraints(
    opti: ca.Opti,
    knot_evaluators: Sequence[Any],
    X: ca.MX,
    N: int,
) -> None:
    """Add terminal constraints to Opti."""
    if len(knot_evaluators) <= N - 1:
        return

    for con in knot_evaluators[N - 1].constraints:
        if isinstance(con, GoalConstraint):
            opti.subject_to(X[list(con.inds), N - 1] == np.asarray(con.xf))
        elif isinstance(con, CircleConstraint):
            for xc, yc, rad in zip(con.xc, con.yc, con.radius, strict=False):
                opti.subject_to(
                    -((X[con.xi, N - 1] - float(xc)) ** 2) - (X[con.yi, N - 1] - float(yc)) ** 2 + float(rad) ** 2 <= 0
                )
        elif isinstance(con, SphereConstraint):
            for xc, yc, zc, rad in zip(con.xc, con.yc, con.zc, con.radius, strict=False):
                opti.subject_to(
                    -((X[con.xi, N - 1] - float(xc)) ** 2)
                    - ((X[con.yi, N - 1] - float(yc)) ** 2)
                    - ((X[con.zi, N - 1] - float(zc)) ** 2)
                    + float(rad) ** 2
                    <= 0
                )
        elif isinstance(con, QuatVecEq):
            q = X[list(con.qind), N - 1]
            q_norm = ca.norm_2(q)
            q_unit = q / ca.if_else(q_norm > 1e-12, q_norm, 1.0)
            dq = ca.dot(np.asarray(con.qf), q_unit)
            qf_sign = ca.if_else(dq < 0.0, -np.asarray(con.qf), np.asarray(con.qf))
            opti.subject_to(q_unit[:3] == qf_sign[:3])


def _add_primal_bounds(
    opti: ca.Opti,
    problem: Problem,
    X: ca.MX,
    U: ca.MX,
    n: int,
    m: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Add variable bounds to Opti and return flattened (zL, zU) arrays."""
    if not hasattr(problem.constraints, "primal_bounds"):
        return None, None

    xL, xU, uL, uU = problem.constraints.primal_bounds()
    for i in range(m):
        u_min_i = np.min(uL[:, i])
        u_max_i = np.max(uU[:, i])
        if np.isfinite(u_min_i) or np.isfinite(u_max_i):
            opti.subject_to(opti.bounded(u_min_i, U[i, :], u_max_i))

    for i in range(n):
        x_min_i = np.min(xL[:, i])
        x_max_i = np.max(xU[:, i])
        if np.isfinite(x_min_i) or np.isfinite(x_max_i):
            opti.subject_to(opti.bounded(x_min_i, X[i, 1:-1], x_max_i))

    zL, zU = _flatten_bounds(xL, xU, uL, uU)
    return zL, zU


def build_casadi_from_problem(
    problem: Problem,
    x0: ArrayLike,
    dt: float | ArrayLike = 0.05,
    *,
    integrator: str = "rk4",
) -> CasadiProblem:
    """Build a matching CasADi Opti problem automatically from a trajopt.Problem instance.

    Parameters
    ----------
    problem : Problem
        trajopt Problem containing model, objective, constraints, and horizon.
    x0 : ArrayLike
        Initial state vector of shape (n,).
    dt : float | ArrayLike, optional
        Time step duration. Defaults to 0.05.
    integrator : str, optional
        Integration scheme ("rk4" or "euler"). Defaults to "rk4".

    Returns
    -------
    CasadiProblem
        Configured CasADi direct transcription problem.
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    x0_arr = np.asarray(x0, dtype=np.float64)

    opti = ca.Opti()
    X = opti.variable(n, N)
    U = opti.variable(m, N - 1)

    dyn_fn = _get_casadi_dyn_fn(problem.model)
    step_fn = rk4_step if integrator.lower() == "rk4" else euler_step

    cost = _build_objective_expression(problem.obj, X, U, N)
    opti.minimize(cost)

    opti.subject_to(X[:, 0] == x0_arr)

    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (N - 1,))
    knot_evaluators = problem.constraints.knot_evaluators if problem.constraints is not None else ()
    _add_stage_constraints(
        opti=opti,
        knot_evaluators=knot_evaluators,
        X=X,
        U=U,
        step_fn=step_fn,
        dyn_fn=dyn_fn,
        dt_arr=dt_arr,
        N=N,
    )
    _add_terminal_constraints(
        opti=opti,
        knot_evaluators=knot_evaluators,
        X=X,
        N=N,
    )
    zL, zU = _add_primal_bounds(
        opti=opti,
        problem=problem,
        X=X,
        U=U,
        n=n,
        m=m,
    )

    return CasadiProblem(
        opti=opti,
        X=X,
        U=U,
        cost=cost,
        N=N,
        n=n,
        m=m,
        dt=dt_arr,
        x0=x0_arr,
        z_min=zL,
        z_max=zU,
    )


def assert_setups_match(
    problem: Problem,
    casadi_problem: CasadiProblem,
    x0: ArrayLike,
    dt: float | ArrayLike,
) -> None:
    """Assert that the trajopt Problem and CasadiProblem agree identically on their setup parameters."""
    assert int(problem.N) == casadi_problem.N, f"Horizon mismatch: {problem.N} != {casadi_problem.N}"
    assert int(problem.model.n) == casadi_problem.n, f"State dim mismatch: {problem.model.n} != {casadi_problem.n}"
    assert int(problem.model.m) == casadi_problem.m, f"Control dim mismatch: {problem.model.m} != {casadi_problem.m}"

    np.testing.assert_allclose(
        np.asarray(x0, dtype=float),
        casadi_problem.x0,
        atol=1e-12,
        err_msg="Initial state x0 mismatch between formulations",
    )

    dt_py = np.broadcast_to(np.asarray(dt, dtype=float), (casadi_problem.N - 1,))
    dt_cas = np.broadcast_to(np.asarray(casadi_problem.dt, dtype=float), (casadi_problem.N - 1,))
    np.testing.assert_allclose(dt_py, dt_cas, atol=1e-12, err_msg="Step duration dt mismatch between formulations")

    if (
        hasattr(problem.constraints, "primal_bounds")
        and casadi_problem.z_min is not None
        and casadi_problem.z_max is not None
    ):
        xL, xU, uL, uU = problem.constraints.primal_bounds()
        zL, zU = _flatten_bounds(xL, xU, uL, uU)

        np.testing.assert_allclose(
            zL,
            casadi_problem.z_min,
            atol=1e-12,
            err_msg="Primal lower bounds z_min mismatch between formulations",
        )
        np.testing.assert_allclose(
            zU,
            casadi_problem.z_max,
            atol=1e-12,
            err_msg="Primal upper bounds z_max mismatch between formulations",
        )


def canonical_dual_blocks(problem: Problem) -> list[tuple[str, int, int]]:
    """Return the transcribed problem's dual row blocks as (name, start, width), in eval_g order.

    Both formulations emit the initial condition first and then, per knot, the dynamics defect
    followed by that knot's path constraints, so this decomposition indexes either side's dual
    vector. It stops at the last constraint row: variable bounds live outside it.
    """
    n = int(problem.model.n)
    N = int(problem.N)
    knot_p = list(problem.constraints.p)

    blocks: list[tuple[str, int, int]] = [("initial condition", 0, n)]
    offset = n
    for k in range(N - 1):
        blocks.append((f"dynamics defect {k}", offset, n))
        offset += n
        if knot_p[k]:
            blocks.append((f"path constraints {k}", offset, knot_p[k]))
            offset += knot_p[k]
    if knot_p[N - 1]:
        blocks.append((f"terminal constraints {N - 1}", offset, knot_p[N - 1]))

    return blocks


def assert_dual_block_parity(
    problem: Problem,
    trajopt_res: SolverResult,
    casadi_res: CasadiResult,
    *,
    tol_dual: float = 1e-4,
) -> None:
    """Assert the dual blocks whose rows exist in both formulations agree, block by block.

    Only the transcribed constraint rows are compared. The box bounds are deliberately excluded:
    trajopt hands them to Ipopt as variable limits, where their multipliers come back through
    `mu` rather than `mult_g`, while `_add_primal_bounds` gives CasADi general constraint rows
    for them -- collapsed to a knot-invariant min/max envelope, ordered by variable rather than
    by knot, and skipping the first and last state knots. Those rows are a different set, not a
    permutation of the same one, so no elementwise comparison of them would mean anything.

    Parameters
    ----------
    problem : Problem
        The trajopt problem both formulations transcribe, read for the row layout.
    trajopt_res : SolverResult
        Result whose `lam` holds the canonical constraint duals.
    casadi_res : CasadiResult
        Result whose `mult_g` holds the same rows first, then CasADi's bound rows.
    tol_dual : float, optional
        Maximum tolerated relative dual error per block. Defaults to 1e-4.
    """
    lam_py = np.asarray(trajopt_res.lam, dtype=float).flatten()
    lam_cas = np.asarray(casadi_res.mult_g, dtype=float).flatten()

    blocks = canonical_dual_blocks(problem)
    n_compared = blocks[-1][1] + blocks[-1][2]

    assert len(lam_py) == n_compared, (
        f"trajopt returned {len(lam_py)} constraint duals against {n_compared} canonical rows"
    )
    assert len(lam_cas) >= n_compared, (
        f"CasADi returned {len(lam_cas)} duals, fewer than the {n_compared} shared constraint rows"
    )

    for name, start, width in blocks:
        a = lam_py[start : start + width]
        b = lam_cas[start : start + width]
        # Scaled by |b| but floored at 1, so a near-zero multiplier stays an absolute test.
        rel_err = float(np.max(np.abs(a - b) / (np.abs(b) + 1.0)))
        assert rel_err <= tol_dual, (
            f"Dual block '{name}' (rows {start}:{start + width}) disagrees: "
            f"max rel err {rel_err:.3e} exceeds {tol_dual:.3e}\n  trajopt: {a}\n  casadi:  {b}"
        )


def assert_parity(
    trajopt_res: SolverResult,
    casadi_res: CasadiResult,
    *,
    tol_state: float = 1e-5,
    tol_control: float = 1e-5,
    tol_cost: float = 1e-5,
    tol_feas: float = 1e-4,
    check_duals: bool = True,
    tol_dual: float = 1e-4,
) -> None:
    """Assert full numerical parity between trajopt solve and CasADi baseline solve."""
    assert trajopt_res.success, f"trajopt Ipopt solve did not succeed: {trajopt_res.message}"
    assert casadi_res.success, f"CasADi Ipopt solve did not succeed: {casadi_res.status}"

    # 1. Primal state parity: ||X*_trajopt - X*_casadi||_inf <= tol_state
    X_py = np.asarray(trajopt_res.trajectory.X, dtype=float)
    X_cas = np.asarray(casadi_res.trajectory.X, dtype=float)
    max_state_err = float(np.max(np.abs(X_py - X_cas)))
    assert max_state_err <= tol_state, (
        f"Maximum absolute state error {max_state_err:.3e} exceeds tolerance {tol_state:.3e}"
    )

    # 2. Primal control parity: ||U*_trajopt - U*_casadi||_inf <= tol_control
    U_py = np.asarray(trajopt_res.trajectory.U, dtype=float)
    U_cas = np.asarray(casadi_res.trajectory.U, dtype=float)
    max_control_err = float(np.max(np.abs(U_py - U_cas)))
    assert max_control_err <= tol_control, (
        f"Maximum absolute control error {max_control_err:.3e} exceeds tolerance {tol_control:.3e}"
    )

    # 3. Objective parity, relative to |J*_casadi| but floored at 1 so that a near-zero
    # optimal cost (the tracking problems reach ~1e-15) does not turn this into a ratio test.
    abs_cost_err = float(abs(trajopt_res.cost - casadi_res.cost))
    scale = max(abs(casadi_res.cost), 1.0)
    rel_cost_err = abs_cost_err / scale
    assert rel_cost_err <= tol_cost, (
        f"Relative objective error {rel_cost_err:.3e} exceeds tolerance {tol_cost:.3e} "
        f"(trajopt: {trajopt_res.cost:.6f}, casadi: {casadi_res.cost:.6f})"
    )

    # 4. Feasibility tolerance on constraint residual.
    # NOTE: evaluate_max_constraint_residual covers only x0, box bounds and the goal, so the
    # trajopt-side violation below is the one that actually sees the dynamics defects.
    assert casadi_res.max_constraint_residual <= tol_feas, (
        f"CasADi constraint residual {casadi_res.max_constraint_residual:.3e} exceeds feasibility tol {tol_feas:.3e}"
    )
    assert trajopt_res.constraint_violation <= tol_feas, (
        f"trajopt constraint violation {trajopt_res.constraint_violation:.3e} exceeds feasibility tol {tol_feas:.3e}"
    )

    # 5. Dual multiplier agreement
    if check_duals:
        assert "mult_g" in trajopt_res.info, "mult_g not found in trajopt_res.info"
        assert len(casadi_res.mult_g) > 0, "casadi_res.mult_g is empty"
        mult_g_py = np.asarray(trajopt_res.info["mult_g"], dtype=float).flatten()
        mult_g_cas = np.asarray(casadi_res.mult_g, dtype=float).flatten()
        assert len(mult_g_py) == len(mult_g_cas), (
            f"Dual multiplier vector length mismatch: trajopt has {len(mult_g_py)}, CasADi has {len(mult_g_cas)}"
        )
        dual_diff = np.abs(mult_g_py - mult_g_cas)
        max_dual_err = float(np.max(dual_diff))
        # Scaled by |mult_g_cas| but floored at 1, so a near-zero multiplier stays an absolute test.
        rel_dual_err = float(np.max(dual_diff / (np.abs(mult_g_cas) + 1.0)))
        assert rel_dual_err <= tol_dual, (
            f"Dual multipliers disagree: max abs err {max_dual_err:.3e}, max rel err {rel_dual_err:.3e}"
        )
