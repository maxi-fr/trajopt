from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.constraint_list import BuiltConstraintList, ConstraintList
from trajopt.costs.objective import Objective
from trajopt.dynamics.base import (
    AbstractModel,
    ContinuousDynamics,
    DiscreteDynamics,
    DiscretizedDynamics,
    IntegratorCallable,
)
from trajopt.dynamics.integrators import RK4, Integrator
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import trajectory_to_z, z_to_trajectory


def _extract_discrete_model(problem: "Problem | AbstractModel") -> DiscreteDynamics:
    """Extract or construct a DiscreteDynamics model from Problem or AbstractModel."""
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


class Problem(eqx.Module):
    """Problem structure holding model, objective, constraints, horizon, and integrator.

    Parameters
    ----------
    model : AbstractModel
        Continuous or discrete dynamical model.
    obj : Objective
        Cost objective with stacked parameters.
    constraints : BuiltConstraintList | ConstraintList | None, optional
        Registered or fused constraint list. Defaults to empty ConstraintList.
    N : int | None, optional
        Horizon length. Defaults to obj.N.
    integrator : Integrator | IntegratorCallable | None, optional
        Integrator instance for continuous models. Defaults to None.
    """

    model: AbstractModel
    obj: Objective
    constraints: BuiltConstraintList
    N: int = eqx.field(static=True)
    integrator: Integrator | IntegratorCallable | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        model: AbstractModel,
        obj: Objective,
        constraints: BuiltConstraintList | ConstraintList | None = None,
        N: int | None = None,
        integrator: Integrator | IntegratorCallable | None = None,
    ) -> None:
        n = int(model.n)
        m = int(model.m)
        N_val = int(N if N is not None else obj.N)

        if constraints is None:
            cl = ConstraintList(n=n, m=m, N=N_val)
            built_con = cl.build()
        elif isinstance(constraints, ConstraintList):
            built_con = constraints.build()
        elif isinstance(constraints, BuiltConstraintList):
            built_con = constraints
        else:
            msg = f"Unsupported constraints type: {type(constraints).__name__}"
            raise TypeError(msg)

        self.model = model
        self.obj = obj
        self.constraints = built_con
        self.N = N_val
        self.integrator = integrator


class MPCState(eqx.Module):
    """Per-step MPC state holding initial condition, goal, trajectory, multipliers, and metadata.

    Parameters
    ----------
    x0 : jax.Array
        Initial state condition of shape (n,).
    t0 : jax.Array
        Initial timestamp scalar of shape ().
    xf : jax.Array
        Goal state vector of shape (n,).
    lam : jax.Array
        Constraint dual multipliers vector of shape (P,).
    mu : jax.Array
        Primal variable bounds multipliers vector of shape (N * n + (N - 1) * m,).
    Z : jax.Array
        Warm-start flat primal trajectory vector of shape (N * n + (N - 1) * m,).
    dt : jax.Array
        Step duration array of shape (N - 1,).
    n : int
        State dimension.
    m : int
        Control dimension.
    N : int
        Horizon length in knot points.
    """

    x0: jax.Array
    t0: jax.Array
    xf: jax.Array
    lam: jax.Array
    mu: jax.Array
    Z: jax.Array
    dt: jax.Array
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)

    @classmethod
    def initial(  # noqa: PLR0913 -- Initial state factory takes 7 arguments
        cls,
        problem: Problem,
        x0: jax.Array | Sequence[float],
        *,
        t0: float | jax.Array = 0.0,
        xf: jax.Array | Sequence[float] | None = None,
        dt: float | jax.Array = 0.05,
        initial_trajectory: Trajectory | None = None,
        initial_z: jax.Array | None = None,
    ) -> "MPCState":
        """Create an initial MPCState from Problem structure and boundary conditions.

        Parameters
        ----------
        problem : Problem
            Problem instance defining model dimensions and constraint structures.
        x0 : jax.Array | Sequence[float]
            Initial measured state vector of shape (n,).
        t0 : float | jax.Array, optional
            Initial timestamp. Defaults to 0.0.
        xf : jax.Array | Sequence[float] | None, optional
            Target goal state vector of shape (n,). Defaults to zeros.
        dt : float | jax.Array, optional
            Step duration (scalar or array of shape (N - 1,)). Defaults to 0.05.
        initial_trajectory : Trajectory | None, optional
            Initial trajectory guess. Defaults to repeating x0 with zero controls.
        initial_z : jax.Array | None, optional
            Flat initial guess vector of shape (N * n + (N - 1) * m,).

        Returns
        -------
        MPCState
            Initial per-step state instance.
        """
        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)

        x0_arr = jnp.asarray(x0, dtype=jnp.float64)
        t0_arr = jnp.asarray(t0, dtype=jnp.float64)
        dt_arr = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N - 1,))
        xf_arr = jnp.asarray(xf, dtype=jnp.float64) if xf is not None else jnp.zeros(n, dtype=jnp.float64)

        if initial_z is not None:
            z_init = jnp.asarray(initial_z, dtype=jnp.float64)
        elif initial_trajectory is not None:
            z_init = trajectory_to_z(initial_trajectory.X, initial_trajectory.U)
        else:
            X0 = jnp.repeat(x0_arr[None, :], N, axis=0)
            U0 = jnp.zeros((N - 1, m), dtype=jnp.float64)
            z_init = trajectory_to_z(X0, U0)

        P_total = n + (N - 1) * n + sum(problem.constraints.p)
        lam = jnp.zeros(P_total, dtype=jnp.float64)
        mu = jnp.zeros(len(z_init), dtype=jnp.float64)

        return cls(
            x0=x0_arr,
            t0=t0_arr,
            xf=xf_arr,
            lam=lam,
            mu=mu,
            Z=z_init,
            dt=dt_arr,
            n=n,
            m=m,
            N=N,
        )

    def with_measurement(self, x: jax.Array | Sequence[float], t: float | jax.Array) -> "MPCState":
        """Return a new MPCState with updated measured initial state x0 and timestamp t0.

        Parameters
        ----------
        x : jax.Array | Sequence[float]
            New measured state of shape (n,).
        t : float | jax.Array
            New timestamp scalar.

        Returns
        -------
        MPCState
            New state instance with updated measurement.
        """
        x_arr = jnp.asarray(x, dtype=self.x0.dtype)
        t_arr = jnp.asarray(t, dtype=self.t0.dtype)

        X, U = z_to_trajectory(self.Z, self.N, self.n, self.m)
        X_new = X.at[0].set(x_arr)
        Z_new = trajectory_to_z(X_new, U)

        return MPCState(
            x0=x_arr,
            t0=t_arr,
            xf=self.xf,
            lam=self.lam,
            mu=self.mu,
            Z=Z_new,
            dt=self.dt,
            n=self.n,
            m=self.m,
            N=self.N,
        )

    def with_goal(self, xf: jax.Array | Sequence[float]) -> "MPCState":
        """Return a new MPCState with updated goal state xf.

        Parameters
        ----------
        xf : jax.Array | Sequence[float]
            New goal state of shape (n,).

        Returns
        -------
        MPCState
            New state instance with updated goal.
        """
        xf_arr = jnp.asarray(xf, dtype=self.xf.dtype)
        return MPCState(
            x0=self.x0,
            t0=self.t0,
            xf=xf_arr,
            lam=self.lam,
            mu=self.mu,
            Z=self.Z,
            dt=self.dt,
            n=self.n,
            m=self.m,
            N=self.N,
        )

    def shift(self, dt: float | jax.Array | None = None) -> "MPCState":
        """Shift the primal trajectory and timestamps forward for MPC warm-starting.

        Parameters
        ----------
        dt : float | jax.Array | None, optional
            Step duration of the completed step. Defaults to self.dt[0].

        Returns
        -------
        MPCState
            New state instance with warm-start trajectory shifted forward.
        """
        X, U = z_to_trajectory(self.Z, self.N, self.n, self.m)
        new_X = jnp.concatenate([X[1:], X[-1:]], axis=0)
        new_U = jnp.concatenate([U[1:], U[-1:]], axis=0)
        new_Z = trajectory_to_z(new_X, new_U)

        dt_step = self.dt[0] if (self.dt.ndim > 0 and len(self.dt) > 0) else self.dt
        step_val = dt_step if dt is None else jnp.asarray(dt, dtype=self.t0.dtype)
        new_t0 = self.t0 + step_val
        new_x0 = new_X[0]

        return MPCState(
            x0=new_x0,
            t0=new_t0,
            xf=self.xf,
            lam=self.lam,
            mu=self.mu,
            Z=new_Z,
            dt=self.dt,
            n=self.n,
            m=self.m,
            N=self.N,
        )

    def states(self) -> jax.Array:
        """Return stacked state trajectory X of shape (N, n)."""
        X, _ = z_to_trajectory(self.Z, self.N, self.n, self.m)
        return X

    def controls(self) -> jax.Array:
        """Return stacked control trajectory U of shape (N - 1, m)."""
        _, U = z_to_trajectory(self.Z, self.N, self.n, self.m)
        return U

    def initial_states(self, X0: jax.Array) -> "MPCState":
        """Return a new MPCState with states in Z replaced by X0.

        Parameters
        ----------
        X0 : jax.Array
            New states of shape (N, n).

        Returns
        -------
        MPCState
            New state instance with updated states.
        """
        _, U = z_to_trajectory(self.Z, self.N, self.n, self.m)
        Z_new = trajectory_to_z(jnp.asarray(X0, dtype=self.Z.dtype), U)
        return MPCState(
            x0=self.x0,
            t0=self.t0,
            xf=self.xf,
            lam=self.lam,
            mu=self.mu,
            Z=Z_new,
            dt=self.dt,
            n=self.n,
            m=self.m,
            N=self.N,
        )

    def initial_controls(self, U0: jax.Array) -> "MPCState":
        """Return a new MPCState with controls in Z replaced by U0.

        Parameters
        ----------
        U0 : jax.Array
            New controls of shape (N - 1, m).

        Returns
        -------
        MPCState
            New state instance with updated controls.
        """
        X, _ = z_to_trajectory(self.Z, self.N, self.n, self.m)
        Z_new = trajectory_to_z(X, jnp.asarray(U0, dtype=self.Z.dtype))
        return MPCState(
            x0=self.x0,
            t0=self.t0,
            xf=self.xf,
            lam=self.lam,
            mu=self.mu,
            Z=Z_new,
            dt=self.dt,
            n=self.n,
            m=self.m,
            N=self.N,
        )

    def to_trajectory(self) -> Trajectory:
        """Convert state to a Trajectory instance."""
        X, U = z_to_trajectory(self.Z, self.N, self.n, self.m)
        t_arr = self.t0 + jnp.concatenate([jnp.zeros(1, dtype=self.Z.dtype), jnp.cumsum(self.dt)])
        return Trajectory(X=X, U=U, t=t_arr, dt=self.dt)


def states(state: MPCState) -> jax.Array:
    """Return stacked state trajectory X of shape (N, n)."""
    return state.states()


def controls(state: MPCState) -> jax.Array:
    """Return stacked control trajectory U of shape (N - 1, m)."""
    return state.controls()


def initial_states(state: MPCState, X0: jax.Array) -> MPCState:
    """Return a new MPCState with states in Z replaced by X0."""
    return state.initial_states(X0)


def initial_controls(state: MPCState, U0: jax.Array) -> MPCState:
    """Return a new MPCState with controls in Z replaced by U0."""
    return state.initial_controls(U0)


def cost(problem: Problem, state: MPCState) -> jax.Array:
    """Evaluate objective scalar cost J(state.Z) given Problem and MPCState."""
    from trajopt.transcription.transcription import eval_f  # noqa: PLC0415 -- avoid circular import

    return eval_f(problem, state.Z, state.t0, state.dt, state.xf)


def rollout(
    problem: Problem | AbstractModel,
    state: MPCState | Trajectory,
    x0: jax.Array | None = None,
) -> Trajectory:
    """Forward simulate dynamical system given Problem and MPCState or Model and Trajectory.

    Parameters
    ----------
    problem : Problem | AbstractModel
        Problem instance or continuous/discrete dynamical model.
    state : MPCState | Trajectory
        MPCState holding initial state and controls, or Trajectory holding controls and step durations.
    x0 : jax.Array | None, optional
        Initial state condition of shape (n,). Defaults to state's initial state.

    Returns
    -------
    Trajectory
        Simulated state and control trajectory.
    """
    from trajopt.dynamics.rollout import rollout as _rollout_traj  # noqa: PLC0415 -- avoid circular import
    from trajopt.dynamics.rollout import rollout_states  # noqa: PLC0415 -- avoid circular import

    if isinstance(problem, Problem) and isinstance(state, MPCState):
        discrete_model = _extract_discrete_model(problem)
        x0_val = state.x0 if x0 is None else jnp.asarray(x0, dtype=state.Z.dtype)
        U_controls = state.controls()
        X_sim = rollout_states(discrete_model, x0_val, U_controls, t=state.t0, dt=state.dt)
        t_arr = state.t0 + jnp.concatenate([jnp.zeros(1, dtype=state.Z.dtype), jnp.cumsum(state.dt)])
        return Trajectory(X=X_sim, U=U_controls, t=t_arr, dt=state.dt)

    if isinstance(problem, (DiscreteDynamics, ContinuousDynamics)) and isinstance(state, Trajectory):
        return _rollout_traj(problem, state, x0=x0)

    msg = f"Unsupported argument types for rollout: problem={type(problem).__name__}, state={type(state).__name__}"
    raise TypeError(msg)


def solve(
    problem: Problem,
    state: MPCState,
    *,
    solver: str | Callable[..., Any] = "ipopt",
    options: Mapping[str, Any] | None = None,
) -> MPCState:
    """Solve the optimal control problem and return an updated MPCState with optimal trajectory and multipliers.

    Parameters
    ----------
    problem : Problem
        Problem structure containing model, objective, constraints, and horizon.
    state : MPCState
        Current per-step state holding initial state, warm-start trajectory, and goal.
    solver : str | Any, optional
        Solver backend to use ("ipopt", "osqp", "clarabel") or a callable. Defaults to "ipopt".
    options : Mapping[str, Any] | None, optional
        Solver options passed to backend (e.g. max_iter, tol, print_level, verbose).

    Returns
    -------
    MPCState
        New state containing optimal trajectory Z, dual multipliers lam, mu, and boundary conditions.
    """
    res: Any
    if isinstance(solver, str):
        solver_name = solver.strip().lower()
        if solver_name == "ipopt":
            from trajopt.transcription.ipopt import solve_ipopt  # noqa: PLC0415 -- avoid circular import

            res = solve_ipopt(problem=problem, x0=state, options=options)
        elif solver_name == "osqp":
            from trajopt.transcription.osqp import solve_osqp  # noqa: PLC0415 -- avoid circular import

            res = solve_osqp(problem=problem, x0=state, options=options)
        elif solver_name == "clarabel":
            from trajopt.transcription.clarabel import solve_clarabel  # noqa: PLC0415 -- avoid circular import

            res = solve_clarabel(problem=problem, x0=state, options=options)
        else:
            msg = f"Unknown solver backend: '{solver}'. Expected 'ipopt', 'osqp', 'clarabel', or callable."
            raise ValueError(msg)
    elif callable(solver):
        solver_fn: Any = solver
        res = solver_fn(problem=problem, x0=state, options=options)
    else:
        msg = f"Invalid solver type: {type(solver).__name__}. Expected str or callable."
        raise TypeError(msg)

    mult_con = res.info.get("mult_g")
    if mult_con is None:
        mult_con = res.info.get("y")
    if mult_con is None:
        mult_con = res.info.get("z")
    lam = jnp.asarray(mult_con, dtype=state.Z.dtype) if mult_con is not None else state.lam

    mult_x_L = res.info.get("mult_x_L")
    mult_x_U = res.info.get("mult_x_U")
    if mult_x_U is not None and mult_x_L is not None:
        mu = jnp.asarray(mult_x_U - mult_x_L, dtype=state.Z.dtype)
    else:
        mu = state.mu

    return MPCState(
        x0=state.x0,
        t0=state.t0,
        xf=state.xf,
        lam=lam,
        mu=mu,
        Z=res.Z,
        dt=state.dt,
        n=state.n,
        m=state.m,
        N=state.N,
    )
