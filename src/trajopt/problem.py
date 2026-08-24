from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.constraint_list import BuiltConstraintList, ConstraintList
from trajopt.costs.objective import Objective
from trajopt.dynamics.base import AbstractModel, DiscreteDynamics, IntegratorCallable
from trajopt.dynamics.integrators import Integrator
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import trajectory_to_z, z_to_trajectory

if TYPE_CHECKING:
    from trajopt.expansions import Expansion
    from trajopt.transcription.result import Solver, SolverStatus


class Problem(eqx.Module):
    """Problem structure holding model, objective, constraints, and horizon.

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
        Integrator instance for continuous models. Defaults to None, meaning RK4.
    """

    model: DiscreteDynamics
    obj: Objective
    constraints: BuiltConstraintList
    N: int = eqx.field(static=True)

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

        cl = ConstraintList(n=n, m=m, N=N_val) if constraints is None else constraints
        built_con = cl.build()

        self.model = model.discretize(integrator)
        self.obj = obj
        self.constraints = built_con
        self.N = N_val

    def cost_expansion(self, traj: Trajectory) -> "Expansion":
        """Stacked first- and second-order cost expansion in error coordinates along traj."""
        return self.obj.cost_expansion(traj, self.model)

    def dynamics_expansion(self, traj: Trajectory) -> "Expansion":
        """Stacked first-order dynamics expansion in error coordinates along traj."""
        return self.model.dynamics_expansion(traj)

    def augmented_lagrangian_expansion(
        self,
        traj: Trajectory,
        expansion: "Expansion",
        lam: "Sequence[jax.Array] | jax.Array | None" = None,
        mu: "float | jax.Array" = 1.0,
    ) -> "Expansion":
        """Add augmented Lagrangian gradient and Hessian contributions into an existing Expansion."""
        return self.constraints.augmented_lagrangian_expansion(traj, expansion, lam, mu, self.model)

    def solve(self, state: "MPCState", solver: "Solver | None" = None) -> "MPCState":
        """Solve this problem from `state` with `solver`, returning an updated MPCState.

        Parameters
        ----------
        state : MPCState
            Current per-step state holding initial state, warm-start trajectory, and goal.
        solver : Solver | None, optional
            Solver backend object (e.g. ``Ipopt()``, ``OSQP(operating_point=...)``). Defaults to
            None, meaning ``Ipopt()``, resolved here rather than at import time to avoid a cycle
            between this module and the transcription layer.

        Returns
        -------
        MPCState
            New state containing optimal trajectory Z, dual multipliers lam, mu, and status.
        """
        if solver is None:
            from trajopt.transcription.ipopt import Ipopt  # noqa: PLC0415 -- avoid an import cycle

            solver = Ipopt()

        from trajopt.transcription.result import normalize_status  # noqa: PLC0415 -- avoid an import cycle

        res = solver.solve(self, state)

        lam = jnp.asarray(res.lam, dtype=state.Z.dtype) if len(res.lam) > 0 else state.lam
        mu = jnp.asarray(res.mu, dtype=state.Z.dtype) if len(res.mu) > 0 else state.mu

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
            status=normalize_status(success=res.success, message=res.message),
        )


class MPCState(eqx.Module):
    """Per-step MPC state holding initial condition, goal, trajectory, multipliers, and metadata.

    Parameters
    ----------
    x0 : jax.Array
        Initial state condition of shape (n,).
    t0 : jax.Array
        Initial timestamp scalar of shape ().
    xf : jax.Array | None
        Run-time goal state vector of shape (n,), or None when the goal is baked into the
        objective and the constraints at build time.
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
    status : SolverStatus | None
        Normalized outcome of the last solve ("converged", "infeasible", "iteration_limit",
        "error"), or None before any solve has run.
    """

    x0: jax.Array
    t0: jax.Array
    xf: jax.Array | None
    lam: jax.Array
    mu: jax.Array
    Z: jax.Array
    dt: jax.Array
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    status: "SolverStatus | None" = eqx.field(static=True, default=None)

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
            Run-time goal state vector of shape (n,), read by a goal-regulating objective and by
            any GoalConstraint. Defaults to None, leaving both at their build-time goal.

        Raises
        ------
        ValueError
            If xf is given but nothing in the problem reads it.
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

        if xf is None:
            xf_arr = None
        elif problem.obj.regulates_to_goal or problem.constraints.has_goal_constraint():
            xf_arr = jnp.asarray(xf, dtype=jnp.float64)
        else:
            msg = (
                f"xf was given but nothing in the problem reads it: the {type(problem.obj.stage_cost).__name__} "
                f"objective carries a reference of its own rather than regulating to a goal, and no "
                f"GoalConstraint is registered. Build the objective with LQRObjective to regulate to xf, or "
                f"add a GoalConstraint, or leave xf unset."
            )
            raise ValueError(msg)

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
            status=self.status,
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

        Raises
        ------
        ValueError
            If this state was built without a goal, since nothing was checked to read one.
        """
        if self.xf is None:
            msg = "This MPCState was built without a goal. Pass xf to MPCState.initial to make the goal run-time."
            raise ValueError(msg)
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
            status=self.status,
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
            status=self.status,
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
            status=self.status,
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
            status=self.status,
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


