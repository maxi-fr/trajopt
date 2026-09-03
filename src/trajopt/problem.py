import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.constraint_list import BuiltConstraintList, ConstraintList
from trajopt.costs.objective import Objective
from trajopt.dynamics.base import AbstractModel, DiscreteDynamics, IntegratorCallable
from trajopt.dynamics.integrators import Integrator
from trajopt.program import program_for
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z, _z_to_trajectory

if TYPE_CHECKING:
    from trajopt.expansions import Expansion
    from trajopt.solvers.al import ALConstraints
    from trajopt.transcription.result import Solver


class BoundaryConditions(eqx.Module):
    """Traced boundary data of one solve: where the trajectory starts, when, and what it aims at.

    Every field is an array leaf and none is `eqx.field(static=True)`, so an instance can be
    handed to a jitted solver core as an ordinary traced argument: moving the target between MPC
    steps changes values, not the pytree, and forces no recompile. That is the whole point of the
    type, and the reason the target no longer lives fused into the Objective's linear terms.

    A goal point is just a constant reference window, so one mechanism serves regulation and
    tracking alike.

    Parameters
    ----------
    x0 : jax.Array
        Initial state of shape (n,).
    t0 : jax.Array
        Initial timestamp of shape ().
    X_ref : jax.Array | None
        Reference states of shape (N, n) the quadratic objective is retargeted onto, or None to
        leave the objective at the reference it was built with.
    U_ref : jax.Array | None
        Reference controls of shape (N - 1, m), paired with X_ref and None exactly when it is.
    """

    x0: jax.Array
    t0: jax.Array
    X_ref: jax.Array | None = None
    U_ref: jax.Array | None = None

    @property
    def xf(self) -> jax.Array | None:
        """Terminal reference state of shape (n,), the run-time goal constraints read, or None."""
        return None if self.X_ref is None else self.X_ref[-1]

    def retarget(self, obj: Objective) -> Objective:
        """Objective aimed at this reference window, or `obj` unchanged when there is none."""
        if self.X_ref is None or self.U_ref is None or not obj.is_quadratic:
            return obj
        return obj.with_reference(self.X_ref, self.U_ref)


def retarget_to_goal(obj: Objective, xf: jax.Array | None) -> Objective:
    """Objective regulated to the run-time goal xf of shape (n,), held constant over the horizon.

    A goal point is a constant reference window, so regulation and tracking go through the one
    `with_reference` mechanism. Returns `obj` untouched when there is no goal, or when the cost is
    not quadratic and so exposes no linear terms to retarget.
    """
    if xf is None or not obj.is_quadratic:
        return obj
    xf_arr = jnp.asarray(xf)
    X_ref = jnp.broadcast_to(xf_arr, (obj.N, xf_arr.shape[-1]))
    U_ref = jnp.zeros((obj.N - 1, obj.m), dtype=xf_arr.dtype)
    return obj.with_reference(X_ref, U_ref)


def retarget_problem(problem: "Problem", bc: BoundaryConditions | None) -> "Problem":
    """Problem whose objective is aimed at `bc`'s reference window, unchanged when there is none.

    Called at the top of every traced solver core: `bc` arrives as a traced argument, so the
    rebuilt objective holds tracers and the core compiles once for every target.
    """
    if bc is None:
        return problem
    obj = bc.retarget(problem.obj)
    return problem if obj is problem.obj else eqx.tree_at(lambda p: p.obj, problem, obj)


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

    def solve(self, state: "MPCState", solver: "Solver | None" = None) -> "MPCState":
        """Solve this problem from `state` with `solver`'s Program, returning an updated MPCState.

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
            New state containing optimal trajectory Z and dual multipliers lam, mu. The solve's
            status is on the returned `SolverResult`, which this delegator drops; read it by
            calling the solver (or its Program) directly.
        """
        if solver is None:
            from trajopt.transcription.ipopt import Ipopt  # noqa: PLC0415 -- avoid an import cycle

            solver = Ipopt()

        res = program_for(solver, self).solve(state)

        # A backend that returns no duals (an empty vector) leaves the state's own untouched.
        lam = jnp.asarray(res.lam, dtype=state.Z.dtype) if len(res.lam) > 0 else state.lam
        mu = jnp.asarray(res.mu, dtype=state.Z.dtype) if len(res.mu) > 0 else state.mu

        # AL solvers (ticket 29) populate `res.al` with their padded per-knot duals/penalties for
        # MPCState warm-starting; every other backend leaves it absent, so the prior al survives.
        al = getattr(res, "al", None)

        return dataclasses.replace(state, lam=lam, mu=mu, Z=res.Z, al=al if al is not None else state.al)

    def cost(self, state: "MPCState") -> jax.Array:
        """Evaluate objective scalar cost J(state.Z) for this problem at `state`."""
        from trajopt.transcription.transcription import eval_f  # noqa: PLC0415 -- avoid circular import

        return eval_f(self, state.Z, state.t0, state.dt, state.bc)


class MPCState(eqx.Module):
    """Per-step MPC state holding boundary conditions, trajectory, multipliers, and metadata.

    Parameters
    ----------
    bc : BoundaryConditions
        Traced boundary data: initial state x0 of shape (n,), timestamp t0, and the optional
        reference window the objective and any GoalConstraint are retargeted onto.
    lam : jax.Array
        Constraint dual multipliers vector of shape (P,).
    mu : jax.Array
        Primal variable bounds multipliers vector, one entry per primal variable of the
        transcription.
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
    al : ALConstraints | None
        Padded augmented-Lagrangian duals and penalties (ticket 28), or None before an AL solve
        has populated them. Distinct from `lam` / `mu`, which keep their transcription meaning;
        carried as a pytree field so AL warm-starts survive across MPC steps.
    """

    bc: BoundaryConditions
    lam: jax.Array
    mu: jax.Array
    Z: jax.Array
    dt: jax.Array
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    al: "ALConstraints | None" = None

    @property
    def x0(self) -> jax.Array:
        """Initial state of shape (n,), delegated to the boundary conditions."""
        return self.bc.x0

    @property
    def t0(self) -> jax.Array:
        """Initial timestamp of shape (), delegated to the boundary conditions."""
        return self.bc.t0

    @property
    def xf(self) -> jax.Array | None:
        """Run-time goal of shape (n,) -- the last knot of the reference window -- or None."""
        return self.bc.xf

    @classmethod
    def initial(  # noqa: PLR0913 -- Initial state factory takes 8 arguments
        cls,
        problem: Problem,
        x0: jax.Array | Sequence[float],
        *,
        t0: float | jax.Array = 0.0,
        xf: jax.Array | Sequence[float] | None = None,
        reference: Trajectory | None = None,
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
            Run-time goal state of shape (n,), held constant over the horizon as the reference
            window a quadratic objective is retargeted onto and any GoalConstraint reads.
            Defaults to None, which leaves a shape-only `LQRObjective` regulating to the origin
            and any GoalConstraint at its build-time xf. Rejected when the objective already
            tracks a reference, which a constant goal would flatten.
        reference : Trajectory | None, optional
            Full reference window of N knot points, used in place of a constant goal when the
            target varies over the horizon. Its last state serves as the run-time goal. Defaults
            to None.
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

        Raises
        ------
        ValueError
            If a target is given but nothing in the problem reads it, if both xf and reference
            are given, or if xf is given against an objective that already tracks a reference.
        """
        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)

        x0_arr = jnp.asarray(x0, dtype=jnp.float64)
        t0_arr = jnp.asarray(t0, dtype=jnp.float64)
        dt_arr = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N - 1,))

        if xf is not None and reference is not None:
            msg = "Pass either xf (a constant goal) or reference (a window), not both."
            raise ValueError(msg)
        if xf is not None and problem.obj.carries_reference:
            msg = (
                "This objective already tracks a build-time reference, so a constant goal xf would "
                "silently overwrite it at every knot point. Pass reference=Trajectory(...) with the "
                "window you want tracked, or build the objective with LQRObjective, which carries "
                "shape only."
            )
            raise ValueError(msg)
        if (xf is not None or reference is not None) and not (
            problem.obj.is_quadratic or problem.constraints.has_goal_constraint()
        ):
            msg = (
                f"A run-time target was given but nothing in the problem reads it: the "
                f"{type(problem.obj.stage_cost).__name__} objective is not quadratic, so it exposes no linear "
                f"terms to retarget, and no GoalConstraint is registered. Use a quadratic objective, or add a "
                f"GoalConstraint, or leave the target unset."
            )
            raise ValueError(msg)

        if reference is not None:
            X_ref = jnp.asarray(reference.X, dtype=jnp.float64)
            U_ref = jnp.asarray(reference.U, dtype=jnp.float64)
        elif xf is not None:
            X_ref = jnp.repeat(jnp.asarray(xf, dtype=jnp.float64)[None, :], N, axis=0)
            U_ref = jnp.zeros((N - 1, m), dtype=jnp.float64)
        else:
            X_ref, U_ref = None, None

        if initial_z is not None:
            z_init = jnp.asarray(initial_z, dtype=jnp.float64)
        elif initial_trajectory is not None:
            z_init = _trajectory_to_z(initial_trajectory.X, initial_trajectory.U)
        else:
            X0 = jnp.repeat(x0_arr[None, :], N, axis=0)
            U0 = jnp.zeros((N - 1, m), dtype=jnp.float64)
            z_init = _trajectory_to_z(X0, U0)

        P_total = n + (N - 1) * n + sum(problem.constraints.p)
        lam = jnp.zeros(P_total, dtype=jnp.float64)
        mu = jnp.zeros(len(z_init), dtype=jnp.float64)

        return cls(
            bc=BoundaryConditions(x0=x0_arr, t0=t0_arr, X_ref=X_ref, U_ref=U_ref),
            lam=lam,
            mu=mu,
            Z=z_init,
            dt=dt_arr,
            n=n,
            m=m,
            N=N,
        )

    def with_measurement(self, x: jax.Array | Sequence[float], t: float | jax.Array) -> "MPCState":
        """Return a new MPCState with updated measured initial state x0 of shape (n,) and timestamp t0."""
        x_arr = jnp.asarray(x, dtype=self.x0.dtype)
        t_arr = jnp.asarray(t, dtype=self.t0.dtype)

        X, U = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        Z_new = _trajectory_to_z(X.at[0].set(x_arr), U)

        return dataclasses.replace(self, bc=dataclasses.replace(self.bc, x0=x_arr, t0=t_arr), Z=Z_new)

    def with_goal(self, xf: jax.Array | Sequence[float]) -> "MPCState":
        """Return a new MPCState whose reference window is the constant goal state xf of shape (n,).

        Raises
        ------
        ValueError
            If this state was built without a target, since nothing was checked to read one.
        """
        if self.bc.X_ref is None:
            msg = "This MPCState was built without a goal. Pass xf to MPCState.initial to make the goal run-time."
            raise ValueError(msg)
        X_ref = jnp.repeat(jnp.asarray(xf, dtype=self.bc.X_ref.dtype)[None, :], self.N, axis=0)
        return dataclasses.replace(self, bc=dataclasses.replace(self.bc, X_ref=X_ref))

    def with_reference(self, reference: Trajectory) -> "MPCState":
        """Return a new MPCState whose reference window is `reference`, a Trajectory of N knot points."""
        return dataclasses.replace(
            self,
            bc=dataclasses.replace(
                self.bc,
                X_ref=jnp.asarray(reference.X, dtype=jnp.float64),
                U_ref=jnp.asarray(reference.U, dtype=jnp.float64),
            ),
        )

    def shift(self, dt: float | jax.Array | None = None) -> "MPCState":
        """Shift the primal trajectory and timestamps forward by dt for MPC warm-starting, defaulting to self.dt[0]."""
        X, U = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        new_X = jnp.concatenate([X[1:], X[-1:]], axis=0)
        new_U = jnp.concatenate([U[1:], U[-1:]], axis=0)
        new_Z = _trajectory_to_z(new_X, new_U)

        dt_step = self.dt[0] if (self.dt.ndim > 0 and len(self.dt) > 0) else self.dt
        step_val = dt_step if dt is None else jnp.asarray(dt, dtype=self.t0.dtype)

        return dataclasses.replace(
            self,
            bc=dataclasses.replace(self.bc, x0=new_X[0], t0=self.t0 + step_val),
            Z=new_Z,
        )

    @property
    def states(self) -> jax.Array:
        """Stacked state trajectory X of shape (N, n), unpacked from Z on each access."""
        X, _ = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        return X

    @property
    def controls(self) -> jax.Array:
        """Stacked control trajectory U of shape (N - 1, m), unpacked from Z on each access."""
        _, U = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        return U

    def with_states(self, X0: jax.Array) -> "MPCState":
        """Return a new MPCState with the states in Z replaced by X0 of shape (N, n)."""
        _, U = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        return dataclasses.replace(self, Z=_trajectory_to_z(jnp.asarray(X0, dtype=self.Z.dtype), U))

    def with_controls(self, U0: jax.Array) -> "MPCState":
        """Return a new MPCState with the controls in Z replaced by U0 of shape (N - 1, m)."""
        X, _ = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        return dataclasses.replace(self, Z=_trajectory_to_z(X, jnp.asarray(U0, dtype=self.Z.dtype)))

    def to_trajectory(self) -> Trajectory:
        """Convert state to a Trajectory instance."""
        X, U = _z_to_trajectory(self.Z, self.N, self.n, self.m)
        t_arr = self.t0 + jnp.concatenate([jnp.zeros(1, dtype=self.Z.dtype), jnp.cumsum(self.dt)])
        return Trajectory(X=X, U=U, t=t_arr, dt=self.dt)
