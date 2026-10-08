from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.constraint_list import BuiltConstraintList, BuiltKnotConstraint, ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.indexed import IndexedConstraint
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.base import CostFunction, QuadraticCostFunction
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost, QuadraticCost
from trajopt.dynamics.base import AbstractModel, DiscreteDynamics, IntegratorCallable
from trajopt.dynamics.integrators import Integrator

if TYPE_CHECKING:
    from trajopt.constraints.base import Constraint
    from trajopt.expansions import Expansion
    from trajopt.models.transforms import LinearTrajectoryModel
    from trajopt.trajectory import Trajectory


class _EpigraphModel(DiscreteDynamics):
    """Model accepting original controls followed by unused epigraph variables."""

    original: DiscreteDynamics

    def __init__(self, original: DiscreteDynamics) -> None:
        """Double the control dimension while retaining state and output dimensions."""
        super().__init__(n=original.n, m=2 * original.m, ne=original.ne, p=original.p)
        self.original = original

    def discrete_dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array, dt: float | jax.Array) -> jax.Array:
        """Advance the original model using the first m control coordinates."""
        return self.original.discrete_dynamics(x, u[: self.original.m], t, dt)

    def state_jacobian(
        self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0, *args: float | jax.Array
    ) -> jax.Array:
        """Return the original dynamics state Jacobian of shape (n, n)."""
        return self.original.state_jacobian(x, u[: self.original.m], t, *args)

    def control_jacobian(
        self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0, *args: float | jax.Array
    ) -> jax.Array:
        """Append zero epigraph columns to the original control Jacobian."""
        original = self.original.control_jacobian(x, u[: self.original.m], t, *args)
        return jnp.pad(original, ((0, 0), (0, self.original.m)))

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Retain the original model's error-state coordinates."""
        return self.original.state_diff(x, x0)

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Retain the original error-state Jacobian of shape (n, ne)."""
        return self.original.errstate_jacobian(x)

    def output(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate original output using only original controls when present."""
        return self.original.output(x, None if u is None else u[: self.original.m], t)

    def output_state_jacobian(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Return the original output state Jacobian of shape (p, n)."""
        return self.original.output_state_jacobian(x, None if u is None else u[: self.original.m], t)

    def output_control_jacobian(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        """Append zero epigraph columns to the original output control Jacobian."""
        original = self.original.output_control_jacobian(x, u[: self.original.m], t)
        return jnp.pad(original, ((0, 0), (0, self.original.m)))

    def has_control_feedthrough(self) -> bool:
        """Retain whether original controls feed directly into output."""
        return self.original.has_control_feedthrough()


class _EpigraphStageCost(CostFunction):
    """Lift a nonquadratic stage cost and add a linear epigraph cost."""

    original: CostFunction
    weight: jax.Array

    def __init__(self, original: CostFunction, weight: float | jax.Array) -> None:
        """Store the original stage cost and positive linear slack price."""
        super().__init__(n=original.n, m=2 * original.m)
        self.original = original
        self.weight = jnp.asarray(weight)

    def evaluate(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate original stage cost plus weight times the epigraph variables."""
        if u is None:
            msg = "Augmented control [u, s] is required."
            raise ValueError(msg)
        m = self.original.m
        return self.original.evaluate(x, u[:m], t) + self.weight * jnp.sum(u[m:])

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate all original stage costs and add each stage's epigraph price."""
        m = self.original.m
        return self.original.stage_costs(X, U[:, :m], t) + self.weight * jnp.sum(U[:, m:], axis=1)

    def unstacked(self, k: int) -> CostFunction:
        """Return a cost for knot k, including its original stage parameters."""
        return _EpigraphStageCost(self.original.unstacked(k), self.weight)


def _epigraph_objective(obj: Objective, weight: float) -> Objective:
    """Lift an Objective onto doubled controls and price the epigraph coordinates."""
    stage = obj.stage_cost
    m = obj.m
    linear = jnp.concatenate([obj.linear_control_cost, jnp.full((obj.N - 1, m), weight)], axis=-1)
    if isinstance(stage, DiagonalCost):
        lifted = DiagonalCost(
            Q=stage.Q,
            R=jnp.pad(stage.R, ((0, 0), (0, m))),
            q=stage.q,
            r=jnp.concatenate([stage.r, jnp.full((obj.N - 1, m), weight)], axis=-1),
            c=stage.c,
        )
    elif isinstance(stage, QuadraticCostFunction):
        dense = stage.to_quadratic()
        lifted = QuadraticCost(
            Q=dense.Q,
            R=jnp.pad(dense.R, ((0, 0), (0, m), (0, m))),
            H=jnp.pad(dense.H, ((0, 0), (0, m), (0, 0))) if dense.H is not None else None,
            q=dense.q,
            r=jnp.concatenate([dense.r, jnp.full((obj.N - 1, m), weight)], axis=-1),
            c=dense.c,
        )
    else:
        lifted = _EpigraphStageCost(stage, weight)
    return Objective(stage_cost=lifted, terminal_cost=obj.terminal_cost, N=obj.N, linear_control_cost=linear)


def _epigraph_constraints(original: BuiltConstraintList) -> BuiltConstraintList:
    """Lift Knot Point rows and bounds and remap Horizon indices to doubled controls."""
    n, m, N = original.n, original.m, original.N
    eye = jnp.eye(m)
    A = jnp.block([[eye, -eye], [-eye, -eye]])
    epigraph = LinearConstraint(n=n, m=2 * m, A=A, b=jnp.zeros(2 * m), inds=range(n, n + 2 * m))
    evaluators = []
    for k, evaluator in enumerate(original.knot_evaluators):
        lifted: list[Constraint] = [
            GoalConstraint(n=n, xf=con.xf, inds=con.inds, m=2 * m)
            if isinstance(con, GoalConstraint)
            else IndexedConstraint(n=n, m=2 * m, constraint=con, ix=range(n), iu=range(m))
            for con in evaluator.constraints
        ]
        if k < N - 1:
            lifted.append(epigraph)
        evaluators.append(BuiltKnotConstraint(lifted, n=n, m=2 * m, is_terminal=(k == N - 1)))

    old_stage, new_stage = n + m, n + 2 * m

    def new_index(i: int) -> int:
        """Map an original Primal Vector coordinate into the augmented vector."""
        if i < (N - 1) * old_stage:
            k, offset = divmod(i, old_stage)
            return k * new_stage + offset
        return (N - 1) * new_stage + i - (N - 1) * old_stage

    horizon = [
        LinearHorizonConstraint(con.A, con.b, sense=con.cone, inds=tuple(new_index(i) for i in con.inds))
        for con in original.horizon_constraints
    ]
    xL, xU, uL, uU = original.primal_bounds()
    shape = (N - 1, m)
    uL = np.concatenate([uL, np.full(shape, -np.inf)], axis=1)
    uU = np.concatenate([uU, np.full(shape, np.inf)], axis=1)
    return BuiltConstraintList(evaluators, n=n, m=2 * m, N=N, bounds=(xL, xU, uL, uU), horizon_constraints=horizon)


def add_l1_epigraph(problem: "Problem", weight: float) -> "Problem":
    """Add exact weight * sum(abs(u)) by augmenting controls as [u, s] at each Knot Point.

    The returned Problem has control width 2m. Trajectories and reference windows supplied to it
    therefore need controls [u, s] of shape (N - 1, 2m). The original Problem is unchanged.
    """
    if not isinstance(weight, (int, float, np.floating)) or not np.isfinite(weight) or weight <= 0:
        msg = "weight must be a finite positive scalar."
        raise ValueError(msg)
    return Problem(
        model=_EpigraphModel(problem.model),
        obj=_epigraph_objective(problem.obj, float(weight)),
        constraints=_epigraph_constraints(problem.constraints),
        N=problem.N,
        dt=problem.dt,
        binary_control_indices=problem.binary_control_indices,
    )


class BoundaryConditions(eqx.Module):
    """Traced boundary data of one solve: where the Trajectory starts, when, and what it aims at.

    Every field is an array leaf and none is `eqx.field(static=True)`, so an instance can be
    handed to a jitted solver core as an ordinary traced argument: moving the target between MPC
    steps changes values, not the pytree, and forces no recompile. That is the whole point of the
    type, and the reason the target no longer lives fused into the Objective's linear terms.

    The window and the goal are separate concepts. `X_ref`/`U_ref` are what the quadratic
    objective tracks over the horizon; `xf` is the destination terminal constraints bind. They
    coincide only when the window is a constant goal window, which is why one leaf sufficed until
    the window was allowed to move.

    Parameters
    ----------
    x0 : jax.Array
        Initial state of shape (n,).
    t0 : jax.Array
        Initial timestamp of shape ().
    X_ref : jax.Array | None
        Reference states of shape (N, n) the quadratic Objective is retargeted onto, or None to
        leave the Objective at the reference it was built with.
    U_ref : jax.Array | None
        Reference controls of shape (N - 1, m), paired with X_ref and None exactly when it is.
    xf : jax.Array | None
        Terminal goal of shape (n,) that a GoalConstraint binds, or None to leave every goal
        constraint at the target it was built with.
    """

    x0: jax.Array
    t0: jax.Array
    X_ref: jax.Array | None = None
    U_ref: jax.Array | None = None
    xf: jax.Array | None = None

    def retarget(self, obj: Objective) -> Objective:
        """Objective aimed at this reference window, regulated to `xf` when there is no window."""
        if self.X_ref is None or self.U_ref is None:
            return retarget_to_goal(obj, self.xf)
        if not obj.is_quadratic:
            return obj
        return obj.with_reference(self.X_ref, self.U_ref)


def retarget_to_goal(obj: Objective, xf: jax.Array | None) -> Objective:
    """Objective regulated to the run-time goal xf of shape (n,), held constant over the Horizon.

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
    """Problem whose Objective is aimed at `bc`'s reference window, unchanged when there is none.

    Called at the top of every traced solver core: `bc` arrives as a traced argument, so the
    rebuilt Objective holds tracers and the core compiles once for every target.
    """
    if bc is None:
        return problem
    obj = bc.retarget(problem.obj)
    return problem if obj is problem.obj else eqx.tree_at(lambda p: p.obj, problem, obj)


class Problem(eqx.Module):
    """Problem structure holding model, objective, constraints, and the horizon's time grid.

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
    dt : float | jax.Array, optional
        Step durations of the horizon, a scalar or an array of shape (N - 1,). Structural rather
        than per-step data: the time grid a Program is compiled against does not move as the
        horizon recedes. Defaults to 0.05.
    integrator : Integrator | IntegratorCallable | None, optional
        Integrator instance for continuous models. Defaults to None, meaning RK4.
    binary_control_indices : Sequence[int], optional
        Unique control coordinates constrained to [0, 1] at each nonterminal Knot Point.
    """

    model: DiscreteDynamics
    obj: Objective
    constraints: BuiltConstraintList
    N: int = eqx.field(static=True)
    dt: jax.Array
    binary_control_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913, PLR0917 -- the six pieces that define a transcription
        self,
        model: AbstractModel,
        obj: Objective,
        constraints: BuiltConstraintList | ConstraintList | None = None,
        N: int | None = None,
        dt: float | jax.Array = 0.05,
        integrator: Integrator | IntegratorCallable | None = None,
        binary_control_indices: Sequence[int] = (),
    ) -> None:
        """Build a Problem and intersect declared binary control coordinates with box bounds."""
        n = int(model.n)
        m = int(model.m)
        N_val = int(N if N is not None else obj.N)

        cl = ConstraintList(n=n, m=m, N=N_val) if constraints is None else constraints
        built_con = cl.build()
        binary = tuple(int(i) for i in binary_control_indices)
        if len(set(binary)) != len(binary) or any(i < 0 or i >= m for i in binary):
            msg = f"Binary control indices must be unique and in [0, {m})."
            raise ValueError(msg)
        if binary:
            xL, xU, uL, uU = built_con.primal_bounds()
            uL, uU = uL.copy(), uU.copy()
            uL[:, binary] = np.maximum(uL[:, binary], 0.0)
            uU[:, binary] = np.minimum(uU[:, binary], 1.0)
            if np.any(uL > uU):
                msg = "Binary control bounds contradict existing control bounds."
                raise ValueError(msg)
            built_con = BuiltConstraintList(
                built_con.knot_evaluators,
                n,
                m,
                N_val,
                bounds=(xL, xU, uL, uU),
                horizon_constraints=built_con.horizon_constraints,
            )

        self.model = model.discretize(integrator)
        self.obj = obj
        self.constraints = built_con
        self.N = N_val
        self.dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N_val - 1,))
        self.binary_control_indices = binary

    def cost_expansion(self, traj: "Trajectory") -> "Expansion":
        """Stacked first- and second-order cost expansion in error coordinates along traj."""
        return self.obj.cost_expansion(traj, self.model)

    def dynamics_expansion(self, traj: "Trajectory") -> "Expansion":
        """Stacked first-order dynamics expansion in error coordinates along traj."""
        return self.model.dynamics_expansion(traj)

    def operating_trajectory(
        self,
        operating_point: "Trajectory | jax.Array | None" = None,
        t0: float | jax.Array = 0.0,
    ) -> "Trajectory":
        """Operating Point as a Trajectory on this problem's time grid, the origin when None.

        Parameters
        ----------
        operating_point : Trajectory | jax.Array | None, optional
            Expansion point as a Trajectory or a flat Primal Vector of shape (N * n + (N - 1) * m,).
            A Trajectory is returned with its own time grid; a flat vector is given this problem's.
        t0 : float | jax.Array, optional
            Initial timestamp of the grid a flat vector is placed on. Defaults to 0.0.
        """
        from trajopt.trajectory import Trajectory as _Trajectory  # noqa: PLC0415 -- avoid an import cycle
        from trajopt.transcription.layout import _z_to_trajectory, operating_point_z  # noqa: PLC0415 -- same

        if isinstance(operating_point, _Trajectory):
            return operating_point

        z_op = operating_point_z(self, operating_point)
        X, U = _z_to_trajectory(z_op, self.N, int(self.model.n), int(self.model.m))
        t = jnp.asarray(t0, dtype=jnp.float64) + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(self.dt)])
        return _Trajectory(X=X, U=U, t=t, dt=self.dt)

    def linearize(
        self,
        operating_point: "Trajectory | jax.Array | None" = None,
        t0: float | jax.Array = 0.0,
    ) -> "LinearTrajectoryModel":
        """Linearize this problem's dynamics about an Operating Point, in error coordinates.

        The same linearization `Model.linearize` performs, reached from the NLP tier's currency:
        the Operating Point may be the flat Primal Vector the transcription works in, which is
        placed on the problem's own time grid before the model is asked for its Jacobians. The
        stagewise tier's per-knot (A_k, B_k) and the dynamics block of the NLP's constraint
        Jacobian therefore come from one computation.

        Parameters
        ----------
        operating_point : Trajectory | jax.Array | None, optional
            Expansion point as a Trajectory or a flat Primal Vector of shape (N * n + (N - 1) * m,).
            Defaults to None, meaning the origin.
        t0 : float | jax.Array, optional
            Initial timestamp of the horizon. Defaults to 0.0.

        Returns
        -------
        LinearTrajectoryModel
            Linearized model exposing stacked Jacobians A of shape (N - 1, ne, ne) and
            B of shape (N - 1, ne, m).
        """
        return self.model.linearize(self.operating_trajectory(operating_point, t0))
