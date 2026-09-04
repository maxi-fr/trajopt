from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.constraint_list import BuiltConstraintList, ConstraintList
from trajopt.costs.objective import Objective
from trajopt.dynamics.base import AbstractModel, DiscreteDynamics, IntegratorCallable
from trajopt.dynamics.integrators import Integrator

if TYPE_CHECKING:
    from trajopt.expansions import Expansion
    from trajopt.models.transforms import LinearTrajectoryModel
    from trajopt.trajectory import Trajectory


class BoundaryConditions(eqx.Module):
    """Traced boundary data of one solve: where the trajectory starts, when, and what it aims at.

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
        Reference states of shape (N, n) the quadratic objective is retargeted onto, or None to
        leave the objective at the reference it was built with.
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
    """

    model: DiscreteDynamics
    obj: Objective
    constraints: BuiltConstraintList
    N: int = eqx.field(static=True)
    dt: jax.Array

    def __init__(  # noqa: PLR0913, PLR0917 -- the six pieces that define a transcription
        self,
        model: AbstractModel,
        obj: Objective,
        constraints: BuiltConstraintList | ConstraintList | None = None,
        N: int | None = None,
        dt: float | jax.Array = 0.05,
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
        self.dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N_val - 1,))

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
