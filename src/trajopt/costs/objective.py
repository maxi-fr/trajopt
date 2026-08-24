from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.costs.base import CostFunction, QuadraticCostFunction
from trajopt.costs.quadratic import promote_weights
from trajopt.trajectory import Trajectory

if TYPE_CHECKING:
    from trajopt.dynamics.base import AbstractModel
    from trajopt.expansions import Expansion

_MIN_HORIZON = 2


def _quadratic(cost: CostFunction) -> QuadraticCostFunction:
    """Narrow a cost to a quadratic one, the only kind exposing weight parameters."""
    if not isinstance(cost, QuadraticCostFunction):
        msg = f"{type(cost).__name__} does not expose the quadratic parameters Q, R, H, q, r, c"
        raise TypeError(msg)
    return cost


class Objective(eqx.Module):
    """Stage and terminal costs with their parameters stacked over the horizon.

    Stacked stage parameters have shapes Q (N-1, n) if diagonal or (N-1, n, n) if dense,
    R (N-1, m) or (N-1, m, m), H (N-1, m, n), q (N-1, n), r (N-1, m) and c (N-1,). Terminal
    parameters have shapes Q_f (n,) or (n, n), q_f (n,) and c_f ().

    Parameters
    ----------
    stage_cost : CostFunction
        Stage cost, either already stacked over the N - 1 stages or a single-knot cost that is
        repeated over the horizon.
    terminal_cost : CostFunction | None, optional
        Terminal cost. Required when stage_cost is already stacked, otherwise derived from
        stage_cost.
    N : int | None, optional
        Horizon length in knot points. Required when stage_cost is not stacked.
    regulates_to_goal : bool, optional
        Whether every linear state term encodes one constant goal state as q = -Q xf, so that
        `with_goal` can retarget the objective. Only LQRObjective sets this. Defaults to False.
    """

    stage_cost: CostFunction
    terminal_cost: CostFunction
    N: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    regulates_to_goal: bool = eqx.field(static=True)

    def __init__(
        self,
        stage_cost: CostFunction,
        terminal_cost: CostFunction | None = None,
        N: int | None = None,
        *,
        regulates_to_goal: bool = False,
    ) -> None:
        if stage_cost.is_stacked:
            n_knots = int(_quadratic(stage_cost).Q.shape[0]) + 1
            if N is not None and n_knots != N:
                msg = f"Provided N ({N}) does not match stacked stage cost length ({n_knots})"
                raise ValueError(msg)
            if terminal_cost is None:
                msg = "terminal_cost must be provided when stage_cost is already stacked."
                raise ValueError(msg)
            N_val = n_knots
            st_cost = stage_cost
            term_cost = terminal_cost
        else:
            if N is None:
                msg = "Horizon length N must be specified when stage_cost is not stacked."
                raise ValueError(msg)
            if N < _MIN_HORIZON:
                msg = f"Horizon N must be at least 2, got {N}"
                raise ValueError(msg)
            N_val = int(N)
            st_cost = stage_cost.stacked(N_val)
            term_cost = stage_cost.as_terminal() if terminal_cost is None else terminal_cost

        # Homogenize the two costs when one is diagonal and the other dense
        if (
            isinstance(st_cost, QuadraticCostFunction)
            and isinstance(term_cost, QuadraticCostFunction)
            and type(st_cost) is not type(term_cost)
        ):
            st_cost = st_cost.to_quadratic()
            term_cost = term_cost.to_quadratic()

        self.stage_cost = st_cost
        self.terminal_cost = term_cost
        self.N = N_val
        self.n = int(stage_cost.n)
        self.m = int(stage_cost.m)
        self.regulates_to_goal = bool(regulates_to_goal)

    @property
    def Q(self) -> jax.Array:  # noqa: N802
        """Stacked state weights of shape (N-1, n) if diagonal or (N-1, n, n) if dense."""
        return _quadratic(self.stage_cost).Q

    @property
    def R(self) -> jax.Array:  # noqa: N802
        """Stacked control weights of shape (N-1, m) if diagonal or (N-1, m, m) if dense."""
        return _quadratic(self.stage_cost).R

    @property
    def H(self) -> jax.Array | None:  # noqa: N802
        """Stacked cross-coupling weights of shape (N-1, m, n), or None if the cost is diagonal."""
        return _quadratic(self.stage_cost).H

    @property
    def q(self) -> jax.Array:
        """Stacked linear state terms of shape (N-1, n)."""
        return _quadratic(self.stage_cost).q

    @property
    def r(self) -> jax.Array:
        """Stacked linear control terms of shape (N-1, m)."""
        return _quadratic(self.stage_cost).r

    @property
    def c(self) -> jax.Array:
        """Stacked constant terms of shape (N-1,)."""
        return _quadratic(self.stage_cost).c

    @property
    def Q_f(self) -> jax.Array:  # noqa: N802
        """Terminal state weights of shape (n,) if diagonal or (n, n) if dense."""
        return _quadratic(self.terminal_cost).Q

    @property
    def q_f(self) -> jax.Array:
        """Terminal linear state term of shape (n,)."""
        return _quadratic(self.terminal_cost).q

    @property
    def c_f(self) -> jax.Array:
        """Terminal constant term of shape ()."""
        return _quadratic(self.terminal_cost).c

    @property
    def is_diag(self) -> bool:
        """Whether both cost Hessians are strictly diagonal."""
        return self.stage_cost.is_diag and self.terminal_cost.is_diag

    @property
    def is_blockdiag(self) -> bool:
        """Whether the stage cost Hessian is block diagonal."""
        return self.stage_cost.is_blockdiag

    @property
    def is_quadratic(self) -> bool:
        """Whether both stage and terminal costs are quadratic."""
        return isinstance(self.stage_cost, QuadraticCostFunction) and isinstance(
            self.terminal_cost, QuadraticCostFunction
        )

    def cost(self, traj: Trajectory) -> jax.Array:
        """Total cost sum_k l_k over the trajectory, as a scalar, in one batched pass."""
        stage_c = self.stage_cost.stage_costs(traj.X[:-1], traj.U, traj.t[:-1])
        return jnp.sum(stage_c) + self.terminal_cost.evaluate(traj.X[-1], None, traj.t[-1])

    def cost_expansion(self, traj: Trajectory, model: "AbstractModel | None" = None) -> "Expansion":
        """Stacked first- and second-order cost expansion in error coordinates along traj."""
        from trajopt.expansions import _cost_expansion  # noqa: PLC0415 -- avoid an import cycle

        return _cost_expansion(self, traj, model)

    def invert(self) -> "Objective":
        """Objective holding the inverted stage and terminal cost parameters."""
        return Objective(
            stage_cost=self.stage_cost.invert(),
            terminal_cost=self.terminal_cost.invert(),
            N=self.N,
        )

    def update_reference(self, trajectory: Trajectory, start: int = 0) -> "Objective":
        """Objective tracking trajectory from knot point start, keeping the current weights."""
        return update_reference(self, trajectory, start=start)

    def with_goal(self, xf: jax.Array) -> "Objective":
        """Objective retargeted to goal state xf of shape (n,), keeping Q, R, H, r and c.

        Rewrites only the linear state terms as q = -Q xf, the `set_LQR_goal!` of
        TrajectoryOptimization.jl. The constant term c is left at its build-time value, so a
        moved goal shifts the reported cost by a constant without moving the minimizer. Safe
        under trace: xf flows into array leaves only, so a goal that changes between MPC steps
        does not recompile.

        Raises
        ------
        TypeError
            If the objective does not regulate to a single constant goal state, in which case
            its linear terms are a reference of their own and overwriting them would discard it.
        """
        if not self.regulates_to_goal:
            msg = (
                f"{type(self.stage_cost).__name__} objective does not regulate to a goal state, so it cannot be "
                f"retargeted by xf; its linear terms hold a reference of their own. Use update_reference instead."
            )
            raise TypeError(msg)
        return eqx.tree_at(
            lambda o: (o.stage_cost.q, o.terminal_cost.q),
            self,
            (
                -type(_quadratic(self.stage_cost)).matvec(self.Q, xf),
                -type(_quadratic(self.terminal_cost)).matvec(self.Q_f, xf),
            ),
        )

    def __len__(self) -> int:
        """Return the horizon length N."""
        return self.N

    def __getitem__(self, idx: int) -> CostFunction:
        """Return the cost function at knot point idx, which may be negative.

        A stacked stage cost is sliced at idx; a stage cost carrying no horizon axis, such as a
        `QuatGeodesicCost` repeated over the horizon, is returned whole.
        """
        if not isinstance(idx, int):
            msg = f"Objective index must be an integer, got {type(idx).__name__}"
            raise TypeError(msg)
        k = self.N + idx if idx < 0 else idx
        if k < 0 or k >= self.N:
            msg = f"Objective index {idx} out of range for horizon of length {self.N}"
            raise IndexError(msg)
        if k == self.N - 1:
            return self.terminal_cost
        return self.stage_cost.unstacked(k)


def LQRObjective(  # noqa: N802, PLR0913, PLR0917
    Q: jax.Array,
    R: jax.Array,
    Qf: jax.Array,
    xf: jax.Array,
    N: int,
    uf: jax.Array | None = None,
) -> Objective:
    """Construct an LQR tracking objective with stacked-constant parameters over horizon N.

    Formula:
    (x_N - xf)^T Qf (x_N - xf) + sum_{k=0}^{N-2} [ (x_k - xf)^T Q (x_k - xf) + (u_k - uf)^T R (u_k - uf) ]

    Weights are held diagonally only if all three are diagonal vectors; otherwise all three are
    embedded as dense matrices.

    Parameters
    ----------
    Q : jax.Array
        State weights of shape (n,) if diagonal or (n, n) if dense.
    R : jax.Array
        Control weights of shape (m,) if diagonal or (m, m) if dense.
    Qf : jax.Array
        Terminal state weights of shape (n,) if diagonal or (n, n) if dense.
    xf : jax.Array
        Goal state of shape (n,).
    N : int
        Horizon length in knot points.
    uf : jax.Array | None, optional
        Goal control of shape (m,). Defaults to zeros.
    """
    xf_arr = jnp.asarray(xf)
    cost_cls, (Q_arr, R_arr, Qf_arr) = promote_weights(Q, R, Qf)
    m = int(R_arr.shape[-1])
    uf_arr = jnp.zeros(m, dtype=xf_arr.dtype) if uf is None else jnp.asarray(uf, dtype=xf_arr.dtype)

    stage_cost = cost_cls.tracking(
        jnp.repeat(Q_arr[None], N - 1, axis=0),
        jnp.repeat(R_arr[None], N - 1, axis=0),
        xf_arr,
        uf_arr,
    )
    terminal_cost = cost_cls.terminal_tracking(Qf_arr, xf_arr, m)
    return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N, regulates_to_goal=True)


def TrackingObjective(  # noqa: N802
    Q: jax.Array,
    R: jax.Array,
    trajectory: Trajectory,
    Qf: jax.Array | None = None,
) -> Objective:
    """Construct a time-varying tracking objective from a reference trajectory.

    Weights are held diagonally only if all three are diagonal vectors; otherwise all three are
    embedded as dense matrices.

    Parameters
    ----------
    Q : jax.Array
        State weights of shape (n,) if diagonal or (n, n) if dense.
    R : jax.Array
        Control weights of shape (m,) if diagonal or (m, m) if dense.
    trajectory : Trajectory
        Reference trajectory holding target states and controls.
    Qf : jax.Array | None, optional
        Terminal state weights of shape (n,) or (n, n). Defaults to Q.
    """
    N = trajectory.N
    cost_cls, (Q_arr, R_arr, Qf_arr) = promote_weights(Q, R, Q if Qf is None else Qf)

    stage_cost = cost_cls.tracking(
        jnp.repeat(Q_arr[None], N - 1, axis=0),
        jnp.repeat(R_arr[None], N - 1, axis=0),
        trajectory.X[:-1],
        trajectory.U,
    )
    terminal_cost = cost_cls.terminal_tracking(Qf_arr, trajectory.X[-1], trajectory.m)
    return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)


def update_reference(
    obj: Objective,
    trajectory: Trajectory,
    start: int = 0,
) -> Objective:
    """Rebuild the linear and constant terms of a tracking objective from a reference trajectory.

    Parameters
    ----------
    obj : Objective
        Existing tracking objective, whose weights Q, R and Q_f are carried over unchanged.
    trajectory : Trajectory
        New reference trajectory, which must hold at least start + N knot points.
    start : int, optional
        Index of the knot point in trajectory that the horizon starts at. Defaults to 0.
    """
    N = obj.N
    if start < 0:
        msg = f"start index must be non-negative, got {start}"
        raise ValueError(msg)
    if start + N > trajectory.N:
        msg = f"Reference trajectory length ({trajectory.N}) is insufficient for horizon {N} from start {start}"
        raise ValueError(msg)
    if not isinstance(obj.stage_cost, QuadraticCostFunction):
        msg = f"Cannot update the reference of a non-quadratic stage cost {type(obj.stage_cost).__name__}"
        raise TypeError(msg)

    cost_cls = type(obj.stage_cost)
    X_ref = trajectory.X[start : start + N]
    U_ref = trajectory.U[start : start + N - 1]

    stage_cost = cost_cls.tracking(obj.Q, obj.R, X_ref[:-1], U_ref)
    terminal_cost = cost_cls.terminal_tracking(obj.Q_f, X_ref[-1], obj.m)
    return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)
