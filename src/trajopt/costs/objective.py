"""Objective formulation holding stacked stage-cost and terminal-cost parameters."""

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.costs.base import CostFunction
from trajopt.costs.quadratic import DiagonalCost, QuadraticCost
from trajopt.trajectory import Trajectory

_MIN_HORIZON = 2
_EXPECTED_NDIM_1D = 1
_EXPECTED_NDIM_2D = 2
_EXPECTED_NDIM_3D = 3


def _stack_diagonal(
    cost: DiagonalCost,
    N: int,
    terminal_cost: CostFunction | None,
) -> tuple[DiagonalCost, CostFunction]:
    Q_stacked = jnp.repeat(cost.Q[None, :], N - 1, axis=0)
    R_stacked = jnp.repeat(cost.R[None, :], N - 1, axis=0)
    q_stacked = jnp.repeat(cost.q[None, :], N - 1, axis=0)
    r_stacked = jnp.repeat(cost.r[None, :], N - 1, axis=0)
    c_stacked = jnp.repeat(jnp.atleast_1d(cost.c), N - 1, axis=0)
    st = DiagonalCost(
        Q=Q_stacked,
        R=R_stacked,
        q=q_stacked,
        r=r_stacked,
        c=c_stacked,
        terminal=False,
    )
    term = (
        DiagonalCost(
            Q=cost.Q,
            q=cost.q,
            c=cost.c,
            terminal=True,
            m=cost.m,
        )
        if terminal_cost is None
        else terminal_cost
    )
    return st, term


def _stack_quadratic(
    cost: QuadraticCost,
    N: int,
    terminal_cost: CostFunction | None,
) -> tuple[QuadraticCost, CostFunction]:
    Q_stacked = jnp.repeat(cost.Q[None, :, :], N - 1, axis=0)
    R_stacked = jnp.repeat(cost.R[None, :, :], N - 1, axis=0)
    H_stacked = jnp.repeat(cost.H[None, :, :], N - 1, axis=0)
    q_stacked = jnp.repeat(cost.q[None, :], N - 1, axis=0)
    r_stacked = jnp.repeat(cost.r[None, :], N - 1, axis=0)
    c_stacked = jnp.repeat(jnp.atleast_1d(cost.c), N - 1, axis=0)
    st = QuadraticCost(
        Q=Q_stacked,
        R=R_stacked,
        H=H_stacked,
        q=q_stacked,
        r=r_stacked,
        c=c_stacked,
        terminal=False,
    )
    term = (
        QuadraticCost(
            Q=cost.Q,
            q=cost.q,
            c=cost.c,
            terminal=True,
            m=cost.m,
        )
        if terminal_cost is None
        else terminal_cost
    )
    return st, term


class Objective(eqx.Module):
    """Stacked objective holding stage-cost and terminal-cost parameters.

    Parameters
    ----------
    stage_cost : CostFunction
        Stage cost with parameters stacked over the horizon (N - 1 stages) or single-knot cost.
    terminal_cost : CostFunction | None, optional
        Terminal cost parameter object. If None, derived from stage_cost.
    N : int | None, optional
        Horizon length (number of knot points). Required if stage_cost is unstacked.
    """

    stage_cost: CostFunction
    terminal_cost: CostFunction
    N: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)

    def __init__(
        self,
        stage_cost: CostFunction,
        terminal_cost: CostFunction | None = None,
        N: int | None = None,
    ) -> None:
        n = int(stage_cost.n)
        m = int(stage_cost.m)

        is_stacked = (isinstance(stage_cost, DiagonalCost) and stage_cost.Q.ndim == _EXPECTED_NDIM_2D) or (
            isinstance(stage_cost, QuadraticCost) and stage_cost.Q.ndim == _EXPECTED_NDIM_3D
        )

        if is_stacked and isinstance(stage_cost, (DiagonalCost, QuadraticCost)):
            n_knots = int(stage_cost.Q.shape[0]) + 1
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

            if isinstance(stage_cost, DiagonalCost):
                st_cost, term_cost = _stack_diagonal(stage_cost, N_val, terminal_cost)
            elif isinstance(stage_cost, QuadraticCost):
                st_cost, term_cost = _stack_quadratic(stage_cost, N_val, terminal_cost)
            else:
                st_cost = stage_cost
                term_cost = terminal_cost if terminal_cost is not None else stage_cost

        # Homogenize types if mixing DiagonalCost and QuadraticCost
        if isinstance(st_cost, QuadraticCost) and isinstance(term_cost, DiagonalCost):
            term_cost = term_cost.to_quadratic()
        elif isinstance(st_cost, DiagonalCost) and isinstance(term_cost, QuadraticCost):
            st_cost = st_cost.to_quadratic()

        self.stage_cost = st_cost
        self.terminal_cost = term_cost
        self.N = N_val
        self.n = n
        self.m = m

    @property
    def Q(self) -> jax.Array:  # noqa: N802
        """Stacked state weight parameters of shape (N-1, n) or (N-1, n, n)."""
        if isinstance(self.stage_cost, (DiagonalCost, QuadraticCost)):
            return self.stage_cost.Q
        msg = f"{type(self.stage_cost).__name__} does not expose quadratic weight matrix Q"
        raise AttributeError(msg)

    @property
    def R(self) -> jax.Array:  # noqa: N802
        """Stacked control weight parameters of shape (N-1, m) or (N-1, m, m)."""
        if isinstance(self.stage_cost, (DiagonalCost, QuadraticCost)):
            return self.stage_cost.R
        msg = f"{type(self.stage_cost).__name__} does not expose quadratic weight matrix R"
        raise AttributeError(msg)

    @property
    def H(self) -> jax.Array | None:  # noqa: N802
        """Stacked cross-coupling weight parameters of shape (N-1, m, n) if dense."""
        if isinstance(self.stage_cost, QuadraticCost):
            return self.stage_cost.H
        return None

    @property
    def q(self) -> jax.Array:
        """Stacked linear state terms of shape (N-1, n)."""
        if isinstance(self.stage_cost, (DiagonalCost, QuadraticCost)):
            return self.stage_cost.q
        msg = f"{type(self.stage_cost).__name__} does not expose linear vector q"
        raise AttributeError(msg)

    @property
    def r(self) -> jax.Array:
        """Stacked linear control terms of shape (N-1, m)."""
        if isinstance(self.stage_cost, (DiagonalCost, QuadraticCost)):
            return self.stage_cost.r
        msg = f"{type(self.stage_cost).__name__} does not expose linear vector r"
        raise AttributeError(msg)

    @property
    def c(self) -> jax.Array:
        """Stacked constant terms of shape (N-1,)."""
        if isinstance(self.stage_cost, (DiagonalCost, QuadraticCost)):
            return self.stage_cost.c
        msg = f"{type(self.stage_cost).__name__} does not expose constant c"
        raise AttributeError(msg)

    @property
    def Qf(self) -> jax.Array:  # noqa: N802
        """Terminal state weight parameters of shape (n,) or (n, n)."""
        if isinstance(self.terminal_cost, (DiagonalCost, QuadraticCost)):
            return self.terminal_cost.Q
        msg = f"{type(self.terminal_cost).__name__} does not expose quadratic weight matrix Qf"
        raise AttributeError(msg)

    @property
    def Q_f(self) -> jax.Array:  # noqa: N802
        """Alias for Qf."""
        return self.Qf

    @property
    def qf(self) -> jax.Array:
        """Terminal linear state term of shape (n,)."""
        if isinstance(self.terminal_cost, (DiagonalCost, QuadraticCost)):
            return self.terminal_cost.q
        msg = f"{type(self.terminal_cost).__name__} does not expose linear vector qf"
        raise AttributeError(msg)

    @property
    def q_f(self) -> jax.Array:
        """Alias for qf."""
        return self.qf

    @property
    def cf(self) -> jax.Array:
        """Terminal constant term scalar ()."""
        if isinstance(self.terminal_cost, (DiagonalCost, QuadraticCost)):
            return self.terminal_cost.c
        msg = f"{type(self.terminal_cost).__name__} does not expose constant cf"
        raise AttributeError(msg)

    @property
    def c_f(self) -> jax.Array:
        """Alias for cf."""
        return self.cf

    @property
    def is_diag(self) -> bool:
        """Whether the objective cost functions are diagonal."""
        return isinstance(self.stage_cost, DiagonalCost) and isinstance(self.terminal_cost, DiagonalCost)

    @property
    def is_diagonal(self) -> bool:
        """Alias for is_diag."""
        return self.is_diag

    @property
    def is_blockdiag(self) -> bool:
        """Whether the objective cost Hessians are block diagonal."""
        if self.is_diag:
            return True
        if isinstance(self.stage_cost, QuadraticCost):
            return self.stage_cost.is_blockdiag
        return False

    def cost(self, traj: Trajectory) -> jax.Array:
        """Evaluate total objective cost over a trajectory in a single batched pass.

        Parameters
        ----------
        traj : Trajectory
            Trajectory holding stacked states X, controls U, and timestamps t.

        Returns
        -------
        jax.Array
            Total scalar cost value.
        """
        X_stage = traj.X[:-1]
        U_stage = traj.U
        t_stage = traj.t[:-1]

        if isinstance(self.stage_cost, DiagonalCost):
            stage_c = (
                0.5 * jnp.sum(self.Q * (X_stage**2), axis=-1)
                + 0.5 * jnp.sum(self.R * (U_stage**2), axis=-1)
                + jnp.sum(self.q * X_stage, axis=-1)
                + jnp.sum(self.r * U_stage, axis=-1)
                + self.c
            )
        elif isinstance(self.stage_cost, QuadraticCost):
            H_mat = self.H if self.H is not None else jnp.zeros((self.N - 1, self.m, self.n), dtype=self.Q.dtype)

            def eval_quad_stage(  # noqa: PLR0913, PLR0917
                x: jax.Array,
                u: jax.Array,
                q_mat: jax.Array,
                r_mat: jax.Array,
                h_mat: jax.Array,
                q_vec: jax.Array,
                r_vec: jax.Array,
                c_val: jax.Array,
            ) -> jax.Array:
                val = (
                    0.5 * jnp.dot(x, q_mat @ x)
                    + 0.5 * jnp.dot(u, r_mat @ u)
                    + jnp.dot(q_vec, x)
                    + jnp.dot(r_vec, u)
                    + c_val
                )
                if not self.is_blockdiag:
                    val = val + jnp.dot(u, h_mat @ x)
                return val

            stage_c = jax.vmap(eval_quad_stage)(
                X_stage,
                U_stage,
                self.Q,
                self.R,
                H_mat,
                self.q,
                self.r,
                self.c,
            )
        else:
            stage_c = jax.vmap(self.stage_cost.evaluate)(X_stage, U_stage, t_stage)

        X_term = traj.X[-1]
        t_term = traj.t[-1]
        term_c = self.terminal_cost.evaluate(X_term, None, t_term)

        return jnp.sum(stage_c) + term_c

    def evaluate(self, traj: Trajectory) -> jax.Array:
        """Evaluate total cost over trajectory (alias for cost).

        Parameters
        ----------
        traj : Trajectory
            Trajectory holding states and controls.

        Returns
        -------
        jax.Array
            Scalar cost value.
        """
        return self.cost(traj)

    def invert(self) -> "Objective":
        """Return a new Objective with inverted Hessian parameters.

        Returns
        -------
        Objective
            New Objective with inverted stage and terminal cost parameters.
        """
        new_stage = self.stage_cost.invert()
        new_term = self.terminal_cost.invert()
        return Objective(stage_cost=new_stage, terminal_cost=new_term, N=self.N)

    def update_reference(self, trajectory: Trajectory, start: int = 0) -> "Objective":
        """Update reference tracking parameters from a trajectory slice.

        Parameters
        ----------
        trajectory : Trajectory
            New reference trajectory.
        start : int, optional
            Starting index in trajectory. Defaults to 0.

        Returns
        -------
        Objective
            New Objective with updated tracking reference.
        """
        return update_reference(self, trajectory, start=start)

    def __len__(self) -> int:
        """Return the horizon length N."""
        return self.N

    def __getitem__(self, idx: int) -> CostFunction:
        """Return the unstacked cost function for knot point idx.

        Parameters
        ----------
        idx : int
            Knot point index in [0, N - 1] or negative index.

        Returns
        -------
        CostFunction
            Unstacked cost function at knot point idx.
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

        if isinstance(self.stage_cost, DiagonalCost):
            return DiagonalCost(
                Q=self.Q[k],
                R=self.R[k],
                q=self.q[k],
                r=self.r[k],
                c=self.c[k],
                terminal=False,
            )
        if isinstance(self.stage_cost, QuadraticCost):
            H_k = self.H[k] if self.H is not None else None
            return QuadraticCost(
                Q=self.Q[k],
                R=self.R[k],
                H=H_k,
                q=self.q[k],
                r=self.r[k],
                c=self.c[k],
                terminal=False,
            )
        return self.stage_cost


def cost(obj: Objective, traj: Trajectory) -> jax.Array:
    """Evaluate total objective cost over a trajectory in a single batched pass.

    Parameters
    ----------
    obj : Objective
        Objective instance.
    traj : Trajectory
        Trajectory to evaluate.

    Returns
    -------
    jax.Array
        Total scalar cost.
    """
    return obj.cost(traj)


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

    Parameters
    ----------
    Q : jax.Array
        State weighting vector (n,) or matrix (n, n).
    R : jax.Array
        Control weighting vector (m,) or matrix (m, m).
    Qf : jax.Array
        Terminal state weighting vector (n,) or matrix (n, n).
    xf : jax.Array
        Goal state of shape (n,).
    N : int
        Horizon length (number of knot points).
    uf : jax.Array | None, optional
        Goal control of shape (m,). Defaults to zeros.

    Returns
    -------
    Objective
        Constructed stacked Objective.
    """
    xf_arr = jnp.asarray(xf)
    n = int(xf_arr.shape[0])
    Q_arr = jnp.asarray(Q)
    R_arr = jnp.asarray(R)
    Qf_arr = jnp.asarray(Qf)

    if Q_arr.ndim == _EXPECTED_NDIM_2D and Q_arr.shape[0] == Q_arr.shape[1] and R_arr.ndim <= _EXPECTED_NDIM_1D:
        Q_arr = jnp.diag(Q_arr)
    if R_arr.ndim == _EXPECTED_NDIM_2D and R_arr.shape[0] == R_arr.shape[1] and Q_arr.ndim <= _EXPECTED_NDIM_1D:
        R_arr = jnp.diag(R_arr)
    if Qf_arr.ndim == _EXPECTED_NDIM_2D and Qf_arr.shape[0] == Qf_arr.shape[1] and Q_arr.ndim <= _EXPECTED_NDIM_1D:
        Qf_arr = jnp.diag(Qf_arr)

    m = int(R_arr.shape[-1])
    uf_arr = jnp.zeros(m, dtype=xf_arr.dtype) if uf is None else jnp.asarray(uf, dtype=xf_arr.dtype)

    if Q_arr.ndim == _EXPECTED_NDIM_1D and R_arr.ndim == _EXPECTED_NDIM_1D and Qf_arr.ndim == _EXPECTED_NDIM_1D:
        q_stage = -Q_arr * xf_arr
        r_stage = -R_arr * uf_arr
        c_stage = 0.5 * jnp.sum(Q_arr * (xf_arr**2)) + 0.5 * jnp.sum(R_arr * (uf_arr**2))

        Q_stacked = jnp.repeat(Q_arr[None, :], N - 1, axis=0)
        R_stacked = jnp.repeat(R_arr[None, :], N - 1, axis=0)
        q_stacked = jnp.repeat(q_stage[None, :], N - 1, axis=0)
        r_stacked = jnp.repeat(r_stage[None, :], N - 1, axis=0)
        c_stacked = jnp.repeat(jnp.asarray(c_stage)[None], N - 1, axis=0)

        qf = -Qf_arr * xf_arr
        cf = 0.5 * jnp.sum(Qf_arr * (xf_arr**2))

        stage_cost = DiagonalCost(
            Q=Q_stacked,
            R=R_stacked,
            q=q_stacked,
            r=r_stacked,
            c=c_stacked,
            terminal=False,
        )
        terminal_cost = DiagonalCost(Q=Qf_arr, q=qf, c=cf, terminal=True, m=m)
        return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)

    Q_mat = jnp.diag(Q_arr) if Q_arr.ndim == _EXPECTED_NDIM_1D else Q_arr
    R_mat = jnp.diag(R_arr) if R_arr.ndim == _EXPECTED_NDIM_1D else R_arr
    Qf_mat = jnp.diag(Qf_arr) if Qf_arr.ndim == _EXPECTED_NDIM_1D else Qf_arr

    q_stage = -Q_mat @ xf_arr
    r_stage = -R_mat @ uf_arr
    c_stage = 0.5 * jnp.dot(xf_arr, Q_mat @ xf_arr) + 0.5 * jnp.dot(uf_arr, R_mat @ uf_arr)

    Q_stacked = jnp.repeat(Q_mat[None, :, :], N - 1, axis=0)
    R_stacked = jnp.repeat(R_mat[None, :, :], N - 1, axis=0)
    H_stacked = jnp.zeros((N - 1, m, n), dtype=Q_mat.dtype)
    q_stacked = jnp.repeat(q_stage[None, :], N - 1, axis=0)
    r_stacked = jnp.repeat(r_stage[None, :], N - 1, axis=0)
    c_stacked = jnp.repeat(jnp.asarray(c_stage)[None], N - 1, axis=0)

    qf = -Qf_mat @ xf_arr
    cf = 0.5 * jnp.dot(xf_arr, Qf_mat @ xf_arr)

    stage_cost = QuadraticCost(
        Q=Q_stacked,
        R=R_stacked,
        H=H_stacked,
        q=q_stacked,
        r=r_stacked,
        c=c_stacked,
        terminal=False,
    )
    terminal_cost = QuadraticCost(Q=Qf_mat, q=qf, c=cf, terminal=True, m=m)
    return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)


def TrackingObjective(  # noqa: N802
    Q: jax.Array,
    R: jax.Array,
    trajectory: Trajectory,
    Qf: jax.Array | None = None,
) -> Objective:
    """Construct a time-varying tracking objective from a reference trajectory.

    Parameters
    ----------
    Q : jax.Array
        State weighting vector (n,) or matrix (n, n).
    R : jax.Array
        Control weighting vector (m,) or matrix (m, m).
    trajectory : Trajectory
        Reference trajectory holding target states and controls.
    Qf : jax.Array | None, optional
        Terminal state weighting. Defaults to Q.

    Returns
    -------
    Objective
        Constructed tracking Objective with stacked time-varying linear terms.
    """
    N = trajectory.N
    m = trajectory.m
    X = trajectory.X
    U = trajectory.U

    Q_arr = jnp.asarray(Q)
    R_arr = jnp.asarray(R)
    Qf_arr = jnp.asarray(Q if Qf is None else Qf)

    if Q_arr.ndim == _EXPECTED_NDIM_2D and Q_arr.shape[0] == Q_arr.shape[1] and R_arr.ndim <= _EXPECTED_NDIM_1D:
        Q_arr = jnp.diag(Q_arr)
    if R_arr.ndim == _EXPECTED_NDIM_2D and R_arr.shape[0] == R_arr.shape[1] and Q_arr.ndim <= _EXPECTED_NDIM_1D:
        R_arr = jnp.diag(R_arr)
    if Qf_arr.ndim == _EXPECTED_NDIM_2D and Qf_arr.shape[0] == Qf_arr.shape[1] and Q_arr.ndim <= _EXPECTED_NDIM_1D:
        Qf_arr = jnp.diag(Qf_arr)

    if Q_arr.ndim == _EXPECTED_NDIM_1D and R_arr.ndim == _EXPECTED_NDIM_1D and Qf_arr.ndim == _EXPECTED_NDIM_1D:
        Q_stacked = jnp.repeat(Q_arr[None, :], N - 1, axis=0)
        R_stacked = jnp.repeat(R_arr[None, :], N - 1, axis=0)
        X_stage = X[:-1]
        U_stage = U
        q_stage = -Q_stacked * X_stage
        r_stage = -R_stacked * U_stage
        c_stage = 0.5 * jnp.sum(Q_stacked * (X_stage**2), axis=-1) + 0.5 * jnp.sum(R_stacked * (U_stage**2), axis=-1)

        X_term = X[-1]
        qf = -Qf_arr * X_term
        cf = 0.5 * jnp.sum(Qf_arr * (X_term**2))

        stage_cost = DiagonalCost(
            Q=Q_stacked,
            R=R_stacked,
            q=q_stage,
            r=r_stage,
            c=c_stage,
            terminal=False,
        )
        terminal_cost = DiagonalCost(Q=Qf_arr, q=qf, c=cf, terminal=True, m=m)
        return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)

    Q_mat = jnp.diag(Q_arr) if Q_arr.ndim == _EXPECTED_NDIM_1D else Q_arr
    R_mat = jnp.diag(R_arr) if R_arr.ndim == _EXPECTED_NDIM_1D else R_arr
    Qf_mat = jnp.diag(Qf_arr) if Qf_arr.ndim == _EXPECTED_NDIM_1D else Qf_arr

    Q_stacked = jnp.repeat(Q_mat[None, :, :], N - 1, axis=0)
    R_stacked = jnp.repeat(R_mat[None, :, :], N - 1, axis=0)
    H_stacked = jnp.zeros((N - 1, m, trajectory.n), dtype=Q_mat.dtype)

    X_stage = X[:-1]
    U_stage = U
    q_stage = -jnp.einsum("kij,kj->ki", Q_stacked, X_stage)
    r_stage = -jnp.einsum("kij,kj->ki", R_stacked, U_stage)
    c_stage = 0.5 * jnp.einsum("ki,kij,kj->k", X_stage, Q_stacked, X_stage) + 0.5 * jnp.einsum(
        "ki,kij,kj->k", U_stage, R_stacked, U_stage
    )

    X_term = X[-1]
    qf = -Qf_mat @ X_term
    cf = 0.5 * jnp.dot(X_term, Qf_mat @ X_term)

    stage_cost = QuadraticCost(
        Q=Q_stacked,
        R=R_stacked,
        H=H_stacked,
        q=q_stage,
        r=r_stage,
        c=c_stage,
        terminal=False,
    )
    terminal_cost = QuadraticCost(Q=Qf_mat, q=qf, c=cf, terminal=True, m=m)
    return Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)


def update_reference(
    obj: Objective,
    trajectory: Trajectory,
    start: int = 0,
) -> Objective:
    """Update tracking reference terms from a trajectory without changing Q/R weights.

    Parameters
    ----------
    obj : Objective
        Existing tracking objective.
    trajectory : Trajectory
        New reference trajectory.
    start : int, optional
        Starting index in trajectory. Defaults to 0.

    Returns
    -------
    Objective
        New Objective instance with updated linear and constant terms.
    """
    N = obj.N
    if start < 0:
        msg = f"start index must be non-negative, got {start}"
        raise ValueError(msg)
    if start + N > trajectory.N:
        msg = f"Reference trajectory length ({trajectory.N}) is insufficient for horizon {N} from start {start}"
        raise ValueError(msg)

    X_ref = trajectory.X[start : start + N]
    U_ref = trajectory.U[start : start + N - 1]

    if isinstance(obj.stage_cost, DiagonalCost):
        X_stage = X_ref[:-1]
        U_stage = U_ref
        q_stage = -obj.Q * X_stage
        r_stage = -obj.R * U_stage
        c_stage = 0.5 * jnp.sum(obj.Q * (X_stage**2), axis=-1) + 0.5 * jnp.sum(obj.R * (U_stage**2), axis=-1)

        X_term = X_ref[-1]
        qf = -obj.Q_f * X_term
        cf = 0.5 * jnp.sum(obj.Q_f * (X_term**2))

        new_stage = DiagonalCost(
            Q=obj.Q,
            R=obj.R,
            q=q_stage,
            r=r_stage,
            c=c_stage,
            terminal=False,
        )
        new_term = DiagonalCost(Q=obj.Q_f, q=qf, c=cf, terminal=True, m=obj.m)
        return Objective(stage_cost=new_stage, terminal_cost=new_term, N=N)

    if isinstance(obj.stage_cost, QuadraticCost):
        X_stage = X_ref[:-1]
        U_stage = U_ref
        q_stage = -jnp.einsum("kij,kj->ki", obj.Q, X_stage)
        r_stage = -jnp.einsum("kij,kj->ki", obj.R, U_stage)
        c_stage = 0.5 * jnp.einsum("ki,kij,kj->k", X_stage, obj.Q, X_stage) + 0.5 * jnp.einsum(
            "ki,kij,kj->k", U_stage, obj.R, U_stage
        )

        X_term = X_ref[-1]
        qf = -obj.Q_f @ X_term
        cf = 0.5 * jnp.dot(X_term, obj.Q_f @ X_term)

        new_stage = QuadraticCost(
            Q=obj.Q,
            R=obj.R,
            H=obj.H,
            q=q_stage,
            r=r_stage,
            c=c_stage,
            terminal=False,
        )
        new_term = QuadraticCost(Q=obj.Q_f, q=qf, c=cf, terminal=True, m=obj.m)
        return Objective(stage_cost=new_stage, terminal_cost=new_term, N=N)

    msg = f"Cannot update tracking reference on non-quadratic objective with stage cost {type(obj.stage_cost).__name__}"
    raise TypeError(msg)
