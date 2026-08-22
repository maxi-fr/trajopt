import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import AbstractModel, ContinuousDynamics, DiscreteDynamics
from trajopt.dynamics.rollout import _discretize, _step_durations_and_times
from trajopt.trajectory import Trajectory

_EXPECTED_NDIM_2D = 2


def control_rate_cost(
    R_delta: jax.Array,
    n: int,
    m: int,
) -> QuadraticCost:
    """Build a quadratic stage cost penalizing control rate delta_u = u_k - u_{k-1}.

    Parameters
    ----------
    R_delta : jax.Array
        Weight matrix of shape (m, m) or diagonal weight vector of shape (m,).
    n : int
        Original state dimension.
    m : int
        Control dimension.

    Returns
    -------
    QuadraticCost
        Stage cost on augmented state x_aug = [x; u_prev] of shape (n + m,) and control u of shape (m,).
    """
    R_arr = jnp.asarray(R_delta)
    R_mat = jnp.diag(R_arr) if R_arr.ndim == 1 else R_arr
    dtype = R_mat.dtype
    n_aug = n + m

    z_nn = jnp.zeros((n, n), dtype=dtype)
    z_nm = jnp.zeros((n, m), dtype=dtype)
    z_mn = jnp.zeros((m, n), dtype=dtype)

    top_row = jnp.hstack([z_nn, z_nm])
    bot_row = jnp.hstack([z_mn, R_mat])
    Q_aug = jnp.vstack([top_row, bot_row])

    R_aug = R_mat
    H_aug = jnp.hstack([z_mn, -R_mat])

    return QuadraticCost(
        Q=Q_aug,
        R=R_aug,
        H=H_aug,
        q=jnp.zeros(n_aug, dtype=dtype),
        r=jnp.zeros(m, dtype=dtype),
        c=0.0,
        terminal=False,
    )


class ControlRateModel(DiscreteDynamics):
    """Discrete dynamics model augmented with previous control u_{k-1} as extra state variables.

    Parameters
    ----------
    model : DiscreteDynamics
        Underlying discrete dynamics model.
    """

    model: DiscreteDynamics

    def __init__(self, model: DiscreteDynamics) -> None:
        super().__init__(n=model.n + model.m, m=model.m, ne=model.ne + model.m)
        self.model = model

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate discrete dynamics for state x of shape (n + m,) and control u of shape (m,).

        Parameters
        ----------
        x : jax.Array
            Augmented state vector [x_orig, u_prev] of shape (n + m,).
        u : jax.Array
            Control input vector of shape (m,).
        t : float | jax.Array
            Timestamp.
        dt : float | jax.Array
            Step duration.

        Returns
        -------
        jax.Array
            Next augmented state vector [x_next_orig, u] of shape (n + m,).
        """
        x_orig = x[: self.model.n]
        x_next_orig = self.model.discrete_dynamics(x_orig, u, t, dt)
        return jnp.concatenate([x_next_orig, u])

    def state_diff(self, x: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute augmented error state dx = x (-) x0 of shape (ne + m,).

        Parameters
        ----------
        x : jax.Array
            Augmented state vector of shape (n + m,).
        x0 : jax.Array
            Augmented reference state vector of shape (n + m,).

        Returns
        -------
        jax.Array
            Augmented error state vector of shape (ne + m,).
        """
        dx_orig = self.model.state_diff(x[: self.model.n], x0[: self.model.n])
        du = x[self.model.n :] - x0[self.model.n :]
        return jnp.concatenate([dx_orig, du])

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate error-state Jacobian G = dx / d(delta x) of shape (n + m, ne + m).

        Parameters
        ----------
        x : jax.Array
            Augmented state vector of shape (n + m,).

        Returns
        -------
        jax.Array
            Augmented error-state Jacobian blockdiag(G_orig, I_m) of shape (n + m, ne + m).
        """
        G_orig = self.model.errstate_jacobian(x[: self.model.n])
        eye_m = jnp.eye(self.model.m, dtype=x.dtype)
        z_nm = jnp.zeros((self.model.n, self.model.m), dtype=x.dtype)
        z_mne = jnp.zeros((self.model.m, self.model.ne), dtype=x.dtype)
        top = jnp.hstack([G_orig, z_nm])
        bot = jnp.hstack([z_mne, eye_m])
        return jnp.vstack([top, bot])


def with_control_rate_penalty(
    model: AbstractModel,
    R_delta: jax.Array,
) -> tuple[ControlRateModel, QuadraticCost]:
    """Augment a model's state with previous control u_{k-1} to preserve Markovian stage separability.

    Parameters
    ----------
    model : AbstractModel
        Continuous or discrete dynamics model. Continuous models are discretized with RK4.
    R_delta : jax.Array
        Control rate penalty weight matrix (m, m) or diagonal vector (m,).

    Returns
    -------
    tuple[ControlRateModel, QuadraticCost]
        Augmented discrete model and the corresponding stage cost function.
    """
    discrete_model = _discretize(model) if isinstance(model, ContinuousDynamics) else model
    if not isinstance(discrete_model, DiscreteDynamics):
        msg = f"Cannot discretize model {type(model).__name__}"
        raise TypeError(msg)

    aug_model = ControlRateModel(discrete_model)
    cost = control_rate_cost(R_delta, discrete_model.n, discrete_model.m)
    return aug_model, cost


class LinearTrajectoryModel(eqx.Module):
    """Linear time-varying trajectory model holding stacked state and control Jacobians.

    Parameters
    ----------
    A : jax.Array
        Stacked state Jacobians of shape (N - 1, ne, ne).
    B : jax.Array
        Stacked control Jacobians of shape (N - 1, ne, m).
    """

    A: jax.Array
    B: jax.Array
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    ne: int = eqx.field(static=True)
    N: int = eqx.field(static=True)

    def __init__(
        self,
        A: jax.Array,
        B: jax.Array,
        n: int,
        m: int,
        ne: int,
    ) -> None:
        self.A = jnp.asarray(A)
        self.B = jnp.asarray(B)
        self.n = n
        self.m = m
        self.ne = ne
        self.N = int(self.A.shape[0]) + 1


def linearize_about(
    model: AbstractModel,
    X_ref: Trajectory | jax.Array,
    U_ref: jax.Array | None = None,
    t: jax.Array | float | None = None,
    dt: jax.Array | float | None = None,
) -> LinearTrajectoryModel:
    """Linearize a dynamics model about a reference trajectory.

    Produces stacked discrete state Jacobians A of shape (N-1, ne, ne) and control Jacobians
    B of shape (N-1, ne, m) in error coordinates along the horizon.

    Parameters
    ----------
    model : AbstractModel
        Dynamics model to linearize. If continuous, discretized with RK4 by default.
    X_ref : Trajectory | jax.Array
        Reference trajectory holding states of shape (N, n), or Trajectory instance.
    U_ref : jax.Array | None, optional
        Reference controls of shape (N-1, m). Required when X_ref is an array.
    t : jax.Array | float | None, optional
        Timestamps of shape (N,) or initial timestamp.
    dt : jax.Array | float | None, optional
        Step durations of shape (N-1,) or scalar step duration. Defaults to 0.01.

    Returns
    -------
    LinearTrajectoryModel
        Linearized model exposing stacked Jacobians A and B.
    """
    if isinstance(X_ref, Trajectory):
        traj = X_ref
        X_arr = traj.X
        U_arr = traj.U
        t_arr = traj.t
        dt_arr = traj.dt
    else:
        if U_ref is None:
            msg = "U_ref must be provided when X_ref is an array."
            raise ValueError(msg)
        X_arr = jnp.asarray(X_ref)
        U_arr = jnp.asarray(U_ref)
        if U_arr.ndim != _EXPECTED_NDIM_2D:
            msg = f"Reference controls U_ref must have 2 dimensions (N-1, m), got shape {U_arr.shape}"
            raise ValueError(msg)
        dt_arr, t_arr = _step_durations_and_times(t, dt, n_steps=U_arr.shape[0], dtype=X_arr.dtype)

    discrete_model = _discretize(model) if isinstance(model, ContinuousDynamics) else model
    if not isinstance(discrete_model, DiscreteDynamics):
        msg = f"Cannot extract discrete dynamics from model {type(model).__name__}"
        raise TypeError(msg)

    def step_jacobians(
        xk: jax.Array,
        uk: jax.Array,
        x_next: jax.Array,
        tk: float | jax.Array,
        dtk: float | jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        Ak_raw = discrete_model.state_jacobian(xk, uk, tk, dtk)
        Bk_raw = discrete_model.control_jacobian(xk, uk, tk, dtk)
        Gk = discrete_model.errstate_jacobian(xk)
        G_next = discrete_model.errstate_jacobian(x_next)
        A_bar = G_next.T @ Ak_raw @ Gk
        B_bar = G_next.T @ Bk_raw
        return A_bar, B_bar

    A_stacked, B_stacked = jax.vmap(step_jacobians)(
        X_arr[:-1],
        U_arr,
        X_arr[1:],
        t_arr[:-1],
        dt_arr,
    )

    return LinearTrajectoryModel(
        A=A_stacked,
        B=B_stacked,
        n=discrete_model.n,
        m=discrete_model.m,
        ne=discrete_model.ne,
    )
