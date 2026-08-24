from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.constraints.constraint_list import BuiltConstraintList, BuiltKnotConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost, QuadraticCost
from trajopt.dynamics.base import AbstractModel
from trajopt.trajectory import Trajectory

_EXPECTED_NDIM_2D = 2
_EXPECTED_NDIM_3D = 3


class _KnotALContext(NamedTuple):
    evaluator: BuiltKnotConstraint
    bounds: BuiltKnotConstraint
    lam_k: jax.Array
    mu_k: jax.Array
    tk: float | jax.Array

    @property
    def p(self) -> int:
        """Total penalised row count at this knot: constraint rows then bound rows."""
        return self.evaluator.p + self.bounds.p


def _validate_expansion(exp: "Expansion") -> None:
    """Validate shape and dimension consistency for an Expansion instance."""
    ndim_specs = [
        (exp.A, _EXPECTED_NDIM_3D, "Dynamics Jacobian A must have 3 dimensions (N-1, ne, ne)"),
        (exp.B, _EXPECTED_NDIM_3D, "Dynamics Jacobian B must have 3 dimensions (N-1, ne, m)"),
        (exp.q, _EXPECTED_NDIM_2D, "State gradient q must have 2 dimensions (N, ne)"),
        (exp.r, _EXPECTED_NDIM_2D, "Control gradient r must have 2 dimensions (N-1, m)"),
        (exp.Q, _EXPECTED_NDIM_3D, "State Hessian Q must have 3 dimensions (N, ne, ne)"),
        (exp.R, _EXPECTED_NDIM_3D, "Control Hessian R must have 3 dimensions (N-1, m, m)"),
        (exp.H, _EXPECTED_NDIM_3D, "Cross Hessian H must have 3 dimensions (N-1, m, ne)"),
    ]
    for arr, expected_ndim, msg_prefix in ndim_specs:
        if arr.ndim != expected_ndim:
            msg = f"{msg_prefix}, got shape {arr.shape}"
            raise ValueError(msg)

    n_knots = exp.q.shape[0]
    ne_dim = exp.q.shape[1]
    m_dim = exp.r.shape[1]

    shape_specs = [
        (exp.A, (n_knots - 1, ne_dim, ne_dim), "A"),
        (exp.B, (n_knots - 1, ne_dim, m_dim), "B"),
        (exp.r, (n_knots - 1, m_dim), "r"),
        (exp.Q, (n_knots, ne_dim, ne_dim), "Q"),
        (exp.R, (n_knots - 1, m_dim, m_dim), "R"),
        (exp.H, (n_knots - 1, m_dim, ne_dim), "H"),
    ]
    for arr, expected_shape, name in shape_specs:
        if arr.shape != expected_shape:
            msg = f"{name} shape {arr.shape} inconsistent with expected {expected_shape}"
            raise ValueError(msg)


class Expansion(eqx.Module):
    """Stacked first- and second-order expansions of dynamics, cost, and augmented Lagrangian.

    Expansions are sized in the error dimension (ne) rather than the state dimension (n),
    with the attitude / error-state Jacobian G_k applied internally.

    Parameters
    ----------
    A : jax.Array
        Stacked discrete state Jacobians of shape (N-1, ne, ne).
    B : jax.Array
        Stacked discrete control Jacobians of shape (N-1, ne, m).
    q : jax.Array
        Stacked cost / AL state gradients of shape (N, ne).
    r : jax.Array
        Stacked cost / AL control gradients of shape (N-1, m).
    Q : jax.Array
        Stacked cost / AL state Hessians of shape (N, ne, ne).
    R : jax.Array
        Stacked cost / AL control Hessians of shape (N-1, m, m).
    H : jax.Array
        Stacked cost / AL cross-coupling Hessians of shape (N-1, m, ne).
    """

    A: jax.Array
    B: jax.Array
    q: jax.Array
    r: jax.Array
    Q: jax.Array
    R: jax.Array
    H: jax.Array

    def __init__(  # noqa: PLR0913 -- Expansion holds all 7 Taylor expansion blocks
        self,
        *,
        A: jax.Array,
        B: jax.Array,
        q: jax.Array,
        r: jax.Array,
        Q: jax.Array,
        R: jax.Array,
        H: jax.Array,
    ) -> None:
        self.A = jnp.asarray(A)
        self.B = jnp.asarray(B)
        self.q = jnp.asarray(q)
        self.r = jnp.asarray(r)
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.H = jnp.asarray(H)
        _validate_expansion(self)

    @property
    def N(self) -> int:  # noqa: N802 -- uppercase N follows standard optimal control notation for horizon length
        """Number of knot points in the expansion."""
        return int(self.q.shape[0])

    @property
    def ne(self) -> int:
        """Error-state dimension."""
        return int(self.q.shape[1])

    @property
    def m(self) -> int:
        """Control dimension."""
        return int(self.r.shape[1])

    @classmethod
    def zeros(
        cls,
        N: int,
        ne: int,
        m: int,
        dtype: jnp.dtype | type = jnp.float64,
    ) -> "Expansion":
        """Construct an Expansion with all zero blocks."""
        return cls(
            A=jnp.zeros((N - 1, ne, ne), dtype=dtype),
            B=jnp.zeros((N - 1, ne, m), dtype=dtype),
            q=jnp.zeros((N, ne), dtype=dtype),
            r=jnp.zeros((N - 1, m), dtype=dtype),
            Q=jnp.zeros((N, ne, ne), dtype=dtype),
            R=jnp.zeros((N - 1, m, m), dtype=dtype),
            H=jnp.zeros((N - 1, m, ne), dtype=dtype),
        )

    @classmethod
    def zeros_like(cls, expansion: "Expansion") -> "Expansion":
        """Construct a zero Expansion with matching dimensions and dtype."""
        return cls(
            A=jnp.zeros_like(expansion.A),
            B=jnp.zeros_like(expansion.B),
            q=jnp.zeros_like(expansion.q),
            r=jnp.zeros_like(expansion.r),
            Q=jnp.zeros_like(expansion.Q),
            R=jnp.zeros_like(expansion.R),
            H=jnp.zeros_like(expansion.H),
        )

    def __add__(self, other: "Expansion") -> "Expansion":
        """Add two Expansion instances component-wise."""
        return Expansion(
            A=self.A + other.A,
            B=self.B + other.B,
            q=self.q + other.q,
            r=self.r + other.r,
            Q=self.Q + other.Q,
            R=self.R + other.R,
            H=self.H + other.H,
        )

    def __sub__(self, other: "Expansion") -> "Expansion":
        """Subtract other Expansion from self component-wise."""
        return Expansion(
            A=self.A - other.A,
            B=self.B - other.B,
            q=self.q - other.q,
            r=self.r - other.r,
            Q=self.Q - other.Q,
            R=self.R - other.R,
            H=self.H - other.H,
        )


def _dynamics_expansion(model: AbstractModel, state: Trajectory) -> Expansion:
    """Compute the stacked first-order dynamics expansion in error coordinates.

    Parameters
    ----------
    model : AbstractModel
        Model exposing continuous or discrete dynamics.
    state : Trajectory
        Trajectory holding stacked states X of shape (N, n), controls U of shape (N-1, m),
        times t of shape (N,), and step durations dt of shape (N-1,).

    Returns
    -------
    Expansion
        Expansion holding stacked A of shape (N-1, ne, ne) and B of shape (N-1, ne, m).
    """
    discrete_model = model.discretize()

    X = state.X
    U = state.U
    t = state.t
    dt = state.dt

    N = state.N
    ne = discrete_model.ne
    m = discrete_model.m

    def step_expansion(
        xk: jax.Array,
        uk: jax.Array,
        x_next: jax.Array,
        tk: float | jax.Array,
        dtk: float | jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        Ak = discrete_model.state_jacobian(xk, uk, tk, dtk)
        Bk = discrete_model.control_jacobian(xk, uk, tk, dtk)
        Gk = discrete_model.errstate_jacobian(xk)
        G_next = discrete_model.errstate_jacobian(x_next)
        A_bar = G_next.T @ Ak @ Gk
        B_bar = G_next.T @ Bk
        return A_bar, B_bar

    A_stacked, B_stacked = jax.vmap(step_expansion)(X[:-1], U, X[1:], t[:-1], dt)

    dtype = X.dtype
    q_zero = jnp.zeros((N, ne), dtype=dtype)
    r_zero = jnp.zeros((N - 1, m), dtype=dtype)
    Q_zero = jnp.zeros((N, ne, ne), dtype=dtype)
    R_zero = jnp.zeros((N - 1, m, m), dtype=dtype)
    H_zero = jnp.zeros((N - 1, m, ne), dtype=dtype)

    return Expansion(
        A=A_stacked,
        B=B_stacked,
        q=q_zero,
        r=r_zero,
        Q=Q_zero,
        R=R_zero,
        H=H_zero,
    )


def _stage_cost_expansion(
    obj: Objective,
    X_stage: jax.Array,
    U_stage: jax.Array,
    t_stage: jax.Array,
    model: AbstractModel | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute stacked stage cost derivatives in error coordinates."""
    n = obj.n
    m = obj.m
    dtype = X_stage.dtype
    n_stage = X_stage.shape[0]

    if isinstance(obj.stage_cost, DiagonalCost):
        Q_diag = obj.Q
        R_diag = obj.R
        q_diag = obj.q
        r_diag = obj.r

        gx = Q_diag * X_stage + q_diag
        gu = R_diag * U_stage + r_diag
        Hxx = jax.vmap(jnp.diag)(Q_diag)
        Huu = jax.vmap(jnp.diag)(R_diag)
        Hux = jnp.zeros((n_stage, m, n), dtype=dtype)
    elif isinstance(obj.stage_cost, QuadraticCost):
        Q_mat = obj.Q
        R_mat = obj.R
        H_mat = obj.H if obj.H is not None else jnp.zeros((n_stage, m, n), dtype=dtype)
        q_vec = obj.q
        r_vec = obj.r

        gx = jnp.einsum("nij,nj->ni", Q_mat, X_stage) + jnp.einsum("nji,nj->ni", H_mat, U_stage) + q_vec
        gu = jnp.einsum("nij,nj->ni", R_mat, U_stage) + jnp.einsum("nij,nj->ni", H_mat, X_stage) + r_vec
        Hxx = Q_mat
        Huu = R_mat
        Hux = H_mat
    else:

        def single_step(
            xk: jax.Array,
            uk: jax.Array,
            tk: float | jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            gx_k = jax.grad(lambda x_: obj.stage_cost.evaluate(x_, uk, tk))(xk)
            gu_k = jax.grad(lambda u_: obj.stage_cost.evaluate(xk, u_, tk))(uk)
            Hxx_k = jax.hessian(lambda x_: obj.stage_cost.evaluate(x_, uk, tk))(xk)
            Huu_k = jax.hessian(lambda u_: obj.stage_cost.evaluate(xk, u_, tk))(uk)
            Hux_k = jax.jacobian(lambda x_in: jax.grad(lambda u_in: obj.stage_cost.evaluate(x_in, u_in, tk))(uk))(xk)
            return gx_k, gu_k, Hxx_k, Huu_k, Hux_k

        gx, gu, Hxx, Huu, Hux = jax.vmap(single_step)(X_stage, U_stage, t_stage)

    if model is not None:
        G_stage = jax.vmap(model.errstate_jacobian)(X_stage)
    else:
        G_stage = jnp.broadcast_to(jnp.eye(n, dtype=dtype), (n_stage, n, n))

    q_bar = jnp.einsum("nji,nj->ni", G_stage, gx)
    r_bar = gu
    Q_bar = jnp.einsum("nki,nkl,nlj->nij", G_stage, Hxx, G_stage)
    R_bar = Huu
    H_bar = jnp.einsum("nik,nkj->nij", Hux, G_stage)

    return q_bar, r_bar, Q_bar, R_bar, H_bar


def _terminal_cost_expansion(
    obj: Objective,
    x_term: jax.Array,
    t_term: float | jax.Array,
    model: AbstractModel | None,
) -> tuple[jax.Array, jax.Array]:
    """Compute terminal cost gradient and Hessian in error coordinates."""
    n = obj.n

    if isinstance(obj.terminal_cost, DiagonalCost):
        gx_term = obj.Q_f * x_term + obj.q_f
        Hxx_term = jnp.diag(obj.Q_f)
    elif isinstance(obj.terminal_cost, QuadraticCost):
        gx_term = obj.Q_f @ x_term + obj.q_f
        Hxx_term = obj.Q_f
    else:
        gx_term = jax.grad(lambda x_: obj.terminal_cost.evaluate(x_, None, t_term))(x_term)
        Hxx_term = jax.hessian(lambda x_: obj.terminal_cost.evaluate(x_, None, t_term))(x_term)

    G_term = model.errstate_jacobian(x_term) if model is not None else jnp.eye(n, dtype=x_term.dtype)
    q_term = G_term.T @ gx_term
    Q_term = G_term.T @ Hxx_term @ G_term

    return q_term, Q_term


def _cost_expansion(
    obj: Objective,
    state: Trajectory,
    model: AbstractModel | None = None,
) -> Expansion:
    """Compute the stacked first- and second-order cost expansion in error coordinates.

    Parameters
    ----------
    obj : Objective
        Objective containing stage and terminal costs.
    state : Trajectory
        Trajectory holding stacked states X of shape (N, n), controls U of shape (N-1, m),
        and times t of shape (N,).
    model : AbstractModel | None, optional
        Model defining the error state coordinates. Defaults to None, meaning Euclidean.

    Returns
    -------
    Expansion
        Expansion holding stacked q of shape (N, ne), r of shape (N-1, m), Q of shape (N, ne, ne),
        R of shape (N-1, m, m), and H of shape (N-1, m, ne) in error coordinates.
    """
    resolved_model = model

    X = state.X
    U = state.U
    t = state.t
    N = state.N
    m = obj.m
    ne = resolved_model.ne if resolved_model is not None else obj.n
    dtype = X.dtype

    # 1. Stage cost expansions k = 0, ..., N-2
    q_st, r_st, Q_st, R_st, H_st = _stage_cost_expansion(obj, X[:-1], U, t[:-1], resolved_model)

    # 2. Terminal cost expansion k = N-1
    q_term, Q_term = _terminal_cost_expansion(obj, X[-1], t[-1], resolved_model)

    # 3. Assemble stacked arrays
    q_stacked = jnp.concatenate([q_st, jnp.expand_dims(q_term, 0)], axis=0)
    Q_stacked = jnp.concatenate([Q_st, jnp.expand_dims(Q_term, 0)], axis=0)

    A_zero = jnp.zeros((N - 1, ne, ne), dtype=dtype)
    B_zero = jnp.zeros((N - 1, ne, m), dtype=dtype)

    return Expansion(
        A=A_zero,
        B=B_zero,
        q=q_stacked,
        r=r_st,
        Q=Q_stacked,
        R=R_st,
        H=H_st,
    )


def _evaluate_knot_penalty(
    ctx: _KnotALContext,
    x_in: jax.Array,
    u_in: jax.Array | None,
) -> jax.Array:
    """Evaluate augmented Lagrangian penalty for a knot point's constraints and box bounds."""
    pen_sum = jnp.zeros((), dtype=x_in.dtype)
    off = 0
    for c in (*ctx.evaluator.constraints, *ctx.bounds.constraints):
        p_c = c.p
        lam_c = ctx.lam_k[off : off + p_c]
        val_c = c.evaluate(x_in, u_in, ctx.tk)
        shifted = val_c + lam_c / ctx.mu_k
        proj = c.cone.project_dual(shifted)
        pen = jnp.dot(lam_c, proj) + 0.5 * ctx.mu_k * jnp.dot(proj, proj)
        pen_sum = pen_sum + pen
        off += p_c
    return pen_sum


def _stage_al_derivatives(
    ctx: _KnotALContext,
    xk: jax.Array,
    uk: jax.Array,
    Gk: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate stage AL gradients and Hessian blocks in error coordinates."""

    def penalty_fn(x_: jax.Array, u_: jax.Array) -> jax.Array:
        return _evaluate_knot_penalty(ctx, x_, u_)

    gx = jax.grad(lambda x_: penalty_fn(x_, uk))(xk)
    gu = jax.grad(lambda u_: penalty_fn(xk, u_))(uk)
    Hxx = jax.hessian(lambda x_: penalty_fn(x_, uk))(xk)
    Huu = jax.hessian(lambda u_: penalty_fn(xk, u_))(uk)
    Hux = jax.jacobian(lambda x_in: jax.grad(lambda u_in: penalty_fn(x_in, u_in))(uk))(xk)

    q_bar = Gk.T @ gx
    r_bar = gu
    Q_bar = Gk.T @ Hxx @ Gk
    R_bar = Huu
    H_bar = Hux @ Gk
    return q_bar, r_bar, Q_bar, R_bar, H_bar


def _term_al_derivatives(
    ctx: _KnotALContext,
    xk: jax.Array,
    Gk: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate terminal AL gradient and Hessian blocks in error coordinates."""

    def penalty_fn(x_: jax.Array) -> jax.Array:
        return _evaluate_knot_penalty(ctx, x_, None)

    gx = jax.grad(penalty_fn)(xk)
    Hxx = jax.hessian(penalty_fn)(xk)
    q_bar = Gk.T @ gx
    Q_bar = Gk.T @ Hxx @ Gk
    return q_bar, Q_bar


def _parse_multipliers(
    lam: Sequence[jax.Array] | jax.Array | None,
    p_list: Sequence[int],
    n_knots: int,
    dtype: jnp.dtype | type,
) -> list[jax.Array]:
    """Parse multiplier inputs into per-knot arrays matching constraint dimensions."""
    if lam is None:
        return [jnp.zeros(p_k, dtype=dtype) for p_k in p_list]
    if isinstance(lam, (list, tuple)):
        return [jnp.asarray(lam_k, dtype=dtype) for lam_k in lam]
    lam_arr = jnp.asarray(lam, dtype=dtype)
    if lam_arr.ndim == 1 and lam_arr.shape[0] == sum(p_list):
        lam_list: list[jax.Array] = []
        offset = 0
        for p_k in p_list:
            lam_list.append(lam_arr[offset : offset + p_k])
            offset += p_k
        return lam_list
    if lam_arr.ndim == _EXPECTED_NDIM_2D and lam_arr.shape[0] == n_knots:
        return [lam_arr[k, : p_list[k]] for k in range(n_knots)]
    return [jnp.zeros(p_k, dtype=dtype) for p_k in p_list]


def _parse_penalties(
    mu: float | jax.Array,
    n_knots: int,
    dtype: jnp.dtype | type,
) -> list[jax.Array]:
    """Parse penalty inputs into per-knot scalar arrays."""
    mu_arr = jnp.asarray(mu, dtype=dtype)
    if mu_arr.ndim == 0:
        return [mu_arr] * n_knots
    if mu_arr.ndim == 1 and mu_arr.shape[0] == n_knots:
        return [mu_arr[k] for k in range(n_knots)]
    return [mu_arr] * n_knots


def _knot_al_expansion(
    ctx: _KnotALContext,
    xk: jax.Array,
    uk: jax.Array | None,
    model: AbstractModel | None,
) -> tuple[jax.Array, jax.Array | None, jax.Array, jax.Array | None, jax.Array | None]:
    """Compute AL gradient and Hessian contributions at a single knot point."""
    dtype = xk.dtype
    n_x = int(xk.shape[0])
    ne = model.ne if model is not None else n_x
    m = int(uk.shape[0]) if uk is not None else int(ctx.evaluator.m)
    Gk = model.errstate_jacobian(xk) if model is not None else jnp.eye(n_x, dtype=dtype)

    if ctx.p == 0:
        q_al = jnp.zeros(ne, dtype=dtype)
        Q_al = jnp.zeros((ne, ne), dtype=dtype)
        r_al = jnp.zeros(m, dtype=dtype) if uk is not None else None
        R_al = jnp.zeros((m, m), dtype=dtype) if uk is not None else None
        H_al = jnp.zeros((m, ne), dtype=dtype) if uk is not None else None
        return q_al, r_al, Q_al, R_al, H_al

    if not ctx.evaluator.is_terminal and uk is not None:
        q_al, r_al, Q_al, R_al, H_al = _stage_al_derivatives(ctx, xk, uk, Gk)
        return q_al, r_al, Q_al, R_al, H_al

    q_al, Q_al = _term_al_derivatives(ctx, xk, Gk)
    return q_al, None, Q_al, None, None


def _augmented_lagrangian_expansion(  # noqa: PLR0913, PLR0917 -- AL expansion takes constraints, state, expansion, lam, mu, model
    constraints: BuiltConstraintList,
    state: Trajectory,
    expansion: Expansion,
    lam: Sequence[jax.Array] | jax.Array | None = None,
    mu: float | jax.Array = 1.0,
    model: AbstractModel | None = None,
) -> Expansion:
    """Add augmented Lagrangian gradient and Hessian contributions into an existing Expansion.

    Parameters
    ----------
    constraints : BuiltConstraintList
        Built constraint list holding active constraints and active knot-point ranges.
        Box bounds are penalised alongside the constraint rows, since a native solver has no
        variable bounds of its own; each knot's multipliers run constraint rows then bound rows.
    state : Trajectory
        Trajectory holding stacked states X of shape (N, n), controls U of shape (N-1, m),
        and times t of shape (N,).
    expansion : Expansion
        Existing Expansion to which the augmented Lagrangian contributions are added.
    lam : Sequence[jax.Array] | jax.Array | None, optional
        Lagrange multiplier vectors per knot point or concatenated 1D vector.
    mu : float | jax.Array, optional
        Penalty parameter (scalar or array of length N). Default is 1.0.
    model : AbstractModel | None, optional
        Model defining error state coordinates. Defaults to None, meaning Euclidean.

    Returns
    -------
    Expansion
        New Expansion with augmented Lagrangian gradient and Hessian terms added.
    """
    built_constraints = constraints
    resolved_model = model

    if len(built_constraints.knot_evaluators) == 0:
        return expansion

    X = state.X
    U = state.U
    t = state.t
    N = state.N
    dtype = X.dtype

    p_penalised = tuple(
        ev.p + bd.p
        for ev, bd in zip(built_constraints.knot_evaluators, built_constraints.bound_evaluators, strict=True)
    )
    lam_list = _parse_multipliers(lam, p_penalised, N, dtype)
    mu_list = _parse_penalties(mu, N, dtype)

    q_al_list: list[jax.Array] = []
    r_al_list: list[jax.Array] = []
    Q_al_list: list[jax.Array] = []
    R_al_list: list[jax.Array] = []
    H_al_list: list[jax.Array] = []

    for k in range(N):
        ctx = _KnotALContext(
            evaluator=built_constraints.knot_evaluators[k],
            bounds=built_constraints.bound_evaluators[k],
            lam_k=lam_list[k],
            mu_k=mu_list[k],
            tk=t[k],
        )
        xk = X[k]
        uk = U[k] if k < N - 1 else None

        q_k, r_k, Q_k, R_k, H_k = _knot_al_expansion(ctx, xk, uk, resolved_model)
        q_al_list.append(q_k)
        Q_al_list.append(Q_k)
        if r_k is not None:
            r_al_list.append(r_k)
        if R_k is not None:
            R_al_list.append(R_k)
        if H_k is not None:
            H_al_list.append(H_k)

    q_al_stacked = jnp.stack(q_al_list, axis=0)
    Q_al_stacked = jnp.stack(Q_al_list, axis=0)
    r_al_stacked = jnp.stack(r_al_list, axis=0)
    R_al_stacked = jnp.stack(R_al_list, axis=0)
    H_al_stacked = jnp.stack(H_al_list, axis=0)

    new_q = expansion.q + q_al_stacked
    new_r = expansion.r + r_al_stacked
    new_Q = expansion.Q + Q_al_stacked
    new_R = expansion.R + R_al_stacked
    new_H = expansion.H + H_al_stacked

    return Expansion(
        A=expansion.A,
        B=expansion.B,
        q=new_q,
        r=new_r,
        Q=new_Q,
        R=new_R,
        H=new_H,
    )
