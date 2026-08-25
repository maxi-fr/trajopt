import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import ZeroCone
from trajopt.constraints.constraint_list import BuiltConstraintList
from trajopt.dynamics.base import AbstractModel
from trajopt.expansions import Expansion
from trajopt.solvers.options import SolverOptions
from trajopt.trajectory import Trajectory


class ALConstraints(eqx.Module):
    """Padded per-knot, per-row augmented-Lagrangian duals and penalties.

    Stores lambda using Altro's **non-conic** sign convention (reference Finding E):
    `lambda_bar = lambda + mu * c`, with penalty cost `+ lambda' c`. This is neither Altro's
    conic convention (`lambda_bar = lambda - mu * c`) nor the convention the code this module
    replaces used (shifting by `c + lambda / mu` and projecting onto the dual cone). Ticket 31's
    conic path must convert explicitly at its boundary rather than reinterpret this field.

    Rows are laid out per knot as `[constraint rows (0 .. p_cons_max) | x_upper(n) | x_lower(n) |
    u_upper(m) | u_lower(m)]`, padded to a fixed `p_max = p_cons_max + 2n + 2m` across every knot.
    `row_mask` marks which rows are structurally real at that knot (fewer constraint rows than
    `p_cons_max`, an infinite bound, or a terminal knot's absent control rows); masked rows carry
    `lam = 0` and are excluded from every reduction (cost, gradient, Hessian, violation, updates).

    Parameters
    ----------
    lam : jax.Array
        Padded dual multipliers of shape (N, p_max).
    mu : jax.Array
        Padded penalty parameters of shape (N, p_max).
    row_mask : jax.Array
        Boolean mask of shape (N, p_max), True where the row is a real constraint/bound row.
    is_equality : jax.Array
        Boolean mask of shape (N, p_max), True where the row maps to Altro's `Equality`
        (`ZeroCone`); False maps to `Inequality` (`NegativeOrthant`, including every box bound).
    p_cons_max : int
        Padded width of the constraint-only block (excludes the box-bound rows).
    """

    lam: jax.Array
    mu: jax.Array
    row_mask: jax.Array
    is_equality: jax.Array
    p_cons_max: int = eqx.field(static=True)

    @classmethod
    def build(cls, constraints: BuiltConstraintList, penalty_initial: float = 1.0) -> "ALConstraints":
        """Allocate a fresh padded AL layout for `constraints`, lambda=0 and mu=penalty_initial on real rows.

        Structural (row_mask, is_equality) computation is eager Python/NumPy over `constraints`,
        run once when a Problem is set up rather than under trace, since it depends only on
        constraint structure, not on any trajectory.
        """
        N = constraints.N
        n = constraints.n
        m = constraints.m
        p_cons_max = max(constraints.p) if constraints.p else 0

        row_mask_cons = np.zeros((N, p_cons_max), dtype=bool)
        is_eq_cons = np.zeros((N, p_cons_max), dtype=bool)
        for k, ev in enumerate(constraints.knot_evaluators):
            off = 0
            for c in ev.constraints:
                p_c = c.p
                is_eq_cons[k, off : off + p_c] = isinstance(c.cone, ZeroCone)
                off += p_c
            row_mask_cons[k, : ev.p] = True

        x_upper = np.asarray(constraints.x_upper)
        x_lower = np.asarray(constraints.x_lower)
        u_upper_pad = np.concatenate([np.asarray(constraints.u_upper), np.full((1, m), np.inf)], axis=0)
        u_lower_pad = np.concatenate([np.asarray(constraints.u_lower), np.full((1, m), -np.inf)], axis=0)

        row_mask_bound = np.concatenate(
            [np.isfinite(x_upper), np.isfinite(x_lower), np.isfinite(u_upper_pad), np.isfinite(u_lower_pad)],
            axis=-1,
        )
        is_eq_bound = np.zeros((N, 2 * n + 2 * m), dtype=bool)

        row_mask = jnp.asarray(np.concatenate([row_mask_cons, row_mask_bound], axis=-1))
        is_equality = jnp.asarray(np.concatenate([is_eq_cons, is_eq_bound], axis=-1))

        p_max = p_cons_max + 2 * n + 2 * m
        lam = jnp.zeros((N, p_max), dtype=jnp.float64)
        mu = jnp.where(row_mask, jnp.asarray(penalty_initial, dtype=jnp.float64), 0.0)

        return cls(lam=lam, mu=mu, row_mask=row_mask, is_equality=is_equality, p_cons_max=p_cons_max)


def _evaluate_constraint_block(
    constraints: BuiltConstraintList,
    X: jax.Array,
    U: jax.Array,
    T: jax.Array,
    p_cons_max: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Scatter each structural constraint group's evaluate/jacobian into a padded (N, p_cons_max, ...) block.

    Iterates `constraints.groups`, a tuple bounded by the number of structurally distinct knots
    (typically O(1): stage vs. terminal), not by N -- each group is filled with a single `vmap`
    over its knot indices, so this loop does not unroll per knot under trace.
    """
    N, n = X.shape
    m = U.shape[1]
    dtype = X.dtype

    C = jnp.zeros((N, p_cons_max), dtype=dtype)
    Jx = jnp.zeros((N, p_cons_max, n), dtype=dtype)
    Ju = jnp.zeros((N, p_cons_max, m), dtype=dtype)

    for g in constraints.groups:
        p_g = g.evaluator.p
        if p_g == 0:
            continue
        knots = jnp.asarray(g.knots)
        c_g = g.evaluate(X, U, T)
        jx_g, ju_g = g.jacobian(X, U, T)
        C = C.at[knots, :p_g].set(c_g)
        Jx = Jx.at[knots, :p_g, :].set(jx_g)
        if not g.evaluator.is_terminal:
            Ju = Ju.at[knots, :p_g, :].set(ju_g)

    return C, Jx, Ju


def _evaluate_bound_block(
    constraints: BuiltConstraintList,
    X: jax.Array,
    U: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate box-bound residual rows [x_upper | x_lower | u_upper | u_lower] and constant Jacobians.

    Pure vectorized array ops over all N knots at once (no python loop, no vmap needed); infinite
    bounds are replaced with a safe finite stand-in since `ALConstraints.row_mask` excludes those
    rows from every reduction regardless of the (otherwise meaningless) residual value there.
    """
    N, n = X.shape
    m = U.shape[1]
    dtype = X.dtype

    x_upper, x_lower = constraints.x_upper, constraints.x_lower
    U_pad = jnp.concatenate([U, jnp.zeros((1, m), dtype=dtype)], axis=0)
    u_upper_pad = jnp.concatenate([constraints.u_upper, jnp.full((1, m), jnp.inf, dtype=dtype)], axis=0)
    u_lower_pad = jnp.concatenate([constraints.u_lower, jnp.full((1, m), -jnp.inf, dtype=dtype)], axis=0)

    x_upper_safe = jnp.where(jnp.isfinite(x_upper), x_upper, 0.0)
    x_lower_safe = jnp.where(jnp.isfinite(x_lower), x_lower, 0.0)
    u_upper_safe = jnp.where(jnp.isfinite(u_upper_pad), u_upper_pad, 0.0)
    u_lower_safe = jnp.where(jnp.isfinite(u_lower_pad), u_lower_pad, 0.0)

    C_bound = jnp.concatenate(
        [X - x_upper_safe, x_lower_safe - X, U_pad - u_upper_safe, u_lower_safe - U_pad],
        axis=-1,
    )

    I_n = jnp.eye(n, dtype=dtype)
    I_m = jnp.eye(m, dtype=dtype)
    Jx_row = jnp.concatenate([I_n, -I_n, jnp.zeros((2 * m, n), dtype=dtype)], axis=0)
    Ju_row = jnp.concatenate([jnp.zeros((2 * n, m), dtype=dtype), I_m, -I_m], axis=0)
    Jx_bound = jnp.broadcast_to(Jx_row, (N, 2 * n + 2 * m, n))
    Ju_bound = jnp.broadcast_to(Ju_row, (N, 2 * n + 2 * m, m))

    return C_bound, Jx_bound, Ju_bound


def evaluate_al_constraints(
    al: ALConstraints,
    constraints: BuiltConstraintList,
    model: AbstractModel | None,
    traj: Trajectory,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate padded per-knot constraint and box-bound residuals and Jacobians in error coordinates.

    Parameters
    ----------
    al : ALConstraints
        AL layout providing `p_cons_max`.
    constraints : BuiltConstraintList
        Built constraint list holding constraint groups and box-bound limits.
    model : AbstractModel | None
        Model defining error-state coordinates. None means Euclidean (G = I).
    traj : Trajectory
        Trajectory holding stacked states X, controls U, and times t.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        C of shape (N, p_max), dc/dx in error coordinates of shape (N, p_max, ne), and dc/du of
        shape (N, p_max, m).
    """
    X, U, T = traj.X, traj.U, traj.t
    C_cons, Jx_cons, Ju_cons = _evaluate_constraint_block(constraints, X, U, T, al.p_cons_max)
    C_bound, Jx_bound, Ju_bound = _evaluate_bound_block(constraints, X, U)

    C = jnp.concatenate([C_cons, C_bound], axis=-1)
    Jx = jnp.concatenate([Jx_cons, Jx_bound], axis=1)
    Ju = jnp.concatenate([Ju_cons, Ju_bound], axis=1)

    n = X.shape[1]
    dtype = X.dtype
    if model is not None:
        G_all = jax.vmap(model.errstate_jacobian)(X)
    else:
        G_all = jnp.broadcast_to(jnp.eye(n, dtype=dtype), (X.shape[0], n, n))
    Jx_err = jnp.einsum("kpn,kne->kpe", Jx, G_all)

    return C, Jx_err, Ju


def _active_penalty(al: ALConstraints, C: jax.Array) -> jax.Array:
    """Row-wise active penalty `a`: `mu` where `(equality | c >= 0 | lam > 0) & row_mask`, else 0.

    Reduces to Altro's `Equality` branch (`a = mu` unconditionally) since `is_equality` forces the
    active test true, and to its `Inequality` branch (`a = ((c>=0)|(lam>0)) * mu`) otherwise.
    """
    active = (al.is_equality | (C >= 0.0) | (al.lam > 0.0)) & al.row_mask
    return jnp.where(active, al.mu, 0.0)


def _lam_bar(al: ALConstraints, C: jax.Array, a: jax.Array) -> jax.Array:
    """Row-wise `lambda_bar = lambda + a * c`, masked rows contributing lambda = 0."""
    lam_masked = jnp.where(al.row_mask, al.lam, 0.0)
    return jnp.where(al.row_mask, lam_masked + a * C, 0.0)


def al_cost(al: ALConstraints, C: jax.Array) -> jax.Array:
    """Augmented Lagrangian penalty cost, Altro's `alcost`: sum of `lambda' c + 0.5 * a * c^2`.

    Parameters
    ----------
    al : ALConstraints
        Current duals and penalties.
    C : jax.Array
        Padded constraint residuals of shape (N, p_max), from `evaluate_al_constraints`.

    Returns
    -------
    jax.Array
        Scalar penalty cost, masked rows contributing 0.
    """
    a = _active_penalty(al, C)
    lam_masked = jnp.where(al.row_mask, al.lam, 0.0)
    per_row = jnp.where(al.row_mask, lam_masked * C + 0.5 * a * C**2, 0.0)
    return jnp.sum(per_row)


def al_grad_hess(
    al: ALConstraints,
    C: jax.Array,
    Jx_err: jax.Array,
    Ju: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Augmented Lagrangian gradient and Gauss-Newton Hessian blocks, Altro's `algrad!` / `alhess!`.

    The Hessian is `J' diag(a) J` (Gauss-Newton), not `jax.hessian` of the penalty: exact for
    affine constraints, PSD by construction for nonlinear ones (reference §7.1).

    Parameters
    ----------
    al : ALConstraints
        Current duals and penalties.
    C : jax.Array
        Padded constraint residuals of shape (N, p_max).
    Jx_err : jax.Array
        Padded state Jacobian in error coordinates, shape (N, p_max, ne).
    Ju : jax.Array
        Padded control Jacobian, shape (N, p_max, m).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
        `(grad_x, grad_u, Hxx, Huu, Hux)` of shapes (N, ne), (N, m), (N, ne, ne), (N, m, m),
        (N, m, ne). `grad_u`, `Huu`, `Hux` are structurally zero at the terminal knot.
    """
    a = _active_penalty(al, C)
    lam_bar = _lam_bar(al, C, a)

    grad_x = jnp.einsum("kpe,kp->ke", Jx_err, lam_bar)
    grad_u = jnp.einsum("kpm,kp->km", Ju, lam_bar)
    Hxx = jnp.einsum("kpe,kp,kpf->kef", Jx_err, a, Jx_err)
    Huu = jnp.einsum("kpm,kp,kpn->kmn", Ju, a, Ju)
    Hux = jnp.einsum("kpm,kp,kpe->kme", Ju, a, Jx_err)
    return grad_x, grad_u, Hxx, Huu, Hux


def add_al_expansion(
    expansion: Expansion,
    al: ALConstraints,
    C: jax.Array,
    Jx_err: jax.Array,
    Ju: jax.Array,
) -> Expansion:
    """Add augmented Lagrangian gradient and Hessian contributions into an existing Expansion."""
    grad_x, grad_u, Hxx, Huu, Hux = al_grad_hess(al, C, Jx_err, Ju)
    return Expansion(
        A=expansion.A,
        B=expansion.B,
        q=expansion.q + grad_x,
        r=expansion.r + grad_u[:-1],
        Q=expansion.Q + Hxx,
        R=expansion.R + Huu[:-1],
        H=expansion.H + Hux[:-1],
    )


def dual_update(al: ALConstraints, C: jax.Array, options: SolverOptions) -> ALConstraints:
    """Update lambda, Altro's `dualupdate!`: equality `lam <- lam + mu*c`, inequality `lam <- max(0, lam + mu*c)`.

    Uses the full `mu`, not the active-gated `a` (Altro's dual update always applies the full
    penalty). Both branches are then clamped to `+-dual_max`, matching the unconditional clamp
    Altro applies after the cone-specific branch. Masked rows stay at lambda = 0.
    """
    raw = al.lam + al.mu * C
    updated = jnp.where(al.is_equality, raw, jnp.maximum(0.0, raw))
    clamped = jnp.clip(updated, -options.dual_max, options.dual_max)
    new_lam = jnp.where(al.row_mask, clamped, 0.0)
    return eqx.tree_at(lambda a: a.lam, al, new_lam)


def penalty_update(al: ALConstraints, options: SolverOptions) -> ALConstraints:
    """Update mu, Altro's `penaltyupdate!`: `mu <- clamp(mu * penalty_scaling, 0, penalty_max)`.

    Applied unconditionally to every real row (no active-set gating); masked rows are left frozen,
    keeping them inert rather than accumulating scaling that is never read.
    """
    scaled = jnp.clip(al.mu * options.penalty_scaling, 0.0, options.penalty_max)
    new_mu = jnp.where(al.row_mask, scaled, al.mu)
    return eqx.tree_at(lambda a: a.mu, al, new_mu)


def max_violation(al: ALConstraints, C: jax.Array) -> jax.Array:
    """Max constraint violation, Altro's `max_violation`: infinity-norm of `Pi_K(c) - c` over real rows.

    Equality rows project to 0 (`viol = -c`); inequality rows project onto the negative orthant
    (`viol = min(0, c) - c = -max(0, c)`).
    """
    viol = jnp.where(al.is_equality, -C, -jnp.maximum(0.0, C))
    viol_masked = jnp.where(al.row_mask, jnp.abs(viol), 0.0)
    return jnp.max(viol_masked)


def max_penalty(al: ALConstraints) -> jax.Array:
    """Max penalty parameter over real rows, Altro's `max_penalty`; 0 if there are no real rows."""
    mu_masked = jnp.where(al.row_mask, al.mu, -jnp.inf)
    has_rows = jnp.any(al.row_mask)
    return jnp.where(has_rows, jnp.max(mu_masked), 0.0)
