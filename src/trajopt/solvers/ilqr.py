import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg

from trajopt.expansions import Expansion
from trajopt.solvers.options import SolverOptions


class DynamicRegularization(eqx.Module):
    """Backward-pass regularization state, ported from Altro's `DynamicRegularization`.

    Parameters
    ----------
    rho : jax.Array
        Regularization added to `Quu` before inversion, as a float scalar.
    drho : jax.Array
        Regularization derivative term used to scale `rho` on the next increase/decrease.
    """

    rho: jax.Array
    drho: jax.Array

    @classmethod
    def initial(cls, options: SolverOptions) -> "DynamicRegularization":
        """Construct the regularization state Altro starts a solve with (`reg = (bp_reg_initial, 0)`)."""
        return cls(
            rho=jnp.asarray(options.bp_reg_initial, dtype=jnp.float64),
            drho=jnp.asarray(0.0, dtype=jnp.float64),
        )


def increase_regularization(
    reg: DynamicRegularization,
    options: SolverOptions,
) -> DynamicRegularization:
    """Raise `(rho, drho)` after a failed backward-pass sweep, matching `increaseregularization!`."""
    rho_dot = options.bp_reg_increase_factor
    rho_min = options.bp_reg_min
    drho = jnp.maximum(reg.drho * rho_dot, rho_dot)
    rho = jnp.maximum(reg.rho * drho, rho_min)
    return DynamicRegularization(rho=rho, drho=drho)


def decrease_regularization(
    reg: DynamicRegularization,
    options: SolverOptions,
) -> DynamicRegularization:
    """Lower `(rho, drho)` once per backward pass, matching `decreaseregularization!`."""
    rho_dot = options.bp_reg_increase_factor
    rho_min = options.bp_reg_min
    drho = jnp.minimum(reg.drho / rho_dot, 1.0 / rho_dot)
    rho = jnp.maximum(rho_min, reg.rho * drho)
    return DynamicRegularization(rho=rho, drho=drho)


class BackwardPassResult(eqx.Module):
    """Output of one backward pass: the affine policy, cost-to-go, and expected decrease.

    Parameters
    ----------
    K : jax.Array
        Feedback gains of shape `(N-1, m, ne)`.
    d : jax.Array
        Feedforward terms of shape `(N-1, m)`.
    S_x : jax.Array
        Cost-to-go gradient per knot of shape `(N, ne)`.
    S_xx : jax.Array
        Cost-to-go Hessian per knot of shape `(N, ne, ne)`.
    dV : jax.Array
        Expected cost decrease terms `[sum(d'Qu), sum(0.5 d'Quu d)]`, shape `(2,)`.
    regularization : DynamicRegularization
        Regularization state after this backward pass (post decrease).
    failed : jax.Array
        Whether the retry loop exhausted `bp_reg_max` while still failing, as a bool scalar.
    """

    K: jax.Array
    d: jax.Array
    S_x: jax.Array
    S_xx: jax.Array
    dV: jax.Array  # noqa: N815 -- ports Altro's ΔV field name verbatim
    regularization: DynamicRegularization
    failed: jax.Array


class _SweepResult(eqx.Module):
    """One reversed scan over knots at a fixed rho; see `BackwardPassResult` for field meaning."""

    K: jax.Array
    d: jax.Array
    S_x: jax.Array
    S_xx: jax.Array
    dV: jax.Array  # noqa: N815 -- ports Altro's ΔV field name verbatim
    failed: jax.Array


def _knot_step(
    carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    k: jax.Array,
    expansion: Expansion,
    rho: jax.Array,
    ne: int,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """One knot of the reversed Riccati recursion, regularizing `Quu` by `rho` (finding G)."""
    S_x, S_xx, dV1, dV2, failed = carry

    A = expansion.A[k]
    B = expansion.B[k]
    q_k = expansion.q[k]
    r_k = expansion.r[k]
    Q_k = expansion.Q[k]
    R_k = expansion.R[k]
    H_k = expansion.H[k]

    Qx = A.T @ S_x + q_k
    Qu = B.T @ S_x + r_k
    Qxx = A.T @ S_xx @ A + Q_k
    Quu = B.T @ S_xx @ B + R_k
    Qux = B.T @ S_xx @ A + H_k

    m = Quu.shape[0]
    Quu_reg = Quu + rho * jnp.eye(m, dtype=Quu.dtype)

    L = jnp.linalg.cholesky(Quu_reg)
    step_failed = jnp.any(jnp.isnan(L))

    rhs = jnp.concatenate([Qux, Qu[:, None]], axis=1)
    Kd = -jax.scipy.linalg.cho_solve((L, True), rhs)
    K_k = Kd[:, :ne]
    d_k = Kd[:, ne]

    S_x_new = Qx + K_k.T @ (Quu @ d_k) + K_k.T @ Qu + Qux.T @ d_k
    S_xx_new = Qxx + K_k.T @ (Quu @ K_k) + K_k.T @ Qux + Qux.T @ K_k
    S_xx_new = 0.5 * (S_xx_new + S_xx_new.T)

    dV1_new = dV1 + jnp.dot(d_k, Qu)
    dV2_new = dV2 + 0.5 * jnp.dot(d_k, Quu @ d_k)
    failed_new = failed | step_failed

    new_carry = (S_x_new, S_xx_new, dV1_new, dV2_new, failed_new)
    outputs = (K_k, d_k, S_x_new, S_xx_new)
    return new_carry, outputs


def _sweep(expansion: Expansion, rho: jax.Array) -> _SweepResult:
    """Run one full reversed Riccati recursion over all knots at a fixed `rho`.

    Always runs every knot to completion, carrying a `failed` flag rather than exiting early
    (ticket 25: `jnp.linalg.cholesky` returns NaN instead of raising, so an indefinite `Quu` at
    one knot poisons the rest of the sweep with NaNs but never aborts it).
    """
    ne = expansion.ne
    N = expansion.N

    S_x_terminal = expansion.q[-1]
    S_xx_terminal = expansion.Q[-1]
    init_carry = (
        S_x_terminal,
        S_xx_terminal,
        jnp.asarray(0.0, dtype=expansion.q.dtype),
        jnp.asarray(0.0, dtype=expansion.q.dtype),
        jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
    )
    ks = jnp.arange(N - 2, -1, -1)

    final_carry, (K_rev, d_rev, Sx_rev, Sxx_rev) = jax.lax.scan(
        lambda c, k: _knot_step(c, k, expansion, rho, ne),
        init_carry,
        ks,
    )
    _, _, dV1, dV2, failed = final_carry

    K = jnp.flip(K_rev, axis=0)
    d = jnp.flip(d_rev, axis=0)
    S_x = jnp.concatenate([jnp.flip(Sx_rev, axis=0), S_x_terminal[None]], axis=0)
    S_xx = jnp.concatenate([jnp.flip(Sxx_rev, axis=0), S_xx_terminal[None]], axis=0)
    dV = jnp.stack([dV1, dV2])

    return _SweepResult(K=K, d=d, S_x=S_x, S_xx=S_xx, dV=dV, failed=failed)


def backward_pass(
    expansion: Expansion,
    regularization: DynamicRegularization,
    options: SolverOptions,
) -> BackwardPassResult:
    """Compute the affine iLQR policy, cost-to-go, and expected decrease for one backward pass.

    Retries the reversed Riccati recursion under `lax.while_loop` while it fails and `rho` has
    not exceeded `options.bp_reg_max` (finding F: unlike Altro, this bounds the retry so a
    persistently indefinite `Quu` cannot hang). Regularization is increased once per retry and
    decreased exactly once at the end, mirroring `increaseregularization!` /
    `decreaseregularization!`. `K` and `d` are solved against the regularized `Quu_reg = Quu +
    rho*I` (finding G); the cost-to-go update uses the unregularized `Quu`/`Qux`, matching
    `Altro.backwardpass!`.
    """
    sweep0 = _sweep(expansion, regularization.rho)

    def cond(carry: tuple[_SweepResult, jax.Array, jax.Array]) -> jax.Array:
        sweep, rho, _drho = carry
        return sweep.failed & (rho <= options.bp_reg_max)

    def body(carry: tuple[_SweepResult, jax.Array, jax.Array]) -> tuple[_SweepResult, jax.Array, jax.Array]:
        _sweep_prev, rho, drho = carry
        new_reg = increase_regularization(DynamicRegularization(rho=rho, drho=drho), options)
        new_sweep = _sweep(expansion, new_reg.rho)
        return new_sweep, new_reg.rho, new_reg.drho

    final_sweep, final_rho, final_drho = jax.lax.while_loop(
        cond,
        body,
        (sweep0, regularization.rho, regularization.drho),
    )

    final_reg = decrease_regularization(DynamicRegularization(rho=final_rho, drho=final_drho), options)

    return BackwardPassResult(
        K=final_sweep.K,
        d=final_sweep.d,
        S_x=final_sweep.S_x,
        S_xx=final_sweep.S_xx,
        dV=final_sweep.dV,
        regularization=final_reg,
        failed=final_sweep.failed,
    )
