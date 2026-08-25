from dataclasses import dataclass, field
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from trajopt.costs.objective import Objective
from trajopt.dynamics.base import AbstractModel
from trajopt.expansions import Expansion
from trajopt.problem import MPCState, Problem
from trajopt.solvers.options import SolverOptions, SolverStats, TerminationStatus, to_solver_status
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z, _z_to_trajectory, parse_solver_initial_state
from trajopt.transcription.result import SolverStatus

_EMPTY = np.zeros(0, dtype=np.float64)


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


class RolloutResult(eqx.Module):
    """Closed-loop rollout output: the simulated trajectory and its first guard failure.

    Parameters
    ----------
    X : jax.Array
        Rolled-out states of shape (N, n).
    U : jax.Array
        Rolled-out controls of shape (N-1, m).
    failed : jax.Array
        Whether `‖x‖∞` or `‖u‖∞` exceeded `max_state_value` / `max_control_value`, or went
        NaN, at any knot, as a bool scalar.
    status : jax.Array
        `TerminationStatus.STATE_LIMIT` or `CONTROL_LIMIT` for the first knot that failed, or
        `UNSOLVED` if none did, as an int32 scalar (matching `Altro.rollout!`'s reuse of
        `UNSOLVED` to mean "no limit hit").
    """

    X: jax.Array
    U: jax.Array
    failed: jax.Array
    status: jax.Array


def rollout_closed_loop(  # noqa: PLR0913, PLR0917 -- model, nominal, gains, alpha, and options are all load-bearing
    model: AbstractModel,
    nominal: Trajectory,
    K: jax.Array,
    d: jax.Array,
    alpha: jax.Array | float,
    options: SolverOptions,
) -> RolloutResult:
    """Closed-loop rollout `u_k = ubar_k + K_k @ dx_k + alpha*d_k`, matching `Altro.rollout!`.

    `dx_k` is `model.state_diff(xbar_k, x_nom_k)`: the error between the trajectory being
    rolled out and `nominal` (Altro's `Z`, fixed for the whole line search), not the previous
    knot's state. A `lax.scan` cannot stop early on a guard violation the way `rollout!` does
    (reference §7.3 item 5), so every knot is always computed; `failed`/`status` latch onto the
    first knot that violates a guard and later knots' garbage values are meant to be discarded
    by the caller, not trusted.

    Parameters
    ----------
    model : AbstractModel
        Dynamics model; continuous models are discretized with RK4.
    nominal : Trajectory
        Reference trajectory `Z` supplying `X`, `U`, `t`, `dt` for the feedback law and guards'
        step timing. Only `X[0]` seeds the rollout; the rest of `nominal.X` feeds `state_diff`.
    K : jax.Array
        Feedback gains of shape `(N-1, m, ne)`.
    d : jax.Array
        Feedforward terms of shape `(N-1, m)`.
    alpha : jax.Array | float
        Line-search step length scaling the feedforward term.
    options : SolverOptions
        Supplies `max_state_value` and `max_control_value`.

    Returns
    -------
    RolloutResult
        The rolled-out `X`, `U`, and the first guard failure, if any.
    """
    discrete_model = model.discretize()
    X_nom, U_nom, t, dt = nominal.X, nominal.U, nominal.t, nominal.dt
    x0 = X_nom[0]

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        xbar_k, failed, status = carry
        x_nom_k, u_nom_k, K_k, d_k, t_k, dt_k = inputs

        dx_k = discrete_model.state_diff(xbar_k, x_nom_k)
        u_k = u_nom_k + K_k @ dx_k + alpha * d_k
        x_next = discrete_model.discrete_dynamics(xbar_k, u_k, t_k, dt_k)

        max_x = jnp.max(jnp.abs(x_next))
        state_bad = (max_x > options.max_state_value) | jnp.isnan(max_x)
        max_u = jnp.max(jnp.abs(u_k))
        control_bad = (max_u > options.max_control_value) | jnp.isnan(max_u)

        this_status = jnp.where(
            state_bad,
            jnp.int32(TerminationStatus.STATE_LIMIT),
            jnp.where(control_bad, jnp.int32(TerminationStatus.CONTROL_LIMIT), jnp.int32(TerminationStatus.UNSOLVED)),
        )
        new_failed = failed | state_bad | control_bad
        new_status = jnp.where(failed, status, this_status)

        return (x_next, new_failed, new_status), (x_next, u_k)

    init_carry = (x0, jnp.asarray(False), jnp.int32(TerminationStatus.UNSOLVED))  # noqa: FBT003 -- traced bool scalar
    (_, failed, status), (X_rest, U_out) = jax.lax.scan(
        step,
        init_carry,
        (X_nom[:-1], U_nom, K, d, t[:-1], dt),
    )
    X_out = jnp.concatenate([x0[None], X_rest], axis=0)
    return RolloutResult(X=X_out, U=U_out, failed=failed, status=status)


class ForwardPassResult(eqx.Module):
    """Output of the iLQR line search, matching `Altro.forwardpass!`.

    Parameters
    ----------
    trajectory : Trajectory
        The accepted `Z̄` on success, or the restored `Z` (`nominal`) when no step is taken
        (the expected-decrease-too-small exit, or line-search exhaustion).
    alpha : jax.Array
        The accepted step length, or 0 when no step is taken.
    J : jax.Array
        The accepted cost, `J_prev` when no step is taken, or NaN when the final cost still
        exceeds `J_prev` (`COST_INCREASE`).
    expected : jax.Array
        The last-computed expected decrease `-alpha*(dV[0] + alpha*dV[1])`.
    z : jax.Array
        The last-computed Armijo-style ratio `(J_prev - J) / expected`.
    ls_failed : jax.Array
        Whether the search exhausted `options.iterations_linesearch` without accepting a step.
    status : jax.Array
        `TerminationStatus.COST_INCREASE` if the final cost exceeds `J_prev`, else `UNSOLVED`.
    regularization : DynamicRegularization
        Regularization state after the search (increased on a no-step or exhaustion exit).
    """

    trajectory: Trajectory
    alpha: jax.Array
    J: jax.Array
    expected: jax.Array
    z: jax.Array
    ls_failed: jax.Array
    status: jax.Array
    regularization: DynamicRegularization


class _LineSearchCarry(NamedTuple):
    i: jax.Array
    alpha: jax.Array
    done: jax.Array
    ls_failed: jax.Array
    J: jax.Array
    z: jax.Array
    expected: jax.Array
    X: jax.Array
    U: jax.Array
    rho: jax.Array
    drho: jax.Array


def _line_search_step(  # noqa: PLR0913, PLR0917 -- one iteration needs the full line-search context
    carry: _LineSearchCarry,
    model: AbstractModel,
    obj: Objective,
    nominal: Trajectory,
    K: jax.Array,
    d: jax.Array,
    dV: jax.Array,
    J_prev: jax.Array,
    options: SolverOptions,
) -> _LineSearchCarry:
    """One iteration of `Altro.forwardpass!`'s line search loop, all four exits included.

    Reproduces the control flow exactly rather than as mutually exclusive branches: a rollout
    guard failure only decays `alpha` and skips everything else (finding J's "continue" quirk);
    otherwise cost, the expected decrease, and the Armijo-style ratio `z` are always
    recomputed, the no-step and iteration-exhaustion checks are independent (both can fire on
    the same iteration and each bumps regularization, per reference §7.3 item 3), and only
    acceptance or exhaustion (not a guard failure) ends the search.
    """
    reg = DynamicRegularization(rho=carry.rho, drho=carry.drho)
    rollout = rollout_closed_loop(model, nominal, K, d, carry.alpha, options)
    good = ~rollout.failed

    J_roll = obj.cost(Trajectory(X=rollout.X, U=rollout.U, t=nominal.t, dt=nominal.dt))
    expected_c = -carry.alpha * (dV[0] + carry.alpha * dV[1])

    no_step = good & (expected_c > 0) & (expected_c < options.expected_decrease_tolerance)
    z_c = jnp.where(no_step, jnp.inf, jnp.where(expected_c > 0, (J_prev - J_roll) / expected_c, -1.0))

    accept = good & (z_c >= options.line_search_lower_bound) & (z_c <= options.line_search_upper_bound)
    is_max_iter = carry.i + 1 == options.iterations_linesearch
    exhausted = good & ~accept & is_max_iter

    reg_after_no_step = jax.tree.map(
        lambda new, old: jnp.where(no_step, new, old), increase_regularization(reg, options), reg
    )
    reg_after_exhaustion = jax.tree.map(
        lambda new, old: jnp.where(exhausted, new, old),
        increase_regularization(reg_after_no_step, options),
        reg_after_no_step,
    )
    final_reg = DynamicRegularization(
        rho=jnp.where(exhausted, reg_after_exhaustion.rho + options.bp_reg_fp, reg_after_exhaustion.rho),
        drho=reg_after_exhaustion.drho,
    )

    should_exit = good & (no_step | accept | exhausted)

    new_alpha = jnp.where(
        should_exit, jnp.where(accept, carry.alpha, 0.0), carry.alpha * options.line_search_decrease_factor
    )
    new_J = jnp.where(should_exit, jnp.where(accept, J_roll, J_prev), carry.J)
    new_X = jnp.where(should_exit, jnp.where(accept, rollout.X, nominal.X), carry.X)
    new_U = jnp.where(should_exit, jnp.where(accept, rollout.U, nominal.U), carry.U)
    new_ls_failed = carry.ls_failed | exhausted
    new_done = carry.done | should_exit

    return _LineSearchCarry(
        i=carry.i + 1,
        alpha=new_alpha,
        done=new_done,
        ls_failed=new_ls_failed,
        J=new_J,
        z=jnp.where(good, z_c, carry.z),
        expected=jnp.where(good, expected_c, carry.expected),
        X=new_X,
        U=new_U,
        rho=jnp.where(good, final_reg.rho, carry.rho),
        drho=jnp.where(good, final_reg.drho, carry.drho),
    )


def forward_pass(  # noqa: PLR0913, PLR0917 -- model, objective, nominal, policy, dV, J_prev, reg, options all needed
    model: AbstractModel,
    obj: Objective,
    nominal: Trajectory,
    K: jax.Array,
    d: jax.Array,
    dV: jax.Array,
    J_prev: jax.Array,
    regularization: DynamicRegularization,
    options: SolverOptions,
) -> ForwardPassResult:
    """Line search over `alpha` for the affine policy `(K, d)`, matching `Altro.forwardpass!`.

    Runs `rollout_closed_loop` under `lax.while_loop`, up to `options.iterations_linesearch`
    times, halving `alpha` between attempts. See `_line_search_step` for the four exits (guard
    retry, no-step, acceptance, exhaustion) and reference §4.3 / §7.3 item 3 for why this is not
    a plain Armijo backtrack: the acceptance ratio `z = (J_prev - J) / expected` has both a
    lower and an upper bound. If the final cost still exceeds `J_prev` -- including the
    guard-exhaustion quirk where every rollout fails and `J` never leaves `Inf` (finding J) --
    the returned cost is NaN and `status` is `COST_INCREASE`; ticket 27's convergence check must
    treat that NaN as non-convergence, not as success.
    """

    def cond(carry: _LineSearchCarry) -> jax.Array:
        return (~carry.done) & (carry.i < options.iterations_linesearch)

    def body(carry: _LineSearchCarry) -> _LineSearchCarry:
        return _line_search_step(carry, model, obj, nominal, K, d, dV, J_prev, options)

    init = _LineSearchCarry(
        i=jnp.int32(0),
        alpha=jnp.asarray(1.0, dtype=J_prev.dtype),
        done=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar
        ls_failed=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar
        J=jnp.asarray(jnp.inf, dtype=J_prev.dtype),
        z=jnp.asarray(jnp.inf, dtype=J_prev.dtype),
        expected=jnp.asarray(jnp.inf, dtype=J_prev.dtype),
        X=nominal.X,
        U=nominal.U,
        rho=regularization.rho,
        drho=regularization.drho,
    )

    final = jax.lax.while_loop(cond, body, init)

    cost_increase = J_prev < final.J
    reported_J = jnp.where(cost_increase, jnp.nan, final.J)
    status = jnp.where(cost_increase, jnp.int32(TerminationStatus.COST_INCREASE), jnp.int32(TerminationStatus.UNSOLVED))

    return ForwardPassResult(
        trajectory=Trajectory(X=final.X, U=final.U, t=nominal.t, dt=nominal.dt),
        alpha=final.alpha,
        J=reported_J,
        expected=final.expected,
        z=final.z,
        ls_failed=final.ls_failed,
        status=status,
        regularization=DynamicRegularization(rho=final.rho, drho=final.drho),
    )


def _per_knot_gradient(d: jax.Array, U: jax.Array) -> jax.Array:
    """Per-knot normalized feedforward magnitude `max_i |d_k[i]| / (|u_k[i]| + 1)`, shape `(N-1,)`."""
    return jnp.max(jnp.abs(d) / (jnp.abs(U) + 1.0), axis=-1)


def _feedforward_gradient(d: jax.Array, U: jax.Array) -> jax.Array:
    """Compute the normalized feedforward magnitude `mean_k max_i |d_k[i]| / (|u_k[i]| + 1)`.

    Not a cost gradient: it is the primal optimality residual `gradient_tolerance` compares
    against, computed on the feedforward term `d` from the backward pass paired with the
    *accepted* controls `U` (ticket 27).
    """
    return jnp.mean(_per_knot_gradient(d, U))


class ILQRCarry(NamedTuple):
    """Traced `lax.while_loop` state for one iLQR solve; see `ilqr_solve` for field meaning."""

    i: jax.Array
    trajectory: Trajectory
    regularization: DynamicRegularization
    stats: SolverStats
    done: jax.Array
    status: jax.Array


def _ilqr_step(
    carry: ILQRCarry,
    problem: Problem,
    options: SolverOptions,
    cost_tolerance: jax.Array,
    gradient_tolerance: jax.Array,
) -> ILQRCarry:
    """One iLQR iteration: cost, expansions, backward pass, forward pass, accept, record, check.

    Follows reference §4.1's order exactly: `J_prev` on the trajectory carried in, then
    expansions, backward pass, forward pass, an unconditional accept, then `dJ` and `gradient`
    computed on the accepted trajectory. `dJ_zero_counter` compares the raw (possibly NaN) `dJ`
    to exactly `0.0`, matching Julia's `dJ ≈ 0` against a literal zero (default `isapprox`
    tolerances make that an exact-equality test); `NaN == 0.0` is `False`, so a failed forward
    pass neither increments the counter nor satisfies the cost-convergence criterion (ticket 27:
    reproduced by letting IEEE NaN comparisons propagate, not by special-casing NaN).

    `cost_tolerance`/`gradient_tolerance` are traced scalars (ticket 29's effective intermediate
    or final AL tolerance pair, or plain `options.cost_tolerance`/`gradient_tolerance` for a bare
    iLQR solve), passed straight through to `_evaluate_convergence`.
    """
    traj = carry.trajectory
    obj = problem.obj

    J_prev = obj.cost(traj)
    expansion = problem.dynamics_expansion(traj) + problem.cost_expansion(traj)
    bp = backward_pass(expansion, carry.regularization, options)
    fp = forward_pass(problem.model, obj, traj, bp.K, bp.d, bp.dV, J_prev, bp.regularization, options)

    new_traj = fp.trajectory
    dJ = J_prev - fp.J
    grad = _feedforward_gradient(bp.d, new_traj.U)

    dJ_zero = dJ == 0.0
    new_dJ_counter = jnp.where(dJ_zero, carry.stats.dJ_zero_counter + 1, jnp.int32(0))

    idx = carry.i
    iter_num = carry.i + 1
    stats = carry.stats
    new_stats = SolverStats(
        iterations=iter_num,
        cost=stats.cost.at[idx].set(fp.J),
        dJ=stats.dJ.at[idx].set(dJ),
        c_max=stats.c_max.at[idx].set(jnp.zeros((), dtype=stats.c_max.dtype)),
        gradient=stats.gradient.at[idx].set(grad),
        penalty_max=stats.penalty_max.at[idx].set(jnp.zeros((), dtype=stats.penalty_max.dtype)),
        dJ_zero_counter=new_dJ_counter,
        ls_failed=fp.ls_failed,
    )

    status = _evaluate_convergence(
        dJ, grad, fp.ls_failed, fp.J, new_dJ_counter, iter_num, cost_tolerance, gradient_tolerance, options
    )
    done = status != jnp.int32(TerminationStatus.UNSOLVED)

    return ILQRCarry(
        i=iter_num,
        trajectory=new_traj,
        regularization=fp.regularization,
        stats=new_stats,
        done=done,
        status=status,
    )


def _evaluate_convergence(  # noqa: PLR0913, PLR0917 -- one per Altro's evaluate_convergence input
    dJ: jax.Array,
    grad: jax.Array,
    ls_failed: jax.Array,
    J: jax.Array,
    dJ_zero_counter: jax.Array,
    iter_num: jax.Array,
    cost_tolerance: jax.Array,
    gradient_tolerance: jax.Array,
    options: SolverOptions,
) -> jax.Array:
    """Decide whether an iLQR iteration converged, matching `Altro.evaluate_convergence`.

    First-match-wins in Altro's declared order (reference §4.4, ticket 27): the cost criterion
    (needing all three of `0 <= dJ < cost_tolerance`, `grad < gradient_tolerance`, and
    `not ls_failed`), then max iterations, then `dJ_zero_counter > dJ_counter_limit`, then max
    cost. A NaN `dJ` or `J` makes every comparison touching it `False` under IEEE semantics, so
    it neither converges nor is mistaken for `MAXIMUM_COST` -- it just fails to match any exit
    (ticket 27: reproduced by letting NaN propagate rather than special-casing it).

    `cost_tolerance` and `gradient_tolerance` are passed as traced scalars rather than read off
    `options` (ticket 29: AL's per-outer-iteration intermediate/final tolerance pair is computed
    in its loop carry and threaded through here, so `options` itself is never traced).

    Returns
    -------
    jax.Array
        `TerminationStatus` ordinal as an int32 scalar, or `UNSOLVED` if no exit fired.
    """
    cost_converged = (dJ >= 0.0) & (dJ < cost_tolerance) & (grad < gradient_tolerance) & (~ls_failed)
    max_iters_hit = iter_num >= options.iterations
    no_progress = dJ_zero_counter > options.dJ_counter_limit
    max_cost_hit = options.max_cost_value < J

    return jnp.where(
        cost_converged,
        jnp.int32(TerminationStatus.SOLVE_SUCCEEDED),
        jnp.where(
            max_iters_hit,
            jnp.int32(TerminationStatus.MAX_ITERATIONS),
            jnp.where(
                no_progress,
                jnp.int32(TerminationStatus.NO_PROGRESS),
                jnp.where(
                    max_cost_hit,
                    jnp.int32(TerminationStatus.MAXIMUM_COST),
                    jnp.int32(TerminationStatus.UNSOLVED),
                ),
            ),
        ),
    )


def ilqr_solve(
    problem: Problem,
    trajectory: Trajectory,
    options: SolverOptions,
    *,
    cost_tolerance: jax.Array | float | None = None,
    gradient_tolerance: jax.Array | float | None = None,
) -> tuple[Trajectory, SolverStats, jax.Array]:
    """Traced iLQR core, matching `Altro.iLQRSolver`'s `initialize!` + `solve!` loop.

    A pure `(problem, trajectory, options) -> (trajectory, stats, status)` function built from
    one `lax.while_loop`, jittable and vmappable end to end with `options` static. `trajectory`
    is the warm-start guess; only its `X[0]`, `U`, `t`, `dt` are used, since `initialize!` does
    an **open-loop** rollout (`problem.model.rollout`) rather than reusing any cached gains
    (ticket 27 discards Altro's `closed_loop_initial_rollout` option and its docstring claim).

    Parameters
    ----------
    problem : Problem
        Supplies the model, objective, and the `dynamics_expansion` / `cost_expansion` methods
        the loop body delegates to.
    trajectory : Trajectory
        Warm-start guess; only `X[0]`, `U`, `t`, `dt` are read.
    options : SolverOptions
        Static solve configuration; must not be traced.
    cost_tolerance : jax.Array | float | None, optional
        Overrides `options.cost_tolerance` when given, as a possibly-traced scalar (ticket 29:
        AL computes its per-outer-iteration effective tolerance in its loop carry and passes it
        here rather than rebuilding `options` with a traced field). Defaults to None, meaning
        `options.cost_tolerance`.
    gradient_tolerance : jax.Array | float | None, optional
        Same as `cost_tolerance`, for `options.gradient_tolerance`. Defaults to None.

    Returns
    -------
    tuple[Trajectory, SolverStats, jax.Array]
        The accepted trajectory at exit, the stats history (buffers sized `options.iterations`,
        untrimmed), and the exit `TerminationStatus` ordinal as an int32 scalar.
    """
    init_traj = problem.model.rollout(trajectory)
    cost_tol = jnp.asarray(options.cost_tolerance if cost_tolerance is None else cost_tolerance, dtype=jnp.float64)
    grad_tol = jnp.asarray(
        options.gradient_tolerance if gradient_tolerance is None else gradient_tolerance, dtype=jnp.float64
    )

    init_carry = ILQRCarry(
        i=jnp.int32(0),
        trajectory=init_traj,
        regularization=DynamicRegularization.initial(options),
        stats=SolverStats.create(options),
        done=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
        status=jnp.int32(TerminationStatus.UNSOLVED),
    )

    def cond(carry: ILQRCarry) -> jax.Array:
        return (~carry.done) & (carry.i < options.iterations)

    def body(carry: ILQRCarry) -> ILQRCarry:
        return _ilqr_step(carry, problem, options, cost_tol, grad_tol)

    final = jax.lax.while_loop(cond, body, init_carry)
    return final.trajectory, final.stats, final.status


def build_warm_start(problem: Problem, state: MPCState) -> tuple[Problem, Trajectory]:
    """Build the eager warm-start trajectory and goal-overridden problem from `state`.

    Shared by `ILQR.solve` and `AL.solve` (ticket 29 wraps ticket 27's `.solve()` boundary):
    both parse `state.Z` into `(X, U)`, build the absolute time grid from `state.t0`/`state.dt`,
    and override `problem`'s objective goal from `state.xf` when the objective regulates to a
    runtime goal.

    Parameters
    ----------
    problem : Problem
        Problem to warm-start; its `obj` may be goal-overridden in the returned copy.
    state : MPCState
        Per-step state supplying the flat primal `Z`, `t0`, `dt`, and optional runtime goal `xf`.

    Returns
    -------
    tuple[Problem, Trajectory]
        `(problem_eff, init_traj)`: `problem` with its goal overridden if applicable, and the
        warm-start trajectory built from `state.Z`.
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    _x0_arr, t0_arr, dt_arr, xf_val, z0 = parse_solver_initial_state(state)
    assert z0 is not None  # noqa: S101 -- MPCState.Z is never None; the shared helper's type is just loose
    dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))
    t_arr = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])
    X0, U0 = _z_to_trajectory(z0, N, n, m)
    init_traj = Trajectory(X=X0, U=U0, t=t_arr, dt=dt_arr)

    problem_eff = problem
    if xf_val is not None and problem.obj.regulates_to_goal:
        problem_eff = eqx.tree_at(lambda p: p.obj, problem, problem.obj.with_goal(xf_val))

    return problem_eff, init_traj


def _trim_stats(stats: SolverStats, n_iter: int) -> SolverStats:
    """Slice a finished solve's fixed-size stats buffers down to the completed iteration count."""
    return SolverStats(
        iterations=stats.iterations,
        cost=stats.cost[:n_iter],
        dJ=stats.dJ[:n_iter],
        c_max=stats.c_max[:n_iter],
        gradient=stats.gradient[:n_iter],
        penalty_max=stats.penalty_max[:n_iter],
        dJ_zero_counter=stats.dJ_zero_counter,
        ls_failed=stats.ls_failed,
    )


class ILQRResult(NamedTuple):
    """Result of a native iLQR solve, satisfying the `SolverResult` protocol.

    Parameters
    ----------
    trajectory : Trajectory
        Optimal state and control trajectory.
    success : bool
        Whether the core exited with `TerminationStatus.SOLVE_SUCCEEDED`.
    status : int
        `TerminationStatus` ordinal the traced core exited with.
    message : str
        `TerminationStatus` member name -- the precise Altro exit reason, kept for diagnostics.
    solver_status : SolverStatus
        `status` mapped through `to_solver_status`'s table; the authoritative public status
        `Problem.solve` uses for `MPCState.status`, rather than guessing from `message`.
    cost : float
        Final objective value.
    Z : jax.Array
        Optimal flat primal vector.
    info : dict[str, Any]
        Holds the trimmed `SolverStats` history under `"stats"`.
    iterations : int, optional
        Number of completed iLQR iterations. Defaults to 0.
    constraint_violation : float, optional
        Always 0.0: an unconstrained iLQR has no constraints. Defaults to 0.0.
    lam : np.ndarray, optional
        Always empty: an unconstrained iLQR has no duals. Defaults to empty.
    mu : np.ndarray, optional
        Always empty: an unconstrained iLQR has no duals. Defaults to empty.
    """

    trajectory: Trajectory
    success: bool
    status: int
    message: str
    solver_status: SolverStatus
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    iterations: int = 0
    constraint_violation: float = 0.0
    lam: np.ndarray = _EMPTY
    mu: np.ndarray = _EMPTY


@dataclass(frozen=True)
class ILQR:
    """Native iLQR solver backend, satisfying the `Solver` protocol for an unconstrained problem.

    A thin eager wrapper over the traced `ilqr_solve` core (ticket 27): `.solve()` builds the
    warm-start trajectory from `state`, calls the jitted core, then converts the traced status
    int and stats buffers into `success` / `message` / `info` at the boundary -- work that
    cannot happen inside a trace. Swapping `ILQR()` for `Ipopt()` in `problem.solve(state,
    solver=...)` is then a one-word change.

    Parameters
    ----------
    options : SolverOptions, optional
        Static solve configuration. Defaults to `SolverOptions()`.
    """

    options: SolverOptions = field(default_factory=SolverOptions)

    def solve(self, problem: Problem, state: MPCState) -> ILQRResult:
        """Run the traced iLQR core from `state`'s warm-start trajectory and boundary-convert the result."""
        problem_eff, init_traj = build_warm_start(problem, state)

        final_traj, stats, status_int = ilqr_solve(problem_eff, init_traj, self.options)

        status = TerminationStatus(int(status_int))
        n_iter = int(stats.iterations)

        return ILQRResult(
            trajectory=final_traj,
            success=status == TerminationStatus.SOLVE_SUCCEEDED,
            status=int(status_int),
            message=status.name,
            solver_status=to_solver_status(status),
            cost=float(problem_eff.obj.cost(final_traj)),
            Z=_trajectory_to_z(final_traj.X, final_traj.U),
            info={"stats": _trim_stats(stats, n_iter)},
            iterations=n_iter,
        )
