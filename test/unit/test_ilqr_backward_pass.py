import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.expansions import Expansion
from trajopt.solvers.ilqr import (
    DynamicRegularization,
    backward_pass,
    decrease_regularization,
    increase_regularization,
)
from trajopt.solvers.options import SolverOptions


def _lqr_expansion(N: int, ne: int, m: int, *, seed: int = 0) -> tuple[Expansion, list, list, list, list]:
    """Build a time-varying, regulation-about-origin (q=r=0) LQ expansion for testing."""
    rng = np.random.default_rng(seed)
    As, Bs, Qs, Rs = [], [], [], []
    for _ in range(N - 1):
        A = rng.normal(size=(ne, ne)) * 0.3 + np.eye(ne)
        B = rng.normal(size=(ne, m)) * 0.3
        Q = rng.normal(size=(ne, ne))
        Q = Q @ Q.T + np.eye(ne)
        R = rng.normal(size=(m, m))
        R = R @ R.T + np.eye(m)
        As.append(A)
        Bs.append(B)
        Qs.append(Q)
        Rs.append(R)
    Qf = rng.normal(size=(ne, ne))
    Qf = Qf @ Qf.T + np.eye(ne)

    exp = Expansion(
        A=jnp.asarray(np.stack(As)),
        B=jnp.asarray(np.stack(Bs)),
        q=jnp.zeros((N, ne)),
        r=jnp.zeros((N - 1, m)),
        Q=jnp.asarray(np.concatenate([np.stack(Qs), Qf[None]], axis=0)),
        R=jnp.asarray(np.stack(Rs)),
        H=jnp.zeros((N - 1, m, ne)),
    )
    return exp, As, Bs, Qs, [*Rs, Qf]


def _analytic_lqr_gains(
    As: list[np.ndarray],
    Bs: list[np.ndarray],
    Qs: list[np.ndarray],
    Rs: list[np.ndarray],
    Qf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference time-varying discrete Riccati recursion, K = -inv(Quu) @ Qux, unregularized."""
    N = len(As) + 1
    S = Qf
    Ks: list[np.ndarray] = [np.zeros_like(Bs[0].T)] * (N - 1)
    for k in range(N - 2, -1, -1):
        A, B, Q, R = As[k], Bs[k], Qs[k], Rs[k]
        Quu = R + B.T @ S @ B
        Qux = B.T @ S @ A
        Qxx = Q + A.T @ S @ A
        K = -np.linalg.solve(Quu, Qux)
        S = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        S = 0.5 * (S + S.T)
        Ks[k] = K
    return np.stack(Ks), S


def test_backward_pass_lqr_matches_analytic_gains() -> None:
    """At rho = 0 the returned K matches the closed-form time-varying LQR gains to 1e-10."""
    N, ne, m = 6, 3, 2
    exp, As, Bs, Qs, RsQf = _lqr_expansion(N, ne, m)
    Rs, Qf = RsQf[:-1], RsQf[-1]

    options = SolverOptions()
    reg = DynamicRegularization.initial(options)
    result = backward_pass(exp, reg, options)

    K_ref, S_ref = _analytic_lqr_gains(As, Bs, Qs, Rs, Qf)

    np.testing.assert_allclose(np.asarray(result.K), K_ref, atol=1e-10)
    np.testing.assert_allclose(np.asarray(result.d), np.zeros((N - 1, m)), atol=1e-10)
    np.testing.assert_allclose(np.asarray(result.S_xx[0]), S_ref, atol=1e-8)
    assert not bool(result.failed)


def test_backward_pass_indefinite_quu_triggers_retry() -> None:
    """An indefinite Quu forces the rho-retry loop; increases follow the 1.6x state machine."""
    N, ne, m = 4, 2, 1
    rng = np.random.default_rng(1)
    A = rng.normal(size=(ne, ne)) * 0.1 + np.eye(ne)
    Q = np.eye(ne)
    # B = 0 makes Quu = R at every knot, regardless of S_xx, so indefiniteness is uniform.
    exp = Expansion(
        A=jnp.asarray(np.stack([A] * (N - 1))),
        B=jnp.zeros((N - 1, ne, m)),
        q=jnp.zeros((N, ne)),
        r=jnp.zeros((N - 1, m)),
        Q=jnp.asarray(np.stack([Q] * N)),
        R=jnp.asarray(np.stack([-np.eye(m)] * (N - 1))),
        H=jnp.zeros((N - 1, m, ne)),
    )

    options = SolverOptions()
    reg = DynamicRegularization.initial(options)
    result = backward_pass(exp, reg, options)

    assert not bool(result.failed)

    expected_reg = reg
    n_increases = 0
    while True:
        quu_reg = -1.0 + float(expected_reg.rho)
        if quu_reg > 0:
            break
        expected_reg = increase_regularization(expected_reg, options)
        n_increases += 1
    expected_final = decrease_regularization(expected_reg, options)

    assert n_increases > 0
    np.testing.assert_allclose(float(result.regularization.rho), float(expected_final.rho), rtol=1e-10)
    np.testing.assert_allclose(float(result.regularization.drho), float(expected_final.drho), rtol=1e-10)


def test_backward_pass_retry_bounded_terminates() -> None:
    """A permanently indefinite Quu exhausts bp_reg_max and returns rather than hanging."""
    N, ne, m = 3, 2, 1
    exp = Expansion(
        A=jnp.asarray(np.stack([np.eye(ne)] * (N - 1))),
        B=jnp.zeros((N - 1, ne, m)),
        q=jnp.zeros((N, ne)),
        r=jnp.zeros((N - 1, m)),
        Q=jnp.asarray(np.stack([np.eye(ne)] * N)),
        R=jnp.asarray(np.stack([-1.0e12 * np.eye(m)] * (N - 1))),
        H=jnp.zeros((N - 1, m, ne)),
    )

    options = SolverOptions(bp_reg_max=1.0)
    reg = DynamicRegularization.initial(options)

    jitted = jax.jit(functools.partial(backward_pass, options=options))
    result = jitted(exp, reg)

    assert bool(result.failed)
    assert float(result.regularization.rho) > options.bp_reg_max


def test_backward_pass_uses_regularized_quu_for_gains() -> None:
    """K and d at rho > 0 differ from the unregularized solve, distinguishing finding G."""
    N, ne, m = 4, 2, 2
    exp, _As, _Bs, _Qs, _RsQf = _lqr_expansion(N, ne, m, seed=2)

    options = SolverOptions()
    zero_reg = DynamicRegularization.initial(options)
    reg_rho = DynamicRegularization(rho=jnp.asarray(5.0), drho=jnp.asarray(1.0))

    result_unreg = backward_pass(exp, zero_reg, options)
    result_reg = backward_pass(exp, reg_rho, options)

    assert not bool(result_unreg.failed)
    assert not bool(result_reg.failed)
    assert not np.allclose(np.asarray(result_unreg.K), np.asarray(result_reg.K))

    # rho=5 shouldn't need any retries here, so the returned rho is just decrease(5).
    expected_reg = decrease_regularization(reg_rho, options)
    np.testing.assert_allclose(float(result_reg.regularization.rho), float(expected_reg.rho))


def test_backward_pass_is_jittable_with_static_options() -> None:
    """backward_pass runs unchanged under jax.jit with SolverOptions held static."""
    N, ne, m = 5, 2, 1
    exp, *_ = _lqr_expansion(N, ne, m, seed=3)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    jitted = jax.jit(functools.partial(backward_pass, options=options))
    eager = backward_pass(exp, reg, options)
    traced = jitted(exp, reg)

    np.testing.assert_allclose(np.asarray(traced.K), np.asarray(eager.K))
    np.testing.assert_allclose(np.asarray(traced.d), np.asarray(eager.d))


@pytest.mark.parametrize(
    ("rho", "drho"),
    [(0.0, 1.0), (1e-6, 2.0), (100.0, 5.0)],
)
def test_increase_decrease_regularization_state_machine(rho: float, drho: float) -> None:
    """increase/decrease match Altro's rho = max(rho*drho, rho_min) / drho update exactly."""
    options = SolverOptions()
    reg = DynamicRegularization(rho=jnp.asarray(rho), drho=jnp.asarray(drho))

    increased = increase_regularization(reg, options)
    expected_drho_up = max(drho * options.bp_reg_increase_factor, options.bp_reg_increase_factor)
    expected_rho_up = max(rho * expected_drho_up, options.bp_reg_min)
    assert float(increased.drho) == pytest.approx(expected_drho_up)
    assert float(increased.rho) == pytest.approx(expected_rho_up)

    decreased = decrease_regularization(reg, options)
    expected_drho_down = min(drho / options.bp_reg_increase_factor, 1.0 / options.bp_reg_increase_factor)
    expected_rho_down = max(options.bp_reg_min, rho * expected_drho_down)
    assert float(decreased.drho) == pytest.approx(expected_drho_down)
    assert float(decreased.rho) == pytest.approx(expected_rho_down)


def test_decrease_regularization_runs_once_per_backward_pass() -> None:
    """A successful (no-retry) backward pass applies decrease exactly once to the input rho."""
    N, ne, m = 4, 2, 1
    exp, *_ = _lqr_expansion(N, ne, m, seed=4)
    options = SolverOptions()
    reg = DynamicRegularization(rho=jnp.asarray(0.1), drho=jnp.asarray(1.2))

    result = backward_pass(exp, reg, options)
    assert not bool(result.failed)

    expected = decrease_regularization(reg, options)
    np.testing.assert_allclose(float(result.regularization.rho), float(expected.rho))
    np.testing.assert_allclose(float(result.regularization.drho), float(expected.drho))
