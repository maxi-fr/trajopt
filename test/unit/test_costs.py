"""Unit tests for cost functions and stacked objectives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs import (
    DiagonalCost,
    GenericCost,
    LQRCost,
    LQRObjective,
    Objective,
    QuadraticCost,
    TrackingObjective,
    cost,
    update_reference,
)
from trajopt.trajectory import Trajectory


def test_diagonal_cost_evaluation() -> None:
    n, m = 4, 2
    Q = jnp.array([1.0, 2.0, 3.0, 4.0])
    R = jnp.array([0.5, 1.5])
    q = jnp.array([0.1, -0.2, 0.3, -0.4])
    r = jnp.array([-0.1, 0.2])
    c = 0.75

    dcost = DiagonalCost(Q=Q, R=R, q=q, r=r, c=c)
    assert dcost.n == n
    assert dcost.m == m
    assert dcost.is_diag
    assert dcost.is_blockdiag

    x = jnp.array([1.0, -1.0, 2.0, -2.0])
    u = jnp.array([0.5, -0.5])

    # 1. Scalar evaluation
    val = dcost.evaluate(x, u)
    expected_val = 0.5 * jnp.sum(Q * x**2) + 0.5 * jnp.sum(R * u**2) + jnp.dot(q, x) + jnp.dot(r, u) + c
    np.testing.assert_allclose(val, expected_val, rtol=1e-14, atol=1e-14)

    # 2. Gradient
    grad = dcost.gradient(x, u)
    expected_gx = Q * x + q
    expected_gu = R * u + r
    expected_grad = jnp.concatenate([expected_gx, expected_gu])
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-14, atol=1e-14)

    # 3. Hessian
    hess = dcost.hessian(x, u)
    expected_hess = jnp.diag(jnp.concatenate([Q, R]))
    np.testing.assert_allclose(hess, expected_hess, rtol=1e-14, atol=1e-14)

    # 4. Inversion
    dcost_inv = dcost.invert()
    np.testing.assert_allclose(dcost_inv.Q, 1.0 / Q, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(dcost_inv.R, 1.0 / R, rtol=1e-14, atol=1e-14)
    h_inv = dcost.hessian_inverse(x, u)
    np.testing.assert_allclose(h_inv, jnp.diag(jnp.concatenate([1.0 / Q, 1.0 / R])), rtol=1e-14, atol=1e-14)


def test_diagonal_cost_terminal() -> None:
    n = 3
    Q = jnp.array([2.0, 4.0, 6.0])
    q = jnp.array([-0.5, 0.5, -1.0])
    c = 1.25

    dterm = DiagonalCost(Q=Q, q=q, c=c, terminal=True)
    assert dterm.terminal
    assert dterm.n == n

    x = jnp.array([0.5, -1.5, 2.5])
    val = dterm.evaluate(x)
    expected_val = 0.5 * jnp.sum(Q * x**2) + jnp.dot(q, x) + c
    np.testing.assert_allclose(val, expected_val, rtol=1e-14, atol=1e-14)

    grad = dterm.gradient(x)
    expected_grad = Q * x + q
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-14, atol=1e-14)

    hess = dterm.hessian(x)
    expected_hess = jnp.diag(Q)
    np.testing.assert_allclose(hess, expected_hess, rtol=1e-14, atol=1e-14)

    term_inv = dterm.invert()
    np.testing.assert_allclose(term_inv.Q, 1.0 / Q, rtol=1e-14, atol=1e-14)


def test_quadratic_cost_dense_and_cross_coupling() -> None:
    n, m = 3, 2
    Q = jnp.array(
        [
            [2.0, 0.5, 0.1],
            [0.5, 3.0, 0.2],
            [0.1, 0.2, 4.0],
        ]
    )
    R = jnp.array(
        [
            [1.0, 0.2],
            [0.2, 2.0],
        ]
    )
    H = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.1],
        ]
    )
    q = jnp.array([0.1, -0.2, 0.3])
    r = jnp.array([-0.4, 0.5])
    c = 2.0

    qcost = QuadraticCost(Q=Q, R=R, H=H, q=q, r=r, c=c)
    assert qcost.n == n
    assert qcost.m == m
    assert not qcost.is_blockdiag

    x = jnp.array([1.0, -0.5, 0.25])
    u = jnp.array([-0.3, 0.7])

    # 1. Scalar evaluate
    val = qcost.evaluate(x, u)
    expected_val = 0.5 * x @ (Q @ x) + 0.5 * u @ (R @ u) + u @ (H @ x) + q @ x + r @ u + c
    np.testing.assert_allclose(val, expected_val, rtol=1e-14, atol=1e-14)

    # 2. Gradient
    grad = qcost.gradient(x, u)
    expected_gx = Q @ x + q + H.T @ u
    expected_gu = R @ u + r + H @ x
    expected_grad = jnp.concatenate([expected_gx, expected_gu])
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-14, atol=1e-14)

    # 3. Hessian
    hess = qcost.hessian(x, u)
    expected_hess = jnp.block([[Q, H.T], [H, R]])
    np.testing.assert_allclose(hess, expected_hess, rtol=1e-14, atol=1e-14)

    # 4. Inversion
    qcost_inv = qcost.invert()
    G_inv = jnp.linalg.inv(expected_hess)
    np.testing.assert_allclose(qcost_inv.Q, G_inv[:n, :n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qcost_inv.R, G_inv[n:, n:], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qcost_inv.H, G_inv[n:, :n], rtol=1e-12, atol=1e-12)


def test_quadratic_cost_block_diagonal_shortcut() -> None:
    Q = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    R = jnp.array([[4.0, 0.5], [0.5, 1.0]])

    qcost = QuadraticCost(Q=Q, R=R)
    assert qcost.is_blockdiag

    qcost_inv = qcost.invert()
    np.testing.assert_allclose(qcost_inv.Q, jnp.linalg.inv(Q), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(qcost_inv.R, jnp.linalg.inv(R), rtol=1e-14, atol=1e-14)


def test_lqr_cost_helper() -> None:
    Q_diag = jnp.array([1.0, 2.0, 3.0])
    R_diag = jnp.array([0.5, 1.5])
    xf = jnp.array([1.0, -1.0, 2.0])
    uf = jnp.array([0.2, -0.3])

    # Diagonal LQR cost
    lqr_diag = LQRCost(Q_diag, R_diag, xf, uf)
    assert isinstance(lqr_diag, DiagonalCost)
    np.testing.assert_allclose(lqr_diag.q, -Q_diag * xf, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(lqr_diag.r, -R_diag * uf, rtol=1e-14, atol=1e-14)
    expected_c = 0.5 * jnp.sum(Q_diag * xf**2) + 0.5 * jnp.sum(R_diag * uf**2)
    np.testing.assert_allclose(lqr_diag.c, expected_c, rtol=1e-14, atol=1e-14)

    # Dense LQR cost
    Q_dense = jnp.diag(Q_diag)
    R_dense = jnp.diag(R_diag)
    lqr_dense = LQRCost(Q_dense, R_dense, xf, uf)
    assert isinstance(lqr_dense, QuadraticCost)
    np.testing.assert_allclose(lqr_dense.q, -Q_dense @ xf, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(lqr_dense.r, -R_dense @ uf, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(lqr_dense.c, expected_c, rtol=1e-14, atol=1e-14)


def test_generic_cost() -> None:
    n, m = 2, 1

    def stage_fn(x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del t
        return jnp.sin(x[0]) ** 2 + jnp.cos(x[1]) ** 2 + u[0] ** 4

    gcost = GenericCost(cost_fn=stage_fn, n=n, m=m)

    x = jnp.array([0.5, -0.3])
    u = jnp.array([0.8])

    # 1. Evaluate
    val = gcost.evaluate(x, u)
    expected_val = jnp.sin(0.5) ** 2 + jnp.cos(-0.3) ** 2 + 0.8**4
    np.testing.assert_allclose(val, expected_val, rtol=1e-14, atol=1e-14)

    # 2. Gradient via AD
    grad = gcost.gradient(x, u)
    gx0 = 2.0 * jnp.sin(0.5) * jnp.cos(0.5)
    gx1 = -2.0 * jnp.cos(-0.3) * jnp.sin(-0.3)
    gu0 = 4.0 * 0.8**3
    expected_grad = jnp.array([gx0, gx1, gu0])
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-12, atol=1e-12)

    # 3. Hessian via AD
    hess = gcost.hessian(x, u)
    h00 = 2.0 * (jnp.cos(0.5) ** 2 - jnp.sin(0.5) ** 2)
    h11 = -2.0 * (jnp.cos(-0.3) ** 2 - jnp.sin(-0.3) ** 2)
    h22 = 12.0 * 0.8**2
    expected_hess = jnp.diag(jnp.array([h00, h11, h22]))
    np.testing.assert_allclose(hess, expected_hess, rtol=1e-12, atol=1e-12)


def test_lqr_objective_stacked_and_cost_evaluation() -> None:
    n, m, N = 3, 2, 11
    Q = jnp.array([1.0, 2.0, 3.0])
    R = jnp.array([0.1, 0.2])
    Qf = jnp.array([10.0, 20.0, 30.0])
    xf = jnp.array([1.0, 0.0, -1.0])
    uf = jnp.array([0.5, -0.5])

    obj = LQRObjective(Q, R, Qf, xf, N, uf=uf)
    assert len(obj) == N
    assert obj.N == N
    assert obj.n == n
    assert obj.m == m
    assert obj.is_diag

    # Verify stacked shapes
    assert obj.Q.shape == (N - 1, n)
    assert obj.R.shape == (N - 1, m)
    assert obj.q.shape == (N - 1, n)
    assert obj.r.shape == (N - 1, m)
    assert obj.c.shape == (N - 1,)
    assert obj.Qf.shape == (n,)
    assert obj.qf.shape == (n,)
    assert obj.cf.shape == ()

    # Create dummy trajectory
    rng = np.random.default_rng(42)
    X = jnp.array(rng.standard_normal((N, n)))
    U = jnp.array(rng.standard_normal((N - 1, m)))
    t = jnp.linspace(0.0, 1.0, N)
    dt = jnp.diff(t)
    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    # Evaluate cost on trajectory
    total_cost = cost(obj, traj)

    # Calculate expected sum
    expected_stage = 0.0
    for k in range(N - 1):
        xk = X[k]
        uk = U[k]
        expected_stage += 0.5 * jnp.sum(Q * (xk - xf) ** 2) + 0.5 * jnp.sum(R * (uk - uf) ** 2)
    expected_term = 0.5 * jnp.sum(Qf * (X[-1] - xf) ** 2)
    expected_total = expected_stage + expected_term

    np.testing.assert_allclose(total_cost, expected_total, rtol=1e-14, atol=1e-14)

    # Test JIT compilation of cost
    jit_cost = jax.jit(cost)
    jit_val = jit_cost(obj, traj)
    np.testing.assert_allclose(jit_val, expected_total, rtol=1e-14, atol=1e-14)


def test_tracking_objective_and_update_reference() -> None:
    n, m, N = 2, 1, 10
    Q = jnp.array([2.0, 3.0])
    R = jnp.array([0.5])
    Qf = jnp.array([20.0, 30.0])

    rng = np.random.default_rng(123)
    X_ref1 = jnp.array(rng.standard_normal((N, n)))
    U_ref1 = jnp.array(rng.standard_normal((N - 1, m)))
    t1 = jnp.linspace(0.0, 1.0, N)
    dt1 = jnp.diff(t1)
    traj_ref1 = Trajectory(X=X_ref1, U=U_ref1, t=t1, dt=dt1)

    obj = TrackingObjective(Q=Q, R=R, trajectory=traj_ref1, Qf=Qf)
    assert obj.N == N
    assert obj.is_diag

    # Cost of traj_ref1 against itself must be 0.0
    c_zero = cost(obj, traj_ref1)
    np.testing.assert_allclose(c_zero, 0.0, atol=1e-14)

    # Update reference with a longer trajectory and non-zero start index
    N_long = 25
    X_long = jnp.array(rng.standard_normal((N_long, n)))
    U_long = jnp.array(rng.standard_normal((N_long - 1, m)))
    t_long = jnp.linspace(0.0, 2.5, N_long)
    dt_long = jnp.diff(t_long)
    traj_long = Trajectory(X=X_long, U=U_long, t=t_long, dt=dt_long)

    start_idx = 5
    obj_updated = update_reference(obj, traj_long, start=start_idx)

    # Extract target slice
    target_slice = Trajectory(
        X=X_long[start_idx : start_idx + N],
        U=U_long[start_idx : start_idx + N - 1],
        t=t_long[start_idx : start_idx + N],
        dt=dt_long[start_idx : start_idx + N - 1],
    )

    c_slice_zero = cost(obj_updated, target_slice)
    np.testing.assert_allclose(c_slice_zero, 0.0, atol=1e-14)

    # Cost of a different trajectory against obj_updated
    X_test = jnp.array(rng.standard_normal((N, n)))
    U_test = jnp.array(rng.standard_normal((N - 1, m)))
    traj_test = Trajectory(X=X_test, U=U_test, t=t1, dt=dt1)

    total_c = cost(obj_updated, traj_test)

    expected_total = 0.0
    for k in range(N - 1):
        xk = X_test[k]
        uk = U_test[k]
        x_ref_k = X_long[start_idx + k]
        u_ref_k = U_long[start_idx + k]
        expected_total += 0.5 * jnp.sum(Q * (xk - x_ref_k) ** 2) + 0.5 * jnp.sum(R * (uk - u_ref_k) ** 2)
    expected_total += 0.5 * jnp.sum(Qf * (X_test[-1] - X_long[start_idx + N - 1]) ** 2)

    np.testing.assert_allclose(total_c, expected_total, rtol=1e-14, atol=1e-14)


def test_objective_indexing_and_properties() -> None:
    N = 5
    Q = jnp.array([1.0, 2.0])
    R = jnp.array([3.0])
    Qf = jnp.array([10.0, 20.0])
    xf = jnp.array([0.0, 0.0])

    obj = LQRObjective(Q, R, Qf, xf, N)
    assert len(obj) == N

    # Stage knot points
    for k in range(N - 1):
        cost_k = obj[k]
        assert isinstance(cost_k, DiagonalCost)
        assert not cost_k.terminal
        np.testing.assert_allclose(cost_k.Q, Q)
        np.testing.assert_allclose(cost_k.R, R)

    # Terminal knot point
    term_k = obj[N - 1]
    assert isinstance(term_k, DiagonalCost)
    assert term_k.terminal
    np.testing.assert_allclose(term_k.Q, Qf)

    # Negative indexing
    assert obj[-1] is obj.terminal_cost

    with pytest.raises(IndexError):
        _ = obj[N]
    with pytest.raises(IndexError):
        _ = obj[-N - 1]
