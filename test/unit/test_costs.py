import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs import (
    DiagonalCost,
    GenericCost,
    LieLQRCost,
    LQRCost,
    LQRObjective,
    Objective,
    QuadraticCost,
    QuatGeodesicCost,
    TrackingObjective,
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

    obj = LQRObjective(Q, R, Qf, N).with_reference(jnp.broadcast_to(xf, (N, n)), jnp.broadcast_to(uf, (N - 1, m)))
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
    assert obj.Q_f.shape == (n,)
    assert obj.q_f.shape == (n,)
    assert obj.c_f.shape == ()

    # Create dummy trajectory
    rng = np.random.default_rng(42)
    X = jnp.array(rng.standard_normal((N, n)))
    U = jnp.array(rng.standard_normal((N - 1, m)))
    t = jnp.linspace(0.0, 1.0, N)
    dt = jnp.diff(t)
    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    # Evaluate cost on trajectory
    total_cost = obj.cost(traj)

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
    jit_cost = jax.jit(Objective.cost)
    jit_val = jit_cost(obj, traj)
    np.testing.assert_allclose(jit_val, expected_total, rtol=1e-14, atol=1e-14)


def test_mixed_diagonal_and_dense_weights() -> None:
    n, m, N = 2, 2, 4
    Q = jnp.array([1.0, 2.0])
    R = jnp.array([[2.0, 0.7], [0.7, 3.0]])  # dense: the off-diagonal weight must survive
    Qf = jnp.array([5.0, 6.0])
    xf = jnp.array([0.5, -0.5])

    obj = LQRObjective(Q, R, Qf, N).with_reference(jnp.broadcast_to(xf, (N, n)), jnp.zeros((N - 1, m)))
    assert not obj.is_diag
    assert obj.Q.shape == (N - 1, n, n)
    assert obj.R.shape == (N - 1, m, m)
    assert obj.Q_f.shape == (n, n)
    np.testing.assert_allclose(obj.R[0], R, rtol=1e-14, atol=1e-14)

    rng = np.random.default_rng(7)
    X = jnp.array(rng.standard_normal((N, n)))
    U = jnp.array(rng.standard_normal((N - 1, m)))
    t = jnp.linspace(0.0, 1.0, N)
    traj = Trajectory(X=X, U=U, t=t, dt=jnp.diff(t))

    expected_total = 0.5 * jnp.sum(Qf * (X[-1] - xf) ** 2)
    for k in range(N - 1):
        dx = X[k] - xf
        expected_total += 0.5 * jnp.sum(Q * dx**2) + 0.5 * U[k] @ (R @ U[k])

    np.testing.assert_allclose(obj.cost(traj), expected_total, rtol=1e-12, atol=1e-12)

    # A dense R is never truncated to its diagonal behind the user's back
    with pytest.raises(ValueError, match="DiagonalCost weights"):
        DiagonalCost(Q=Q, R=R)


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
    c_zero = obj.cost(traj_ref1)
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

    c_slice_zero = obj_updated.cost(target_slice)
    np.testing.assert_allclose(c_slice_zero, 0.0, atol=1e-14)

    # Cost of a different trajectory against obj_updated
    X_test = jnp.array(rng.standard_normal((N, n)))
    U_test = jnp.array(rng.standard_normal((N - 1, m)))
    traj_test = Trajectory(X=X_test, U=U_test, t=t1, dt=dt1)

    total_c = obj_updated.cost(traj_test)

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
    obj = LQRObjective(Q, R, Qf, N)
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


def test_objective_indexing_leaves_unstacked_stage_cost_whole() -> None:
    """Assert a stage cost with no horizon axis is shared, not sliced, when m collides with N - 1."""
    n, m, N = 13, 4, 5  # m == N - 1, the shape collision that a leading-dimension test cannot see
    xf = jnp.zeros(n).at[6].set(1.0)
    stage_cost = QuatGeodesicCost(Q=jnp.ones(n), R=jnp.full((m,), 0.01), q_ref=xf[3:7], m=m)
    term_cost = QuatGeodesicCost(Q=jnp.ones(n), q_ref=xf[3:7], terminal=True)
    obj = Objective(stage_cost=stage_cost, terminal_cost=term_cost, N=N)

    assert not obj.stage_cost.is_stacked
    for k in range(N - 1):
        assert obj[k] is obj.stage_cost

    # The cost stays evaluable, which is what the sliced parameters broke
    obj[0].evaluate(jnp.zeros(n), jnp.zeros(m))


def test_objective_indexing_slices_stacked_cost_with_broadcast_parameters() -> None:
    """Assert stacked costs carry the horizon axis on every parameter, including defaulted ones."""
    n, m, N = 3, 4, 5  # m == N - 1 again, this time on a genuinely stacked cost
    Q = jnp.tile(jnp.eye(n), (N - 1, 1, 1))
    R = jnp.tile(jnp.eye(m), (N - 1, 1, 1))
    stage_cost = QuadraticCost(Q=Q, R=R)
    assert stage_cost.is_stacked
    assert stage_cost.q.shape == (N - 1, n)
    assert stage_cost.r.shape == (N - 1, m)
    assert stage_cost.c.shape == (N - 1,)

    obj = Objective(stage_cost=stage_cost, terminal_cost=QuadraticCost(Q=jnp.eye(n), terminal=True, m=m))
    cost_0 = obj[0]
    assert isinstance(cost_0, QuadraticCost)
    assert cost_0.Q.shape == (n, n)
    assert cost_0.R.shape == (m, m)
    assert cost_0.H.shape == (m, n)
    assert cost_0.q.shape == (n,)
    assert cost_0.r.shape == (m,)
    assert cost_0.c.shape == ()


def test_quat_geodesic_cost_evaluation_and_subgradient_branches() -> None:
    """Assert QuatGeodesicCost evaluation and gradient on both subgradient branches."""
    n, m = 13, 4
    Q = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
    R = jnp.array([0.1, 0.1, 0.1, 0.1])
    q_ref = jnp.array([0.0, 0.0, 0.0, 1.0])
    w = 2.5

    cost = QuatGeodesicCost(Q=Q, R=R, q_ref=q_ref, w=w, qind=(3, 4, 5, 6))
    assert cost.n == n
    assert cost.m == m
    assert not cost.terminal

    u = jnp.array([0.5, -0.5, 1.0, -1.0])

    # Branch 1: q_ref^T q > 0 (e.g. q is close to q_ref)
    q_pos = jnp.array([0.1, 0.2, 0.3, 0.9])
    q_pos = q_pos / jnp.linalg.norm(q_pos)
    x_pos = jnp.array([1.0, 2.0, 3.0, *q_pos, 0.5, -0.5, 1.0, 0.1, -0.1, 0.2])

    dq_pos = float(jnp.dot(q_ref, q_pos))
    assert dq_pos > 0.0

    val_pos = cost.evaluate(x_pos, u)
    expected_quad_pos = 0.5 * jnp.sum(Q * (x_pos**2)) + 0.5 * jnp.sum(R * (u**2))
    expected_geo_pos = w * (1.0 - dq_pos)
    np.testing.assert_allclose(val_pos, expected_quad_pos + expected_geo_pos, rtol=1e-14)

    # Gradient check on Branch 1 (dq > 0 => grad_q = -w * q_ref)
    grad_pos = cost.gradient(x_pos, u)
    assert grad_pos.shape == (n + m,)
    expected_gx_pos = Q * x_pos
    expected_gx_pos = expected_gx_pos.at[3:7].set(-w * q_ref)
    expected_gu_pos = R * u
    np.testing.assert_allclose(grad_pos[:n], expected_gx_pos, rtol=1e-12)
    np.testing.assert_allclose(grad_pos[n:], expected_gu_pos, rtol=1e-12)

    # Branch 2: q_ref^T q < 0 (double-cover antipodal branch: q is close to -q_ref)
    q_neg = -q_pos
    x_neg = jnp.array([1.0, 2.0, 3.0, *q_neg, 0.5, -0.5, 1.0, 0.1, -0.1, 0.2])

    dq_neg = float(jnp.dot(q_ref, q_neg))
    assert dq_neg < 0.0

    val_neg = cost.evaluate(x_neg, u)
    expected_quad_neg = 0.5 * jnp.sum(Q * (x_neg**2)) + 0.5 * jnp.sum(R * (u**2))
    expected_geo_neg = w * (1.0 + dq_neg)
    # Geodesic penalty should be identical for q and -q (double cover!)
    np.testing.assert_allclose(expected_geo_neg, expected_geo_pos, rtol=1e-14)
    np.testing.assert_allclose(val_neg, expected_quad_neg + expected_geo_neg, rtol=1e-14)

    # Gradient check on Branch 2 (dq < 0 => grad_q = +w * q_ref)
    grad_neg = cost.gradient(x_neg, u)
    expected_gx_neg = Q * x_neg
    expected_gx_neg = expected_gx_neg.at[3:7].set(+w * q_ref)
    np.testing.assert_allclose(grad_neg[:n], expected_gx_neg, rtol=1e-12)
    np.testing.assert_allclose(grad_neg[n:], expected_gu_pos, rtol=1e-12)

    # Hessian check: second derivative of min(1+dq, 1-dq) is zero
    hess_pos = cost.hessian(x_pos, u)
    assert hess_pos.shape == (n + m, n + m)
    np.testing.assert_allclose(hess_pos, jnp.diag(jnp.concatenate([Q, R])), atol=1e-14)


def test_lie_lqr_cost_helper() -> None:
    """Assert LieLQRCost constructor builds tracking cost with zero error at goal."""
    xf = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    uf = jnp.array([1.2, 1.2, 1.2, 1.2])
    Q = jnp.ones(13)
    R = jnp.full(4, 0.1)

    stage_cost = LieLQRCost(Q=Q, R=R, xf=xf, uf=uf, w=4.0)
    term_cost = LieLQRCost(Q=Q, R=R, xf=xf, terminal=True, w=4.0)

    # At goal state and control, cost must be 0
    val_stage = stage_cost.evaluate(xf, uf)
    np.testing.assert_allclose(val_stage, 0.0, atol=1e-14)

    val_term = term_cost.evaluate(xf)
    np.testing.assert_allclose(val_term, 0.0, atol=1e-14)

    # Double-cover goal state: xf with -qf must also give 0 cost
    xf_antipodal = xf.at[3:7].set(-xf[3:7])
    val_antipodal = stage_cost.evaluate(xf_antipodal, uf)
    np.testing.assert_allclose(val_antipodal, 0.0, atol=1e-14)
