import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import trajopt._env
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.affine import AffineModel
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt, _IpoptCallback
from trajopt.transcription.layout import (
    _trajectory_to_z,
    _z_to_trajectory,
    compute_constraint_violation,
)


def _make_problem(
    N: int = 13,
    b: int = 6,
    *,
    n: int = 2,
    m: int = 1,
) -> Problem:
    """Build a linear AffineModel Problem with specified Horizon and block size."""
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    model = AffineModel(A=A, B=B)
    Q = jnp.eye(n)
    R = jnp.eye(m)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    return Problem(
        model=model,
        obj=obj,
        N=N,
        dt=0.1,
        control_block_steps=b,
    )


def test_reduced_nlp_dimensions_and_trajectory_shapes() -> None:
    """NLP vector has length N * n + B * m and returned Trajectory has native dimensions."""
    pytest.importorskip("cyipopt")

    N = 13
    b = 6
    H = N - 1
    B = H // b
    n = 2
    m = 1
    problem = _make_problem(N=N, b=b, n=n, m=m)

    cb = Ipopt().transcription_callback(problem, jnp.zeros(n), 0.0, 0.1)
    _jac_rows, jac_cols = cb.jacobianstructure()
    hess_rows, hess_cols = cb.hessianstructure()

    expected_nlp_len = N * n + B * m
    assert max(jac_cols) < expected_nlp_len
    assert max(hess_rows) < expected_nlp_len
    assert max(hess_cols) < expected_nlp_len

    x0 = jnp.array([1.0, 0.0])
    res = MPC(problem, Ipopt(options={"print_level": 0}), x0=x0).solve()

    assert res.success
    assert res.trajectory.X.shape == (13, 2)
    assert res.trajectory.U.shape == (12, 1)
    assert len(res.Z) == N * n + H * m

    u_np = np.asarray(res.trajectory.U)
    np.testing.assert_allclose(u_np[:6], np.repeat(u_np[0:1], 6, axis=0))
    np.testing.assert_allclose(u_np[6:], np.repeat(u_np[6:7], 6, axis=0))


def test_warm_start_reduction_and_expansion() -> None:
    """Warm starts reduce by taking the first native control per block and expand to block-constant controls."""
    pytest.importorskip("cyipopt")

    N = 7
    b = 3
    H = N - 1
    B = H // b
    n = 2
    m = 1
    problem = _make_problem(N=N, b=b, n=n, m=m)

    U_native = jnp.arange(1, H + 1, dtype=jnp.float64)[:, None]
    X_native = jnp.ones((N, n), dtype=jnp.float64)
    from trajopt.transcription.layout import reduce_controls

    U_reduced = reduce_controls(U_native, b)
    assert U_reduced.shape == (B, m)
    # First native control in each block is taken without averaging or rounding:
    np.testing.assert_allclose(np.asarray(U_reduced), np.asarray(U_native[::b]))
    assert float(U_reduced[0, 0]) == 1.0  # U_native[0] is 1.0
    assert float(U_reduced[1, 0]) == 4.0  # U_native[3] is 4.0

    dt_arr = jnp.full(H, 0.1)
    ws = WarmStart.cold(
        problem, X_native[0], initial_trajectory=Trajectory(X=X_native, U=U_native, t=jnp.zeros(N), dt=dt_arr)
    )
    bc = BoundaryConditions(x0=X_native[0], t0=jnp.array(0.0))
    prog = Program(problem, solver=Ipopt(options={"print_level": 0, "max_iter": 50}))
    res = prog.solve(bc, ws)
    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (H, m)
    u_res = np.asarray(res.trajectory.U)
    np.testing.assert_allclose(u_res[:3], np.repeat(u_res[0:1], 3, axis=0))
    np.testing.assert_allclose(u_res[3:6], np.repeat(u_res[3:4], 3, axis=0))


def _make_pendulum_cb_and_data() -> tuple[_IpoptCallback, np.ndarray, np.ndarray, float, int, int]:
    N = 5
    b = 2
    H = N - 1
    B = H // b
    n = 2
    m = 1

    model = Pendulum()
    Q = jnp.eye(n)
    R = jnp.eye(m)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    problem = Problem(model=model, obj=obj, N=N, dt=0.05, integrator=RK4(), control_block_steps=b)

    x0 = jnp.array([0.2, -0.1])
    cb = Ipopt().transcription_callback(problem, x0, 0.0, 0.05)

    nlp_len = N * n + B * m
    z = np.sin(np.arange(nlp_len, dtype=np.float64) * 0.7) + 0.1

    P_total = n + H * n + sum(problem.constraints.p)
    lam = np.cos(np.arange(P_total, dtype=np.float64) * 0.5) * 0.2
    obj_factor = 1.3
    return cb, z, lam, obj_factor, nlp_len, P_total


def test_callback_gradient_matches_finite_differences() -> None:
    """Verify callback gradient matches central finite differences on blocked Problems."""
    pytest.importorskip("cyipopt")

    cb, z, _, _, nlp_len, _ = _make_pendulum_cb_and_data()
    grad_analytic = cb.gradient(z)
    eps = 1e-6
    grad_fd = np.zeros_like(grad_analytic)
    for i in range(nlp_len):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        grad_fd[i] = (cb.objective(zp) - cb.objective(zm)) / (2 * eps)
    np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4, atol=1e-5)


def test_callback_jacobian_matches_finite_differences() -> None:
    """Verify callback sparse Jacobian matches central finite differences without duplicates."""
    pytest.importorskip("cyipopt")

    cb, z, _, _, nlp_len, P_total = _make_pendulum_cb_and_data()
    jac_val = cb.jacobian(z)
    jac_rows, jac_cols = cb.jacobianstructure()
    coords_j = list(zip(jac_rows, jac_cols, strict=True))
    assert len(coords_j) == len(set(coords_j)), "Jacobian sparsity must not contain duplicate coordinates"
    jac_sparse = sp.coo_matrix((jac_val, (jac_rows, jac_cols)), shape=(P_total, nlp_len)).toarray()

    eps = 1e-6
    jac_fd = np.zeros((P_total, nlp_len), dtype=np.float64)
    for i in range(nlp_len):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        jac_fd[:, i] = (cb.constraints(zp) - cb.constraints(zm)) / (2 * eps)
    np.testing.assert_allclose(jac_sparse, jac_fd, rtol=1e-4, atol=1e-5)


def test_callback_hessian_matches_finite_differences() -> None:
    """Verify callback sparse exact Lagrangian Hessian matches finite differences."""
    pytest.importorskip("cyipopt")

    cb, z, lam, obj_factor, nlp_len, P_total = _make_pendulum_cb_and_data()
    jac_rows, jac_cols = cb.jacobianstructure()
    hess_val = cb.hessian(z, lam, obj_factor)
    hess_rows, hess_cols = cb.hessianstructure()
    coords_h = list(zip(hess_rows, hess_cols, strict=True))
    assert len(coords_h) == len(set(coords_h)), "Hessian sparsity must not contain duplicate coordinates"
    assert np.all(hess_rows >= hess_cols), "Hessian sparsity must be strictly lower-triangular"
    hess_lower = sp.coo_matrix((hess_val, (hess_rows, hess_cols)), shape=(nlp_len, nlp_len)).toarray()
    hess_sym = hess_lower + hess_lower.T - np.diag(np.diag(hess_lower))

    def lagrangian_grad(z_in: np.ndarray) -> np.ndarray:
        grad_f = cb.gradient(z_in)
        j_val = cb.jacobian(z_in)
        j_mat = sp.coo_matrix((j_val, (jac_rows, jac_cols)), shape=(P_total, nlp_len))
        return obj_factor * grad_f + j_mat.T.dot(lam)

    eps = 1e-6
    hess_fd = np.zeros((nlp_len, nlp_len), dtype=np.float64)
    for j in range(nlp_len):
        zp = z.copy()
        zm = z.copy()
        zp[j] += eps
        zm[j] -= eps
        hess_fd[:, j] = (lagrangian_grad(zp) - lagrangian_grad(zm)) / (2 * eps)
    np.testing.assert_allclose(hess_sym, hess_fd, rtol=1e-4, atol=1e-5)


def test_zero_canonical_multipliers_for_b_gt_1() -> None:
    """For b > 1, IpoptResult.lam and mu are canonical-sized zeros, replacing prior WarmStart duals."""
    pytest.importorskip("cyipopt")

    N = 7
    b = 3
    H = N - 1
    n = 2
    m = 1
    problem = _make_problem(N=N, b=b, n=n, m=m)

    x0 = jnp.array([1.0, 0.0])
    mpc = MPC(problem, Ipopt(options={"print_level": 0}), x0=x0)

    # Seed nonzero multipliers in warm start
    P_total = n + H * n + sum(problem.constraints.p)
    nz_primal = N * n + H * m
    seeded_lam = jnp.ones(P_total, dtype=jnp.float64) * 5.0
    seeded_mu = jnp.ones(nz_primal, dtype=jnp.float64) * 3.0
    mpc._ws = WarmStart(Z=mpc.warm_start.Z, lam=seeded_lam, mu=seeded_mu)  # noqa: SLF001 -- seeding test multipliers

    res = mpc.solve()
    assert res.success
    assert res.lam.shape == (P_total,)
    assert res.mu.shape == (nz_primal,)
    np.testing.assert_allclose(res.lam, 0.0)
    np.testing.assert_allclose(res.mu, 0.0)

    # MPC's warm start was updated with the canonical zeros
    np.testing.assert_allclose(np.asarray(mpc.warm_start.lam), 0.0)
    np.testing.assert_allclose(np.asarray(mpc.warm_start.mu), 0.0)


def test_block_constancy_in_constraint_violation() -> None:
    """compute_constraint_violation checks that published controls are block-constant."""
    N = 7
    b = 3
    n = 2
    m = 1
    problem = _make_problem(N=N, b=b, n=n, m=m)

    x0 = jnp.zeros(n)
    # Block constant controls: [1.0, 1.0, 1.0] and [2.0, 2.0, 2.0]
    U_const = jnp.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])

    # Roll out dynamics so defect violation is zero
    X_list = [x0]
    for k in range(N - 1):
        X_list.append(problem.model.discrete_dynamics(X_list[-1], U_const[k], 0.0, 0.1))
    X = jnp.stack(X_list)
    Z_const = _trajectory_to_z(X, U_const)

    viol_const = compute_constraint_violation(problem, Z_const, x0, t0=0.0, dt=0.1)
    assert viol_const < 1e-6

    # Perturb one control in the block
    U_varying = jnp.array([[1.0], [1.5], [1.0], [2.0], [2.0], [2.0]])
    Z_varying = _trajectory_to_z(X, U_varying)
    viol_varying = compute_constraint_violation(problem, Z_varying, x0, t0=0.0, dt=0.1)

    assert viol_varying >= 0.5


def test_failed_solve_expansion_and_multipliers() -> None:
    """Failed solves expand iterate to native Trajectory and canonical Z and return canonical zero Multipliers."""
    pytest.importorskip("cyipopt")

    N = 7
    b = 3
    H = N - 1
    n = 2
    m = 1
    problem = _make_problem(N=N, b=b, n=n, m=m)

    x0 = jnp.array([1.0, 0.0])
    # max_iter = 0 forces failure
    res = MPC(problem, Ipopt(options={"print_level": 0, "max_iter": 0}), x0=x0).solve()

    assert not res.success
    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (H, m)
    assert len(res.Z) == N * n + H * m

    P_total = n + H * n + sum(problem.constraints.p)
    nz_primal = N * n + H * m
    assert res.lam.shape == (P_total,)
    assert res.mu.shape == (nz_primal,)
    np.testing.assert_allclose(res.lam, 0.0)
    np.testing.assert_allclose(res.mu, 0.0)


def test_multiplier_behavior_b_eq_1_preserved() -> None:
    """For b = 1, canonical Multiplier output and dual warm starting are preserved."""
    pytest.importorskip("cyipopt")

    N = 7
    n = 2
    m = 1
    problem = _make_problem(N=N, b=1, n=n, m=m)

    x0 = jnp.array([1.0, 0.0])
    mpc = MPC(problem, Ipopt(options={"print_level": 0}), x0=x0)
    res = mpc.solve()

    assert res.success
    assert len(res.lam) > 0
    assert not np.all(res.lam == 0.0)
    # Warm start retains solved multipliers
    np.testing.assert_allclose(np.asarray(mpc.warm_start.lam), res.lam)


def test_objective_not_scaled_by_b() -> None:
    """The Objective is evaluated on native states and expanded controls without scaling stage weights by b."""
    from trajopt.transcription.layout import _trajectory_to_nlp_z

    N = 7
    b = 3
    H = N - 1
    n = 2
    m = 1
    problem = _make_problem(N=N, b=b, n=n, m=m)

    X = jnp.ones((N, n), dtype=jnp.float64) * 0.5
    U_blk = jnp.array([[2.0], [3.0]])
    U_expanded = jnp.repeat(U_blk, b, axis=0)

    z_nlp = _trajectory_to_nlp_z(X, U_blk, b)
    cb = Ipopt().transcription_callback(problem, X[0], 0.0, 0.1)

    obj_val = cb.objective(np.asarray(z_nlp, dtype=np.float64))

    # Compute unscaled native cost: sum of all native stage costs + terminal cost
    stage_costs = problem.obj.stage_cost.stage_costs(X[:-1], U_expanded, jnp.arange(H, dtype=jnp.float64) * 0.1)
    term_cost = problem.obj.terminal_cost.evaluate(X[-1], None, H * 0.1)
    expected_cost = float(jnp.sum(stage_costs) + term_cost)

    np.testing.assert_allclose(obj_val, expected_cost, rtol=1e-6)


def test_blocked_binary_homotopy_multiple_shooting() -> None:
    """Ipopt handles blocked binary Problems with binary homotopy continuation."""
    pytest.importorskip("cyipopt")

    N = 5
    b = 2
    H = N - 1
    B = H // b
    n = 2
    m = 1

    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B_mat = jnp.array([[0.0], [0.1]])
    model = AffineModel(A=A, B=B_mat)
    Q = jnp.eye(n)
    R = jnp.eye(m)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    problem = Problem(
        model=model,
        obj=obj,
        N=N,
        dt=0.1,
        control_block_steps=b,
        binary_control_indices=(0,),
    )

    cb = Ipopt().transcription_callback(problem, jnp.zeros(n), 0.0, 0.1)
    assert len(cb.binary_z_indices) == B

    x0 = jnp.array([0.05, 0.0])
    res = MPC(
        problem, Ipopt(options={"print_level": 0}, beta_initial=0.1, beta_growth=10.0, max_passes=3), x0=x0
    ).solve()

    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (H, m)
    u_np = np.asarray(res.trajectory.U)
    # Check block constancy
    np.testing.assert_allclose(u_np[:2], np.repeat(u_np[0:1], 2, axis=0))
    np.testing.assert_allclose(u_np[2:4], np.repeat(u_np[2:3], 2, axis=0))
    if res.success:
        for u_val in u_np.ravel():
            assert min(abs(u_val), abs(u_val - 1.0)) < 1e-3


def test_blocked_multiple_shooting_with_linear_horizon_constraints() -> None:
    """Multiple shooting correctly maps and differentiates linear Horizon constraints under control blocking."""
    pytest.importorskip("cyipopt")

    N = 7
    b = 3
    H = N - 1
    n = 2
    m = 1
    A = jnp.eye(n)
    B = jnp.ones((n, m))
    model = AffineModel(A=A, B=B)
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)

    # Constraint summing native controls in block 0: u0, u1, u2 at canonical indices (2, 5, 8)
    cl = ConstraintList(n=n, m=m, N=N)
    inds = (2, 5, 8)
    A_hz = jnp.array([[1.0, 2.0, 3.0]])
    b_hz = jnp.array([5.0])
    hz = LinearHorizonConstraint(A=A_hz, b=b_hz, inds=inds)
    cl.add_horizon_constraint(hz)

    prob = Problem(model=model, obj=obj, constraints=cl, N=N, dt=0.1, control_block_steps=b)
    cb = Ipopt().transcription_callback(prob, jnp.zeros(n), 0.0, 0.1)

    nlp_len = N * n + (H // b) * m
    z = np.ones(nlp_len, dtype=np.float64)
    jac_val = cb.jacobian(z)
    jac_rows, jac_cols = cb.jacobianstructure()
    coords = list(zip(jac_rows, jac_cols, strict=True))
    assert len(coords) == len(set(coords)), "Horizon constraint columns must not introduce duplicate sparse coordinates"

    eps = 1e-6
    P_total = len(cb.constraints(z))
    jac_fd = np.zeros((P_total, nlp_len), dtype=np.float64)
    for i in range(nlp_len):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        jac_fd[:, i] = (cb.constraints(zp) - cb.constraints(zm)) / (2 * eps)

    jac_sparse = sp.coo_matrix((jac_val, (jac_rows, jac_cols)), shape=(P_total, nlp_len)).toarray()
    np.testing.assert_allclose(jac_sparse, jac_fd, rtol=1e-4, atol=1e-5)

    res = MPC(prob, Ipopt(options={"print_level": 0}), x0=jnp.zeros(n)).solve()
    assert res.success
