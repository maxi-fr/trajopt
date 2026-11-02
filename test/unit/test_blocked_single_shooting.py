import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.affine import AffineModel
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.layout import (
    _trajectory_to_z,
    _z_to_trajectory,
    blocked_control_bounds,
    compute_constraint_violation,
    reduce_controls,
)
from trajopt.transcription.result import constraint_row_count
from trajopt.transcription.single_shooting import (
    SingleShooting,
    _primal_bounds,
    _SingleShootingCallback,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_h,
    eval_jac_g,
    single_shooting_dimensions,
)


def _make_blocked_problem(
    N: int = 7,
    b: int = 3,
    *,
    n: int = 2,
    m: int = 1,
    constraints: ConstraintList | None = None,
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
        constraints=constraints,
        N=N,
        dt=0.1,
        control_block_steps=b,
    )


def test_single_shooting_dimensions_blocked() -> None:
    """single_shooting_dimensions returns (B * m, p_user) where B = (N - 1) // b."""
    n, m, N = 2, 1, 13
    b = 6
    B = (N - 1) // b
    problem = _make_blocked_problem(N=N, b=b, n=n, m=m)
    n_u, p_user = single_shooting_dimensions(problem)
    assert n_u == B * m
    assert p_user == 0

    # With constraints
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(GoalConstraint(n=n, xf=jnp.zeros(n)), N - 1)
    prob_c = _make_blocked_problem(N=N, b=b, n=n, m=m, constraints=constraints)
    n_u_c, p_user_c = single_shooting_dimensions(prob_c)
    assert n_u_c == B * m
    assert p_user_c == n


def test_blocked_primal_bounds() -> None:
    """Control bounds are intersected per-block bounds of shape (B * m,)."""
    N, b, n, m = 7, 3, 2, 1
    H = N - 1
    B = H // b

    constraints = ConstraintList(n=n, m=m, N=N)
    # Block 0: k=0,1,2
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-2.0], u_max=[2.0]), 0)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-1.5], u_max=[3.0]), 1)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-3.0], u_max=[1.2]), 2)
    # Block 1: k=3,4,5
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-1.0], u_max=[2.0]), 3)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-2.0], u_max=[1.5]), 4)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-0.8], u_max=[1.8]), 5)

    prob = _make_blocked_problem(N=N, b=b, n=n, m=m, constraints=constraints)
    z_l, z_u = _primal_bounds(prob)
    assert z_l.shape == (B * m,)
    assert z_u.shape == (B * m,)

    expected_bl_l, expected_bl_u = blocked_control_bounds(prob)
    np.testing.assert_allclose(z_l, expected_bl_l.reshape(-1))
    np.testing.assert_allclose(z_u, expected_bl_u.reshape(-1))
    np.testing.assert_allclose(z_l, [-1.5, -0.8])
    np.testing.assert_allclose(z_u, [1.2, 1.5])


def test_warm_start_reduction_and_expansion() -> None:
    """Warm starts reduce canonical controls by U[::b] and expand to native block-constant controls."""
    pytest.importorskip("cyipopt")

    N, b, n, m = 7, 3, 2, 1
    H = N - 1
    B = H // b
    problem = _make_blocked_problem(N=N, b=b, n=n, m=m)

    U_native = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=jnp.float64)
    X_native = jnp.zeros((N, n), dtype=jnp.float64)
    dt_arr = jnp.full(H, 0.1)

    traj = Trajectory(X=X_native, U=U_native, t=jnp.zeros(N), dt=dt_arr)
    ws = WarmStart.cold(problem, X_native[0], initial_trajectory=traj)

    _, U_guess = ws.unpack(problem)
    U_reduced = reduce_controls(U_guess, b)
    assert U_reduced.shape == (B, m)
    np.testing.assert_allclose(np.asarray(U_reduced), np.asarray(U_native[::b]))
    assert float(U_reduced[0, 0]) == 1.0
    assert float(U_reduced[1, 0]) == 4.0

    bc = BoundaryConditions(x0=X_native[0], t0=jnp.array(0.0))
    prog = Program(problem, solver=SingleShooting(Ipopt(options={"print_level": 0, "max_iter": 50})))
    res = prog.solve(bc, ws)

    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (H, m)
    assert len(res.Z) == N * n + H * m

    u_res = np.asarray(res.trajectory.U)
    np.testing.assert_allclose(u_res[:3], np.repeat(u_res[0:1], 3, axis=0))
    np.testing.assert_allclose(u_res[3:6], np.repeat(u_res[3:4], 3, axis=0))


def _make_pendulum_cb_and_data() -> tuple[_SingleShootingCallback, np.ndarray, np.ndarray, float, int, int]:
    N, b, n, m = 5, 2, 2, 1
    H = N - 1
    B = H // b

    model = Pendulum()
    Q = jnp.eye(n)
    R = jnp.eye(m)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    problem = Problem(model=model, obj=obj, N=N, dt=0.05, integrator=RK4(), control_block_steps=b)

    x0 = jnp.array([0.2, -0.1])
    cb = SingleShooting(Ipopt(), hessian="dense").transcription_callback(problem, x0, jnp.array(0.0), jnp.array(0.05))

    n_u, p_user = single_shooting_dimensions(problem)
    assert n_u == B * m

    u = np.sin(np.arange(n_u, dtype=np.float64) * 0.7) + 0.1
    lam = np.cos(np.arange(p_user, dtype=np.float64) * 0.5) * 0.2
    obj_factor = 1.3
    return cb, u, lam, obj_factor, n_u, p_user


def test_callback_gradient_matches_finite_differences() -> None:
    """Callback gradient matches central finite differences w.r.t block controls."""
    pytest.importorskip("cyipopt")

    cb, u, _, _, n_u, _ = _make_pendulum_cb_and_data()
    grad_analytic = cb.gradient(u)
    eps = 1e-6
    grad_fd = np.zeros_like(grad_analytic)
    for i in range(n_u):
        up = u.copy()
        um = u.copy()
        up[i] += eps
        um[i] -= eps
        grad_fd[i] = (cb.objective(up) - cb.objective(um)) / (2 * eps)

    np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4, atol=1e-5)


def test_callback_jacobian_and_hessian_matches_finite_differences() -> None:
    """Callback Jacobian and lower-triangular Hessian match central finite differences on smooth blocked Problem."""
    pytest.importorskip("cyipopt")

    N, b, n, m = 5, 2, 2, 1
    H = N - 1
    B = H // b

    model = Pendulum()
    Q = jnp.eye(n)
    R = jnp.eye(m)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    # Add a smooth nonlinear/linear constraint so p_user > 0
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([0.5])),
        range(H),
    )
    problem = Problem(
        model=model, obj=obj, constraints=constraints, N=N, dt=0.05, integrator=RK4(), control_block_steps=b
    )

    x0 = jnp.array([0.2, -0.1])
    cb = SingleShooting(Ipopt(), hessian="dense").transcription_callback(problem, x0, jnp.array(0.0), jnp.array(0.05))

    n_u, p_user = single_shooting_dimensions(problem)
    assert n_u == B * m
    assert p_user == H

    u = np.sin(np.arange(n_u, dtype=np.float64) * 0.7) + 0.1
    lam = np.cos(np.arange(p_user, dtype=np.float64) * 0.5) * 0.2
    obj_factor = 1.3

    # Check Jacobian
    jac_val = cb.jacobian(u)
    jac_rows, jac_cols = cb.jacobianstructure()
    jac_mat = np.zeros((p_user, n_u), dtype=np.float64)
    jac_mat[jac_rows, jac_cols] = jac_val

    eps = 1e-6
    jac_fd = np.zeros((p_user, n_u), dtype=np.float64)
    for j in range(n_u):
        up = u.copy()
        um = u.copy()
        up[j] += eps
        um[j] -= eps
        jac_fd[:, j] = (cb.constraints(up) - cb.constraints(um)) / (2 * eps)

    np.testing.assert_allclose(jac_mat, jac_fd, rtol=1e-4, atol=1e-5)

    # Check Hessian
    hess_val = cb.hessian(u, lam, obj_factor)
    hess_rows, hess_cols = cb.hessianstructure()
    assert np.all(hess_rows >= hess_cols), "Hessian sparsity must be lower-triangular"
    hess_mat = np.zeros((n_u, n_u), dtype=np.float64)
    hess_mat[hess_rows, hess_cols] = hess_val
    hess_sym = hess_mat + hess_mat.T - np.diag(np.diag(hess_mat))

    def lagrangian_grad(u_in: np.ndarray) -> np.ndarray:
        grad_f = cb.gradient(u_in)
        j_val = cb.jacobian(u_in)
        j_m = np.zeros((p_user, n_u), dtype=np.float64)
        j_m[jac_rows, jac_cols] = j_val
        return obj_factor * grad_f + j_m.T.dot(lam)

    hess_fd = np.zeros((n_u, n_u), dtype=np.float64)
    for j in range(n_u):
        up = u.copy()
        um = u.copy()
        up[j] += eps
        um[j] -= eps
        hess_fd[:, j] = (lagrangian_grad(up) - lagrangian_grad(um)) / (2 * eps)

    np.testing.assert_allclose(hess_sym, hess_fd, rtol=1e-4, atol=1e-5)


def test_zero_canonical_multipliers_for_b_gt_1() -> None:
    """For b > 1, SingleShooting returns canonical-sized zero lam and mu arrays and does not consume prior Multipliers."""
    pytest.importorskip("cyipopt")

    N, b, n, m = 7, 3, 2, 1
    H = N - 1
    problem = _make_blocked_problem(N=N, b=b, n=n, m=m)

    x0 = jnp.array([1.0, 0.0])
    mpc = MPC(problem, SingleShooting(Ipopt(options={"print_level": 0})), x0=x0)

    P_total = constraint_row_count(problem)
    nz_primal = N * n + H * m
    seeded_lam = jnp.ones(P_total, dtype=jnp.float64) * 5.0
    seeded_mu = jnp.ones(nz_primal, dtype=jnp.float64) * 3.0
    mpc._ws = WarmStart(Z=mpc.warm_start.Z, lam=seeded_lam, mu=seeded_mu)  # noqa: SLF001 -- seeding test Multipliers

    res = mpc.solve()
    assert res.success
    assert res.lam.shape == (P_total,)
    assert res.mu.shape == (nz_primal,)
    np.testing.assert_allclose(res.lam, 0.0)
    np.testing.assert_allclose(res.mu, 0.0)

    # MPC's warm start was updated with canonical zeros
    np.testing.assert_allclose(np.asarray(mpc.warm_start.lam), 0.0)
    np.testing.assert_allclose(np.asarray(mpc.warm_start.mu), 0.0)


def test_multiplier_behavior_b_eq_1_preserved() -> None:
    """For b = 1, adjoint-recovered canonical Multipliers and Multiplier warm starting are preserved."""
    pytest.importorskip("cyipopt")

    N, n, m = 7, 2, 1
    problem = _make_blocked_problem(N=N, b=1, n=n, m=m)

    x0 = jnp.array([1.0, 0.0])
    mpc = MPC(problem, SingleShooting(Ipopt(options={"print_level": 0})), x0=x0)
    res = mpc.solve()

    assert res.success
    assert len(res.lam) > 0
    assert not np.all(res.lam == 0.0)
    np.testing.assert_allclose(np.asarray(mpc.warm_start.lam), res.lam)


def test_failed_solve_expansion_and_multipliers() -> None:
    """Failed single-shooting solves expand block controls, reconstruct canonical Z, and return zero Multipliers."""
    pytest.importorskip("cyipopt")

    N, b, n, m = 7, 3, 2, 1
    H = N - 1
    problem = _make_blocked_problem(N=N, b=b, n=n, m=m)

    x0 = jnp.array([1.0, 0.0])
    res = MPC(problem, SingleShooting(Ipopt(options={"print_level": 0, "max_iter": 0})), x0=x0).solve()

    assert not res.success
    assert res.trajectory.X.shape == (N, n)
    assert res.trajectory.U.shape == (H, m)
    assert len(res.Z) == N * n + H * m

    P_total = constraint_row_count(problem)
    nz_primal = N * n + H * m
    assert res.lam.shape == (P_total,)
    assert res.mu.shape == (nz_primal,)
    np.testing.assert_allclose(res.lam, 0.0)
    np.testing.assert_allclose(res.mu, 0.0)


def test_numerical_agreement_with_multiple_shooting() -> None:
    """Blocked SingleShooting ('lbfgs' and 'dense') and Ipopt multiple shooting agree on cost, controls, and states."""
    pytest.importorskip("cyipopt")

    N, b, n, m = 7, 3, 2, 1

    # Well-conditioned linear-quadratic problem with:
    # 1. Unequal native control bounds across a block
    # 2. Scalar state constraint active mid-block
    constraints = ConstraintList(n=n, m=m, N=N)
    # Block 0: k=0,1,2
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-2.0], u_max=[2.0]), 0)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-1.5], u_max=[3.0]), 1)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-3.0], u_max=[1.2]), 2)
    # Block 1: k=3,4,5
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-1.0], u_max=[2.0]), 3)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-2.0], u_max=[1.5]), 4)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-0.8], u_max=[1.8]), 5)

    # Active mid-block state constraint at k=2 (block 0 covers 0, 1, 2)
    # Require x_2[0] <= 0.895 (unconstrained would be ~0.90)
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([0.895]), sense=NegativeOrthant()),
        2,
    )

    problem = _make_blocked_problem(N=N, b=b, n=n, m=m, constraints=constraints)
    x0 = jnp.array([1.0, -0.5])

    opts = {"tol": 1e-8, "constr_viol_tol": 1e-8, "print_level": 0, "max_iter": 200}

    # 1. Multiple shooting
    res_ms = MPC(problem, Ipopt(options=opts), x0=x0).solve()
    assert res_ms.success

    # 2. Blocked single shooting (dense exact Hessian)
    res_ss_dense = MPC(problem, SingleShooting(Ipopt(options=opts), hessian="dense"), x0=x0).solve()
    assert res_ss_dense.success

    # 3. Blocked single shooting (lbfgs limited-memory)
    res_ss_lbfgs = MPC(problem, SingleShooting(Ipopt(options=opts), hessian="lbfgs"), x0=x0).solve()
    assert res_ss_lbfgs.success

    # Costs agree within solver tolerance
    np.testing.assert_allclose(res_ss_dense.cost, res_ms.cost, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_ss_lbfgs.cost, res_ms.cost, rtol=1e-4, atol=1e-5)

    # Controls agree
    np.testing.assert_allclose(res_ss_dense.trajectory.U, res_ms.trajectory.U, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_ss_lbfgs.trajectory.U, res_ms.trajectory.U, rtol=1e-4, atol=1e-5)

    # States agree
    np.testing.assert_allclose(res_ss_dense.trajectory.X, res_ms.trajectory.X, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_ss_lbfgs.trajectory.X, res_ms.trajectory.X, rtol=1e-4, atol=1e-5)

    # Mid-block state constraint at k=2 is satisfied and active
    x2_dense = float(res_ss_dense.trajectory.X[2, 0])
    assert x2_dense <= 0.8951
    assert abs(x2_dense - 0.895) < 1e-4  # active

    # Single shooting satisfies dynamics Defects to machine precision
    rolled = problem.model.rollout(res_ss_dense.trajectory)
    np.testing.assert_allclose(rolled.X, res_ss_dense.trajectory.X, atol=1e-12)
    rolled_lbfgs = problem.model.rollout(res_ss_lbfgs.trajectory)
    np.testing.assert_allclose(rolled_lbfgs.X, res_ss_lbfgs.trajectory.X, atol=1e-12)
