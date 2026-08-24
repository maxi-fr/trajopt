import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import SecondOrderCone
from trajopt.constraints import (
    ConstraintList,
    ControlBound,
    GoalConstraint,
    NormConstraint,
    StateBound,
)
from trajopt.constraints.rotations import QuatVecEq
from trajopt.costs import (
    DiagonalCost,
    GenericCost,
    LieLQRCost,
    LQRObjective,
    Objective,
    QuadraticCost,
    QuatGeodesicCost,
    TrackingObjective,
)
from trajopt.dynamics import (
    RK4,
    ContinuousDynamics,
    DiscretizedDynamics,
    Euler,
)
from trajopt.expansions import Expansion
from trajopt.models import Cartpole, DubinsCar, Pendulum, Quadrotor
from trajopt.problem import Problem
from trajopt.rotations.quaternion import Quaternion
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z, _z_to_trajectory
from trajopt.transcription.transcription import constraints_and_jac, eval_grad_f, hessian


def test_expansion_data_structure() -> None:
    N, ne, m = 6, 4, 2
    exp = Expansion.zeros(N=N, ne=ne, m=m)

    assert exp.N == N
    assert exp.ne == ne
    assert exp.m == m

    assert exp.A.shape == (N - 1, ne, ne)
    assert exp.B.shape == (N - 1, ne, m)
    assert exp.q.shape == (N, ne)
    assert exp.r.shape == (N - 1, m)
    assert exp.Q.shape == (N, ne, ne)
    assert exp.R.shape == (N - 1, m, m)
    assert exp.H.shape == (N - 1, m, ne)

    # Test arithmetic operations
    exp2 = Expansion(
        A=jnp.ones((N - 1, ne, ne)),
        B=2.0 * jnp.ones((N - 1, ne, m)),
        q=3.0 * jnp.ones((N, ne)),
        r=4.0 * jnp.ones((N - 1, m)),
        Q=5.0 * jnp.ones((N, ne, ne)),
        R=6.0 * jnp.ones((N - 1, m, m)),
        H=7.0 * jnp.ones((N - 1, m, ne)),
    )

    sum_exp = exp + exp2
    np.testing.assert_allclose(sum_exp.A, exp2.A)
    np.testing.assert_allclose(sum_exp.B, exp2.B)
    np.testing.assert_allclose(sum_exp.q, exp2.q)
    np.testing.assert_allclose(sum_exp.r, exp2.r)
    np.testing.assert_allclose(sum_exp.Q, exp2.Q)
    np.testing.assert_allclose(sum_exp.R, exp2.R)
    np.testing.assert_allclose(sum_exp.H, exp2.H)

    diff_exp = sum_exp - exp2
    np.testing.assert_allclose(diff_exp.A, exp.A)

    zeros_like_exp = Expansion.zeros_like(exp2)
    np.testing.assert_allclose(zeros_like_exp.A, jnp.zeros_like(exp2.A))
    np.testing.assert_allclose(zeros_like_exp.q, jnp.zeros_like(exp2.q))

    # Test validation checks
    with pytest.raises(ValueError, match="Dynamics Jacobian A must have 3 dimensions"):
        Expansion(
            A=jnp.zeros((N - 1, ne)),
            B=exp.B,
            q=exp.q,
            r=exp.r,
            Q=exp.Q,
            R=exp.R,
            H=exp.H,
        )

    with pytest.raises(ValueError, match="inconsistent with expected"):
        Expansion(
            A=jnp.zeros((N, ne, ne)),  # wrong horizon length
            B=exp.B,
            q=exp.q,
            r=exp.r,
            Q=exp.Q,
            R=exp.R,
            H=exp.H,
        )


def test_expansion_jit_pytree() -> None:
    N, ne, m = 4, 3, 2
    exp = Expansion.zeros(N=N, ne=ne, m=m)

    @jax.jit
    def scale_expansion(e: Expansion, factor: float) -> Expansion:
        return Expansion(
            A=e.A * factor,
            B=e.B * factor,
            q=e.q * factor,
            r=e.r * factor,
            Q=e.Q * factor,
            R=e.R * factor,
            H=e.H * factor,
        )

    scaled = scale_expansion(exp, 2.5)
    assert scaled.A.shape == (N - 1, ne, ne)
    assert scaled.q.shape == (N, ne)


def test_dynamics_expansion_discrete_finite_differences() -> None:
    model = Cartpole()
    discrete = DiscretizedDynamics(model, RK4())

    n, m, N = model.n, model.m, 5
    dt = 0.05

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
    )

    exp = discrete.dynamics_expansion(traj)

    assert exp.A.shape == (N - 1, n, n)
    assert exp.B.shape == (N - 1, n, m)
    assert exp.q.shape == (N, n)
    assert exp.r.shape == (N - 1, m)
    assert exp.Q.shape == (N, n, n)
    assert exp.R.shape == (N - 1, m, m)
    assert exp.H.shape == (N - 1, m, n)

    # Cost fields should be zero
    np.testing.assert_allclose(exp.q, jnp.zeros((N, n)))
    np.testing.assert_allclose(exp.r, jnp.zeros((N - 1, m)))
    np.testing.assert_allclose(exp.Q, jnp.zeros((N, n, n)))
    np.testing.assert_allclose(exp.R, jnp.zeros((N - 1, m, m)))
    np.testing.assert_allclose(exp.H, jnp.zeros((N - 1, m, n)))

    # Central finite differences verification of discrete dynamics Jacobians
    eps = 1e-6
    for k in range(N - 1):
        xk = jnp.array(X_np[k])
        uk = jnp.array(U_np[k])
        tk = t_np[k]
        dtk = dt

        # State Jacobian A_k
        A_fd = np.zeros((n, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus = np.array(discrete.discrete_dynamics(xk + dx, uk, tk, dtk))
            f_minus = np.array(discrete.discrete_dynamics(xk - dx, uk, tk, dtk))
            A_fd[:, i] = (f_plus - f_minus) / (2.0 * eps)

        # Control Jacobian B_k
        B_fd = np.zeros((n, m))
        for j in range(m):
            du = np.zeros(m)
            du[j] = eps
            f_plus = np.array(discrete.discrete_dynamics(xk, uk + du, tk, dtk))
            f_minus = np.array(discrete.discrete_dynamics(xk, uk - du, tk, dtk))
            B_fd[:, j] = (f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(np.array(exp.A[k]), A_fd, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(exp.B[k]), B_fd, rtol=1e-5, atol=1e-5)


def test_dynamics_expansion_problem_wrapper() -> None:
    model = Pendulum()
    obj = LQRObjective(jnp.ones(2), jnp.ones(1), jnp.ones(2), jnp.zeros(2), 6)
    problem = Problem(model=model, obj=obj, integrator=Euler())

    traj = Trajectory(
        X=jnp.zeros((6, 2)),
        U=jnp.zeros((5, 1)),
        t=jnp.linspace(0.0, 0.5, 6),
        dt=jnp.ones(5) * 0.1,
    )

    exp = problem.dynamics_expansion(traj)
    assert exp.A.shape == (5, 2, 2)
    assert exp.B.shape == (5, 2, 1)


def test_cost_expansion_diagonal_lqr() -> None:
    n, m, N = 3, 2, 6
    Q_diag = jnp.array([2.0, 1.5, 0.5])
    R_diag = jnp.array([0.8, 1.2])
    Qf_diag = jnp.array([10.0, 8.0, 5.0])
    xf = jnp.array([1.0, -0.5, 2.0])
    uf = jnp.array([0.1, -0.2])

    obj = LQRObjective(
        Q=Q_diag,
        R=R_diag,
        Qf=Qf_diag,
        xf=xf,
        N=N,
        uf=uf,
    )

    rng = np.random.default_rng(7)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    traj = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.linspace(0.0, 0.5, N),
        dt=jnp.diff(jnp.linspace(0.0, 0.5, N)),
    )

    exp = obj.cost_expansion(traj)

    assert exp.q.shape == (N, n)
    assert exp.r.shape == (N - 1, m)
    assert exp.Q.shape == (N, n, n)
    assert exp.R.shape == (N - 1, m, m)
    assert exp.H.shape == (N - 1, m, n)
    np.testing.assert_allclose(exp.A, jnp.zeros((N - 1, n, n)))

    # Analytical verification
    for k in range(N - 1):
        xk = X_np[k]
        uk = U_np[k]
        expected_qk = np.array(Q_diag) * (xk - np.array(xf))
        expected_rk = np.array(R_diag) * (uk - np.array(uf))
        expected_Qk = np.diag(np.array(Q_diag))
        expected_Rk = np.diag(np.array(R_diag))
        expected_Hk = np.zeros((m, n))

        np.testing.assert_allclose(np.array(exp.q[k]), expected_qk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.r[k]), expected_rk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.Q[k]), expected_Qk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.R[k]), expected_Rk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.H[k]), expected_Hk, rtol=1e-12, atol=1e-12)

    # Terminal knot point
    expected_q_term = np.array(Qf_diag) * (X_np[-1] - np.array(xf))
    expected_Q_term = np.diag(np.array(Qf_diag))
    np.testing.assert_allclose(np.array(exp.q[-1]), expected_q_term, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(exp.Q[-1]), expected_Q_term, rtol=1e-12, atol=1e-12)


def test_cost_expansion_dense_quadratic_cross_coupling() -> None:
    n, m, N = 3, 2, 5
    rng = np.random.default_rng(99)
    A_Q = rng.standard_normal((n, n))
    Q_mat = jnp.array(A_Q.T @ A_Q + 2.0 * np.eye(n))
    A_R = rng.standard_normal((m, m))
    R_mat = jnp.array(A_R.T @ A_R + 1.0 * np.eye(m))
    H_mat = jnp.array([[0.3, -0.2, 0.4], [0.1, 0.5, -0.1]])
    q_vec = jnp.array([0.5, -0.2, 0.1])
    r_vec = jnp.array([-0.3, 0.4])
    c_val = 1.25

    stage_cost = QuadraticCost(
        Q=Q_mat,
        R=R_mat,
        H=H_mat,
        q=q_vec,
        r=r_vec,
        c=c_val,
    )
    term_cost = QuadraticCost(
        Q=Q_mat * 2.0,
        q=q_vec * 2.0,
        c=c_val,
        terminal=True,
        m=m,
    )
    obj = Objective(stage_cost=stage_cost, terminal_cost=term_cost, N=N)

    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    traj = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.linspace(0.0, 0.4, N),
        dt=jnp.diff(jnp.linspace(0.0, 0.4, N)),
    )

    exp = obj.cost_expansion(traj)

    for k in range(N - 1):
        xk = X_np[k]
        uk = U_np[k]
        expected_qk = np.array(Q_mat) @ xk + np.array(H_mat).T @ uk + np.array(q_vec)
        expected_rk = np.array(R_mat) @ uk + np.array(H_mat) @ xk + np.array(r_vec)

        np.testing.assert_allclose(np.array(exp.q[k]), expected_qk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.r[k]), expected_rk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.Q[k]), np.array(Q_mat), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.R[k]), np.array(R_mat), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp.H[k]), np.array(H_mat), rtol=1e-12, atol=1e-12)

    expected_q_term = 2.0 * np.array(Q_mat) @ X_np[-1] + 2.0 * np.array(q_vec)
    expected_Q_term = 2.0 * np.array(Q_mat)
    np.testing.assert_allclose(np.array(exp.q[-1]), expected_q_term, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(exp.Q[-1]), expected_Q_term, rtol=1e-12, atol=1e-12)


def _setup_generic_cost_fixture() -> tuple[Objective, Trajectory, np.ndarray, np.ndarray]:
    n, m, N = 2, 1, 5

    def nonlinear_stage(x: jax.Array, u: jax.Array | None, t: float | jax.Array = 0.0) -> jax.Array:
        del t
        assert u is not None
        return jnp.sin(x[0]) + (x[1] ** 4) + (u[0] ** 2) * jnp.cos(u[0]) + x[0] * u[0]

    def nonlinear_term(x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        del u, t
        return (x[0] ** 3) + jnp.exp(x[1])

    stage = GenericCost(cost_fn=nonlinear_stage, n=n, m=m, terminal=False)
    term = GenericCost(cost_fn=nonlinear_term, n=n, m=m, terminal=True)
    obj = Objective(stage_cost=stage, terminal_cost=term, N=N)

    rng = np.random.default_rng(55)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    traj = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.linspace(0.0, 0.4, N),
        dt=jnp.diff(jnp.linspace(0.0, 0.4, N)),
    )
    return obj, traj, X_np, U_np


def test_cost_expansion_generic_stage_gradients() -> None:
    obj, traj, X_np, U_np = _setup_generic_cost_fixture()
    exp = obj.cost_expansion(traj)
    eps = 1e-5
    n, m, N = 2, 1, 5
    stage = obj.stage_cost

    for k in range(N - 1):
        xk = jnp.array(X_np[k])
        uk = jnp.array(U_np[k])

        q_fd = np.zeros(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            c_plus = float(stage.evaluate(xk + dx, uk))
            c_minus = float(stage.evaluate(xk - dx, uk))
            q_fd[i] = (c_plus - c_minus) / (2.0 * eps)
        np.testing.assert_allclose(np.array(exp.q[k]), q_fd, rtol=1e-5, atol=1e-5)

        r_fd = np.zeros(m)
        for j in range(m):
            du = np.zeros(m)
            du[j] = eps
            c_plus = float(stage.evaluate(xk, uk + du))
            c_minus = float(stage.evaluate(xk, uk - du))
            r_fd[j] = (c_plus - c_minus) / (2.0 * eps)
        np.testing.assert_allclose(np.array(exp.r[k]), r_fd, rtol=1e-5, atol=1e-5)


def test_cost_expansion_generic_stage_hessians() -> None:
    obj, traj, X_np, U_np = _setup_generic_cost_fixture()
    exp = obj.cost_expansion(traj)
    eps = 1e-5
    n, m, N = 2, 1, 5
    stage = obj.stage_cost

    for k in range(N - 1):
        xk = jnp.array(X_np[k])
        uk = jnp.array(U_np[k])

        Q_fd = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dxi = np.zeros(n)
                dxj = np.zeros(n)
                dxi[i] = eps
                dxj[j] = eps
                f_pp = float(stage.evaluate(xk + dxi + dxj, uk))
                f_pm = float(stage.evaluate(xk + dxi - dxj, uk))
                f_mp = float(stage.evaluate(xk - dxi + dxj, uk))
                f_mm = float(stage.evaluate(xk - dxi - dxj, uk))
                Q_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
        np.testing.assert_allclose(np.array(exp.Q[k]), Q_fd, rtol=1e-4, atol=1e-4)

        R_fd = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                dui = np.zeros(m)
                duj = np.zeros(m)
                dui[i] = eps
                duj[j] = eps
                f_pp = float(stage.evaluate(xk, uk + dui + duj))
                f_pm = float(stage.evaluate(xk, uk + dui - duj))
                f_mp = float(stage.evaluate(xk, uk - dui + duj))
                f_mm = float(stage.evaluate(xk, uk - dui - duj))
                R_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
        np.testing.assert_allclose(np.array(exp.R[k]), R_fd, rtol=1e-4, atol=1e-4)

        H_fd = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                du = np.zeros(m)
                dx = np.zeros(n)
                du[i] = eps
                dx[j] = eps
                f_pp = float(stage.evaluate(xk + dx, uk + du))
                f_pm = float(stage.evaluate(xk - dx, uk + du))
                f_mp = float(stage.evaluate(xk + dx, uk - du))
                f_mm = float(stage.evaluate(xk - dx, uk - du))
                H_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
        np.testing.assert_allclose(np.array(exp.H[k]), H_fd, rtol=1e-4, atol=1e-4)


def test_cost_expansion_generic_terminal() -> None:
    obj, traj, X_np, _ = _setup_generic_cost_fixture()
    exp = obj.cost_expansion(traj)
    eps = 1e-5
    n = 2
    term = obj.terminal_cost

    x_term = jnp.array(X_np[-1])
    q_term_fd = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        c_plus = float(term.evaluate(x_term + dx, None))
        c_minus = float(term.evaluate(x_term - dx, None))
        q_term_fd[i] = (c_plus - c_minus) / (2.0 * eps)
    np.testing.assert_allclose(np.array(exp.q[-1]), q_term_fd, rtol=1e-5, atol=1e-5)

    Q_term_fd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dxi = np.zeros(n)
            dxj = np.zeros(n)
            dxi[i] = eps
            dxj[j] = eps
            f_pp = float(term.evaluate(x_term + dxi + dxj, None))
            f_pm = float(term.evaluate(x_term + dxi - dxj, None))
            f_mp = float(term.evaluate(x_term - dxi + dxj, None))
            f_mm = float(term.evaluate(x_term - dxi - dxj, None))
            Q_term_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
    np.testing.assert_allclose(np.array(exp.Q[-1]), Q_term_fd, rtol=1e-4, atol=1e-4)


def test_augmented_lagrangian_expansion_goal_equality() -> None:
    n, m, N = 3, 2, 6
    xf = jnp.array([1.0, 2.0, -1.0])
    gcon = GoalConstraint(n=n, xf=xf, m=m)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(gcon, N - 1)
    built_cons = cl.build()

    obj = LQRObjective(jnp.ones(n), jnp.ones(m), jnp.ones(n), xf, N)
    rng = np.random.default_rng(10)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    traj = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.linspace(0.0, 0.5, N),
        dt=jnp.diff(jnp.linspace(0.0, 0.5, N)),
    )

    base_exp = obj.cost_expansion(traj)

    lam_term = jnp.array([0.5, -0.8, 1.2])
    mu = 4.0
    lam_list = [jnp.zeros(0) for _ in range(N - 1)] + [lam_term]

    al_exp = built_cons.augmented_lagrangian_expansion(traj, base_exp, lam=lam_list, mu=mu)

    # Stages 0..N-2 should have no AL changes
    for k in range(N - 1):
        np.testing.assert_allclose(al_exp.q[k], base_exp.q[k])
        np.testing.assert_allclose(al_exp.Q[k], base_exp.Q[k])
        np.testing.assert_allclose(al_exp.r[k], base_exp.r[k])
        np.testing.assert_allclose(al_exp.R[k], base_exp.R[k])
        np.testing.assert_allclose(al_exp.H[k], base_exp.H[k])

    # Terminal stage should have exact AL contribution
    c_term = X_np[-1] - np.array(xf)
    expected_q_al = 2.0 * np.array(lam_term) + mu * c_term
    expected_Q_al = mu * np.eye(n)

    q_diff = np.array(al_exp.q[-1] - base_exp.q[-1])
    Q_diff = np.array(al_exp.Q[-1] - base_exp.Q[-1])

    np.testing.assert_allclose(q_diff, expected_q_al, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Q_diff, expected_Q_al, rtol=1e-12, atol=1e-12)


def test_augmented_lagrangian_expansion_bounds_inequality() -> None:
    n, m, N = 2, 1, 5
    x_max = jnp.array([2.0, 1.0])
    u_max = jnp.array([0.5])

    sbnd = StateBound(n=n, x_max=x_max, m=m)
    ubnd = ControlBound(m=m, u_max=u_max, n=n)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(sbnd, range(N - 1))
    cl.add_constraint(ubnd, range(N - 1))
    built_cons = cl.build()

    obj = LQRObjective(jnp.ones(n), jnp.ones(m), jnp.ones(n), jnp.zeros(n), N)

    # Choose states such that knot 0 violates bounds (active) and knot 1 is well below (inactive)
    X = jnp.array(
        [
            [2.5, 1.2],  # Violates x_max (active)
            [-1.0, -0.5],  # Well below x_max (inactive)
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    U = jnp.array(
        [
            [1.0],  # Violates u_max (active)
            [-2.0],  # Inactive
            [0.0],
            [0.0],
        ]
    )
    traj = Trajectory(X=X, U=U, t=jnp.linspace(0.0, 0.4, N), dt=jnp.diff(jnp.linspace(0.0, 0.4, N)))

    base_exp = obj.cost_expansion(traj)
    mu = 10.0
    lam_stage = jnp.array([0.1, 0.2, 0.3])
    lam_list = [lam_stage] * (N - 1) + [jnp.zeros(0)]

    al_exp = built_cons.augmented_lagrangian_expansion(traj, base_exp, lam=lam_list, mu=mu)

    # Knot 0 (active violation) -> should have significant positive Hessian contributions
    Q_al_0 = np.array(al_exp.Q[0] - base_exp.Q[0])
    R_al_0 = np.array(al_exp.R[0] - base_exp.R[0])
    np.testing.assert_allclose(Q_al_0, mu * np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(R_al_0, mu * np.eye(m), rtol=1e-12, atol=1e-12)

    # Knot 1 (inactive) -> shift is negative, max(0, shift) = 0 -> AL contribution is 0
    Q_al_1 = np.array(al_exp.Q[1] - base_exp.Q[1])
    R_al_1 = np.array(al_exp.R[1] - base_exp.R[1])
    np.testing.assert_allclose(Q_al_1, np.zeros((n, n)), atol=1e-12)
    np.testing.assert_allclose(R_al_1, np.zeros((m, m)), atol=1e-12)


def test_augmented_lagrangian_expansion_soc() -> None:
    n, m, N = 3, 1, 4
    norm_con = NormConstraint(n=n, m=m, val=2.0, inds=[0, 1], sense=SecondOrderCone())
    assert isinstance(norm_con.cone, SecondOrderCone)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(norm_con, range(N - 1))
    built_cons = cl.build()

    obj = LQRObjective(jnp.ones(n), jnp.ones(m), jnp.ones(n), jnp.zeros(n), N)

    # Knot 0 outside cone (||v|| = 3.0 > s = 2.0) -> active penalty
    X = jnp.array(
        [
            [3.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    U = jnp.zeros((N - 1, m))
    traj = Trajectory(X=X, U=U, t=jnp.linspace(0.0, 0.3, N), dt=jnp.diff(jnp.linspace(0.0, 0.3, N)))

    base_exp = obj.cost_expansion(traj)
    mu = 5.0
    lam_knot = jnp.array([0.1, 0.2, 0.3])
    lam_list = [lam_knot] * (N - 1) + [jnp.zeros(0)]

    al_exp = built_cons.augmented_lagrangian_expansion(traj, base_exp, lam=lam_list, mu=mu)

    # Verify that AL gradient and Hessian at knot 0 match finite differences of AL penalty
    eps = 1e-5
    xk = X[0]
    uk = U[0]

    def al_pen(x_in: jax.Array, u_in: jax.Array) -> jax.Array:
        val = norm_con.evaluate(x_in, u_in, 0.0)
        shifted = val + lam_knot / mu
        proj = norm_con.cone.project_dual(shifted)
        return jnp.dot(lam_knot, proj) + 0.5 * mu * jnp.dot(proj, proj)

    # Gradient q_al
    q_fd = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        p_plus = float(al_pen(xk + dx, uk))
        p_minus = float(al_pen(xk - dx, uk))
        q_fd[i] = (p_plus - p_minus) / (2.0 * eps)

    q_al_actual = np.array(al_exp.q[0] - base_exp.q[0])
    np.testing.assert_allclose(q_al_actual, q_fd, rtol=1e-4, atol=1e-4)

    # Hessian Q_al
    Q_fd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dxi = np.zeros(n)
            dxj = np.zeros(n)
            dxi[i] = eps
            dxj[j] = eps
            p_pp = float(al_pen(xk + dxi + dxj, uk))
            p_pm = float(al_pen(xk + dxi - dxj, uk))
            p_mp = float(al_pen(xk - dxi + dxj, uk))
            p_mm = float(al_pen(xk - dxi - dxj, uk))
            Q_fd[i, j] = (p_pp - p_pm - p_mp + p_mm) / (4.0 * eps * eps)

    Q_al_actual = np.array(al_exp.Q[0] - base_exp.Q[0])
    np.testing.assert_allclose(Q_al_actual, Q_fd, rtol=1e-4, atol=1e-4)


def test_error_coordinates_with_mock_attitude_jacobian() -> None:
    # Test error coordinates on a mock system where n=3, ne=2, and G(x) is 3x2 with G^T G = I_2
    class MockManifoldModel(ContinuousDynamics):
        def __init__(self) -> None:
            super().__init__(n=3, m=1, ne=2)

        def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
            del t
            return jnp.array([x[0] + u[0], x[1] * 2.0, x[2] + x[0] ** 2])

        def errstate_jacobian(self, x: jax.Array) -> jax.Array:
            del x
            # Constant 3x2 matrix with orthonormal columns
            return jnp.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )

    model = MockManifoldModel()
    discrete = DiscretizedDynamics(model, RK4())
    N = 4
    n, ne, m = model.n, model.ne, model.m

    X = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5],
        ]
    )
    U = jnp.array([[0.5], [-0.2], [0.8]])
    traj = Trajectory(X=X, U=U, t=jnp.linspace(0.0, 0.3, N), dt=jnp.diff(jnp.linspace(0.0, 0.3, N)))

    # 1. Dynamics expansion in error coordinates
    exp_dyn = discrete.dynamics_expansion(traj)
    assert exp_dyn.A.shape == (N - 1, ne, ne)
    assert exp_dyn.B.shape == (N - 1, ne, m)
    assert exp_dyn.q.shape == (N, ne)
    assert exp_dyn.Q.shape == (N, ne, ne)

    G = model.errstate_jacobian(X[0])
    for k in range(N - 1):
        Ak_state = discrete.state_jacobian(X[k], U[k], traj.t[k], traj.dt[k])
        Bk_state = discrete.control_jacobian(X[k], U[k], traj.t[k], traj.dt[k])
        expected_A_bar = G.T @ Ak_state @ G
        expected_B_bar = G.T @ Bk_state
        np.testing.assert_allclose(exp_dyn.A[k], expected_A_bar, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exp_dyn.B[k], expected_B_bar, rtol=1e-12, atol=1e-12)

    # 2. Cost expansion in error coordinates
    obj = LQRObjective(jnp.ones(n), jnp.ones(m), jnp.ones(n), jnp.zeros(n), N)
    problem = Problem(model=model, obj=obj, N=N)
    exp_cost = problem.cost_expansion(traj)

    assert exp_cost.q.shape == (N, ne)
    assert exp_cost.Q.shape == (N, ne, ne)
    assert exp_cost.r.shape == (N - 1, m)
    assert exp_cost.R.shape == (N - 1, m, m)
    assert exp_cost.H.shape == (N - 1, m, ne)

    for k in range(N):
        expected_qk = G.T @ (jnp.ones(n) * X[k])
        expected_Qk = G.T @ jnp.diag(jnp.ones(n)) @ G
        np.testing.assert_allclose(exp_cost.q[k], expected_qk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exp_cost.Q[k], expected_Qk, rtol=1e-12, atol=1e-12)


def _sample_quadrotor_trajectory(n_knots: int = 5) -> tuple[Quadrotor, Trajectory]:
    """Helper to generate a Quadrotor model and a valid sample trajectory."""
    model = Quadrotor()
    m = 4
    dt = 0.05
    rng = np.random.default_rng(123)
    X_list = []
    for _ in range(n_knots):
        r = rng.standard_normal(3)
        q = rng.standard_normal(4)
        q = q / np.linalg.norm(q)
        v = rng.standard_normal(3)
        omega = rng.standard_normal(3)
        X_list.append(np.concatenate([r, q, v, omega]))
    X = jnp.array(np.stack(X_list, axis=0))
    U = jnp.array(rng.standard_normal((n_knots - 1, m)))
    t = jnp.linspace(0.0, dt * (n_knots - 1), n_knots)
    traj = Trajectory(X=X, U=U, t=t, dt=jnp.diff(t))
    return model, traj


def test_quadrotor_sandwiched_dynamics_expansion() -> None:
    """Assert Quadrotor dynamics expansions are properly sandwiched in 12-dimensional error state coordinates."""
    model, traj = _sample_quadrotor_trajectory(n_knots=5)
    discrete = DiscretizedDynamics(model, RK4())
    ne, m, N = 12, 4, 5

    exp_dyn = discrete.dynamics_expansion(traj)
    assert exp_dyn.A.shape == (N - 1, ne, ne)
    assert exp_dyn.B.shape == (N - 1, ne, m)
    assert exp_dyn.ne == ne

    for k in range(N - 1):
        G_k = model.errstate_jacobian(traj.X[k])
        G_next = model.errstate_jacobian(traj.X[k + 1])
        assert G_k.shape == (13, 12)
        assert G_next.shape == (13, 12)

        Ak_state = discrete.state_jacobian(traj.X[k], traj.U[k], traj.t[k], traj.dt[k])
        Bk_state = discrete.control_jacobian(traj.X[k], traj.U[k], traj.t[k], traj.dt[k])

        A_bar = G_next.T @ Ak_state @ G_k
        B_bar = G_next.T @ Bk_state
        assert A_bar.shape == (12, 12)
        assert B_bar.shape == (12, 4)

        np.testing.assert_allclose(exp_dyn.A[k], A_bar, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exp_dyn.B[k], B_bar, rtol=1e-12, atol=1e-12)


def test_quadrotor_sandwiched_cost_and_al_expansion() -> None:
    """Assert Quadrotor cost and AL expansions are properly sandwiched in 12-dimensional error coordinates."""
    model, traj = _sample_quadrotor_trajectory(n_knots=5)
    n, ne, m, N = 13, 12, 4, 5

    xf = traj.X[-1]
    uf = jnp.zeros(m)
    stage_cost = LieLQRCost(Q=jnp.ones(13), R=jnp.ones(m), xf=xf, uf=uf, w=2.0)
    term_cost = LieLQRCost(Q=jnp.ones(13), R=jnp.ones(m), xf=xf, terminal=True, w=2.0)
    obj = Objective(stage_cost=stage_cost, terminal_cost=term_cost, N=N)
    problem = Problem(model=model, obj=obj, N=N)

    exp_cost = problem.cost_expansion(traj)
    assert exp_cost.q.shape == (N, ne)
    assert exp_cost.Q.shape == (N, ne, ne)
    assert exp_cost.r.shape == (N - 1, m)
    assert exp_cost.R.shape == (N - 1, m, m)
    assert exp_cost.H.shape == (N - 1, m, ne)

    cons = ConstraintList(n=n, m=m, N=N)
    q_target = xf[3:7]
    cons.add_constraint(QuatVecEq(n=n, qf=q_target, m=m), N - 1)
    built_cons = cons.build()

    lam_list = [jnp.zeros(built_cons.p[k]) for k in range(N)]
    lam_list[-1] = jnp.array([0.5, -0.5, 0.2])
    al_exp = problem.augmented_lagrangian_expansion(traj, exp_cost, lam=lam_list, mu=10.0)
    assert al_exp.q.shape == (N, ne)
    assert al_exp.Q.shape == (N, ne, ne)

    assert al_exp.r.shape == (N - 1, m)
    assert al_exp.R.shape == (N - 1, m, m)
    assert al_exp.H.shape == (N - 1, m, ne)


def _euclidean_dubins_problem(weights: str) -> tuple[Problem, Trajectory, float, jax.Array]:
    """Build a Dubins problem and a random trajectory to expand about, plus its (t0, dt).

    DubinsCar is Euclidean (ne == n, G = I), which is what makes the engine's error-coordinate
    output directly comparable to the transcription's state-coordinate derivatives. `weights`
    selects the stacked DiagonalCost or the stacked QuadraticCost closed form inside
    `_stage_cost_expansion`. Every weight varies knot to knot, and the dense case carries a
    nonzero cross term, so a comparison that lost track of the knot index or dropped the cross
    block could not still pass.
    """
    N, dt, t0 = 6, 0.1, 0.3
    model = DubinsCar()
    n, m = int(model.n), int(model.m)

    rng = np.random.default_rng(20260824)
    scale = 1.0 + jnp.arange(N - 1, dtype=jnp.float64)[:, None]
    Q_diag = scale * jnp.array([1.0, 2.0, 0.5])
    R_diag = scale * jnp.array([0.3, 0.7])
    q_lin = jnp.asarray(rng.standard_normal((N - 1, n)))
    r_lin = jnp.asarray(rng.standard_normal((N - 1, m)))
    Qf_diag = jnp.array([10.0, 20.0, 5.0])

    if weights == "diagonal":
        stage_cost = DiagonalCost(Q=Q_diag, R=R_diag, q=q_lin, r=r_lin)
        terminal_cost = DiagonalCost(Q=Qf_diag, terminal=True, m=m)
    else:
        cross = jnp.asarray(rng.standard_normal((N - 1, m, n)))
        stage_cost = QuadraticCost(
            Q=jax.vmap(jnp.diag)(Q_diag),
            R=jax.vmap(jnp.diag)(R_diag),
            H=cross,
            q=q_lin,
            r=r_lin,
        )
        terminal_cost = QuadraticCost(Q=jnp.diag(Qf_diag), terminal=True, m=m)
    obj = Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)

    problem = Problem(model=model, obj=obj, constraints=ConstraintList(n, m, N), N=N, integrator=RK4())

    dt_arr = jnp.full((N - 1,), dt)
    traj = Trajectory(
        X=jnp.asarray(rng.standard_normal((N, n))),
        U=jnp.asarray(rng.standard_normal((N - 1, m))),
        t=t0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(dt_arr)]),
        dt=dt_arr,
    )
    return problem, traj, t0, dt_arr


def _unpack_tril(values: np.ndarray, d: int) -> np.ndarray:
    """Rebuild a symmetric (d, d) matrix from its lower-triangular nonzeros in row-major order."""
    out = np.zeros((d, d))
    rows, cols = np.tril_indices(d)
    out[rows, cols] = values
    return out + np.tril(out, -1).T


@pytest.mark.parametrize("weights", ["diagonal", "dense"])
def test_engine_agrees_with_transcription_on_euclidean_derivatives(weights: str) -> None:
    """Verify the expansion engine and the NLP transcription compute the same derivatives at G = I.

    The engine is the seam the native solvers will consume and nothing consumes it yet, so its
    closed-form cost paths have had no oracle but themselves. Where the error dimension equals
    the state dimension the two are computing the same quantities in the same coordinates, which
    makes the transcription -- exercised on every Ipopt solve -- exactly that missing oracle.
    """
    problem, traj, t0, dt = _euclidean_dubins_problem(weights)
    N, n, m = int(problem.N), int(problem.model.n), int(problem.model.m)
    Z = _trajectory_to_z(traj.X, traj.U)

    # 1. Cost gradients against eval_grad_f, unpacked from the flat Z layout.
    cost_exp = problem.cost_expansion(traj)
    grad_X, grad_U = _z_to_trajectory(eval_grad_f(problem, Z, t0, dt), N, n, m)
    np.testing.assert_allclose(np.asarray(cost_exp.q), np.asarray(grad_X), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cost_exp.r), np.asarray(grad_U), rtol=1e-12, atol=1e-12)

    # 2. Cost Hessian blocks against the Lagrangian Hessian with the multipliers zeroed, which
    #    leaves only the objective term.
    hess = np.asarray(hessian(problem, Z, t0=t0, dt=dt, obj_factor=1.0, lam=None))
    stage_nnz = (n + m) * (n + m + 1) // 2
    for k in range(N - 1):
        Hk = _unpack_tril(hess[k * stage_nnz : (k + 1) * stage_nnz], n + m)
        np.testing.assert_allclose(np.asarray(cost_exp.Q[k]), Hk[:n, :n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(cost_exp.R[k]), Hk[n:, n:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(cost_exp.H[k]), Hk[n:, :n], rtol=1e-12, atol=1e-12)
    H_term = _unpack_tril(hess[(N - 1) * stage_nnz :], n)
    np.testing.assert_allclose(np.asarray(cost_exp.Q[-1]), H_term, rtol=1e-12, atol=1e-12)

    # 3. Dynamics Jacobians against the defect rows of the constraint Jacobian, which carry
    #    -[A_k, B_k] followed by the identity block of x_{k+1}.
    dyn_exp = problem.dynamics_expansion(traj)
    _, jac = constraints_and_jac(problem, Z, traj.X[0], t0, dt)
    jac_np = np.asarray(jac)
    offset = n * n  # the initial-condition identity block
    for k in range(N - 1):
        AB = -jac_np[offset : offset + n * (n + m)].reshape(n, n + m)
        offset += n * (n + m)
        np.testing.assert_allclose(np.asarray(dyn_exp.A[k]), AB[:, :n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(dyn_exp.B[k]), AB[:, n:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jac_np[offset : offset + n * n].reshape(n, n), np.eye(n))
        offset += n * n
    assert offset == len(jac_np), "the constraint Jacobian carries rows this walk did not account for"
