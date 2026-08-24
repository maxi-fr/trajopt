import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.benchmarks import quadrotor_obstacle_benchmark
from trajopt.cones import NegativeOrthant, PositiveOrthant, ZeroCone
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import CircleConstraint
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.layout import (
    _trajectory_to_z,
    _z_to_trajectory,
    constraint_bounds,
    primal_bounds,
)
from trajopt.transcription.sparsity import (
    hessian_sparsity_pattern,
    jacobian_sparsity_pattern,
)
from trajopt.transcription.transcription import (
    constraints_and_jac,
    cost_and_grad,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_h,
    eval_jac_g,
    hessian,
)


def test_primal_vector_interleaving_roundtrip() -> None:
    """Verify that Z interleaves states and controls with trailing terminal state and round-trips."""
    N = 4
    n = 3
    m = 2

    # Deterministic test arrays
    X = jnp.arange(N * n, dtype=jnp.float64).reshape((N, n))
    U = jnp.arange(100, 100 + (N - 1) * m, dtype=jnp.float64).reshape((N - 1, m))

    Z = _trajectory_to_z(X, U)
    expected_len = N * n + (N - 1) * m
    assert Z.shape == (expected_len,)

    # Verify interleaving: [x0, u0, x1, u1, x2, u2, x3]
    expected_Z = jnp.concatenate(
        [
            X[0],
            U[0],
            X[1],
            U[1],
            X[2],
            U[2],
            X[3],
        ]
    )
    np.testing.assert_allclose(Z, expected_Z)

    # Round trip
    X_rec, U_rec = _z_to_trajectory(Z, N, n, m)
    np.testing.assert_allclose(X_rec, X)
    np.testing.assert_allclose(U_rec, U)


def test_primal_bounds() -> None:
    """Verify primal variable bounds extraction from ConstraintList."""
    N = 3
    n = 2
    m = 1
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(StateBound(n=n, m=m, x_min=[-1.0, -2.0], x_max=[1.0, 2.0]), range(N))
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))

    # Build dummy problem
    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    zL, zU = primal_bounds(prob)
    expected_zL = np.array([-1.0, -2.0, -5.0, -1.0, -2.0, -5.0, -1.0, -2.0])
    expected_zU = np.array([1.0, 2.0, 5.0, 1.0, 2.0, 5.0, 1.0, 2.0])
    np.testing.assert_allclose(zL, expected_zL)
    np.testing.assert_allclose(zU, expected_zU)


def test_constraint_bounds() -> None:
    """Verify constraint lower and upper bounds for initial condition, dynamics, and stage constraints."""
    N = 3
    n = 2
    m = 1
    cl = ConstraintList(n=n, m=m, N=N)
    # CircleConstraint has NegativeOrthant (c <= 0 -> gL = -inf, gU = 0)
    cl.add_constraint(CircleConstraint(n=n, xc=[0.0], yc=[0.0], radius=[0.5], m=m), 0)
    # GoalConstraint has ZeroCone (c == 0 -> gL = 0, gU = 0)
    cl.add_constraint(GoalConstraint(n=n, xf=[np.pi, 0.0]), N - 1)

    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    gL, gU = constraint_bounds(prob)
    # P = n (x0) + (N-1)*n (dyn) + sum(p) = 2 + 4 + 1 + 2 = 9
    assert len(gL) == 9
    assert len(gU) == 9

    # Initial condition (2 zeros)
    np.testing.assert_allclose(gL[:2], [0.0, 0.0])
    np.testing.assert_allclose(gU[:2], [0.0, 0.0])
    # Dynamics defect 0 (2 zeros)
    np.testing.assert_allclose(gL[2:4], [0.0, 0.0])
    np.testing.assert_allclose(gU[2:4], [0.0, 0.0])
    # Stage constraint 0: CircleConstraint (p=1, NegativeOrthant: [-inf, 0])
    assert gL[4] == -np.inf
    assert gU[4] == 0.0
    # Dynamics defect 1 (2 zeros)
    np.testing.assert_allclose(gL[5:7], [0.0, 0.0])
    np.testing.assert_allclose(gU[5:7], [0.0, 0.0])
    # Terminal constraint (p=2, ZeroCone: [0, 0])
    np.testing.assert_allclose(gL[7:9], [0.0, 0.0])
    np.testing.assert_allclose(gU[7:9], [0.0, 0.0])


def test_sparsity_pattern_dimensions_pure_function() -> None:
    """Verify that build-time COO sparsity patterns are pure functions of dimensions."""
    N = 4
    n = 2
    m = 1
    p = (1, 0, 1, 2)  # constraints at each knot point

    jac_rows, jac_cols = jacobian_sparsity_pattern(N, n, m, p)
    assert isinstance(jac_rows, np.ndarray)
    assert isinstance(jac_cols, np.ndarray)
    assert jac_rows.ndim == 1
    assert jac_cols.ndim == 1
    assert len(jac_rows) == len(jac_cols)

    # Expected nonzeros count breakdown:
    # Initial state condition block has dimension n*n = 4
    # Dynamics blocks across horizon have (N-1) * (n*(n+m) + n*n) = 30 nonzeros
    # Stage constraints have p0*(n+m) + p1*(n+m) + p2*(n+m) + p3*n = 10 nonzeros
    # Summing all nonzeros yields 4 + 30 + 10 = 44 nonzeros
    assert len(jac_rows) == 44

    hess_rows, hess_cols = hessian_sparsity_pattern(N, n, m)
    # Stage knots 0..2 each have block size n+m=3 with 6 lower-triangular entries
    # Terminal knot has block size n=2 with 3 lower-triangular entries
    # Total lower-triangular nonzeros count is 18 + 3 = 21
    assert len(hess_rows) == 21
    assert len(hess_cols) == 21
    # Verify lower triangular property: row >= col
    assert np.all(hess_rows >= hess_cols)


def test_runtime_jacobian_value_ordering_matches_build_time_pattern() -> None:
    """Assert directly that runtime Jacobian values match the build-time pattern against dense AD."""
    N = 3
    n = 2
    m = 1

    model = Pendulum()
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(CircleConstraint(n=n, xc=[0.5], yc=[0.2], radius=[0.3], m=m), 0)
    cl.add_constraint(GoalConstraint(n=n, xf=[np.pi, 0.0]), N - 1)

    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.diag(jnp.array([0.1]))
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    x0 = jnp.array([0.1, 0.2])
    t0 = 0.0
    dt = 0.05

    # Random test primal vector Z
    key = jax.random.PRNGKey(42)
    n_Z = N * n + (N - 1) * m
    Z = jax.random.normal(key, (n_Z,))

    # 1. Build-time pattern
    p_seq = tuple(int(pk) for pk in prob.constraints.p)
    jac_rows, jac_cols = jacobian_sparsity_pattern(N, n, m, p_seq)

    # 2. Runtime values
    _c_val, jac_val = constraints_and_jac(prob, Z, x0, t0, dt)

    # 3. Dense Jacobian via autodiff of full constraint vector function
    def dense_constraints_fn(Z_in: jax.Array) -> jax.Array:
        c_out, _ = constraints_and_jac(prob, Z_in, x0, t0, dt)
        return c_out

    J_dense = jax.jacobian(dense_constraints_fn)(Z)

    # 4. Compare nonzeros in pattern order
    J_dense_np = np.asarray(J_dense)
    jac_val_np = np.asarray(jac_val)
    expected_values = J_dense_np[jac_rows, jac_cols]
    np.testing.assert_allclose(jac_val_np, expected_values, atol=1e-12, rtol=1e-12)

    # 5. Assert all structural zeros outside pattern are strictly zero
    mask = np.zeros_like(J_dense_np, dtype=bool)
    mask[jac_rows, jac_cols] = True
    zeros_outside = J_dense_np[~mask]
    np.testing.assert_allclose(zeros_outside, 0.0, atol=1e-12)


def test_runtime_hessian_value_ordering_matches_build_time_pattern() -> None:
    """Assert directly that runtime Hessian values match the build-time pattern against dense AD."""
    N = 3
    n = 2
    m = 1

    model = Pendulum()
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(CircleConstraint(n=n, xc=[0.5], yc=[0.2], radius=[0.3], m=m), 0)
    cl.add_constraint(GoalConstraint(n=n, xf=[np.pi, 0.0]), N - 1)

    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.diag(jnp.array([0.1]))
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    x0 = jnp.array([0.1, 0.2])
    t0 = 0.0
    dt = 0.05

    key = jax.random.PRNGKey(123)
    n_Z = N * n + (N - 1) * m
    Z = jax.random.normal(key, (n_Z,))
    P_total = n + (N - 1) * n + sum(prob.constraints.p)
    lam = jax.random.normal(key, (P_total,))
    obj_factor = 1.0

    # 1. Build-time pattern
    hess_rows, hess_cols = hessian_sparsity_pattern(N, n, m)

    # 2. Runtime values
    hess_val = hessian(prob, Z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam)

    # 3. Dense Hessian via autodiff of full Lagrangian function
    def dense_lagrangian_fn(Z_in: jax.Array) -> jax.Array:
        cost, _ = cost_and_grad(prob, Z_in, t0, dt)
        c_out, _ = constraints_and_jac(prob, Z_in, x0, t0, dt)
        return obj_factor * cost + jnp.dot(lam, c_out)

    H_dense = jax.hessian(dense_lagrangian_fn)(Z)
    H_dense_np = np.asarray(H_dense)
    hess_val_np = np.asarray(hess_val)

    # 4. Compare lower-triangular entries in pattern order
    expected_values = H_dense_np[hess_rows, hess_cols]
    np.testing.assert_allclose(hess_val_np, expected_values, atol=1e-12, rtol=1e-12)

    # 5. Assert off-diagonal blocks are strictly zero
    mask = np.zeros_like(H_dense_np, dtype=bool)
    mask[hess_rows, hess_cols] = True
    mask[hess_cols, hess_rows] = True  # symmetric
    zeros_outside = H_dense_np[~mask]
    np.testing.assert_allclose(zeros_outside, 0.0, atol=1e-12)


def test_cartpole_swingup_ipopt_solve() -> None:
    """End-to-end cartpole swing-up with bounded actuation and terminal goal constraint."""
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    t0 = 0.0

    model = Cartpole()
    n = model.n
    m = model.m

    # Downward initial state
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    # Upright goal state
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])

    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    # Control bound: force between -20N and 20N
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    # Terminal goal constraint: exactly upright
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)

    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())
    state = MPCState.initial(prob, x0=x0, t0=t0, dt=dt)

    # Solve with Ipopt
    res = Ipopt(options={"max_iter": 200, "tol": 1e-4, "print_level": 0}).solve(prob, state)

    assert res.success, f"Ipopt failed to converge: {res.message}"

    traj = res.trajectory
    assert isinstance(traj, Trajectory)
    assert traj.N == N
    assert traj.n == n
    assert traj.m == m

    # Assert initial state satisfied
    np.testing.assert_allclose(traj.X[0], x0, atol=1e-4)

    # Assert terminal goal constraint satisfied
    np.testing.assert_allclose(traj.X[-1], xf, atol=1e-3)

    # Assert control bounds respected
    assert np.all(traj.U >= -20.0 - 1e-5)
    assert np.all(traj.U <= 20.0 + 1e-5)

    # Assert dynamics defects are satisfied
    dmodel = RK4(model)
    for k in range(N - 1):
        x_next_sim = dmodel.discrete_dynamics(traj.X[k], traj.U[k], traj.t[k], traj.dt[k])
        np.testing.assert_allclose(traj.X[k + 1], x_next_sim, atol=1e-4)


def test_compiled_phases_exist_and_match_callbacks() -> None:
    """Verify the four independently compiled phases and individual solver callback helpers."""
    N = 3
    n = 2
    m = 1

    model = Pendulum()
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=[np.pi, 0.0]), N - 1)

    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    x0 = jnp.array([0.0, 0.0])
    t0 = 0.0
    dt = 0.1
    Z = jnp.zeros(N * n + (N - 1) * m)
    P_total = n + (N - 1) * n + sum(prob.constraints.p)
    lam = jnp.zeros(P_total)
    obj_factor = 1.0

    # Phase 1: cost_and_grad vs eval_f / eval_grad_f
    c_val, g_val = cost_and_grad(prob, Z, t0, dt)
    f_val = eval_f(prob, Z, t0, dt)
    grad_f_val = eval_grad_f(prob, Z, t0, dt)
    np.testing.assert_allclose(c_val, f_val)
    np.testing.assert_allclose(g_val, grad_f_val)

    # Phase 2: constraints_and_jac vs eval_g / eval_jac_g
    con_val, j_val = constraints_and_jac(prob, Z, x0, t0, dt)
    g_out = eval_g(prob, Z, x0, t0, dt)
    jac_g_out = eval_jac_g(prob, Z, x0, t0, dt)
    np.testing.assert_allclose(con_val, g_out)
    np.testing.assert_allclose(j_val, jac_g_out)

    # Phase 3: hessian vs eval_h
    h_val = hessian(prob, Z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam)
    eval_h_val = eval_h(prob, Z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam)
    np.testing.assert_allclose(h_val, eval_h_val)


def test_callback_allocates_no_sparse_matrices() -> None:
    """Verify that solver callbacks return flat numpy arrays without sparse matrix allocation."""
    from trajopt.transcription.ipopt import _IpoptCallback

    N = 3
    n = 2
    m = 1

    model = Pendulum()
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=[np.pi, 0.0]), N - 1)

    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    x0 = jnp.array([0.0, 0.0])
    cb = _IpoptCallback(problem=prob, x0=x0, t0=0.0, dt=0.05)

    z = np.zeros(N * n + (N - 1) * m)
    P_total = n + (N - 1) * n + sum(prob.constraints.p)
    lagrange = np.zeros(P_total)

    # Callbacks must return standard flat 1D numpy arrays
    obj_val = cb.objective(z)
    assert isinstance(obj_val, float)

    grad_val = cb.gradient(z)
    assert isinstance(grad_val, np.ndarray)
    assert grad_val.ndim == 1

    con_val = cb.constraints(z)
    assert isinstance(con_val, np.ndarray)
    assert con_val.ndim == 1

    jac_val = cb.jacobian(z)
    assert isinstance(jac_val, np.ndarray)
    assert jac_val.ndim == 1

    hess_val = cb.hessian(z, lagrange)
    assert isinstance(hess_val, np.ndarray)
    assert hess_val.ndim == 1


def test_unconstrained_solve() -> None:
    """Verify that Ipopt solves an optimal control problem with no registered stage constraints."""
    pytest.importorskip("cyipopt")

    N = 10
    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([0.0, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, N=N, integrator=RK4())

    x0 = jnp.array([0.5, 0.0])
    state = MPCState.initial(prob, x0=x0, dt=0.05)
    res = Ipopt(options={"print_level": 0}).solve(prob, state)
    assert res.success
    np.testing.assert_allclose(res.trajectory.X[0], x0, atol=1e-4)


def test_hessian_with_unstacked_stage_cost_at_colliding_horizon() -> None:
    """Assert the Lagrangian Hessian builds when the control dimension equals N - 1."""
    prob, state, _ = quadrotor_obstacle_benchmark(N=5)  # m == 4 == N - 1
    hess_val = hessian(prob, state.Z, dt=state.dt, xf=state.xf)

    hess_rows, _ = hessian_sparsity_pattern(prob.N, prob.model.n, prob.model.m)
    assert hess_val.shape == hess_rows.shape
    assert np.all(np.isfinite(np.asarray(hess_val)))
