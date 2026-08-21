"""Cross-verification tests comparing Python costs and objectives against TrajectoryOptimization.jl."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs import (
    DiagonalCost,
    LQRCost,
    LQRObjective,
    Objective,
    QuadraticCost,
    TrackingObjective,
    cost,
    update_reference,
)
from trajopt.trajectory import Trajectory


@pytest.mark.julia
def test_cross_diagonal_cost(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    Q_np = np.array([2.5, 3.2, 1.8, 4.0])
    R_np = np.array([0.8, 1.5])
    q_np = np.array([0.1, -0.4, 0.2, -0.3])
    r_np = np.array([-0.5, 0.6])
    c_np = 1.75

    dcost_py = DiagonalCost(Q=jnp.array(Q_np), R=jnp.array(R_np), q=jnp.array(q_np), r=jnp.array(r_np), c=c_np)

    jl_create = jl.seval("""
    function (Qd, Rd, q, r, c)
        TO.DiagonalCost(SVector{4,Float64}(Qd...), SVector{2,Float64}(Rd...), MVector{4,Float64}(q...), MVector{2,Float64}(r...), Float64(c))
    end
    """)
    dcost_jl = jl_create(Q_np, R_np, q_np, r_np, c_np)

    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, -0.5, 2.0, -1.5]),
        np.array([-2.1, 3.4, -0.7, 1.2]),
    ]
    test_controls = [
        np.array([0.0, 0.0]),
        np.array([0.5, -0.8]),
        np.array([-1.2, 2.3]),
    ]

    jl_eval = jl.seval(
        "function (cost, x, u) RD.evaluate(cost, SVector{4,Float64}(x...), SVector{2,Float64}(u...)) end"
    )
    jl_grad = jl.seval("""
    function (cost, x, u)
        z = TO.KnotPoint(SVector{4,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        grad = zeros(6)
        RD.gradient!(cost, grad, z)
        grad
    end
    """)
    jl_hess = jl.seval("""
    function (cost, x, u)
        z = TO.KnotPoint(SVector{4,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        hess = zeros(6, 6)
        RD.hessian!(cost, hess, z)
        hess
    end
    """)
    jl_invert = jl.seval("""
    function (cost)
        Ginv = zeros(6, 6)
        TO.invert!(Ginv, cost)
        Ginv
    end
    """)

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 1. Scalar cost (tol 1e-14)
            val_py = float(dcost_py.evaluate(x_jax, u_jax))
            val_jl = float(jl_eval(dcost_jl, x_np, u_np))
            np.testing.assert_allclose(val_py, val_jl, rtol=1e-14, atol=1e-14)

            # 2. Gradient (tol 1e-12)
            grad_py = np.array(dcost_py.gradient(x_jax, u_jax))
            grad_jl = np.array(jl_grad(dcost_jl, x_np, u_np))
            np.testing.assert_allclose(grad_py, grad_jl, rtol=1e-12, atol=1e-12)

            # 3. Hessian (tol 1e-12)
            hess_py = np.array(dcost_py.hessian(x_jax, u_jax))
            hess_jl = np.array(jl_hess(dcost_jl, x_np, u_np))
            np.testing.assert_allclose(hess_py, hess_jl, rtol=1e-12, atol=1e-12)

    # 4. Inverted Hessian (tol 1e-12)
    inv_py = np.array(dcost_py.hessian_inverse())
    inv_jl = np.array(jl_invert(dcost_jl))
    np.testing.assert_allclose(inv_py, inv_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_quadratic_cost_dense_and_cross_coupling(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 3, 2
    rng = np.random.default_rng(101)
    # Generate positive definite Q and R
    A_Q = rng.standard_normal((n, n))
    Q_np = A_Q.T @ A_Q + 2.0 * np.eye(n)
    A_R = rng.standard_normal((m, m))
    R_np = A_R.T @ A_R + 1.0 * np.eye(m)
    H_np = np.array([[0.2, -0.1, 0.3], [0.4, 0.0, -0.2]])
    q_np = np.array([0.5, -0.2, 0.1])
    r_np = np.array([-0.3, 0.4])
    c_np = 3.5

    qcost_py = QuadraticCost(
        Q=jnp.array(Q_np),
        R=jnp.array(R_np),
        H=jnp.array(H_np),
        q=jnp.array(q_np),
        r=jnp.array(r_np),
        c=c_np,
    )

    jl_create = jl.seval("""
    function (Q, R, H, q, r, c)
        TO.QuadraticCost(SMatrix{3,3,Float64}(Q), SMatrix{2,2,Float64}(R), SMatrix{2,3,Float64}(H), MVector{3,Float64}(q...), MVector{2,Float64}(r...), Float64(c), checks=false)
    end
    """)
    qcost_jl = jl_create(Q_np, R_np, H_np, q_np, r_np, c_np)

    jl_eval = jl.seval(
        "function (cost, x, u) RD.evaluate(cost, SVector{3,Float64}(x...), SVector{2,Float64}(u...)) end"
    )
    jl_grad = jl.seval("""
    function (cost, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        grad = zeros(5)
        RD.gradient!(cost, grad, z)
        grad
    end
    """)
    jl_hess = jl.seval("""
    function (cost, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        hess = zeros(5, 5)
        RD.hessian!(cost, hess, z)
        if !TO.is_blockdiag(cost)
            hess[1:3, 4:5] .= cost.H'
        end
        hess
    end
    """)
    jl_invert = jl.seval("""
    function (cost)
        Ginv = zeros(5, 5)
        TO.invert!(Ginv, cost)
        Ginv
    end
    """)

    test_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.2, -0.8, 0.4]),
        np.array([-0.5, 2.1, -1.3]),
    ]
    test_controls = [
        np.array([0.0, 0.0]),
        np.array([0.7, -1.1]),
        np.array([-0.3, 0.9]),
    ]

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 1. Scalar cost (tol 1e-14)
            val_py = float(qcost_py.evaluate(x_jax, u_jax))
            val_jl = float(jl_eval(qcost_jl, x_np, u_np))
            np.testing.assert_allclose(val_py, val_jl, rtol=1e-14, atol=1e-14)

            # 2. Gradient (tol 1e-12)
            grad_py = np.array(qcost_py.gradient(x_jax, u_jax))
            grad_jl = np.array(jl_grad(qcost_jl, x_np, u_np))
            np.testing.assert_allclose(grad_py, grad_jl, rtol=1e-12, atol=1e-12)

            # 3. Hessian (tol 1e-12)
            hess_py = np.array(qcost_py.hessian(x_jax, u_jax))
            hess_jl = np.array(jl_hess(qcost_jl, x_np, u_np))
            np.testing.assert_allclose(hess_py, hess_jl, rtol=1e-12, atol=1e-12)

    # 4. Inverted Hessian (tol 1e-12)
    inv_py = np.array(qcost_py.hessian_inverse())
    inv_jl = np.array(jl_invert(qcost_jl))
    np.testing.assert_allclose(inv_py, inv_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_lqr_objective_and_cost(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m, N = 3, 2, 10
    Q_diag = np.array([2.0, 3.0, 1.5])
    R_diag = np.array([0.5, 1.2])
    Qf_diag = np.array([10.0, 15.0, 8.0])
    xf_np = np.array([1.0, -0.5, 2.0])
    uf_np = np.array([0.2, -0.3])

    obj_py = LQRObjective(
        Q=jnp.array(Q_diag),
        R=jnp.array(R_diag),
        Qf=jnp.array(Qf_diag),
        xf=jnp.array(xf_np),
        N=N,
        uf=jnp.array(uf_np),
    )

    jl_create_lqr = jl.seval("""
    function (Qd, Rd, Qfd, xf, N, uf)
        Q = Diagonal(SVector{3,Float64}(Qd...))
        R = Diagonal(SVector{2,Float64}(Rd...))
        Qf = Diagonal(SVector{3,Float64}(Qfd...))
        TO.LQRObjective(Q, R, Qf, SVector{3,Float64}(xf...), N, uf=SVector{2,Float64}(uf...))
    end
    """)
    obj_jl = jl_create_lqr(Q_diag, R_diag, Qf_diag, xf_np, N, uf_np)

    # Generate test trajectory
    rng = np.random.default_rng(55)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    dt = 0.05
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_py = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.array(t_np),
        dt=jnp.array(np.diff(t_np)),
    )

    jl_create_traj = jl.seval("""
    function (X, U, N, dt)
        kps = [
            TO.KnotPoint(
                SVector{3,Float64}(X[k, :]...),
                k < N ? SVector{2,Float64}(U[k, :]...) : SVector{2,Float64}(0.0, 0.0),
                (k - 1) * dt,
                k < N ? dt : 0.0
            ) for k = 1:N
        ]
        TO.SampledTrajectory(kps)
    end
    """)
    traj_jl = jl_create_traj(X_np, U_np, N, dt)

    # Total cost comparison (tol 1e-14)
    cost_py = float(cost(obj_py, traj_py))
    cost_jl = float(jl.TO.cost(obj_jl, traj_jl))
    np.testing.assert_allclose(cost_py, cost_jl, rtol=1e-14, atol=1e-14)

    # Per-knot gradient, Hessian, and inversion check
    jl_knot_grad = jl.seval("""
    function (obj, traj, k)
        grad = zeros(k < length(obj) ? 5 : 3)
        RD.gradient!(obj[k], grad, traj[k])
        grad
    end
    """)
    jl_knot_hess = jl.seval("""
    function (obj, traj, k)
        sz = k < length(obj) ? 5 : 3
        hess = zeros(sz, sz)
        RD.hessian!(obj[k], hess, traj[k])
        hess
    end
    """)
    jl_knot_invert = jl.seval("""
    function (obj, k)
        sz = k < length(obj) ? 5 : 3
        Ginv = zeros(sz, sz)
        TO.invert!(Ginv, obj[k])
        Ginv
    end
    """)

    for k in range(N):
        cost_k_py = obj_py[k]
        xk = jnp.array(X_np[k])
        uk = jnp.array(U_np[k]) if k < N - 1 else None

        grad_k_py = np.array(cost_k_py.gradient(xk, uk))
        grad_k_jl = np.array(jl_knot_grad(obj_jl, traj_jl, k + 1))
        np.testing.assert_allclose(grad_k_py, grad_k_jl, rtol=1e-12, atol=1e-12)

        hess_k_py = np.array(cost_k_py.hessian(xk, uk))
        hess_k_jl = np.array(jl_knot_hess(obj_jl, traj_jl, k + 1))
        np.testing.assert_allclose(hess_k_py, hess_k_jl, rtol=1e-12, atol=1e-12)

        inv_k_py = np.array(cost_k_py.hessian_inverse())
        inv_k_jl = np.array(jl_knot_invert(obj_jl, k + 1))
        np.testing.assert_allclose(inv_k_py, inv_k_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_tracking_objective(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m, N = 2, 1, 8
    Q_diag = np.array([3.0, 2.0])
    R_diag = np.array([0.7])
    Qf_diag = np.array([12.0, 8.0])

    rng = np.random.default_rng(77)
    X_ref = rng.standard_normal((N, n))
    U_ref = rng.standard_normal((N - 1, m))
    dt = 0.1
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_ref_py = Trajectory(
        X=jnp.array(X_ref),
        U=jnp.array(U_ref),
        t=jnp.array(t_np),
        dt=jnp.array(np.diff(t_np)),
    )

    jl_create_traj = jl.seval("""
    function (X, U, N, dt)
        kps = [
            TO.KnotPoint(
                SVector{2,Float64}(X[k, :]...),
                k < N ? SVector{1,Float64}(U[k, :]...) : SVector{1,Float64}(0.0),
                (k - 1) * dt,
                k < N ? dt : 0.0
            ) for k = 1:N
        ]
        TO.SampledTrajectory(kps)
    end
    """)
    traj_ref_jl = jl_create_traj(X_ref, U_ref, N, dt)

    obj_track_py = TrackingObjective(
        Q=jnp.array(Q_diag),
        R=jnp.array(R_diag),
        trajectory=traj_ref_py,
        Qf=jnp.array(Qf_diag),
    )

    jl_create_tracking = jl.seval("""
    function (Qd, Rd, Qfd, traj)
        Q = Diagonal(SVector{2,Float64}(Qd...))
        R = Diagonal(SVector{1,Float64}(Rd...))
        Qf = Diagonal(SVector{2,Float64}(Qfd...))
        TO.TrackingObjective(Q, R, traj, Qf=Qf)
    end
    """)
    obj_track_jl = jl_create_tracking(Q_diag, R_diag, Qf_diag, traj_ref_jl)

    # 1. Cost of target trajectory itself should be zero
    cost_self_py = float(cost(obj_track_py, traj_ref_py))
    cost_self_jl = float(jl.TO.cost(obj_track_jl, traj_ref_jl))
    np.testing.assert_allclose(cost_self_py, 0.0, atol=1e-14)
    np.testing.assert_allclose(cost_self_jl, 0.0, atol=1e-14)

    # 2. Cost of a perturbed trajectory
    X_pert = X_ref + 0.1 * rng.standard_normal((N, n))
    U_pert = U_ref + 0.1 * rng.standard_normal((N - 1, m))
    traj_pert_py = Trajectory(
        X=jnp.array(X_pert),
        U=jnp.array(U_pert),
        t=jnp.array(t_np),
        dt=jnp.array(np.diff(t_np)),
    )
    traj_pert_jl = jl_create_traj(X_pert, U_pert, N, dt)

    cost_pert_py = float(cost(obj_track_py, traj_pert_py))
    cost_pert_jl = float(jl.TO.cost(obj_track_jl, traj_pert_jl))
    np.testing.assert_allclose(cost_pert_py, cost_pert_jl, rtol=1e-14, atol=1e-14)

    # 3. Per-knot gradient, Hessian, and inversion on tracking objective
    jl_track_grad = jl.seval("""
    function (obj, traj, k)
        grad = zeros(k < length(obj) ? 3 : 2)
        RD.gradient!(obj[k], grad, traj[k])
        grad
    end
    """)
    jl_track_hess = jl.seval("""
    function (obj, traj, k)
        sz = k < length(obj) ? 3 : 2
        hess = zeros(sz, sz)
        RD.hessian!(obj[k], hess, traj[k])
        hess
    end
    """)
    jl_track_invert = jl.seval("""
    function (obj, k)
        sz = k < length(obj) ? 3 : 2
        Ginv = zeros(sz, sz)
        TO.invert!(Ginv, obj[k])
        Ginv
    end
    """)

    for k in range(N):
        cost_k_py = obj_track_py[k]
        xk = jnp.array(X_pert[k])
        uk = jnp.array(U_pert[k]) if k < N - 1 else None

        grad_k_py = np.array(cost_k_py.gradient(xk, uk))
        grad_k_jl = np.array(jl_track_grad(obj_track_jl, traj_pert_jl, k + 1))
        np.testing.assert_allclose(grad_k_py, grad_k_jl, rtol=1e-12, atol=1e-12)

        hess_k_py = np.array(cost_k_py.hessian(xk, uk))
        hess_k_jl = np.array(jl_track_hess(obj_track_jl, traj_pert_jl, k + 1))
        np.testing.assert_allclose(hess_k_py, hess_k_jl, rtol=1e-12, atol=1e-12)

        inv_k_py = np.array(cost_k_py.hessian_inverse())
        if k < N - 1:
            inv_k_jl = np.array(jl_track_invert(obj_track_jl, k + 1))
            np.testing.assert_allclose(inv_k_py, inv_k_jl, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_allclose(inv_k_py, np.diag(1.0 / Qf_diag), rtol=1e-12, atol=1e-12)
