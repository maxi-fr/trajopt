from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import (
    ConstraintList,
    ControlBound,
    GoalConstraint,
    StateBound,
)
from trajopt.costs import (
    LQRObjective,
    TrackingObjective,
)
from trajopt.dynamics import (
    RK4,
    DiscretizedDynamics,
)
from trajopt.expansions import Expansion
from trajopt.models import Cartpole
from trajopt.trajectory import Trajectory


@pytest.mark.julia
def test_cross_dynamics_expansion_cartpole(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays; const RD = RobotDynamics")

    model_jl = jl.seval("RobotZoo.Cartpole()")
    model_py = Cartpole()
    discrete_py = DiscretizedDynamics(model_py, RK4())

    n, m, N = 4, 1, 8
    dt = 0.05

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_py = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
    )

    exp_py = discrete_py.dynamics_expansion(traj_py)

    jl_rk4_jac = jl.seval("""
    function (model, x, u, t, dt)
        integ = RD.RK4(4, 1)
        ForwardDiff.jacobian(z_ -> RD.integrate(integ, model, z_[1:4], z_[5:5], t, dt), [x; u])
    end
    """)

    for k in range(N - 1):
        xk = X_np[k]
        uk = U_np[k]
        tk = t_np[k]

        J_jl = np.array(jl_rk4_jac(model_jl, xk, uk, tk, dt))
        A_jl = J_jl[:, :n]
        B_jl = J_jl[:, n:]

        np.testing.assert_allclose(np.array(exp_py.A[k]), A_jl, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.B[k]), B_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_cost_expansion_lqr_objective(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("""
    using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays
    const TO = TrajectoryOptimization
    const RD = RobotDynamics
    """)

    n, m, N = 3, 2, 8
    Q_diag = np.array([2.5, 3.2, 1.8])
    R_diag = np.array([0.8, 1.5])
    Qf_diag = np.array([12.0, 15.0, 8.0])
    xf_np = np.array([1.0, -0.5, 2.0])
    uf_np = np.array([0.2, -0.3])
    dt = 0.05

    obj_py = LQRObjective(
        Q=jnp.array(Q_diag),
        R=jnp.array(R_diag),
        Qf=jnp.array(Qf_diag),
        xf=jnp.array(xf_np),
        N=N,
        uf=jnp.array(uf_np),
    )

    jl_create_lqr = jl.seval("""
    function (Qd, Rd, Qfd, xf, uf)
        Q = Diagonal(SVector{3,Float64}(Qd...))
        R = Diagonal(SVector{2,Float64}(Rd...))
        Qf = Diagonal(SVector{3,Float64}(Qfd...))
        xf_v = SVector{3,Float64}(xf...)
        uf_v = SVector{2,Float64}(uf...)
        TO.LQRObjective(Q, R, Qf, xf_v, 8, uf=uf_v)
    end
    """)
    obj_jl = jl_create_lqr(Q_diag, R_diag, Qf_diag, xf_np, uf_np)

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_py = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
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

    exp_py = obj_py.cost_expansion(traj_py)

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

    for k in range(N - 1):
        grad_jl = np.array(jl_knot_grad(obj_jl, traj_jl, k + 1))
        hess_jl = np.array(jl_knot_hess(obj_jl, traj_jl, k + 1))

        np.testing.assert_allclose(np.array(exp_py.q[k]), grad_jl[:n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.r[k]), grad_jl[n:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.Q[k]), hess_jl[:n, :n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.R[k]), hess_jl[n:, n:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.H[k]), hess_jl[n:, :n], rtol=1e-12, atol=1e-12)

    # Terminal knot
    grad_term_jl = np.array(jl_knot_grad(obj_jl, traj_jl, N))
    hess_term_jl = np.array(jl_knot_hess(obj_jl, traj_jl, N))
    np.testing.assert_allclose(np.array(exp_py.q[-1]), grad_term_jl, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(exp_py.Q[-1]), hess_term_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_cost_expansion_tracking_objective(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("""
    using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays
    const TO = TrajectoryOptimization
    const RD = RobotDynamics
    """)

    n, m, N = 2, 1, 6
    Q_diag = np.array([3.0, 2.0])
    R_diag = np.array([0.7])
    Qf_diag = np.array([12.0, 8.0])
    dt = 0.1

    rng = np.random.default_rng(77)
    X_ref = rng.standard_normal((N, n))
    U_ref = rng.standard_normal((N - 1, m))
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_ref_py = Trajectory(
        X=jnp.array(X_ref),
        U=jnp.array(U_ref),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
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

    # Evaluate at a perturbed trajectory
    X_pert = X_ref + 0.1 * rng.standard_normal((N, n))
    U_pert = U_ref + 0.1 * rng.standard_normal((N - 1, m))
    traj_pert_py = Trajectory(
        X=jnp.array(X_pert),
        U=jnp.array(U_pert),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
    )
    traj_pert_jl = jl_create_traj(X_pert, U_pert, N, dt)

    exp_py = obj_track_py.cost_expansion(traj_pert_py)

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

    for k in range(N - 1):
        grad_jl = np.array(jl_track_grad(obj_track_jl, traj_pert_jl, k + 1))
        hess_jl = np.array(jl_track_hess(obj_track_jl, traj_pert_jl, k + 1))

        np.testing.assert_allclose(np.array(exp_py.q[k]), grad_jl[:n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.r[k]), grad_jl[n:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.Q[k]), hess_jl[:n, :n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.R[k]), hess_jl[n:, n:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(exp_py.H[k]), hess_jl[n:, :n], rtol=1e-12, atol=1e-12)

    grad_term_jl = np.array(jl_track_grad(obj_track_jl, traj_pert_jl, N))
    hess_term_jl = np.array(jl_track_hess(obj_track_jl, traj_pert_jl, N))
    np.testing.assert_allclose(np.array(exp_py.q[-1]), grad_term_jl, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(exp_py.Q[-1]), hess_term_jl, rtol=1e-12, atol=1e-12)
