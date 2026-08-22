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
from trajopt.expansions import (
    Expansion,
    augmented_lagrangian_expansion,
    cost_expansion,
    dynamics_expansion,
)
from trajopt.models import Cartpole, Pendulum
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

    exp_py = dynamics_expansion(discrete_py, traj_py)

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
def test_cross_dynamics_expansion_pendulum(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays; const RD = RobotDynamics")

    model_jl = jl.seval("RobotZoo.Pendulum()")
    model_py = Pendulum()
    discrete_py = DiscretizedDynamics(model_py, RK4())

    n, m, N = 2, 1, 6
    dt = 0.05

    rng = np.random.default_rng(123)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_py = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
    )

    exp_py = dynamics_expansion(discrete_py, traj_py)

    jl_rk4_jac = jl.seval("""
    function (model, x, u, t, dt)
        integ = RD.RK4(2, 1)
        ForwardDiff.jacobian(z_ -> RD.integrate(integ, model, z_[1:2], z_[3:3], t, dt), [x; u])
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

    exp_py = cost_expansion(obj_py, traj_py)

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

    exp_py = cost_expansion(obj_track_py, traj_pert_py)

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


@pytest.mark.julia
def test_cross_augmented_lagrangian_expansion(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("""
    using TrajectoryOptimization, RobotDynamics, LinearAlgebra, StaticArrays, ForwardDiff
    const TO = TrajectoryOptimization
    const RD = RobotDynamics
    """)

    n, m, N = 3, 2, 5
    dt = 0.1

    x_goal = np.array([1.0, 2.0, 3.0])
    x_min = np.array([-2.0, -2.0, -2.0])
    x_max = np.array([2.0, 2.0, 2.0])
    u_min = np.array([-1.0, -1.0])
    u_max = np.array([1.0, 1.0])

    cons_py = ConstraintList(n=n, m=m, N=N)
    goal_con = GoalConstraint(n=n, xf=jnp.array(x_goal), m=m)
    st_bound = StateBound(n=n, x_min=x_min, x_max=x_max, m=m)
    ctrl_bound = ControlBound(m=m, u_min=u_min, u_max=u_max, n=n)
    cons_py.add_constraint(goal_con, N - 1)
    cons_py.add_constraint(st_bound, range(N - 1))
    cons_py.add_constraint(ctrl_bound, range(N - 1))
    built_cons = cons_py.build()

    rng = np.random.default_rng(99)
    X_np = rng.standard_normal((N, n))
    U_np = rng.standard_normal((N - 1, m))
    t_np = np.linspace(0.0, dt * (N - 1), N)

    traj_py = Trajectory(
        X=jnp.array(X_np),
        U=jnp.array(U_np),
        t=jnp.array(t_np),
        dt=jnp.diff(t_np),
    )

    base_exp = Expansion.zeros(N=N, ne=n, m=m)
    mu = 4.0

    lam_list = []
    for k in range(N):
        p_k = built_cons.p[k]
        lam_list.append(jnp.array(rng.standard_normal(p_k)))

    al_exp_py = augmented_lagrangian_expansion(built_cons, traj_py, base_exp, lam=lam_list, mu=mu)

    jl_al_grad_hess = jl.seval("""
    function (X, U, x_min, x_max, u_min, u_max, x_goal, lam_stages, lam_term, mu, N)
        grads_x = []
        grads_u = []
        hesses_xx = []
        hesses_uu = []
        hesses_ux = []

        for k = 1:(N-1)
            xk = X[k, :]
            uk = U[k, :]
            lam_k = lam_stages[k]

            stage_pen = function (z)
                x = z[1:3]
                u = z[4:5]
                c_x = [x - x_max; x_min - x]
                c_u = [u - u_max; u_min - u]
                c_val = [c_x; c_u]
                shifted = c_val + lam_k / mu
                proj = max.(0.0, shifted)
                dot(lam_k, proj) + 0.5 * mu * dot(proj, proj)
            end

            g = ForwardDiff.gradient(stage_pen, [xk; uk])
            H = ForwardDiff.hessian(stage_pen, [xk; uk])

            push!(grads_x, g[1:3])
            push!(grads_u, g[4:5])
            push!(hesses_xx, H[1:3, 1:3])
            push!(hesses_uu, H[4:5, 4:5])
            push!(hesses_ux, H[4:5, 1:3])
        end

        x_term = X[N, :]
        term_pen = function (x)
            c_val = x - x_goal
            shifted = c_val + lam_term / mu
            proj = shifted
            dot(lam_term, proj) + 0.5 * mu * dot(proj, proj)
        end

        g_term = ForwardDiff.gradient(term_pen, x_term)
        H_term = ForwardDiff.hessian(term_pen, x_term)

        (grads_x, grads_u, hesses_xx, hesses_uu, hesses_ux, g_term, H_term)
    end
    """)

    lam_stages_np = [np.array(lam_list[k]) for k in range(N - 1)]
    lam_term_np = np.array(lam_list[-1])

    gx_jl, gu_jl, Hxx_jl, Huu_jl, Hux_jl, g_term_jl, H_term_jl = jl_al_grad_hess(
        X_np, U_np, x_min, x_max, u_min, u_max, x_goal, lam_stages_np, lam_term_np, mu, N
    )

    for k in range(N - 1):
        np.testing.assert_allclose(np.array(al_exp_py.q[k]), np.array(gx_jl[k]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(al_exp_py.r[k]), np.array(gu_jl[k]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(al_exp_py.Q[k]), np.array(Hxx_jl[k]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(al_exp_py.R[k]), np.array(Huu_jl[k]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(al_exp_py.H[k]), np.array(Hux_jl[k]), rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(np.array(al_exp_py.q[-1]), np.array(g_term_jl), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(al_exp_py.Q[-1]), np.array(H_term_jl), rtol=1e-12, atol=1e-12)
