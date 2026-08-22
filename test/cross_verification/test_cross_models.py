from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.dynamics import (
    RK4,
    DiscretizedDynamics,
    Euler,
    ImplicitMidpoint,
)
from trajopt.expansions import dynamics_expansion
from trajopt.models import DubinsCar, Pendulum, Quadrotor
from trajopt.rotations.quaternion import Quaternion
from trajopt.trajectory import Trajectory


@pytest.mark.julia
def test_model_parameters_match_robotzoo(jl_to: Any) -> None:
    """Assert that default parameters of Pendulum, DubinsCar, and Quadrotor match RobotZoo bit-for-bit."""
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, LinearAlgebra")

    # 1. Pendulum parameter match assertion
    jl_pendulum = jl.seval("RobotZoo.Pendulum()")
    py_pendulum = Pendulum()

    assert float(py_pendulum.mass) == float(jl_pendulum.mass)
    assert float(py_pendulum.len) == float(jl_pendulum.len)
    assert float(py_pendulum.b) == float(jl_pendulum.b)
    assert float(py_pendulum.lc) == float(jl_pendulum.lc)
    assert float(py_pendulum.I) == float(jl_pendulum.I)
    assert float(py_pendulum.g) == float(jl_pendulum.g)

    assert py_pendulum.n == int(jl.seval("RobotDynamics.state_dim(RobotZoo.Pendulum())"))
    assert py_pendulum.m == int(jl.seval("RobotDynamics.control_dim(RobotZoo.Pendulum())"))

    # 2. DubinsCar parameter match assertion
    jl_car = jl.seval("RobotZoo.DubinsCar()")
    py_car = DubinsCar()

    assert float(py_car.radius) == float(jl_car.radius)
    assert py_car.n == int(jl.seval("RobotDynamics.state_dim(RobotZoo.DubinsCar())"))
    assert py_car.m == int(jl.seval("RobotDynamics.control_dim(RobotZoo.DubinsCar())"))

    # 3. Quadrotor parameter match assertion
    jl_quad = jl.seval("RobotZoo.Quadrotor()")
    py_quad = Quadrotor()

    assert float(py_quad.mass) == float(jl_quad.mass)
    jl_J = jl.seval("diag(Array(RobotZoo.Quadrotor().J))")
    np.testing.assert_allclose(py_quad.J, np.array(jl_J))
    np.testing.assert_allclose(py_quad.gravity, np.array(jl_quad.gravity))
    assert float(py_quad.motor_dist) == float(jl_quad.motor_dist)
    assert float(py_quad.kf) == float(jl_quad.kf)
    assert float(py_quad.km) == float(jl_quad.km)

    assert py_quad.n == int(jl.seval("RobotDynamics.state_dim(RobotZoo.Quadrotor())"))
    assert py_quad.m == int(jl.seval("RobotDynamics.control_dim(RobotZoo.Quadrotor())"))


@pytest.mark.julia
def test_pendulum_continuous_dynamics_and_jacobians_cross(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays")

    jl_model = jl.seval("RobotZoo.Pendulum()")
    py_model = Pendulum()

    jl_eval_dyn = jl.seval("""
    function (model, x, u)
        RobotDynamics.dynamics(model, SVector{2,Float64}(x...), SVector{1,Float64}(u...))
    end
    """)

    jl_eval_jac = jl.seval("""
    function (model, x, u)
        z = [x; u]
        ForwardDiff.jacobian(z_ -> RobotDynamics.dynamics(model, z_[1:2], z_[3:3]), z)
    end
    """)

    test_states = [
        np.array([0.0, 0.0]),
        np.array([0.5, -1.2]),
        np.array([np.pi / 2, 2.0]),
        np.array([-np.pi / 3, -0.8]),
        np.array([np.pi, 0.0]),
        np.array([2.5, 3.5]),
    ]

    test_controls = [
        np.array([0.0]),
        np.array([1.5]),
        np.array([-2.4]),
        np.array([0.1]),
        np.array([10.0]),
    ]

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # Continuous dynamics comparison (tol 1e-14)
            xdot_py = py_model.dynamics(x_jax, u_jax)
            xdot_jl = np.array(jl_eval_dyn(jl_model, x_np, u_np))
            np.testing.assert_allclose(xdot_py, xdot_jl, rtol=1e-14, atol=1e-14)

            # Continuous Jacobian comparison (tol 1e-12)
            J_py = py_model.jacobian(x_jax, u_jax)
            J_jl = np.array(jl_eval_jac(jl_model, x_np, u_np))
            np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_dubins_car_continuous_dynamics_and_jacobians_cross(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays")

    jl_model = jl.seval("RobotZoo.DubinsCar()")
    py_model = DubinsCar()

    jl_eval_dyn = jl.seval("""
    function (model, x, u)
        RobotDynamics.dynamics(model, SVector{3,Float64}(x...), SVector{2,Float64}(u...))
    end
    """)

    jl_eval_jac = jl.seval("""
    function (model, x, u)
        z = [x; u]
        ForwardDiff.jacobian(z_ -> RobotDynamics.dynamics(model, z_[1:3], z_[4:5]), z)
    end
    """)

    test_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.5]),
        np.array([-2.5, 3.2, np.pi / 3]),
        np.array([0.5, -1.0, -np.pi / 4]),
        np.array([10.0, -5.0, np.pi]),
    ]

    test_controls = [
        np.array([0.0, 0.0]),
        np.array([1.5, -0.2]),
        np.array([-2.0, 1.0]),
        np.array([3.0, 0.0]),
        np.array([0.5, 2.5]),
    ]

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # Continuous dynamics comparison (tol 1e-14)
            xdot_py = py_model.dynamics(x_jax, u_jax)
            xdot_jl = np.array(jl_eval_dyn(jl_model, x_np, u_np))
            np.testing.assert_allclose(xdot_py, xdot_jl, rtol=1e-14, atol=1e-14)

            # Continuous Jacobian comparison (tol 1e-12)
            J_py = py_model.jacobian(x_jax, u_jax)
            J_jl = np.array(jl_eval_jac(jl_model, x_np, u_np))
            np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_pendulum_integrators_cross_verification(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays, LinearAlgebra")

    jl_model = jl.seval("RobotZoo.Pendulum()")
    py_model = Pendulum()

    py_euler = Euler(py_model)
    py_rk4 = RK4(py_model)
    py_mid = ImplicitMidpoint(py_model, iters=10)

    jl.seval("""
    jl_pend_rk4 = RobotDynamics.RK4(2, 1)
    jl_pend_euler = RobotDynamics.Euler(2, 1)

    function eval_pend_euler_step(model, x, u, t, dt)
        RobotDynamics.integrate(jl_pend_euler, model, SVector{2,Float64}(x...), SVector{1,Float64}(u...), t, dt)
    end

    function eval_pend_euler_jac(model, x, u, t, dt)
        ForwardDiff.jacobian(z_ -> RobotDynamics.integrate(jl_pend_euler, model, z_[1:2], z_[3:3], t, dt), [x; u])
    end

    function eval_pend_rk4_step(model, x, u, t, dt)
        RobotDynamics.integrate(jl_pend_rk4, model, SVector{2,Float64}(x...), SVector{1,Float64}(u...), t, dt)
    end

    function eval_pend_rk4_jac(model, x, u, t, dt)
        ForwardDiff.jacobian(z_ -> RobotDynamics.integrate(jl_pend_rk4, model, z_[1:2], z_[3:3], t, dt), [x; u])
    end

    function eval_pend_mid_step(model, x, u, t, dt; iters=10)
        xn = copy(x)
        for iter in 1:iters
            xmid = (x + xn) / 2
            fmid = RobotDynamics.dynamics(model, xmid, u, t + dt/2)
            r = x + dt * fmid - xn
            dfdx = ForwardDiff.jacobian(x_ -> RobotDynamics.dynamics(model, x_, u, t + dt/2), xmid)
            A = I - (dt/2) * dfdx
            dx = A \\ r
            xn += dx
        end
        return xn
    end

    function eval_pend_mid_jac(model, x, u, t, dt; iters=10)
        ForwardDiff.jacobian(z_ -> eval_pend_mid_step(model, z_[1:2], z_[3:3], t, dt; iters=iters), [x; u])
    end
    """)

    jl_euler_step = jl.seval("eval_pend_euler_step")
    jl_euler_jac = jl.seval("eval_pend_euler_jac")
    jl_rk4_step = jl.seval("eval_pend_rk4_step")
    jl_rk4_jac = jl.seval("eval_pend_rk4_jac")
    jl_mid_step = jl.seval("eval_pend_mid_step")
    jl_mid_jac = jl.seval("eval_pend_mid_jac")

    test_states = [
        np.array([0.0, 0.0]),
        np.array([0.5, -1.2]),
        np.array([np.pi / 2, 2.0]),
        np.array([-np.pi / 3, -0.8]),
        np.array([np.pi, 0.0]),
    ]

    test_controls = [
        np.array([0.0]),
        np.array([1.5]),
        np.array([-2.4]),
        np.array([0.1]),
        np.array([10.0]),
    ]

    dt = 0.05
    t = 0.0

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 1. Euler step (1e-14) and Jacobian (1e-12)
            xnext_euler_py = py_euler.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_euler_jl = np.array(jl_euler_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_euler_py, xnext_euler_jl, rtol=1e-14, atol=1e-14)

            J_euler_py = py_euler.jacobian(x_jax, u_jax, t, dt)
            J_euler_jl = np.array(jl_euler_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_euler_py, J_euler_jl, rtol=1e-12, atol=1e-12)

            # 2. RK4 step (1e-14) and Jacobian (1e-12)
            xnext_rk4_py = py_rk4.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_rk4_jl = np.array(jl_rk4_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_rk4_py, xnext_rk4_jl, rtol=1e-14, atol=1e-14)

            J_rk4_py = py_rk4.jacobian(x_jax, u_jax, t, dt)
            J_rk4_jl = np.array(jl_rk4_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_rk4_py, J_rk4_jl, rtol=1e-12, atol=1e-12)

            # 3. Implicit Midpoint step (1e-14) and Jacobian (1e-12)
            xnext_mid_py = py_mid.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_mid_jl = np.array(jl_mid_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_mid_py, xnext_mid_jl, rtol=1e-14, atol=1e-14)

            J_mid_py = py_mid.jacobian(x_jax, u_jax, t, dt)
            J_mid_jl = np.array(jl_mid_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_mid_py, J_mid_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_dubins_car_integrators_cross_verification(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays, LinearAlgebra")

    jl_model = jl.seval("RobotZoo.DubinsCar()")
    py_model = DubinsCar()

    py_euler = Euler(py_model)
    py_rk4 = RK4(py_model)
    py_mid = ImplicitMidpoint(py_model, iters=10)

    jl.seval("""
    jl_car_rk4 = RobotDynamics.RK4(3, 2)
    jl_car_euler = RobotDynamics.Euler(3, 2)

    function eval_car_euler_step(model, x, u, t, dt)
        RobotDynamics.integrate(jl_car_euler, model, SVector{3,Float64}(x...), SVector{2,Float64}(u...), t, dt)
    end

    function eval_car_euler_jac(model, x, u, t, dt)
        ForwardDiff.jacobian(z_ -> RobotDynamics.integrate(jl_car_euler, model, z_[1:3], z_[4:5], t, dt), [x; u])
    end

    function eval_car_rk4_step(model, x, u, t, dt)
        RobotDynamics.integrate(jl_car_rk4, model, SVector{3,Float64}(x...), SVector{2,Float64}(u...), t, dt)
    end

    function eval_car_rk4_jac(model, x, u, t, dt)
        ForwardDiff.jacobian(z_ -> RobotDynamics.integrate(jl_car_rk4, model, z_[1:3], z_[4:5], t, dt), [x; u])
    end

    function eval_car_mid_step(model, x, u, t, dt; iters=10)
        xn = copy(x)
        for iter in 1:iters
            xmid = (x + xn) / 2
            fmid = RobotDynamics.dynamics(model, xmid, u, t + dt/2)
            r = x + dt * fmid - xn
            dfdx = ForwardDiff.jacobian(x_ -> RobotDynamics.dynamics(model, x_, u, t + dt/2), xmid)
            A = I - (dt/2) * dfdx
            dx = A \\ r
            xn += dx
        end
        return xn
    end

    function eval_car_mid_jac(model, x, u, t, dt; iters=10)
        ForwardDiff.jacobian(z_ -> eval_car_mid_step(model, z_[1:3], z_[4:5], t, dt; iters=iters), [x; u])
    end
    """)

    jl_euler_step = jl.seval("eval_car_euler_step")
    jl_euler_jac = jl.seval("eval_car_euler_jac")
    jl_rk4_step = jl.seval("eval_car_rk4_step")
    jl_rk4_jac = jl.seval("eval_car_rk4_jac")
    jl_mid_step = jl.seval("eval_car_mid_step")
    jl_mid_jac = jl.seval("eval_car_mid_jac")

    test_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.5]),
        np.array([-2.5, 3.2, np.pi / 3]),
        np.array([0.5, -1.0, -np.pi / 4]),
        np.array([10.0, -5.0, np.pi]),
    ]

    test_controls = [
        np.array([0.0, 0.0]),
        np.array([1.5, -0.2]),
        np.array([-2.0, 1.0]),
        np.array([3.0, 0.0]),
        np.array([0.5, 2.5]),
    ]

    dt = 0.05
    t = 0.0

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 1. Euler step (1e-14) and Jacobian (1e-12)
            xnext_euler_py = py_euler.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_euler_jl = np.array(jl_euler_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_euler_py, xnext_euler_jl, rtol=1e-14, atol=1e-14)

            J_euler_py = py_euler.jacobian(x_jax, u_jax, t, dt)
            J_euler_jl = np.array(jl_euler_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_euler_py, J_euler_jl, rtol=1e-12, atol=1e-12)

            # 2. RK4 step (1e-14) and Jacobian (1e-12)
            xnext_rk4_py = py_rk4.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_rk4_jl = np.array(jl_rk4_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_rk4_py, xnext_rk4_jl, rtol=1e-14, atol=1e-14)

            J_rk4_py = py_rk4.jacobian(x_jax, u_jax, t, dt)
            J_rk4_jl = np.array(jl_rk4_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_rk4_py, J_rk4_jl, rtol=1e-12, atol=1e-12)

            # 3. Implicit Midpoint step (1e-14) and Jacobian (1e-12)
            xnext_mid_py = py_mid.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_mid_jl = np.array(jl_mid_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_mid_py, xnext_mid_jl, rtol=1e-14, atol=1e-14)

            J_mid_py = py_mid.jacobian(x_jax, u_jax, t, dt)
            J_mid_jl = np.array(jl_mid_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_mid_py, J_mid_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_custom_parameters_cross(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays")

    # 1. Custom Pendulum
    mass, length, b, lc, I_val, g = 1.8, 0.9, 0.15, 0.45, 0.35, 9.80665
    jl_pendulum = jl.RobotZoo.Pendulum(mass=mass, len=length, b=b, lc=lc, I=I_val, g=g)
    py_pendulum = Pendulum(mass=mass, len=length, b=b, lc=lc, I=I_val, g=g)

    jl_eval_dyn_p = jl.seval("""
    function (model, x, u)
        RobotDynamics.dynamics(model, SVector{2,Float64}(x...), SVector{1,Float64}(u...))
    end
    """)
    jl_eval_jac_p = jl.seval("""
    function (model, x, u)
        z = [x; u]
        ForwardDiff.jacobian(z_ -> RobotDynamics.dynamics(model, z_[1:2], z_[3:3]), z)
    end
    """)

    x_p = np.array([0.7, -1.5])
    u_p = np.array([2.5])
    np.testing.assert_allclose(
        py_pendulum.dynamics(jnp.array(x_p), jnp.array(u_p)),
        np.array(jl_eval_dyn_p(jl_pendulum, x_p, u_p)),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        py_pendulum.jacobian(jnp.array(x_p), jnp.array(u_p)),
        np.array(jl_eval_jac_p(jl_pendulum, x_p, u_p)),
        rtol=1e-12,
        atol=1e-12,
    )

    # 2. Custom DubinsCar
    radius = 0.35
    jl_car = jl.RobotZoo.DubinsCar(radius=radius)
    py_car = DubinsCar(radius=radius)

    assert float(py_car.radius) == float(jl_car.radius)


@pytest.mark.julia
def test_quadrotor_continuous_dynamics_and_jacobians_cross(jl_to: Any) -> None:
    r"""Assert Quadrotor continuous dynamics and Jacobians match RobotZoo under quaternion conversion.

    Conversion:
    x_jl = T_13 @ x_py
    xdot_jl = T_13 @ xdot_py
    df/dx_jl = T_13 @ (df/dx_py) @ T_13^T
    df/du_jl = T_13 @ (df/du_py)
    """
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays, LinearAlgebra")

    jl_model = jl.seval("RobotZoo.Quadrotor()")
    py_model = Quadrotor()

    T_quat = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    T_13 = np.block(
        [
            [np.eye(3), np.zeros((3, 4)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((4, 3)), T_quat, np.zeros((4, 3)), np.zeros((4, 3))],
            [np.zeros((3, 3)), np.zeros((3, 4)), np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 4)), np.zeros((3, 3)), np.eye(3)],
        ]
    )

    jl_eval_dyn = jl.seval("""
    function (model, x, u)
        RobotDynamics.dynamics(model, SVector{13,Float64}(x...), SVector{4,Float64}(u...))
    end
    """)

    jl_eval_jac = jl.seval("""
    function (model, x, u)
        z = [x; u]
        ForwardDiff.jacobian(z_ -> RobotDynamics.dynamics(model, z_[1:13], z_[14:17]), z)
    end
    """)

    rng = np.random.default_rng(300)
    for _ in range(20):
        r = rng.standard_normal(3)
        q_raw = rng.standard_normal(4)
        q = q_raw / np.linalg.norm(q_raw)
        v = rng.standard_normal(3)
        omega = rng.standard_normal(3)
        u = rng.uniform(0.5, 3.0, size=4)

        x_py = np.concatenate([r, q, v, omega])
        x_jl = T_13 @ x_py

        # 1. Continuous dynamics comparison (tol 1e-14)
        xdot_py = np.array(py_model.dynamics(jnp.array(x_py), jnp.array(u)))
        xdot_jl = np.array(jl_eval_dyn(jl_model, x_jl, u))

        # 1a. Position derivative r_dot = v
        np.testing.assert_allclose(xdot_py[:3], xdot_jl[:3], rtol=1e-14, atol=1e-14)

        # 1b. Linear velocity derivative v_dot
        # In Python: vdot = g + (1/m) * R(q)^T @ F_body
        # In Julia: vdot = g + (1/m) * q_jl * F_body
        np.testing.assert_allclose(xdot_py[7:10], xdot_jl[7:10], rtol=1e-14, atol=1e-14)

        # 1c. Angular velocity derivative omega_dot
        np.testing.assert_allclose(xdot_py[10:13], xdot_jl[10:13], rtol=1e-14, atol=1e-14)

        # 1d. Quaternion kinematics qdot:
        # In Python (JPL): qdot_py = 0.5 * Xi(q) @ omega_py
        # In Julia (Hamilton scalar-first): qdot_jl = Rotations.kinematics(q_jl, omega_jl)
        # Relation: T_quat @ qdot_py = -qdot_jl or qdot_jl?
        # Note: h = T @ q => hdot = T @ qdot_py.
        # But Rotations.kinematics(h, omega) in Julia uses right multiplication 0.5 * h * [0; omega]
        # or left multiplication 0.5 * [0; omega] * h?
        # Let's check relation: T_quat @ xdot_py[3:7] vs xdot_jl[3:7]
        np.testing.assert_allclose(T_quat @ xdot_py[3:7], xdot_jl[3:7], rtol=1e-14, atol=1e-14)

        # 2. Continuous Jacobians comparison (tol 1e-12)
        J_py = np.array(py_model.jacobian(jnp.array(x_py), jnp.array(u)))
        J_jl = np.array(jl_eval_jac(jl_model, x_jl, u))

        A_py = J_py[:, :13]
        B_py = J_py[:, 13:]
        A_jl = J_jl[:, :13]
        B_jl = J_jl[:, 13:]

        # Continuous state and control Jacobians match under the 13-state basis transformation:
        np.testing.assert_allclose(T_13 @ A_py @ T_13.T, A_jl, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(T_13 @ B_py, B_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_quadrotor_sandwiched_dynamics_expansion_cross(jl_to: Any) -> None:
    r"""Assert RK4 Quadrotor error-state dynamics expansion matches Julia TrajectoryOptimization at 1e-12.

    Relation between error-state expansions:
    A_bar_jl = E(q_next) @ A_bar_py @ E(q_k)^T
    B_bar_jl = E(q_next) @ B_bar_py
    where E(q) = blockdiag(I3, -R(q)^T, I3, I3)
    """
    jl = jl_to
    jl.seval("""
    using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays, LinearAlgebra, Rotations
    const RD = RobotDynamics
    """)

    jl_model = jl.seval("RobotZoo.Quadrotor()")
    py_model = Quadrotor()
    discrete_py = DiscretizedDynamics(py_model, RK4())

    T_quat = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    T_13 = np.block(
        [
            [np.eye(3), np.zeros((3, 4)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((4, 3)), T_quat, np.zeros((4, 3)), np.zeros((4, 3))],
            [np.zeros((3, 3)), np.zeros((3, 4)), np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 4)), np.zeros((3, 3)), np.eye(3)],
        ]
    )

    jl_step_jac = jl.seval("""
    function (model, x_jl, u, t, dt)
        integ = RD.RK4(13, 4)
        ForwardDiff.jacobian(z_ -> RD.integrate(integ, model, z_[1:13], z_[14:17], t, dt), [x_jl; u])
    end
    """)

    jl_diff = getattr(jl.Rotations, "∇differential")

    dt = 0.05
    t = 0.0
    rng = np.random.default_rng(301)

    for _ in range(10):
        # Generate valid initial state and control
        r0 = rng.standard_normal(3)
        q0_raw = rng.standard_normal(4)
        q0 = q0_raw / np.linalg.norm(q0_raw)
        v0 = rng.standard_normal(3)
        omega0 = rng.standard_normal(3)
        x0_py = np.concatenate([r0, q0, v0, omega0])
        u_py = rng.uniform(0.8, 2.5, size=4)

        # Step forward in Python to get x1_py and normalize quaternion
        x1_py = np.array(discrete_py.discrete_dynamics(jnp.array(x0_py), jnp.array(u_py), t, dt))
        q1_norm = np.linalg.norm(x1_py[3:7])
        x1_py[3:7] /= q1_norm

        # Python Jacobians and error-state sandwich
        Ak_py = np.array(discrete_py.state_jacobian(jnp.array(x0_py), jnp.array(u_py), t, dt))
        Bk_py = np.array(discrete_py.control_jacobian(jnp.array(x0_py), jnp.array(u_py), t, dt))
        G0_py = np.array(py_model.errstate_jacobian(jnp.array(x0_py)))
        G1_py = np.array(py_model.errstate_jacobian(jnp.array(x1_py)))

        A_bar_py = G1_py.T @ Ak_py @ G0_py  # (12, 12)
        B_bar_py = G1_py.T @ Bk_py  # (12, 4)

        # Julia step and error-state expansion
        x0_jl = T_13 @ x0_py
        x1_jl = T_13 @ x1_py

        J_jl = np.array(jl_step_jac(jl_model, x0_jl, u_py, t, dt))
        Ak_jl = J_jl[:, :13]
        Bk_jl = J_jl[:, 13:]

        # Julia error-state Jacobians G_jl(x0_jl), G_jl(x1_jl)
        q0_quat_jl = jl.Rotations.UnitQuaternion(x0_jl[3], x0_jl[4], x0_jl[5], x0_jl[6])
        q1_quat_jl = jl.Rotations.UnitQuaternion(x1_jl[3], x1_jl[4], x1_jl[5], x1_jl[6])

        G0_rot_jl = np.array(jl_diff(q0_quat_jl))
        G1_rot_jl = np.array(jl_diff(q1_quat_jl))

        G0_jl = np.block(
            [
                [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((4, 3)), 0.5 * G0_rot_jl, np.zeros((4, 3)), np.zeros((4, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
            ]
        )
        G1_jl = np.block(
            [
                [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((4, 3)), 0.5 * G1_rot_jl, np.zeros((4, 3)), np.zeros((4, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
            ]
        )

        A_bar_jl = G1_jl.T @ Ak_jl @ G0_jl
        B_bar_jl = G1_jl.T @ Bk_jl

        # In this representation where x_jl = T_13 @ x_py and G_jl = T_13 @ G_py,
        # the error-state vectors are in identical coordinates:
        np.testing.assert_allclose(A_bar_py, A_bar_jl, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(B_bar_py, B_bar_jl, rtol=1e-12, atol=1e-12)
