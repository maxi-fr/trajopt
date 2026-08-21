"""Cross-verification tests comparing Python Pendulum and DubinsCar models against Julia RobotZoo/RobotDynamics."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.dynamics import (
    RK4,
    Euler,
    ImplicitMidpoint,
)
from trajopt.models import DubinsCar, Pendulum


@pytest.mark.julia
def test_model_parameters_match_robotzoo(jl_to: Any) -> None:
    """Assert that default parameters of Pendulum and DubinsCar match RobotZoo bit-for-bit."""
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics")

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
