"""Cross-verification tests comparing Python integrators (Euler, RK4, ImplicitMidpoint) against Julia RobotDynamics."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.dynamics import (
    RK4,
    Euler,
    ImplicitMidpoint,
)
from trajopt.models import Cartpole


@pytest.mark.julia
def test_cartpole_integrators_cross_verification(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays, LinearAlgebra")

    jl_model = jl.seval("RobotZoo.Cartpole()")
    py_model = Cartpole()

    # Discretized dynamics models in Python
    py_euler = Euler(py_model)
    py_rk4 = RK4(py_model)
    py_mid = ImplicitMidpoint(py_model, iters=10)

    # Julia evaluator helpers
    jl.seval("""
    jl_rk4 = RobotDynamics.RK4(4, 1)
    jl_euler = RobotDynamics.Euler(4, 1)

    function eval_euler_step(model, x, u, t, dt)
        RobotDynamics.integrate(jl_euler, model, SVector{4,Float64}(x...), SVector{1,Float64}(u...), t, dt)
    end

    function eval_euler_jac(model, x, u, t, dt)
        ForwardDiff.jacobian(z_ -> RobotDynamics.integrate(jl_euler, model, z_[1:4], z_[5:5], t, dt), [x; u])
    end

    function eval_rk4_step(model, x, u, t, dt)
        RobotDynamics.integrate(jl_rk4, model, SVector{4,Float64}(x...), SVector{1,Float64}(u...), t, dt)
    end

    function eval_rk4_jac(model, x, u, t, dt)
        ForwardDiff.jacobian(z_ -> RobotDynamics.integrate(jl_rk4, model, z_[1:4], z_[5:5], t, dt), [x; u])
    end

    function eval_implicit_midpoint_step(model, x, u, t, dt; iters=10)
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

    function eval_implicit_midpoint_jac(model, x, u, t, dt; iters=10)
        ForwardDiff.jacobian(z_ -> eval_implicit_midpoint_step(model, z_[1:4], z_[5:5], t, dt; iters=iters), [x; u])
    end
    """)

    jl_euler_step = jl.seval("eval_euler_step")
    jl_euler_jac = jl.seval("eval_euler_jac")
    jl_rk4_step = jl.seval("eval_rk4_step")
    jl_rk4_jac = jl.seval("eval_rk4_jac")
    jl_mid_step = jl.seval("eval_implicit_midpoint_step")
    jl_mid_jac = jl.seval("eval_implicit_midpoint_jac")

    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, -0.3, 0.4]),
        np.array([-1.5, np.pi / 3, 2.0, -1.0]),
        np.array([0.5, -np.pi / 4, -0.5, 3.0]),
        np.array([2.0, np.pi, 0.0, 0.0]),
    ]

    test_controls = [
        np.array([0.0]),
        np.array([1.5]),
        np.array([-3.2]),
        np.array([0.1]),
        np.array([10.0]),
    ]

    dt = 0.05
    t = 0.0

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 1. Euler Integrator step (1e-14) and Jacobian (1e-12)
            xnext_euler_py = py_euler.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_euler_jl = np.array(jl_euler_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_euler_py, xnext_euler_jl, rtol=1e-14, atol=1e-14)

            J_euler_py = py_euler.jacobian(x_jax, u_jax, t, dt)
            J_euler_jl = np.array(jl_euler_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_euler_py, J_euler_jl, rtol=1e-12, atol=1e-12)

            # 2. RK4 Integrator step (1e-14) and Jacobian (1e-12)
            xnext_rk4_py = py_rk4.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_rk4_jl = np.array(jl_rk4_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_rk4_py, xnext_rk4_jl, rtol=1e-14, atol=1e-14)

            J_rk4_py = py_rk4.jacobian(x_jax, u_jax, t, dt)
            J_rk4_jl = np.array(jl_rk4_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_rk4_py, J_rk4_jl, rtol=1e-12, atol=1e-12)

            # 3. Implicit Midpoint Integrator step (1e-14) and Jacobian (1e-12)
            xnext_mid_py = py_mid.discrete_dynamics(x_jax, u_jax, t, dt)
            xnext_mid_jl = np.array(jl_mid_step(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(xnext_mid_py, xnext_mid_jl, rtol=1e-14, atol=1e-14)

            J_mid_py = py_mid.jacobian(x_jax, u_jax, t, dt)
            J_mid_jl = np.array(jl_mid_jac(jl_model, x_np, u_np, t, dt))
            np.testing.assert_allclose(J_mid_py, J_mid_jl, rtol=1e-12, atol=1e-12)
