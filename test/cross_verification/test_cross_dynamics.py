"""Cross-verification tests comparing Python dynamics models against Julia RobotZoo/RobotDynamics."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.models import Cartpole


@pytest.mark.julia
def test_cartpole_continuous_dynamics_cross(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays")

    jl_model = jl.seval("RobotZoo.Cartpole()")
    py_model = Cartpole()

    # Define Julia evaluator helper
    jl_eval_dyn = jl.seval("""
    function (model, x, u)
        RobotDynamics.dynamics(model, SVector{4,Float64}(x...), SVector{1,Float64}(u...))
    end
    """)

    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, -0.3, 0.4]),
        np.array([-1.5, np.pi / 3, 2.0, -1.0]),
        np.array([0.5, -np.pi / 4, -0.5, 3.0]),
        np.array([2.0, np.pi, 0.0, 0.0]),  # Upright equilibrium
    ]

    test_controls = [
        np.array([0.0]),
        np.array([1.5]),
        np.array([-3.2]),
        np.array([0.1]),
        np.array([10.0]),
    ]

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 1. Continuous dynamics comparison (tol 1e-14)
            xdot_py = py_model.dynamics(x_jax, u_jax)
            xdot_jl = np.array(jl_eval_dyn(jl_model, x_np, u_np))
            np.testing.assert_allclose(xdot_py, xdot_jl, rtol=1e-14, atol=1e-14)


@pytest.mark.julia
def test_cartpole_continuous_jacobian_cross(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays")

    jl_model = jl.seval("RobotZoo.Cartpole()")
    py_model = Cartpole()

    # Define Julia Jacobian evaluator helper
    jl_eval_jac = jl.seval("""
    function (model, x, u)
        z = [x; u]
        ForwardDiff.jacobian(z_ -> RobotDynamics.dynamics(model, z_[1:4], z_[5:5]), z)
    end
    """)

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

    for x_np in test_states:
        for u_np in test_controls:
            x_jax = jnp.array(x_np)
            u_jax = jnp.array(u_np)

            # 2. Continuous Jacobian comparison (tol 1e-12)
            J_py = py_model.jacobian(x_jax, u_jax)
            J_jl = np.array(jl_eval_jac(jl_model, x_np, u_np))
            np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cartpole_custom_parameters_cross(jl_to: Any) -> None:
    jl = jl_to
    jl.seval("using RobotZoo, RobotDynamics, ForwardDiff, StaticArrays")

    mc, mp, pole_length, g = 2.5, 0.8, 1.2, 9.80665

    jl_model = jl.RobotZoo.Cartpole(mc, mp, pole_length, g)
    py_model = Cartpole(mc=mc, mp=mp, l=pole_length, g=g)

    jl_eval_dyn = jl.seval("""
    function (model, x, u)
        RobotDynamics.dynamics(model, SVector{4,Float64}(x...), SVector{1,Float64}(u...))
    end
    """)
    jl_eval_jac = jl.seval("""
    function (model, x, u)
        z = [x; u]
        ForwardDiff.jacobian(z_ -> RobotDynamics.dynamics(model, z_[1:4], z_[5:5]), z)
    end
    """)

    x_np = np.array([0.3, -0.5, 1.2, -0.8])
    u_np = np.array([-4.5])

    x_jax = jnp.array(x_np)
    u_jax = jnp.array(u_np)

    xdot_py = py_model.dynamics(x_jax, u_jax)
    xdot_jl = np.array(jl_eval_dyn(jl_model, x_np, u_np))
    np.testing.assert_allclose(xdot_py, xdot_jl, rtol=1e-14, atol=1e-14)

    J_py = py_model.jacobian(x_jax, u_jax)
    J_jl = np.array(jl_eval_jac(jl_model, x_np, u_np))
    np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)
