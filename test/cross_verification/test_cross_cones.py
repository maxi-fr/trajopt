"""Cross-verification tests comparing Python cone implementations against TrajectoryOptimization.jl."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone


# The Julia names contain "!" and a Unicode nabla, so none of them is a valid Python identifier.
# They have to be reached with getattr rather than attribute syntax.
def _jl_project(jl_to: Any) -> Any:
    return getattr(jl_to.TO, "projection!")


def _jl_jacobian(jl_to: Any) -> Any:
    return getattr(jl_to.TO, "∇projection!")


def _jl_hessian(jl_to: Any) -> Any:
    return getattr(jl_to.TO, "∇²projection!")


@pytest.mark.julia
def test_zero_cone_cross(jl_to: Any) -> None:
    cone_py = ZeroCone()
    cone_jl = jl_to.TO.ZeroCone()

    x_np = np.array([1.0, -2.0, 3.0])
    b_np = np.array([0.5, -0.2, 0.1])
    x = jnp.array(x_np)
    b = jnp.array(b_np)

    # 1. Projection check
    px_py = cone_py.project(x)
    px_jl = np.zeros_like(x_np)
    _jl_project(jl_to)(cone_jl, px_jl, x_np)
    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)

    # 2. Jacobian check
    J_py = cone_py.jacobian(x)
    J_jl = np.zeros((len(x_np), len(x_np)))
    _jl_jacobian(jl_to)(cone_jl, J_jl, x_np)
    np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)

    # 3. Hessian contraction check
    H_py = cone_py.hessian(x, b)
    H_jl = np.zeros((len(x_np), len(x_np)))
    _jl_hessian(jl_to)(cone_jl, H_jl, x_np, b_np)
    np.testing.assert_allclose(H_py, H_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_negative_orthant_cross(jl_to: Any) -> None:
    cone_py = NegativeOrthant()
    cone_jl = jl_to.TO.NegativeOrthant()

    x_np = np.array([-2.0, 3.0, 0.0])
    b_np = np.array([0.5, -0.2, 0.1])
    x = jnp.array(x_np)
    b = jnp.array(b_np)

    # 1. Projection check
    px_py = cone_py.project(x)
    px_jl = np.zeros_like(x_np)
    _jl_project(jl_to)(cone_jl, px_jl, x_np)
    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)

    # 2. Jacobian check
    J_py = cone_py.jacobian(x)
    J_jl = np.zeros((len(x_np), len(x_np)))
    _jl_jacobian(jl_to)(cone_jl, J_jl, x_np)
    np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)

    # 3. Hessian contraction check
    H_py = cone_py.hessian(x, b)
    H_jl = np.zeros((len(x_np), len(x_np)))
    _jl_hessian(jl_to)(cone_jl, H_jl, x_np, b_np)
    np.testing.assert_allclose(H_py, H_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_positive_orthant_cross(jl_to: Any) -> None:
    cone_py = PositiveOrthant()
    cone_jl = jl_to.TO.PositiveOrthant()

    x_np = np.array([-2.0, 3.0, 0.0])
    x = jnp.array(x_np)

    # 1. Projection check against Julia out-of-place projection
    px_py = cone_py.project(x)
    px_jl = np.array(jl_to.TO.projection(cone_jl, x_np))
    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)


@pytest.mark.julia
def test_second_order_cone_cross(jl_to: Any) -> None:
    cone_py = SecondOrderCone()
    cone_jl = jl_to.TO.SecondOrderCone()

    test_vectors = [
        # Inside cone
        np.array([1.0, 1.0, 2.0]),
        np.array([0.5, -0.2, 0.1, 0.8]),
        # Below dual cone
        np.array([1.0, 1.0, -2.0]),
        np.array([2.0, 3.0, 1.0, -10.0]),
        # Outside cone (positive and negative scalar)
        np.array([2.0, 3.0, 1.0, 1.0]),
        np.array([2.0, 3.0, 1.0, -1.0]),
        # Boundary cases: ||v|| == s and ||v|| == -s
        np.array([3.0, 4.0, 5.0]),
        np.array([3.0, 4.0, -5.0]),
        # Zero norm boundary cases
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
    ]

    for x_np in test_vectors:
        n = len(x_np)
        b_np = np.linspace(0.1, 0.9, n)
        x = jnp.array(x_np)
        b = jnp.array(b_np)

        # 1. Projection check (tol 1e-14)
        px_py = cone_py.project(x)
        px_jl = np.zeros_like(x_np)
        _jl_project(jl_to)(cone_jl, px_jl, x_np)
        np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)

        # 2. Jacobian check (tol 1e-12)
        J_py = cone_py.jacobian(x)
        J_jl = np.zeros((n, n))
        _jl_jacobian(jl_to)(cone_jl, J_jl, x_np)
        np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)

        # 3. Hessian contraction check (tol 1e-12)
        H_py = cone_py.hessian(x, b)
        H_jl = np.zeros((n, n))
        _jl_hessian(jl_to)(cone_jl, H_jl, x_np, b_np)
        np.testing.assert_allclose(H_py, H_jl, rtol=1e-12, atol=1e-12)
