"""Cross-verification tests comparing Python cone implementations against TrajectoryOptimization.jl."""

import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone


@pytest.mark.julia
def test_zero_cone_cross(jl_to) -> None:
    cone_py = ZeroCone()
    cone_jl = jl_to.TO.ZeroCone()

    x = np.array([1.0, -2.0, 3.0])
    px_py = cone_py.project(x)
    px_jl = np.zeros_like(x)
    jl_to.TO.projection_b(cone_jl, px_jl, x)

    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)


@pytest.mark.julia
def test_negative_orthant_cross(jl_to) -> None:
    cone_py = NegativeOrthant()
    cone_jl = jl_to.TO.NegativeOrthant()

    x = np.array([-2.0, 3.0, 0.0])
    px_py = cone_py.project(x)
    px_jl = np.zeros_like(x)
    jl_to.TO.projection_b(cone_jl, px_jl, x)

    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)


@pytest.mark.julia
def test_second_order_cone_cross(jl_to) -> None:
    cone_py = SecondOrderCone()
    cone_jl = jl_to.TO.SecondOrderCone()

    test_vectors = [
        np.array([1.0, 1.0, 2.0]),  # inside
        np.array([1.0, 1.0, -2.0]),  # below
        np.array([2.0, 3.0, 1.0, 1.0]),  # outside
        np.array([0.5, -0.2, 0.1, 0.8]),  # inside
    ]

    for x in test_vectors:
        # 1. Projection check
        px_py = cone_py.project(x)
        px_jl = np.zeros_like(x)
        jl_to.TO.projection_b(cone_jl, px_jl, x)
        np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)

        # 2. Jacobian check
        J_py = cone_py.jacobian(x)
        J_jl = np.zeros((len(x), len(x)))
        jl_to.TO.grad_projection_b(cone_jl, J_jl, x)
        np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)
