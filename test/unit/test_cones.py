"""Unit tests for cone sets and projections."""

import numpy as np

from trajopt.cones import NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone


def test_zero_cone() -> None:
    cone = ZeroCone()
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(cone.project(x), np.zeros(3))
    np.testing.assert_allclose(cone.jacobian(x), np.zeros((3, 3)))


def test_negative_orthant() -> None:
    cone = NegativeOrthant()
    x = np.array([-2.0, 3.0, 0.0])
    expected_proj = np.array([-2.0, 0.0, 0.0])
    expected_jac = np.diag([1.0, 0.0, 1.0])
    np.testing.assert_allclose(cone.project(x), expected_proj)
    np.testing.assert_allclose(cone.jacobian(x), expected_jac)


def test_positive_orthant() -> None:
    cone = PositiveOrthant()
    x = np.array([-2.0, 3.0, 0.0])
    expected_proj = np.array([0.0, 3.0, 0.0])
    expected_jac = np.diag([0.0, 1.0, 1.0])
    np.testing.assert_allclose(cone.project(x), expected_proj)
    np.testing.assert_allclose(cone.jacobian(x), expected_jac)


def test_second_order_cone_regions() -> None:
    cone = SecondOrderCone()

    # Inside cone: ||[1, 1]||_2 = sqrt(2) <= 2
    x_inside = np.array([1.0, 1.0, 2.0])
    assert cone.status(x_inside) == "inside"
    np.testing.assert_allclose(cone.project(x_inside), x_inside)
    np.testing.assert_allclose(cone.jacobian(x_inside), np.eye(3))

    # Below dual cone: ||[1, 1]||_2 = sqrt(2) <= -(-2)
    x_below = np.array([1.0, 1.0, -2.0])
    assert cone.status(x_below) == "below"
    np.testing.assert_allclose(cone.project(x_below), np.zeros(3))
    np.testing.assert_allclose(cone.jacobian(x_below), np.zeros((3, 3)))

    # Outside cone: x = [2, 3, 1, 1], ||[2, 3, 1]|| = sqrt(14) ≈ 3.74 > 1
    x_outside = np.array([2.0, 3.0, 1.0, 1.0])
    assert cone.status(x_outside) == "outside"
    cone.project(x_outside)
    J = cone.jacobian(x_outside)

    # Numerical finite difference check for Jacobian
    eps = 1e-7
    n = len(x_outside)
    J_num = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J_num[:, i] = (cone.project(x_outside + dx) - cone.project(x_outside - dx)) / (2 * eps)

    np.testing.assert_allclose(J, J_num, rtol=1e-5, atol=1e-5)
