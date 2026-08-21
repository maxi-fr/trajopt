"""Unit tests for Ipopt solver integration via cyipopt."""

import numpy as np
import pytest

import trajopt


def test_ipopt_rosenbrock() -> None:
    """Verify that cyipopt solves the 2D Rosenbrock optimization problem."""
    cyipopt = pytest.importorskip("cyipopt")

    # Minimize 2D Rosenbrock function: f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2
    def objective(x: np.ndarray) -> float:
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def gradient(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                200.0 * (x[1] - x[0] ** 2),
            ]
        )

    x0 = np.array([0.0, 0.0])
    res = cyipopt.minimize_ipopt(objective, x0, jac=gradient)

    assert res.success, f"Ipopt failed to converge: {res.message}"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)
