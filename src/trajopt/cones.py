"""Conic sets, projections, and derivatives for trajectory optimization constraints."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractCone(ABC):
    """Abstract base class for convex cones."""

    @abstractmethod
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project vector x onto the cone."""

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Jacobian of the projection at x: ∇Π(x)."""

    def hessian(self, x: np.ndarray, _b: np.ndarray) -> np.ndarray:
        """Evaluate the Hessian contraction ∇²Π(x)[b]. Default is zero matrix."""
        n = len(x)
        return np.zeros((n, n), dtype=x.dtype)


class ZeroCone(AbstractCone):
    """Zero cone representing equality constraints g(x) = 0."""

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project vector x onto the zero cone."""
        return np.zeros_like(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Jacobian of the projection onto the zero cone."""
        n = len(x)
        return np.zeros((n, n), dtype=x.dtype)


class NegativeOrthant(AbstractCone):
    """Negative orthant representing inequality constraints h(x) <= 0."""

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project vector x onto the negative orthant."""
        return np.minimum(0.0, x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Jacobian of the projection onto the negative orthant."""
        return np.diag((x <= 0.0).astype(x.dtype))


class PositiveOrthant(AbstractCone):
    """Positive orthant representing inequality constraints h(x) >= 0."""

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project vector x onto the positive orthant."""
        return np.maximum(0.0, x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Jacobian of the projection onto the positive orthant."""
        return np.diag((x >= 0.0).astype(x.dtype))


class SecondOrderCone(AbstractCone):
    """Second-order cone (Lorentz cone / ice cream cone): ||v||_2 <= s, where x = [v; s]."""

    def status(self, x: np.ndarray, _eps: float = 1e-10) -> str:
        """Determine region status of x relative to the cone."""
        v = x[:-1]
        s = x[-1]
        a = np.linalg.norm(v)
        if a <= -s:
            return "below"
        if a <= s:
            return "inside"
        return "outside"

    def project(self, x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Project vector x onto the second-order cone."""
        v = x[:-1]
        s = x[-1]
        a = np.linalg.norm(v)

        if a <= -s:
            return np.zeros_like(x)
        if a <= s:
            return x.copy()
        safe_a = max(a, eps)
        scale = 0.5 * (1.0 + s / safe_a)
        return np.append(scale * v, scale * safe_a)

    def jacobian(self, x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Evaluate the Jacobian of the projection onto the second-order cone."""
        n = len(x)
        v = x[:-1]
        s = x[-1]
        a = np.linalg.norm(v)

        if a <= -s:
            return np.zeros((n, n), dtype=x.dtype)
        if a <= s:
            return np.eye(n, dtype=x.dtype)
        safe_a = max(a, eps)
        I_v = np.eye(n - 1, dtype=x.dtype)
        vvT = np.outer(v, v)

        # Top-left block: 0.5 * ((1 + s/a)*I - (s/a^3)*v*v^T)
        J_vv = 0.5 * ((1.0 + s / safe_a) * I_v - (s / (safe_a**3)) * vvT)
        # Top-right block: 0.5 * (v / a)
        J_vs = 0.5 * (v / safe_a).reshape(-1, 1)
        # Bottom-left block: 0.5 * ((1 + s/a)*(v^T / a) - (s / a^2)*v^T) = 0.5 * (v^T / a)
        J_sv = 0.5 * (v / safe_a).reshape(1, -1)
        # Bottom-right scalar: 0.5
        J_ss = np.array([[0.5]], dtype=x.dtype)

        J_top = np.hstack([J_vv, J_vs])
        J_bot = np.hstack([J_sv, J_ss])
        return np.vstack([J_top, J_bot])
