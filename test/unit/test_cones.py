import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt import _env
from trajopt.cones import (
    AbstractCone,
    NegativeOrthant,
    PositiveOrthant,
    SecondOrderCone,
    ZeroCone,
)


def test_jax_x64_enabled() -> None:
    """Verify that 64-bit precision is enabled upon package import."""
    assert _env.__version__ == "0.1.0"
    assert jax.config.jax_enable_x64 is True
    # Verify default-constructed JAX arrays are float64
    arr_float = jnp.array([1.0, 2.0, 3.0])
    assert arr_float.dtype == jnp.float64
    arr_zeros = jnp.zeros(4)
    assert arr_zeros.dtype == jnp.float64
    arr_ones = jnp.ones((2, 2))
    assert arr_ones.dtype == jnp.float64


def test_abstract_cone_hierarchy() -> None:
    """Verify all cones inherit from eqx.Module and AbstractCone."""
    for cone_cls in (ZeroCone, NegativeOrthant, PositiveOrthant, SecondOrderCone):
        assert issubclass(cone_cls, AbstractCone)
        assert issubclass(cone_cls, eqx.Module)
        cone = cone_cls()
        assert isinstance(cone, eqx.Module)


def test_zero_cone() -> None:
    cone = ZeroCone()
    x = jnp.array([1.0, -2.0, 3.0])
    b = jnp.array([0.5, -0.1, 0.4])

    np.testing.assert_allclose(cone.project(x), jnp.zeros(3))
    np.testing.assert_allclose(cone.jacobian(x), jnp.zeros((3, 3)))
    np.testing.assert_allclose(cone.hessian(x, b), jnp.zeros((3, 3)))

    # JIT verification
    jit_proj = jax.jit(cone.project)
    jit_jac = jax.jit(cone.jacobian)
    jit_hess = jax.jit(cone.hessian)

    np.testing.assert_allclose(jit_proj(x), jnp.zeros(3))
    np.testing.assert_allclose(jit_jac(x), jnp.zeros((3, 3)))
    np.testing.assert_allclose(jit_hess(x, b), jnp.zeros((3, 3)))


def test_negative_orthant() -> None:
    cone = NegativeOrthant()
    x = jnp.array([-2.0, 3.0, 0.0])
    b = jnp.array([1.0, 2.0, 3.0])
    expected_proj = jnp.array([-2.0, 0.0, 0.0])
    expected_jac = jnp.diag(jnp.array([1.0, 0.0, 1.0]))

    np.testing.assert_allclose(cone.project(x), expected_proj)
    np.testing.assert_allclose(cone.jacobian(x), expected_jac)
    np.testing.assert_allclose(cone.hessian(x, b), jnp.zeros((3, 3)))

    # JIT verification
    jit_proj = jax.jit(cone.project)
    jit_jac = jax.jit(cone.jacobian)
    jit_hess = jax.jit(cone.hessian)

    np.testing.assert_allclose(jit_proj(x), expected_proj)
    np.testing.assert_allclose(jit_jac(x), expected_jac)
    np.testing.assert_allclose(jit_hess(x, b), jnp.zeros((3, 3)))


def test_positive_orthant() -> None:
    cone = PositiveOrthant()
    x = jnp.array([-2.0, 3.0, 0.0])
    b = jnp.array([1.0, 2.0, 3.0])
    expected_proj = jnp.array([0.0, 3.0, 0.0])
    expected_jac = jnp.diag(jnp.array([0.0, 1.0, 1.0]))

    np.testing.assert_allclose(cone.project(x), expected_proj)
    np.testing.assert_allclose(cone.jacobian(x), expected_jac)
    np.testing.assert_allclose(cone.hessian(x, b), jnp.zeros((3, 3)))

    # JIT verification
    jit_proj = jax.jit(cone.project)
    jit_jac = jax.jit(cone.jacobian)
    jit_hess = jax.jit(cone.hessian)

    np.testing.assert_allclose(jit_proj(x), expected_proj)
    np.testing.assert_allclose(jit_jac(x), expected_jac)
    np.testing.assert_allclose(jit_hess(x, b), jnp.zeros((3, 3)))


def test_second_order_cone_regions() -> None:
    cone = SecondOrderCone()
    b = jnp.array([0.5, -0.3, 0.8])

    # Inside cone: ||[1, 1]||_2 = sqrt(2) <= 2
    x_inside = jnp.array([1.0, 1.0, 2.0])
    np.testing.assert_allclose(cone.project(x_inside), x_inside)
    np.testing.assert_allclose(cone.jacobian(x_inside), jnp.eye(3))
    np.testing.assert_allclose(cone.hessian(x_inside, b), jnp.zeros((3, 3)))

    # Below dual cone: ||[1, 1]||_2 = sqrt(2) <= -(-2)
    x_below = jnp.array([1.0, 1.0, -2.0])
    np.testing.assert_allclose(cone.project(x_below), jnp.zeros(3))
    np.testing.assert_allclose(cone.jacobian(x_below), jnp.zeros((3, 3)))
    np.testing.assert_allclose(cone.hessian(x_below, b), jnp.zeros((3, 3)))

    # Outside cone: x = [2, 3, 1, 1], ||[2, 3, 1]|| = sqrt(14) ≈ 3.74 > 1
    x_outside = jnp.array([2.0, 3.0, 1.0, 1.0])
    b4 = jnp.array([0.2, 0.4, -0.1, 0.3])
    px_outside = cone.project(x_outside)
    assert not jnp.isnan(px_outside).any()
    J = cone.jacobian(x_outside)
    H = cone.hessian(x_outside, b4)

    # Numerical finite difference check for Jacobian
    eps = 1e-7
    n = len(x_outside)
    J_num = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J_num[:, i] = np.array((cone.project(x_outside + dx) - cone.project(x_outside - dx)) / (2 * eps))

    np.testing.assert_allclose(J, J_num, rtol=1e-7, atol=1e-7)

    # Finite difference check for Hessian contraction
    H_num = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J_plus = cone.jacobian(x_outside + dx)
        J_minus = cone.jacobian(x_outside - dx)
        H_num[:, i] = np.array(((J_plus.T @ b4) - (J_minus.T @ b4)) / (2 * eps))

    np.testing.assert_allclose(H, H_num, rtol=1e-6, atol=1e-6)


def test_second_order_cone_boundary_and_corner_cases() -> None:
    cone = SecondOrderCone()

    # 1. Boundary a = s (boundary of primal cone)
    x_bnd_primal = jnp.array([3.0, 4.0, 5.0])
    np.testing.assert_allclose(cone.project(x_bnd_primal), x_bnd_primal, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cone.jacobian(x_bnd_primal), jnp.eye(3), rtol=1e-12, atol=1e-12)
    b3 = jnp.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(cone.hessian(x_bnd_primal, b3), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)

    # 2. Boundary a = -s (boundary of dual cone)
    x_bnd_dual = jnp.array([3.0, 4.0, -5.0])
    np.testing.assert_allclose(cone.project(x_bnd_dual), jnp.zeros(3), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cone.jacobian(x_bnd_dual), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cone.hessian(x_bnd_dual, b3), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)

    # 3. Zero norm inside: v = [0, 0], s = 2.0 (strictly inside)
    x_zero_inside = jnp.array([0.0, 0.0, 2.0])
    assert not jnp.isnan(cone.project(x_zero_inside)).any()
    assert not jnp.isnan(cone.jacobian(x_zero_inside)).any()
    assert not jnp.isnan(cone.hessian(x_zero_inside, b3)).any()
    np.testing.assert_allclose(cone.project(x_zero_inside), x_zero_inside, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cone.jacobian(x_zero_inside), jnp.eye(3), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cone.hessian(x_zero_inside, b3), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)

    # 4. Zero norm below: v = [0, 0], s = -2.0 (strictly below)
    x_zero_below = jnp.array([0.0, 0.0, -2.0])
    assert not jnp.isnan(cone.project(x_zero_below)).any()
    assert not jnp.isnan(cone.jacobian(x_zero_below)).any()
    assert not jnp.isnan(cone.hessian(x_zero_below, b3)).any()
    np.testing.assert_allclose(cone.project(x_zero_below), jnp.zeros(3), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cone.jacobian(x_zero_below), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cone.hessian(x_zero_below, b3), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)

    # 5. Origin: v = [0, 0], s = 0.0
    x_origin = jnp.array([0.0, 0.0, 0.0])
    assert not jnp.isnan(cone.project(x_origin)).any()
    assert not jnp.isnan(cone.jacobian(x_origin)).any()
    assert not jnp.isnan(cone.hessian(x_origin, b3)).any()
    np.testing.assert_allclose(cone.project(x_origin), jnp.zeros(3), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cone.jacobian(x_origin), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cone.hessian(x_origin, b3), jnp.zeros((3, 3)), rtol=1e-12, atol=1e-12)

    # 6. 1D Cone (v is empty)
    x_1d_pos = jnp.array([3.0])
    b1 = jnp.array([1.0])
    np.testing.assert_allclose(cone.project(x_1d_pos), jnp.array([3.0]))
    np.testing.assert_allclose(cone.jacobian(x_1d_pos), jnp.array([[1.0]]))
    np.testing.assert_allclose(cone.hessian(x_1d_pos, b1), jnp.array([[0.0]]))

    x_1d_neg = jnp.array([-3.0])
    np.testing.assert_allclose(cone.project(x_1d_neg), jnp.array([0.0]))
    np.testing.assert_allclose(cone.jacobian(x_1d_neg), jnp.array([[0.0]]))
    np.testing.assert_allclose(cone.hessian(x_1d_neg, b1), jnp.array([[0.0]]))


def test_second_order_cone_jit() -> None:
    cone = SecondOrderCone()
    x = jnp.array([2.0, 3.0, 1.0, 1.0])
    b = jnp.array([0.1, 0.2, 0.3, 0.4])

    jit_proj = jax.jit(cone.project)
    jit_jac = jax.jit(cone.jacobian)
    jit_hess = jax.jit(cone.hessian)

    np.testing.assert_allclose(jit_proj(x), cone.project(x), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(jit_jac(x), cone.jacobian(x), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(jit_hess(x, b), cone.hessian(x, b), rtol=1e-14, atol=1e-14)


def test_identity_cone() -> None:
    from trajopt.cones import IdentityCone

    cone = IdentityCone()
    x = jnp.array([1.5, -2.5, 3.0])
    b = jnp.array([0.1, 0.2, 0.3])

    np.testing.assert_allclose(cone.project(x), x)
    np.testing.assert_allclose(cone.jacobian(x), jnp.eye(3))
    np.testing.assert_allclose(cone.hessian(x, b), jnp.zeros((3, 3)))
    np.testing.assert_allclose(cone.project_dual(x), jnp.zeros_like(x))


def test_cone_duals_and_project_dual() -> None:
    from trajopt.cones import IdentityCone

    zero_cone = ZeroCone()
    assert isinstance(zero_cone.dual(), IdentityCone)
    x = jnp.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(zero_cone.project_dual(x), x)

    neg_cone = NegativeOrthant()
    assert isinstance(neg_cone.dual(), PositiveOrthant)
    np.testing.assert_allclose(neg_cone.project_dual(jnp.array([-2.0, 0.0, 3.0])), jnp.array([0.0, 0.0, 3.0]))

    pos_cone = PositiveOrthant()
    assert isinstance(pos_cone.dual(), NegativeOrthant)
    np.testing.assert_allclose(pos_cone.project_dual(jnp.array([-2.0, 0.0, 3.0])), jnp.array([-2.0, 0.0, 0.0]))

    soc = SecondOrderCone()
    assert isinstance(soc.dual(), SecondOrderCone)
    x_soc = jnp.array([2.0, 3.0, 1.0, 1.0])
    np.testing.assert_allclose(soc.project_dual(x_soc), soc.project(x_soc))
