from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.dynamics.base import RigidBody
from trajopt.rotations.quaternion import (
    Quaternion,
    attitude_jacobian,
    error_map,
    to_hamilton,
)

SQRT_HALF = float(np.sqrt(0.5))


def jpl_axis_angle(axis: list[float] | np.ndarray, angle: float) -> Quaternion:
    """JPL quaternion for a rotation of ``angle`` about unit ``axis``."""
    axis_arr = np.asarray(axis, float)
    axis_norm = np.linalg.norm(axis_arr)
    unit_axis = axis_arr / axis_norm if axis_norm > 0 else axis_arr
    return Quaternion(unit_axis * np.sin(angle / 2.0), float(np.cos(angle / 2.0)))


def random_pairs(seed: int, count: int) -> list[tuple[Quaternion, Quaternion]]:
    """Generate random unit-quaternion pairs."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(count):
        v1 = rng.normal(size=4)
        v1 /= np.linalg.norm(v1)
        v2 = rng.normal(size=4)
        v2 /= np.linalg.norm(v2)
        pairs.append((Quaternion.from_array(v1), Quaternion.from_array(v2)))
    return pairs


# A pure-x against a pure-y rotation: vector parts are orthogonal, cross product is maximal.
X_Y_PAIR = (jpl_axis_angle([1.0, 0.0, 0.0], 0.9), jpl_axis_angle([0.0, 1.0, 0.0], -1.3))


def _py_to_julia_unitquat(jl: Any, q: Quaternion) -> Any:
    """Convert Python JPL quaternion to Julia UnitQuaternion (Hamilton scalar-first: w, x, y, z)."""
    h = to_hamilton(q)  # [-v0, -v1, -v2, w]
    return jl.Rotations.UnitQuaternion(float(h[3]), float(h[0]), float(h[1]), float(h[2]))


@pytest.mark.julia
def test_rotation_matrix_cross(jl_to: Any) -> None:
    """Assert direct JAX rotation matrix matches Julia RotMatrix for bridged quaternions."""
    jl = jl_to
    jl.seval("using Rotations")

    for q, _ in [X_Y_PAIR, *random_pairs(seed=100, count=50)]:
        q_jl = _py_to_julia_unitquat(jl, q)
        r_jl = np.array(jl.Rotations.RotMatrix(q_jl))
        r_py = np.array(q.to_rot_mat())
        np.testing.assert_allclose(r_py, r_jl, rtol=1e-14, atol=1e-14)


@pytest.mark.julia
def test_vector_rotation_cross(jl_to: Any) -> None:
    """Assert passive vector rotation matches Julia rotation matrix-vector product."""
    jl = jl_to
    jl.seval("using Rotations, StaticArrays")

    rng = np.random.default_rng(101)
    for q, _ in [X_Y_PAIR, *random_pairs(seed=101, count=30)]:
        v = rng.normal(size=3)
        v_jl = jl.SVector(float(v[0]), float(v[1]), float(v[2]))
        q_jl = _py_to_julia_unitquat(jl, q)

        rot_jl = np.array(jl.Rotations.RotMatrix(q_jl) * v_jl)
        rot_py = np.array(q.apply(v))
        np.testing.assert_allclose(rot_py, rot_jl, rtol=1e-14, atol=1e-14)


@pytest.mark.julia
def test_quaternion_error_operand_ordering_cross(jl_to: Any) -> None:
    r"""Assert relation derived in ticket 02 against live Julia Rotations.jl error quaternion.

    Derived relation:
    vec(delta_q_Julia) = -R(q_ref)^T vec(q_err)
    scalar(delta_q_Julia) = scalar(q_err)
    where q_err = q (x)_JPL q_ref^-1 and delta_q_Julia = q_ref_jl \ q_jl.
    """
    jl = jl_to
    jl.seval("using Rotations, StaticArrays")

    for q, q_ref in [X_Y_PAIR, *random_pairs(seed=102, count=100)]:
        q_jl = _py_to_julia_unitquat(jl, q)
        q_ref_jl = _py_to_julia_unitquat(jl, q_ref)

        # In Julia: delta_q = q_ref_jl \ q_jl
        delta_q_jl = jl.seval("function(q, q0) q0 \\ q end")(q_jl, q_ref_jl)
        delta_q_arr = np.array(jl.Rotations.params(delta_q_jl))  # Julia UnitQuaternion params: [w, x, y, z]

        scalar_jl = float(delta_q_arr[0])
        vec_jl = delta_q_arr[1:4]

        # Python quantities
        q_err = q.error_to(q_ref)
        r_ref = np.array(q_ref.to_rot_mat())

        # Assert derived relation: vec(delta_q_jl) = -R(q_ref)^T @ q_err.vec
        expected_vec_jl = -r_ref.T @ np.array(q_err.vec)
        np.testing.assert_allclose(vec_jl, expected_vec_jl, rtol=1e-13, atol=1e-13)
        assert scalar_jl == pytest.approx(float(q_err.scalar), abs=1e-13)


@pytest.mark.julia
def test_explicit_rotation_error_map_cross(jl_to: Any) -> None:
    r"""Assert error maps passed explicitly to Julia Rotations.rotation_error.

    Tests CayleyMap and MRPMap via Julia's Rotations.rotation_error(R1, R2, map)
    which evaluates map⁻¹(R2 \ R1).
    Asserts map(delta_q_Julia) = -R(q_ref)^T map(q_err_Python).
    """
    jl = jl_to
    jl.seval("using Rotations, StaticArrays")

    for q, q_ref in [X_Y_PAIR, *random_pairs(seed=103, count=50)]:
        q_jl = _py_to_julia_unitquat(jl, q)
        q_ref_jl = _py_to_julia_unitquat(jl, q_ref)

        q_err = q.error_to(q_ref)
        v_err = np.array(q_err.vec)
        w_err = float(q_err.scalar)
        r_ref = np.array(q_ref.to_rot_mat())

        # 1. CayleyMap (v / w in Rotations.jl) passed explicitly to Julia Rotations.rotation_error
        cayley_jl = np.array(jl.Rotations.rotation_error(q_jl, q_ref_jl, jl.Rotations.CayleyMap()))
        cayley_py = v_err / w_err
        np.testing.assert_allclose(cayley_jl, -r_ref.T @ cayley_py, rtol=1e-13, atol=1e-13)

        # 2. MRPMap (2 * v / (1 + w) in Rotations.jl) passed explicitly to Julia Rotations.rotation_error
        mrp_jl = np.array(jl.Rotations.rotation_error(q_jl, q_ref_jl, jl.Rotations.MRPMap()))
        mrp_py = 2.0 * v_err / (1.0 + w_err)
        np.testing.assert_allclose(mrp_jl, -r_ref.T @ mrp_py, rtol=1e-13, atol=1e-13)

        # 3. Multiplicative small-angle error delta_theta = 2 * vec(q_err)
        dtheta_py = np.array(error_map(q, q_ref))
        delta_q_jl = jl.seval("function(q, q0) q0 \\ q end")(q_jl, q_ref_jl)
        dtheta_jl = 2.0 * np.array(jl.Rotations.params(delta_q_jl))[1:4]
        np.testing.assert_allclose(dtheta_jl, -r_ref.T @ dtheta_py, rtol=1e-13, atol=1e-13)


@pytest.mark.julia
def test_degenerate_coaxial_and_identity_cross(jl_to: Any) -> None:
    """Assert coaxial rotations and identity reference match degenerate behavior."""
    jl = jl_to
    jl.seval("using Rotations")

    # Coaxial pair
    q_coax = jpl_axis_angle([0.0, 0.0, 1.0], 0.9)
    q_ref_coax = jpl_axis_angle([0.0, 0.0, 1.0], -1.3)
    q_jl = _py_to_julia_unitquat(jl, q_coax)
    q_ref_jl = _py_to_julia_unitquat(jl, q_ref_coax)

    delta_q_jl = jl.seval("function(q, q0) q0 \\ q end")(q_jl, q_ref_jl)
    delta_q_arr = np.array(jl.Rotations.params(delta_q_jl))
    vec_jl = delta_q_arr[1:4]

    r_ref = np.array(q_ref_coax.to_rot_mat())
    q_err = q_coax.error_to(q_ref_coax)

    # For coaxial rotations, R_ref^T @ vec(A) == vec(A), so relation holds degenerately
    np.testing.assert_allclose(vec_jl, -r_ref.T @ np.array(q_err.vec), atol=1e-15)

    # Identity reference
    q_rand = jpl_axis_angle([1.0, 2.0, -0.5], 1.1)
    q_ref_id = Quaternion(np.zeros(3), 1.0)
    q_jl = _py_to_julia_unitquat(jl, q_rand)
    q_ref_jl = _py_to_julia_unitquat(jl, q_ref_id)

    delta_q_jl = jl.seval("function(q, q0) q0 \\ q end")(q_jl, q_ref_jl)
    delta_q_arr = np.array(jl.Rotations.params(delta_q_jl))
    vec_jl = delta_q_arr[1:4]
    q_err = q_rand.error_to(q_ref_id)

    np.testing.assert_allclose(vec_jl, -np.array(q_err.vec), atol=1e-15)


@pytest.mark.julia
def test_attitude_jacobian_cross(jl_to: Any) -> None:
    r"""Assert derived relation between Python attitude Jacobian and Julia ∇differential.

    Derivation:
    In Python (JPL):
        delta_q = G_py(q) @ delta_theta_py, where G_py(q) = 0.5 * Xi(q)
    In Julia (Hamilton scalar-first):
        delta_h = (0.5 * ∇diff(h)) @ delta_theta_jl
    Since h = T @ q with T mapping JPL scalar-last to Hamilton scalar-first,
        delta_h = T @ delta_q = T @ G_py(q) @ delta_theta_py
    Using the derived relation delta_theta_jl = -R(q)^T @ delta_theta_py:
        (0.5 * ∇diff(h)) @ (-R(q)^T @ delta_theta_py) = T @ G_py(q) @ delta_theta_py
    Therefore:
        T @ G_py(q) = -0.5 * ∇diff(h) @ R(q)^T
    or equivalently:
        0.5 * ∇diff(h) = -T @ G_py(q) @ R(q)
    """
    jl = jl_to
    jl.seval("using Rotations")
    jl_diff = getattr(jl.Rotations, "∇differential")

    # Permutation matrix T mapping [x, y, z, w] to [w, -x, -y, -z]
    T = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )

    # 1. Test on non-degenerate orthogonal pair X_Y_PAIR
    for q, _ in [X_Y_PAIR, *random_pairs(seed=200, count=50)]:
        q_jl = _py_to_julia_unitquat(jl, q)
        G_jl = np.array(jl_diff(q_jl))
        G_py = np.array(attitude_jacobian(q))
        R_ref = np.array(q.to_rot_mat())

        # Assert derived relation: T @ G_py = -0.5 * G_jl @ R^T
        expected_TG_py = -0.5 * G_jl @ R_ref.T
        np.testing.assert_allclose(T @ G_py, expected_TG_py, rtol=1e-12, atol=1e-12)


class _TestRigidBody(RigidBody):
    def __init__(self, m: int = 4) -> None:
        super().__init__(m=m)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del u, t
        return jnp.zeros_like(x)


@pytest.mark.julia
def test_rigid_body_errstate_and_jacobian_cross(jl_to: Any) -> None:
    """Assert RigidBody state_diff and errstate_jacobian match Julia LieState across random states."""
    jl = jl_to
    jl.seval("using Rotations, StaticArrays, LinearAlgebra")

    model_py = _TestRigidBody(m=4)

    T_quat = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
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

    rng = np.random.default_rng(202)
    for q, q_ref in [X_Y_PAIR, *random_pairs(seed=202, count=30)]:
        r = rng.standard_normal(3)
        r0 = rng.standard_normal(3)
        v = rng.standard_normal(3)
        v0 = rng.standard_normal(3)
        omega = rng.standard_normal(3)
        omega0 = rng.standard_normal(3)

        x_py = jnp.array(np.concatenate([r, q.to_array(), v, omega]))
        x0_py = jnp.array(np.concatenate([r0, q_ref.to_array(), v0, omega0]))

        # 1. Error state comparison
        dx_py = np.array(model_py.state_diff(x_py, x0_py))

        q_jl = _py_to_julia_unitquat(jl, q)
        q_ref_jl = _py_to_julia_unitquat(jl, q_ref)
        delta_q_jl = jl.seval("function(q, q0) q0 \\ q end")(q_jl, q_ref_jl)
        dtheta_jl = 2.0 * np.array(jl.Rotations.params(delta_q_jl))[1:4]

        # In Julia error state: [dr, dtheta_jl, dv, domega]
        dx_jl_expected = np.concatenate([r - r0, dtheta_jl, v - v0, omega - omega0])
        # Mapping relation between Python and Julia error states
        R_ref = np.array(q_ref.to_rot_mat())
        E_mat = np.block(
            [
                [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), -R_ref.T, np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
            ]
        )
        np.testing.assert_allclose(dx_jl_expected, E_mat @ dx_py, rtol=1e-12, atol=1e-12)

        # 2. errstate_jacobian comparison
        G_py = np.array(model_py.errstate_jacobian(x_py))
        assert G_py.shape == (13, 12)

        jl_diff = getattr(jl.Rotations, "∇differential")
        G_rot_jl = np.array(jl_diff(q_jl))
        G_jl = np.block(
            [
                [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((4, 3)), 0.5 * G_rot_jl, np.zeros((4, 3)), np.zeros((4, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
            ]
        )
        R_curr = np.array(q.to_rot_mat())
        E_curr = np.block(
            [
                [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), -R_curr.T, np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
            ]
        )
        # Assert T_13 @ G_py == G_jl @ E_curr
        np.testing.assert_allclose(T_13 @ G_py, G_jl @ E_curr, rtol=1e-12, atol=1e-12)
