from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from trajopt.rotations.quaternion import (
    Quaternion,
    attitude_jacobian,
    error_map,
    from_hamilton,
    to_hamilton,
)

SQRT_HALF = float(np.sqrt(0.5))


# --------------------------------------------------------------------------------------
# Stage 1: Independent Hamilton witness and standalone bridge verification.
# Verified against known values BEFORE any conjugated comparison exists.
# --------------------------------------------------------------------------------------


def ham_mul(a: Sequence[float] | np.ndarray | jax.Array, b: Sequence[float] | np.ndarray | jax.Array) -> np.ndarray:
    """Hamilton product a (x) b, scalar-last storage."""
    va, wa = np.asarray(a, float)[:3], float(np.asarray(a, float)[3])
    vb, wb = np.asarray(b, float)[:3], float(np.asarray(b, float)[3])
    return np.array([*(wa * vb + wb * va + np.cross(va, vb)), wa * wb - float(np.dot(va, vb))])


def ham_conj(a: Sequence[float] | np.ndarray | jax.Array) -> np.ndarray:
    """Hamilton conjugate [-v, w]."""
    arr = np.asarray(a, float)
    return np.array([-arr[0], -arr[1], -arr[2], arr[3]])


def ham_matrix(a: Sequence[float] | np.ndarray | jax.Array) -> np.ndarray:
    """Active rotation matrix of a unit Hamilton quaternion, scalar-last."""
    v, w = np.asarray(a, float)[:3], float(np.asarray(a, float)[3])
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return (w * w - float(np.dot(v, v))) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew


def jpl_axis_angle(axis: Sequence[float] | np.ndarray, angle: float) -> Quaternion:
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


def test_hamilton_algebra_defining_relations():
    """Verify Hamilton witness against defining relations ij = k, jk = i, ki = j, i^2 = -1, ji = -k."""
    i = np.array([1.0, 0.0, 0.0, 0.0])
    j = np.array([0.0, 1.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0, 0.0])
    one = np.array([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(ham_mul(i, j), k, atol=1e-15)
    np.testing.assert_allclose(ham_mul(j, k), i, atol=1e-15)
    np.testing.assert_allclose(ham_mul(k, i), j, atol=1e-15)
    np.testing.assert_allclose(ham_mul(i, i), -one, atol=1e-15)
    np.testing.assert_allclose(ham_mul(j, i), -k, atol=1e-15)


def test_bridge_against_hardcoded_hamilton_values():
    """Verify Hamilton bridge against hardcoded values only."""
    # JPL q = [sin(t/2) n, cos(t/2)] is the passive frame rotation by t about n.
    # Its Hamilton image is the active rotation by -t about the same axis, (v, w) -> (-v, w).
    q = jpl_axis_angle([1.0, 0.0, 0.0], np.pi / 2)
    np.testing.assert_allclose(q.to_array(), [SQRT_HALF, 0.0, 0.0, SQRT_HALF], atol=1e-15)
    np.testing.assert_allclose(to_hamilton(q), [-SQRT_HALF, 0.0, 0.0, SQRT_HALF], atol=1e-15)

    # Active matrix of that Hamilton quaternion is the hardcoded -90 deg x-rotation.
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(ham_matrix(to_hamilton(q)), expected, atol=1e-15)


def test_bridge_is_an_involution_and_self_inverse():
    """Verify to_hamilton and from_hamilton form a self-inverse bijection."""
    identity = Quaternion(np.zeros(3), 1.0)
    np.testing.assert_allclose(to_hamilton(identity), [0.0, 0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(from_hamilton(to_hamilton(identity)).to_array(), identity.to_array(), atol=1e-15)

    q = jpl_axis_angle([0.0, 1.0, 0.0], 0.7)
    h = to_hamilton(q)
    q_recovered = from_hamilton(h)
    np.testing.assert_allclose(q_recovered.to_array(), q.to_array(), atol=1e-15)
    np.testing.assert_allclose(to_hamilton(q_recovered), h, atol=1e-15)


def test_bridge_preserves_rotation_matrix():
    """Verify R_JPL(q) == R_Ham(to_hamilton(q))."""
    for axis, angle in [([1.0, 0, 0], 0.4), ([0, 1.0, 0], 1.1), ([0, 0, 1.0], -2.3)]:
        q = jpl_axis_angle(axis, angle)
        np.testing.assert_allclose(q.to_rot_mat(), ham_matrix(to_hamilton(q)), atol=1e-14)


def test_bridge_is_product_isomorphism():
    """Verify B(a (x)_JPL b) == B(a) (x)_Ham B(b)."""
    for a, b in random_pairs(seed=0, count=50):
        np.testing.assert_allclose(to_hamilton(a * b), ham_mul(to_hamilton(a), to_hamilton(b)), atol=1e-14)


# --------------------------------------------------------------------------------------
# Stage 2: JPL Product, conjugate, inverse, and passive vector rotation
# --------------------------------------------------------------------------------------


def test_jpl_product_and_associativity():
    """Verify JPL product associativity and identity element."""
    identity = Quaternion(np.zeros(3), 1.0)
    for a, _ in random_pairs(seed=10, count=20):
        np.testing.assert_allclose((a * identity).to_array(), a.to_array(), atol=1e-15)
        np.testing.assert_allclose((identity * a).to_array(), a.to_array(), atol=1e-15)

    q1 = jpl_axis_angle([1.0, 0.0, 0.0], 0.5)
    q2 = jpl_axis_angle([0.0, 1.0, 0.0], 0.8)
    q3 = jpl_axis_angle([0.0, 0.0, 1.0], -0.3)
    np.testing.assert_allclose(((q1 * q2) * q3).to_array(), (q1 * (q2 * q3)).to_array(), atol=1e-15)


def test_jpl_conjugate_and_inverse():
    """Verify conjugate equals inverse for unit quaternions and q * q^-1 == identity."""
    for a, _ in random_pairs(seed=20, count=20):
        a_inv = a.inverse()
        a_conj = a.conjugate()
        np.testing.assert_allclose(a_inv.to_array(), a_conj.to_array(), atol=1e-15)
        np.testing.assert_allclose(a_conj.vec, -a.vec, atol=1e-15)
        assert float(a_conj.scalar) == pytest.approx(float(a.scalar), abs=1e-15)

        prod = a * a_inv
        np.testing.assert_allclose(prod.to_array(), [0.0, 0.0, 0.0, 1.0], atol=1e-14)


def test_passive_vector_rotation():
    """Verify vector rotation is passive form consistent with R(q) @ v."""
    # Rotation of +90 deg about x transforms inertial y-axis [0, 1, 0] to body z-axis [0, 0, -1]
    q = jpl_axis_angle([1.0, 0.0, 0.0], np.pi / 2)
    v = np.array([0.0, 1.0, 0.0])
    v_rot = q.apply(v)
    expected = np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(v_rot, expected, atol=1e-15)

    # q.apply(v) must match q.to_rot_mat() @ v for arbitrary vectors and rotations
    rng = np.random.default_rng(30)
    for q_rand, _ in random_pairs(seed=30, count=30):
        vec = rng.normal(size=3)
        np.testing.assert_allclose(q_rand.apply(vec), q_rand.to_rot_mat() @ vec, atol=1e-14)
        assert np.linalg.norm(q_rand.apply(vec)) == pytest.approx(np.linalg.norm(vec), abs=1e-14)


# --------------------------------------------------------------------------------------
# Stage 3: Direct rotation matrix vs SciPy witness
# --------------------------------------------------------------------------------------


def test_direct_rotation_matrix_matches_scipy():
    """Verify direct JAX rotation matrix against SciPy interop without calling SciPy in to_rot_mat."""
    for q, _ in random_pairs(seed=40, count=50):
        r_direct = q.to_rot_mat()
        r_scipy = q.to_scipy().as_matrix()
        np.testing.assert_allclose(r_direct, r_scipy, atol=1e-14)

        # Assert orthogonality and proper rotation
        np.testing.assert_allclose(r_direct.T @ r_direct, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(r_direct), 1.0, atol=1e-14)


def test_from_scipy_and_to_scipy_roundtrip():
    """Verify conversion to and from SciPy Rotation."""
    rot = Rotation.from_euler("xyz", [15.0, -30.0, 60.0], degrees=True)
    q = Quaternion.from_scipy(rot)
    rot_recovered = q.to_scipy()
    np.testing.assert_allclose(q.to_rot_mat(), rot.as_matrix(), atol=1e-14)
    np.testing.assert_allclose(rot_recovered.as_matrix(), rot.as_matrix(), atol=1e-14)


# --------------------------------------------------------------------------------------
# Stage 4: Kinematics matrix Xi(q) and derivative dq/dt
# --------------------------------------------------------------------------------------


def test_kinematics_matrix_and_derivative():
    """Verify Xi(q) matrix and quaternion derivative from body angular velocity."""
    q = jpl_axis_angle([0.0, 0.0, 1.0], 0.6)
    xi = q.xi()
    assert xi.shape == (4, 3)

    # Xi(q) definition: [w*I + [v x]; -v^T]
    w = float(q.scalar)
    x, y, z = float(q.vec[0]), float(q.vec[1]), float(q.vec[2])
    expected_xi = np.array(
        [
            [w, -z, y],
            [z, w, -x],
            [-y, x, w],
            [-x, -y, -z],
        ]
    )
    np.testing.assert_allclose(xi, expected_xi, atol=1e-15)

    # Derivative dq/dt = 0.5 * Xi(q) @ omega
    omega = np.array([0.1, -0.2, 0.5])
    q_dot = q.kinematics(omega)
    assert q_dot.shape == (4,)
    np.testing.assert_allclose(q_dot, 0.5 * expected_xi @ omega, atol=1e-15)

    # Scalar-first option
    xi_sf = q.xi(scalar_first=True)
    assert xi_sf.shape == (4, 3)
    np.testing.assert_allclose(xi_sf[0, :], expected_xi[3, :], atol=1e-15)
    np.testing.assert_allclose(xi_sf[1:, :], expected_xi[:3, :], atol=1e-15)


def test_attitude_jacobian():
    """Verify attitude_jacobian returns 0.5 * Xi(q) of shape (4, 3)."""
    q = jpl_axis_angle([1.0, -1.0, 0.5], 1.2)
    G = attitude_jacobian(q)
    assert G.shape == (4, 3)
    np.testing.assert_allclose(G, 0.5 * q.xi(), atol=1e-15)


# --------------------------------------------------------------------------------------
# Stage 5: Multiplicative error map
# --------------------------------------------------------------------------------------


def test_error_map_is_multiplicative_small_angle():
    """Verify error_map produces 2 * vec(q (x) q_ref^-1) without division or branch guard."""
    for q, q_ref in random_pairs(seed=50, count=30):
        q_err = q.error_to(q_ref)
        dtheta = error_map(q, q_ref)
        assert dtheta.shape == (3,)
        np.testing.assert_allclose(dtheta, 2.0 * q_err.vec, atol=1e-15)

    # Identical attitudes have zero error
    q_same = jpl_axis_angle([0.5, 0.5, 0.0], 0.9)
    np.testing.assert_allclose(error_map(q_same, q_same), np.zeros(3), atol=1e-15)


# --------------------------------------------------------------------------------------
# Stage 6: Double-cover behaviour
# --------------------------------------------------------------------------------------


def test_double_cover_behaviour():
    """Verify q and -q produce identical rotation matrices, vector rotations, and error magnitudes."""
    for q, q_ref in random_pairs(seed=60, count=30):
        q_neg = -q

        # Same rotation matrix
        np.testing.assert_allclose(q.to_rot_mat(), q_neg.to_rot_mat(), atol=1e-15)

        # Same vector rotation
        v = np.array([1.2, -0.4, 0.7])
        np.testing.assert_allclose(q.apply(v), q_neg.apply(v), atol=1e-15)

        # Same error magnitude
        err_pos = q.error_to(q_ref)
        err_neg = q_neg.error_to(q_ref)
        assert np.linalg.norm(err_pos.vec) == pytest.approx(np.linalg.norm(err_neg.vec), abs=1e-15)
        assert np.linalg.norm(error_map(q, q_ref)) == pytest.approx(np.linalg.norm(error_map(q_neg, q_ref)), abs=1e-15)


# --------------------------------------------------------------------------------------
# Stage 7: JAX PyTree registration, JIT, vmap, grad, and scan compatibility
# --------------------------------------------------------------------------------------


def test_quaternion_is_pytree_and_survives_jit():
    """Verify Quaternion round trips through jax.jit as a compiled value."""

    @jax.jit
    def rotate_point(q: Quaternion, pt: jax.Array) -> jax.Array:
        return q.apply(pt)

    @jax.jit
    def compose_rotations(q1: Quaternion, q2: Quaternion) -> Quaternion:
        return q1 * q2

    q1 = jpl_axis_angle([1.0, 0.0, 0.0], 0.3)
    q2 = jpl_axis_angle([0.0, 1.0, 0.0], 0.7)
    pt = jnp.array([1.0, 2.0, 3.0])

    res_pt = rotate_point(q1, pt)
    expected_pt = q1.apply(pt)
    np.testing.assert_allclose(res_pt, expected_pt, atol=1e-15)

    res_q = compose_rotations(q1, q2)
    expected_q = q1 * q2
    np.testing.assert_allclose(res_q.to_array(), expected_q.to_array(), atol=1e-15)


def test_quaternion_vmap():
    """Verify Quaternion works under jax.vmap across batches."""
    q_list = [jpl_axis_angle([1.0, 0.0, 0.0], 0.1 * i) for i in range(5)]
    q_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *q_list)

    @jax.vmap
    def batch_to_rot_mat(q: Quaternion) -> jax.Array:
        return q.to_rot_mat()

    mats = batch_to_rot_mat(q_batch)
    assert mats.shape == (5, 3, 3)
    for i, q in enumerate(q_list):
        np.testing.assert_allclose(mats[i], q.to_rot_mat(), atol=1e-15)


def test_quaternion_autodiff():
    """Verify gradients flow through Quaternion operations."""

    def loss(q_arr: jax.Array, target_mat: jax.Array) -> jax.Array:
        q = Quaternion.from_array(q_arr)
        return jnp.sum((q.to_rot_mat() - target_mat) ** 2)

    q0 = jpl_axis_angle([0.0, 1.0, 0.0], 0.5)
    target = jpl_axis_angle([0.0, 1.0, 0.0], 0.6).to_rot_mat()

    grad_fn = jax.grad(loss)
    g = grad_fn(q0.to_array(), target)
    assert g.shape == (4,)
    assert not jnp.any(jnp.isnan(g))
