"""Numerical verification of the JPL/Hamilton attitude-error operand-ordering relation.

Companion to ``docs/quaternion_operand_ordering.md``, which carries the symbolic derivation.
This module pins the result numerically so it cannot silently drift.

Test order matters and is load-bearing. The Hamilton bridge is verified first against
hardcoded Hamilton values and an independently written Hamilton product, before any test
compares a bridged quantity to a JPL one. A sign error in the bridge therefore cannot cancel
a sign error in a kernel and produce a green suite over a wrong implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs"))

from quaternion import Quaternion  # ty: ignore[unresolved-import]

SQRT_HALF = float(np.sqrt(0.5))


# --------------------------------------------------------------------------------------
# An independent Hamilton implementation. Written from the textbook Hamilton formula, with
# no reference to docs/quaternion.py, so that it is an independent witness rather than a
# restatement of the code under test. Storage is scalar-last, [v1, v2, v3, w].
# --------------------------------------------------------------------------------------


def ham_mul(a, b):
    """Hamilton product a (x) b, scalar-last storage."""
    va, wa = np.asarray(a, float)[:3], float(a[3])
    vb, wb = np.asarray(b, float)[:3], float(b[3])
    return np.array([*(wa * vb + wb * va + np.cross(va, vb)), wa * wb - va @ vb])


def ham_conj(a):
    """Hamilton conjugate, which is the inverse for a unit quaternion."""
    a = np.asarray(a, float)
    return np.array([-a[0], -a[1], -a[2], a[3]])


def ham_matrix(a):
    """Active rotation matrix of a unit Hamilton quaternion, scalar-last."""
    v, w = np.asarray(a, float)[:3], float(a[3])
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew


def to_hamilton(q: Quaternion):
    """The bridge under test: JPL scalar-last to Hamilton scalar-last, ``(v, w) -> (-v, w)``."""
    return np.array([*(-q.vec), q.scalar])


def jpl_axis_angle(axis, angle) -> Quaternion:
    """JPL quaternion for a rotation of ``angle`` about unit ``axis``."""
    axis = np.asarray(axis, float)
    return Quaternion(axis * np.sin(angle / 2.0), float(np.cos(angle / 2.0)))


def random_pairs(seed, count):
    """Random unit-quaternion pairs ``(q, q_ref)``, so relations are checked generally."""
    rng = np.random.default_rng(seed)
    return [tuple(Quaternion.from_array(v / np.linalg.norm(v)) for v in rng.normal(size=(2, 4))) for _ in range(count)]


# --------------------------------------------------------------------------------------
# Stage 1 -- the independent Hamilton witness itself
# --------------------------------------------------------------------------------------


def test_hamilton_product_matches_known_values():
    i = np.array([1.0, 0.0, 0.0, 0.0])
    j = np.array([0.0, 1.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0, 0.0])
    one = np.array([0.0, 0.0, 0.0, 1.0])

    # The defining relations of the Hamilton algebra: ij = k, jk = i, ki = j, i^2 = -1, ji = -k.
    np.testing.assert_allclose(ham_mul(i, j), k, atol=1e-15)
    np.testing.assert_allclose(ham_mul(j, k), i, atol=1e-15)
    np.testing.assert_allclose(ham_mul(k, i), j, atol=1e-15)
    np.testing.assert_allclose(ham_mul(i, i), -one, atol=1e-15)
    np.testing.assert_allclose(ham_mul(j, i), -k, atol=1e-15)


def test_hamilton_matrix_matches_known_values():
    # Active rotation by +90 degrees about x maps y -> z.
    q = np.array([SQRT_HALF, 0.0, 0.0, SQRT_HALF])
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(ham_matrix(q), expected, atol=1e-15)


# --------------------------------------------------------------------------------------
# Stage 2 -- the bridge, against hardcoded Hamilton values only.
#
# No conjugated comparison exists yet at this point in the file.
# --------------------------------------------------------------------------------------


def test_bridge_against_hardcoded_hamilton_values():
    # JPL q = [sin(t/2) n, cos(t/2)] is the *frame* (passive) rotation by t about n. Its
    # Hamilton image is the active rotation by -t about the same axis, so the vector part
    # flips and the scalar part does not.
    q = jpl_axis_angle([1.0, 0.0, 0.0], np.pi / 2)
    np.testing.assert_allclose(q.to_array(), [SQRT_HALF, 0.0, 0.0, SQRT_HALF], atol=1e-15)
    np.testing.assert_allclose(to_hamilton(q), [-SQRT_HALF, 0.0, 0.0, SQRT_HALF], atol=1e-15)

    # And that Hamilton quaternion's active matrix is the hardcoded -90 degree x-rotation.
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(ham_matrix(to_hamilton(q)), expected, atol=1e-15)


def test_bridge_is_an_involution_and_fixes_the_identity():
    identity = Quaternion(np.zeros(3), 1.0)
    np.testing.assert_allclose(to_hamilton(identity), [0.0, 0.0, 0.0, 1.0], atol=1e-15)

    q = jpl_axis_angle([0.0, 1.0, 0.0], 0.7)
    round_trip = to_hamilton(Quaternion.from_array(to_hamilton(q)))
    np.testing.assert_allclose(round_trip, q.to_array(), atol=1e-15)


def test_bridge_preserves_the_rotation_matrix():
    # The reference's to_rot_mat is, by construction, the active Hamilton matrix of the
    # bridged quaternion. Checking that against the independent Hamilton matrix formula
    # verifies the bridge's geometric meaning without a quaternion-to-quaternion comparison.
    for axis, angle in [([1.0, 0, 0], 0.4), ([0, 1.0, 0], 1.1), ([0, 0, 1.0], -2.3)]:
        q = jpl_axis_angle(axis, angle)
        np.testing.assert_allclose(q.to_rot_mat(), ham_matrix(to_hamilton(q)), atol=1e-14)


def test_bridge_is_a_product_isomorphism():
    # B(a (x)_JPL b) == B(a) (x)_Ham B(b). Only reached after the bridge is independently pinned.
    for a, b in random_pairs(seed=0, count=50):
        np.testing.assert_allclose(to_hamilton(a * b), ham_mul(to_hamilton(a), to_hamilton(b)), atol=1e-14)


# --------------------------------------------------------------------------------------
# Stage 3 -- the operand-ordering relation
# --------------------------------------------------------------------------------------


def julia_error(q: Quaternion, q_ref: Quaternion):
    """What ``RobotDynamics.state_diff`` builds: ``q_ref^-1 (x)_Ham q`` on bridged operands."""
    return ham_mul(ham_conj(to_hamilton(q_ref)), to_hamilton(q))


def bridged_python_error(q: Quaternion, q_ref: Quaternion):
    """The bridge applied to the reference's ``q (x)_JPL q_ref^-1``."""
    return to_hamilton(q.error_to(q_ref))


# A pure-x against a pure-y rotation: the vector parts are orthogonal, so the cross-product
# term is at its largest and operand order demonstrably matters.
X_Y_PAIR = (jpl_axis_angle([1.0, 0.0, 0.0], 0.9), jpl_axis_angle([0.0, 1.0, 0.0], -1.3))


def test_orderings_genuinely_differ_on_the_x_y_pair():
    q, q_ref = X_Y_PAIR
    a, c = bridged_python_error(q, q_ref), julia_error(q, q_ref)

    assert not np.allclose(a, c), "test pair is uninformative: the two orderings agree"
    assert not np.allclose(a, -c), "test pair is uninformative: the two differ by a global sign"

    # They differ by exactly -2 (x cross y) in the vector part, where x and y are the bridged
    # vector parts, and agree exactly in the scalar part.
    x, y = to_hamilton(q)[:3], to_hamilton(q_ref)[:3]
    np.testing.assert_allclose(a[:3] - c[:3], -2.0 * np.cross(x, y), atol=1e-14)
    assert a[3] == pytest.approx(c[3], abs=1e-15)


def test_error_orderings_are_a_similarity_transform():
    """C = p_ref^-1 (x) A (x) p_ref, for a general pair and for the x/y pair specifically."""
    for q, q_ref in [X_Y_PAIR, *random_pairs(seed=1, count=200)]:
        p, p_ref = to_hamilton(q), to_hamilton(q_ref)
        a, c = bridged_python_error(q, q_ref), julia_error(q, q_ref)
        np.testing.assert_allclose(c, ham_mul(ham_mul(ham_conj(p_ref), a), p_ref), atol=1e-13)
        np.testing.assert_allclose(a, ham_mul(ham_mul(p, c), ham_conj(p)), atol=1e-13)


def test_vector_parts_are_related_by_the_reference_rotation():
    """vec(C) = R_JPL(q_ref)^T vec(A) = -R_JPL(q_ref)^T vec(q_err), the shipping form."""
    for q, q_ref in [X_Y_PAIR, *random_pairs(seed=2, count=200)]:
        r_ref = q_ref.to_rot_mat()
        a, c = bridged_python_error(q, q_ref), julia_error(q, q_ref)
        q_err = q.error_to(q_ref)

        np.testing.assert_allclose(c[:3], r_ref.T @ a[:3], atol=1e-13)
        np.testing.assert_allclose(c[:3], -r_ref.T @ q_err.vec, atol=1e-13)
        assert c[3] == pytest.approx(q_err.scalar, abs=1e-13)


def test_relation_survives_any_radial_error_map():
    """Both orderings have equal scalar parts and equal vector-part norms.

    Every error map in play here -- dtheta = 2v, the Cayley map v/w, and the exponential map
    -- rescales the vector part by a factor depending only on w and the norm of v. Both
    orderings take the same factor, so the rotation relation passes through the map unchanged
    and holds for whichever map the cross-test selects.
    """
    for q, q_ref in [X_Y_PAIR, *random_pairs(seed=3, count=200)]:
        a, c = bridged_python_error(q, q_ref), julia_error(q, q_ref)
        assert a[3] == pytest.approx(c[3], abs=1e-13)
        assert np.linalg.norm(a[:3]) == pytest.approx(np.linalg.norm(c[:3]), abs=1e-13)


# --------------------------------------------------------------------------------------
# Stage 4 -- the degenerate cases, recorded so they are never used as the test case
# --------------------------------------------------------------------------------------


def test_parallel_vector_parts_make_the_orderings_coincide():
    """Two rotations about a common axis are useless as a cross-test: the orderings agree.

    The two definitions differ only in the sign of x cross y. Coaxial rotations have parallel
    vector parts, so that term vanishes and the equality holds for *any* operand convention.
    Such a pair would pass the cross-test against a reversed implementation.
    """
    q = jpl_axis_angle([0.0, 0.0, 1.0], 0.9)
    q_ref = jpl_axis_angle([0.0, 0.0, 1.0], -1.3)

    a, c = bridged_python_error(q, q_ref), julia_error(q, q_ref)
    np.testing.assert_allclose(a, c, atol=1e-15)

    # The rotation in the general relation degenerates to a no-op on this particular vector,
    # which is the precise sense in which the case carries no information.
    np.testing.assert_allclose(q_ref.to_rot_mat().T @ a[:3], a[:3], atol=1e-15)


def test_identity_reference_is_also_degenerate():
    """q_ref = identity is the other uninformative case.

    R_ref becomes the identity and the relation collapses to a plain vector-part negation,
    which cannot distinguish the two operand orders.
    """
    axis = np.array([1.0, 2.0, -0.5])
    q = jpl_axis_angle(axis / np.linalg.norm(axis), 1.1)
    q_ref = Quaternion(np.zeros(3), 1.0)

    np.testing.assert_allclose(q_ref.to_rot_mat(), np.eye(3), atol=1e-15)
    np.testing.assert_allclose(bridged_python_error(q, q_ref), julia_error(q, q_ref), atol=1e-15)
