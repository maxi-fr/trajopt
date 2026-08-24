import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs"))

from quaternion import Quaternion  # ty: ignore[unresolved-import]

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
