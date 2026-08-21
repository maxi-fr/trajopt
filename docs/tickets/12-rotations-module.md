# 12 — Rotations module

**What to build:** Quaternion algebra a traced, differentiated kernel can use. A caller can
compose rotations, rotate vectors, take attitude kinematics, compute an attitude error, and
convert to and from the Hamilton convention for interoperability — all under autodiff, with no
untraceable code in any hot path.

The convention is JPL with scalar-last storage, following the reference implementation. Two
details in that reference do not survive the port unchanged: the rotation-matrix conversion
routes through SciPy, which cannot be traced and must be reimplemented directly; and the
container becomes a registered pytree so quaternions can flow through compiled code as values.

Test ordering matters here and is not negotiable. The Hamilton bridge is verified first, standing
alone against known values. Only then are the conjugated comparisons against Julia written.
Reversing that order allows a sign error in the bridge to cancel a sign error in a kernel and
produce a green suite over a wrong implementation.

**Blocked by:** 01 — Numerics foundation and cones in JAX; 02 — Quaternion operand-ordering
derivation.

**Spec:** Section 7 (rotations, conventions and error map). The reference implementation is
`docs/quaternion.py`; the operand-ordering relation comes from ticket 02.

## Acceptance criteria

- [ ] The JPL product, conjugate, inverse, and vector rotation are implemented, with the vector
      rotation being the passive form consistent with the product convention
- [ ] The kinematics matrix and the quaternion derivative from a body-frame angular velocity are
      implemented
- [ ] The rotation matrix is computed directly rather than through SciPy; SciPy appears only in
      interoperability helpers and tests
- [ ] The error map produces the multiplicative small-angle error, requiring no division and no
      branch guard
- [ ] The Hamilton bridge is implemented and shown to be its own inverse
- [ ] The bridge is verified independently against known Hamilton values before any conjugated
      cross-test exists
- [ ] Cross-tests against Julia pass the matching error map explicitly rather than relying on the
      Julia default, and assert the relation derived in ticket 02
- [ ] Double-cover behaviour is tested: a quaternion and its negation produce the same rotation
      and the same error magnitude
- [ ] Quaternion values are registered pytrees and survive a round trip through compiled code
