# 02 — Quaternion operand-ordering derivation

**What to build:** A settled, written-down answer to the one open question in the specification:
how the Python attitude error relates to the Julia one.

The reference implementation computes the error as `q ⊗ q_ref⁻¹` under the JPL product. Julia
computes it as `q_ref⁻¹ ⊗ q` under the Hamilton product. These share a scalar part but their
vector parts differ in the sign of the cross-product term, so they are not related by a global
sign, and no cross-test of the rotation code can be written until the exact relation is known.

This is a derivation, not a feature. Nothing ships from it except a proof, a numerical check,
and a specification edit — but it gates the entire SO(3) strand, and it has a real chance of
forcing a convention change, which is why it runs first and in parallel.

**Blocked by:** None — can start immediately.

**Status:** Done. Derivation in `docs/quaternion_operand_ordering.md`, numerical check in
`test/unit/test_quaternion_ordering.py`, specification section 7 updated. No convention change
was required.

**Spec:** Section 7 (rotations), especially the subsection marked unresolved; Appendix A
(conventions). The reference implementation is `docs/quaternion.py`.

## Acceptance criteria

- [x] The relation between the two error definitions is derived symbolically for a general pair
      of quaternions, not fitted to an example
- [x] The derivation is verified numerically on a rotation pair whose vector parts have a nonzero
      cross product — for example a rotation about one axis composed against a rotation about
      another — so that operand order demonstrably matters
- [x] The Hamilton bridge conversion is verified independently against known Hamilton values,
      before any conjugated comparison exists, so a sign error in the bridge cannot cancel a sign
      error in a kernel
- [x] A degenerate case where the two orderings coincide, meaning parallel vector parts, is
      included and shown to be uninformative — documenting why it must not be used as the test
      case
- [x] The specification's unresolved subsection is replaced with the derived relation
- [x] If the derivation shows the chosen conventions cannot be reconciled cleanly, the required
      convention change is written up as a decision rather than worked around
