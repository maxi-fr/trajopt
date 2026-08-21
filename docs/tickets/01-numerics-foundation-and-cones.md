# 01 — Numerics foundation and cones in JAX

**What to build:** The numerical ground floor everything else stands on, proven by making the
cone library work. A caller can project a vector onto any of the four cones, get the Jacobian of
that projection, and get the second-derivative contraction, and every one of those answers
matches TrajectoryOptimization.jl to full double precision.

The foundation half is unglamorous but blocking: JAX defaults to 32-bit floats, so until 64-bit
precision is enforced at import, every `1e-14` tolerance in the specification is unreachable and
every cross-test fails in a way that looks like a math error rather than a configuration error.

**Blocked by:** None — can start immediately.

**Spec:** Section 4 (backend, data model, and compilation), section 9 (cones and projections),
section 15 (verification strategy).

## Acceptance criteria

- [ ] 64-bit precision is enabled when the package is imported, and a test asserts that a
      default-constructed array is 64-bit
- [ ] `equinox` is a declared dependency and the lockfile and exported requirements are
      regenerated
- [ ] All four cones — zero, negative orthant, positive orthant, second-order — implement
      projection, projection Jacobian, and the second-derivative contraction
- [ ] The second-order cone contains no Python branching on traced values; region selection is
      branchless and the division by the vector norm is guarded against zero
- [ ] The second-derivative contraction is autodiff-derived rather than hand-written
- [ ] Cross-verification against Julia covers the second-order cone in all three regions —
      inside, outside, below the dual cone — plus the boundary cases where the norm is zero and
      where the norm exactly equals the scalar part
- [ ] Tolerances hold at `1e-14` for projections and `1e-12` for derivatives
- [ ] The pre-existing NumPy cone implementation and its tests are fully replaced, not left
      alongside
