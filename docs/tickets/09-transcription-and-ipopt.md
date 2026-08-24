# 09 — NLP transcription and the first Ipopt solve

**What to build:** The tracer bullet. A caller defines a cartpole swing-up — dynamics, LQR cost,
goal constraint, control bounds — and it solves to optimality through Ipopt. This is the first
slice where every layer built so far is exercised end to end by a real optimization.

The interesting work is sparse assembly, and it is the largest chunk of genuinely novel
implementation in the project: no library does it. The solver's structure callback fires once and
its values callback fires every iteration, returning a flat array in the structure's order. So
the sparsity pattern is computed at build time from the dimensions alone, and at runtime a
batched pass produces dense per-knot blocks whose values are placed into a preallocated array in
exactly that order. If the ordering disagrees with the pattern, the failure mode is a wrong
answer rather than an error — so the ordering needs a direct test, not only an end-to-end one.

**Blocked by:** 08 — Expansion engine, Euclidean path.

**Spec:** Section 12 (NLP transcription), section 4 (compilation units), section 5 (why the flat
primal vector is not the storage of record).

## Acceptance criteria

- [x] The primal vector interleaves states and controls with a trailing terminal state, and the
      interleaving is owned by the transcription layer alone — trajectory storage stays
      struct-of-arrays
- [x] The constraint vector composes the initial-state condition, the dynamics defects, and the
      stage constraints in a documented order. Box bounds are absent from it: `ConstraintList.build`
      hoists them into the primal variable limits, since a duplicated bound row has gradient `e_i`
      — the same as the variable bound already active — and so degenerates the active set
- [x] The sparsity pattern is computed at build time as a pure function of the dimensions, using
      host arrays rather than traced ones
- [x] Per-knot Jacobian blocks are treated as dense; structural zeros inside a block are not
      exploited
- [x] A test asserts directly that the runtime value ordering matches the build-time pattern
      ordering, independent of whether any solve converges
- [x] Four independently compiled phases exist, matching the solver's objective, gradient,
      constraint, constraint-Jacobian, and Hessian callbacks
- [x] No sparse matrix objects are allocated inside the iteration loop
- [x] Cartpole swing-up solves to optimality with bounded actuation and a terminal goal
      constraint
