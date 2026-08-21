# 07 — Constraint catalog and fused ConstraintList

**What to build:** The ability to attach constraints to specified ranges of knot points and
evaluate them, with values and Jacobians matching Julia. A caller adds a goal constraint at the
terminal knot, control bounds everywhere, and an obstacle constraint over a mid-horizon range,
and gets back one concatenated constraint vector per knot point.

The structural decision this ticket implements: constraints and their active ranges are fused at
build time into a single per-knot function. There is no runtime grouping by type and no runtime
batching logic. This resolves the indexed-constraint wrapper during tracing, so batched
evaluation never encounters a Python-level composition. The cost is that trace time scales with
constraint count and any structural change invalidates the trace — acceptable precisely because
structural change inside the control loop is already forbidden.

**Blocked by:** 04 — Integrators and rollout.

**Spec:** Section 10 (constraint catalog and ConstraintList), section 9 (cones and projections).

## Acceptance criteria

- [ ] The state-only, control-only, and combined constraint kinds exist, with the zero Jacobian
      blocks implied rather than stored
- [ ] The catalog covers goal, state bound, control bound, combined bound, linear, circle,
      sphere, collision, norm in both its quadratic and second-order-cone forms, and the indexed
      wrapper
- [ ] Explicit and implicit dynamics constraints are both implemented
- [ ] Constraints register against an active knot-point index range, with a dimension check at
      registration time
- [ ] Total constraint dimension per knot point is queryable across the horizon
- [ ] Box bounds expose a path that maps onto solver variable limits rather than becoming rows of
      the constraint vector
- [ ] A build step fuses all registered constraints into one concatenated function per knot
      point, evaluable in a single batched pass
- [ ] A test covers a knot point carrying several constraints of different types and output
      dimensions, confirming the concatenation order is deterministic
- [ ] Jacobians are autodiff-derived; the analytic forms in the specification are used as
      expected values in tests, not as implementation
- [ ] Cross-verification covers values and both Jacobian blocks for the whole catalog across
      active knot points at `1e-12`
- [ ] The Julia signature-mode and differentiation-mode fields are absent, not ported
