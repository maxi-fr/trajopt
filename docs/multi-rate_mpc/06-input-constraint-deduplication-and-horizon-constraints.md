# 06 — Input-constraint deduplication and linear Horizon constraints in both shooting formulations

**What to build:** Deduplicate repeated time-independent input-only Knot Point constraints within each control block and support `LinearHorizonConstraint` rows across both `Ipopt` multiple shooting and `SingleShooting` for `control_block_steps >= 1`.

**Blocked by:** 04 — Blocked continuous multiple shooting in `Ipopt`, 05 — Blocked continuous `SingleShooting`.

## Acceptance criteria

- [ ] Built-in constraints and wrappers (`LinearConstraint`, `IndexedConstraint`, `OutputConstraint`, and epigraph constraints) accurately classify whether they read state coordinates (`uses_state()`) and whether they are time-independent input-only constraints; custom constraints default conservatively to native evaluation at every applicable Knot Point.
- [ ] Within each control block of `b` native intervals, structurally equivalent time-independent input-only constraint groups (same expression, coefficients, right-hand side, Cone, and coordinate mapping) are collapsed so only the first applicable native occurrence in that block is retained in the NLP, even when that occurrence is not at the block's first Knot Point.
- [ ] State-dependent, mixed state/control, time-dependent, and terminal constraints retain every registered native evaluation; distinct input constraints or constraints with non-overlapping active ranges are not collapsed.
- [ ] A Kirchhoff-style control equality registered across all `H` native Knot Points produces exactly `B = H // b` retained equality rows in both shooting formulations, solves without rank-deficiency from duplicate block rows, and still passes full native feasibility verification on all `H` original native rows.
- [ ] Both `Ipopt` and `SingleShooting` support `LinearHorizonConstraint` (both selected-index `inds` and full-vector forms) at the trailing position of the NLP and canonical constraint vectors for `b >= 1`.
- [ ] Repeated canonical control columns in a `LinearHorizonConstraint` that map to the same independent block control accumulate their coefficients (`b` active native intervals in one block contribute `b` times the per-step weight to an activation budget), preserving native budget units without rounding or dividing by `b`.
- [ ] In `SingleShooting`, `LinearHorizonConstraint` rows are evaluated on the canonical Primal Vector reconstructed from the native Rollout and expanded controls, so Horizon rows that read state coordinates include exact Rollout contributions in both the dense constraint Jacobian and the dense Lagrangian Hessian (verified against central finite differences).
