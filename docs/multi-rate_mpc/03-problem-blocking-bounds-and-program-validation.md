# 03 — Structural control blocking on `Problem`, block bounds, and `Program` pairing guards

**What to build:** Add structural uniform control blocking (`control_block_steps`) to `Problem`, preserve it across supported `Problem` transformations, compute and validate intersected per-block control bounds (including binary domain checks), and enforce supported `Problem`/Backend pairings at `Program` construction and direct Backend entry points before compilation or optimization begins.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] `Problem` accepts a static structural field `control_block_steps: int = 1` (`b`), validating at construction that `b` is a positive integer (rejecting `bool`, `float`, and `b < 1`), that `H = N - 1 >= 1`, and that `H` is divisible by `b`.
- [ ] Blocking does not alter `Problem.N`, model state/control dimensions, `Problem.dt`, integration, or the canonical Primal Vector length `N * n + H * m`.
- [ ] `retarget_problem` and `add_l1_epigraph` preserve `control_block_steps` on the returned `Problem`; `add_l1_epigraph` blocks both original and epigraph slack coordinates under the same block size `b`.
- [ ] Native binary control bounds are intersected with `[0, 1]` before block bounds are calculated; for each block of `b` native intervals and each control coordinate, the block lower bound is the maximum native lower bound in the block and the block upper bound is the minimum native upper bound in the block, preserving infinite bounds and fixed controls (`lower == upper`).
- [ ] If any block's intersected lower bound exceeds its upper bound, or if a declared binary control coordinate's intersected block interval contains neither `0.0` nor `1.0` (even when the continuous interval is nonempty, such as `[0.2, 0.8]`), validation fails with a `ValueError` before IPOPT is invoked.
- [ ] Constructing `Program(problem, solver)` (and therefore `MPC(problem, solver, ...)`) as well as direct Backend `solve` calls validate the `Problem`/solver combination upfront before compilation or numerical work:
  - `Ipopt` and `SingleShooting(Ipopt)` are accepted for continuous and binary Problems, `control_block_steps >= 1`, and `LinearHorizonConstraint` rows (removing the `isinstance(self.solver, Ipopt)` rejection of `SingleShooting` on binary and Horizon-constrained Problems).
  - `SingleShooting` continues to reject any finite state box bounds, including terminal state box bounds.
  - All other Backends and Native Solvers (`ALTRO`, `iLQR`, `AugmentedLagrangianSolver`, `SQP`, `OSQP`, `Clarabel`, `PIQP`, `SLSQP`) explicitly reject `control_block_steps > 1` with a `ValueError` while keeping their existing `control_block_steps == 1` behavior intact.
  - Constraint Cones that the IPOPT representation cannot express (`SecondOrderCone` or other non-interval Cones) are rejected explicitly rather than reinterpreted as scalar equalities.
