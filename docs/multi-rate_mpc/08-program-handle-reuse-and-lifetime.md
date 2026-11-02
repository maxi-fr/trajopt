# 08 — Persistent IPOPT handle reuse and zero-recompilation across `Program` lifetime

**What to build:** Build and retain one external `cyipopt.Problem` handle and one set of compiled numerical callback kernels per `Program` for both `Ipopt` and `SingleShooting`, updating boundary conditions, run-time targets, and continuation `beta` as dynamic numerical data across solves and continuation passes without handle reconstruction or JAX retracing.

**Blocked by:** 06 — Input-constraint deduplication and linear Horizon constraints in both shooting formulations, 07 — Shared binary continuation across `Ipopt` and `SingleShooting`.

## Acceptance criteria

- [ ] A `Program` configured with `Ipopt` or `SingleShooting(Ipopt)` constructs its shooting layout, sparsity structures, compiled JAX callback kernels, and external `cyipopt.Problem` handle once and retains them in `Program` across continuation passes and subsequent `Program.solve` / `MPC.solve` calls.
- [ ] Updating `x0`, `t0`, `X_ref`, `U_ref`, `xf` (via `MPC.advance`, `MPC.measure`, `MPC.set_goal`, `MPC.set_reference`, `MPC.push_reference`), and continuation `beta` passes updated numerical values to the retained callback and handle without reconstructing the `Problem`, reallocating `cyipopt.Problem`, or retracing compiled JAX functions after the first warm-up solve.
- [ ] Per-solve state (`beta`, `iteration_count`, pass count, status, and homotopy diagnostics) is initialized afresh at the start of every `solve` call so a prior solve's failure, final `beta`, or iteration count never leaks into the next solve attempt.
- [ ] Instrumentation tests verify both (a) single `cyipopt.Problem` handle allocation across multiple `MPC.advance` / `MPC.measure` / `MPC.solve` cycles (including binary continuation passes and a failed solve followed by a successful solve) and (b) zero additional JAX traces of the objective, gradient, constraint, Jacobian, and Hessian kernels after warm-up when varying `x0`, `t0`, `X_ref`, `U_ref`, `xf`, and `beta`.
- [ ] Root domain documentation (`CONTEXT.md` and `docs/adr/` if any architectural decision is amended) and public docstrings are reconciled with count-based advancement, uniform control blocking, canonical `SingleShooting` Multipliers, and `Program`-owned IPOPT handle lifetime.
