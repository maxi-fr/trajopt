# 07 — Shared binary continuation across `Ipopt` and `SingleShooting`

**What to build:** Share the binary penalty continuation loop across `Ipopt` multiple shooting and `SingleShooting` for both unblocked (`b = 1`) and blocked (`b > 1`) Problems, using native-weighted binary penalties so continuation strength is invariant to block size.

**Blocked by:** 04 — Blocked continuous multiple shooting in `Ipopt`, 05 — Blocked continuous `SingleShooting`, 06 — Input-constraint deduplication and linear Horizon constraints in both shooting formulations.

## Acceptance criteria

- [ ] `SingleShooting` reads `beta_initial`, `beta_growth`, `max_passes`, and `binary_tolerance` directly from its `Ipopt` instance and validates them with the same rules as `Ipopt` before solving a Problem with `binary_control_indices`.
- [ ] Both `Ipopt` and `SingleShooting` execute the same continuation schedule: pass 0 solves the unpenalized continuous relaxation (`beta = 0.0`), subsequent passes set `beta = beta_initial * beta_growth ** (pass_index - 1)`, internal reduced primal and dual iterates (`lam`, `mu`) are carried from pass to pass within a solve, and a failed pass terminates the continuation loop immediately.
- [ ] For block size `b`, each declared binary block coordinate `a` contributes the native-weighted penalty `b * beta * a * (1 - a)` to the callback objective, `b * beta * (1 - 2 * a)` to the reduced objective gradient, and `-2 * b * beta * obj_factor` to the corresponding reduced Hessian diagonal entry (verified against finite differences at fractional controls for both `b = 1` and `b > 1`). Both `"lbfgs"` and `"dense"` Hessian modes in `SingleShooting` use the same continuation penalty schedule.
- [ ] Neither formulation rounds primal iterates to manufacture integrality or feasibility. `success` is `True` if and only if IPOPT returns an accepted termination status, at least one penalized pass (`pass_count > 1`) completes, maximum binary distance `max(min(|a|, |1 - a|))` is at most `binary_tolerance`, and canonical expanded constraint violation is at most `constr_viol_tol`.
- [ ] Reaching `max_passes` with fractional binary controls or failing the initial continuous relaxation returns `success=False` alongside the actual expanded iterate, unpenalized retargeted Objective `cost`, canonical `constraint_violation`, and `info` diagnostics (`homotopy_beta`, native-weighted `homotopy_penalty`, `binary_distance`, and `homotopy_passes`).
- [ ] On deterministic blocked binary Problems with a native Horizon activation budget (`H = 12, b = 6`, budget `6` allowing one active block, budget `< 6` allowing none, and a non-multiple budget such as `8`), both `Ipopt` and `SingleShooting` match the enumerated block-assignment oracle.
