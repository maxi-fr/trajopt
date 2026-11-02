# 05 — Blocked continuous `SingleShooting`

**What to build:** Enable blocked continuous trajectory optimization (`control_block_steps >= 1`) in `SingleShooting` across both `"lbfgs"` and `"dense"` exact Hessian modes through `Program.solve`, `MPC.solve`, and `SingleShooting.transcription_callback`. `SingleShooting` optimizes `B * m` independent block controls, expands them to `H` native controls before rolling out all `N` native states through the discrete dynamics, and returns native-grid Trajectories and canonical-sized results that agree with blocked multiple shooting.

**Blocked by:** 02 — Canonical Multipliers, Cone feasibility, and dual warm starts in `SingleShooting`, 03 — Structural control blocking on `Problem`, block bounds, and `Program` pairing guards, 04 — Blocked continuous multiple shooting in `Ipopt`.

## Acceptance criteria

- [ ] `single_shooting_dimensions` and `SingleShooting.transcription_callback` size the internal NLP decision vector to `B * m` (where `B = (N - 1) // b`) and apply the intersected per-block control bounds of shape `(B * m,)`.
- [ ] Warm-start controls are reduced from the canonical `WarmStart` by taking the first native control of each block (`U[::b]`); warm-start states are discarded for the numerical solve and recovered by native Rollout from `x0` and expanded controls.
- [ ] Every callback evaluation expands the `B` block controls into `H = N - 1` native controls and integrates all `H` native transitions with their native `t` and `dt`, so Stage Costs, terminal Cost, output maps, and state-dependent Knot Point constraints see all `N` native states rather than a block-boundary subsampling.
- [ ] `SingleShooting.transcription_callback` differentiates Objective and retained constraints with respect to the `B * m` block control vector in both `"lbfgs"` and `"dense"` Hessian modes; `callback.gradient`, `callback.jacobian`, and lower-triangular `callback.hessian` match central finite differences on a smooth blocked Problem (`b > 1`).
- [ ] Both successful and failed `SingleShooting` solves expand the returned block control vector to native `U` of shape `(H, m)`, roll out native `X` of shape `(N, n)`, reconstruct canonical `Z`, and compute canonical native constraint violation (including block-constancy verification).
- [ ] For `b > 1`, `SingleShooting` exports canonical-sized zero `lam` and `mu` arrays and does not consume canonical Multipliers from preceding solves; for `b = 1`, it preserves the adjoint-recovered canonical Multipliers and dual warm-start behavior from Ticket 02.
- [ ] On a well-conditioned continuous linear-quadratic Problem with a unique optimum and `b > 1` (including unequal native control bounds across a block and a scalar state constraint active mid-block), blocked `SingleShooting` (`"lbfgs"` and `"dense"`) and blocked `Ipopt` multiple shooting agree on optimal cost, expanded controls, native states, and constraint residuals within solver tolerance.
