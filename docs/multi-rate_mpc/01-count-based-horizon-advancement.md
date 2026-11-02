# 01 — Count-based horizon advancement (`MPC.advance`)

**What to build:** Replace the duration-based `MPC.shift` interface with count-based `MPC.advance(steps=1)` across the receding-horizon driver, warm-start helpers, simulation controller, benchmarks, examples, and tests. Advancing by `steps` moves the canonical warm start (states, controls, canonical Multipliers, and padded augmented-Lagrangian state), the reference window, and the clock forward by `steps` native intervals in the structural `Problem` time grid without solving, running a Rollout, reevaluating the Objective, or modifying the latest solver result.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [x] `MPC.shift` and its duration argument are removed with no compatibility alias; `MPC.advance(steps=1)` is the sole advancement method and all package callers, examples, and tests use it.
- [x] `steps` accepts only integers in `[0, H]` (where `H = N - 1`) and rejects booleans, fractional values, negative values, and `steps > H` with a `ValueError` or `TypeError` before mutating any driver state.
- [x] `MPC.advance(0)` is a complete no-op: warm start, `t0`, reference window, pending pushed reference point, terminal goal `xf`, and `result` remain unchanged.
- [x] For `steps >= 1`, warm-start states `X` and controls `U` advance by `steps` native Knot Points, holding their final native entries to fill the entering tail.
- [x] Canonical constraint Multipliers `lam`, bound Multipliers `mu`, and padded augmented-Lagrangian state `al` advance by applying the existing one-step shift and mask rules `steps` times, including initial-state pin retention, terminal-row resets, stage row-count mismatch handling, trailing Horizon Multiplier retention, and real-row penalty hold next to terminal padding.
- [x] `t0` advances by the sum of the first `steps` native durations in the structural `Problem.dt` array (`steps * dt` on a uniform grid; the consumed prefix sum `sum(dt[:steps])` on a nonuniform grid), while `Problem.dt` itself remains unchanged.
- [x] Reference states `X_ref` and controls `U_ref` advance by `steps` Knot Points using endpoint hold semantics: a pending `push_reference` point is appended on the first internal one-step shift and consumed once, and subsequent steps in the same `advance` call hold that new endpoint.
- [x] The terminal goal `xf` remains unchanged across `advance`, and `set_reference` continues to replace the tracked reference window wholesale.
- [x] Calling `MPC.measure(x, t)` after `MPC.advance` replaces both `BoundaryConditions.x0` (and `t0`) and the first Knot Point state `X[0]` in the canonical warm start before the next solve, so the shifted predicted state is never used as the measured initial state.
- [x] On a uniform grid, `MPC.advance(6)` produces identical canonical Primal Vector `Z`, `lam`, `mu`, `al`, reference window, and `t0` to six consecutive `MPC.advance(1)` calls across cases with a pending reference push, no pending push, constant references, moving references, and an independent terminal goal.
