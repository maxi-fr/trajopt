# ADR 0006: Carrying augmented-Lagrangian duals and penalties across horizon steps

## Status

Partially accepted. Three defects in the carry are fixed; the penalty-carry question itself is
deliberately left open, and this ADR records why.

## Context

`SolverOptions` exposes `reset_duals` and `reset_penalties`. In Altro.jl these apply once, at the
start of a whole solve. In trajopt they also govern what one receding-horizon step hands to the
next, through `WarmStart.shift` and `WarmStart.al` -- a layer Altro.jl does not have at all.

Only the default `(reset_duals=True, reset_penalties=True)` worked. The 2x2 factorial on the
kicked cartpole of `test/unit/test_mpc_characterization.py` (N = 40, 40 closed-loop steps, a
`KICK` of 1.5 injected into the pole rate at step 20):

| `reset_duals` | `reset_penalties` | recovers | final theta err | max violation | AL iters |
| --- | --- | --- | --- | --- | --- |
| True | True (default) | yes | 0.161 | 5.8e-7 | 137 |
| True | False | no | 3.106 | 1.5e-4 | 47 |
| False | True | no | 12.52 | 5.0e+3 (NaN cost) | 95 |
| False | False | no | 3.072 | 0.0 | 45 |

Three separate mechanisms were found.

1. **A termination bug on the penalty side.** Inherited penalties are stiff enough that the very
   first inner solve of the next step is already feasible. `_evaluate_al_convergence` then exits
   after one outer iteration, skipping the dual and penalty updates -- so both freeze for the rest
   of the run. Measured: lambda pinned at 193.5 and mu at 1e5 for the final twenty steps. The
   controller degenerates into open-loop replay of a stale multiplier set.
2. **Stale multipliers against fresh penalties.** `(reset_duals=False, reset_penalties=True)`
   carries lambda forward (compounding to 2.5e4) while resetting mu to 1.0. That is a linear
   multiplier term with no matching quadratic: the augmented objective is unbounded below in the
   directions the stale multipliers favour, which is the NaN cost in the table.
3. **A mask leak in the shift.** `ALConstraints.build` pads the control bounds at knot N - 1 with
   +-inf. `shift` drags that padding onto knot N - 2, so a *real* row could inherit a padded
   source row's zero penalty and be left unconstrained.

## Decision

Three fixes, each with its own test.

- **`_shifted_al` holds its own value rather than zeroing.** A real destination row whose source
  row is padding keeps its own `lam` and `mu` instead of taking the padding's. This is the same
  "hold the last real value" rule the rest of `WarmStart.shift` already follows. Zeroing, and the
  simpler `live = al.row_mask`, both leave a real row at zero penalty.
- **`_evaluate_al_convergence` requires one dual update before it may exit on feasibility.**
  Under `reset_penalties=False` only, `converged_violation` is additionally gated on
  `iter_num > 1`. This is a scoped divergence from the Altro.jl port, invisible under default
  options where the branch is never taken. Note that ADR 0007 has since put two *unscoped*
  divergences into the same function, so "invisible under default options" is a claim about this
  gate alone, not about `_evaluate_al_convergence` as a whole, and the cross-verification against
  the live Julia solver is per-formula and per-transition rather than a whole-history replay.
- **`(reset_duals=False, reset_penalties=True)` is rejected at the call site.** `AL.solve`,
  `Altro.solve` and `BoxQP.solve` raise `ValueError` rather than producing the NaN above.
  Carrying a multiplier without its penalty is not a configuration worth supporting.

## Consequences

- The one-iteration exit is gone: `reset_penalties=False` now takes 2 outer iterations per step.
- **`(True, False)` and `(False, False)` still do not recover.** The freeze was traded for a
  ratchet: mu now grows by `penalty_scaling = 10.0` every step from step 0 and saturates at
  `penalty_max = 1e8` by step 7 -- thirteen steps *before* the kick arrives. At 1e8 the true cost
  is eight orders of magnitude below the penalty term, so the solve is a feasibility projection
  with no cost gradient left to steer with. The `(False, False)` row's max violation of 0.0 is
  that, not success.
- `reset_duals=False` remains unusable in practice. The default `(True, True)` is the only
  supported receding-horizon configuration, and nothing in the repo relies on the others.
- ADR 0007's per-row penalty cap does not change this, and on this problem does not fire at all.
  Gated on the active set, the cap can only bind on a row that is actually in the augmented cost,
  and such a row has its residual driven towards zero, at which point
  $2 \cdot \text{max\_cost\_value} / c^2$ is no cap. Re-measured on the kicked cartpole with the cap
  disabled, the `(False, False)` run is bit-identical: final pole angle error 3.46 rad either way,
  against 0.16 rad on the supported default. The cap is the one mechanism in the codebase that can
  pull a carried `mu` *down*, so it looks like a candidate for option 2 below. It is not one. Option
  2 remains open and unimplemented.

## What is deliberately not decided

The ratchet needs a rule for how penalties carry *between* solves. Three candidates, none
implemented:

1. **Decay mu on shift** -- divide the carried penalty by `penalty_scaling`, so a step that needed
   the stiffness keeps it and a quiet step relaxes. Cheapest, but the decay rate is a free
   parameter with no principled value.
2. **Cap per-step growth** -- clamp the carried mu below `penalty_max`, e.g. at the value the
   previous step actually converged with. Preserves the within-solve schedule, only bounds what
   crosses the seam.
3. **Reset mu while keeping lam** -- the mirror of the rejected `(False, True)`, and equally
   ill-posed for the same reason. Recorded to be dismissed, not chosen.

These are not held back because they are risky, but because the design question underneath them is
unanswered: **what does carrying stiffness across a horizon step mean when the constraint set has
moved?** A penalty earned against knot k's constraint is being applied to knot k - 1's.

Note also that the usual objection does not apply here. Altro.jl has no receding-horizon driver,
so penalty carry between solves has no Julia counterpart to diverge from. Options 1 and 2 act in
`WarmStart.shift` or the MPC driver, outside `al_solve` -- port fidelity is not what blocks them.
