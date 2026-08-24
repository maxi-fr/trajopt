# 16 — Full benchmark suite and timing harness

**What to build:** The evidence that the library does what it was built for. All three benchmark
problems solve correctly and fast enough, with timing broken down finely enough to say where the
time actually goes.

The three problems are chosen to cover disjoint failure modes: cartpole swing-up is
underactuated with bounded actuation and state limits, quadrotor obstacle avoidance exercises
attitude tracking on the rotation group through keep-out zones, and the Dubins car is
nonholonomic with corridor constraints and a tracking objective.

The timing breakdown matters more than a single number. A closed-loop rate that misses its
deadline is only actionable if it is clear whether the time went into assembling sparse
structures, evaluating derivatives, or the solver itself.

**Blocked by:** 10 — Pure-CasADi baseline and end-to-end parity; 11 — Problem/MPCState split and
the zero-recompile loop; 13 — RigidBody, Quadrotor, and error-state expansions.

**Spec:** Section 15 (verification strategy, benchmarking and standing risks), section 14
(models).

## Acceptance criteria

- [x] All three benchmark problems are formulated and solve to optimality
- [x] The quadrotor problem navigates around spherical keep-out zones while tracking an attitude
      reference
- [x] The Dubins problem enforces corridor constraints alongside a tracking objective, and the
      corridor is active rather than merely satisfied: the tracking reference bulges laterally
      past it, so several knots sit on the wall carrying nonzero multipliers. The cartpole's
      cart position limit is tightened until it binds for the same reason
- [x] Each problem has a matching independent CasADi formulation and meets the state, control,
      objective, and dual parity tolerances. Dual parity is asserted block by block over the rows
      the two formulations share — the initial condition, the dynamics costates, and the path and
      terminal constraint rows. Box bounds are excluded: trajopt hands them to Ipopt as variable
      limits, whose multipliers arrive as `mu` rather than `mult_g`, while the CasADi baseline
      gives them general constraint rows, collapsed to a knot-invariant envelope and ordered by
      variable rather than by knot. Those rows are a different set, not a permutation of the same
      one
- [x] Timing is reported separately for transcription setup, per-iteration derivative evaluation,
      solver runtime, and closed-loop rate
- [x] Closed-loop measurement reports sustained frequency and latency jitter, not just a mean
- [x] Warm-start speedup over a receding horizon is quantified
- [x] Benchmarks run under the existing benchmark tooling and can be excluded from ordinary test
      runs
