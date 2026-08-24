# 33 — ALTRO driver and end-to-end scenarios

**What to build:** `problem.solve(state, solver=ALTRO())` runs both phases and hits machine-
precision constraint satisfaction. This is the ticket that makes the whole series demonstrable:
the three scenario solves from reference §8.2 — LQR, bound-constrained cartpole, and a swing-up
with a goal constraint — all match Julia's `Altro.ALTROSolver` on final cost, trajectory, status,
and iteration count, and the option-parity suite shows the two implementations tracking each other
as you vary tolerances and penalties.

**Blocked by:** 32.

## Architecture

`src/trajopt/solvers/altro.py`. A driver, not an algorithm — everything it needs exists by now.

**Phase logic, from reference §6 with two corrections.** The shape is: shortcut to plain iLQR when
there are no constraints; if `projected_newton` is on, loosen the AL phase's constraint tolerance
to `projected_newton_tolerance` (1e-3) so AL stops early and leaves the polish to PN; run AL; then
run PN if the violation still exceeds the real tolerance.

Correction one, finding I: reference §6's backup check omits its guard. Altro requires
`status <= SOLVE_SUCCEEDED` before upgrading the status to `SOLVE_SUCCEEDED`, so a
`MAX_ITERATIONS_OUTER` exit is never upgraded even when PN drove the violation under tolerance.
That is arguably wrong, but it is the behaviour parity tests will hold you to. Reproduce it and
note it.

Correction two, also finding I: `c_max` is read from the AL stats **cache** when
`iterations > 1`, and recomputed from the constraints only otherwise. The two can differ, because
the cached value is from before the last inner solve.

**The `projected_newton_tolerance < 0` branch needs ticket 29's `kickout_max_penalty`.** When the
tolerance is negative, Altro sets the AL constraint tolerance to zero and turns on
`kickout_max_penalty` — the branch that throws upstream (finding B). Ticket 29 implemented it as
intended; this is where it gets exercised.

**Tolerance mutation becomes a threaded value.** Altro mutates `opts.constraint_tolerance` and
restores it. Compute both tolerances up front and pass each phase the one it needs.

**Status decisions are ordinal.** `status <= SOLVE_SUCCEEDED` and `status == MAX_ITERATIONS_OUTER`
gate whether PN runs at all — finding C. With `TerminationStatus` as an ordered `IntEnum` these are
plain comparisons on the carried int.

## End-to-end scenarios

Reference §8.2's list, layered on the phase-level parity tests from earlier tickets:

- **LQR.** Linear dynamics, quadratic cost, no constraints — takes the unconstrained shortcut,
  converges in one iteration, gains match analytic LQR on both sides.
- **Box.** Cartpole with `ControlBound` and a goal. Compare cost, trajectory, λ, μ, and iteration
  count against Julia. Also run it through ticket 30's box-QP path and record the comparison.
- **ALTRO.** Swing-up with `GoalConstraint`. Both sides reach
  `max_violation < constraint_tolerance` and match terminal cost.
- **Options parity.** Same kwargs on both sides — `penalty_initial`, `penalty_scaling`,
  `cost_tolerance`, `gradient_tolerance`, `constraint_tolerance`, the `line_search_*` family, the
  `bp_reg_*` family — comparing trajectories each time. This is where a divergence that a single
  default-configuration test hides will show up.

Reference §8.2's closing note about `use_static = Val(false)` not being portable is right: use
`StaticReturn` consistently on the Julia side and the same pendulum, cartpole, and quadrotor models
the existing cross tests use.

**Existing adapter tests should now run against `ALTRO()`.** The whole point of the protocol
conformance from ticket 27 is that a native solver substitutes for `Ipopt()` without touching the
call site. Prove it on at least one existing test file rather than asserting it in a docstring.

## Acceptance criteria

- [ ] `problem.solve(state, solver=ALTRO())` runs both phases and reaches `max_violation < constraint_tolerance` on a constrained cartpole.
- [ ] An unconstrained problem takes the plain-iLQR shortcut without constructing any AL or PN state.
- [ ] The AL phase's constraint tolerance is loosened to `projected_newton_tolerance` when PN is enabled, and both phases receive their tolerance as a value with no option mutation.
- [ ] The `projected_newton_tolerance < 0` branch works end to end via `kickout_max_penalty`.
- [ ] The backup check reproduces its `status <= SOLVE_SUCCEEDED` guard: a test that exits `MAX_ITERATIONS_OUTER` with a converged violation asserts the status is **not** upgraded.
- [ ] `c_max` comes from the AL stats cache when `iterations > 1`, matching upstream.
- [ ] All four §8.2 end-to-end scenarios pass against Julia to 1e-8, including the option-parity sweep across at least eight distinct option settings.
- [ ] At least one existing solver-adapter test file runs green against `ALTRO()` with only the solver object changed.
- [ ] A whole `ALTRO()` solve runs under `jax.jit` and under `jax.vmap` over a batch of initial states.
- [ ] pre-commit hooks pass
