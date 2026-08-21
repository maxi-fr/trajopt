# 04 — Integrators and rollout

**What to build:** The ability to step a model forward in time and simulate a whole horizon. A
caller hands a cartpole an initial state and a sequence of controls and gets back a trajectory
whose every state and every discrete Jacobian matches Julia.

This completes what ticket 03 started: 03 makes a model evaluable at a point, this makes it
evaluable over time. It is also where the first performance-shaped decision becomes real —
forward simulation is a compiled scan, never a Python loop over knot points, because a Python
loop over the horizon is the interpreter-overhead failure mode the specification calls out.

**Blocked by:** 03 — Trajectory storage and the model interface.

**Spec:** Section 6 (numerical integrators, rollout).

## Acceptance criteria

- [x] Explicit RK4, Euler, and implicit midpoint integrators are implemented
- [x] Each integrator composes with the discretized dynamics type from ticket 03 without
      special-casing the model
- [x] Forward simulation uses a compiled scan rather than a Python loop over knot points, and a
      test or benchmark demonstrates the horizon length does not multiply interpreter overhead
- [x] Rollout sets the first state from the initial condition and propagates the remainder from
      the stored controls
- [x] Integration accuracy is verified against an analytically integrable system, so an
      integrator bug cannot hide behind a matching-but-wrong cross-test
- [x] Cross-verification covers the discrete step and the discrete state and control Jacobians
      for all three integrators, at `1e-14` for steps and `1e-12` for Jacobians
