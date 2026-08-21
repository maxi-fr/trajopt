# 03 — Trajectory storage and the model interface

**What to build:** The ability to define a dynamical system and hold a trajectory of it. A
caller defines a cartpole, evaluates its continuous dynamics and Jacobians at a point, and gets
answers matching Julia — and can store a horizon of states, controls, and times.

Trajectory storage is struct-of-arrays because batched evaluation over knot points is the whole
performance argument for the backend. The knot-point abstraction survives as a read-only view
for callers, never as the storage itself: a list of per-knot Python objects cannot be traced.

The model interface carries the manifold seam. Euclidean models declare only their state and
control dimensions; the error-state dimension and the state-difference and error-Jacobian
operations default so that no Euclidean model ever mentions the manifold. That default is what
lets the rigid-body work later be an override rather than a special case threaded through every
consumer.

**Blocked by:** 01 — Numerics foundation and cones in JAX.

**Spec:** Section 5 (trajectory storage), section 6 (dynamics types and the model interface),
section 1 (invariants).

## Acceptance criteria

- [ ] Trajectory storage holds states, controls, times, and step durations as stacked arrays,
      with control storage one entry shorter than state storage
- [ ] A knot-point view exposes state, control, time, duration, and terminal status without
      owning any storage
- [ ] Trajectory operations that would mutate instead return new trajectories: setting states,
      setting controls, shifting the initial time, and shifting forward one step for
      warm-starting
- [ ] Continuous, discrete, and discretized dynamics types exist, with the discretized form
      wrapping a continuous model and an integrator
- [ ] The error-state dimension defaults to the state dimension, the state difference defaults to
      subtraction, and the error Jacobian defaults to the identity
- [ ] A test confirms a Euclidean model can be defined without referencing any manifold concept
- [ ] Model parameters are traced values and dimensions are compile-time metadata, following the
      pytree split in the specification
- [ ] A cartpole model is implemented with parameters matched to RobotZoo
- [ ] Cross-verification covers the continuous dynamics and its state and control Jacobians at
      `1e-14` and `1e-12` respectively
