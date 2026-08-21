# 05 — Remaining Euclidean benchmark models

**What to build:** The other two Euclidean benchmark systems, so the dynamics layer is verified
against more than one model before anything is built on top of it. A caller can simulate a
pendulum or a Dubins car with the same interface the cartpole uses, and both match Julia.

Parameter matching is the point of this ticket, not the dynamics equations. A model whose
gravity or length differs from RobotZoo's produces a cross-test that passes on its own terms
while verifying nothing, which is worse than having no cross-test at all.

**Blocked by:** 04 — Integrators and rollout.

**Spec:** Section 14 (models), section 15 (verification strategy).

## Acceptance criteria

- [ ] Pendulum and Dubins car models are implemented against the same model interface as the
      cartpole
- [ ] Every physical parameter is matched to the corresponding RobotZoo model, and the match is
      asserted in a test rather than assumed
- [ ] Cross-verification covers continuous dynamics, discrete steps, and Jacobians for both
      models across all three integrators
- [ ] The quadrotor is explicitly out of scope here and is noted as belonging to the rotations
      strand
