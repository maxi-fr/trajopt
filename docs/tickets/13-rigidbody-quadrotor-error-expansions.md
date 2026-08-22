# 13 — RigidBody, Quadrotor, and error-state expansions

**What to build:** The point where the two strands merge. A caller defines a quadrotor with a
quaternion in its state, rolls it out, and gets expansions in error coordinates that are
correctly dimensioned and match Julia — with the attitude Jacobian sandwich applied inside the
engine, invisible to every consumer.

This is where the twelve-dimensional error state meets the thirteen-dimensional state vector.
The attitude Jacobian maps error variations into state variations, and it is that direction —
not its transpose — that makes the expansion sandwich dimensionally consistent. Its rotation
block is exactly the kinematics matrix scaled by one half, already built and tested in the
rotations module.

The reason expansions must be in error coordinates rather than state coordinates becomes
concrete here: in state coordinates the quaternion's unit-norm direction is a null direction of
the Hessian, and a factorization will fail on it.

**One derivation is still outstanding and belongs to this ticket.** Ticket 02 settled the
operand ordering for the *error*, but not for the *attitude Jacobian*. Julia builds its rotation
block from `Rotations.∇differential`, documented as the Jacobian of `lmult(q) QuatMap(ϕ)` — a
right multiplicative perturbation, consistent with its `q_ref⁻¹ ⊗ q` error. The scaled
kinematics matrix used here is the Jacobian of a left perturbation, which bridges to a left
Hamilton perturbation. The two therefore differ in the same way the two error definitions do,
and the relation between them has to be derived before the attitude-Jacobian cross-test is
written, by the same standard ticket 02 set: derived, not fitted.

**Blocked by:** 08 — Expansion engine, Euclidean path; 12 — Rotations module.

**Spec:** Section 7 (attitude Jacobian, error-state expansions, model structure declaration,
geodesic quaternion cost), section 11 (expansions), section 14 (models).

## Acceptance criteria

- [x] A rigid-body model base exists with the state laid out as position, attitude, linear
      velocity, angular velocity, and with the error dimension one less than the state dimension
- [x] The attitude Jacobian is block diagonal with identity blocks for the Euclidean components
      and the scaled kinematics matrix for the attitude component
- [x] The attitude Jacobian maps error variations into state variations, and a shape assertion in
      the test suite pins that direction
- [x] The left-versus-right perturbation relation between the Python and Julia attitude Jacobians
      is derived symbolically and verified numerically on a non-degenerate pair, before the
      attitude-Jacobian cross-test asserts anything
- [x] A quadrotor model is implemented with every parameter matched to RobotZoo, including the
      motor mixing matrix
- [x] The quadrotor cross-test converts both the state vector and the Jacobians, since RobotZoo
      stores its quaternion in the opposite convention and ordering
- [x] The expansion engine applies the attitude Jacobian sandwich to dynamics, cost, and
      constraint expansions
- [x] Euclidean models still produce results identical to before this ticket, confirming the
      identity path is unchanged
- [x] The geodesic quaternion cost is implemented with its double-cover branch, and the
      quaternion attitude equality constraint is added to the catalog
- [x] Cross-verification covers the attitude Jacobian, the error state, the sandwiched dynamics
      expansion, and the geodesic cost including both branches of its subgradient, at `1e-12`
