# 08 — Expansion engine, Euclidean path

**What to build:** The shared derivative engine that both the NLP transcription and, later, the
native solvers consume. A caller hands it a problem and a trajectory and gets back the stacked
first- and second-order expansions of the dynamics and the cost, plus the augmented Lagrangian
contributions folded into the same structure.

This module is cut before its first consumer exists, deliberately. Built later, the transcription
layer's derivative code would harden into a de facto engine with a solver-shaped interface, and
the native solver work would have to fight it.

Expansions are returned in error coordinates, with the attitude Jacobian applied inside. For
Euclidean models the error dimension equals the state dimension and that Jacobian is the
identity, which the compiler folds away — so this ticket builds the Euclidean path while
establishing the interface the manifold path will fill in.

**Blocked by:** 06 — Stacked objective and cost evaluation; 07 — Constraint catalog and fused
ConstraintList.

**Spec:** Section 11 (expansions), section 7 (error-state expansions), section 9 (cone
projections, used by the augmented Lagrangian term).

## Acceptance criteria

- [x] An expansion structure holds stacked dynamics Jacobians, cost gradients, and cost Hessian
      blocks, sized in the error dimension rather than the state dimension
- [x] Three composable pure functions produce it: dynamics expansion, cost expansion, and
      augmented Lagrangian expansion
- [x] The augmented Lagrangian function takes the multipliers and penalty and returns a new
      expansion with its contributions added into the existing gradient and Hessian fields, not
      into a separate structure
- [x] The augmented Lagrangian uses the dual-cone projection of the shifted constraint value,
      consistent with the cone module
- [x] All quantities are in error coordinates; no consumer applies the attitude Jacobian sandwich
      itself
- [x] Every expansion is verified against finite differences of the corresponding evaluation
      function
- [x] Cross-verification against Julia covers whichever expansion quantities have a direct Julia
      counterpart
