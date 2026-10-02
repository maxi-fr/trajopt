# Binary control homotopy with Ipopt

## Problem Statement

`trajopt` treats every control coordinate as continuous. A caller can bound a control to
`[0, 1]`, but cannot declare that it must finish near one of those endpoints. This prevents
mixed-integer optimal control problems with binary decisions at control Knot Points from being
solved through the existing `MPC` and Ipopt path.

The current Constraint List also holds only constraints evaluated at individual Knot Points.
It cannot express a linear restriction that couples decisions across the Horizon, such as a
limit on the sum of one control coordinate over all control Knot Points. A user should be able
to formulate that restriction without introducing a bookkeeping state into the dynamics.

## Solution

Let a Problem declare selected control coordinates binary. The declaration applies at every
nonterminal Knot Point. Problem construction adds their `[0, 1]` bounds. The coordinates remain
part of the ordinary control array and Primal Vector, so callers can use them in their own
models, Objectives, and constraints.

When Ipopt solves such a Problem, it runs a penalty term homotopy. Starting with a continuous
relaxation, it repeatedly solves the same constrained OCP while increasing the coefficient
`beta` of the additive Objective term `sum beta * a * (1 - a)` over all declared binary control
entries. Each solve starts from the previous Primal Vector and Multipliers. The sequence succeeds
only if the final continuous solve succeeds, its ordinary constraints satisfy the solver's
feasibility criterion, and every declared coordinate is within the configured binary tolerance
of 0 or 1.

Add linear Horizon constraints to the Constraint List. A caller supplies coefficients over the
Primal Vector and a Cone, allowing linear equalities or inequalities that couple Knot Points.
Ipopt includes these rows in its constraint bounds, values, Jacobian, Multiplier layout, and
constraint violation. This general mechanism allows a sum limit without making activation,
budgets, or particular physical dynamics part of the binary feature.

## Implementation Decisions

### Decision variables and bounds

- A binary decision is an existing control coordinate. There is no separate activation vector,
  no extra state, and no activation-specific dynamics adapter. A user's model receives the
  expanded control vector and decides which coordinates affect its dynamics.
- `Problem` accepts a structural declaration of binary control indices. It validates that each
  index is unique and within the model's control dimension. The same indices apply at every
  control Knot Point of the Horizon; the terminal Knot Point has no control.
- Problem construction adds `[0, 1]` bounds for those coordinates to the built constraint
  bounds, without mutating a caller-owned Constraint List. Existing bounds still intersect with
  the binary bounds. Contradictory bounds fail during Problem construction.
- `Trajectory.U`, the Primal Vector, `WarmStart`, and the existing solver result shape retain
  their current layouts. A binary coordinate appears wherever an ordinary control coordinate
  appears. Code applying physical controls to a plant selects the physical coordinates.

### Objective and continuation

- The homotopy term is added to the caller's Objective during an Ipopt solve. It does not
  replace any stage-cost term. In particular, run-time reference retargeting must not erase
  caller-supplied linear costs on binary coordinates.
- The penalty coefficient is solve data, not a new structural Problem. Changing it between
  homotopy passes must not rebuild the Problem or force a fresh JAX compilation for each value.
  Ipopt's objective value, gradient, and exact Lagrangian Hessian must all see the same
  coefficient. For each binary coordinate, the added gradient is `beta * (1 - 2a)` and the
  added diagonal Hessian entry is `-2 * beta`, scaled by Ipopt's objective factor.
- The Ipopt configuration owns the initial positive coefficient, growth factor, maximum number
  of passes, and binary tolerance. The first pass solves the unpenalized relaxation. Later
  passes increase the coefficient until the tolerance is met or the pass limit is reached.
- The Primal Vector and compatible Multipliers from one pass initialize the next. A failed
  continuous solve ends the sequence; it is not reported as binary convergence. Values are
  never rounded after the solve, since rounding could violate dynamics or other constraints.
- Ipopt remains the existing Backend and keeps the existing Solver interface. `MPC.solve()`
  requires no new call pattern. Backends and Native Solvers without binary support must reject a
  Problem with a binary declaration rather than silently solving its relaxation.

### Horizon constraints

- The Constraint List gains a general linear constraint whose coefficients act on the complete
  Primal Vector. Its row sense uses the existing Cone vocabulary. It is distinct from the
  current `LinearConstraint`, which acts at one Knot Point.
- Horizon rows follow the existing per-knot rows in a documented canonical order. Their
  Multipliers occupy corresponding trailing entries in `WarmStart.lam` and solver results.
- The Ipopt transcription appends these rows to constraint values, bounds, and Jacobian
  structure. Linear Horizon constraints have zero constraint Hessian, so the existing exact
  Lagrangian Hessian needs no cross-Knot blocks for them.
- Constraint violation includes Horizon rows. A Backend or Native Solver that does not support
  them must reject the Problem explicitly. This specification requires their solution through
  Ipopt, not through every existing solver.

### Result meaning

- `success` means both an accepted Ipopt termination and satisfaction of the binary tolerance.
  If the homotopy exhausts its pass limit with fractional binary controls, the result is a
  failure even when the final continuous NLP converged.
- `cost` is the caller's original, retargeted Objective at the returned Trajectory. The
  homotopy penalty and final coefficient are available separately in the result's diagnostic
  information, together with the maximum binary distance to `{0, 1}` and the pass count.
- `constraint_violation` keeps its existing meaning for dynamics, bounds, and registered
  constraints, including Horizon constraints. Binary distance is a separate diagnostic; it is
  not hidden inside `constraint_violation`.
- This method seeks a binary-feasible local result. It provides no global mixed-integer
  optimality certificate.

## Testing Decisions

Tests exercise caller-visible behavior at the `Problem`, Ipopt, and `MPC` interfaces. They do
not assert on private continuation helpers, callback call counts, or intermediate array layout.
The main acceptance test solves a small OCP with a binary control coordinate through `MPC`,
checks ordinary feasibility and binary distance, and verifies that a linear Horizon constraint
changes the accepted solution as expected. The test model is intentionally simple; it does not
encode the user's particular current-activation OCP.

- At the Problem seam, a binary declaration produces `[0, 1]` control bounds at every control
  Knot Point. Invalid indices, repeated indices, and contradictory pre-existing bounds fail
  before solving. A Problem without a declaration retains its current bounds and behavior.
- At the constraint seam, a linear Horizon equality or inequality evaluates over the intended
  Primal Vector entries, appears in Ipopt's constraint values and Jacobian, and contributes to
  measured constraint violation. A whole-Horizon sum is the representative case.
- At the Ipopt seam, a fractional optimum of the relaxed problem becomes binary within
  tolerance under homotopy. The returned cost equals the original Objective value, while
  diagnostics report the final coefficient and binary distance. A pass-limit case reports
  failure without rounding its Primal Vector.
- At the MPC seam, successive solves and `shift()` retain the binary control coordinates and
  warm start the next Horizon. A moving reference does not remove an independently supplied
  linear cost on a binary coordinate.
- An unsupported solver rejects Problems containing binary declarations or Horizon constraints
  explicitly. Ordinary continuous Problems keep their existing solve behavior.

Existing transcription tests establish prior art for objective derivatives, sparse constraint
rows, and the Primal Vector layout. Existing Problem and MPC tests establish prior art for
boundary retargeting, warm starts, and Horizon shifts. Existing Ipopt tests cover the Backend
adapter.

## Out of Scope

- A dedicated mixed-integer Backend, branch and bound, and global optimality certificates.
- Integer values other than binary 0 and 1, binary state coordinates, and binary declarations
  that vary by Knot Point.
- An activation-specific model, current-limit helper, activation charge, or budget policy.
  Callers formulate those using their own dynamics, Objective, per-knot constraints, and linear
  Horizon constraints.
- Nonlinear constraints that couple multiple Knot Points. The new Horizon constraint is linear.
- Solving binary Problems or linear Horizon constraints with Backends or Native Solvers other
  than Ipopt.

## Further Notes

The binary penalty is nonconvex because `a * (1 - a)` is concave over `[0, 1]`. Different warm
starts can lead Ipopt to different local results. The acceptance criterion is feasibility and
binary tolerance, not a claim that the returned trajectory is globally best.

The structural split follows the accepted Program design: `Problem` declares the control domain
and constraints, `BoundaryConditions` carries per-solve targets, and Ipopt owns its continuation
state. The result and unsupported-solver behavior follow the accepted requirement that a
Backend state which problem it actually solved.
