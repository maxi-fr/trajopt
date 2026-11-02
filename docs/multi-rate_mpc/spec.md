# Multirate MPC and blocked controls

## Problem Statement

A downstream neurostimulation application needs to distinguish the native prediction interval,
the interval between optimization attempts, and the interval over which planned controls remain
constant. Its Waveform Predictor advances every 10 ms. The controller must continue absorbing
measurements every 10 ms while solving every 60 ms. It must support both executing six distinct
planned controls between solves and executing one control held for six native intervals.

The package currently exposes a one-Knot Point horizon shift whose optional duration changes
only the clock. Supplying 60 ms to a Problem with 10 ms native intervals therefore shifts the
warm start and reference by one Knot Point while moving time by six intervals. This is an
incorrect warm start for the next solve.

The existing shooting Transcriptions also assume independent controls at every nonterminal
Knot Point. Repeating controls after an unrestricted optimization does not solve the blocked
Problem. Adding equality rows between native controls enforces blocking but retains the large
NLP and introduces avoidable redundancy. The required representation optimizes independent
block controls directly while preserving native dynamics, constraints, Objective evaluation,
and public Trajectories.

Binary control coordinates must participate in blocking. The existing IPOPT Backend uses
penalty continuation for binary feasibility in multiple shooting. Single shooting currently
lacks that continuation and linear Horizon constraint support. Both shooting formulations must
support continuous and binary blocked Problems under the same public contract.

Two existing issues affect this work. Single shooting returns reduced Multipliers where the MPC
driver expects canonical full-grid Multipliers; the downstream application drops incompatible
vectors with a wrapper. The IPOPT shooting paths also construct fresh external NLP handles
inside repeated solves despite the Program's role as the persistent compiled and allocated form
of a Problem. Multirate support must resolve these issues rather than transfer them to callers.

## Solution

Provide count-based horizon advancement and uniform control blocking through the existing
Problem, Program, MPC, and IPOPT shooting interfaces.

The controller remains responsible for deciding when to solve, absorbing measurements, copying
and executing a command prefix, and handling failures. The package advances a warm start by an
explicit number of consumed native intervals, accepts the newly measured state, and solves the
same structural Problem through the same Program.

Both single and multiple shooting optimize one independent control vector per uniform block.
Every control coordinate, including declared binary coordinates and supported auxiliary
coordinates, follows the same blocking pattern. Both formulations return native-grid states,
controls, time stamps, canonical Primal Vectors, and canonical-sized Multiplier vectors.

The default block size of one retains the existing admissible control sequences. This work
deliberately replaces the ambiguous duration-based shift interface rather than retaining a
compatibility alias. Unsupported Backends and Native Solvers reject non-unit blocking when
paired with the Problem.

## Implementation Decisions

### 1. Ownership and module responsibilities

- Problem owns immutable blocking structure alongside the model, Objective, constraints,
  Horizon, time grid, and binary declarations.
- Boundary Conditions continue to carry only per-solve array data: measured initial state,
  initial time, reference window, and terminal goal. Block size and binary declarations remain
  structural rather than becoming Boundary Conditions.
- Program owns the shooting layout, retained constraint-row selection, compiled numerical
  functions, and live IPOPT handle for its structural Problem and Backend configuration.
- MPC owns the current Boundary Conditions, canonical warm start, reference cursor, and latest
  result. It does not own a replanning scheduler or a queue of commands for the plant.
- The shooting Transcriptions share block expansion, native-guess reduction, bound intersection,
  eligible input-row deduplication, and continuation behavior. Their state treatment remains
  distinct: single shooting uses a Rollout; multiple shooting optimizes native states.
- Canonical Trajectory evaluation remains available independently of a Backend's reduced NLP
  representation. Consumers evaluating Costs or feasibility never need to construct block
  variables.
- Changes follow the accepted Program and cost-target design, solver-result honesty, full-state
  NLP coordinate convention, output-function design, and existing augmented-Lagrangian shift
  rules. This work does not change the Native Solvers' error-state coordinates or penalty-carry
  policy.

### 2. Dimensions and terminology

- Let N be the Horizon in native Knot Points, H = N - 1 the number of native prediction
  transitions, n the model state dimension, m its control dimension, b the block size, and
  B = H / b the number of independent control vectors.
- Native states have shape (N, n). Native controls have shape (H, m). Independent block controls
  have shape (B, m).
- The canonical Primal Vector retains its existing interleaved native layout and length
  N *n + H* m. Canonical initial-state, Defect, per-Knot Point, terminal, and trailing Horizon
  row ordering remains unchanged.
- A Backend's internal NLP vector is distinguished from the canonical Primal Vector. Single
  shooting has B *m internal variables. Multiple shooting has N* n + B * m internal variables.
- The retained NLP constraint count is distinct from the canonical constraint count. Single
  shooting eliminates the initial-state and Defect equations; both Transcriptions may collapse
  eligible repeated input rows. Dimension helpers and callback descriptions must state which
  representation they describe.
- For H = 102, b = 6, and m = 3, there are 51 independent scalar controls rather than 306.
  Both formulations still return 102 native control vectors and 103 native state vectors.

### 3. Problem construction and supported combinations

- Add structural control_block_steps to Problem, defaulting to one.
- Accept positive integer block sizes, excluding booleans and fractional values. Require at
  least one native transition and H divisible by b. Invalid declarations fail during Problem
  construction rather than being truncated, rounded, or partially blocked.
- Blocking counts native transitions. It does not change N, model dimensions, the structural
  time grid, integration, or the meaning of the model's duration argument.
- Preserve blocking through retargeting, epigraph lifting, and other package-owned operations
  that return a supported transformed Problem. A transformation that cannot preserve the
  declaration must reject it explicitly.
- Binary control indices retain their existing coordinate meaning and validation. Their
  native bounds intersect with [0, 1] before block bounds are calculated.
- Multiple shooting through Ipopt supports continuous and binary blocking, native state bounds,
  existing supported Knot Point constraints, and linear Horizon constraints.
- Single shooting through SingleShooting with Ipopt supports continuous and binary blocking,
  existing supported Knot Point constraints, and linear Horizon constraints. It continues to
  reject state box bounds, including terminal state box bounds. Existing general state-dependent
  constraint rows remain valid because states are differentiable functions of block controls.
- Other Backends and Native Solvers reject b greater than one explicitly. Their existing b = 1
  behavior remains intact; count-based advancement alone does not imply blocked-solve support.
- Validate the Problem/Backend combination when constructing Program, and therefore when
  constructing MPC, before compilation or optimization. Direct invocation of a Backend must not
  bypass the same validation. Support is determined by an explicit supported pairing rather
  than inferred merely from the presence of a solve method.
- Recognize the supported SingleShooting/Ipopt combination for binary and Horizon constraints;
  the current top-level Ipopt-only type check must not reject this valid pairing.

### 4. Count-based horizon advancement

- Replace MPC.shift with MPC.advance(steps=1). Remove the duration argument and the old method;
  update package tests, examples, and documentation to the count-based interface. No deprecated
  alias or duration-to-count inference is required.
- A count is an integer excluding booleans. Zero is a no-op that leaves the warm start, clock,
  reference, pending reference push, and latest result unchanged. Reject negative, fractional,
  and greater-than-H counts before mutating driver state. A consumed execution prefix cannot
  exceed the plan's available native controls.
- Advance the warm-start states and controls by the requested count, holding their final native
  entries to fill the tail. Apply the existing one-step canonical Multiplier and augmented-
  Lagrangian rules repeatedly, including terminal-row resets, row-count mismatch handling,
  trailing Horizon Multipliers, and padding masks.
- Advance time by the sum of the first steps native durations in the structural Problem grid.
  For a uniform grid this equals steps times native dt. The time grid itself remains structural
  and unchanged, consistent with the existing Program design. On a nonuniform grid, the count
  refers to the prefix of the current plan; repeated separate one-count calls are not a claim
  that the structural duration array has been shifted.
- Advance reference states and controls by the same count using existing endpoint hold
  semantics. A pending pushed point is appended during the first internal one-count reference
  shift and consumed once; subsequent fills hold the new endpoint. No new reference sequence
  is inferred from a single push. A zero-count operation does not consume a push.
- Keep the separately specified terminal goal unchanged. Continue to support wholesale
  replacement of a reference window through the existing reference interface.
- The shifted predicted initial state is only a warm-start value. MPC.measure must replace both
  the measured initial Boundary Condition and the initial state in the canonical warm start
  before the next solve. Its supplied timestamp remains authoritative.
- Advancing does not solve, evaluate a planned Rollout, reevaluate the Objective, or modify the
  previously returned result. It preserves the Problem, Program, handles, and compiled kernels.
- Canonical primal and reference advancement accepts counts that are not block-aligned. Such a
  shifted control guess may cease to be blockwise constant; the next solve applies the specified
  native-guess reduction. When the count is a multiple of b, block alignment is preserved and
  endpoint fill supplies a constant terminal block.
- Count-based advancement also remains usable with existing b = 1 Native Solvers. It inherits
  their current augmented-Lagrangian carry limitations without changing them.

### 5. Native-plan reduction and expansion

- Expand each independent control vector into b consecutive native controls. Native control
  j uses independent vector floor(j / b), for j from zero through H - 1.
- Accept existing canonical warm starts and native control guesses. Reduce a guess by taking
  the first native control in each block, including its binary and auxiliary coordinates.
  Do not average, round, or choose a control based on its magnitude.
- In single shooting, discard guessed states for the numerical solve and recover states by
  native Rollout from the measured initial state and expanded controls.
- In multiple shooting, retain the native state guess. Replacing its controls with reduced and
  expanded controls may introduce Defects in the initial iterate; feasibility is the Backend's
  job, as with other multiple-shooting warm starts.
- Successful blocked plans expand to exactly equal controls within every block. Failed results
  also use the same expansion, while preserving the actual returned iterate and failure status.
- Expand controls and reconstruct canonical Z before publishing results. MPC.states,
  MPC.controls, warm-start access, and Trajectory evaluation continue to expose the native grid.
- Blocking applies to the full model control vector. Physical-coordinate selection, reduced
  electrode coordinates, and application inference adapters remain caller concerns. Existing
  supported package transformations must preserve their mathematical control bounds and Costs.

### 6. Objective and dynamics semantics

- Evaluate the original Objective on native states, expanded controls, and native time stamps.
  Keep every native Stage Cost and the original terminal Cost. Preserve output-based Costs,
  supported stacked parameters, independent linear control costs, and run-time retargeting.
- Blocking changes admissible control sequences, not Cost units. Do not divide or multiply
  application weights by b or by the replanning interval.
- Each native transition still calls the original learned or integrated dynamics with its own
  native duration and time. Blocking never substitutes a longer learned transition.
- Multiple shooting retains the initial-state equation and every native Defect. Defect j uses
  native states j and j + 1 and the independent control for floor(j / b).
- Single shooting evaluates all H transitions from the measured state. State-dependent Costs,
  output maps, and constraints see the resulting native states rather than a block-boundary
  subsampling.
- General nonuniform structural durations retain their existing meaning. Uniform blocking is
  uniform in count; it need not imply equal physical duration per block on a nonuniform grid.

### 7. Native constraints, deduplication, and block bounds

- State-dependent, state/control-dependent, and time-dependent rows retain every applicable
  native evaluation. Terminal constraints retain their native terminal evaluation and goal.
- Collapse repeated input-only rows only when their equivalence is structurally established:
  the same input expression, coefficients, right-hand side, Cone, coordinate mapping, and no
  time dependence. Retain distinct constraints and constraints with different active ranges.
- Shared object identity or known time-independent constraint structure may establish equality;
  approximate numerical equality and sampled Jacobians must not establish it. Be conservative
  for arbitrary custom constraints.
- Existing LinearConstraint currently reports control usage but does not explicitly identify
  control-only dependence through uses_state. Establish a sound input-dependence classification
  for built-in linear constraints and supported wrappers. Custom constraints default to native
  evaluation unless their dependency and invariance contract proves deduplication valid.
- For every eligible repeated constraint within a block, keep its first applicable native
  occurrence, even when that occurrence is not at the block's first Knot Point. Remove only
  equivalent later occurrences within that same block. Deduplicate complete constraint groups
  so Cone semantics are preserved; general scalar-row algebraic rank reduction is unnecessary.
- In particular, one Kirchhoff equality registered at all native control Knot Points contributes
  one equality per independent block, avoiding b identical equalities on the same control.
- For each control coordinate, a block's lower bound is the maximum native lower bound in that
  block, and its upper bound is the minimum native upper bound. This includes binary bounds.
  Reject a lower bound exceeding the upper bound before optimization. Equal endpoints are valid
  fixed controls. Preserve infinite bounds.
- A binary block whose intersected interval contains neither zero nor one is structurally
  infeasible for the declared binary domain and must fail validation, even if the continuous
  interval is nonempty. Never relax the binary declaration to accept such a block.
- Preserve state bounds in multiple shooting at every native state variable. Preserve supported
  transformed physical-control bounds as mathematical constraints; blocking must not substitute
  bounds on reduced coordinates for a caller's physical bounds.
- Unsupported constraint Cones must be rejected where the IPOPT representation cannot express
  their existing semantics. This work does not reinterpret a general Cone as scalar equality.

### 8. Linear Horizon constraints

- Both shooting formulations support the existing linear Horizon constraint interface over the
  canonical native Primal Vector, including selected canonical indices and full-vector forms.
- Evaluate these rows after canonical expansion. In multiple shooting they remain affine in
  the internal states and block controls. In single shooting, rows involving state coordinates
  become generally nonlinear after Rollout and need the corresponding Jacobian and Hessian.
- Do not apply the multiple-shooting assumption of zero Horizon constraint Hessian to single
  shooting when a row reads a rolled-out state.
- Repeated canonical control indices that map to one independent block variable accumulate
  their coefficients. Keep the original right-hand side, Cone, and canonical indexing.
- An activation budget remains a sum over native intervals. One six-interval active block
  contributes six units to a native activation count. A budget that is not a multiple of b is
  accepted as written; the package does not round the budget or reinterpret it as block count.
- Keep Horizon rows in registration order after the retained Knot Point rows for each NLP, and
  in their existing trailing position in canonical results and feasibility evaluation.

### 9. Derivatives and sparse multiple shooting

- All numerical callbacks differentiate with respect to the internal NLP vector, not a
  full native control vector mislabeled with reduced dimensions.
- For the linear control expansion S, native-control gradients reduce by multiplication with
  the transpose of S, and exact control Hessians transform by S transpose times the native
  Hessian times S. Mixed state/control blocks transform consistently.
- Multiple-shooting Defect and native path rows reference the shared block-control columns.
  Construct their sparse structure for this layout. If native contributions share a sparse
  entry, sum them rather than returning duplicate entries as distinct nonzeros.
- Transform exact Lagrangian Hessians, including shared control/control entries, state/control
  entries, objective factor, retained constraint Multipliers, and binary penalty curvature.
  Sparse multiple shooting remains sparse; converting its whole Hessian to dense solely for
  blocking is outside the intended implementation.
- Single shooting may retain its existing dense Jacobian and exact dense Hessian modes, now
  sized for B * m controls and all retained rows, including Horizon rows.
- Retain limited-memory Hessian support. Both exact and limited-memory modes must solve the
  same blocked mathematical Problem and use the same continuation coefficient.
- Dimension queries, callback bounds, sparsity, initial guesses, results, and violation reporting
  must derive from one consistent layout. Native evaluation helpers must remain clearly distinct
  from callbacks accepting reduced vectors.

### 10. Binary continuation in both shooting formulations

- Reuse the existing Ipopt configuration for the initial coefficient, growth factor, pass limit,
  and binary tolerance. SingleShooting reads these settings from its supplied Ipopt instance.
  No second independent continuation configuration is introduced.
- Share the continuation loop across shooting formulations. Start each new solve with an
  unpenalized continuous relaxation, then increase the coefficient according to the existing
  schedule. Carry the internal primal iterate and compatible internal Multipliers between
  passes. A failed continuous pass terminates the sequence.
- Apply the binary penalty over the expanded native binary controls. For a uniform block of b
  equal binary values a, its contribution is b times beta times a times (1 - a). The reduced
  gradient contribution is b times beta times (1 - 2a), and its Hessian diagonal contribution
  is minus two times b times beta, scaled by the objective factor.
- This preserves the existing native-grid penalty meaning and avoids an implicit change to
  continuation strength when block size changes. The penalty is additional numerical solve
  machinery, not part of the caller's reported Objective.
- Success requires accepted IPOPT termination, completion of the required continuation passes,
  all declared binary coordinates within binary tolerance of zero or one, and ordinary expanded
  Problem feasibility within the configured feasibility tolerance. A fractional result at the
  pass limit is failure even when its continuous NLP status is acceptable.
- Never round the iterate to manufacture integrality or feasibility. Binary distance remains a
  separate diagnostic from ordinary constraint violation.
- Report the original retargeted Objective as cost, excluding the continuation penalty. Preserve
  diagnostics for final coefficient, penalty, pass count, and maximum binary distance. Report
  the native-weighted penalty consistently across both shooting formulations.
- This is binary-feasible local optimization through continuation. There is no global
  mixed-integer optimality certificate and no requirement that arbitrary feasible Problems
  converge from every warm start.

### 11. Canonical results and Multiplier policy

- Every result exposes native Trajectory arrays, canonical Z, and canonical-sized lam and mu.
  Never publish reduced NLP vectors through these canonical fields. Raw backend arrays may
  remain explicitly named in backend diagnostics.
- Canonical lam includes the initial-state pin, all native Defects and user rows, terminal rows,
  and trailing Horizon rows. Canonical mu has one signed bound Multiplier per canonical Primal
  Vector entry, using the existing upper-minus-lower convention.
- For b = 1 multiple shooting, retain existing canonical Multiplier behavior and warm starts.
- For b = 1 single shooting, place retained user and Horizon Multipliers into their canonical
  rows, and place control-bound Multipliers into native control positions. State-bound entries
  are zero because this Transcription rejects state bounds. Recover eliminated initial-state
  and Defect Multipliers from state stationarity by a backward adjoint calculation using the
  original Objective, native dynamics, and user/Horizon constraint terms, with the existing
  Defect sign convention. Test the resulting canonical stationarity residual on a deterministic
  smooth Problem. Do not expose control-only vectors as canonical Multipliers.
- A b = 1 single-shooting warm start may recover internal user/Horizon and control-bound
  Multipliers from the canonical representation. Eliminated canonical equation Multipliers are
  not supplied as additional IPOPT rows.
- For b greater than one, deliberately reset exported canonical lam and mu to zero for both
  Transcriptions. This avoids inventing a nonunique mapping for collapsed rows and intersected
  block bounds. The zero arrays replace previous driver Multipliers, including after failure;
  an empty vector must not silently preserve stale canonical values.
- Blocked IPOPT solves do not consume canonical Multipliers from preceding MPC solves. Internal
  reduced Multipliers remain valid between continuation passes of the same solve and can be
  reused there. Retaining the external handle does not implicitly retain these across solves.
- Validate nonempty result Multiplier dimensions at the driver seam so malformed custom results
  fail with a clear error before any later shift. Existing intentional absence of Multipliers
  from unrelated supported result types may retain its existing semantics; these IPOPT paths
  always return explicit canonical-sized vectors.
- Canonical augmented-Lagrangian state is absent/reset for these IPOPT solves. Native Solver
  advancement continues to follow the existing augmented-Lagrangian mask and fill rules.
- The downstream canonicalization wrapper becomes unnecessary once it adopts this contract.
  No compatibility wrapper is added inside the package.

### 12. Feasibility and failure reporting

- Compute ordinary constraint violation from the expanded native Trajectory and canonical
  Primal Vector, including state/control bounds, initial-state pin, native Defects, native
  registered rows, terminal constraints, and linear Horizon constraints.
- Inequality violation is distance outside the permitted Cone or interval. A satisfied negative
  residual for a less-than-or-equal row contributes zero; taking its absolute value is incorrect.
  Correct the existing single-shooting metric as part of this work.
- Verify every original native row even when the NLP deduplicates equivalent rows. Verify native
  bounds even when the NLP uses intersected block bounds. Include a check that published controls
  satisfy the declared block structure.
- Failure preserves the actual expanded iterate, original Objective, ordinary violation,
  integrality diagnostics, and native backend status. It never substitutes a zero-current plan
  or reports a fallback plan as an optimized result.
- MPC may keep the failed primal iterate as its next warm-start guess under its existing
  behavior. Executing it, invalidating a queued prefix, and choosing a fallback are controller
  policies outside this package.

### 13. Program lifetime and compiled work

- Build and retain one external IPOPT NLP handle per Program for its fixed shooting layout and
  Backend configuration. Reuse it across continuation passes and subsequent solves.
- Update measured state, initial time, reference arrays, terminal target, and continuation
  coefficient as numerical data. They must not reconstruct the Problem, change the Program's
  structural identity, or specialize compiled kernels to each new numerical value.
- Keep the compiled callback surface stable, including optional target/reference structure
  established at construction. A genuinely different structural Problem or Backend option set
  requires a new Program; mutating one in place is unsupported.
- Initialize continuation and per-solve diagnostics afresh on each solve, even while reusing
  numerical functions and handles. A prior success, failure, final coefficient, or iteration
  counter must not leak into the next attempt.
- Advancing, measuring, changing supported run-time targets, and changing beta must cause no
  additional handle construction or compilation after the initialized numerical paths are warm.
  Track handle allocation and actual tracing/compilation separately; a stable Python function
  object alone is insufficient evidence of no retracing.
- No runtime claim requires IPOPT to retain a factorization between solves. Handle reuse and
  avoiding repeated compilation are the specified guarantees.

### 14. Migration and downstream integration contract

- Prefer clean breaking changes. Migrate package-owned uses of shift to advance and document the
  new count semantics. Existing external callers must update explicitly.
- Keep native evaluation interfaces and reduced NLP callback interfaces distinguishable. If a
  helper changes representation, update its documented contract and package callers together;
  avoid shape guessing and compatibility dispatch based on vector length.
- The downstream application subclasses Problem and reconstructs Problems in its builders and
  wrappers. It must forward the new structural field explicitly where it constructs a Problem.
- The application also uses Transcription helpers directly in candidate and binary-homotopy
  implementations. These consumers must migrate to the reduced contract or reject blocking.
  This specification does not promise compatibility for private helper imports.
- At a replanning boundary, the caller advances by the consumed native prefix, measures the
  fully absorbed state, solves, and copies the unshifted plan for playback. Skip advancement
  before the first solve. Native ticks between boundaries require no package solve or planned
  Rollout. An r = b prefix preserves block alignment; r and the service schedule remain caller
  settings rather than additional Problem fields.

## Testing Decisions

### Test philosophy and confirmed seams

The user confirmed four seams: MPC execution, Program.solve through both shooting formulations,
exposed Transcription callbacks, and Program lifetime. Tests use the highest seam that can
demonstrate each contract. Small deterministic Problems establish numerical behavior; tests do
not reproduce the neurostimulation model or require an external application checkout.

Prior art includes the existing Problem/MPC reference and warm-start tests, single-versus-multiple
shooting agreement tests, binary continuation and Horizon-budget tests, MPC characterization
tests, augmented-Lagrangian mask tests, and Program/core reuse tests. Extend those behaviors
rather than testing new private helper names. Allocation/compilation instrumentation is justified
only for the explicit lifetime guarantees.

### Acceptance matrix

| Seam | Cases | Required evidence |
| --- | --- | --- |
| Problem construction | Defaults, invalid block counts, indivisible Horizon, transformed Problems | Unit default; clear early failures; blocking preserved through supported transforms. |
| Program construction | Both IPOPT formulations; unsupported Native Solvers/Backends | Supported continuous/binary/Horizon combinations construct; unsupported blocking fails before numerical work. |
| MPC advance | Counts zero, one, six, H; invalid types/ranges | Correct native tail fill, time, reference, pending push, unchanged terminal goal; invalid calls leave state unchanged. |
| MPC advance | Canonical Multipliers and padded augmented-Lagrangian state | Repeated existing one-count fill/mask behavior, including real rows next to terminal padding. |
| MPC measure | Advance followed by a different measured state | Next Boundary Conditions and result use measurement, not the shifted predicted state. |
| Program.solve | H = 12, b = 6, both formulations | Two independent control vectors; twelve returned controls; thirteen states; canonical result dimensions. |
| Objective evaluation | Explicit expansion; changing reference and time | Original native Cost, terminal Cost, output evaluation, and time spacing preserved. |
| Constraints | Kirchhoff, mixed state/control, time-dependent custom rows | One eligible input equality per block; every required native state/time evaluation retained. |
| Bounds | Unequal native limits, fixed controls, infinities, empty intersection | Tightest limits preserved; infeasible intersections rejected before IPOPT. |
| Binary bounds | Endpoints available/unavailable within block | Binary declaration respected; intervals excluding both endpoints rejected. |
| Horizon constraints | Activation sum, selected indices, state-dependent reduced rows | Native units and coefficient accumulation preserved; single-shooting derivatives include Rollout effects. |
| Binary solve | Successful continuation, pass limit, failed continuous pass | Native-weighted penalty; no rounding; honest status, original Cost, distance and pass diagnostics. |
| Results | Unit and blocked solves, success and failure | Canonical Multipliers or specified zero resets; no reduced arrays in canonical fields or stale carry. |
| Feasibility | Satisfied/violated equalities, inequalities, bounds and Defects | Native recomputation agrees; satisfied inequalities contribute zero. |
| Callback derivatives | Continuous/binary; exact/limited-memory configuration | Reduced dimensions and sparsity correct; objective/constraint derivatives agree with independent differences. |
| Program lifetime | Repeated advance/measure/solve and continuation | Same Program and handle; stable compilation after warm-up; fresh per-attempt diagnostics. |

### Detailed numerical oracles

- Use a small linear-quadratic continuous Problem with a unique optimum to compare blocked
  single and multiple shooting. Check objective agreement, expanded controls, native dynamics,
  bounds, and constraint residuals within explicit tolerances. Compare controls only where the
  optimum is unique and well conditioned.
- Compare b = 1 with the established successful unblocked mathematical behavior. The intentional
  interface removal and earlier validation timing are not regressions to preserve.
- For the six-count advancement oracle, use a uniform grid and compare all canonical primal,
  Multiplier, augmented-Lagrangian, reference, and clock values against six one-count advances.
  Add a nonuniform-grid case checking the documented consumed-prefix duration sum separately.
- Test one pending reference push during multi-count advancement, no pending push, zero count
  with a pending push, constant references, moving references, and an independent terminal goal.
- Seed nonzero canonical Multipliers before a blocked solve and verify replacement with zeros
  afterward, followed by successful advancement and another solve. Test b = 1 single shooting
  without any downstream wrapper and verify the recovered canonical stationarity residual.
- Construct a block whose tightest lower and upper bounds occur at different native Knot Points.
  Verify the expanded plan satisfies both. Include empty continuous intersection and nonempty
  continuous intersection with no permitted binary endpoint as distinct validation cases.
- Use a scalar state constraint active inside a block rather than solely at its first Knot Point.
  Add a time-dependent input row whose limits differ at native times, ensuring it is retained.
- Use a Kirchhoff-style equality with enough controls to make the original repeated-row count
  problematic after reduction. A successful feasible solve and the exposed retained row count
  establish that duplicate equalities were removed correctly.
- For tiny binary Problems, enumerate independent block assignments and solve or evaluate the
  remaining continuous Problem to establish an oracle. On selected deterministic Problems with
  a known reachable optimum, compare feasibility and objective to that oracle. This does not
  establish a global guarantee for arbitrary continuation solves.
- An H = 12, b = 6 activation budget of six permits at most one active block. A budget below six
  permits none. Include a nonmultiple budget to catch accidental conversion to block counts.
- Evaluate penalty derivatives at fractional controls, where the penalty and its curvature are
  nonzero. Check the factor b, objective factor, and zero contribution on continuous coordinates.
- Use central differences for objective gradients and constraint Jacobians; use differences of
  Lagrangian gradients for Hessians. Reconstruct sparse matrices from reported structure and
  values, including accumulated shared entries. Choose smooth test data away from nonsmooth
  points. Include a single-shooting Horizon row involving nonlinear rolled-out state coordinates.
- Run multiple solves through one Program while varying measured state, time, reference, goal,
  and continuation beta. Instrument actual tracing as well as external handle construction.
  Include a failed solve followed by a successful solve to expose stale diagnostics or beta.

### Verification workflow and completion criteria

- Establish failing behavioral tests for the existing count ambiguity, single-shooting canonical
  Multiplier mismatch, missing single-shooting binary/Horizon support, and repeated IPOPT handle
  construction before implementing their fixes.
- Implement and validate canonical results and count advancement first; add blocking through
  both shooting formulations, then shared binary continuation and native Horizon semantics.
  Exact derivative and lifetime evidence is required before claiming either formulation complete.
- Keep numerical tests small and deterministic. Tests requiring IPOPT identify the optional
  dependency explicitly; final acceptance requires running them in an environment with IPOPT,
  rather than considering skipped numerical tests sufficient.
- Use uv for targeted pytest, type, and lint checks. Follow the repository's verification
  requirements and explicit user overrides, and investigate failures without unrelated cleanup.
  A documentation-only change does not claim that the future implementation acceptance tests
  already pass.
- Delivery requires all four continuous/binary by single/multiple shooting combinations, canonical
  result and advancement tests, constraint/derivative evidence, explicit unsupported-pairing
  failures, native-grid regression checks, and separate handle/compilation reuse evidence.

## Out of Scope

- Blocked iLQR, ALTRO, AL, BoxQP, projected Newton, SQP, OSQP, Clarabel, or other unsupported
  Native Solvers and Backends. Their ordinary supported b = 1 operation remains available.
- Arbitrary block schedules, different schedules per control coordinate, nonuniform block
  counts, partial terminal blocks, and changing block structure inside an existing Program.
- Binary states, general integer domains, branch and bound, a new mixed-integer Backend, global
  optimality certificates, or guaranteed continuation convergence.
- New nonlinear Horizon constraint interfaces. Existing linear Horizon constraints may acquire
  nonlinear reduced expressions through single-shooting Rollout, which is in scope.
- Replacing the existing state-bound rejection in single shooting with a new path-constraint
  feature, changing manifold coordinates, or redesigning augmented-Lagrangian penalty carry.
- Measurement estimation, readiness gating, warm-up release, native service scheduling,
  replanning ratios, command playback, physical-current logging, fallback selection, deadline
  metrics, and application-specific activation or delivered-charge policies.
- Changes to the downstream neuro application, its CasADi implementation, inference adapters,
  candidate controllers, configuration files, plotting, scoring, or experiments. Their migration
  requirements are documented here, but their implementation is separate work.
- New application weight scaling, slew constraints, a hard charge target, learned-model
  retraining, deadline compensation, or changing the Predictor's native rate.
- A promised speedup proportional to block size, preserved closed-loop application performance,
  or reuse of IPOPT's internal factorization beyond what its existing interface provides.

## Further Notes

This specification records the scope agreed in the conversation: both shooting formulations,
continuous and binary controls, explicit rejection by unsupported Native Solvers and Backends,
and clean breaking interfaces. It expands the downstream proposal's initial single-shooting,
continuous-only scope intentionally. The confirmed test seams are part of acceptance.

The downstream example uses 10 ms native intervals, a six-control execution prefix, and six-step
blocking. These are motivating values, not hardcoded package settings. Distinct-control
multirate execution uses block size one with multi-count advancement; blocked execution adds
the reduced control Transcription. Either policy preserves native prediction geometry.

Before implementation, reconcile affected public documentation with the accepted Program,
solver-result, output-map, and binary-continuation decisions. Extend the single root domain
glossary only for new terms that implementation actually introduces; avoid a feature-local
competing glossary. Record any necessary architectural amendments through the repository's
existing ADR practice rather than silently contradicting accepted decisions.
