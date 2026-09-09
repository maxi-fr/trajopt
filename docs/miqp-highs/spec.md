# Mixed-Integer Quadratic Programming Backend (HiGHS)

**Wrong**: highspy only serves a MI-LP backend not QP, as far as i know

## Problem Statement

Optimal control problems in robotics and cyber-physical systems frequently require discrete decisions alongside continuous state evolution. Common examples include thruster allocation with discrete on/off firing levels, hybrid contact mode scheduling in legged locomotion, gear selection in vehicle powertrains, and obstacle avoidance over partitioned convex regions.

Currently, `trajopt` operates exclusively on continuous decision variables. Its existing full-space **Backends** (`Ipopt`, `OSQP`, `Clarabel`) and **Native Solvers** (`ILQR`, `AL`, `ALTRO`) assume smooth, real vector spaces. Non-linear mixed-integer programming (MINLP) via external solvers like BONMIN or SCIP incurs prohibitive solve latencies (seconds to minutes per solve) and suffers from severe Python binding friction on modern platforms.

In contrast, Mixed-Integer Quadratic Programming (MIQP) with modern branch-and-cut solvers like HiGHS can solve convex quadratic subproblems with discrete variables in 5 to 50 milliseconds. This latency makes MIQP viable for online, receding-horizon **MPC**.

The gap in `trajopt` is the lack of a discrete variable declaration contract and an MIQP **Backend** that consumes the transcribed **Quadratic Subproblem** without compromising continuous workflows.

## Solution

A new transcription backend, `Highs`, wrapping the official `highspy` C++ interface:

1. Introduce an integrality contract into problem formulation and transcription layout.
2. Extend the layout engine to map discrete channels into an integrality mask over the full **Primal Vector**.
3. Re-use the existing **Quadratic Subproblem** transcription engine to assemble the sparse objective Hessian and linear constraint Jacobian in canonical CSC format.
4. Pass the CSC matrices, bounds, and variable integrality vector to `highspy.Highs`.
5. Solve the MIQP via HiGHS branch-and-cut.
6. Return a typed `HighsResult` satisfying the shared solver result protocol, allowing execution inside `MPC`.

## Implementation Decisions

### Settled Architectural Decisions

#### Placement and Substrate

The solver is a **transcription backend**, located in the transcription layer beside `OSQP` and `Clarabel`. Like OSQP, HiGHS is a compiled C++ library executing host-side in Python. It does not execute inside JAX-traced loops and does not participate in native solver options. It provides its own frozen configuration dataclass for solver options (e.g. time limits, optimality tolerances, MIP gap tolerances).

#### Subproblem Assembly and Sparsity

The backend reuses the canonical **Quadratic Subproblem** assembly from the transcription layer:

- The objective Hessian $H$ and linear constraint matrix $A$ are extracted in canonical CSC format.
- Row bounds ($l \le A \Delta z \le u$) and primal column bounds ($z_L \le z \le z_U$) are aligned with existing transcription helpers.
- Variables are handed to HiGHS directly in sparse format without intermediate dense conversions.

#### Integrality Propagation in Layout

The layout engine maps discrete variable metadata into a full-horizon integrality vector matching the flat **Primal Vector** $Z$:

- Continuous variables are marked with continuous variable types.
- Discrete variables are marked with integer variable types.
- When variable bounds are $[0, 1]$, HiGHS automatically treats integer variables as binary.

---

### Open Decisions to Resolve in Planning

#### Open Decision 1: Discrete Variable Declaration API

How should users specify which variables are discrete?

- **Option 1A: Problem-level parameter (`discrete_controls` on `Problem`)**
  - *Description:* Declare discrete control channels as a tuple of channel indices when constructing the problem.
  - *Pros:*
    - Control channels are uniform across all stage **Knot Points**.
    - Integrality is an intrinsic variable domain property ($u \in \mathbb{Z}^m$) rather than an algebraic residual constraint $c(x, u) \in \mathcal{K}$.
    - Keeps constraint collections strictly focused on differentiable residuals and cones.
    - Continuous backends (`Ipopt`, `OSQP`, `Clarabel`) can inspect the problem at build time and immediately raise a descriptive configuration error.
  - *Cons:*
    - Does not support stage-varying integrality (e.g. discrete controls for the first $k$ knots, relaxed continuous controls thereafter).
    - Restricted to control channels; does not accommodate discrete state coordinates or auxiliary slack variables directly.

- **Option 1B: Constraint-based declaration in `ConstraintList` (e.g. `BinaryControl`, `IntegerControl`)**
  - *Description:* Add an explicit integrality constraint to the problem's constraint list over a specified range of knot indices.
  - *Pros:*
    - Fits the familiar `add_constraint(con, knot_indices)` pattern.
    - Supports knot-specific discrete horizons.
    - Can naturally extend to discrete states or auxiliary variables.
  - *Cons:*
    - Integrality has no differentiable residual, Jacobian, or cone projection.
    - Continuous solvers and expansion engines must add special-case filtering to ignore or reject non-differentiable integrality constraints during derivative evaluation and transcription.

- **Option 1C: Bound-level metadata**
  - *Description:* Attach an integrality flag to variable bound constraints (e.g. marking a `ControlBound` as integer).
  - *Pros:* Integrality is often paired with specific lower/upper limits (e.g. binary $[0, 1]$).
  - *Cons:* Conflates geometric bounds with numerical variable types.

---

#### Open Decision 2: Problem Class Scope & Non-linear Strategy

What scope of optimal control problems should the backend support initially?

- **Option 2A: Single-Pass Convex MIQP (Linear/Affine Hybrid Systems Only)**
  - *Description:* Strictly target problems with linear/affine dynamics and linear constraints. HiGHS solves the exact global problem in one pass.
  - *Pros:*
    - Clean, deterministic global solutions without outer-loop convergence tuning.
    - Fast solve times (5–30 ms) suitable for high-rate MPC.
    - Smallest implementation surface area.
  - *Cons:*
    - Cannot handle non-linear dynamics directly without manual external linearization.

- **Option 2B: Single-Pass Subproblem Solve (RTI Building Block)**
  - *Description:* For non-linear problems, linearize once about an **Operating Point** (e.g. the shifted trajectory from the prior step) and solve one MIQP per time step.
  - *Pros:*
    - Replicates the existing behavior of `OSQP` and `Clarabel` in `trajopt`.
    - Directly enables Real-Time Iteration (RTI) receding-horizon MPC for non-linear systems without multi-iteration latency.
  - *Cons:*
    - A single linearization about an operating point may have poor predictive accuracy if the discrete mode changes significantly.

- **Option 2C: Full Sequential MIQP (MISQP) Outer Loop**
  - *Description:* Implement an iterative outer loop that repeatedly linearizes non-linear dynamics, solves an MIQP, evaluates constraint defects, updates trust regions on continuous variables, and iterates until convergence.
  - *Pros:*
    - Solves non-linear hybrid problems to convergence.
  - *Cons:*
    - Substantially higher complexity (requires trust-region radius adaptation, integer step locking, and merit evaluations).
    - Multi-iteration solve times may exceed online MPC latency budgets.

---

#### Open Decision 3: Logical and Mode-Switching Constraint Modeling

How should discrete switching logic (e.g. contact active vs. inactive, collision avoidance side choice) be expressed?

- **Option 3A: Direct Linear Inequalities (Raw Big-M)**
  - *Description:* Users formulate switching logic as standard linear inequalities ($A x + B u \le b$) using the existing linear constraint catalog.
  - *Pros:*
    - Zero new constraint classes or abstractions required.
    - Directly reuses the existing constraint Jacobian transcription pipeline.
  - *Cons:*
    - Users must manually tune big-M values. Poorly chosen bounds can lead to numerical ill-conditioning or loose LP relaxations during branch-and-cut.

- **Option 3B: Dedicated Hybrid Modeling Helpers (e.g. `SwitchedDynamics`, `IndicatorConstraint`)**
  - *Description:* Introduce structured helper classes that automatically compute big-M bounds or map to HiGHS indicator constraints.
  - *Pros:*
    - Safer, higher-level user API that prevents invalid big-M choices.
  - *Cons:*
    - Expands the library's domain model and API surface area.

---

#### Open Decision 4: Multiplier and Dual Recovery Policy

How should the backend report dual variables (multipliers) when discrete variables are present?

- **Option 4A: Return empty dual vectors**
  - *Description:* Set `lam` and `mu` to empty arrays when discrete variables are active.
  - *Pros:* Mathematically accurate; KKT dual variables are not well-defined for mixed-integer solutions due to the discrete lattice.
  - *Cons:* Downstream consumers that expect dual arrays for warm-starting or sensitivity analysis receive empty values.

- **Option 4B: Fixed-integer continuous resolve**
  - *Description:* After finding the optimal integer assignment, fix integer variables to their optimal values and perform a continuous QP resolve to obtain dual multipliers for continuous constraints and bounds.
  - *Pros:* Provides dual variables and constraint shadow prices for continuous states and dynamics.
  - *Cons:* Requires a second continuous QP solve, adding computational overhead to each step.

---

## Testing Decisions

A good test asserts on **external observable behavior**:

- The returned **Trajectory** states and controls satisfy the discrete requirements (e.g. discrete controls strictly conform to the integer lattice).
- Costs and constraint violations match analytical or verified reference values.
- Receding-horizon **MPC** steps execute cleanly without state corruption across horizon shifts.
- Infeasible or misconfigured problems report appropriate failure statuses.

### Seam 1: Pure Continuous Parity

Test the backend on standard continuous benchmark problems (e.g. double integrator, linear tracking) with no discrete variables.

- Assert that the backend converges to the identical optimal cost, trajectory, and status as `OSQP` and `Clarabel`.
- Verifies that matrix ingestion and basic subproblem handling match existing solvers.

### Seam 2: Switched Hybrid Control (MIQP Unit Test)

A dedicated hybrid test problem: a double integrator with switched thruster modes (e.g. discrete control $u_0 \in \{-1, 0, 1\}$ or binary forward/reverse firing).

- Assert that the solver successfully converges.
- Assert that every control entry along the entire **Horizon** strictly satisfies the integer lattice within floating-point tolerance ($\epsilon < 10^{-6}$).
- Assert that the continuous state tracks the target position while satisfying dynamics defects.

### Seam 3: Disjunctive Region Selection (Big-M Logic)

A test problem featuring obstacle avoidance or target selection between disjoint convex regions using binary variables:

- Assert that the solver selects the lower-cost convex region.
- Assert that the inactive region constraint is relaxed by the big-M term, while the active region constraint binds.

### Seam 4: Closed-Loop Receding-Horizon MPC (Highest Seam)

Drive the switched hybrid system using `MPC` and the backend over multiple simulation steps:

- Assert that the warm start advances correctly on `shift()`.
- Assert that closed-loop states converge to the goal.
- Assert that execution time per step remains within the real-time threshold (< 50 ms).

### Seam 5: Incompatibility Guards

- Assert that instantiating a continuous backend (`Ipopt`, `OSQP`, `Clarabel`) with a problem containing discrete declarations raises an immediate, descriptive error rather than failing silently.

### Prior Art

The tests mirror the structure of existing backend unit tests for OSQP and Clarabel, as well as the closed-loop simulation tests in the example suite.

## Out of Scope

The following items are out of scope for this specification:

- **Non-linear MINLP Solvers:** No integration with BONMIN, SCIP, or Couenne.
- **Discrete State Variables:** States remain continuous integrations of differential equations across stages; discrete behavior enters through controls and constraints.
- **Custom Branch-and-Cut Heuristics:** HiGHS handles all tree search, presolve, and cutting planes natively.
- **Automatic Big-M Bound Tightening:** Users provide explicit upper/lower bounds when formulating logical constraints.

## Further Notes

Adding `highspy` requires adding it as an optional dependency in `pyproject.toml` under the `solvers` extra:

```toml
[project.optional-dependencies]
solvers = [
    "cyipopt>=1.3.0",
    "osqp>=0.6.2",
    "clarabel>=0.6.0",
    "highspy>=1.7.0",
]
```

The integration preserves `trajopt`'s core design: fast, differentiable continuous optimal control by default, with an efficient bridge to modern external combinatorial solvers when hybrid decisions are required.
