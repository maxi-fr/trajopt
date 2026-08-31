# Vanilla SQP backend

## Problem Statement

`trajopt` has three transcription backends and one native solver family. None of them is a
sequential quadratic programming method.

- `Ipopt` solves the nonlinear program properly, but it is an interior-point method behind a
  C++ extension: opaque, heavyweight, and impossible to read as a teaching artefact.
- `OSQP` and `Clarabel` solve exactly one convex approximation, expanded about an
  `operating_point`. The `OSQP` docstring already tells the user what is missing: *"drive it
  down by re-solving with the operating point set to the previous solution."* That instruction
  describes an SQP outer loop the library does not provide, so every user who needs one
  hand-rolls it, without a merit function, a line search, or a Hessian update.
- The native `ALTRO` family is trajectory-space and JAX-traced. Its line search is welded to a
  closed-loop nonlinear rollout and cannot serve a full-space step.

The gap is a small, readable sequential quadratic programming solver: one that repeatedly
linearizes the nonlinear program, solves the resulting quadratic program with OSQP, and takes a
globalized step. The reference point is CasADi's `sqpmethod` — deliberately vanilla, with a
dense quasi-Newton Hessian and an L1 merit line search, and nothing else.

## Solution

A new transcription backend, `SQP`, that sits beside `Ipopt`, `OSQP` and `Clarabel` and speaks
the same adapter interface: a frozen dataclass with a `solve(problem, state)` method returning
an `SQPResult`.

Each iteration:

1. Evaluate the objective, its gradient, the constraint vector and the constraint Jacobian at
   the current iterate.
2. Assemble a quadratic program in the step variable: the Hessian approximation as the
   quadratic term, the objective gradient as the linear term, the constraint Jacobian and the
   identity stacked as the row block, with row bounds shifted by the current constraint and
   iterate values.
3. Solve that quadratic program with OSQP, yielding a step and its multipliers.
4. Update the L1 penalty parameter from the quadratic program's multipliers.
5. Backtrack along the step until the L1 merit function satisfies an Armijo condition against a
   short non-monotone memory of recent merit values.
6. Update the Hessian approximation from the accepted step.
7. Test convergence on primal and dual infeasibility separately.

The scope discipline is explicit: this is a *vanilla* SQP. Where a textbook improvement would
cost more than a little complexity it is left out and recorded in **Out of Scope**, so the
omissions read as decisions rather than oversights.

## Implementation Decisions

### Placement and substrate

The solver is a **transcription backend**, not a native solver. OSQP is a C extension that
cannot execute inside a traced JAX loop, so the outer iteration must be host-side Python. This
means the solver does not participate in `SolverOptions`, `SolverStats` or `TerminationStatus`,
and is not jittable end to end the way `ALTRO` is. It gets its own small options dataclass
instead.

### Derivative source

The solver is built on the **full-space derivative callbacks already used by the `Ipopt`
backend** — objective value, objective gradient, constraint vector, constraint Jacobian — plus
the existing primal-bound and constraint-bound helpers.

It deliberately does **not** reuse the `OSQP` backend's internal quadratic-program assembly.
That path is a second, independently written linearization of the same problem, and reusing it
would create a third caller that must then stay consistent with two others. The chosen path is
already `jit`-compiled and already validated against Ipopt.

One consequence is load-bearing and simplifies the design: the constraint-vector callback emits
rows in **canonical order** — the initial condition, then each knot's dynamics defect followed
by that knot's constraint rows. The `OSQP` and `Clarabel` backends emit *blocked* order and must
permute their duals back into canonical order. Building on the canonical path means `SQP`'s
multipliers are already canonical, so no permutation step exists, and the existing dual
warm-start helper plugs in directly.

### Hessian approximation

Two modes, selected by an option:

- **Damped BFGS (default).** A dense quasi-Newton approximation of the Lagrangian Hessian.
  Curvature pairs are formed from the accepted step and the change in the Lagrangian gradient
  evaluated at the *same* multipliers. Powell damping is applied whenever the curvature
  condition fails, which keeps the matrix positive definite unconditionally. That guarantee is
  load-bearing beyond convergence quality: OSQP requires a positive semidefinite quadratic term,
  and Powell damping supplies it without any separate convexification routine. The matrix is
  initialized to the identity at the start of every solve.
- **Gauss-Newton.** The objective's own second-order model, with the constraint-curvature term
  dropped. Positive semidefinite for the quadratic costs the library ships, and it needs no
  update rule. Nearly free to implement, because the cost-expansion helper already exists and is
  already used by the `OSQP` backend.

The **exact Lagrangian Hessian is not offered**, even though a callback for it exists. It is
indefinite away from the solution, so it would need a convexification routine that neither other
mode requires.

The BFGS matrix is dense, making that mode quadratic in the primal dimension. At the horizons
the benchmark suite uses this is immaterial — the largest problem has a few hundred primal
variables — but it is a real constraint at long horizons and belongs in the docstring rather
than being discovered.

### Globalization

An **L1 merit function**: the objective plus a penalty parameter times the one-norm of the
constraint violation. Backtracking is Armijo, against the directional derivative predicted by
the quadratic program.

The penalty parameter is **updated from the quadratic program's multipliers** each iteration,
increasing monotonically toward a threshold above the multiplier infinity-norm and capped at a
maximum. A fixed penalty is rejected: too small and the search accepts steps that worsen
feasibility, too large and it stalls.

Acceptance is **non-monotone** over a short memory of recent merit values, as CasADi's
`sqpmethod` does. The Armijo test compares against the maximum merit over that memory rather
than the immediately preceding value. This costs a handful of lines and partially mitigates the
Maratos effect — the rejection of good steps near the solution that afflicts every monotone
merit line search — without any of the machinery a full remedy needs.

A line search failure, meaning the step length falls below a floor, terminates the solve rather
than triggering a fallback.

### Quadratic program handling

The subproblem is set up and solved fresh each iteration.

**Second-order cone constraints are accepted**, unlike the direct `OSQP` backend, which rejects
them. The difference is structural rather than an added feature: the direct backend must hand a
cone through to a solver with no cone support, whereas an SQP linearizes the constraint
*function*, and a linearized cone constraint is an ordinary linear constraint. The rejection
branch simply is not written. The resulting local model is a poor one, and the docstring should
say so.

If OSQP reports the subproblem primal infeasible — the linearized constraints are mutually
inconsistent — the solve **terminates and reports infeasibility**. There is no relaxation, no
elastic mode, no restoration phase.

### Convergence and status

Primal and dual infeasibility are tested **separately against independent tolerances**,
following CasADi rather than Ipopt: maximum constraint violation for the primal test, the
infinity-norm of the Lagrangian gradient for the dual test. Ipopt's multiplier-scaled criterion
is not ported.

Termination reasons map onto the library's existing four-value solver status vocabulary:
convergence, iteration limit, subproblem infeasibility, and line search failure.

### Result type and warm starting

`SQPResult` mirrors the shape of `IpoptResult` and `OSQPResult` — trajectory, success flag,
status, message, cost, flat primal vector, info dictionary, iteration count, constraint
violation, constraint duals, bound duals — so the existing cross-backend tests can treat it as
one more member of the family.

Warm starting carries the **primal iterate and the duals** through the existing helpers. The
Hessian approximation and the penalty parameter are **reset at the start of every solve**.
Carrying them across MPC steps would help, but it needs a staleness guard to avoid being worse
than a reset after a disturbance, and that guard is more machinery than the gain justifies.

### Benchmark integration

`SQP` is registered alongside the existing backends in the benchmark module so the configurable
benchmark can include it.

## Testing Decisions

A good test here asserts on **external behaviour of the adapter**: the returned trajectory, the
cost, the constraint violation, the reported status, the duals. It does not reach into the
iteration loop, the Hessian matrix, or the penalty parameter's trajectory. The one exception is
iteration *count* as a proxy for warm-start effectiveness, which the repository already treats
as observable behaviour in its existing adapter tests.

### Seam 1 — the adapter's own behaviour

A dedicated unit test module, modelled on the existing `OSQP` backend tests. A small
double-integrator problem, solved; asserts on result type, success, iteration count, cost,
constraint violation, trajectory shapes, and the initial state being respected. A second test
adds control and state bounds plus a linear constraint and asserts the bounds hold at the
solution. A third asserts that a second-order cone problem *solves* rather than raising — the
behavioural difference from the direct `OSQP` backend.

Both Hessian modes are exercised. A nonlinear problem where the constraint-curvature term is
nonzero — the quadrotor's spherical keep-out is the natural choice — is what distinguishes them.

### Seam 2 — parity with the other backends

`SQP` joins the existing parametrized cross-backend tests rather than getting bespoke copies:
the common adapter interface test, the test that backends agree on the duals of a shared
optimum, the dual warm-start test, and the problem-definition-invariance test.

This is the highest available seam. It asserts the thing that actually matters — that `SQP` is
indistinguishable from the other backends where it should be — and it is nearly free, because
each of those tests is already parametrized over backend classes.

### Seam 3 — cross-verification against CasADi

The strongest seam, and the one that speaks directly to the reference implementation. The
cross-verification suite already contains a CasADi baseline module that builds an equivalent
CasADi problem from a `Problem`, plus parity assertions for trajectories, costs and per-block
duals. `SQP` gets a parity test on the benchmark problems through that existing machinery.

The comparison is on the **converged solution**, not on the iterate sequence. Matching CasADi's
`sqpmethod` iterate for iterate would be fragile — it depends on Hessian initialization,
backtracking constants and merit memory details that are not part of any contract. Matching the
optimum it converges to is both robust and the property that matters.

The CasADi baseline currently hardcodes Ipopt as its solver. Letting it also target `sqpmethod`
is a small extension and worth doing, because it upgrades "our SQP finds the same optimum as an
interior-point method" into the sharper "our SQP finds the same optimum as the reference SQP".

### Prior art

The existing backend unit tests supply the single-adapter pattern; the shared adapter test
module supplies the parametrized cross-backend pattern; the CasADi cross-verification module
supplies the parity-assertion helpers and the problem-translation layer.

## Out of Scope

Each of the following was designed and then deliberately cut for costing more than a little
complexity.

- **Elastic mode / permanent L1-penalty subproblem.** Would guarantee a feasible subproblem
  always, at the cost of roughly doubling the quadratic program's variable count, an index map
  from constraint rows to slack pairs, a penalty-exactness guard, and a check that the slacks
  actually vanished at convergence. The concrete price of omitting it: the quadrotor benchmark
  initializes on a straight line passing through the keep-out sphere, so its linearized
  subproblem may be inconsistent and the solve may fail where Ipopt succeeds. A feasible initial
  guess avoids this.
- **Second-order correction.** The standard Maratos remedy, partially substituted for by the
  non-monotone merit memory at a fraction of the cost.
- **Trust region.** The classical pairing for an L1-penalty SQP, and cheap to express here since
  it is only a tightening of the step bounds. Rejected because radius management — shrink and
  grow factors, acceptance ratio thresholds, an initial radius — is a larger tuning surface than
  the line search's two constants.
- **Warm-started OSQP factorization across iterations.** Reusing the symbolic factorization by
  updating matrix values in place, rather than setting the subproblem up fresh, is the single
  largest performance gain available. It requires assembling the row block from a fixed
  structural sparsity pattern with explicit zeros retained, never from dense blocks, because
  dropped numerical zeros would silently change the pattern between iterations. The pattern
  helper already exists. This is the first optimization to reach for if profiling shows setup
  cost dominating.
- **Adaptive subproblem tolerance.** Loosening the OSQP tolerance early and tightening it as the
  outer iteration converges. Cheap, but nonstandard and absent from the reference.
- **Exact Lagrangian Hessian mode.** Would require convexification.
- **Ipopt's multiplier-scaled KKT criterion.** Replaced by separate primal and dual tolerances.
- **Persisting the Hessian approximation and penalty parameter across MPC solves.**
- **Feasibility restoration of any kind.**

## Further Notes

The `ALTRO` line search is **not** reusable here, despite the initial hope. It consumes an
affine feedback policy and rolls the nonlinear dynamics closed-loop, so dynamic feasibility
holds at every trial point by construction, and it accepts on a cost-only ratio against the
backward pass's predicted decrease. An SQP step is a full-space direction that *violates* the
dynamics: there is no policy, no rollout, and no predicted decrease of that kind, and cost alone
is the wrong acceptance test. The Projected Newton line search is closer in shape — it does
backtrack along a flat direction — but accepts on constraint violation alone, which is equally
wrong. The L1 merit backtrack is written fresh.

The direct `OSQP` backend's `operating_point` mechanism is, in retrospect, a single iteration of
this solver without globalization or a Hessian update. The two should stay separate — the direct
backend's value is that it is exactly one convex solve — but the relationship is worth a sentence
in both docstrings.

Two documentation follow-ups are open. The repository has no `CONTEXT.md` despite the agent
instructions referencing one, and the terms this design settles — Operating Point, Merit
Function, Damped BFGS, Elastic Mode — are natural first entries. An architecture decision record
covering the vanilla-scope cuts is also warranted, since a future reader hitting the quadrotor
failure mode will otherwise re-litigate the elastic-mode decision from scratch.
