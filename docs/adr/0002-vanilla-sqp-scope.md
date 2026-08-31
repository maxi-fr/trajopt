# ADR 0002: The SQP Backend is deliberately vanilla

## Status

Accepted.

## Context

`src/trajopt/transcription/sqp.py` adds a sequential quadratic programming Backend: it
repeatedly linearizes the transcribed nonlinear program, solves the resulting quadratic program
with OSQP, and takes a step globalized by an L1 Merit Function line search, with Damped BFGS as
the default second-order model.

The design brief was explicitly CasADi's `sqpmethod` — a deliberately basic SQP — and not a
competitor to Ipopt. During design a fuller solver was worked out and then cut back. Several of
those cuts are invisible in the finished code: a reader sees only their absence, and each one is
a well-known technique whose omission looks like an oversight rather than a decision. One of
them has a reproducible failure mode on a benchmark problem this repository ships.

This ADR records the cuts so the next reader checks them against a decision instead of
re-deriving the argument, and so a failing quadrotor solve is diagnosed rather than patched.

## Decision

### No Elastic Mode; an inconsistent subproblem ends the solve

When linearized constraints are mutually inconsistent, OSQP reports the subproblem primal
infeasible and the solve terminates with an infeasible status. There is no relaxation and no
restoration phase of any kind.

Elastic Mode was designed in full — penalized slacks on every constraint row, making the
subproblem unconditionally feasible — and rejected on cost: it roughly doubles the quadratic
program's variable count, and needs an index map from constraint rows to slack pairs, a
guard keeping the Penalty Parameter above the exactness threshold, and a check that the slacks
actually vanished at convergence. Without that last check the solver converges neatly to an
infeasible point and reports success, which is a worse failure than not converging.

**This has a concrete consequence, not a theoretical one.** The quadrotor obstacle benchmark
initializes on a straight line from start to goal, which passes through the spherical keep-out
zone. Its linearized subproblem may be inconsistent, and the SQP Backend may fail there while
Ipopt succeeds. A feasible initial guess avoids it. This is the expected behaviour of the
solver as specified, not a bug.

### Built on the Ipopt derivative path, not the OSQP Backend's assembly

Two linearizations of the same problem already existed: the `eval_g` / `eval_jac_g` family the
Ipopt Backend uses, and the hand-rolled block assembly inside the OSQP Backend. The SQP Backend
uses the former.

This is surprising on its face — two OSQP-driven Backends that share no assembly code — so it is
worth the record. Reusing the OSQP Backend's assembly would have made a third caller of a path
that must then stay consistent with two others, and the Ipopt path is already `jit`-compiled and
already validated against Ipopt's own answers. It also emits constraint rows in canonical order,
so the SQP Backend's Multipliers need no permutation, while the OSQP and Clarabel Backends must
permute theirs back.

### Second-order cone constraints are accepted, unlike the OSQP Backend

The OSQP Backend rejects them with a `TypeError`; the SQP Backend solves them. The asymmetry is
structural rather than a feature: a direct Backend must pass the Cone through to a solver with no
cone support, whereas an SQP linearizes the constraint *function*, and a linearized cone
constraint is an ordinary linear constraint. The local model is poor, but it is not invalid.

### A line search, not a trust region

An L1-penalty SQP is classically a trust-region method, and a trust region would be unusually
cheap here — only a tightening of the step bounds, no extra rows or variables. It was rejected
because radius management (shrink and grow factors, acceptance ratio thresholds, an initial
radius) is a larger tuning surface than the line search's two constants. A Damped BFGS model is
positive definite, so the subproblem is bounded below and the line search is well-posed without
one.

### Non-monotone merit memory instead of a second-order correction

Every monotone Merit Function line search suffers the Maratos effect: good steps are rejected
near the solution and the step length collapses. The standard remedy is a second-order
correction. Instead the Armijo test compares against the maximum Merit Function value over a
short memory of recent iterations, as `sqpmethod` does — a few lines for most of the benefit.

### No exact Lagrangian Hessian mode

A callback for it exists and is used by the Ipopt Backend. It is indefinite away from the
solution, and OSQP requires a positive semidefinite quadratic term, so offering it would mean
writing a convexification routine that neither Damped BFGS (positive definite by construction)
nor Gauss-Newton (positive semidefinite for the shipped costs) needs.

### Separate primal and dual tolerances, not Ipopt's scaled criterion

Convergence is tested as maximum constraint violation against one tolerance and the Lagrangian
gradient's infinity-norm against another, following CasADi. Ipopt's Multiplier-scaled KKT error
is not ported. Head-to-head timings against Ipopt therefore compare two different notions of
"solved" and should not be read as precise.

### The subproblem is set up fresh each iteration

Reusing OSQP's symbolic factorization by updating matrix values in place is the largest
performance gain available and was deliberately deferred, not overlooked. It requires assembling
the row block from a fixed structural sparsity pattern with explicit zeros retained — never from
dense blocks, whose dropped numerical zeros would silently change the pattern between iterations
and corrupt the update. `jacobian_sparsity_pattern` already supplies the pattern. This is the
first optimization to reach for if profiling shows setup cost dominating.

## Consequences

The SQP Backend is readable end to end and has few knobs, at the price of being less robust than
Ipopt on problems with poor initial guesses. It is not intended to replace Ipopt for hard
problems; it is intended to be a correct, inspectable SQP that the benchmark can include and that
a reader can follow.

The Damped BFGS matrix is dense, making that mode quadratic in the size of the Primal Vector. At
the horizons the benchmark suite uses (a few hundred primal variables at most) this is
immaterial, but it bounds the useful Horizon length, and Gauss-Newton is the mode to reach for
when that bound bites.

Reversing any single cut is contained work, since each was designed before being dropped. Elastic
Mode is the largest and would touch subproblem assembly, the Penalty Parameter update, and the
convergence test together.
