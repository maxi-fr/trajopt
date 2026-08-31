# ADR 0003: A solver states its constraint violation, and says so when it answers a different problem

## Status

Accepted.

## Context

`src/trajopt/benchmarks.py` gained `compare_solvers`, which runs one problem through several
solvers and tabulates the results side by side. Building it exposed a gap in what a
`SolverResult` promises.

`constraint_violation` was an optional field defaulting to `0.0`. Seven of the eight result types
assigned it; `ILQRResult` did not, and its docstring said "Always 0.0: an unconstrained iLQR has
no constraints." That is true of the solver and false of the problem. `ILQR` accepts any
`Problem`, ignores whatever constraints it carries, and returned a perfect feasibility score for
a trajectory that could be far outside them. In a comparison table sorted by anything, iLQR wins
by not doing the work.

The same shape of problem applies to the `OSQP` and `Clarabel` Backends. Both solve a single
convex subproblem built about the Operating Point. Their reported numbers are accurate, but they
describe a linearization of the problem rather than the problem, and nothing in the result says
so.

## Decision

### `constraint_violation` is mandatory

The field moved above `iterations` in all eight result `NamedTuple`s and lost its default. A new
solver cannot now return a result without stating how feasible it is, and the field ordering is
what enforces it rather than a convention someone has to remember. `ILQR.solve` computes it at
the eager boundary with `compute_constraint_violation`, the same function the Backends use.

`ALTRO`'s unconstrained-shortcut path states `0.0` explicitly, because there the claim is true:
`is_unconstrained()` has already established that there are no constraint rows and no finite box
bounds.

### A solver that answers a different problem warns

`ILQR.solve` warns when the problem it is handed is not structurally unconstrained. `OSQP.solve`
and `Clarabel.solve` warn unconditionally that they solve one linearization about the Operating
Point.

The warnings live in the solvers rather than in the benchmark harness. A user who swaps `Ipopt()`
for `ILQR()` in `problem.solve(state, solver=...)` — the one-word change the `Solver` protocol
exists to make possible — needs to hear it too, and the harness then gets it for free.

The alternative was to refuse: raise on a constrained problem, or have the harness skip such a
pairing. Both were rejected. "Unconstrained iLQR is 40x faster and leaves the cart 0.3 outside
its limit" is a real answer to "which solver works best for my case", and it is only available if
the solve runs and the violation is measured.

### The comparison harness recomputes cost and violation

`compare_solvers` scores every row itself, from the returned Primal Vector, via `eval_f` and
`compute_constraint_violation`. It does not read `result.cost` or `result.constraint_violation`
into the table.

Making the field mandatory fixes the one result that lied. It does not make the eight results
comparable, and that is a separate problem: ALTRO reports its augmented Lagrangian `c_max`, PN
its active-set residual, the Backends the transcription's violation including Defects, and iLQR
evaluates a retargeted objective. Each is right by its own definition. A column is only readable
down its length if one definition produced every entry in it.

The solvers' own numbers stay reachable — each `SolverRow` keeps the result it came from — so a
disagreement between claimed and recomputed violation is visible rather than hidden.

## Consequences

- A new solver must state its `constraint_violation`, and if it ignores part of the problem it
  must warn. Neither is optional and neither can be forgotten silently.
- `ILQR.solve` now pays a `compute_constraint_violation` call per solve, in the MPC loop the jit
  cache exists to make fast. It is unconditional rather than gated on the problem carrying
  constraints: the violation also covers Defects, which every problem has.
- The warnings fire on every call, with no dedupe state on the solver instances. Python's default
  filter shows one per call site; a caller who wants silence uses `warnings.filterwarnings`.
- `compare_solvers` does not declare a winner. Whether speed, feasibility, or cost decides is the
  caller's question, and a solver that is fastest because it ignored a constraint is not the
  answer to it.
