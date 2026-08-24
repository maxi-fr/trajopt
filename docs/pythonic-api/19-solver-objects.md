# 19 — Solver backends become objects

**What to build:** choosing a solver means constructing one — `problem.solve(state, solver=OSQP(operating_point=traj))` — instead of passing a magic string and hoping the options paired with it are legal for that backend. Illegal pairings stop type-checking rather than raising at run time. After a solve, the state knows whether it succeeded.

**Blocked by:** 18.

## Why

Solver selection is a string compared against three literals in an if/elif chain, with a
fourth branch accepting an arbitrary callable. The chain carries two run-time errors that
exist *only* because the arguments are untyped: passing `operating_point` to Ipopt raises
(Ipopt solves the nonlinear problem directly and has no operating point), and passing it
alongside a callable raises (nothing would forward it). Both are the type system's job.
Give `operating_point` to `OSQP` and `Clarabel` as a field and withhold it from `Ipopt`, and
neither error can be written.

The arbitrary-callable escape hatch has zero callers in `src/` or `test/`. It goes.

## What changes

A `Solver` protocol — `solve(problem, state) -> SolverResult` — lives beside the backends that
implement it, with `SolverResult` promising the three fields the problem actually reads: the
primal vector and the two multiplier vectors. The three existing backend result types keep
their own richer shapes and satisfy the protocol structurally. Use `typing.Protocol`, not an
ABC: the backends share no implementation, only a shape, and an ABC would impose inheritance
on three wrappers around foreign C libraries.

`Ipopt`, `OSQP`, `Clarabel` are frozen dataclasses. `Ipopt` carries native options.
`OSQP` and `Clarabel` carry native options and `operating_point`. `problem.solve(state, solver=None)`
defaults to Ipopt, resolved lazily inside the call — a module-level default would close an
import cycle between the problem and the transcription layer.

`MPCState` gains a `status` field recording how the last solve ended, normalized across
backends to a small vocabulary rather than passing through native codes: converged, infeasible,
iteration limit, error. The backends' own result types keep their native status and message for
anyone who needs the detail. This is the one field this refactor *adds* — justified because for
MPC "did that solve succeed" is the question you most need answered, and today `solve` discards
it entirely. Keep the field static on the pytree; it is metadata, never traced.

The free functions `solve_ipopt`, `solve_osqp`, `solve_clarabel` are deleted along with the
module-level `solve`. Their work moves into the corresponding `.solve` methods.

## The migration is not purely mechanical — read this before starting

The free backend functions accept far more than the protocol will: as well as a `Problem` they
take **either** an `MPCState` **or** a raw initial-state array, plus loose `t0`, `dt`, `xf`,
`initial_trajectory` and `initial_z` keywords, which an internal parser folds into a state.
The protocol takes a `Problem` and an `MPCState`, full stop — keeping the union would preserve
exactly the wart this ticket exists to remove.

So the roughly 47 call sites across seven test files split into two kinds. Those already
passing an `MPCState` are a spelling change. Those passing a raw array plus keywords must
construct `MPCState.initial(problem, x0, ...)` first. Budget for the second kind; it is the
bulk of the work in this ticket and the reason it is not a one-sitting rewrite. The assertions
in those tests do not change — only how the input state is built.

## Acceptance criteria

- [ ] A `Solver` protocol and a `SolverResult` protocol exist; the three backend result types satisfy `SolverResult` structurally with no inheritance added.
- [ ] `Ipopt`, `OSQP`, `Clarabel` are frozen dataclasses; `operating_point` is a field of the latter two only.
- [ ] `problem.solve(state, solver=None)` defaults to Ipopt without a module-level import cycle.
- [ ] The string dispatch chain, the callable-solver branch, and both `operating_point` run-time errors are gone.
- [ ] `solve_ipopt`, `solve_osqp`, `solve_clarabel` and the module-level `solve` are deleted.
- [ ] `MPCState.status` records a normalized outcome; each backend maps its native status onto it.
- [ ] Every test call site constructs an `MPCState` and calls a solver object; no test asserts on a raw-array entry path.
- [ ] Full suite green with no assertion changed.
