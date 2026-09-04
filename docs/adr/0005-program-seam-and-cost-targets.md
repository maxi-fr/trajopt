# ADR 0005: A Program is a Problem's compiled form, and a cost carries shape without a target

## Status

Accepted.

## Context

`MPCState` held everything a receding-horizon loop needed in one object: the problem, the current
state and time, the goal, the warm start, the last result, and a `JitCacheSlot` holding the
compiled solver core. Running an MPC loop with it recompiled the traced core on **every step**.

The cause was not one bug but three, each with the same shape: data that changes every step was
reaching `jax.jit` as a compile-time constant rather than as a traced argument.

1. **The goal was baked into the objective.** `LQRObjective(Q, R, Qf, N, xf)` folded `xf` into the
   cost's linear and constant terms at construction. Moving the goal meant building a new
   objective, which meant a structurally different pytree, which meant a retrace.
2. **The cache key was object identity.** `JitCacheSlot` kept a compiled closure keyed on the
   objects it closed over, so anything that rebuilt one of those objects invalidated it.
3. **Boundary data was carried as static fields.** Anything marked `eqx.field(static=True)`
   participates in the pytree's *structure*, so a new value is a new structure and a new trace.

A closed-loop cartpole run took 20 compiles over 20 steps. Compilation dominated, and the cache
existed precisely to prevent that.

## Decision

### Four objects, split by what changes and when

- **`Problem`** — purely structural and immutable: model, objective *shape*, constraints, `N`,
  `dt`. It says what the problem is.
- **`BoundaryConditions`** — `x0`, `t0`, and the reference window `X_ref`/`U_ref`, plus the
  terminal goal `xf` (ADR: separating the goal from the window). **All array leaves, zero static
  fields**, so the whole object passes as a traced jit argument.
- **`Program`** — one solver's compiled, allocated form of a Problem: its jitted cores and its
  live C handles. Mutable, eager-side, per-solver.
- **`MPC`** — the driver: a Program, its BoundaryConditions, and the reference cursor.

The rule that makes this work: **the Problem and the solver's static configuration are the
Program's identity; everything else is a traced argument.** A different static configuration means
a different Program, not a silent retrace of an existing one.

### A cost carries shape, not a target

`LQRObjective(Q, R, Qf, N)` no longer takes `xf`. Its linear and constant terms start at zero —
which regulates to the origin — and the target arrives as traced data through
`BoundaryConditions`, applied by `Objective.with_reference` inside the traced core.

A goal point is not a special case: it is a constant reference window. Regulation and tracking go
through the one `with_reference` mechanism rather than two parallel paths.

`Objective.carries_reference` reports whether a cost's linear terms are *already* aimed at a
build-time reference. It reads concrete values, so it belongs to eager setup and not to a traced
core, and it is what lets `MPC` reject a constant goal aimed at a `TrackingObjective` — which
would silently flatten the tracked window at every knot.

### A Program is deliberately not a pytree

`vmap` over a Program is given up on purpose. A Program holds live C solver handles, which cannot
be traced. This was accepted explicitly rather than discovered: batching a *Problem* is still
possible, and batching a *solve* is what is lost.

## Consequences

- **The closed loop compiles its traced core exactly once.** This is pinned by a test asserting
  the compile count is `== 1`, not "small". That test is the definition of done for this work, and
  it discriminates: forcing a cache miss makes it read 20.
- **A `Program` field must never become a traced value, and a `BoundaryConditions` field must never
  become static.** Either mistake reintroduces per-step recompilation, silently — nothing fails,
  it just gets slow. The compile-count test is the only thing that catches it.
- **`Program.handles` gives eager backends somewhere to keep live state.** OSQP uses it to reuse a
  factorization across steps rather than setting one up each time.
- **A goal-carrying objective built the old way now regulates to the origin instead.** There is no
  compatibility shim: `LQRObjective` lost its `xf` parameter rather than deprecating it.
- **`Problem.solve` and `Problem.cost` are gone.** A Problem is structural; asking it to solve
  itself required it to own a solver and a cache, which is what `Program` and `MPC` now do.

## Deliberate debt

Recorded here because it is invisible from the code that looks finished:

- `TrackingObjective` and `update_reference` still fuse shape and target at build time. Only
  `LQRObjective` was unfused. The `carries_reference` guard exists because that conflict is still
  reachable.
- `Objective.with_reference` **overwrites** the linear terms rather than composing with them, so
  applying it to a cost that already carries a reference discards the original.
