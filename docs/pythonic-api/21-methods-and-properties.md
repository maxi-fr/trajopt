# 21 — Methods and properties replace the free-function API

**What to build:** the API in the README snippet. Ask an object for what it knows —
`opt_state.states`, `problem.cost(state)` — instead of passing it to a function named after
the thing you wanted. This is the ticket the whole sequence exists for; the ones before it
clear the dispatch chains out of the way so this one is a rename rather than a rewrite.

**Blocked by:** 19, 20.

## Why

The module-level `states`, `controls`, `initial_states`, `initial_controls` are one-line
wrappers around methods that already exist on `MPCState`. They are pure Julia-idiom residue:
a function taking its receiver as the first argument. Deleting them is the whole change.

`cost(problem, state)` follows `solve` onto the problem for the same reason `solve` went there
in ticket 19 — the state is data, the problem is what knows how to evaluate against it.

`Trajectory` is inconsistent with itself: dimensions are properties while `states()`,
`controls()` and `times()` are *methods that return public fields*. Two names for one field is
one too many, and `traj.X` is both shorter and what every internal caller already uses.

## What changes

`MPCState.states` and `.controls` become properties. They unpack the flat primal vector on each
access rather than reading a field, so reading both unpacks twice — accepted: it is a reshape of
an already-materialized array, not worth an API wart to avoid.

`Trajectory.states()`, `.controls()` and `.times()` are deleted in favour of `.X`, `.U`, `.t`.

Both immutable types settle on `with_` for "returns a new instance", joining the
`with_measurement` / `with_goal` pair that already reads that way. `MPCState.initial_states` and
`initial_controls` become `with_states` and `with_controls` — the old names sat confusingly
beside the new `states` property while meaning something entirely different. `Trajectory.set_states`
and `set_controls` become `with_states` and `with_controls` too: `set_` on a frozen pytree
describes something the type cannot do.

`Problem.cost(state)` replaces the free `cost`. The free `states`, `controls`, `initial_states`,
`initial_controls` are deleted.

The flat-vector layout conversions become private to the transcription layer. They are a
transcription implementation detail that leaked into the public API; `MPCState.Z` and
`.to_trajectory()` are the user-facing surface for the same information. Roughly 28 internal
call sites, three in tests.

Note this ticket renames names that tickets 19 and 20 have just finished rewriting. That
overlap is why the sequence is stacked rather than parallel — do not attempt to land this
before them.

## Acceptance criteria

- [ ] `MPCState.states` and `.controls` are properties.
- [ ] `Trajectory.states()`, `.controls()`, `.times()` are deleted; callers use `.X`, `.U`, `.t`.
- [ ] Both types expose `with_states` / `with_controls`; `initial_states`, `initial_controls`, `set_states`, `set_controls` are gone.
- [ ] `problem.cost(state)` exists; the free `cost`, `states`, `controls`, `initial_states`, `initial_controls` are deleted.
- [ ] The flat-vector layout conversions are private to the transcription layer and absent from the package's public names.
- [ ] The README snippet at the top of the overview runs verbatim.
- [ ] Full suite green with no assertion changed.
