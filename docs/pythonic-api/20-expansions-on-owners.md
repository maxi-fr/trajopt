# 20 — Expansions answer to their owners

**What to build:** ask the object that holds the information for its expansion —
`problem.cost_expansion(traj)`, `model.dynamics_expansion(traj)` — instead of calling a free
function that accepts several unrelated types and works out at run time which one it got.

**Blocked by:** 18.

## Why

Three expansion entry points each take a union: cost expansion accepts a problem *or* an
objective, dynamics expansion accepts a problem *or* a model, augmented-Lagrangian expansion
accepts a problem *or* a built constraint list *or* an unbuilt one. Behind them sit three
private helpers whose entire job is to unpick those unions — and one of them silently returns
`None` when it recognises nothing, so a wrong type produces a confusing failure further down
rather than at the call.

Giving each owner the method makes the union impossible to *express*, which is the point. The
helpers are then unreferenced and deleted, not ported.

## What changes

`Problem` gains `cost_expansion(traj)`, `dynamics_expansion(traj)` and
`augmented_lagrangian_expansion(traj, expansion, lam, mu)`, each delegating to the owner that
actually holds the data. `Objective` gains `cost_expansion(traj, model=None)`. `AbstractModel`
gains `dynamics_expansion(traj)`. `BuiltConstraintList` gains the augmented-Lagrangian
expansion; an unbuilt `ConstraintList` reaches it through `.build()` from ticket 18, so the
third arm of that union resolves by polymorphism rather than by branch.

The implementations stay where they are, as private module functions, with the new methods as
one-line delegates. Relocating ~500 lines of scan and `jacfwd` machinery into three modules
that currently know nothing about expansions would scatter a body of math that reads as a unit,
and would put the dynamics base module and the expansion module in a live import cycle. The
win being bought here is the deleted unions; delegation buys it in full.

Names keep the `_expansion` noun form because they return an `Expansion` — `expand_cost` would
imply mutation. Spell out `augmented_lagrangian_expansion` rather than abbreviating; it has
seven call sites total and terseness buys nothing.

`linearize_about(model, ...)` becomes `model.linearize(traj)` and loses its
`Trajectory | array` overload along with the separate reference-controls argument that only
existed to feed the array arm. Callers holding loose arrays construct a `Trajectory` — which is
what the array arm was reconstructing internally anyway. Six test call sites.

## Acceptance criteria

- [ ] `Problem`, `Objective`, `AbstractModel` and `BuiltConstraintList` expose the expansion methods listed above.
- [ ] The three private extraction helpers are deleted; no expansion entry point takes a union type.
- [ ] Expansion implementations remain in their current module as private functions; the methods are delegates.
- [ ] No import cycle is introduced between the dynamics, costs, constraints and expansion modules.
- [ ] `model.linearize(traj)` accepts a `Trajectory` only; `linearize_about` is deleted.
- [ ] Full suite green with no assertion changed.
