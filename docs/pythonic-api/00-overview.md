# Pythonic API refactor — Tickets 18–23

The public API was written in a Julia idiom: free functions taking the object they act on as
their first argument, with `isinstance` chains standing in for method dispatch. This sequence
moves it to methods on the owning type and replaces the dispatch chains with polymorphism.

Before:

```python
opt_state = solve(prob, state)
X = states(opt_state)
U = controls(opt_state)
```

After:

```python
opt_state = prob.solve(state)
X = opt_state.states
U = opt_state.controls
```

## Rules that apply to every ticket

**No shims.** Old and new spellings never coexist. Each ticket changes a form and all of its
call sites in the same commit. This is a departure from expand–contract, and it works because
the repo is one package with no external consumers: `src/` is ~10k lines, the suite is 32 files,
and nothing outside the repo imports `trajopt`.

**Green either side.** The full suite passes before the ticket and after it, with **no assertion
changed** — only call syntax. Any test whose assertions must change is a behaviour change: call
it out explicitly in the commit message rather than absorbing it. Two are known in advance and
are named in their tickets (18, 22).

**Stacked and ordered.** 18 → 19 → 20 → 21 → 22 → 23, merged in sequence. The dependencies are
real: 18's `discretize()` is what lets 20 delete the model-extraction helpers, and 21's renames
land on files 19 and 20 rewrite. Don't reorder to parallelise.

**Prune, don't relocate.** Where a union parameter exists only so an `isinstance` chain can
unpick it, the union goes away with the chain. Where a branch encodes real information the
types can't (see ticket 22's plant adapter), it stays.

**Commit convention.** `<type>(<scope>): <description> (Ticket NN)`, matching tickets 01–17.

## Decisions taken up front, so no ticket relitigates them

- `problem.solve(state)`, not `state.solve(problem)`. The state is data; the problem knows how
  to solve. This costs the fluent chain through `with_measurement`/`shift`, and that's accepted.
- Derived values are properties (`state.states`); stored values are plain fields (`traj.X`).
  A method that only returns a field is deleted, not converted.
- `with_` is the verb for "returns a new instance". `set_` on a frozen pytree is a lie.
- Expansion math stays in one module; the methods on `Problem`/`Objective`/`AbstractModel`
  delegate to it. Splitting ~500 lines of scan/jacfwd machinery across three files would make
  it harder to read and risks an import cycle.
- Solver backends are frozen dataclasses, not `eqx.Module`s. They hold Python config and are
  never traced; making them pytrees invites someone to try tracing an Ipopt handle.
- Native solver options stay a loose `Mapping[str, Any]` pass-through. Typing them is a
  maintenance tax against a large, version-dependent, third-party option set.

## Explicitly out of scope

- `simulate.py`'s `dict | instance` config unions. "Instance or config dict" is the contract
  the external `simulate` framework's `from_config` requires, not incidental configurability.
- `with_control_rate_penalty` stays a free function. It returns a model *and* a cost, so
  neither is naturally the receiver.
- `MPCState` gains no sequence protocol. It is a solver state — duals, warm start, timestamps —
  not a sequence of knot points; indexing it would imply the multipliers index along with it.
- `model.rollout(traj, x0=...)` keeps its `x0` override. It carries information the trajectory
  genuinely doesn't: simulate from *here*, not from the guess's first knot.
