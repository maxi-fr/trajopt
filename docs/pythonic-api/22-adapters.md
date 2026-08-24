# 22 — Adapters follow the new API

**What to build:** the simulate plant and the benchmark harness use the same public API as
everyone else. The plant integrates through trajopt's own integrators with a single code path,
and the benchmarks stop reaching into a solver's private internals to time it.

**Blocked by:** 21.

## Why

Both modules are forced in-scope by the deletions in 19 and 21 — they import the free `solve`
and a private Ipopt callback, and call the accessor methods that became properties. Beyond the
forced edits, both carry warts the earlier tickets have now made cheap to remove.

## The plant

The simulate adapter currently branches twice on model type. Once in the constructor, to pick
an integrator from the external framework when the model is continuous. Once in the dynamics
callback, to decide whether to return a next state or a derivative — because the framework's
base class treats the callback as a derivative when an integrator is configured and as a state
transition when one is not.

Both branches disappear if the plant is always discrete. Take any model, `discretize()` it at
construction (ticket 18 made that a no-op for models already discrete, so config dicts naming
continuous models keep working), configure no framework integrator, and return the next state
unconditionally. This is supported behaviour, not a workaround: the framework's base class
documents that with no integrator the callback returns the next state directly.

The plant's `integrator` parameter is kept but repurposed. It stops being a framework
integrator callable and becomes a trajopt `Integrator` forwarded to `discretize()`, with the
config key resolving a class rather than a function. Plant fidelity is a real knob — the plant
is often integrated more finely than the model the controller linearizes — and dropping the
parameter would throw it away. Repurposing also makes the plant's integrator selection spell
the same way as a `Problem`'s, which is one concept instead of two.

The framework's `dict | instance` config unions stay. That is `from_config`'s required
contract, not incidental configurability.

### Known assertion change

Two adapter tests assert on whether the plant's framework integrator is or is not `None`. With
no framework integrator ever configured, those assertions become meaningless and must be
rewritten to assert on the plant's behaviour instead. This is the second of the two assertion
changes the sequence permits — call it out in the commit message.

## The benchmarks

The timing harness imports a *private* Ipopt callback class and constructs it directly to
measure transcription setup cost. Give the Ipopt backend a public way to hand over its
assembled callback: "how long does transcription setup take" is a legitimate question to ask a
solver, and the private reach-in is the only reason the benchmark module knows Ipopt's
internals at all.

The setup-timing entry point takes either an initial-state array or an `MPCState` and branches
to pull the array out. Take the array only; callers holding a state pass its initial state,
which is exactly what the branch was doing.

The benchmark MPC loop's own continuous/discrete branch is already gone via ticket 18.

## Acceptance criteria

- [ ] The plant discretizes at construction and configures no framework integrator.
- [ ] The plant's dynamics callback has one path and returns the next state unconditionally.
- [ ] The plant's `integrator` parameter and config key accept a trajopt integrator.
- [ ] The two adapter tests asserting on framework-integrator presence are rewritten to assert on behaviour; the change is named in the commit message.
- [ ] The `from_config` `dict | instance` handling is untouched.
- [ ] The Ipopt backend exposes its transcription callback publicly; the benchmarks import nothing private.
- [ ] The setup-timing entry point takes an initial-state array only.
- [ ] Full suite green with no assertion changed beyond the two named above.
