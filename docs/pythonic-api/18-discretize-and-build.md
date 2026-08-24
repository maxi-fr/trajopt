# 18 — `discretize()` and `build()` replace type dispatch

**What to build:** every dynamics model can answer "give me your discrete form" and every
constraint list can answer "give me your built form", so callers stop asking what type they
are holding. A `Problem` resolves its model once at construction and no longer carries a
separate integrator for someone else to remember to apply.

**Blocked by:** None — can start immediately.

## Why

The same continuous/discrete `isinstance` chain is written out in six places: the rollout
helper, the problem's model extractor, twice in the model transforms, in the explicit dynamics
constraint, and in the benchmark MPC loop. Each one asks the model what it is and then does one
of two things. That is method dispatch spelled by hand. One method on each subclass deletes all
six, and deletes the `TypeError` branch each of them carries for a state the type system
already rules out.

`Problem` is the sharpest case. It stores `model` and `integrator` as separate fields, and the
*only* reader of `integrator` in the entire repo is the extractor helper. So the field exists
purely to be recombined later, and until it is recombined, `problem.model` is a half-configured
object: asking it to roll out silently uses a default RK4 even when the problem was built with
`Euler()`. Discretizing at construction removes the trap and the field together.

## What changes

`AbstractModel` declares `discretize(integrator=None) -> DiscreteDynamics`. `DiscreteDynamics`
returns `self`. `ContinuousDynamics` wraps itself with the given integrator, defaulting to RK4.
Every hand-written branch that was doing this is replaced by the call.

`BuiltConstraintList` gains `build() -> Self` returning itself, mirroring the same trick.
`Problem.__init__` then reduces to a sentinel default for `None` followed by an unconditional
`.build()`, with the three-way isinstance chain and its `TypeError` gone.

`Problem` stores `self.model = model.discretize(integrator)` and drops the `integrator` field.
The model-extraction helper in the problem module goes away, and with it the `Problem`-level
`rollout`, whose 2×2 `isinstance` matrix over (Problem | model) × (MPCState | Trajectory) was
that helper's only other caller. Callers roll out through the model:
`problem.model.rollout(state.to_trajectory())`.

Note the behaviour that becomes explicit rather than disappearing: the old problem-level
rollout started from `state.x0`, while rolling out a trajectory starts from its first knot.
Those coincide after `with_measurement` or `shift` but are not guaranteed equal, so callers
that mean the former write `problem.model.rollout(traj, x0=state.x0)` and say so.

## Two branches that must survive

Not every `isinstance` here is dispatch in disguise. The implicit collocation constraint
genuinely requires a continuous model — you cannot ask a discrete model for a derivative —
so its type check is a real precondition and stays. Likewise the explicit dynamics constraint
keeps accepting either form; only the branch that *discretizes* is replaced by the call.

## Known assertion change

The CasADi cross-verification harness looks up its baseline function by `isinstance` on the
concrete model class (`Cartpole`, `DubinsCar`, `Pendulum`, `Quadrotor`) reached through
`problem.model`. Once the problem stores the discretized form, that lookup sees the wrapper.
It must reach through to the wrapped continuous model. This is the price of the decision and
was accepted knowingly; keep the fix to the lookup itself.

## Acceptance criteria

- [ ] `AbstractModel.discretize(integrator=None)` exists; `DiscreteDynamics` returns `self`, `ContinuousDynamics` wraps with the given integrator or RK4.
- [ ] `BuiltConstraintList.build()` returns `self`.
- [ ] All six hand-written continuous/discrete discretization branches are replaced by `.discretize()` calls; no `TypeError` for "neither continuous nor discrete" remains at those sites.
- [ ] `Problem` stores an already-discrete model and has no `integrator` field.
- [ ] `Problem.__init__` has no isinstance chain over constraint types and no `TypeError` branch for them.
- [ ] The problem module's model-extraction helper and its `rollout` are deleted.
- [ ] The implicit collocation constraint still rejects discrete models.
- [ ] CasADi harness resolves its baseline through the wrapped continuous model; cross-verification parity tolerances are unchanged.
- [ ] Full suite green with no assertion changed other than the CasADi lookup noted above.
