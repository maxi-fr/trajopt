# ADR 0004: The two solver tiers linearize in different coordinates

## Status

Accepted, with a known limit. The asymmetry is recorded here rather than resolved.

## Context

The library has two solver tiers, and each linearizes the dynamics for itself.

The stagewise tier — iLQR, AL, ALTRO, PN — goes through `Model.linearize`, which delegates to
`_linearize_about` in `src/trajopt/models/transforms.py`. That function does not return the raw
step Jacobians. It sandwiches them in the error-state maps:

```python
Gk = discrete_model.errstate_jacobian(xk)
G_next = discrete_model.errstate_jacobian(x_next)
A_bar = G_next.T @ Ak_raw @ Gk
B_bar = G_next.T @ Bk_raw
```

The result has shape `(ne, ne)` and `(ne, m)`, where `ne` is the error-state dimension.

The NLP tier — the five `eval_*` callbacks, consumed by Ipopt and by the Quadratic Subproblem
derived from them (ADR: the QP is the NLP's derived form) — does not use the error-state maps at
all. `grep -rn '\.ne\b' src/trajopt/transcription/` returns nothing. The Primal Vector is
`N * n + (N - 1) * m` entries of full state coordinates, and `eval_jac_g`'s defect rows carry the
raw `(n, n)` Jacobian.

For every Euclidean model these coincide: `EuclideanModel` sets `ne = n` and its
`errstate_jacobian` is the identity, so `Gᵀ A G` is `A`. Pendulum, Cartpole, Dubins and
`AffineModel` are all in this class, and for all of them the two tiers linearize the same object
in the same coordinates.

`Quadrotor` is not. It derives from `RigidBody`, whose state carries a JPL unit quaternion:
`n = 13`, `ne = 12`. For it the two tiers produce Jacobians of *different shapes*, describing the
same dynamics in different coordinate systems.

This is not hypothetical. `Quadrotor` is driven through the NLP tier today, in
`test/cross_verification/` and in `examples/04_quadrotor.py`.

## Decision

Both tiers keep their current coordinates. Neither is converted to match the other.

The stagewise tier must use error coordinates. Its whole method is a local quadratic model with an
unconstrained Newton step, and a step taken in full quaternion coordinates leaves the unit sphere.
The three-parameter error state is what makes the step well-posed, and the `Gᵀ A G` sandwich is
how the model reaches those coordinates.

The NLP tier must use full coordinates. It hands a flat Primal Vector to a general-purpose NLP
solver that owns its own iterate and knows nothing about manifolds. There is no seam at which an
error-coordinate parameterization could be re-linearized each iteration without reimplementing the
solver's step. The manifold is expressed to it the way any other requirement is — as a constraint.
`examples/04_quadrotor.py` does exactly that with `QuatVecEq` for terminal attitude.

So the tiers are not inconsistent by oversight. Each takes the only coordinate choice its own
method admits, and the divergence is a property of the two methods rather than a defect in either.

## Consequences

- **The agreement test is limited to Euclidean models, permanently.**
  `test_defect_jacobian_matches_model_linearization` compares the NLP's defect rows against
  `Problem.linearize` and passes on Cartpole. It would fail on `Quadrotor` — on shape, before it
  reached the numbers — and that failure would be correct. The test's docstring states the
  restriction. It cannot be lifted by fixing the test.

- **The cross-tier guarantee is weakest exactly where the dynamics are hardest.** The test exists
  to catch a second linearization drifting from the one Ipopt is handed. For rigid-body models
  that guard does not apply, and nothing else checks the two tiers describe the same system.

- **A trajectory does not transfer between tiers for a rigid body without a coordinate change.**
  Seeding ALTRO from an Ipopt solve — the pattern the closed-loop characterization test uses — is
  a Primal Vector handoff, which is coordinate-free and so unaffected. Anything handing over
  *Jacobians or duals* across the tiers is not, and would need the `G` maps applied or removed.
  No code does this today; it is a trap for code that tries.

- **`ne` is load-bearing but invisible from the NLP tier.** A reader in `src/trajopt/transcription/`
  sees only `n` and can reasonably conclude the error state is not a concept the library has.

## Alternatives rejected

**Transcribe the NLP tier in error coordinates.** The operating point would have to be re-chosen
and the problem re-transcribed every iteration, because error coordinates are defined relative to
a reference the solver is continuously moving away from. Ipopt owns its iterate and offers no hook
for that. It would also mean the QP could no longer be the NLP's derived form, since the two would
no longer share a Primal Vector.

**Drop error coordinates from the stagewise tier and constrain the norm instead.** This makes the
Newton step ill-posed rather than merely inexact: the quaternion's radial direction is a null
direction of the dynamics, so the unconstrained backward pass would be solving a singular system.
Error coordinates exist to remove that direction, not to improve conditioning within it.

**Convert at the seam, on demand.** A `G`-map adapter is straightforward to write and would let a
Jacobian cross tiers. It is not written, because nothing needs it yet and an unused conversion
between two coordinate systems is the kind of code that goes silently stale. This is the natural
change to make when a cross-tier handoff of derivatives is actually required.
