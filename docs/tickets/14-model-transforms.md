# 14 — Model transforms

**What to build:** Two mechanical wrappers that turn a model into a different model. The first
lets a caller penalize control rate without breaking anything; the second linearizes a model
about a reference trajectory.

The control-rate transform exists because of an invariant, not because of convenience. A cost
term coupling consecutive knot points makes the cost Hessian block-tridiagonal rather than block
diagonal. An interior-point solver absorbs that silently as extra nonzeros, so it appears to
work — but differential dynamic programming and its relatives cannot represent it at all, and
the failure would surface much later, inside a Riccati recursion. Augmenting the state with the
previous control converts the coupled term into an ordinary stage cost and preserves stage
separability, at the price of extra state dimensions.

Shipping this transform is what makes it possible to say no to coupled cost terms without saying
no to control-rate penalties.

**Blocked by:** 04 — Integrators and rollout.

**Spec:** Section 1 (invariants, Markovian stage decoupling), section 14 (model transforms).

## Acceptance criteria

- [x] A transform augments a model's state with the previous control and exposes the resulting
      dimensions correctly
- [x] A control-rate penalty expressed through the augmented model produces the same trajectory
      cost as the equivalent coupled penalty computed by hand
- [x] The augmented model's cost Hessian is verified to be block diagonal, not block tridiagonal
- [x] The transform composes with the existing integrators and rollout without special-casing
- [x] A transform linearizes a model about a reference trajectory, producing stacked state and
      control Jacobians along the horizon
- [x] Both transforms work on Euclidean models; behaviour on rigid-body models is either
      supported or explicitly documented as unsupported
