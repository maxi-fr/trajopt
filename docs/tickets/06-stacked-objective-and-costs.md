# 06 — Stacked objective and cost evaluation

**What to build:** The ability to attach a cost to a trajectory and evaluate it, along with its
gradient and Hessian, in one batched pass over the horizon. A caller builds an LQR objective or
a tracking objective, hands it a trajectory, and gets a scalar cost that matches Julia to full
precision.

The structural decision this ticket implements: the objective holds one stage cost and one
terminal cost, homogeneous in type, with parameters stacked over the horizon. It is not a list
of per-knot cost objects. A list of heterogeneous Python objects cannot be traced, and stacked
parameters preserve every capability the list actually provided — constant weights are a stacked
constant, time-varying tracking weights are a stacked array.

**Blocked by:** 03 — Trajectory storage and the model interface.

**Spec:** Section 8 (cost functions and objectives), section 1 (invariants — in particular why
control-rate penalties are not implemented here).

## Acceptance criteria

- [x] The objective stores stage-cost parameters stacked over the horizon and terminal-cost
      parameters separately
- [x] Diagonal and dense quadratic forms are both supported, with the diagonal form storing
      weights as vectors rather than matrices
- [x] LQR tracking construction produces the correct linear and constant terms from a goal state
      and goal control
- [x] A time-varying tracking objective can be built from a reference trajectory, and its
      reference can be updated from a new reference without rebuilding the objective
- [x] A generic user-supplied cost callable is supported and differentiated by autodiff with no
      special-case machinery
- [x] Cost evaluation over a trajectory is a single batched pass plus a reduction, not a Python
      loop
- [x] Analytic Hessian inversion is implemented using the diagonal and block-diagonal shortcuts
- [x] No interface admits a cost term coupling consecutive knot points; such penalties are the
      subject of ticket 14
- [x] Cross-verification covers scalar cost, gradient, Hessian, and inverted Hessian for the
      diagonal, dense, LQR, and tracking variants, at `1e-14` for values and `1e-12` for
      derivatives
