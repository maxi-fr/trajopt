# 30 — Control-limited DDP: box-QP backward pass

**What to build:** control bounds satisfied exactly at every iteration, with no outer loop. Where
ticket 29 penalizes bound violations and drives them down over successive AL iterations, this
solves a bound-constrained QP for the feedforward step inside the backward pass, so every rolled-
out control is feasible by construction. Demonstrable side by side with 29 on the same
bound-constrained cartpole: same problem, no penalty parameters, controls on the bound rather than
near it.

**Blocked by:** 27, 29.

## This one has no Altro oracle

Altro does not implement it. The algorithm is Tassa, Erez, and Todorov's control-limited
differential dynamic programming — the projected-Newton box-QP from `boxQP.m` in their published
release. Everything else in this series is verified against `altro_jl/`; this is verified against
**Clarabel**, which is already a dependency: solve the same subproblem

`
min ½ δuᵀ Quu δu + Quᵀ δu    s.t.   u_lo − ū ≤ δu ≤ u_hi − ū
`

with Clarabel and compare the minimizer and the free/clamped index set. That is a genuinely
independent oracle at exactly the level where mistakes hide, and it needs no new dependency and no
second vendored source. The full-trajectory behaviour is then checked for the properties the
algorithm promises rather than against a reference implementation.

Because there is no reference to defer to, be explicit in the ticket's commit message about which
variant of the projected-Newton box-QP you implemented — they differ in the free-set update rule
and the line search.

## Architecture

`src/trajopt/solvers/boxqp.py`, with `ilqr.py` gaining a way to select which solve computes
`(K, d)` from `(Quu_reg, Qux_reg, Qu)`.

**The box-QP is itself an iterative active-set method, so it is a `while_loop` inside the backward
pass's `scan`.** Nested traced loops are fine but compile slowly; keep the inner iteration count
bounded and the inner state small — free-set mask, δu, and a convergence flag. Expect this to be
the most expensive thing to compile in the whole series and measure it.

**The feedback gain is not the unconstrained gain.** Rows of `K` corresponding to clamped controls
are zeroed: a clamped control does not respond to state deviation. This is the part most easily
got wrong and the part that most changes closed-loop behaviour, so test it directly rather than
only through trajectory cost.

**Routing.** Box-QP handles `ControlBound` only. State bounds and every other constraint still go
through the AL outer loop from ticket 29, exactly as in Tassa's formulation. So the selection is
not "box-QP instead of AL" but "box-QP for control bounds, AL for the rest" — a bound-constrained
problem with a goal constraint uses both. Decide the routing at build time from the constraint
list, not at run time from traced values, and raise clearly when box-QP is requested for a problem
whose control bounds vary per knot in a way the implementation does not support.

**The regularization retry from ticket 25 still applies.** A box-QP on an indefinite `Quu` is no
better posed than a Cholesky on one; keep the ρ loop and `bp_reg_max` bound around it.

## Acceptance criteria

- [ ] A box-QP solver returns the minimizer and free/clamped mask for a bound-constrained QP, and matches Clarabel on the same problem to 1e-8 across randomized `Quu`, `Qu`, and bounds, including cases where every control clamps and where none does.
- [ ] With bounds wide enough to be inactive, the box-QP path reproduces the unconstrained backward pass from ticket 25 to 1e-10.
- [ ] Rows of `K` for clamped controls are zero, asserted directly.
- [ ] A bound-constrained cartpole solves with every rolled-out control inside its bounds at every iteration, not just at convergence.
- [ ] The same problem solved through ticket 29's AL path and through this one reach comparable final cost; the comparison is recorded in the test as documentation of the two approaches.
- [ ] Control bounds route to box-QP while other constraints continue through the AL outer loop, demonstrated on a problem carrying both.
- [ ] Requesting box-QP for an unsupported bound structure raises at build time with a message naming the problem.
- [ ] The regularization retry and `bp_reg_max` bound still apply; a test with an indefinite `Quu` terminates.
- [ ] Compile time for the nested traced loop is measured and recorded in the test suite.
- [ ] Targeted checks green. No full-suite run and no `pre-commit --all-files` gate
