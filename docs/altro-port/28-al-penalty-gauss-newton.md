# 28 — Augmented Lagrangian penalty in Gauss-Newton form

**What to build:** the constraint-side machinery an AL outer loop needs — per-knot, per-row duals
and penalties, the penalty cost and its gradient and Hessian in Altro's Gauss-Newton form, the
dual and penalty updates, and the violation and penalty maxima. Demonstrable without any solver:
given a trajectory and a set of duals, the penalty blocks this produces match
`Altro.alcost` / `algrad!` / `alhess!` to 1e-8, and one dual-and-penalty update step lands on the
same λ and μ as Altro's.

**Blocked by:** 24.

## Architecture

`src/trajopt/solvers/al.py`, replacing `_evaluate_knot_penalty`, `_stage_al_derivatives`,
`_term_al_derivatives`, and `_augmented_lagrangian_expansion` in `expansions.py`. Those go; there
is no shim and no coexistence, matching how tickets 18–23 handled replacement.

**The Hessian becomes `Jᵀ diag(a) J`.** The current route builds a smooth conic penalty and hands
it to `jax.hessian`. Reference §7 is right that the special-cased form is better on all three
counts, and §7.1's measurements are worth trusting: zero divergence for affine constraints (all
box bounds, `GoalConstraint`), a difference proportional to penalty × violation × curvature for
nonlinear ones that vanishes at the solution, and — the reason it matters — a Gauss-Newton block
that is PSD by construction where the full Hessian goes indefinite and stalls the backward pass in
regularization retries.

Cone mapping is clean and needs no new cone code: `ZeroCone` → Altro's `Equality`,
`NegativeOrthant` → `Inequality`. Both agree that a satisfied inequality means `c ≤ 0`, so Altro's
active-set test `(c ≥ 0) | (λ > 0)` transfers unchanged.

**Pin the dual sign convention before writing a line of it.** Finding E: three conventions are in
play. Altro's non-conic path uses `λbar = λ + μ∘c` with cost `+λ'c`. Altro's conic path uses
`λbar = λ − μ∘c`, which for an equality constraint gives `−λ'c + ½μc'c` — the opposite sign on λ.
The code being deleted here uses a third (`shifted = c + λ/μ`, projected onto the **dual** cone).
This ticket implements the **non-conic** convention, and every stored λ in `MPCState` means that.
Ticket 31 adds the conic path and must convert at its boundary, not silently reinterpret.

**Layout: padded per-knot blocks with a row mask, filled per constraint.** Altro keeps one
`ALConstraint` per constraint with a list of knot indices; Python's `BuiltConstraintList`
concatenates every constraint at a knot into one block under a single scalar μ (finding N). Adopt
Altro's grouping, because it is also what makes this traceable: for each constraint, `vmap` its
evaluation and Jacobian over the knot indices it applies to, then scatter the per-row results into
a padded `(N, p_max)` block. The `for k in range(N)` loop being deleted would unroll N times under
trace; this does not. μ becomes per-row, matching Altro, and `_parse_penalties`' per-knot scalar
goes away.

**Duals live on `MPCState` in a new `al` field** — padded λ, padded μ, and the row mask — not in
`lam` / `mu`, which keep their transcription meaning (canonical row order, dynamics and
initial-condition duals). AL has no dynamics rows to put there. The field is a pytree so warm
starts survive across MPC steps.

**Updates are pure functions returning new arrays.** Equality: `λ ← λ + μ∘c`. Inequality:
`λ ← max(0, λ + μ∘c)`. Both clamp to `±dual_max`. Penalties: `μ ← clamp(μ·penalty_scaling, 0,
penalty_max)`, applied to every row unconditionally after every outer iteration. Masked rows must
be inert in all of them.

Violation is `Π_K(c) − c` per knot, with `max_violation` the ∞-norm over knots and constraints —
Python already computes this for the transcription path; reuse it rather than writing a second.

## Julia parity

Reference §8.2 rows 10 through 14: penalty cost, gradient, and Hessian blocks per knot; the
gradient and Hessian deltas added into the cost expansion; λ after one dual update; μ after one
penalty update; `c_max` and `μ_max`.

## Acceptance criteria

- [ ] Penalty cost, gradient, and Hessian are implemented in Altro's equality and inequality forms with the Gauss-Newton Hessian `Jᵀ diag(a) J`; no `jax.hessian` of a penalty remains.
- [ ] `_evaluate_knot_penalty`, `_stage_al_derivatives`, `_term_al_derivatives`, and `_augmented_lagrangian_expansion` are deleted from `expansions.py`, along with `_parse_penalties`' per-knot scalar handling.
- [ ] μ is per-row per-knot; λ and μ live in a new `MPCState.al` field as padded arrays with a row mask, and `lam` / `mu` are untouched.
- [ ] Adding a constraint's penalty blocks costs one `vmap` per constraint, not one traced iteration per knot; a test asserts the traced computation does not scale its jaxpr size with N.
- [ ] For an affine constraint the Gauss-Newton Hessian equals the exact Hessian to 1e-12, confirming reference §7.1's first conclusion.
- [ ] Masked padding rows contribute nothing to cost, gradient, Hessian, violation, or either update.
- [ ] Dual and penalty updates match Altro exactly, including the `max(0, ·)` on inequalities, the `±dual_max` clamp, and the unconditional penalty scaling.
- [ ] The docstring states which of the three sign conventions the stored λ uses.
- [ ] Cross tests match Altro's `alcost` / `algrad!` / `alhess!` / `dualupdate!` / `penaltyupdate!` / `max_violation` / `max_penalty` to 1e-8 for a bound constraint, a goal constraint, and a nonlinear circle constraint.
- [ ] pre-commit hooks pass
