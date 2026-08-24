# 31 — Conic penalty path (`use_conic_cost`)

**What to build:** the generic conic augmented Lagrangian, so constraints in cones other than the
zero cone and the negative orthant — second-order cone constraints in particular — get a correct
penalty instead of being forced through the equality/inequality special cases. Turned on with
`use_conic_cost=True`, off by default, exactly as in Altro. Demonstrable on a second-order cone
constraint that the ticket-28 path cannot express, and on an equality constraint where both paths
must reach the same solution by different routes.

**Blocked by:** 28.

## Architecture

`src/trajopt/solvers/al.py`, as an alternative penalty form selected by the option — not a
separate solver.

**The sign convention flips, and this is the trap.** Finding E: the non-conic path from ticket 28
uses `λbar = λ + μ∘c` with cost `+λ'c`; the conic path uses `λbar = λ − μ∘c`, which for an equality
constraint reduces to `−λ'c + ½μc'c`. Altro presents these as one option toggle, but they store λ
with opposite signs. Two consequences: a λ warm-started under one setting is wrong under the
other, and a parity test that switches the flag on only one side will look like an algebra bug.
Either convert λ at the boundary when the option changes, or refuse to change the option on a
state that already carries duals — decide and enforce it, do not leave it implicit.

**Reference §5.3's gradient formula is wrong; use the code.** Finding D: the reference gives
`grad = −∇c' ∇Π' λs`, copied from Altro's docstring. `algrad!` scales the Jacobian by μ before
applying the projection Jacobian, giving `grad = −∇c' Iμ ∇Π' λs`. The docstring and the code
disagree upstream. The **Hessian** formula in §5.3 is correct as written —
`∇c' Iμ (∇²Π(λs) + ∇Π' Iμ⁻¹ ∇Π) Iμ ∇c` — and note it includes a second-order projection term,
unlike the Gauss-Newton forms in ticket 28.

**Watch the hidden ordering dependency.** In Altro, `alhess!` reads `jac_scaled`, which is only
populated by `algrad!`. Calling the Hessian without the gradient first silently uses a stale
buffer. In a pure functional port this cannot happen — but only if the scaled Jacobian is computed
where it is used rather than passed between functions as an implicit contract. Compute it in both.

**Cone code already exists.** `cones.py` has `project`, `project_dual`, `jacobian`, and `hessian`
for `ZeroCone`, `NegativeOrthant`, `PositiveOrthant`, `IdentityCone`, and `SecondOrderCone`.
Altro's `∇²projection!` maps onto `AbstractCone.hessian`. Nothing new belongs in `cones.py` unless
a parity test proves a formula there is wrong.

**Dual update follows the same flipped convention:** `λ ← Π_{K*}(λ − μ∘c)`. Penalty update is
unchanged from ticket 28 — Altro applies the same scaling to all cones.

## Julia parity

Reference §8.2 rows 10 through 13 again, with `use_conic_cost` set on both sides: penalty cost,
gradient, Hessian, and the dual update, for a second-order cone constraint and for an equality
constraint. The equality case is the one that catches the sign error, because it is the case where
both paths exist and must disagree in exactly the documented way.

## Acceptance criteria

- [ ] Conic penalty cost, gradient, and Hessian are implemented from Altro's **code**, with the `Iμ` factor in the gradient that reference §5.3 omits, and match Julia to 1e-8 on a second-order cone constraint.
- [ ] The Hessian includes the second-order projection term and is verified against a finite-difference Hessian of the conic penalty.
- [ ] The scaled Jacobian is computed where used; no function depends on another having run first.
- [ ] Switching `use_conic_cost` on a state carrying duals either converts λ or raises; the chosen behaviour is tested and documented.
- [ ] On an equality constraint, both penalty paths converge to the same KKT point from the same start, with the λ sign difference asserted explicitly rather than glossed.
- [ ] The dual update uses `λ ← Π_{K*}(λ − μ∘c)` and matches Julia to 1e-8.
- [ ] `use_conic_cost` defaults to `False`; every ticket-28 test passes unchanged with the default.
- [ ] pre-commit hooks pass
