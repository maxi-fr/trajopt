# 32 — Projected Newton polish phase

**What to build:** the second phase of ALTRO. Given a trajectory that the AL phase drove to rough
feasibility (`c_max` around 1e-3), project it onto the constraint manifold to machine precision
with an active-set KKT solve. Demonstrable on its own: hand it a trajectory with a known
violation, and the violation drops by orders of magnitude in two or three steps while the cost
barely moves.

**Blocked by:** 29.

## This is a different formulation from everything before it

Finding L, and it is the single most important thing to internalize before writing code.
Reference §6 and §8.2 present Projected Newton as "one more phase" of the same solver. It is not.
The AL-iLQR phase is **single shooting**: the decision variables are controls, and the dynamics
hold by construction because you roll them out. Projected Newton is **multiple shooting**: it
stacks states and controls into one primal vector and adds the dynamics as explicit equality
constraints, then solves a KKT system over the whole horizon at once.

So this phase has its own primal layout, its own dual layout, and its own row ordering — a second
convention living alongside `transcription/layout.py`'s canonical order. That is the decision
taken up front, and the cost of it is that the two orderings must never be confused. Name the
types so they cannot be: nothing here should be called `Z` or `lam` without a qualifier that says
which layout it belongs to. Ticket 34 records the convention in an ADR.

## Architecture

`src/trajopt/solvers/pn.py`, self-contained assembly.

**Dense KKT, not sparse QDLDL.** Altro builds an upper-triangular sparse KKT matrix and factors it
with QDLDL. For the fixed small `n` and `m` this port targets, a dense factorization is
numerically equivalent and vastly simpler, and it is what makes the whole thing traceable — a
sparse pattern that changes with the active set cannot be a static shape. This is a declared
divergence; note it in the solver docstring.

**The active set has a static shape, which is the central traced-implementation problem.** The
active set changes between iterations, and JAX needs fixed shapes. Assemble the KKT system at full
size with inactive rows masked — zero the row and put a 1 on the diagonal, so inactive multipliers
solve to zero and the factorization stays well-posed. Do not try to resize.

**The loop structure**, from `pn_solve.jl`:

- Outer: `while count <= n_steps && viol > ϵ_feas`. Note this permits **three** projection solves
  at the default `n_steps = 2`, not two — finding M.
- Middle: iterative refinement, up to a hard-coded `max_refinements = 10`, exiting when the
  violation clears tolerance or when `log10(viol)/log10(viol_prev)` falls below `r_threshold`.
- Inner: a violation-based line search halving α up to 10 times, accepting the first step that
  reduces the violation, and **not** updating the active set while it searches.

`max_refinements` and the inner line-search cap are hard-coded constants upstream, not options
(finding M). Keep them constants here too rather than inventing configurability.

**`ρ_primal` regularizes the Hessian blocks** before factorization. `ρ_chol` and `ρ_dual` are dead
upstream and are not ported (finding F).

**`multiplier_projection` is a no-op upstream.** Altro's implementation is commented out and the
call site returns `Inf` (issue #35). Port the projection properly, gated behind the option with
Altro's default of `True` — which makes this port a superset. Any parity test against Julia must
therefore run with the option **off on both sides**; there is nothing on the Julia side to compare
the projection against.

## Julia parity

Reference §8.2 row 16: the active set, the KKT step, and the violation reduction per iteration,
with `multiplier_projection` off on both sides. The active set is the part worth comparing
carefully — the KKT step follows from it.

## Acceptance criteria

- [ ] The PN primal and dual layouts are self-contained, distinctly named, and documented as a second row-ordering convention distinct from `transcription/layout.py`.
- [ ] The KKT system is assembled dense at full size with inactive rows masked; shapes are static across iterations.
- [ ] Given a trajectory with a known violation, PN reduces `max_violation` by at least three orders of magnitude within `n_steps` projection solves.
- [ ] The outer loop permits three projection solves at `n_steps = 2`, matching upstream.
- [ ] Iterative refinement exits on either the tolerance or the convergence-rate criterion; both exits are reached by a test.
- [ ] The inner line search halves α up to ten times, accepts on violation reduction, and does not update the active set while searching.
- [ ] `ρ_primal` regularizes the Hessian blocks; `ρ_chol` and `ρ_dual` do not exist.
- [ ] `multiplier_projection` is implemented and gated, defaults to `True`, and its docstring records that Altro's own version is disabled.
- [ ] Cross tests match Altro's active set and per-iteration violation reduction with `multiplier_projection` off on both sides.
- [ ] The dense-KKT divergence from Altro's QDLDL is stated in the solver docstring.
- [ ] pre-commit hooks pass
