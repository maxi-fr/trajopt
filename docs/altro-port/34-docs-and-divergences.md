# 34 — Docs, glossary, and recorded divergences

**What to build:** someone arriving at this codebase cold can find out that native solvers exist,
which one to reach for, and where they deliberately differ from Altro — without reading the
solvers. The divergences in particular are the deliverable: every one was a decision, and an
undocumented deliberate divergence is indistinguishable from a bug the next time a parity test
fails.

**Blocked by:** 33.

## Architecture

**The domain-doc infrastructure AGENTS.md describes does not exist yet.** It points at a root
`CONTEXT.md`, a `docs/adr/` directory, and `docs/agents/domain.md`; none of the three is present.
Creating a full domain model is out of scope here, but this ticket cannot record an ADR into a
directory that does not exist. Create `docs/adr/` and the first ADR; if a root `CONTEXT.md` is
wanted, that is its own task — flag it rather than improvising one.

**One ADR, covering the divergences as a set**, because they share a rationale: each one trades
literal fidelity to Altro for something this codebase needs more. Record at minimum:

- **Fully traced control flow.** Every loop is `lax.while_loop` / `lax.scan`. Consequence: not
  reverse-mode differentiable; a Cholesky failure costs a full wasted sweep rather than an early
  exit, because a `scan` cannot stop.
- **`bp_reg_max` is live here and dead in Altro.** Finding F. Altro's backward pass can raise ρ
  forever on a failing Cholesky; under `lax.while_loop` that is an unkillable hang, so the bound
  is enforced. This is a behavioural difference, not just a safety net — a problem Altro would
  grind on, this one exits.
- **Gauss-Newton penalty Hessian.** Reference §7's measurements are the justification: exact for
  affine constraints, differing only by penalty × violation × curvature for nonlinear ones, and
  PSD where the full Hessian is indefinite.
- **Dense KKT instead of QDLDL** in Projected Newton, and PN's second row-ordering convention
  distinct from `transcription/layout.py`.
- **`multiplier_projection` is a superset.** Implemented here, disabled upstream (issue #35), so
  parity requires it off on both sides.
- **Eleven dead options discarded**, `bp_reg_type` not ported.
- **Control-limited DDP is not from Altro at all** — Tassa's box-QP, verified against Clarabel.

**The reference document itself needs a pointer.** `docs/altro-jl-reference.md` is now known to
contain fourteen errors, catalogued in `docs/altro-port/00-overview.md`. Anyone reading the
reference without the corrections will reimplement those mistakes. Add a prominent note at the top
of the reference pointing at the corrections list — or fold the corrections into the reference
directly. Either is fine; leaving the two documents disagreeing silently is not.

**Public surface.** `__init__.py` stays empty per the repo convention — imports come from the
module. What needs writing is the README entry: what `ILQR`, `ALTRO`, and the control-limited
variant are for, when to prefer a native solver over the Ipopt adapter, and the one-line swap that
switches between them. Capitalize glossary terms per the repo convention.

**A benchmark comparison closes the loop.** `benchmarks.py` already times the transcription
backends against CasADi. The native solvers' reason for existing is speed on the MPC path; include
the numbers rather than claiming it.

## Acceptance criteria

- [ ] `docs/adr/` exists and holds an ADR recording every divergence listed above, each with its rationale and its observable consequence.
- [ ] `docs/altro-jl-reference.md` points at the corrections in `docs/altro-port/00-overview.md`, or incorporates them; the two no longer disagree silently.
- [ ] The README documents `ILQR`, `ALTRO`, and the control-limited variant: what each is for, when to prefer them over the transcription adapters, and how to swap.
- [ ] Every public solver type and phase function has a one-line docstring; those with array parameters name the shapes, per the repo convention.
- [ ] `__init__.py` is still empty.
- [ ] `benchmarks.py` times a native ALTRO solve against the Ipopt adapter on the same problem, and the numbers are recorded.
- [ ] The absence of a root `CONTEXT.md` is flagged as follow-up rather than improvised.
- [ ] pre-commit hooks pass
