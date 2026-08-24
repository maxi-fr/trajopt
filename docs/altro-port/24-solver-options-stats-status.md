# 24 — Solver options, stats, and termination status

**What to build:** the three value types every native solver in this series needs — the option
set you configure a solve with, the statistics a finished solve hands back, and the reason it
stopped. Nothing solves anything yet. What you can demonstrate is that constructing
`SolverOptions()` gives Altro's defaults, that a `SolverStats` round-trips through `jax.jit`
without leaving the traced world, and that every internal termination reason maps onto the
public four-value `SolverStatus` the rest of the codebase already speaks.

**Blocked by:** None — can start immediately.

## Architecture

`src/trajopt/solvers/options.py`.

**`SolverOptions` is a frozen dataclass, not an `eqx.Module`.** It is static configuration read
during tracing to pick shapes and loop bounds; making it a pytree would invite someone to trace
a tolerance. This matches how `Ipopt` / `OSQP` / `Clarabel` are already modelled — see the
decision recorded in `docs/pythonic-api/00-overview.md`.

Port Altro's names and defaults from reference §3, **minus the eleven dead options** listed in
the overview (finding F). Two of those deletions need a replacement rather than a hole:

- `bp_reg_max` is unread upstream, which is why Altro's backward pass can raise ρ forever on a
  failing Cholesky. Under `lax.while_loop` that hangs with no interrupt. Keep the option, give
  it Altro's stated `1e8` default, and **make it live** — ticket 25 uses it as the loop bound.
  This is a deliberate divergence; record it.
- `bp_reg_type` has only one working value upstream (finding H). Don't port the option; the
  `:control` behaviour is the only behaviour.

**`TerminationStatus` is an ordered `IntEnum`, and the order is load-bearing.** Finding C:
Altro compares `status > SOLVE_SUCCEEDED` and `status <= SOLVE_SUCCEEDED` to decide whether the
AL outer loop breaks and whether ALTRO runs its polish phase. Keep Altro's declaration order
exactly, including `LINESEARCH_FAIL` — which is never set by anything, but occupies ordinal 1
and therefore changes what `> SOLVE_SUCCEEDED` means if you drop it. Inside a traced loop the
status is an `int32` scalar in the carry; the comparisons are `jnp` comparisons on that scalar
and need no special handling.

**`SolverStats` is a pytree of fixed-size buffers plus counters**, because a traced loop cannot
append. Allocate each history (`cost`, `dJ`, `c_max`, `gradient`, `penalty_max`) at length
`options.iterations` and write at the counter index with `.at[i].set(...)`. Trimming to the
counter happens in the eager wrapper at the Python boundary, never inside the trace. Also carry
`dJ_zero_counter` and `ls_failed`, both of which drive control flow in later tickets.

**Public mapping.** Reuse the existing `SolverStatus` literal and sit beside
`normalize_status` in `transcription/result.py` rather than inventing a parallel vocabulary.
Reference §2's suggested table is fine as-is.

## Acceptance criteria

- [ ] `SolverOptions` is a frozen dataclass carrying Altro's live option names and defaults; every value matches `altro_jl/src/solver_opts.jl`, verified by a test that asserts the defaults literally.
- [ ] None of the eleven dead options exists as a field; `bp_reg_type` is absent and `:control` behaviour is unconditional.
- [ ] `bp_reg_max` exists, defaults to `1e8`, and its docstring says it is live here and dead in Altro.
- [ ] `TerminationStatus` is an `IntEnum` whose member order matches Altro's `@enum` declaration exactly, `LINESEARCH_FAIL` included, with a test asserting the ordinals.
- [ ] `SolverStats` is a pytree; constructing one, writing into its buffers, and reading it back survives `jax.jit` with no host callback and no shape change.
- [ ] A mapping function turns any `TerminationStatus` into the existing four-value `SolverStatus`; every enum member is covered, asserted exhaustively.
- [ ] pre-commit hooks pass
