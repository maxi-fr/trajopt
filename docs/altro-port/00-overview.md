# Native iLQR / ALTRO port — Tickets 24–34

Ports the iLQR, augmented-Lagrangian, and ALTRO solvers from `altro_jl/` into `trajopt` as
native JAX solvers, alongside a control-limited DDP backward pass that Altro does not have.
The algorithm and default-parameter oracle is [`docs/altro-jl-reference.md`](../altro-jl-reference.md),
**corrected** by the findings recorded below — where the two disagree, the finding wins,
because each one was verified against `altro_jl/src/` at the pinned commit.

## Decisions taken up front, so no ticket relitigates them

**Fully traced.** Every solver loop is a `lax.while_loop` or `lax.scan`; the whole solve is one
jittable, vmappable function of pytree state. `SolverOptions` is static config, never traced.
This is not a style choice — it is what makes a native solver worth having over the Ipopt
adapter, and it constrains every ticket below.

**Not reverse-mode differentiable.** `lax.while_loop` has no reverse rule and we accept that.
Forward mode works. If gradients through a solve are ever needed, the fix is a `custom_vjp` on
the fixed point, added without touching the inner loops. No ticket here builds toward it.

**Six modules under `src/trajopt/solvers/`:** `options.py`, `ilqr.py`, `boxqp.py`, `al.py`,
`pn.py`, `altro.py`. Not the nine-file mirror of Altro's split that reference §8.3 asks for —
§8.2's phase-by-phase parity tests need named importable *functions*, which these give, not one
module per function. Every module holds a distinct algorithm phase with its own carry type;
none is a single-function file.

**Thin eager wrapper over a traced core.** Each solver is a frozen dataclass satisfying the
existing `Solver` protocol in `transcription/result.py`. `.solve()` calls the jitted core, then
converts the traced status int and the fixed-size stats buffers into `success` / `message` /
`info` at the boundary. Swapping `ILQR()` for `Ipopt()` stays a one-word change. The traced core
stays separately importable for callers who want to jit or vmap a whole solve.

**AL duals get their own `MPCState` field.** `lam` / `mu` keep their transcription meaning
(canonical row order, dynamics and initial-condition duals included). AL's λ/μ are per-knot,
per-constraint-row, and have no dynamics rows to fill, so they live in a new `al` field as
padded arrays with a row mask. Warm-starting duals across MPC steps is most of why AL suits
MPC; it is not optional.

**Global AL parameters, no per-constraint overrides.** Altro's `ConstraintOptions` lets each
constraint override `penalty_initial`, `penalty_scaling`, `penalty_max`, `dual_max`, and
`use_conic_cost`. We port one global set. Per-constraint overrides are configurability nobody
asked for, and they force the padded-row layout to carry per-constraint parameter blocks.

**Effective tolerances are threaded as values.** Altro's `set_tolerances!` mutates the shared
options object mid-solve, and `altro_solve.jl` mutates `constraint_tolerance` on top of that.
Frozen options plus an explicit per-iteration tolerance value in the carry.

**Dead options are discarded.** See finding F. Eleven of Altro's options are never read in
`src/`. Reference §3 names four; the full list is `iterations_inner`, `bp_reg`, `bp_reg_max`,
`square_root`, `save_S`, `static_bp`, `reuse_jacobians`, `active_set_tolerance_al`, `ρ_chol`,
`ρ_dual`, `solve_type`. Plus the logging-only options.

**Projected Newton assembles its own layout.** Not built on `transcription/`. It carries a
second row-ordering convention, documented in ticket 34's ADR.

## Verified corrections to `docs/altro-jl-reference.md`

Each was checked against `altro_jl/src/` at commit `4864df2`. Tickets cite these by letter.

- **A.** Reference §5.5 renders the AL outer-loop convergence check as an ordered first-match
  list. `al_solve.jl` uses four independent `if`s with no early return, so a later check
  overwrites `status`. Converging on the same iteration that exhausts `iterations_outer` exits
  `MAX_ITERATIONS_OUTER`, not `SOLVE_SUCCEEDED`. Only the **iLQR** `evaluate_convergence` is
  genuinely ordered — it returns.
- **B.** The `kickout_max_penalty` branch is broken upstream: `solver.stats.penalty_max[i]`
  references an undefined `i` (should be `iter`). It throws the instant the flag is true, and
  `altro_solve.jl` sets it true whenever `projected_newton_tolerance < 0`. Untested upstream.
- **C.** Reference §2 says `LINESEARCH_FAIL` and `COST_INCREASE` are "set but never compared".
  `LINESEARCH_FAIL` is never *set* anywhere — it appears only in the `@enum` declaration.
  `COST_INCREASE` **is** set by `forwardpass!` and **is** compared, through the enum's *ordinal*:
  `status > SOLVE_SUCCEEDED` in `al_solve.jl`, `status <= SOLVE_SUCCEEDED` in `altro_solve.jl`.
  Declaration order is load-bearing control flow.
- **D.** Reference §5.3's conic gradient `−∇c' ∇Π' λs` is missing a factor. `algrad!` scales the
  Jacobian by μ first, giving `−∇c' Iμ ∇Π' λs`. The reference copied Altro's docstring, which
  disagrees with the code beneath it. The conic *Hessian* formula in §5.3 is correct.
- **E.** The conic path uses the opposite dual sign convention. Non-conic: `λbar = λ + μ∘c`,
  cost `+λ'c`. Conic: `λbar = λ − μ∘c`, reducing for an equality constraint to `−λ'c + ½μc'c`.
  Flipping `use_conic_cost` flips the sign of stored λ. The current Python
  `_evaluate_knot_penalty` uses a *third* convention (`shifted = c + λ/μ`, projected onto the
  **dual** cone).
- **F.** Reference §3's dead-option list is incomplete — full list above. The dangerous omission
  is `bp_reg_max`: because it is never read, Altro's backward pass has **no escape from a
  repeatedly-failing Cholesky**. It restarts at `k = N-1` and raises ρ forever. Under
  `lax.while_loop` that is an unkillable hang, so the bound is mandatory here.
- **G.** Reference §4.2 writes `K = −Quu⁻¹ Qux`. It is `Quu_reg⁻¹`, the regularized matrix, for
  both `K` and `d`. Identical only while ρ = 0.
- **H.** `bp_reg_type = :state` is broken upstream (`Qux_reg .= Qux`, `Qux` undefined). Only
  `:control` executes, so porting the option ports a one-valued knob.
- **I.** Reference §6's backup check omits its guard: the code also requires
  `status <= SOLVE_SUCCEEDED`, so a `MAX_ITERATIONS_OUTER` exit is never upgraded to success even
  when PN drove the violation under tolerance. §6 also omits that `c_max` is read from the AL
  stats *cache* when `iterations > 1` and recomputed only otherwise.
- **J.** Reference §4.3 omits the guard-exhaustion exit. If all 20 rollouts trip the
  state/control limit, `continue` skips the `i == max_iters` block: `ls_failed` stays false, `J`
  stays `Inf`, and the function exits via `J > J_prev` → `COST_INCREASE` → `NaN`.
- **K.** Reference §8.1's environment plan does not work. `trajopt_jl/Project.toml` is a *package*
  project (`name = "TrajectoryOptimization"`), so `Pkg.develop`ing Altro into it makes Altro a
  dependency of TrajectoryOptimization — a cycle. And "no network fetch at test time" is false:
  `trajopt_jl/Manifest.toml` contains none of Octavian, QDLDL_jll, SolverLogging, RobotZoo,
  Crayons, Formatting, TimerOutputs, Interpolations, BenchmarkTools.
- **L.** Reference §6 and §8.2 present Projected Newton as "one more phase". It is a different
  formulation: its own stacked primal `Zdata` and dual `Ydata` with **dynamics as explicit
  equality constraints**, i.e. multiple shooting, not the shooting formulation AL-iLQR uses.
- **M.** `n_steps = 2` with `while count <= max_projection_iters` permits **three** projection
  solves. `max_refinements = 10` and the 10-step inner line search are hard-coded constants, not
  options.
- **N.** Reference §7.3 item 1 understates the layout gap. Altro's μ is per-constraint *and*
  per-row with per-constraint option overrides; Python's `BuiltConstraintList` concatenates every
  constraint at a knot into one row block under one scalar μ.

What held up under checking: all §3 defaults, §4.2's backward-pass algebra and cost-to-go
symmetrization, the regularization state machine (ϕ = 1.6, ρmin = 1e-8, one decrease per backward
pass), §4.3's line-search branches, §4.4's gradient and iLQR convergence order, §5.3's equality
and inequality cost/grad/hess and dual updates, §5.4's violation definition, and §7.1's conclusion
that Gauss-Newton is exact for affine constraints.

## Testing policy for these tickets

The full suite is slow.

**While working: targeted checks only.** `uv run pytest` on the test files this ticket owns,
`uv run ty check`,`uv run ruff check --fix`. That is the whole loop.

**The full pre-commit gate runs once with `git commit`

**Keep the Altro cross tests deselectable.** Ticket 25 adds a second Julia session fixture on top
of the existing `jl_to`. Mark the Altro cross tests so a worker can deselect them by marker, keep the fixture session-
scoped, and leave `jl_to` untouched so the existing cross tests do not pay the Altro load. Within
a ticket, run its Julia parity tests once you believe the Python side is right.

## Ordering

24 → 25 → 26 → 27 gives a working unconstrained iLQR. 28 can start as soon as 24 lands and runs
in parallel with 25–27. 29 needs both branches. 30 and 31 are independent leaves off 29 and 28.
32 → 33 → 34 close it out.

Commit convention, matching tickets 01–23: `<type>(<scope>): <description> (Ticket NN)`.
