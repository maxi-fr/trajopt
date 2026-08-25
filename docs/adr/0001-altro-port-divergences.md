# ADR 0001: Deliberate divergences of the native ALTRO port from Altro.jl

## Status

Accepted.

## Context

Tickets 24-33 ported iLQR, augmented Lagrangian (AL), Box-DDP, Projected Newton (PN), and the
ALTRO driver from `altro_jl/` into `src/trajopt/solvers/` as native JAX solvers
(`options.py`, `ilqr.py`, `al.py`, `boxqp.py`, `pn.py`, `altro.py`). The port is corrected
against `docs/altro-port/00-overview.md`'s findings A-N, each checked against `altro_jl/src/`
at the pinned commit `4864df2`.

A handful of places depart from Altro on purpose rather than by omission. Each one trades
literal fidelity to Altro for something this codebase needs more: traceability under
`lax.while_loop`/`lax.scan`, a JAX-native data layout, or a bug fix upstream never landed. This
ADR records them as a set so a failing parity test against Altro can be checked against a
documented decision instead of re-litigated, and so the next reader does not mistake a deliberate
divergence for an unnoticed bug.

## Decision

### Fully traced control flow

Every solver loop is a `lax.while_loop` or `lax.scan`; the whole solve is one jittable, vmappable
function of pytree state (`docs/altro-port/00-overview.md`, "Decisions taken up front").
`SolverOptions` is a frozen dataclass, never traced, so it can pick loop bounds and shapes.

Consequence: the port is **not reverse-mode differentiable** -- `lax.while_loop` has no reverse
rule. Forward mode works; if gradients through a solve are ever needed, the fix is a `custom_vjp`
on the fixed point, layered on top without touching the inner loops.

Consequence: a `scan`/`while_loop` cannot early-exit mid-sweep, so a failing computation (e.g. a
Cholesky factorization that keeps failing) costs a full wasted sweep rather than an early return,
unlike Altro's eager `while true`/`for` loops which can `break` or early-return the instant a
condition is known.

### `bp_reg_max` is live here, dead in Altro (finding F)

Altro's `SolverOptions.bp_reg_max` is never read by `altro_jl/src/`: nothing bounds how high the
backward pass's regularization `rho` can climb on repeated Cholesky failures. Its `while true`
loop (`backwardpass.jl`) simply restarts at `k = N-1` and raises `rho` by `bp_reg_increase_factor`
forever if the factorization keeps failing -- merely slow in Altro, since a human or a wall-clock
budget eventually kills the process.

Under `lax.while_loop` an unbounded retry loop is an **unkillable hang**: there is no signal to
interrupt a traced loop from outside. `src/trajopt/solvers/ilqr.py`'s backward-pass retry
(`backward_pass`, around the `sweep.failed & (rho <= options.bp_reg_max)` condition) makes
`bp_reg_max` the loop's exit bound. This is a behavioural difference, not just a safety net: a
problem Altro would grind on forever, this port exits with a failure status.

### iLQR's iteration counter resets per call; Altro's is cumulative

Altro's inner iLQR `MAX_ITERATIONS` check in `evaluate_convergence` reads
`solver.stats.iterations`, a counter shared across the *entire* AL outer loop -- it accumulates
over every inner iLQR solve the outer loop runs, not just the current one. This port's
`ilqr_solve` (`src/trajopt/solvers/ilqr.py`) starts its own iteration counter at zero on every
call, including every call `al_solve`'s outer loop makes.

Inert at the default `options.iterations=1000`: with `iterations_outer` capped at 30 and each
inner solve typically converging in a handful of iterations, the cumulative total rarely
approaches 1000 either way. It would bite under a tight iteration budget spread thin across many
outer iterations, where Altro's cumulative counter could hit `MAX_ITERATIONS` mid-outer-loop while
this port's per-call counter would let each inner solve run its full local budget. Not fixed to
match Altro, because a per-call budget is arguably the more useful semantics for a native solver
whose whole point is being called repeatedly (MPC warm-starts); recorded here as a known
difference in what `iterations` limits.

### Box-DDP clips every rolled-out control, beyond the paper's guarantee

The control-limited backward pass (`src/trajopt/solvers/boxqp.py`) is Tassa's box-QP DDP, not
part of Altro at all -- verified independently against Clarabel, not against `altro_jl/`. The
paper's feasibility guarantee is **local**: it bounds the feedforward step `d` at `dx = 0`, i.e.
the box-QP solution is itself bound-feasible only along the nominal trajectory. The actual
closed-loop control applied during rollout is `u = u_bar + K @ dx + alpha * d`, and the `K @ dx`
feedback term is not bound-constrained by the box-QP solve at all.

This port clips every rolled-out control to `u_bounds` in `rollout_closed_loop`
(`src/trajopt/solvers/ilqr.py`), forwarded from `al_solve`/`altro_solve` whenever a box-QP
`solve_kd_builder` is in play. This is a deliberate safeguard beyond what the paper proves,
covering the gap the feedback term leaves open.

### Conic path guards against silently reinterpreted duals (finding E)

Altro's conic and non-conic AL paths use opposite sign conventions for the stored dual: non-conic
`lambda_bar = lambda + mu*c`, conic `lambda_bar = lambda - mu*c`. Flipping `use_conic_cost`
between two solves that warm-start off the same duals would silently reinterpret their sign.

`ALConstraints` (`src/trajopt/solvers/al.py`) carries a static `is_conic` tag set at construction.
`AL.solve` and `ALTRO.solve` raise `ValueError` if a warm-started `state.al`'s tag disagrees with
`options.use_conic_cost`, unless `options.reset_duals=True` (which discards the old duals instead
of reinterpreting them). This is new user-facing behaviour Altro does not have: Altro has no
warm-start guard here at all.

### Projected Newton: dense KKT, not QDLDL

Altro's PN factors its sparse KKT matrix with QDLDL; the active set changes shape between
iterations, and a sparse factorization's fill-in pattern would change with it. JAX requires static
shapes under tracing, so `src/trajopt/solvers/pn.py` assembles a dense masked KKT matrix and
solves it with `jnp.linalg.solve`. Numerically equivalent to Altro's sparse solve at the problem
sizes this port targets; the cost is doing dense linear algebra on what upstream treats as a
sparse system.

### Projected Newton: `multiplier_projection` is a genuine port, a superset of upstream

Altro's `multiplier_projection!` (`pn_solve.jl`) is commented-out dead code -- upstream issue #35
-- and its call site hardcodes the result to `res = Inf`, so it never actually executes upstream.
`src/trajopt/solvers/pn.py`'s `multiplier_projection` is implemented for real, gated behind
`options.multiplier_projection` (default `True`, matching Altro's own default value for an option
whose corresponding code never runs). This port therefore has behaviour Altro's shipped code does
not; parity tests against Altro must set `multiplier_projection=False` on both sides, since
upstream cannot exercise the `True` path at all.

### Projected Newton: `x0` is an explicit parameter, not `trajectory.X[0]`

PN is a different formulation from AL-iLQR: its own stacked primal `Zdata` and dual `Ydata`, with
dynamics as explicit equality constraints -- multiple shooting, not the shooting formulation
AL-iLQR uses (finding L). `X[0]` is therefore a free primal variable in PN's own layout, not fixed
data; `pn_solve` (`src/trajopt/solvers/pn.py`) threads `x0` as an explicit parameter that pins the
initial-condition equality row, mirroring how Altro's own PN solver caches `pn.x0` once from
`prob.x0` at solver construction rather than reading it off the trajectory each time.

### `rho_chol` / `rho_dual` not ported; only `rho_primal` regularizes

Finding F: both are on Altro's dead-option list, never read by `altro_jl/src/`. Only
`options.rho_primal` regularizes PN's dense KKT solve (`H + rho_primal * I` in
`src/trajopt/solvers/pn.py`).

### The ALTRO driver always runs PN's traced core

`altro_solve` (`src/trajopt/solvers/altro.py`) always calls `pn_solve` and selects between AL's
and PN's trajectories with `jnp.where(run_pn, pn_traj, al_traj)` rather than `lax.cond`, matching
this codebase's established style for traced branches elsewhere (e.g. `al.py`'s `inner_failed`
handling). This keeps the whole driver `jax.jit`/`jax.vmap`-able end to end -- a `lax.cond` with a
traced predicate still requires both branches to trace with matching output structure, and
`jnp.where` on the whole pytree is the simpler way to express "run both, pick one" once both sides
are already being traced. The cost is wasted computation whenever PN would not have been needed
(`run_pn` false): PN's projection still runs and its result is simply discarded. Never a
behavioural difference, only a compute cost.

### No per-constraint AL parameter overrides

Altro's `ConstraintOptions` lets each constraint override `penalty_initial`, `penalty_scaling`,
`penalty_max`, `dual_max`, and `use_conic_cost`. This port has one global `SolverOptions` set for
all constraints. Per-constraint overrides are configurability nobody using this port has asked
for, and supporting them would force the padded per-knot, per-constraint-row `ALConstraints`
layout to also carry per-constraint parameter blocks.

### Eleven dead options discarded; `bp_reg_type` not ported

Eleven of Altro's `SolverOptions` fields are never read anywhere in `altro_jl/src/` (finding F,
full list in `docs/altro-port/00-overview.md`): `iterations_inner`, `bp_reg`, `square_root`,
`save_S`, `static_bp`, `reuse_jacobians`, `active_set_tolerance_al`, `rho_chol`, `rho_dual`,
`solve_type`, plus the logging-only `show_summary`/`trim_stats`. None of the eleven exist on
`src/trajopt/solvers/options.py`'s `SolverOptions`.

`bp_reg_type = :state` is separately broken upstream (finding H: `Qux_reg .= Qux` where `Qux` is
undefined at that point), so only `:control` ever executes. Porting the option would port a
one-valued knob; it is dropped, and `:control`'s behaviour is unconditional in this port.

### Gauss-Newton penalty Hessian

The AL penalty cost's Hessian uses the Gauss-Newton approximation throughout
(`src/trajopt/solvers/al.py`), matching Altro. Reference doc §7's measurements, verified during
this port, are the justification: exact for affine constraints, differing from the full Hessian
only by a `penalty * violation * curvature` term for nonlinear ones, and positive semi-definite
where the full Hessian can be indefinite -- a better-conditioned backward pass at the cost of
exactness on curved constraints.

## Consequences

- A parity test against Altro that trips on any of the above is checked against this ADR before
  being treated as a bug. `multiplier_projection=False` must be set on both sides of any PN parity
  test, since Altro's own code cannot exercise the `True` path.
- Gradients through a native solve require a `custom_vjp`, not yet built.
- A pathological problem that repeatedly fails its Cholesky factorization exits with a failure
  status here, where Altro would run indefinitely.
- Warm-starting AL duals across a `use_conic_cost` flip requires `options.reset_duals=True`, or the
  solve raises `ValueError` -- Altro has no equivalent guard, since Julia has no traced-loop
  warm-start path with the same silent-reinterpretation risk.

## Benchmark: native ALTRO vs Ipopt

`src/trajopt/benchmarks.py`'s `measure_altro_vs_ipopt` times `ALTRO().solve()` against
`Ipopt().solve()` on `cartpole_swingup_benchmark`'s N=25 bound-and-goal-constrained cartpole (the
same shape of problem as the ticket 33 cross test), each with one discarded warmup solve first so
neither measurement is dominated by first-call setup. Single run, one machine (Windows, this
development environment), not a statistically averaged number -- reported to close the loop
honestly, not as a performance claim to build on:

| path                                                    | time      |
|----------------------------------------------------------|-----------|
| Ipopt, second (cached) call                               | ~0.51 s   |
| `ALTRO().solve()`, second (still eager, uncompiled) call  | ~9.5 s    |
| `altro_solve` hand-wrapped in `jax.jit`, first (compiling) call | ~9.7 s |
| `altro_solve` hand-wrapped in `jax.jit`, second (cached) call   | ~2.8 s |

**Ipopt is faster on this problem in every configuration measured**, including the hand-jitted,
cache-warm native core. The "Thin eager wrapper over a traced core" decision
(`docs/altro-port/00-overview.md`) means `ALTRO().solve()` does not `jax.jit` `altro_solve` itself
-- every eager call retraces and dispatches each `lax` primitive individually rather than running
one fused XLA computation, which is why the eager number and the jit-compiling number are close:
neither benefits from a cached compilation. Only a caller who explicitly wraps the traced core in
`jax.jit` (closing over `problem` and `options`, per `test_altro.py`'s
`test_altro_solve_is_jittable_and_vmappable_with_static_options`) and reuses that compiled
function across repeated calls sees the ~3.4x drop from eager to cached -- and even that cached
number does not beat Ipopt here. The `jnp.where(run_pn, ...)`-always-runs-PN design (see above)
is one concrete cost contributor: PN's dense KKT solve runs on every `altro_solve` call whether or
not its projection is used.

The design's actual payoff case -- many solves sharing one compilation, e.g. `jax.vmap` over a
batch of initial states, or a receding-horizon loop that jits once and calls the compiled function
many times -- is not what a single `.solve()` call measures, and is not benchmarked here. Recorded
as a real, currently-negative data point rather than omitted or reframed positively; the
`Ipopt`-vs-native tradeoff for a one-off solve currently favors `Ipopt`.

## Follow-up flagged, not resolved

`AGENTS.md`/`CLAUDE.md` describe a root `CONTEXT.md` glossary and `docs/agents/domain.md` as the
project's domain-doc infrastructure; neither exists yet, and this ADR does not create either --
only `docs/adr/` and this first ADR. Building a full domain model is its own task.
