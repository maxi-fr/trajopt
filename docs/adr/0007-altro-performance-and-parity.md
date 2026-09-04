# ADR 0007: ALTRO performance optimizations, conditional Projected Newton, and outer-loop robustness

## Status

Accepted. Partially supersedes ADR 0001.

## Context

ADR 0001 recorded the initial deliberate divergences between the native JAX port of ALTRO
(`src/trajopt/solvers/`) and `Altro.jl` at commit `4864df2`. Benchmarking against Ipopt on reference
problems ($N=25$ Dubins corridor, Cartpole swingup, Quadrotor obstacle) revealed significant
performance overheads and fragile convergence behaviors, investigated and detailed in
`docs/altro_optimization_and_parity.md`.

In particular, three findings warranted architectural decisions diverging from or refining the
original port recorded in ADR 0001:
1. Projected Newton (PN) executed `multiplier_projection` by default (`options.multiplier_projection=True`).
   Upstream Julia had commented this out as dead code (`pn_solve.jl`, issue #35), hardcoding `res = Inf`.
   Assembling and solving the resulting $N_d \times N_d$ Gram matrix added substantial overhead to every
   PN solve.
2. ADR 0001 specified that `altro_solve` always executes PN's traced core and selects trajectories
   using `jnp.where(run_pn, pn_traj, al_traj)` rather than `lax.cond`. Consequently, every solve paid
   the dense $O((N(n+m+p))^3)$ KKT assembly and `jnp.linalg.solve` cost even when the Augmented
   Lagrangian (AL) phase had already driven constraint violation under tolerance, or when the user
   explicitly disabled PN via `options.projected_newton=False`.
3. In `src/trajopt/solvers/al.py`, the outer AL loop aborted immediately whenever inner iLQR returned
   any status with `inner_status > SOLVE_SUCCEEDED`. On non-convex problems (e.g. Quadrotor obstacle)
   where inner iLQR stalled with `NO_PROGRESS` (8) or `MAX_ITERATIONS` (3), this premature abort skipped
   outer Multiplier and Penalty Parameter updates ($\lambda \leftarrow \lambda + \mu c, \mu \leftarrow \phi \mu$).
   This reproduced an upstream Julia bug where `status(solver) > SOLVE_SUCCEEDED && break` in `al_solve.jl`
   aborted on `NO_PROGRESS`, contradicting the authors' own code comment that `NO_PROGRESS` was intended
   to trigger an outer loop update.
4. During iLQR line search in `_ALObjective.cost(traj)`, full constraint Jacobians, bound Jacobians,
   and error-state coordinate einsums were repeatedly computed and discarded on every trial step $\alpha$
   (up to 20 times per iLQR iteration).

This ADR records the decisions resolving these issues within the repository's single-context ADR system.

## Decision

### `multiplier_projection` defaults to `False`

`options.multiplier_projection` in `src/trajopt/solvers/options.py` now defaults to `False` (previously `True`).
Upstream Julia cannot exercise the `True` path because `multiplier_projection!` is commented out.
Disabling this redundant projection by default eliminates the assembly of the $N_d \times N_d$ Gram matrix
and its dense linear solve, cutting PN solve duration on the Dubins benchmark from ~480 ms to ~205 ms
while matching shipped Julia behavior. Users desiring the full projection can still enable it explicitly.

This decision supersedes the default configuration noted in ADR 0001 §"Projected Newton: `multiplier_projection` is a genuine port".

### Conditional execution of Projected Newton via `lax.cond` and eager bypass

The unconditional execution of PN via `jnp.where` recorded in ADR 0001 is superseded:
1. When `options.projected_newton=False` and `options.force_pn=False`, `altro_solve` bypasses `pn_solve`
   entirely at the Python level, returning `al_traj` with empty PN statistics.
2. Inside the traced core, when PN is enabled, `pn_solve` is wrapped in `jax.lax.cond(run_pn, _do_pn, _skip_pn)`.
   When the AL phase already satisfies the constraint tolerance (`run_pn` is false), the expensive dense KKT
   assembly and factorization are skipped during execution.

Both branches trace with identical pytree output structures (`ALTROSolveResult`), preserving end-to-end
traceability under `jax.jit` and `jax.vmap` while eliminating dense KKT overhead whenever PN is unnecessary.

This decision supersedes ADR 0001 §"The ALTRO driver always runs PN's traced core".

### Residual-only evaluation in Augmented Lagrangian line search

`evaluate_al_residuals` in `src/trajopt/solvers/al.py` evaluates constraint residuals $c(x, u)$ and bound
residuals without computing constraint Jacobians $\nabla c$, bound Jacobians, or error-state coordinate mappings.
`_ALObjective.cost(traj)` in the inner iLQR line search, outer step residual evaluation in `_al_step`, and
convergence checking in `altro_solve` call `evaluate_al_residuals`. Full Jacobian evaluations via
`evaluate_al_constraints` are reserved strictly for second-order quadratic model expansions in
`_ALProblem.cost_expansion`.

### Soft inner stalls and outer-loop iteration handling

In `src/trajopt/solvers/al.py`:
- Inner iLQR termination statuses are partitioned into:
  - **Soft stalls**: `TerminationStatus.NO_PROGRESS` and `TerminationStatus.MAX_ITERATIONS`.
    These indicate that inner iLQR cannot make further progress on the current augmented objective landscape.
    The outer AL loop does not treat these as fatal; it proceeds to update Multipliers and scale Penalty
    Parameters, modifying the cost landscape to unstick the Native Solver in subsequent outer iterations.
  - **Fatal errors**: `inner_status > SOLVE_SUCCEEDED` excluding soft stalls (e.g. `NAN_DETECTED`, `STATE_LIMIT`).
    These abort the outer loop immediately.
- In `_evaluate_al_convergence`, `inner_iterations >= options.iterations` is removed from the outer
  loop termination condition `done`. In upstream `Altro.jl`, `solver.stats.iterations` tracked cumulative
  iterations across all outer loops; in this port, `ilqr_solve`'s counter resets per call (ADR 0001).
  Treating single-call iteration limits as outer halts prematurely terminated the solver on iteration 1
  before Multiplier updates could take effect. The outer loop is now strictly bounded by
  `iter_num >= options.iterations_outer`.

This unblocks convergence on constrained problems where the initial guess produces zero-progress inner line searches.

### Per-row penalty capping to prevent `MAXIMUM_COST` line-search aborts

In upstream `Altro.jl`, penalties scale unconditionally as $\mu \leftarrow \phi \mu$ up to `penalty_max = 1e8`.
On complex nonlinear systems with large transient excursions (such as Cartpole swingup), the quadratic penalty
$\frac{1}{2} \mu c^2$ rapidly exceeds `options.max_cost_value = 1e8`, causing inner iLQR's forward pass line
search to abort with `TerminationStatus.MAXIMUM_COST` (7).

In `penalty_update` (`src/trajopt/solvers/al.py`), when constraint residuals $C$ are available, penalties
are capped per constraint row:
$$\mu_{k,j} \le \min\left(\text{penalty\_max}, \frac{2 \cdot \text{max\_cost\_value}}{c_{k,j}^2 + \epsilon}\right)$$
This ensures that quadratic penalties never trigger `MAXIMUM_COST` during line searches while allowing
rows with near-zero violations to scale freely up to `penalty_max`.

## Consequences

- On problems where AL converges to tolerance (such as Dubins corridor), ALTRO executes AL alone, running in
  ~24 ms (over 20x faster than Ipopt) without paying ~76 ms of unneeded PN overhead.
- When PN is required, disabling `multiplier_projection` eliminates redundant dense linear algebra.
- Quadratic penalty evaluation during line search avoids wasteful automatic differentiation and einsum operations,
  yielding a 2x-5x speedup in the inner iLQR line search.
- Hard problems that previously failed immediately on outer iteration 1 with `NO_PROGRESS` now execute outer
  multiplier/penalty updates, matching the intended AL-iLQR theory.
- Cross-verification tests comparing full solves against `Altro.jl` must account for the upstream Julia bug:
  Julia prematurely breaks on `NO_PROGRESS`, whereas this port correctly performs outer updates. Unit-level
  invariant tests in `test/cross_verification/` verify individual update formulas and state transitions.
