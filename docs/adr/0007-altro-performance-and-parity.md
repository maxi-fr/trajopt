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
5. PN's KKT solve was a dense $\left(N_p + N_d\right)^2$ assembly factored by `jnp.linalg.solve`, with
   inactive rows masked to an identity block so their multipliers solve to zero. Upstream factors the
   sparse KKT matrix with QDLDL instead. The dense solve dominates PN's cost and scales cubically in
   the horizon.

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
This keeps quadratic penalties from triggering `MAXIMUM_COST` during line searches while allowing rows
with near-zero violations to scale freely up to `penalty_max`.

**The cap applies only to rows inside `_active_penalty`** -- the same active-set test `al_cost` uses to
decide whether a row's `mu` enters the cost at all. An inequality that is comfortably satisfied has a
large negative residual and an inactive penalty, so it contributes nothing to the cost and there is no
overflow to prevent; capping against its $c^2$ anyway throttled it for no reason and broke whole-solve
parity with Julia. `test_cross_al_solve_cartpole_matches_altro` is the test that pins this down: with
the cap ungated, the cartpole's control lower-bound penalties settle at $8.6 \times 10^6$ where Altro's
ladder puts them at $10^8$, while the trajectory, cost history, violation history and `penalty_max`
history all still match to $10^{-8}$. The divergence is invisible in the solution and visible in the
recorded duals, which is exactly the kind of drift a cross-verification exists to catch. Gated on the
active set, the whole-solve parity holds again.

The residual consequence, kept deliberately: on a converging solve an active row's residual goes to
zero, so the gated cap is inert in ordinary operation and earns its keep only on transient excursions
large enough to overflow. That is the intended scope.

### Sparse QDLDL KKT factorization for Projected Newton

`_solve_kkt_step` in `src/trajopt/solvers/pn.py` no longer assembles the dense masked KKT matrix.
It calls out to QDLDL on the host through `jax.pure_callback`, matching what upstream `Altro.jl`
actually factors. The callback selects the active rows outside the trace, so the system it factors
is $\left(N_p + N_a\right)$ square rather than $\left(N_p + N_d\right)$ with the inactive block
padded out, and it carries a $-10^{-8} I$ dual regularization block, which QDLDL needs for a
quasi-definite factorization and the dense LU solve did not.

Three things follow, and the third is the one that cost a golden file.

1. Shape-polymorphic sparse factorization is possible under `jax.jit` only because the work happens
   on the host. `vmap_method="sequential"` keeps `jax.vmap` working; the price is a host round trip
   per PN iteration and no differentiability through the step.
2. The masking scheme ADR 0001 recorded is gone. There is no inactive identity block, because
   inactive rows are not in the system at all.
3. **The step is not bit-identical to the dense solve.** Selecting rows rather than masking them, an
   LDL^T factorization rather than an LU, and the $10^{-8}$ dual regularization each perturb the
   Newton step at roughly the $10^{-8}$ level. That is inside solver tolerance for a single solve,
   but a receding-horizon loop feeds each solve's output back in as the next one's state, so it
   compounds.

`test/unit/test_mpc_characterization.py`'s recorded closed-loop golden was measured on both sides of
this change. Bisecting the four commits of this work against the golden attributes the whole drift to
this one, and restoring the dense solve reproduces the old golden to the last bit; dropping the dual
regularization to $10^{-12}$ shrinks the drift by five orders of magnitude. On the kicked cartpole the
two runs agree to $10^{-7}$ for 23 steps, then separate, ending 0.07 apart in state and 0.55 in
commanded force. The divergence is a path difference, not a quality difference:

| | dense masked LU (old golden) | QDLDL + dual regularization |
| --- | --- | --- |
| steps reporting `SOLVE_SUCCEEDED` | 40 / 40 | 40 / 40 |
| max constraint violation over the run | 5.8e-7 | 8.8e-7 |
| total closed-loop cost | 2106.65 | 2107.54 |
| steps on which PN ran | 18 | the same 18 |

Both stay under the $10^{-6}$ constraint tolerance, the cost differs by 0.04%, and the qualitative
recovery assertions in the same file -- bounds respected, monotone climb out of the dip, final angle
error under 0.2 rad -- hold either way. The golden was therefore **re-recorded**, not defended.

What this buys: PN factors the system upstream factors, at sparse cost. What it sells: the golden is
now pinned to an external C library's factorization, so a QDLDL or SciPy version bump can move it
again. It is a characterization test, and re-recording it after measuring status, violation and cost
either way is the intended response to that.

### What is no longer claimed about parity

The comment in `_evaluate_al_convergence` used to say the function is bit-identical to Altro's under
default options, and that this is what keeps the Julia parity test and the recorded MPC golden valid.
Neither half survives this ADR. Two of the changes above -- dropping `inner_iterations >= options.iterations`
from `done`, and adding the `penalty_overflow` exit -- apply under default options, so the function
diverges from Altro's for every configuration, not only under `reset_penalties=False`. ADR 0006's
narrower claim, that *its own* `iter_num > 1` gate is invisible under default options, is still true,
and is all that comment should have asserted.

Julia parity is held at the level `test/cross_verification/` actually tests it: update formulas,
expansions, backward and forward passes, and individual state transitions, each against the live Julia
solver. A whole-solve history replay through the outer loop is not something this port claims.

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
- PN's KKT solve is sparse and active-set-sized, at the cost of a host round trip per PN iteration and
  no differentiability through the Newton step.
- The recorded MPC golden was re-measured, not preserved. Any future change to the KKT factorization
  should expect the same, and should re-record only after checking status, violation and cost on both
  sides, as was done here.
- The penalty cap and ADR 0006's penalty carry are independent in effect but overlap in scope, and the
  combination was untested until now. Measured both ways on the kicked cartpole: on the default
  `(reset_duals=True, reset_penalties=True)` the cap never binds -- disabling it reproduces the golden
  bit for bit -- and once gated on the active set it does not bind under `(False, False)` either, where
  the run is likewise bit-identical with the cap off. The cap is therefore not what moved the golden,
  and it is not a fix for ADR 0006's ratchet. `test/unit/test_al_inherited_penalty.py` covers the
  combination: that the cap governs an inherited `mu` and can pull it down, that it leaves a satisfied
  inequality on Altro's ladder, and that a receding-horizon run carrying both duals and penalties stays
  finite and non-fatal across the seam.
- The cap guards the penalty *update*. A `mu` arriving from a warm start is not capped before that
  step's first inner solve reads it. No run has been observed to hit `MAXIMUM_COST` through that gap,
  so nothing is done about it here; it is recorded so the asymmetry is not mistaken for an invariant.
