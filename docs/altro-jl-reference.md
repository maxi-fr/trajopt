# Altro.jl internals — reference for the one-to-one port

> **This document has known errors.** `docs/altro-port/00-overview.md`'s "Verified corrections"
> section (findings A-N) catalogues fourteen places where this reference disagrees with
> `altro_jl/src/` at the pinned commit below -- each finding was checked against the source, and
> where the two disagree, the finding wins. Read the corrections first; sections affected are not
> separately flagged inline. `docs/adr/0001-altro-port-divergences.md` records where the Python
> port then deliberately diverges from Altro on top of those corrections.

Primary source: `Altro.jl` at commit `4864df2bb8ab8f629f451304cbaaa8e0017932d9`
(`main`, `Project.toml` version `0.5.0`), vendored at `altro_jl/` (committed to this repo).
This document is the algorithm and default-parameter oracle for the native
iLQR / box-iLQR / ALTRO solvers in `trajopt`. Every default below was read from
`src/solver_opts.jl`, not from the README; the README table is quoted where it
disagrees.

Altro.jl consumes `TrajectoryOptimization.jl` (vendored at `trajopt_jl/`, v0.7.1)
and its `[compat]` requires `TrajectoryOptimization = "0.7"`, so the vendored
formulation library and Altro can live in one Julia environment.

---

## 1. Architecture map

```text
src/Altro.jl                     module, includes everything below
src/solvers.jl                   AbstractSolver{T} tree, TerminationStatus, stats plumbing
src/solver_opts.jl               SolverOptions{T} (all defaults), SolverStats{T}
src/utils.jl                     helpers (benchmark, triukkt! for PN)
src/infeasible_model.jl          InfeasibleProblem (augmented controls; not needed for the port)

src/ilqr/
  ilqr_solver.jl                 iLQRSolver struct, ctor, regularization state, reset/gain getters
  cost_expansion.jl              StateControlExpansion, CostExpansion, cost_expansion!, error_expansion!
  dynamics_expansion.jl          DynamicsExpansion (full-state A,B and error-state fx,fu), errstate jacobians
  backwardpass.jl                Riccati recursion -> gains (K, d), cost-to-go S, expected decrease ΔV
  forwardpass.jl                 rollout with affine policy + Armijo-style line search
  ilqr_solve.jl                  solve! loop, gradient!, record_iteration!, evaluate_convergence

src/augmented_lagrangian/
  alcon.jl                       ALConstraint: per-constraint λ, μ, penalty cost/grad/hess,
                                 dual/penalty updates, violation
  alconset.jl                    ALConstraintSet: constraint collection, max_violation, max_penalty
  al_objective.jl                ALObjective: obj + conset; cost() and cost_expansion! fold AL in
  al_solver.jl                   ALSolver = iLQRSolver + ALObjective
  al_solve.jl                    outer loop: solve iLQR, dual update, penalty update, convergence

src/altro/
  altro_solver.jl                ALTROSolver = ALSolver + ProjectedNewtonSolver
  altro_solve.jl                 phase 1 AL-iLQR, phase 2 Projected Newton polish

src/direct/                      Projected Newton solver (active-set sparse KKT, QDLDL)
```

### Python mapping targets

| Julia | Python (planned) |
| :--- | :--- |
| `iLQRSolver` | `src/trajopt/solvers/ilqr.py` — plain iLQR |
| `ALSolver` + `ALObjective` + `ALConstraintSet` | `src/trajopt/solvers/altro.py` — box-iLQR and ALTRO share this machinery |
| `ALTROSolver` | `src/trajopt/solvers/altro.py` (outer loop + optional PN phase) |
| `CostExpansion` / `DynamicsExpansion` | existing `trajopt.expansions.Expansion` (already equivalent) |
| `ALObjective.cost_expansion!` | existing `_augmented_lagrangian_expansion` (**not numerically identical, see §6**) |

## 2. Termination statuses

`@enum(TerminationStatus, UNSOLVED, LINESEARCH_FAIL, SOLVE_SUCCEEDED, MAX_ITERATIONS,
MAX_ITERATIONS_OUTER, MAXIMUM_COST, STATE_LIMIT, CONTROL_LIMIT, NO_PROGRESS, COST_INCREASE)`

`LINESEARCH_FAIL` and `COST_INCREASE` are set but never compared by the solvers
(forwardpass returns `NaN` cost, which feeds `dJ = NaN`).

**Decision (user):** keep the public `SolverStatus` 4-value literal
(`converged | infeasible | iteration_limit | error`) as the API surface, but compute and
record the full 10-value status internally in the solver stats (`info` / a `SolverStats`
object), so debugging can see the precise exit reason. Suggested mapping:

| internal (Altro) | public |
| :--- | :--- |
| `SOLVE_SUCCEEDED` | `converged` |
| `MAX_ITERATIONS_OUTER`, `MAXIMUM_COST`, `UNSOLVED` | `infeasible` |
| `MAX_ITERATIONS`, `NO_PROGRESS`, `LINESEARCH_FAIL` | `iteration_limit` |
| `STATE_LIMIT`, `CONTROL_LIMIT`, `COST_INCREASE` | `error` |

This mirrors what the transcription adapters already do (`SolverStatus` in
`transcription/result.py`): native status preserved in the result, normalized vocabulary
upstream.

## 3. SolverOptions defaults (source of truth: `src/solver_opts.jl`)

```text
constraint_tolerance           1e-6     cost_tolerance              1e-4
cost_tolerance_intermediate    1e-4     gradient_tolerance          10.0
gradient_tolerance_intermediate 1.0     expected_decrease_tolerance 1e-10
iterations_inner               300      dJ_counter_limit            10
square_root                    false    line_search_lower_bound     1e-8
line_search_upper_bound        10.0     line_search_decrease_factor 0.5
iterations_linesearch          20       max_cost_value              1e8
max_state_value                1e8      max_control_value           1e8
static_bp                      true     save_S                      false
closed_loop_initial_rollout    false

bp_reg                         false    bp_reg_initial              0.0
bp_reg_increase_factor         1.6      bp_reg_max                  1e8
bp_reg_min                     1e-8     bp_reg_type                 :control
bp_reg_fp                      10.0

use_conic_cost                 false    penalty_initial             1.0
penalty_scaling                10.0     penalty_max                 1e8
dual_max                       1e8
active_set_tolerance_al        1e-3     iterations_outer            30
kickout_max_penalty            false    reset_duals                 true
reset_penalties                true

force_pn                       false    verbose_pn                  false
n_steps                        2        solve_type                  :feasible
projected_newton_tolerance     1e-3     active_set_tolerance_pn     1e-3
multiplier_projection          true     ρ_chol                      1e-2
ρ_primal                       1e-8     ρ_dual                      1e-8
r_threshold                    1.1

dynamics_funsig                StaticReturn()   dynamics_diffmethod  ForwardAD()
projected_newton               true     reuse_jacobians             false
trim_stats                     true     iterations                  1000
show_summary                   true     verbose                     0
```

README disagreements (source wins):

- README says `gradient_tolerance` = `1`, `gradient_tolerance_intermediate` = `10`; source is `10.0` / `1.0`.
- README lists `penalty_initial`/`penalty_scaling` default `NaN` (defer to per-constraint params); source and `ConstraintOptions` use `1.0` / `10.0`.
- `iterations_inner` (300) is defined but **never referenced** in `src/`; the iLQR loop uses `iterations` (1000). Dead option — do not port it.
- `bp_reg` (false) is also unreferenced; regularization is driven by `solver.reg` (ρ, dρ) plus `bp_reg_type`/`bp_reg_fp`.

**Discarded in the port (decision):** `static_bp`, `save_S`, `closed_loop_initial_rollout`,
`square_root` (WIP in Altro), `reuse_jacobians` (performance only), and the logging-only
options (`show_summary`, `trim_stats`, verbose levels ≥ 2) are not ported. The remaining
options are ported with Altro's exact names and defaults.

## 4. iLQR (`src/ilqr/`)

### 4.1 Solve loop (`ilqr_solve.jl`)

```julia
initialize!:  reset stats, ρ = bp_reg_initial, dρ = 0, rollout (open-loop unless
              closed_loop_initial_rollout), copy Z -> Z̄
loop iter = 1..opts.iterations:
    J_prev = cost(obj, Z̄)
    errstate_jacobians! (G)   # identity for Euclidean models
    dynamics_expansion!       # full-state Jacobians, then error-state: fx = G2' A G1, fu = G2' B
    cost_expansion!           # obj gradient/hessian per knot (full state)
    error_expansion!          # Eerr from Efull via G (identity for Euclidean)
    backwardpass!             # gains (K, d), S (cost-to-go), ΔV
    Jnew = forwardpass!       # line search over α; on failure returns NaN
    copyto!(Z, Z̄)             # accept step
    dJ = J_prev - Jnew
    grad = gradient!(solver)  # see 4.4
    record_iteration!
    exit = evaluate_convergence
terminate! (trim stats, tsolve)
```

### 4.2 Backward pass (`backwardpass.jl`)

Terminal: `S[N].xx = E[N].xx`, `S[N].x = E[N].x`. Loop `k = N-1 .. 1`:

```julia
Qx  = A' S[k+1].x  + E[k].x
Qu  = B' S[k+1].x  + E[k].u
Qxx = A' S[k+1].xx A + E[k].xx
Quu = B' S[k+1].xx B + E[k].uu
Qux = B' S[k+1].xx A + E[k].ux          # shape (m, n); E.ux is (m,n)

regularization (bp_reg_type = :control): Quu_reg = Quu + ρ I_m; Qux_reg = Qux
Cholesky(Quu_reg); on failure: increaseregularization!, restart k = N-1, ΔV = 0
solve [K_k; d_k] = Quu_reg \ [-Qux; -Qu]   # one ldiv! on stacked (m, n+1), then negate
  => K = -Quu⁻¹ Qux, d = -Quu⁻¹ Qu     (K, d stored NEGATED: policy is u = ū + K δx + α d)
S[k].x  = Qx + K' Quu d + K' Qu + Qux' d
S[k].xx = Qxx + K' Quu K + K' Qux + Qux' K, then symmetrized: (S + S') / 2
ΔV[1] += d' Qu ;  ΔV[2] += 0.5 d' Quu d
```

Regularization state machine (`DynamicRegularization` ρ, dρ):

- `increaseregularization!`: `dρ = max(dρ·ϕ, ϕ)`, `ρ = max(ρ·dρ, ρmin)` with ϕ = 1.6, ρmin = 1e-8.
- `decreaseregularization!` (called once per backward pass, at the end): `dρ = min(dρ/ϕ, 1/ϕ)`, `ρ = max(ρmin, ρ·dρ)`.
- forward-pass failures also add `bp_reg_fp` (10.0) to ρ.

### 4.3 Forward pass / line search (`forwardpass.jl`)

For α starting at 1.0, up to `iterations_linesearch` (20) times, halving (`α *= 0.5`):

```julia
rollout with u_k = ū_k + K_k δx_k + α d_k,  δx via state_diff (error state)
  guards: ‖x‖∞ or ‖u‖∞ > 1e8 or NaN -> status STATE_LIMIT/CONTROL_LIMIT, α *= ϕ, continue
J = cost(obj, Z̄)
expected = -α (ΔV[1] + α ΔV[2])
z = (J_prev - J) / expected            # Armijo ratio
accept if z_lb ≤ z ≤ z_ub  (1e-8, 10.0)
if 0 < expected < expected_decrease_tolerance (1e-10): α = 0, Z̄ = Z, J = J_prev,
    increaseregularization!
if expected ≤ 0: z = -1 (keep searching)
on max line-search iters: α = 0, Z̄ = Z, J = J_prev, increaseregularization!, ρ += bp_reg_fp,
    stats.ls_failed = true
if J > J_prev: status = COST_INCREASE, return NaN
```

### 4.4 Gradient and convergence (`ilqr_solve.jl`)

- `gradient!`: `grad_k = max_i |d_k[i]| / (|u_k[i]| + 1)`, average over knots — the primal
  optimality residual used by `gradient_tolerance`.
- `evaluate_convergence`, in order:
  1. `0 ≤ dJ < cost_tolerance && grad < gradient_tolerance && !ls_failed` → `SOLVE_SUCCEEDED`
  2. `iterations ≥ opts.iterations` → `MAX_ITERATIONS`
  3. `dJ_zero_counter > dJ_counter_limit` (10 consecutive `dJ ≈ 0`) → `NO_PROGRESS`
  4. `J > max_cost_value` → `MAXIMUM_COST`

## 5. Augmented Lagrangian (`src/augmented_lagrangian/`)

### 5.1 ALObjective

`ALObjective = obj + ALConstraintSet`. `cost()` = unconstrained cost + Σ per-knot AL penalty.
`cost_expansion!` = `cost_expansion!(obj, E, Z)` then per constraint:
`constraint_jacobians!`, `algrad!`, `alhess!`, and `add_alcost_expansion!` adds
`grad`/`hess` into the iLQR cost expansion `E`. **The iLQR inner solver never sees the
constraints — box or otherwise.** This is the composition seam the Python port must
reproduce: box-iLQR = plain iLQR on an AL-modified expansion; ALTRO = outer loop around it.

### 5.2 ALConstraint state (per constraint, per applied knot, per row)

- `λ` duals (p-vector per knot), `μ` penalties (p-vector per knot, all rows start at
  `penalty_initial`), `μinv = 1/μ`, `vals` c(x), `jac` ∇c, `viol`, `c_max`.
- `ConstraintOptions`: per-constraint overrides of `use_conic_cost`, `penalty_initial`,
  `penalty_scaling`, `penalty_max`, `dual_max`; defaults come from `SolverOptions`.

### 5.3 Penalty cost, gradient, Hessian (`use_conic_cost = false`, the default)

Equality:

```julia
alcost  = λ'c + 0.5 c' diag(μ) c
algrad  = ∇c' λbar,        λbar = λ + μ∘c
alhess  = ∇c' diag(μ) ∇c                       # Gauss–Newton, no second-order term
dualupdate: λ ← λ + μ∘c, clamp to ±dual_max
```

Inequality (box bounds map to `Inequality`, so this is the box path):

```julia
active  = (c ≥ 0) | (λ > 0)                    # per row
a       = active ∘ μ
alcost  = λ'c + 0.5 c' diag(a) c
algrad  = ∇c' λbar,        λbar = λ + a∘c
alhess  = ∇c' diag(a) ∇c                       # Gauss–Newton
dualupdate: λ ← max(0, λ + μ∘c), clamp to ±dual_max
```

Generic conic (`use_conic_cost = true`): λbar = λ − μ∘c; λp = Π_{K*}(λbar) (projection onto
dual cone); λs = μinv∘λp; cost = 0.5(λp'λs − λ' diag(μinv) λ); grad = −∇c' ∇Π' λs;
hess = ∇c' Iμ (∇²Π(λs) + ∇Π' Iμ⁻¹ ∇Π) Iμ ∇c (includes second-order projection term);
dualupdate: λ ← Π_{K*}(λ − μ∘c).

Penalty update (all cones): `μ ← clamp(μ·penalty_scaling, 0, penalty_max)`, `μinv = 1/μ`.
Called after every outer iteration unconditionally.

### 5.4 Violation

`viol = Π_K(c) − c` per knot; `max_violation` = max over knots/constraints of `‖viol‖_∞`.
Python already has `max_violation` for the transcription path — reuse it.

### 5.5 ALSolver outer loop (`al_solve.jl`)

```julia
for al_iter = 1..iterations_outer:
    set_tolerances!: if not last iteration, cost_tolerance ← cost_tolerance_intermediate,
        gradient_tolerance ← gradient_tolerance_intermediate
    solve!(ilqr)
    status > SOLVE_SUCCEEDED && break
    J = cost(solver, Z̄); c_max = max_violation(conset); μ_max = max_penalty(conset)
    record outer iteration
    evaluate_convergence:
        c_max < constraint_tolerance              -> SOLVE_SUCCEEDED
        kickout_max_penalty && μ_max ≥ penalty_max -> converged (no status set)
        iterations ≥ opts.iterations              -> MAX_ITERATIONS
        iterations_outer ≥ opts.iterations_outer  -> MAX_ITERATIONS_OUTER
    dualupdate!(conset); penaltyupdate!(conset)
    reset!(ilqr)
```

Note: `reset_duals`/`reset_penalties` options exist but the outer loop shown here never
resets λ/μ between outer iterations (they persist, which is the point of AL warm-starting).
`reset!` of the ALSolver calls `reset!(conset)` (all constraints: `resetparams!`,
`reset_duals!`, `reset_penalties!`) only in `ALSolver`'s own `reset!`, which runs before the
first outer iteration.

## 6. ALTRO (`src/altro/altro_solve.jl`)

```julia
solve!(ALTROSolver):
    if isempty(conSet): solve!(ilqr) and return            # unconstrained shortcut
    if opts.projected_newton:
        if projected_newton_tolerance ≥ 0: AL constraint_tolerance ← projected_newton_tolerance (1e-3)
        else: AL constraint_tolerance ← 0, kickout_max_penalty ← true
    solve!(solver_al)
    if status ≤ SOLVE_SUCCEEDED or force_pn:
        c_max = max_violation(conset)
        if (projected_newton && c_max > constraint_tolerance && status ∈ {≤SUCCEEDED, MAX_ITERATIONS_OUTER}) or force_pn:
            copy Zal -> Zpn; solve!(solver_pn); copy back
        backup check: evaluate constraints; if c_max < constraint_tolerance: status = SOLVE_SUCCEEDED
```

Phase 2 (Projected Newton, `src/direct/`): active-set sparse KKT solve (QDLDL),
iterative refinement with a violation-based line search. `multiplier_projection` is a
no-op in Altro master (issue #35: the projection step is `return Inf`).

**Decision (user):** the PN phase is in scope and runs by default (as in Altro,
`projected_newton = true`, entering when the AL phase gets `c_max` below
`projected_newton_tolerance`). The multiplier projection is ported but gated behind the
`multiplier_projection` option: default `true` to mirror Altro's option value, with the
option explicitly able to disable it. Document in the port that Altro's own implementation
is currently disabled (issue #35), so the Python port's projection is a superset — parity
tests for the projection step compare against the option turned off on both sides.

**Solver choice (decision):** the port uses a **dense KKT solve** instead of Altro's sparse
QDLDL factorization (`src/qdldl.jl`, `triukkt!`). For the port's fixed small n/m this is
numerically equivalent and much simpler; the divergence is noted in the solver docstring.

## 7. Parity-relevant divergences from the current Python code

### 7.1 Quantified: Gauss-Newton vs full penalty Hessian

Measured with `scratch/gn_vs_full_hessian.py` on a cartpole knot point, Altro's default
inequality penalty `P = lam'c + 0.5 c' diag(a) c`, `a = active .* mu`:

| constraint | violation c | rel. Frobenius diff | H_gn eig | H_full eig |
| :--- | ---: | ---: | --- | --- |
| `ControlBound` (affine) | 1.0 | **0.0** | (0, 2) | (0, 2) |
| `GoalConstraint` (affine) | mixed | **0.0** | (0, 2) | (0, 2) |
| `CircleConstraint` (nonlinear) | 0.05 | **0.18** | (0, 1.6) | **(-0.2, 1.4)** |
| `CircleConstraint` (nonlinear) | 0.25 | inf (GN = 0) | (0, 0) | **(-1, 0)** |
| `CircleConstraint`, mu = 2e3 | 0.25 | inf | (0, 0) | **(-1000, 0)** |

Three conclusions:

1. **Zero divergence for every affine constraint** — all box bounds and `GoalConstraint`
   have `∇²c = 0`, so the GN Hessian is exactly the true Hessian. The box rung of the
   ladder has no divergence to worry about.
2. **For nonlinear constraints the difference is `Σᵢ (λᵢ + aᵢcᵢ) ∇²cᵢ`** — it scales as
   penalty × violation × constraint curvature, and vanishes at the solution (c → 0). Both
   variants converge to the same KKT point; only the path differs.
3. **The full Hessian is indefinite** (negative eigenvalues appear exactly where
   `mu*c*hess(c)` dominates), while GN is PSD by construction. Indefinite `Quu` blocks
   fail the Cholesky in the backward pass → regularization restarts → slower inner solves.

### 7.2 Cost of the current full-Hessian approach

- One extra backward-over-backward AD pass per knot per constraint (`jax.hessian` of the
  penalty) versus a `J' diag(a) J` matmul on Jacobians the AL machinery already computes.
- Non-smoothness: the conic penalty's projection has kinks at active-set boundaries; exact
  second derivatives are ill-defined there. GN sidesteps this entirely.
- Indefiniteness as in §7.1, which degrades the Riccati.

There is **no technical obstacle to implementing it like Altro**: `H_gn =
J.T @ diag(a) @ J` is a one-liner in JAX given the constraint Jacobians. The current
`_evaluate_knot_penalty` + `jax.hessian` route was the *generic* choice (one smooth form for
all cones), written before a native solver existed. The port replaces it with the
special-cased equality/inequality forms as the default path and keeps the conic form behind
`use_conic_cost = true`, exactly like Altro.

### 7.3 Remaining divergences

1. **μ is per-row, not per-knot scalar.** Altro keeps a p-vector of penalties per knot
   (equal rows, scaled uniformly). Python's `_parse_penalties` is per-knot scalar.
   Harmless while all rows scale identically, but the λ/μ layout in `MPCState` must match
   for cross-testing.
2. **Gain sign and policy convention.** Altro stores the negated gain `K = −Quu⁻¹ Qux`,
   rollout `u = ū + K δx + α d`. The Python port must use the same sign or invert at the
   boundary.
3. **Line search uses expected decrease** `−α(ΔV[1] + αΔV[2])` with a ratio interval
   `[1e-8, 10]`, plus the `expected_decrease_tolerance` no-step branch and `ρ += bp_reg_fp`
   on line-search exhaustion. Not a plain Armijo backtrack.
4. **Convergence uses both** `dJ < cost_tolerance` **and** `grad < gradient_tolerance`
   (gradient = averaged normalized feedforward), plus the `dJ_zero_counter` no-progress exit.
5. **State/control value limits** (1e8) and NaN checks abort the rollout mid-horizon and
   enter the α-halving loop; the Python `rollout_states` has no such guard.
6. **Error-state expansions** are identity for Euclidean models; for rotation models the
   `G`-chain (`fx = G₂'A G₁`, `fu = G₂'B`) matches `_dynamics_expansion` already.

## 8. Cross-verification setup (proposed)

### 8.1 Environment

Altro is vendored at `altro_jl/` (committed, pinned to commit `4864df2`).

1. Register it with the existing `trajopt_jl/` Julia environment via a path develop:
   `julia --project=trajopt_jl -e 'using Pkg; Pkg.develop(path="../altro_jl"); Pkg.resolve()'`
   — no network fetch at test time, and compat is confirmed (`TrajectoryOptimization = "0.7"`
   matches vendored 0.7.1, `RobotDynamics = "0.4"` matches).
2. Extend `test/conftest.py` with a second session fixture (keep `jl_to` untouched so
   existing cross tests don't pay the Altro load cost): activate `trajopt_jl`,
   `using TrajectoryOptimization`, `using Altro`, and expose
   `Altro.iLQRSolver`, `Altro.ALSolver`, `Altro.ALTROSolver`, `Altro.SolverOptions`,
   `Altro.solve!`. All phase functions below are module functions, reachable through
   juliacall without being exported.

### 8.2 Unit-level parity tests (every phase, not just end-to-end)

Both sides start from an identically built problem (same model, discretization, initial
trajectory — the existing `test_cross_*` machinery already guarantees this). The Python
port must expose the same phase functions as Altro so each test drives both sides one phase
at a time and compares intermediate quantities to ~1e-8.

| # | Julia phase (called via juliacall) | Python counterpart | compared quantity |
| :-- | :--- | :--- | :--- |
| 1 | `dynamics_expansion!`, `errstate_jacobians!`, `error_expansion!` | `Expansion` (A, B) | A, B per knot |
| 2 | `cost_expansion!` + `error_expansion!` | `_cost_expansion` | q, r, Q, R, H per knot |
| 3 | `backwardpass!` | `backward_pass` | K, d, S.x, S.xx, ΔV per knot |
| 4 | `increaseregularization!` / `decreaseregularization!` | reg state machine | ρ, dρ after each event |
| 5 | `rollout!(solver, α)` | `_forward_pass(α)` | Z̄ trajectory for fixed α |
| 6 | `forwardpass!(solver, J_prev)` | `forward_pass` | accepted α, J, expected decrease, ls_failed |
| 7 | `gradient!(solver)` | `gradient` | per-knot and average gradient |
| 8 | `evaluate_convergence(solver)` | `evaluate_convergence` | status decision per state |
| 9 | `initialize!` / `reset!` | `initialize` / `reset` | Z after initial rollout, reg reset |
| 10 | `ALConstraint` cost/grad/hess (`alcost`, `algrad!`, `alhess!`) | AL penalty cost, grad, hess | penalty blocks per knot |
| 11 | `add_alcost_expansion!` | `_augmented_lagrangian_expansion` (GN form) | E.grad/hess deltas |
| 12 | `dualupdate!(conset)` | dual update | λ after one outer step |
| 13 | `penaltyupdate!(conset)` | penalty update | μ after one outer step |
| 14 | `max_violation(conset)` / `max_penalty(conset)` | `max_violation` / `max_penalty` | c_max, μ_max |
| 15 | `ALSolver.solve!` outer loop + `set_tolerances!` | `solve_altro` outer loop | full (cost, c_max, μ_max) history |
| 16 | `ProjectedNewtonSolver` (`_qdldl_solve!`, `update_active_set!`, line search) | PN phase | active set, KKT step, violation reduction |
| 17 | full `ALTROSolver.solve!` | `solve_altro` | final cost, trajectory, status, iterations |

End-to-end scenario tests (layer on top of the phase tests):

- LQR: linear dynamics + quadratic cost → iLQR converges in 1 iteration; gains match the
  analytic LQR gain on both sides.
- Box: cartpole with `ControlBound(u_min, u_max)` and a goal — Julia `ALTROSolver` vs
  Python, compare cost, trajectory, λ/μ, iterations. OSQP/IPopt remain secondary checks.
- ALTRO: swing-up with `GoalConstraint`; assert both reach `max_violation <
  constraint_tolerance` and match terminal cost.
- Options parity: exercise the same kwargs on both sides (`set_options!`-equivalent)
  — `penalty_initial`, `penalty_scaling`, `cost_tolerance`, `gradient_tolerance`,
  `constraint_tolerance`, `line_search_*`, `bp_reg_*` — and compare trajectories.

Julia-side knobs to replicate exactly: `use_static = Val(false)` is not portable; use
`StaticReturn`/`InPlace` consistently on both sides and the same models as the existing
cross tests (pendulum/cartpole/quadrotor from `models/`).

### 8.3 Phase hooks needed in the Python port

The solver module must be structured as the phase functions in the table (mirroring Altro's
file split `ilqr_solver.jl` / `backwardpass.jl` / `forwardpass.jl` / `ilqr_solve.jl` /
`alcon.jl` / `alconset.jl` / `al_solve.jl` / `altro_solve.jl` / `pn_solve.jl`), so a cross
test can run, e.g., five Julia backward passes and five Python backward passes from the same
trajectory and compare gains each time. This also gives free unit tests in `test/unit/`
with no Julia dependency.

## 9. Resolved decisions

- **AL cost form:** mirror Altro's GN Hessian (`J' diag(a) J`) for the default
  equality/inequality path; the existing full-Hessian conic penalty in `expansions.py` is
  replaced. `use_conic_cost` **is ported** (Altro's conic cost/grad/hess with dual-cone
  projections), matching Altro's option semantics.
- **Projected Newton:** ported and run by default; dense KKT instead of QDLDL — divergence
  noted in the solver docstring. `multiplier_projection` ported, gated, default `true`.
- **Status:** public 4-value `SolverStatus`; full 10-value `TerminationStatus` computed and
  recorded internally.
- **Options:** Altro-only options discarded (`static_bp`, `save_S`,
  `closed_loop_initial_rollout`, `square_root`, `reuse_jacobians`); everything else ported
  with Altro's names and defaults.
- **Vendoring:** Altro committed at `altro_jl/`; cross tests develop it into the
  `trajopt_jl/` Julia env (see §8.1).
