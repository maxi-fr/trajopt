# ALTRO Solver Optimization, Parity, and Benchmark Diagnosis

This document provides the complete diagnostic findings, profiling results, theoretical analysis, and implementation roadmap for optimizing the native JAX ALTRO solver (`src/trajopt/solvers/`) and resolving performance regressions against Ipopt and the Julia reference (`Altro.jl`).

An incoming agent continuing this work should use this document as the authoritative specification.

---

## 1. Executive Summary & Benchmark Baseline

Benchmarking was conducted on the three reference problems in `src/trajopt/benchmarks.py` on horizon $N=25$ (discretization step $dt=0.05$s or $0.1$s):

| Problem | Horizon & Structure | Ipopt Runtime | ALTRO Runtime | ALTRO Status & Convergence |
|---|---|---|---|---|
| **Dubins Corridor** | $N=25, dt=0.1$, bounds on $y, v, \omega$ | **550.5 ms** (23 iters) | **24.3 ms** (`PN=False`)<br>**100.8 ms** (`PN=True`) | **22x faster than Ipopt** on AL alone. Projected Newton adds ~76 ms of dense KKT overhead. |
| **Cartpole Swingup** | $N=25, dt=0.05$, bounds on $\|u\| \le 20$, $\|x_0\| \le 0.4$ | **552.9 ms** (47 iters) | 452.1 ms | **FAILED (`MAXIMUM_COST`)**: Cart moves left to $-1.31$m; penalty $\mu$ scales to $10^8$; $0.5 \mu c^2 > 10^8$ triggers `max_cost_value` abort. |
| **Quadrotor Obstacle** | $N=25, dt=0.05$, $SO(3)$ attitude, spherical keep-out | **1107.8 ms** (49 iters) | 958.3 ms | **FAILED (`NO_PROGRESS`)**: Inner iLQR stalls on non-convex obstacle curvature; outer AL immediately aborts on outer iter 1. |

Run the benchmarks again to confirm.

### The Two Core Failure Modes

1. **Premature Outer Abort on Inner Stalls**:
   In `src/trajopt/solvers/al.py:860`, `inner_failed = inner_status > SOLVE_SUCCEEDED`. When inner iLQR exits with `NO_PROGRESS` (8) or `MAX_ITERATIONS` (3), the outer AL loop treats this as a fatal error, immediately aborting the solve and skipping dual/penalty updates.
2. **Dense KKT & Unconditional Execution of Projected Newton (PN)**:
   - PN's dense $O((N(n+m+p))^3)$ system is assembled and solved with `jnp.linalg.solve`, whereas Julia uses sparse `QDLDL` on active constraints only.
   - PN executes dead upstream code (`options.multiplier_projection=True`), adding a second dense solve of dimension $N_d \times N_d$ (cutting runtime by 2.3x when disabled).
   - In `src/trajopt/solvers/altro.py:187`, `altro_solve` always calls `pn_solve` and masks with `jnp.where(run_pn, pn_traj, al_traj)` instead of `jax.lax.cond`, paying the dense solve cost even when AL already converged.
3. **Redundant Jacobian Evaluations during Line Search**:
   `al.py:645` calls `evaluate_al_constraints` inside `_ALObjective.cost(traj)`. Every trial step $\alpha$ in iLQR's line search (up to 20 times per iLQR iteration) computes full constraint Jacobians, bound Jacobians, and error-state einsums, only to discard them.

---

## 2. Upstream Julia Bug & Literature Grounding

### Upstream Reference Flaw in `Altro.jl`

In Julia `altro_jl/src/augmented_lagrangian/al_solve.jl:43`:

```julia
# Check solver status
status(solver) > SOLVE_SUCCEEDED && break
```

In `altro_jl/src/ilqr/ilqr_solve.jl:168-174`:

```julia
# Outer loop update if forward pass is repeatedly unsuccessful
if solver.stats.dJ_zero_counter > solver.opts.dJ_counter_limit
    @log lg "info" "dJ Counter hit max. Terminating" :append
    solver.stats.status = NO_PROGRESS
    return true
end
```

Notice the comment: **`# Outer loop update if forward pass is repeatedly unsuccessful`**.
The original authors intended for `NO_PROGRESS` to terminate the *inner* iLQR iteration so the *outer* Augmented Lagrangian loop could perform an outer multiplier and penalty update ($\lambda \leftarrow \lambda + \mu c(x), \mu \leftarrow \phi \mu$) to modify the cost landscape and unstick the solver.

However, because `status(solver)` in Julia accesses `solver.stats.status` (which was set to `NO_PROGRESS = 8`), line 43 in `al_solve.jl` evaluated `8 > 2` (`SOLVE_SUCCEEDED = 2`) and executed `break`, terminating the entire AL optimization prematurely!

### Primary Literature & Commit History Guidance

The incoming agent should consult the following primary sources when resolving this:

1. **Jackson et al.**: *AL-iLQR tutorial*
   file:///C:/Users/frank/OneDrive%20-%20stud.tu-darmstadt.de/Master/Semester%205/Projekseminar%20Mechatronik/Literature/Controller%20Ideas/AL_iLQR_Tutorial.pdf

2. **`Altro.jl` Repository Commit History**:
   - Pinned submodule: `altro_jl/` at commit `4864df2`.
   - Inspect commits in `RoboticExplorationLab/Altro.jl` touching `al_solve.jl`, `alcon.jl`, and `ilqr_solve.jl` around the handling of `dJ_counter_limit` and `kickout_max_penalty`.

---

## 3. Redesigning Cross-Tests: From Full Solves to Unit-Level Invariants

### Why Monolithic Cross-Tests Failed to Catch Solver Breakdowns

Existing cross-tests in `test/cross_verification/test_cross_altro_al_solve.py` and `test_cross_altro_solve.py` tested an easy Cartpole problem:

- Horizon $T = 5.0$ seconds ($N=101, dt=0.05$).
- Control bounds $\|u\| \le 20.0$.
- Terminal goal constraint $x(T) = [0, \pi, 0, 0]$.
- **Zero state bounds and zero obstacle constraints**.

On this problem, the pendulum swings up with plenty of track space; every single inner iLQR solve converges with `SOLVE_SUCCEEDED` (status 2). The code paths for `inner_status == NO_PROGRESS` or `inner_status == MAX_ITERATIONS` were never triggered. Furthermore, because upstream Julia possesses the exact same break condition on `status > SOLVE_SUCCEEDED`, end-to-end full solve tests reported parity while both solvers were brittle.

### Required Unit-Level Cross-Test Invariants

Instead of comparing monolithic 50-iteration trajectory solves, write targeted unit tests in `test/cross_verification/` and `test/unit/`:

1. **Inner-to-Outer AL State Machine Transition**:
   - Feed synthetic `inner_status` values (`SOLVE_SUCCEEDED`, `NO_PROGRESS`, `MAX_ITERATIONS`, `COST_INCREASE`) into `_al_step`.
   - Verify that `NO_PROGRESS` and `MAX_ITERATIONS` result in dual and penalty updates and continue the outer loop (`done == False`).
   - Verify that true fatal statuses (e.g. `NAN_DETECTED`, `STATE_LIMIT`) terminate with `done == True`.
2. **Penalty Update Rule Invariant**:
   - Compare `dual_update` and `penalty_update` outputs row-by-row against Julia's `dualupdate!` and `penaltyupdate!` in `alcon.jl`.
   - Test penalty clamping at `penalty_max` and verify that `0.5 * mu * c_max^2` cannot overflow `max_cost_value`.
3. **Residual vs. Jacobian Evaluation Equivalence**:
   - Verify that `evaluate_al_residuals(al, constraints, traj)` returns values bitwise identical to `evaluate_al_constraints(al, constraints, model, traj)[0]`.
   - Measure microbenchmarks confirming residual evaluation is $\ge 5\times$ faster than full Jacobian evaluation.
4. **Projected Newton Step Verification**:
   - Test a single step of `_solve_kkt_step` on a known active set against Julia's `_qdldl_solve!` with `multiplier_projection=False`.
   - Verify that inactive constraint rows act as identities without altering primal search directions.

---

## 4. Optimization Roadmap & Action Plan

### Phase 1: JAX Performance Optimizations (Immediate 2x–5x Speedup)

#### 1.1 Residual-Only Evaluation in AL Line Search

- **File**: `src/trajopt/solvers/al.py`
- **Problem**: `_ALObjective.cost(traj)` calls `evaluate_al_constraints`, computing $Jx, Ju$, box bound Jacobians, and `einsum("kpn,kne->kpe", Jx, G_all)` on every line search step.
- **Solution**:
  Create `evaluate_al_residuals(al, constraints, traj) -> jax.Array`:

  ```python
  def _evaluate_constraint_residuals(constraints, X, U, T, p_cons_max):
      N = X.shape[0]
      C = jnp.zeros((N, p_cons_max), dtype=X.dtype)
      for g in constraints.groups:
          p_g = g.evaluator.p
          if p_g == 0:
              continue
          C = C.at[jnp.asarray(g.knots), :p_g].set(g.evaluate(X, U, T))
      return C

  def _evaluate_bound_residuals(constraints, X, U):
      m = U.shape[1]
      x_upper, x_lower = constraints.x_upper, constraints.x_lower
      U_pad = jnp.concatenate([U, jnp.zeros((1, m), dtype=X.dtype)], axis=0)
      u_upper_pad = jnp.concatenate([constraints.u_upper, jnp.full((1, m), jnp.inf, dtype=X.dtype)], axis=0)
      u_lower_pad = jnp.concatenate([constraints.u_lower, jnp.full((1, m), -jnp.inf, dtype=X.dtype)], axis=0)

      x_upper_safe = jnp.where(jnp.isfinite(x_upper), x_upper, 0.0)
      x_lower_safe = jnp.where(jnp.isfinite(x_lower), x_lower, 0.0)
      u_upper_safe = jnp.where(jnp.isfinite(u_upper_pad), u_upper_pad, 0.0)
      u_lower_safe = jnp.where(jnp.isfinite(u_lower_pad), u_lower_pad, 0.0)

      return jnp.concatenate([X - x_upper_safe, x_lower_safe - X, U_pad - u_upper_safe, u_lower_safe - U_pad], axis=-1)

  def evaluate_al_residuals(al, constraints, traj):
      C_cons = _evaluate_constraint_residuals(constraints, traj.X, traj.U, traj.t, al.p_cons_max)
      C_bound = _evaluate_bound_residuals(constraints, traj.X, traj.U)
      return jnp.concatenate([C_cons, C_bound], axis=-1)
  ```

  Update `_ALObjective.cost`, `_al_step` (line 862), `al_solve` (line 1099), and `altro_solve` (lines 174, 191) to use `evaluate_al_residuals`. Only `_ALProblem.cost_expansion` needs full `evaluate_al_constraints`.

#### 1.2 Disable Dead-Code `multiplier_projection` by Default

- **File**: `src/trajopt/solvers/options.py:122`
- **Change**: Change default from `multiplier_projection: bool = True` to `multiplier_projection: bool = False`.
- **Impact**: Eliminates assembling an $N_d \times N_d$ Gram matrix and performing a dense solve. Cuts PN Dubins solve time from 480 ms to 205 ms. Matches shipped Julia behavior where `multiplier_projection!` is commented out.

#### 1.3 Conditional Execution of Projected Newton

- **File**: `src/trajopt/solvers/altro.py:180-195`
- **Problem**: `pn_solve` is always executed, even if `run_pn` is False or `options.projected_newton` is False.
- **Solution**:
  - If `not options.projected_newton and not options.force_pn`: bypass PN entirely in Python (return `al_traj` and dummy stats).
  - When `options.projected_newton` is True: wrap `pn_solve` in `jax.lax.cond(run_pn, lambda: pn_solve(...), lambda: (al_traj, empty_stats, empty_duals, al_status))`.

---

### Phase 2: Augmented Lagrangian Robustness & Convergence Fixes

#### 2.1 Soft Inner Exits in `_al_step`

- **File**: `src/trajopt/solvers/al.py:860-885`
- **Problem**: Any `inner_status > SOLVE_SUCCEEDED` aborts the outer loop.
- **Solution**:
  Distinguish between soft stalls and hard numerical errors:

  ```python
  inner_stalled = (inner_status == jnp.int32(TerminationStatus.NO_PROGRESS)) | (
      inner_status == jnp.int32(TerminationStatus.MAX_ITERATIONS)
  )
  inner_fatal = (inner_status > jnp.int32(TerminationStatus.SOLVE_SUCCEEDED)) & (~inner_stalled)

  conv_status, conv_done = _evaluate_al_convergence(c_max, mu_max, inner_stats.iterations, iter_num, options)
  final_done = inner_fatal | conv_done
  skip_dual_update = inner_fatal | conv_done
  ```

#### 2.2 Penalty Capping vs `max_cost_value`

- **Problem**: With default `penalty_max = 1e8` and `max_cost_value = 1e8`, any violation $c > 1.41$ when $\mu = 10^8$ causes $0.5 \mu c^2 > 10^8$, triggering `MAXIMUM_COST`.
- **Solution**:
  - In `_evaluate_al_convergence`, check `kickout_max_penalty`: if $\mu_{\max} \ge \mu_{\text{threshold}}$, exit with `MAX_ITERATIONS_OUTER` rather than letting quadratic penalties blow past `max_cost_value`.
  - Alternatively, implement Powell's violation reduction check: only scale $\mu \leftarrow \phi \mu$ for constraints where violation failed to decrease by $\gamma = 0.25$; otherwise leave $\mu$ unchanged and update only duals $\lambda$.

---

### Phase 3: Projected Newton Linear Solver Architecture

#### 3.1 Sparse vs. Block-Banded KKT Solvers in JAX

- **Background**: Julia uses sparse `QDLDL`. JAX tracing prefers static shapes.
- **Path Forward**:
  The KKT system for trajectory optimization is block-tridiagonal (dynamics decouple stages $k$ and $k+1$).
  Rather than a generic sparse solver or a flat dense matrix solve:
  - Implement a **block-tridiagonal Riccati / Thomas algorithm** in JAX using `lax.scan`.
  - Stages have static block sizes $(n \times n, n \times m, m \times m)$. Inactive constraint rows can be handled via diagonal masking.
  - Runtime scales as $O(N(n+m)^3)$ instead of dense $O(N^3(n+m)^3)$, reducing PN computation for long horizons by orders of magnitude.

---

### Phase 4: Potential Algorithmic Improvements (Documented in `TODO.md`)

- **Wiring BoxQP into ALTRO Driver**:
  In `boxqp.py`, `BoxQP` solves control-bounded problems without AL penalties by using projected Newton in the backward pass. Julia `Altro.jl` does *not* use BoxQP (it penalizes control bounds via AL). Wiring BoxQP into `ALTRO.solve` is an algorithmic divergence from Julia, but provides substantial robustness on control-limited systems like Cartpole. Recorded in `TODO.md`.

---

## 5. Summary of Implementation Checklist

1. [ ] **`evaluate_al_residuals`**: Add to `al.py` and replace in `cost()`, `_al_step`, and `altro_solve`.
2. [ ] **`multiplier_projection = False`**: Update default in `options.py`.
3. [ ] **Conditional PN**: Wrap `pn_solve` in `jax.lax.cond(run_pn, ...)` in `altro.py`.
4. [ ] **Soft Inner Exits in AL**: Update `_al_step` in `al.py` to allow `NO_PROGRESS` and `MAX_ITERATIONS` to proceed to outer updates.
5. [ ] **Unit-Level Cross-Tests**: Add unit test cases for AL state machine transitions and residual evaluation in `test/cross_verification/`.
6. [ ] **Benchmark Re-evaluation**: Run `scripts/benchmark_altro_vs_ipopt.py` and verify speedups across Dubins, Cartpole, and Quadrotor.
