# 27 — Plain iLQR solves an unconstrained problem

**What to build:** `problem.solve(state, solver=ILQR())` returns an optimized `MPCState`. This is
the first ticket you can point at and say the port works: a linear-quadratic problem converges in
one iteration with gains equal to the analytic LQR solution, and a pendulum swing-up converges to
the same trajectory and cost as `Altro.iLQRSolver` to 1e-8.

**Blocked by:** 26.

## Architecture

`src/trajopt/solvers/ilqr.py` gains the driver; the solver object lives there too.

**Two layers, and the seam between them is where tracing stops.** The traced core is a pure
function `(problem, trajectory, options) -> (trajectory, stats, status_int)` built from a
`lax.while_loop` over iterations, jittable and vmappable end to end. `ILQR` is a frozen dataclass
satisfying the existing `Solver` protocol; its `.solve()` calls the core, then converts at the
boundary — status int to `TerminationStatus` to the public `SolverStatus`, stats buffers trimmed
to the counter, `success` / `message` / `info` synthesized. The protocol's `message: str` and
`info: dict` cannot exist inside a trace, and that is the whole reason for the wrapper.

Conforming to the protocol is what makes `ILQR()` a drop-in for `Ipopt()` across the ~47 existing
call sites. `SolverResult` also wants a flat `Z` and canonical-order `lam` / `mu` — reuse
`transcription/layout.py`'s `_trajectory_to_z` for the former; `lam` / `mu` stay empty here, since
an unconstrained iLQR has no duals. Ticket 29 fills them.

**The loop body follows reference §4.1 exactly**, and the order matters: cost of the *current*
trajectory first, then expansions, backward pass, forward pass, unconditional accept, then `dJ`
and the gradient computed on the **accepted** trajectory. Delegate the expansions to the existing
`problem.dynamics_expansion(traj)` and `problem.cost_expansion(traj)` — no new expansion code.

**`gradient` is not a cost gradient.** It is `mean_k max_i |d_k[i]| / (|u_k[i]| + 1)`, a
normalized feedforward magnitude standing in for the primal optimality residual. Compute it after
the accept, on the new controls.

**Convergence is first-match-wins here, and that is unusual.** The iLQR `evaluate_convergence`
genuinely returns on the first hit, unlike the AL outer loop in ticket 29 (finding A). Under trace
that is a `jnp.where` chain in Altro's order: cost criterion, then max iterations, then
`dJ_zero_counter`, then max cost. The first criterion needs all three of `0 ≤ dJ < cost_tolerance`,
`grad < gradient_tolerance`, and `not ls_failed`.

**NaN `dJ` must not read as convergence.** A failed forward pass returns NaN cost, so `dJ` is NaN.
`0.0 <= NaN` is false, and `NaN ≈ 0` is false, so a NaN iteration neither converges nor increments
`dJ_zero_counter` — it just burns an iteration. Reproduce that rather than special-casing NaN.

`initialize` does an **open-loop** rollout. Altro's docstring claims cached gains are reused, but
that only happens under `closed_loop_initial_rollout`, which we discard along with the claim.

## Julia parity

Reference §8.2 rows 1, 2, 7, 8, 9, and 17: `A`/`B` per knot; `q`/`r`/`Q`/`R`/`H` per knot; the
per-knot and averaged gradient; the convergence decision for each reachable state; the trajectory
after `initialize`; and a full solve compared on final cost, trajectory, status, and iteration
count.

## Acceptance criteria

- [ ] A traced `solve` core runs under `jax.jit` and under `jax.vmap` over a batch of initial states, with `SolverOptions` static in both.
- [ ] `ILQR` is a frozen dataclass satisfying the `Solver` protocol; `problem.solve(state, solver=ILQR())` returns an `MPCState` with a populated `status`.
- [ ] A linear-quadratic problem converges in one iteration and its gains match the analytic time-varying LQR gains to 1e-10.
- [ ] Pendulum and cartpole swing-ups match `Altro.iLQRSolver` on final cost, trajectory, status, and iteration count to 1e-8.
- [ ] `gradient` implements the normalized-feedforward formula, verified per knot against `Altro.gradient!`.
- [ ] Convergence is checked in Altro's order with first-match-wins; each of the four exits is reached by a test.
- [ ] An iteration whose forward pass fails produces a NaN `dJ` that neither converges nor increments `dJ_zero_counter`.
- [ ] Stats history returned from a finished solve is trimmed to the iteration count and contains no trailing zeros.
- [ ] pre-commit hooks pass
