# 26 — Forward pass, rollout, and line search

**What to build:** take the affine policy from the backward pass and find a step along it that
actually reduces cost. Two pieces you can drive independently: a closed-loop rollout at a fixed
step length α, and the line search that picks α. Demonstrable on its own — for a given trajectory
and policy, the rollout matches `Altro.rollout!(solver, α)` knot for knot, and the line search
picks the same α and reports the same accepted cost, expected decrease, and failure flag as
`Altro.forwardpass!`.

**Blocked by:** 25.

## Architecture

`src/trajopt/solvers/ilqr.py`, beside the backward pass.

**The rollout is a `scan` over knots, not a call into `dynamics/rollout.py`.** The existing
`rollout_states` propagates an open-loop control sequence; this one needs the closed-loop form
`u_k = ū_k + K_k δx_k + α d_k` where `δx` comes from the model's error-state difference, and it
needs the numerical guards `rollout_states` has no notion of (reference §7.3 item 5). Rather than
adding an α-and-gains parameter to the existing function and contorting it, write the closed-loop
rollout here and leave `rollout_states` alone.

**Guards abort the sweep, and under trace "abort" means masking.** Altro returns `false` from
`rollout!` the moment `‖x‖∞` or `‖u‖∞` exceeds `1e8` or goes NaN, leaving the remaining knots
unwritten. A `scan` cannot stop early; carry a `failed` flag and let subsequent steps compute
garbage that the flag invalidates. The distinction that matters is which of `STATE_LIMIT` /
`CONTROL_LIMIT` is reported, so carry that as a status int too — first failure wins.

**The line search is a `while_loop` over α** carrying `(α, J, z, status, reg, done)`. Altro's
loop has four exits and they are not mutually exclusive; reproduce all four:

1. Rollout tripped a guard → `α *= ϕ`, retry. Note this path `continue`s **past** the max-iters
   check (finding J), so exhausting all 20 attempts this way leaves `ls_failed` false, `J` at
   `Inf`, and α at `ϕ²⁰` — and the function then exits through the cost-increase check below.
   Reproduce the quirk; a parity test will catch you if you tidy it.
2. `0 < expected < expected_decrease_tolerance` → take no step at all: `α = 0`, restore the
   previous trajectory, `J = J_prev`, increase regularization, exit.
3. `z_lb ≤ z ≤ z_ub` where `z = (J_prev − J) / expected` and `expected = −α(ΔV₁ + αΔV₂)` → accept.
   When `expected ≤ 0`, Altro sets `z = -1` so the interval test fails and the search continues.
4. Iteration 20 with nothing accepted → `α = 0`, restore, `J = J_prev`, increase regularization,
   **then additionally** `ρ += bp_reg_fp`, set `ls_failed`. Exits 2 and 4 can fire on the same
   iteration, which double-increments the regularization. That is upstream behaviour, not a bug
   to fix.

This is not a plain Armijo backtrack and should not be simplified into one (reference §7.3
item 3). The ratio has an **upper** bound as well as a lower one.

**The cost-increase exit is outside the loop.** If the final `J > J_prev`, set `COST_INCREASE`
and return NaN. Ticket 27 must handle a NaN `dJ` without treating it as convergence — Altro's
`0.0 <= dJ` comparison is false for NaN, which is what makes this work.

## Julia parity

Reference §8.2 rows 5 and 6: the rolled-out trajectory for a fixed α, and the accepted α, cost,
expected decrease, and `ls_failed` from a full line search. Drive both from the same trajectory
and the same gains on each side.

## Acceptance criteria

- [ ] A closed-loop rollout takes gains, α, and a nominal trajectory and returns the new trajectory plus a guard status; it runs under `jax.jit`.
- [ ] Guard thresholds and NaN detection fire on both state and control, and report `STATE_LIMIT` / `CONTROL_LIMIT` respectively, first failure winning.
- [ ] With α = 0 and zero feedforward the rollout reproduces the nominal trajectory exactly.
- [ ] The line search reproduces all four exits, each covered by a test that reaches it: guard retry, expected-decrease-too-small, acceptance, and iteration exhaustion.
- [ ] The guard-exhaustion path leaves `ls_failed` false and exits through the cost-increase check, matching finding J.
- [ ] Acceptance uses the two-sided ratio interval `[1e-8, 10]`, with `expected ≤ 0` forcing continuation; a test covers a rejection caused by the **upper** bound.
- [ ] Iteration exhaustion increases regularization *and* adds `bp_reg_fp` to ρ *and* sets `ls_failed`.
- [ ] Cross tests match `Altro.rollout!` and `Altro.forwardpass!` to 1e-8 on pendulum and cartpole, including a case where the line search fails.
- [ ] Targeted checks green — `uv run pytest` on this ticket's own test files, `uv run ty check` on the modules it touched, `uv run ruff check --fix`. No full-suite run and no `pre-commit --all-files` gate: this ticket only adds modules under `solvers/`.
