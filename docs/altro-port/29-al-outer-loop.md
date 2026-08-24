# 29 — Augmented Lagrangian outer loop

**What to build:** constrained problems solve. A cartpole with control bounds and a goal
constraint reaches `max_violation < constraint_tolerance` and matches Julia's `Altro.ALSolver` on
cost, trajectory, λ, μ, and iteration count. The outer loop wraps the iLQR solver from ticket 27
with the penalty machinery from ticket 28, tightening penalties until the constraints hold.

**Blocked by:** 27, 28.

## Architecture

`src/trajopt/solvers/al.py` gains the driver.

**The composition seam is the whole point, and it is one line wide.** Reference §5.1: the iLQR
inner solver never sees a constraint. The AL objective adds its penalty gradient and Hessian into
the cost expansion, and iLQR optimizes that expansion as if it were an unconstrained cost. So this
ticket adds no code inside `ilqr.py` — it supplies a different expansion and a different cost
function. If you find yourself teaching the backward pass about constraints, stop; the design has
gone wrong.

Bound constraints are ordinary `NegativeOrthant` inequality constraints here, handled by the outer
loop like any other. Nothing special happens to them in the backward pass. Ticket 30 adds the
alternative that does treat them specially.

**The outer loop is a `while_loop`** over `iterations_outer`, carrying the trajectory, λ, μ, the
effective tolerances, the stats, and the status int. Body order per reference §5.5: set
tolerances, solve iLQR, break on status, evaluate cost and violation and penalty maxima, record,
check convergence, dual update, penalty update, reset the inner solver.

**`set_tolerances!` becomes a threaded value.** Every iteration but the last uses
`cost_tolerance_intermediate` and `gradient_tolerance_intermediate`; the last uses the real ones.
Altro does this by mutating the shared options object and restoring it afterwards. Compute the
effective pair in the carry instead and pass it to the inner solve — options stay frozen.

**Outer-loop convergence is last-match-wins, and that is a real difference from ticket 27.**
Finding A: `al_solve.jl` runs four independent `if`s with no early return, so a later one
overwrites `status`. Converging on the same iteration that exhausts `iterations_outer` exits
`MAX_ITERATIONS_OUTER`, not `SOLVE_SUCCEEDED`. Reference §5.5's arrow list implies otherwise.
Under trace this is a sequence of `jnp.where` assignments in Altro's order, which reproduces the
overwrite naturally — write it that way deliberately and say so in a comment, or the next reader
will "fix" it.

**`kickout_max_penalty` is broken upstream and cannot be parity-tested.** Finding B: Altro's
branch references an undefined `i` and throws the moment the flag is true. Implement the branch as
Altro clearly intended (`penalty_max[iter] ≥ penalty_max` ends the loop converged, setting no
status), test it against a hand-written expectation rather than against Julia, and note in the
docstring that Julia cannot reach it. Ticket 33 needs this path to work, because ALTRO turns the
flag on whenever `projected_newton_tolerance < 0`.

**The break on inner status is ordinal.** `status > SOLVE_SUCCEEDED` breaks the outer loop —
finding C. With `TerminationStatus` as an ordered `IntEnum` from ticket 24 this is a plain `jnp`
comparison on the carried int and needs nothing special.

**Duals persist across outer iterations.** That is what AL warm-starting means, and reference
§5.5's note is right that the loop never resets them. `reset_duals` / `reset_penalties` apply only
at the start of a whole solve.

The solver object follows ticket 27's pattern — frozen dataclass, `Solver` protocol, thin eager
wrapper. Its `SolverResult` now populates the AL duals into `MPCState.al`.

## Julia parity

Reference §8.2 row 15: the full `(cost, c_max, μ_max)` history over the outer iterations, not just
the final values. A converged endpoint can hide a divergent path.

## Acceptance criteria

- [ ] A cartpole with `ControlBound` and a goal constraint solves to `max_violation < constraint_tolerance`.
- [ ] Nothing in `ilqr.py` changed; the outer loop composes with it purely through the cost expansion and cost function.
- [ ] Effective tolerances are threaded through the carry; `SolverOptions` is never mutated.
- [ ] Outer-loop convergence reproduces the last-match-wins overwrite, with a test that converges and exhausts `iterations_outer` on the same iteration and asserts `MAX_ITERATIONS_OUTER`.
- [ ] The inner-status break uses an ordinal comparison against `SOLVE_SUCCEEDED`.
- [ ] `kickout_max_penalty` works, is tested against a hand-written expectation, and its docstring records that Altro's own branch throws.
- [ ] λ and μ persist across outer iterations and are returned in `MPCState.al` for warm-starting a subsequent solve.
- [ ] Cross tests match `Altro.ALSolver` on the whole `(cost, c_max, μ_max)` history, plus final λ, μ, trajectory, status, and iteration count, to 1e-8.
- [ ] A warm-started second solve from the returned duals converges in strictly fewer outer iterations than a cold one.
- [ ] Targeted checks green. No full-suite run and no `pre-commit --all-files` gate
