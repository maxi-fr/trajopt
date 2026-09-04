# TODOs

* Clean up the __init__.py
* make tests faster
  * new ALTRO tests

* Implement: SQP adapter

* Remove unnecessary configurability. Many if else blocks

* make configurable

* MPCState: .X, .U, .t instead of states, controls, etc.

* solver and problem have solve methods?

* when should a method be implemented by a base class and when should it be a function?

* I want a configurable benchmark: bring your own predictor and decide which solver works the best for your case

* OSQP deprecation warning

* ALTRO profiling and optimization
  * Note: Wire BoxQP (`solve_kd_builder`, `u_bounds`) into `ALTRO.solve` as an optional performance mode for control-bounded problems (potential optimization; diverges from `Altro.jl` which handles all bounds via AL penalties)
  * Note: JAX-native block-tridiagonal Riccati / Thomas KKT solver via `jax.lax.scan` for Projected Newton (long-term native alternative to host `qdldl` callback; eliminates host-device roundtrip overhead, enables GPU/TPU execution with static shapes and $O(N)$ stage complexity)

* SingleShooting: support StateBound as constraint rows (xL <= X(u) <= xU) instead of rejecting it, since box bounds are hoisted into primal bounds before single shooting ever sees them

* Bug: `altro_solve` can report `SOLVE_SUCCEEDED` against a stale violation
  * Under `(reset_duals=False, reset_penalties=False)` one receding-horizon step reports success with a max violation of 0.054. Looks like the status upgrade reading a `c_max` cached from before the last dual update.
  * Predates the ALTRO performance work — reproduces with the per-row penalty cap disabled. Found while merging, not caused by it.
  * The config it shows up under is unsupported (ADR 0006), which is why it went unnoticed; the stale-`c_max` path itself is not config-specific and may be reachable elsewhere.

* The MPC closed-loop golden is pinned to an external factorization
  * `test/golden/mpc_cartpole_closed_loop.npz` now records QDLDL's output (ADR 0007), so a `qdldl` or SciPy bump can move it again at ~1e-8 per step, compounded by the horizon loop.
  * Decide whether the golden should keep asserting bit-level equality or drop to a tolerance that brackets the factorization's noise. Re-recording it on every dependency bump is not a real baseline.

* The penalty cap does not cover the warm-start seam
  * `penalty_update` caps `mu` at the update, so a `mu` arriving from a warm start is read by that step's first inner solve uncapped.
  * No run has been found that hits `MAXIMUM_COST` through this gap, so it is documented rather than guarded. Add the guard if one turns up.

* ADR 0006 option 2 (cap carried penalty growth between solves) is still open
  * The ADR 0007 cap looked like a candidate and is not one: gated on the active set it cannot pull a carried `mu` down on the kicked cartpole. Re-measured bit-identical with the cap off.
  * `reset_duals=False` stays unusable until this is answered, so `(True, True)` remains the only supported receding-horizon config.

* Stale entry above: "MPCState: .X, .U, .t instead of states, controls" — `MPCState` is gone, split into `BoundaryConditions` + `WarmStart` + the `MPC` driver. Re-aim it at `MPC`'s accessors or drop it.

* Cartpole swing-up benchmark does not converge at its default configuration
  * `cartpole_swingup_benchmark()` solved one-shot exits `MAX_ITERATIONS_OUTER` after 9 outer
    iterations with a max violation of 0.908 — the cart position bound is wide open, not merely
    loose.
  * Not a merge regression and not cap-related: identical with the per-row penalty cap gated and
    ungated (ADR 0007). Found while profiling the cap.
  * Decide whether the defaults (`N=25`, `x_pos_bound=0.4`, `iterations_outer`) are simply too
    tight a problem for the iteration budget, or whether the AL phase is stalling. A benchmark
    whose headline problem does not converge measures the wrong thing.

* `MAXIMUM_COST` still fires on the quadrotor benchmark
  * `quadrotor_obstacle_benchmark()` solved one-shot exits `MAXIMUM_COST` after 8 outer iterations
    with a max violation of 1.370 and `penalty_max` at 1e7.
  * This is the exact abort the per-row penalty cap was added to prevent (ADR 0007), so on this
    problem the cap does not do its job — reproduces identically with the cap gated and ungated.
  * Worth finding out whether the cost blows up through a row the cap covers and the ladder simply
    outruns it, or through a path the cap never touches (the warm-start seam above, or the base
    objective rather than the penalty term).
