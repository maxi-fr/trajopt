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
