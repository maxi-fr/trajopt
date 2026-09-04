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
