# TODOs

* Clean up the __init__.py
* make tests faster
  * split workers (2) between cross_verification and others maybe
  * cross tests very slow

* Simulate package integration
  * wrappers for dynamics and controller

* Implement: SQP adapter, ALTRO, iLQR

* Remove unnecessary configurability. Many if else blocks

* all functions of quaternion.py should be made into methods of the class

* Fix `Objective.__getitem__`: it decides a leaf is stacked by testing `leaf.shape[0] == N - 1`,
  so an unstacked leaf whose leading dimension happens to equal `N - 1` gets sliced away.
  Reproduces on the quadrotor at `N = 5`, where `R_stage` has shape `(4,)`: `problem.obj[0].R`
  comes back a scalar and `transcription.hessian` raises
  `matmul input operand 0 must have ndim at least 1`. `CostFunction.is_stacked` already knows
  the answer the shape test is guessing at.

* add examples marimo notebooks

* make configurable

* refactor Julia looking code to be more pythonic:

  ```python
  opt_state = solve(prob, state)
  X = states(opt_state)
  U = controls(opt_state)
  ```
