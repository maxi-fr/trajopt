# TODOs

* Clean up the __init__.py
* make tests faster
  * split workers (2) between cross_verification and others maybe
  * cross tests very slow

* Simulate package integration
  * wrappers for dynamics and controller

* Implement: SQP adapter, ALTRO, iLQR

* Remove unnecessary configurability. Many if else blocks

* add examples marimo notebooks

* make configurable

* refactor Julia looking code to be more pythonic:

  ```python
  opt_state = solve(prob, state)
  X = states(opt_state)
  U = controls(opt_state)
  ```
