# 15 — OSQP and Clarabel adapters

**What to build:** Two more solver backends behind the same transcription layer, so a caller can
switch solvers without reformulating the problem.

Clarabel carries weight beyond being a third option: it is the only backend that accepts
second-order cone constraints natively. Right now the second-order cone has unit tests and
cross-tests but no integration test anywhere — nothing in the project actually solves a problem
that uses it, so the conic machinery is carried through the whole of v1 unproven in situ. This
ticket closes that gap by putting the norm constraint's conic form through a real solve.

**Blocked by:** 09 — NLP transcription and the first Ipopt solve.

**Spec:** Section 12 (NLP transcription), section 9 (cones and projections), section 10 (the norm
constraint's two forms).

## Acceptance criteria

- [ ] An OSQP adapter accepts a transcribed problem and solves it, with its convex-quadratic
      restriction documented
- [ ] A Clarabel adapter accepts a transcribed problem and solves it
- [ ] The norm constraint's second-order-cone form reaches Clarabel as a cone constraint rather
      than being reformulated into an inequality
- [ ] At least one solve exercises the second-order cone end to end, and its solution is verified
      against the same problem expressed with the quadratic norm form through Ipopt
- [ ] Solver selection does not change how a problem is defined
- [ ] Each adapter reports convergence status, iteration count, and constraint violation through
      a common interface
