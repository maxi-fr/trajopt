# 10 — Pure-CasADi baseline and end-to-end parity

**What to build:** An independent second opinion on whether the solver produces correct
trajectories. The same cartpole and Dubins problems are formulated from scratch in CasADi's
direct transcription interface, solved, and compared against the framework's answers.

This is the only end-to-end oracle the project has. Julia cannot fill the role — the vendored
package contains no solver at all, and its NLP transcription type is exported but never defined.
CasADi is also the better oracle on the merits: it is a genuinely independent implementation
rather than a sibling of the code under test, so agreement means something.

Keeping this early, before the SO(3) strand lands, is deliberate. Once parity is proven on
Euclidean problems, a later quadrotor mismatch is unambiguously a rotations bug rather than an
open question about the whole transcription layer.

**Blocked by:** 09 — NLP transcription and the first Ipopt solve.

**Spec:** Section 15 (verification strategy, end-to-end validation), Appendix A (gaps in the
Julia reference, explaining why Julia is not the oracle here).

## Acceptance criteria

- [x] Cartpole and Dubins problems are formulated independently in CasADi, matching
      discretization, cost weights, boundary conditions, constraint sets, and solver options
- [x] The two formulations are asserted to agree on their setup — same horizon, same step
      duration, same bounds — so a parity pass cannot come from comparing different problems
- [x] Maximum absolute state and control error against the CasADi solution is within `1e-5`
- [x] Relative objective value agreement is within `1e-5`
- [x] Maximum constraint residual is within the feasibility tolerance
- [x] Dual multipliers agree under identical solver settings
- [x] The baseline lives as reusable test infrastructure rather than a one-off script, since the
      benchmark suite extends it
