# 11 — Problem/MPCState split and the zero-recompile loop

**What to build:** A closed-loop controller. A caller builds a problem once, then repeatedly
feeds it a measured state and a time, solves, applies the first control, and shifts the
trajectory forward — at a sustained rate, without the compiler ever running again after the
first solve.

The structural decision this ticket implements: problem structure and per-step data are separate
types. The problem holds the model, horizon, cost layout, and constraint structure and never
changes during a run. The per-step state holds the initial state, initial time, goal,
multipliers, and warm-start trajectory, and is always passed as a traced argument.

This split exists for exactly one reason. The initial state, time, and goal change on every
iteration; if any of them becomes a compile-time constant, every control step triggers a
recompile and every deadline is missed. Putting them in a separate type makes that structurally
impossible rather than a discipline someone has to remember.

**Blocked by:** 09 — NLP transcription and the first Ipopt solve.

**Spec:** Section 13 (Problem, MPCState, and the MPC loop), section 4 (the zero-recompile
invariant and the traced/static split).

## Acceptance criteria

- [x] Problem structure and per-step data are distinct types, with nothing that changes per step
      living in the problem
- [x] Model parameters are traced values, so a mass or an obstacle radius can change between
      solves without recompiling
- [x] Dimensions, horizon length, integrator choice, and constraint structure are compile-time
      metadata
- [x] Per-step operations return new values: updating the measurement, updating the goal, and
      shifting the trajectory forward for warm-starting
- [x] The goal state lives in exactly one place and is read as an argument by both the objective
      and the goal constraint, with nothing to keep in sync
- [x] A closed-loop cartpole run drives the system to its goal from a perturbed initial state
- [x] A test asserts the compilation counter stays at zero across 100 consecutive iterations that
      vary the measured state, the time, and the goal
- [x] Warm-starting from the shifted previous solution measurably reduces solver iterations
      compared with a cold start
