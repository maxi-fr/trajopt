# Trajectory Optimization

Solving discrete-time optimal control problems: given a dynamics model, a cost, and constraints,
find the state and control sequence that minimizes cost while satisfying both.

This glossary is grown lazily. It currently covers the vocabulary of problem transcription and
the full-space solvers; the native ALTRO family's own vocabulary (expansions, backward pass,
augmented Lagrangian penalties) is not yet captured here.

## Language

### Problem structure

**Knot Point**:
One discrete time index in the horizon, carrying a state and — except at the terminal index — a
control.
_Avoid_: node, timestep, sample, stage (use Stage Cost only for the cost)

**Horizon**:
The number of Knot Points in the problem, written `N`.
_Avoid_: length, window, trajectory length

**Trajectory**:
A state sequence, control sequence, and their time stamps over the whole Horizon.
_Avoid_: solution, path, rollout (a Rollout is one way to produce a Trajectory, not a synonym)

**Rollout**:
A Trajectory produced by integrating the dynamics forward from an initial state, so it satisfies
the dynamics by construction.
_Avoid_: simulation, forward pass (Forward Pass is iLQR's specific line-search rollout)

**Primal Vector**:
The Trajectory flattened into one decision vector, states and controls interleaved by Knot Point.
_Avoid_: `z` alone in prose, decision variables, flat vector

**Transcription**:
The reformulation of the optimal control problem as a single finite-dimensional nonlinear program
over the Primal Vector.
_Avoid_: discretization (that is the integrator's job), encoding, formulation

**Defect**:
The residual of a dynamics constraint at one Knot Point: the gap between the next state and what
the dynamics predict from the current state and control.
_Avoid_: dynamics error, gap, shooting residual

### Solving

**Backend**:
A solver that consumes the transcribed nonlinear program as a Primal Vector and hands it to an
external library.
_Avoid_: adapter, wrapper, plugin

**Native Solver**:
A solver written against the Trajectory directly rather than the Primal Vector, exploiting the
problem's stagewise structure.
_Avoid_: internal solver, custom solver

Bare **solver** is ambiguous between the two and is best avoided in prose, even though the shared
protocol is named for it.

**Operating Point**:
The Trajectory a Backend expands the problem about when it linearizes the dynamics and
constraints and takes the cost to second order.
_Avoid_: reference trajectory, linearization point, nominal (Nominal is the fixed trajectory an
iLQR line search rolls out against)

**Multiplier**:
The dual variable priced against one constraint row at a solution.
_Avoid_: dual, lambda, Lagrange multiplier (in prose; the symbol is fine in formulas)

**Cone**:
The set a constraint's value is required to lie in, which is what gives the constraint its sense:
equality, inequality, or second-order.
_Avoid_: constraint type, sense, direction

### Globalization

**Merit Function**:
The scalar a line search decreases, combining cost with a penalty on constraint violation, so
that a step can be judged when the iterate satisfies neither alone.
_Avoid_: objective (the Objective is the cost being minimized), penalty function, cost-to-go

**Penalty Parameter**:
The weight the Merit Function places on constraint violation relative to cost.
_Avoid_: sigma, mu, rho in prose, weight

**Damped BFGS**:
A quasi-Newton approximation of the Lagrangian's second derivative, modified so it stays positive
definite even when the curvature it observes is negative.
_Avoid_: Hessian approximation (too broad — Gauss-Newton is one too), quasi-Newton alone

**Elastic Mode**:
A reformulation that adds penalized slack variables to the constraints so a subproblem is always
feasible, used when linearized constraints turn out to be mutually inconsistent.
_Avoid_: relaxation, restoration (Restoration is Ipopt's different mechanism), soft constraints
