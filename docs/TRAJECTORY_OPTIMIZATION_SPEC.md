# TrajectoryOptimization.jl architecture and feature reference

A technical reference for implementing an optimal control and MPC framework in Python.

---

## Table of contents

1. [Core design and scope](#1-core-design-and-scope)
2. [Mathematical problem formulation](#2-mathematical-problem-formulation)
3. [System architecture](#3-system-architecture)
4. [Trajectory and knot point storage](#4-trajectory-and-knot-point-storage)
5. [Dynamics and integration](#5-dynamics-and-integration)
6. [Cost functions and objectives](#6-cost-functions-and-objectives)
7. [Cones and projections](#7-cones-and-projections)
8. [Constraint catalog](#8-constraint-catalog)
9. [ConstraintList management](#9-constraintlist-management)
10. [Rotations on SO(3)](#10-rotations-on-so3)
11. [Problem container and MPC workflow](#11-problem-container-and-mpc-workflow)
12. [First- and second-order expansions](#12-first--and-second-order-expansions)
13. [NLP transcription](#13-nlp-transcription)
14. [Python-specific porting pitfalls](#14-python-specific-porting-pitfalls)
15. [Verification, Julia cross-testing, and CasADi benchmarking](#15-verification-julia-cross-testing-and-casadi-benchmarking)

---

## 1. Core design and scope

`TrajectoryOptimization.jl` solves discrete-time optimal control problems for robotics and aerospace systems. The codebase splits an optimal control problem into modular components:

1. Dynamics models: continuous, discrete, and discretized ODEs with rotation group support.
2. Cost functions and objectives: stage costs and terminal costs, with dedicated quadratic and LQR structures.
3. Constraints and cones: vector constraints mapped to convex cones (equalities, inequalities, second-order cones).
4. Trajectory container: state, control, and time storage along a discrete horizon.
5. Problem definition: links dynamics, costs, constraints, and boundary conditions.
6. Expansion engine: computes Taylor series expansions for DDP, iLQR, Augmented Lagrangian (ALTRO), and NLP solvers.

### Main design choices

**Markovian stage decoupling.** Stage cost $\ell_k(x_k, u_k)$ and stage constraint $c_k(x_k, u_k)$ depend only on the current knot point $k$. Coupling across time happens exclusively through dynamics $x_{k+1} = f(x_k, u_k)$. This structure keeps the cost Hessian and constraint Jacobian block-diagonal.

**Conic formulation for constraints.** Constraints evaluate to a vector $c(x, u)$. Feasibility is evaluated by projecting $c(x, u)$ onto a convex cone $\mathcal{K}$. This handles equality, inequality, and second-order cone constraints through a single mathematical interface.

---

## 2. Mathematical problem formulation

The discrete-time trajectory optimization problem over $N$ knot points is:

$$\begin{aligned}
\min_{x_{1:N}, u_{1:N-1}} \quad & \ell_N(x_N) + \sum_{k=1}^{N-1} \ell_k(x_k, u_k) \\
\text{s.t.} \quad & x_1 = x_0, \\
& x_{k+1} = f_d(x_k, u_k, t_k, \Delta t_k), \quad k = 1, \dots, N-1, \\
& c_k(x_k, u_k) \in \mathcal{K}_k, \quad k = 1, \dots, N-1, \\
& c_N(x_N) \in \mathcal{K}_N
\end{aligned}$$

Variables:
- $x_k \in \mathbb{R}^{n_k}$ is the state at knot point $k$.
- $u_k \in \mathbb{R}^{m_k}$ is the control input at knot point $k$. The terminal knot point $k=N$ has no control input.
- $t_k$ is the timestamp. $\Delta t_k = t_{k+1} - t_k$ is the step duration.
- $\ell_k(x_k, u_k)$ is the scalar stage cost. $\ell_N(x_N)$ is the terminal cost.
- $f_d(\cdot)$ is the discrete dynamics step.
- $c_k(\cdot)$ is a vector constraint function mapping into cone $\mathcal{K}_k$.

---

## 3. System architecture

```
TrajectoryOptimization structure
│
├── Trajectory and knot points
│   ├── KnotPoint (x, u, t, dt)
│   └── SampledTrajectory (list of KnotPoints)
│
├── Dynamics and integration
│   ├── ContinuousDynamics: dx/dt = f(x, u, t)
│   ├── DiscretizedDynamics (RK4, ImplicitMidpoint, Euler)
│   └── DiscreteDynamics: x_{k+1} = f(x_k, u_k)
│
├── Cost functions and objectives
│   ├── CostFunction (base scalar function)
│   │   ├── QuadraticCostFunction (abstract base)
│   │   │   ├── DiagonalCost (Q, R diagonal, H=0)
│   │   │   ├── QuadraticCost (dense Q, R, H, q, r, c)
│   │   │   └── DiagonalQuatCost (quaternion geodesic penalty)
│   │   ├── ErrorQuadratic (manifold error state cost)
│   │   └── Generic / autodiff CostFunction
│   └── Objective (list of N cost functions)
│
├── Conic sets and senses
│   ├── ZeroCone / Equality (g(x) = 0)
│   ├── NegativeOrthant / Inequality (h(x) <= 0)
│   ├── PositiveOrthant (h(x) >= 0)
│   └── SecondOrderCone (||v||_2 <= s)
│
├── Constraints
│   ├── AbstractConstraint
│   │   └── StageConstraint (function of x and u)
│   │       ├── StateConstraint (function of x only)
│   │       │   ├── GoalConstraint (x[inds] == xf)
│   │       │   ├── StateBound (xmin <= x <= xmax)
│   │       │   ├── CircleConstraint (2D obstacle)
│   │       │   ├── SphereConstraint (3D obstacle)
│   │       │   ├── CollisionConstraint (pairwise body distance)
│   │       │   └── QuatVecEq (quaternion attitude equality)
│   │       ├── ControlConstraint (function of u only)
│   │       │   └── ControlBound (umin <= u <= umax)
│   │       ├── LinearConstraint (A * [x; u] - b in cone)
│   │       ├── NormConstraint (||y|| <= a or SOC)
│   │       ├── BoundConstraint (combined box bounds on x and u)
│   │       └── IndexedConstraint (subvector wrapper)
│   └── DynamicsConstraint (collocation constraint)
│
├── ConstraintList (stores constraints and active knot-point index ranges)
│
└── Problem (model, objective, constraints, initial state, goal state, trajectory)
```

---

## 4. Trajectory and knot point storage

### KnotPoint
Stores the state, control, and time at index $k$.

Fields:
- `z`: vector containing $[x_k; u_k] \in \mathbb{R}^{n_k + m_k}$.
- `_x`: index range for the state slice.
- `_u`: index range for the control slice.
- `t`: timestamp $t_k$.
- `dt`: time step duration $\Delta t_k$.

Methods:
- `state(z)`: returns a view or copy of $x_k$.
- `control(z)`: returns a view or copy of $u_k$. Empty at $k = N$.
- `is_terminal(z)`: returns `true` when $k = N$.

### SampledTrajectory
A container holding $N$ `KnotPoint` elements.

Methods:
- `states(traj)`: returns the $N$ state vectors $[x_1, \dots, x_N]$.
- `controls(traj)`: returns the $N-1$ control vectors $[u_1, \dots, u_{N-1}]$.
- `gettimes(traj)`: returns timestamps $[t_1, \dots, t_N]$.
- `setstates!(traj, X)`, `setcontrols!(traj, U)`: updates state or control trajectories in place.
- `setinitialtime!(traj, t0)`: shifts timestamps starting from $t_0$.

---

## 5. Dynamics and integration

### Dynamics types

1. `ContinuousDynamics`: defines $\dot{x} = f(x, u, t)$.
   - `state_dim(model)` returns $n$.
   - `control_dim(model)` returns $m$.
   - `dynamics(model, x, u, t)` returns $\dot{x}$.
   - `jacobian!(model, J, x, u, t)` writes $[\nabla_x f, \nabla_u f] \in \mathbb{R}^{n \times (n+m)}$ into $J$.

2. `DiscreteDynamics`: defines $x_{k+1} = f(x_k, u_k, t_k, \Delta t_k)$.

3. `DiscretizedDynamics`: wraps `ContinuousDynamics` with an integrator.

### Numerical integrators

**Explicit Runge-Kutta 4 (RK4)**:
$$\begin{aligned}
k_1 &= f(x_k, u_k, t_k) \\
k_2 &= f(x_k + \tfrac{\Delta t}{2} k_1, u_k, t_k + \tfrac{\Delta t}{2}) \\
k_3 &= f(x_k + \tfrac{\Delta t}{2} k_2, u_k, t_k + \tfrac{\Delta t}{2}) \\
k_4 &= f(x_k + \Delta t k_3, u_k, t_k + \Delta t) \\
x_{k+1} &= x_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}$$

**Euler integration**:
$$x_{k+1} = x_k + \Delta t \cdot f(x_k, u_k, t_k)$$

**Implicit midpoint**:
$$x_{k+1} - x_k - \Delta t \cdot f\left(\frac{x_k + x_{k+1}}{2}, u_k, t_k + \frac{\Delta t}{2}\right) = 0$$

### Forward simulation (rollout)
`rollout!(prob)` sets $x_1 = x_0$ and propagates $x_{k+1} = f_d(x_k, u_k)$ for $k = 1, \dots, N-1$.

---

## 6. Cost functions and objectives

### Mathematical form
Stage costs and terminal costs use a general quadratic form:
$$\ell(x, u) = \frac{1}{2} x^T Q x + \frac{1}{2} u^T R u + u^T H x + q^T x + r^T u + c$$

### Cost function types

| Class | Purpose | Structure |
| :--- | :--- | :--- |
| `DiagonalCost` | Diagonal state and control weights | $Q = \text{diag}(Q_d)$, $R = \text{diag}(R_d)$, $H = 0$. Inverting cost is $O(n+m)$. |
| `QuadraticCost` | General quadratic cost | Dense matrices $Q \in \mathbb{R}^{n \times n}$, $R \in \mathbb{R}^{m \times m}$, $H \in \mathbb{R}^{m \times n}$. |
| `LQRCost` | Tracking cost builder | Generates $(x - x_f)^T Q (x - x_f) + (u - u_f)^T R (u - u_f)$ with $q = -Q x_f$, $r = -R u_f$, $c = \frac{1}{2} x_f^T Q x_f + \frac{1}{2} u_f^T R u_f$. |
| `DiagonalQuatCost` | Attitude tracking on $SO(3)$ | Adds quaternion geodesic penalty $w \min(1 \pm q_{\text{ref}}^T q)$. |
| `ErrorQuadratic` | Manifold error tracking | Computes $\frac{1}{2}(x \ominus x_d)^T Q (x \ominus x_d)$ using rotation group error states. |
| `GenericCost` | Non-quadratic user cost | Differentiated with autodiff or finite differences. |

### Cost methods
- `evaluate(cost, x, u)`: returns scalar cost $\ell(x, u)$.
- `gradient!(cost, grad, z)`: writes $[\nabla_x \ell; \nabla_u \ell]$ into `grad`.
- `hessian!(cost, hess, z)`: writes $\nabla^2 \ell$ into `hess`.
- `invert!(Ginv, cost)`: inverts the cost Hessian analytically using diagonal or block-diagonal shortcuts.
- `c1 + c2`: adds two quadratic costs and promotes the result to `QuadraticCost`.
- `set_LQR_goal!(cost, xf, [uf])`: updates $q \leftarrow -Q x_f$ and $r \leftarrow -R u_f$ for MPC target updates.

### Objective structure
Holds $N$ cost functions where indices $1 \dots N-1$ are stage costs and index $N$ is the terminal cost.

- `cost(obj, traj)`: evaluates $\sum_{k=1}^N \ell_k(z_k)$.
- `LQRObjective(Q, R, Qf, xf, N)`: creates homogeneous stage LQR costs and a terminal cost.
- `TrackingObjective(Q, R, Z_ref)`: builds time-varying LQR costs tracking trajectory $Z_{\text{ref}}$.
- `update_trajectory!(obj, Z_ref, start=1)`: updates reference states along the tracking objective.

---

## 7. Cones and projections

Every constraint is expressed as $c(x, u) \in \mathcal{K}$.

### Cone definitions

| Cone | Set definition | Projection $\Pi_{\mathcal{K}}(x)$ |
| :--- | :--- | :--- |
| `ZeroCone` / `Equality` | $x = 0$ | $0$ |
| `NegativeOrthant` / `Inequality` | $x \le 0$ | $\min(0, x)$ |
| `PositiveOrthant` | $x \ge 0$ | $\max(0, x)$ |
| `SecondOrderCone` | $\|v\|_2 \le s$ where $x = [v; s]$ | Piecewise projection below |

### Second-order cone projection and derivatives
Let $x = [v; s]$ with $v \in \mathbb{R}^{p-1}$, $s \in \mathbb{R}$, and $a = \|v\|_2$.

1. Projection $\Pi_{\mathcal{K}}(x)$:
   $$\Pi_{\mathcal{K}}(x) = \begin{cases}
   0 & a \le -s \quad (\text{below dual cone}) \\
   x & a \le s \quad (\text{inside cone}) \\
   \frac{1}{2}\left(1 + \frac{s}{a}\right) \begin{bmatrix} v \\ a \end{bmatrix} & a > |s| \quad (\text{outside cone})
   \end{cases}$$

2. First derivative $\nabla \Pi_{\mathcal{K}}(x)$:
   $$\nabla \Pi_{\mathcal{K}}(x) = \begin{cases}
   0 & a \le -s \\
   I & a \le s \\
   \frac{1}{2}\begin{bmatrix} \left(1+\frac{s}{a}\right)I - \frac{s}{a^3} v v^T & \frac{v}{a} \\ \left(1+\frac{s}{a}\right)\frac{v^T}{a} - \frac{s}{a^2} v^T & 1 \end{bmatrix} & a > |s|
   \end{cases}$$

3. Second derivative contraction $\nabla^2 \Pi_{\mathcal{K}}(x)[b]$:
   Used in second-order expansions of the Augmented Lagrangian term $\lambda^T \Pi(c)$.

---

## 8. Constraint catalog

### Base classes
- `StageConstraint`: depends on both state $x_k$ and control $u_k$.
- `StateConstraint`: depends only on state $x_k$. Control Jacobian is zero.
- `ControlConstraint`: depends only on control $u_k$. State Jacobian is zero.

### Defined constraints

**`GoalConstraint` (State equality).**
- Formula: $x_k[\text{inds}] - x_f = 0$.
- Sense: `Equality`.
- Jacobian: $\nabla_x c = I_{\text{inds}}$, $\nabla_u c = 0$.

**`BoundConstraint` / `StateBound` / `ControlBound` (Box bounds).**
- Formula: $x_k - x_{\max} \le 0$, $x_{\min} - x_k \le 0$, $u_k - u_{\max} \le 0$, $u_{\min} - u_k \le 0$.
- Sense: `Inequality`.
- Bound conversion: maps directly to solver primal variable bounds $[z_L, z_U]$.

**`LinearConstraint`.**
- Formula: $A [x_k; u_k] - b \le 0$ (or $= 0$).
- Sense: `Inequality` or `Equality`.
- Jacobian: constant matrix $A$.

**`CircleConstraint` (2D obstacle).**
- Formula: $-(x - x_c)^2 - (y - y_c)^2 + r^2 \le 0$.
- Sense: `Inequality`.
- Jacobian: $\nabla_x c = [-2(x - x_c), -2(y - y_c), 0, \dots]$.

**`SphereConstraint` (3D obstacle).**
- Formula: $-(x - x_c)^2 - (y - y_c)^2 - (z - z_c)^2 + r^2 \le 0$.
- Sense: `Inequality`.
- Jacobian: $\nabla_x c = [-2(x - x_c), -2(y - y_c), -2(z - z_c), 0, \dots]$.

**`CollisionConstraint` (Pairwise body distance).**
- Formula: $r^2 - \|x[\text{body}_1] - x[\text{body}_2]\|^2 \le 0$.
- Sense: `Inequality`.
- Jacobian: $\nabla_{x_1} c = -2(x_1 - x_2)$, $\nabla_{x_2} c = +2(x_1 - x_2)$.

**`NormConstraint`.**
- Quadratic inequality form: $\|y\|_2^2 - a^2 \le 0$.
- Second-order cone form: $[y; a] \in \mathcal{K}_{SOC}$.
- Jacobian for SOC: $\nabla_y c = [I_p; 0]$.

**`IndexedConstraint`.**
- Wraps a constraint defined on a subsystem $(x_{\text{sub}}, u_{\text{sub}})$ to apply to an indexed slice of the full state and control vectors.

**`DynamicsConstraint` (Collocation constraint).**
- Explicit: $x_{k+1} - f_d(x_k, u_k) = 0$. Jacobians are $\nabla_{x_k} c = -\nabla_x f_d$, $\nabla_{u_k} c = -\nabla_u f_d$, $\nabla_{x_{k+1}} c = I$.
- Implicit: $x_k - x_{k+1} + \Delta t \cdot f(\frac{x_k + x_{k+1}}{2}, u_k) = 0$.

---

## 9. ConstraintList management

`ConstraintList` stores constraints and tracks the knot-point ranges where they apply.

### Internal fields
- `nx`: list of state dimensions across knot points.
- `nu`: list of control dimensions across knot points.
- `constraints`: list of constraint objects.
- `inds`: list of knot-point index ranges where each constraint is active.
- `sigs`: evaluation mode (in place or return new).
- `diffs`: differentiation method (autodiff, finite difference, analytic).
- `p`: array of length $N$ storing total constraint dimension at each knot point.

### Key operations
- `add_constraint!(cons, con, inds)`: checks dimensions and registers the constraint over `inds`.
- `num_constraints(cons)`: returns the vector $p = [p_1, \dots, p_N]$.
- `primal_bounds!(zL, zU, cons)`: extracts simple box bounds into global variable limits $z_L \le z \le z_U$.
- Batch evaluation: loops over `inds` to evaluate values and Jacobians along the trajectory.

---

## 10. Rotations on SO(3)

Unit quaternions $q \in \mathbb{H}$ with $\|q\|=1$ live on the 3-sphere $S^3$. Standard vector addition and subtraction do not preserve unit length.

### Error state on manifolds
Instead of Euclidean error $x - x_d$, Lie group models compute the difference:
$$\delta x = x \ominus x_d = \begin{bmatrix} p - p_d \\ \phi(q \cdot q_d^{-1}) \\ v - v_d \\ \omega - \omega_d \end{bmatrix} \in \mathbb{R}^{12}$$
where $\phi(\cdot)$ maps a rotation error quaternion to $\mathbb{R}^3$ via the Cayley map or Modified Rodrigues Parameters (MRP).

### Attitude Jacobian ($G_k$)
The attitude Jacobian maps Euclidean state variations to the 12-dimensional error state:
$$G_k = \frac{\partial (x \ominus x_d)}{\partial x} \in \mathbb{R}^{13 \times 12}$$

### Error expansions in solvers
When a solver operates in error state coordinates:
- Dynamics: $\bar{A}_k = G_{k+1}^T A_k G_k$, $\bar{B}_k = G_{k+1}^T B_k$.
- Cost: $\bar{q}_k = G_k^T \nabla_x \ell_k$, $\bar{Q}_k = G_k^T (\nabla_{xx}^2 \ell_k) G_k$.
- Constraints: $\bar{\nabla}_x c_k = (\nabla_x c_k) G_k$.

### Quaternion cost functions
`DiagonalQuatCost` uses the geodesic penalty:
$$\ell(x) = \frac{1}{2} x^T Q x + w \min(1 + q_{\text{ref}}^T q, 1 - q_{\text{ref}}^T q)$$
This handles the double-cover property where $q$ and $-q$ represent the same physical rotation. The gradient is:
$$\nabla_q \ell = \begin{cases} +w q_{\text{ref}} & q_{\text{ref}}^T q < 0 \\ -w q_{\text{ref}} & q_{\text{ref}}^T q \ge 0 \end{cases}$$

---

## 11. Problem container and MPC workflow

### `Problem` fields
- `model`: list of $N-1$ dynamics steps (supports time-varying or hybrid models).
- `obj`: `Objective` containing $N$ cost functions.
- `constraints`: `ConstraintList`.
- `x0`: initial state vector.
- `xf`: goal state vector.
- `Z`: `SampledTrajectory` storing current states, controls, and timestamps.
- `N`: horizon knot point count.
- `t0`: start time.
- `tf`: final time.

### Problem methods
- `cost(prob)`: evaluates $\sum_{k=1}^N \ell_k(z_k)$.
- `rollout!(prob)`: simulates dynamics forward from $x_0$ using controls in $Z$.
- `states(prob)` / `controls(prob)`: extracts trajectory arrays.
- `initial_states!(prob, X0)`, `initial_controls!(prob, U0)`: sets warm-start trajectories.
- `set_initial_state!(prob, x0)`: updates the initial condition for the next MPC step.
- `set_goal_state!(prob, xf)`: updates target state in problem, objective, and goal constraints.
- `setinitialtime!(prob, t0)`: shifts trajectory timestamps.

### MPC control loop
```
1. Initialize Problem(model, obj, x0, tf, constraints)
2. In each control loop iteration:
   a. prob.set_initial_state!(x_measured)
   b. prob.setinitialtime!(t_current)
   c. Optional: update goal state or reference trajectory
   d. Solve problem using ALTRO, iLQR, or NLP solver
   e. Apply first control input: u_command = controls(prob)[0]
   f. Warm-start next solve by shifting trajectory Z forward by dt
```

---

## 12. First- and second-order expansions

Solvers using Differential Dynamic Programming (DDP), iLQR, or ALTRO require stage-by-stage Taylor expansions:

### Dynamics expansion
At each stage $k = 1, \dots, N-1$:
$$A_k = \nabla_x f(x_k, u_k) \in \mathbb{R}^{n_{k+1} \times n_k}, \quad B_k = \nabla_u f(x_k, u_k) \in \mathbb{R}^{n_{k+1} \times m_k}$$

### Cost expansion
At each stage $k = 1, \dots, N$:
$$q_k = \nabla_x \ell_k, \quad r_k = \nabla_u \ell_k, \quad Q_k = \nabla_{xx}^2 \ell_k, \quad R_k = \nabla_{uu}^2 \ell_k, \quad H_k = \nabla_{ux}^2 \ell_k$$

### Augmented Lagrangian expansion
For constraint $c(x, u) \in \mathcal{K}$ with multiplier $\lambda$ and penalty weight $\mu > 0$:
$$\mathcal{L}_A(x, u, \lambda, \mu) = \ell(x, u) + \lambda^T \Pi_{\mathcal{K}^*}\left(c(x, u) + \frac{\lambda}{\mu}\right) + \frac{\mu}{2} \left\| \Pi_{\mathcal{K}^*}\left(c(x, u) + \frac{\lambda}{\mu}\right) \right\|^2$$
The gradient and Hessian of $\mathcal{L}_A$ add directly into $q_k, r_k, Q_k, R_k, H_k$.

---

## 13. NLP transcription

For general nonlinear solvers like Ipopt or OSQP:

### Primal variable vector ($Z$)
$$Z = [x_1^T, u_1^T, x_2^T, u_2^T, \dots, x_{N-1}^T, u_{N-1}^T, x_N^T]^T \in \mathbb{R}^{N n + (N-1) m}$$

### Constraint vector ($c(Z)$)
$$c(Z) = \begin{bmatrix}
x_1 - x_0 \\
x_2 - f_d(x_1, u_1) \\
c_1(x_1, u_1) \\
\vdots \\
x_N - f_d(x_{N-1}, u_{N-1}) \\
c_N(x_N)
\end{bmatrix} \in \mathbb{R}^P$$

### Sparsity structures
- **Jacobian $\nabla c(Z)$**: block bidiagonal from dynamics and block diagonal from stage constraints.
- **Hessian of Lagrangian $\nabla^2 \mathcal{L}(Z)$**: block diagonal, with non-zero blocks corresponding to stage cost and constraint Hessians.

---

## 14. Python-specific porting pitfalls

Porting this architecture from Julia into Python introduces specific language and runtime pitfalls:

### 1. Loop latency and dynamic dispatch overhead
Julia compiles specialized static arrays and unrolls loops into zero-allocation native code. In Python, looping over $N$ knot points in an outer Augmented Lagrangian loop creates high interpreter overhead and per-call dispatch latency. For high-frequency MPC ($>50\text{ Hz}$), a pure Python loop will miss control deadlines.

*Remedy:* Keep the class interface modular, but compile inner rollout and Riccati passes using JAX (`jax.lax.scan`), Numba, or C++/Cython extensions.

### 2. NumPy array view aliasing and silent mutation
NumPy slicing returns views rather than copies. If a knot point's state is assigned via `kp.x = buffer[idx:idx+n]` without copying, mutating `kp.x` in place during a rollout will corrupt other arrays sharing the same buffer.

*Remedy:* Use `.copy()` during trajectory instantiation and explicit slice assignments (`kp.x[:] = ...`) when writing into pre-allocated memory.

### 3. Autodiff tracing issues with piecewise cone projections
Libraries like JAX and CasADi trace code as static computation graphs. Second-order cone projections use piecewise conditional branches (`if a <= -s`, `elif a <= s`, `else`). Standard Python `if/else` statements fail during JAX tracing or create non-differentiable graph splits.

*Remedy:* Implement cone projections using branchless operations (`jax.lax.cond` or `jnp.where`) and guard divisions with numerical tolerances ($\max(a, 10^{-10})$).

### 4. Polymorphic dispatch vs. JIT compilation
Julia uses multiple dispatch to evaluate a list of heterogeneous cost objects (`Vector{CostFunction}`) without performance loss. In Python, iterating through a list of different Python class instances inside a JIT-compiled loop will break JAX or Numba compilation.

*Remedy:* Normalize common cost types (such as quadratic costs) into contiguous parameter arrays ($Q_{\text{all}}, R_{\text{all}}, q_{\text{all}}, r_{\text{all}}$) so that evaluations can run via vectorized matrix operations.

### 5. Sparse NLP transcription memory blow-up
Generic Python optimizers (like `scipy.optimize.minimize`) default to dense matrix representations if not configured carefully. Treating an $N=100$, $n=12$ collocation problem as dense creates an $O(N^3)$ memory and factorization bottleneck.

*Remedy:* Always construct sparse Jacobian and Hessian patterns using `scipy.sparse.csc_matrix` or pass symbolic sparsity maps to CasADi.

---

## 15. Verification, Julia cross-testing, and CasADi benchmarking

To guarantee 100% mathematical fidelity and eliminate transcription bugs during the port, Python components are verified directly against `TrajectoryOptimization.jl` using **live in-process differential testing (`juliacall`)**, and validated end-to-end against a standalone **CasADi-only implementation**.

### Testing architecture with `juliacall`

Python's `pytest` suite invokes the local Julia package in-process without file I/O:

```python
# tests/cross_verification/test_cones.py
import pytest
import numpy as np
from juliacall import Main as jl

# Load local Julia package from trajopt_jl/
jl.seval('using Pkg; Pkg.activate("trajopt_jl")')
jl.seval("using TrajectoryOptimization; const TO = TrajectoryOptimization")

from trajopt.cones import SecondOrderCone

def test_soc_projection():
    py_cone = SecondOrderCone()
    jl_cone = jl.TO.SecondOrderCone()

    x = np.array([2.0, 3.0, 1.0, 1.0])

    # 1. Verify projection Π(x)
    px_py = py_cone.project(x)
    px_jl = np.zeros_like(x)
    jl.TO.projection_b(jl_cone, px_jl, x)
    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)

    # 2. Verify Jacobian ∇Π(x)
    J_py = py_cone.jacobian(x)
    J_jl = np.zeros((4, 4))
    jl.TO.grad_projection_b(jl_cone, J_jl, x)
    np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)
```

---

### What MUST be cross-tested (Strict numerical parity)

Cross-testing is mandatory for mathematical operations where indexing conventions, signs, coordinate frames, or differentiation paths can cause silent divergence:

| Module / Component | Specific Functions & Targets to Test | Tolerance | Verification Metric |
| :--- | :--- | :--- | :--- |
| **Cones & Projections** | `projection!`, `∇projection!`, `∇²projection!` for `ZeroCone`, `NegativeOrthant`, `PositiveOrthant`, `SecondOrderCone` | `1e-14` (values)<br>`1e-12` (derivatives) | Pointwise projection $\Pi(x)$, Jacobian $\nabla \Pi(x)$, and Hessian contraction $\nabla^2 \Pi(x)[b]$ across all cone regions (inside, outside, below dual cone). |
| **Dynamics & Integrators** | Continuous ODE dynamics, `discrete_dynamics`, and `jacobian!` ($A_k = \nabla_x f, B_k = \nabla_u f$) for RK4, Euler, and Implicit Midpoint. | `1e-14` (steps)<br>`1e-12` (Jacobians) | Output state $x_{k+1}$ and Jacobians $[A_k, B_k]$ evaluated on benchmark models (`Cartpole`, `Pendulum`, `Quadrotor`, `DubinsCar`). |
| **Rotations on $SO(3)$** | Attitude Jacobian $G_k$, error state difference $x \ominus x_d$, modified dynamics $\bar{A}_k = G_{k+1}^T A_k G_k$, geodesic quaternion cost `DiagonalQuatCost`. | `1e-12` | $G_k \in \mathbb{R}^{13 \times 12}$ projection, error vector norm, quaternion subgradient double-cover branches ($\pm w q_{\text{ref}}$). |
| **Cost Functions & Objectives** | `evaluate`, `gradient!`, `hessian!`, and `invert!` for `DiagonalCost`, `QuadraticCost`, `LQRCost`, `TrackingObjective`. | `1e-14` (values)<br>`1e-12` (grads/hess) | Scalar cost $\ell(x,u)$, gradient vector $[q; r]$, Hessian matrix $\begin{bmatrix} Q & H^T \\ H & R \end{bmatrix}$, and analytic inverted Hessian. |
| **Constraint Catalog** | Value $c(x, u)$ and Jacobians $[\nabla_x c, \nabla_u c]$ for `GoalConstraint`, `StateBound`, `ControlBound`, `CircleConstraint`, `SphereConstraint`, `CollisionConstraint`, `LinearConstraint`, `DynamicsConstraint`. | `1e-12` | Constraint evaluation vector and Jacobian matrices across active knot points. |
| **NLP Transcription** | Primal vector layout $Z$, constraint vector $c(Z)$, sparse Jacobian $\nabla c(Z)$ non-zero pattern & values, Lagrangian Hessian $\nabla^2 \mathcal{L}(Z)$ pattern & values. | `1e-12` (values)<br>Exact match (sparsity indices) | Column/row indices and numeric values fed into Ipopt evaluator (`eval_f`, `eval_grad_f`, `eval_g`, `eval_jac_g`, `eval_h`). |
| **End-to-End Problem Solves** | Full trajectory solution $(X, U)$, final objective value, max constraint violation, and solver convergence criteria. | `1e-5` to `1e-6` | Trajectory arrays $[x_1 \dots x_N]$, $[u_1 \dots u_{N-1}]$ and dual multipliers after ALTRO and Ipopt solves on identical problem setups. |

---

### What DOES NOT make sense to cross-test

Testing the following against Julia provides no mathematical verification value and introduces unnecessary coupling:

1. **Python Internal Storage & Memory Layout**:
   - `KnotPoint` property accessors, trajectory slice views, and Python container mechanics.
   - NumPy vs Julia column-major/row-major internal buffer layouts (as long as vectorized mathematical interfaces expose consistent 1D/2D arrays).

2. **Memory Allocation Metrics**:
   - Julia's `@allocated` tests (which verify zero GC allocations in Julia's static array system).
   - In Python/JAX, memory efficiency is evaluated via XLA memory profilers and allocation-free JIT execution rather than Julia heap allocation counters.

3. **Polymorphic Type Hierarchy & Multiple Dispatch**:
   - Julia's abstract type trees (`AbstractConstraint{S,D}`) and multiple dispatch method signatures.
   - Python uses explicit class hierarchies, dataclasses, or functional PyTrees.

4. **Line-Search Backtracking Step Sequences**:
   - Exact per-step line-search $\alpha$ values during intermediate iterations of ALTRO/iLQR.
   - Minor floating-point associativity differences (e.g. FMA instructions in XLA vs LLVM) can cause slight differences in step sizes during intermediate backtracking iterations, even though both solvers converge to the exact same optimal trajectory.

5. **Purely Syntactic Sugar & Builders**:
   - Method chaining utilities, plotting recipes, docstring generation, and high-level CLI wrappers.

---

### Final verification: Full MPC/OCP validation vs. pure CasADi baseline

The final phase of verification validates full trajectory optimization (OCP) and receding-horizon model predictive control (MPC) runs against an independent, pure CasADi-only implementation (`casadi.Opti` direct transcription).

#### 1. Numerical equivalence comparison
Each benchmark problem is formulated identically in both the framework and pure CasADi (matching discretization, cost matrices, initial/terminal conditions, and constraint sets):

- **Primal trajectory parity**: Maximum absolute error across all states and controls:
  $$\|X_{\text{framework}}^* - X_{\text{CasADi}}^*\|_\infty \le 10^{-5}, \quad \|U_{\text{framework}}^* - U_{\text{CasADi}}^*\|_\infty \le 10^{-5}$$
- **Objective function value**: Relative objective agreement:
  $$\frac{|J_{\text{framework}}^* - J_{\text{CasADi}}^*|}{J_{\text{CasADi}}^*} \le 10^{-5}$$
- **Constraint satisfaction & dual multipliers**: Maximum constraint residual $\|c(Z^*)\|_\infty \le \epsilon_{\text{feas}}$ and dual multiplier convergence parity under identical solver settings (e.g., Ipopt tolerances).

#### 2. Solve time and latency benchmarking
Benchmark execution speed, per-iteration timing, and memory overhead across real-world examples:

- **Timing breakdown**:
  1. *Transcription latency*: Time to assemble sparse problem matrices, Jacobian/Hessian structures, and bound vectors.
  2. *Derivative evaluation*: Per-iteration time spent evaluating cost gradients, Jacobians, and Hessians.
  3. *Solver runtime*: Pure solver time (ALTRO vs. Ipopt / CasADi Ipopt).
  4. *Closed-loop MPC rate*: Sustained control loop frequency ($\text{Hz}$), latency jitter, and warm-start speedups over a receding horizon.

- **Benchmark test problems**:
  - **Cartpole Swing-Up**: Underactuated nonlinear system with bounded actuation and state limits.
  - **Quadrotor Obstacle Avoidance**: 3D attitude tracking on $SO(3)$ (quaternion kinematics) navigating around spherical/cylindrical keep-out zones.
  - **Dubins Car / Wheeled Mobile Robot**: Nonholonomic navigation with corridor constraints and tracking objectives.
