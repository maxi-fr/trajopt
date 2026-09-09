# ADR 0008: Output functions and observation maps

## Status

Accepted.

## Context

Optimal control formulations naturally define system dynamics in state coordinates $x \in \mathbb{R}^n$ and
control inputs $u \in \mathbb{R}^m$. In many practical engineering applications, however, objectives,
path constraints, and sensor measurements are most naturally formulated on an output or observation map:

$$y = g(x, u, t) \in \mathbb{R}^p$$

Representative examples include:

- End-effector task-space pose and velocity tracking in manipulators.
- Output sensor observation models (e.g. range-bearing, camera projection, IMU specific force).
- Physical bounds on internal subsystem variables, reaction torques, or joint stresses.

Prior to this work, handling output quantities required users to manually compose operations inside
bespoke cost functions or constraint implementations, or augment the state space with artificial output
variables and integration dynamics. This caused unnecessary code duplication, increased problem
dimensions, introduced artificial equality constraints, and created an architectural impedance mismatch
with external simulation and estimation frameworks such as `simulate`.

## Decision

### 1. Output map as an optional model contract

`AbstractModel` introduces an optional static output dimension `p: int | None`.
Models with non-`None` `p` implement:

- `output(self, x, u=None, t=0.0) -> jax.Array` returning $y \in \mathbb{R}^p$.
- `output_state_jacobian` returning $\partial g / \partial x \in \mathbb{R}^{p \times n}$ via JAX automatic differentiation.
- `output_control_jacobian` returning $\partial g / \partial u \in \mathbb{R}^{p \times m}$ via JAX automatic differentiation.
- `evaluate_output(self, trajectory: Trajectory) -> jax.Array` vectorizing output evaluation across all Knot Points using `jax.vmap` to yield stacked outputs of shape $(N, p)$.

`DiscretizedDynamics` forwards `p`, `output`, and both Jacobians directly to its underlying continuous-time model.

### 2. Linearization in manifold error coordinates

In accordance with ADR 0004, derivatives linearized by stagewise solvers must reside in error coordinates.
In `_linearize_about` (`src/trajopt/models/transforms.py`), when `model.p is not None`:

- At each stage $k \in \{0, \dots, N-2\}$, the output state Jacobian is projected via the error-state map $G_k$:
  $$C_k = \frac{\partial g}{\partial x}(x_k, u_k, t_k) \, G_k \in \mathbb{R}^{p \times n_e}$$
  $$D_k = \frac{\partial g}{\partial u}(x_k, u_k, t_k) \in \mathbb{R}^{p \times m}$$
- At the terminal Knot Point $k = N-1$:
  $$C_{N-1} = \frac{\partial g}{\partial x}(x_{N-1}, \text{None}, t_{N-1}) \, G_{N-1} \in \mathbb{R}^{p \times n_e}$$

`LinearTrajectoryModel` stores the stacked Jacobians $C$ of shape $(N, p, n_e)$ and $D$ of shape $(N-1, p, m)$.
For Euclidean systems ($G_k = I$), $C_k$ coincides with the raw state Jacobian.

### 3. Affine models with native output

`AffineModel` accepts optional output matrices $C \in \mathbb{R}^{p \times n}$, $D \in \mathbb{R}^{p \times m}$,
and offset $d_y \in \mathbb{R}^p$ defining $y = C x + D u + d_y$. As an exact linear model, its output
Jacobians return $C$ and $D$ analytically without AD overhead.

### 4. Composable adapters for costs and constraints

To preserve modularity, output mappings are coupled through adapters rather than solver modifications:

- `OutputCost(CostFunction)` wraps an `AbstractModel` and an inner `CostFunction` defined on $\mathbb{R}^p$.
  Its derivatives are computed through the output composition via AD.
- `pullback_output_cost(model: AffineModel, cost: QuadraticCostFunction) -> QuadraticCost` analytically
  pulls back quadratic output costs through affine dynamics into native `QuadraticCost` on state and control,
  preserving Gauss-Newton structure without introducing slack variables.
- `OutputConstraint(Constraint)` wraps an `AbstractModel` and an inner `Constraint` defined on $\mathbb{R}^p$,
  preserving conic mapping, reporting `uses_control()`, and evaluating constraint residuals and Jacobians.

### 5. Simulation bridge

`TrajOptMeasurement` wraps an `AbstractModel` and implements `simulate.sensor.MeasurementModel`
(`Callable[[float, np.ndarray, np.ndarray], np.ndarray]`), allowing trajopt models to serve directly as
sensor observation maps in `simulate`. `TrajOptDynamics.output` provides direct querying of output signals
during simulation.

## Consequences

- **Separation of concerns**: Dynamics models specify system geometry and outputs; costs and constraints
  specify requirements on those outputs.
- **Manifold correctness**: Error-state projection $C_k = (\partial g/\partial x) G_k$ ensures consistent
  tangent-space linearizations for non-Euclidean systems (such as `RigidBody` attitude states).
- **Analytic performance**: Quadratic output costs on affine systems pull back analytically to standard
  quadratic forms without solver modification or slack variables.
- **Simulation compatibility**: Sensor models in `simulate` connect directly to trajopt models without
  ad-hoc glue code.

## Alternatives rejected

- **State augmentation**: Appending output variables $y$ to state $x$ with trivial dynamics $\dot{y} = 0$
  or algebraic constraints. This expands the Primal Vector, increases solver memory, and converts simple
  stage evaluations into stiff Defect constraints.
- **Transcription slack variables**: Adding $y_k$ as NLP decision variables at each Knot Point with equality
  constraints $y_k - g(x_k, u_k) = 0$. This inflates the Primal Vector and constraint Jacobian sparsity
  structure unnecessarily.
- **Solver-specific output logic**: Embedding output-aware logic directly into Ipopt or ALTRO Backends.
  This violates Backend orthogonality and duplicates logic across Native Solvers.
