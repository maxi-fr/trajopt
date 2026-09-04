# Trajectory optimization in Python: implementation specification

Specification for `trajopt`, an optimal control and MPC framework in Python built on JAX.

`TrajectoryOptimization.jl` (vendored at `trajopt_jl/`) is the mathematical reference and the
correctness oracle during the port. It is not a runtime dependency, and its type hierarchy,
naming, and storage layout are deliberately **not** mirrored. Where this specification
diverges from the Julia package, the divergence is listed in
[Appendix A](#appendix-a-divergences-from-trajectoryoptimizationjl).

---

## Table of contents

1. [Scope and design decisions](#1-scope-and-design-decisions)
2. [Mathematical problem formulation](#2-mathematical-problem-formulation)
3. [Package structure](#3-package-structure)
4. [Backend, data model, and compilation](#4-backend-data-model-and-compilation)
5. [Trajectory storage](#5-trajectory-storage)
6. [Dynamics and integration](#6-dynamics-and-integration)
7. [Rotations: JPL quaternions and the error state](#7-rotations-jpl-quaternions-and-the-error-state)
8. [Cost functions and objectives](#8-cost-functions-and-objectives)
9. [Cones and projections](#9-cones-and-projections)
10. [Constraint catalog and ConstraintList](#10-constraint-catalog-and-constraintlist)
11. [Expansions](#11-expansions)
12. [NLP transcription](#12-nlp-transcription)
13. [Problem, MPC, and the control loop](#13-problem-mpc-and-the-control-loop)
14. [Models](#14-models)
15. [Verification strategy](#15-verification-strategy)
16. [Deferred work](#16-deferred-work)
17. [Appendix A: divergences from TrajectoryOptimization.jl](#appendix-a-divergences-from-trajectoryoptimizationjl)

---

## 1. Scope and design decisions

`trajopt` is a **production MPC library**, not a transliteration of the Julia package. Design
is driven by closed-loop latency and by what JAX can compile, subject to matching the Julia
reference numerically wherever a shared quantity exists.

### Delivery order

1. **v1** — modeling layer, expansion engine, NLP transcription to external solvers
   (Ipopt via `cyipopt`, OSQP, Clarabel).
2. **v2** — native iLQR and ALTRO, consuming the same expansion engine.

The expansion engine (`expansions.py`) is cut as a standalone module from day one, before its
first consumer exists. Cutting it later would let transcription's derivative code harden into
a de facto engine with an Ipopt-shaped interface.

### Settled decisions

| Decision | Choice |
| :--- | :--- |
| Array / autodiff backend | JAX in all evaluation kernels; NumPy only at the solver boundary |
| Pytree mechanism | `equinox` (`eqx.Module`, `eqx.field(static=True)`, `eqx.filter_jit`) |
| State / control dimensions | Fixed scalars `n`, `m` across the horizon |
| Error-state dimension `ne` | Optional; defaults to `n` for Euclidean models |
| Trajectory storage | Struct-of-arrays; `KnotPoint` is a view, never storage |
| Differentiation | AD everywhere, no analytic-Jacobian override |
| Compilation unit | Per-phase, matching Ipopt's four callbacks |
| Rotations | JPL convention, scalar-last |
| Error map | `δθ = 2·vec(q_err)` |
| Objective | One stage cost + one terminal cost, parameters stacked over `k` |
| Naming | Pythonic (`model.rollout(traj)`, `traj.with_states(X)`), no Julia bang-suffix mirroring |
| SO(3) support | v1, not deferred |
| Constraint catalog | Full section 10 catalog ships in v1 |

### Invariants

**Markovian stage decoupling.** Stage cost and stage constraint depend only on knot point
`k`. Coupling across time happens exclusively through the dynamics. This keeps the cost
Hessian block-diagonal and the constraint Jacobian block-bidiagonal, and it is what makes a
Riccati recursion possible at all.

This invariant is load-bearing, not decorative. A cost term coupling `z_k` and `z_{k+1}`
(control-rate or smoothness penalties) makes the Hessian block-**tri**diagonal. Ipopt would
absorb that silently as extra nonzeros; iLQR and DDP **cannot represent it**. Such penalties
are therefore implemented by **state augmentation** — appending `u_{k-1}` to the state via
`models.with_control_rate_penalty` — which restores stage separability at a cost of `m` extra
state dimensions. No coupled cost term is ever added to `Objective`.

**Conic constraint formulation.** Every constraint evaluates to a vector `c(x, u)` whose
feasibility is expressed by membership in a convex cone `K`. Equality, inequality, and
second-order cone constraints share one interface.

**No structural change inside the control loop.** `x0`, `t0`, and `xf` are traced arguments.
Anything that would alter a trace is a build-time concern. See
[section 4](#4-backend-data-model-and-compilation).

---

## 2. Mathematical problem formulation

The discrete-time problem over `N` knot points:

$$\begin{aligned}
\min_{x_{1:N}, u_{1:N-1}} \quad & \ell_N(x_N) + \sum_{k=1}^{N-1} \ell_k(x_k, u_k) \\
\text{s.t.} \quad & x_1 = x_0, \\
& x_{k+1} = f_d(x_k, u_k, t_k, \Delta t_k), \quad k = 1, \dots, N-1, \\
& c_k(x_k, u_k) \in \mathcal{K}_k, \quad k = 1, \dots, N-1, \\
& c_N(x_N) \in \mathcal{K}_N
\end{aligned}$$

- $x_k \in \mathbb{R}^{n}$ — state. $n$ is fixed across the horizon.
- $u_k \in \mathbb{R}^{m}$ — control. The terminal knot point has no control.
- $\delta x_k \in \mathbb{R}^{n_e}$ — error state. $n_e = n$ for Euclidean models;
  $n_e = n - 1$ per unit quaternion in the state.
- $t_k$, $\Delta t_k = t_{k+1} - t_k$ — timestamp and step duration.
- $\ell_k$, $\ell_N$ — stage and terminal cost.
- $f_d$ — discrete dynamics step.
- $c_k$ — constraint vector mapping into cone $\mathcal{K}_k$.

All solver-facing derivative quantities are expressed in **error coordinates** ($n_e$), never
in state coordinates. See [section 11](#11-expansions).

---

## 3. Package structure

```text
src/trajopt/
├── cones.py            ZeroCone, NegativeOrthant, PositiveOrthant, SecondOrderCone
├── rotations/          JPL quaternion algebra, error map, attitude Jacobian, interop
├── dynamics/           ContinuousDynamics, DiscreteDynamics, integrators, rollout
├── costs/              stage + terminal cost, stacked parameters, Objective
├── constraints/        constraint catalog, ConstraintList
├── trajectory.py       struct-of-arrays storage, KnotPoint view
├── problem.py          Problem (structure) and BoundaryConditions (per-step data)
├── program.py          Program (a Problem compiled for one solver) and WarmStart
├── mpc.py              MPC, the receding-horizon driver
├── expansions.py       Expansion: stacked A, B, q, r, Q, R, H in error coordinates
├── transcription/      Z layout, c(Z), COO sparsity pattern, solver adapters
└── models/             benchmark models and model transforms
```

Sub-packages mirror the *functional* split of the Julia ecosystem
(`TrajectoryOptimization.jl`, `RobotDynamics.jl`, `Rotations.jl`, `RobotZoo.jl`) without
vendoring or wrapping any of it. There is no dependency on `jaxlie`, `diffrax`, or `flax`:
matching the Julia reference bit-for-bit requires owning the conventions, and adopting a
third-party rotation library means adopting and then debugging its conventions instead.

---

## 4. Backend, data model, and compilation

### JAX boundary

JAX is used for **evaluation kernels**: dynamics, costs, constraints, cones, projections,
expansions. NumPy appears only where a solver demands concrete host arrays — the Ipopt,
OSQP, and Clarabel adapters in `transcription/`, and the build-time sparsity-pattern
computation.

### Pytrees

All structured objects are `eqx.Module` subclasses.

- **Leaves** (traced): model parameters (mass, length, gravity, inertia), cost matrices `Q`,
  `R`, `H`, `q`, `r`, obstacle centers and radii, bounds, `x0`, `t0`, `xf`, multipliers
  `lam`, penalties `mu`.
- **Static** (`eqx.field(static=True)`): `n`, `m`, `ne`, `N`, integrator choice, constraint
  structure and index ranges, cone types.

Making model parameters leaves means a mass or an obstacle radius can change between solves
without recompiling, and it makes sensitivity with respect to parameters available for free.
`equinox` is preferred over hand-maintained `jax.tree_util.register_dataclass` field tuples
because the traced/static split is declared once per field rather than duplicated in two
tuples per class, where a mistake surfaces as a silent recompile-per-call rather than an
error.

### Compilation units

Four independently jitted phases, matching what Ipopt actually calls at independent rates:

| Phase | Produces | Ipopt callback |
| :--- | :--- | :--- |
| `cost_and_grad` | $J$, $\nabla J$ | `eval_f`, `eval_grad_f` |
| `constraints_and_jac` | $c(Z)$, $\nabla c(Z)$ values | `eval_g`, `eval_jac_g` |
| `hessian` | $\nabla^2\mathcal{L}(Z)$ values | `eval_h` |
| `rollout` | forward simulation | — |

Fusing across these gains nothing, because Ipopt invokes them at different frequencies.

### The zero-recompile invariant

`x0`, `t0`, and `xf` change on **every** MPC iteration. If any of them becomes a trace
constant, every control step triggers a recompile and every deadline is missed. This is
enforced structurally by the `Problem` / `BoundaryConditions` split
([section 13](#13-problem-mpcstate-and-the-mpc-loop)) and asserted by a test that runs 100
MPC iterations and requires the compilation counter to remain at zero.

---

## 5. Trajectory storage

Storage is **struct-of-arrays**, because `vmap` over knot points is the entire performance
argument for the JAX backend and it requires leading-axis-stacked arrays.

| Field | Shape |
| :--- | :--- |
| `X` | `(N, n)` |
| `U` | `(N-1, m)` |
| `t` | `(N,)` |
| `dt` | `(N-1,)` |

`KnotPoint` is retained in the public API as a lightweight **read-only view** constructed on
demand — `kp.x`, `kp.u`, `kp.t`, `kp.dt`, `kp.is_terminal`. It is never the storage of record.
A list of `N` Python objects, as in the Julia design, is precisely the polymorphic container
that cannot be traced.

The flat NLP vector `Z` (section 12) is not the source of truth either: its interleaved
$[x_1, u_1, x_2, u_2, \dots, x_N]$ layout makes `X` and `U` non-contiguous strided views that
cannot be `vmap`ed without a gather. The interleaving is owned exclusively by
`transcription/`.

### Methods

- `X`, `U`, `t` — fields holding the stacked arrays directly.
- `with_states(X)`, `with_controls(U)` — return a new trajectory (arrays are immutable).
- `with_initial_time(t0)` — shift timestamps.
- `shift(dt)` — shift the trajectory forward one step for MPC warm-starting.

---

## 6. Dynamics and integration

### Types

1. `ContinuousDynamics` — $\dot{x} = f(x, u, t)$.
   - `n`, `m`, `ne` — static properties; `ne` defaults to `n`
   - `dynamics(x, u, t) -> Array[n]`
   - `jacobian(x, u, t) -> Array[n, n+m]` — AD-derived
   - `state_diff(x, x0) -> Array[ne]` — defaults to `x - x0`
   - `errstate_jacobian(x) -> Array[n, ne]` — defaults to `I`
2. `DiscreteDynamics` — $x_{k+1} = f_d(x_k, u_k, t_k, \Delta t_k)$.
3. `DiscretizedDynamics` — wraps a `ContinuousDynamics` with an integrator.

Euclidean models never mention the manifold. The three defaults above make `RigidBody` an
override rather than a special case, and every downstream consumer writes $G_k$
unconditionally, paying one identity matmul that `jit` constant-folds away.

### Integrators

**RK4:**

$$\begin{aligned}
k_1 &= f(x_k, u_k, t_k) \\
k_2 &= f(x_k + \tfrac{\Delta t}{2} k_1, u_k, t_k + \tfrac{\Delta t}{2}) \\
k_3 &= f(x_k + \tfrac{\Delta t}{2} k_2, u_k, t_k + \tfrac{\Delta t}{2}) \\
k_4 &= f(x_k + \Delta t\, k_3, u_k, t_k + \Delta t) \\
x_{k+1} &= x_k + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}$$

**Euler:** $x_{k+1} = x_k + \Delta t \cdot f(x_k, u_k, t_k)$

**Implicit midpoint:**
$x_{k+1} - x_k - \Delta t \cdot f\!\left(\tfrac{x_k + x_{k+1}}{2}, u_k, t_k + \tfrac{\Delta t}{2}\right) = 0$

### Rollout

`model.rollout(trajectory)` sets $x_1 = x_0$ and propagates $x_{k+1} = f_d(x_k, u_k)$ using
`jax.lax.scan`. A Python loop over `N` knot points is the loop-latency pitfall verbatim and is
never used in a hot path.

---

## 7. Rotations: JPL quaternions and the error state

`trajopt` uses the **JPL convention with scalar-last storage**, per the reference
implementation in `docs/quaternion.py`. This differs from `Rotations.jl`, which is Hamilton
scalar-first; conversion happens at the cross-test and interop boundary only.

### Convention

| Property | Definition |
| :--- | :--- |
| Storage | $q = [v_1, v_2, v_3, w]$, scalar-last |
| Product | $w = w_1 w_2 - v_1 \cdot v_2$, $\;v = w_2 v_1 + w_1 v_2 - v_1 \times v_2$ |
| Relation to Hamilton | $\text{hamilton}(a, b) = \text{jpl}(b, a)$ |
| Rotation matrix | **Passive** (frame transformation) |
| Conjugate / inverse | $(-v, w)$ |
| Kinematics | $\dot q = \tfrac12 \Xi(q)\,\omega$, body-frame $\omega$ |
| $\Xi(q)$ | $\begin{bmatrix} w I_3 + [v\times] \\ -v^T \end{bmatrix} \in \mathbb{R}^{4\times3}$ |
| Attitude error | $q_{\text{err}} = q \otimes q_{\text{ref}}^{-1}$, body-frame left perturbation |
| Error map | $\delta\theta = 2\,\text{vec}(q_{\text{err}})$ |
| Hamilton bridge | $\text{to\_hamilton}(q) = (-v, w)$, self-inverse |

The minus sign on the cross product is what makes the product JPL rather than Hamilton. The
passive rotation matrix is the self-consistent pairing: mixing an active $R(q)$ with JPL
products is the classic silent sign error.

`to_rot_mat` in the reference implementation routes through
`scipy.spatial.transform.Rotation`, which is not traceable. `rotations/` implements a direct
JAX $q \to R$; `scipy` is confined to interop helpers and tests.

### Error map

The error state is $\delta\theta = 2\,\text{vec}(q_{\text{err}})$, the multiplicative
small-angle error standard in JPL and MEKF work. Three reasons over the Cayley map that
`Rotations.jl` defaults to:

1. The rotation block of $G_k$ becomes exactly $\tfrac12\Xi(q)$, which is already implemented
   and unit-tested.
2. It is linear in $q_{\text{err}}$ — no division, and no `jnp.where` guard in a
   differentiated hot path. The Cayley map's $g = v/w$ diverges as $w \to 0$.
3. It matches the convention in the attitude-MPC and estimation literature, so $G$, the error
   covariance, and any reference implementation agree.

Its cost is a singularity and sign ambiguity at 180°. For tracking MPC this is unreachable,
and none of the section 15 benchmarks performs a large-angle reorientation. The
exponential/log map is the documented upgrade path ([section 16](#16-deferred-work)).

### Attitude Jacobian

$$G_k = \frac{\partial x}{\partial (\delta x)} \in \mathbb{R}^{n \times n_e}$$

For a rigid body with state $[r, q, v, \omega]$ ($n = 13$, $n_e = 12$):

$$G_k = \operatorname{blockdiag}\!\left(I_3,\; \tfrac12 \Xi(q),\; I_3,\; I_3\right) \in \mathbb{R}^{13\times12}$$

Note the direction: $G_k$ maps error-state variations into state variations, which is what
makes the sandwich below dimensionally correct.

### Error-state expansions

- Dynamics: $\bar{A}_k = G_{k+1}^T A_k G_k$, $\;\bar{B}_k = G_{k+1}^T B_k$
- Cost: $\bar{q}_k = G_k^T \nabla_x \ell_k$, $\;\bar{Q}_k = G_k^T (\nabla_{xx}^2 \ell_k) G_k$
- Constraints: $\bar{\nabla}_x c_k = (\nabla_x c_k) G_k$

These are applied **inside** `expansions.py`. No consumer re-derives them.

### Geodesic quaternion cost

$$\ell(x) = \tfrac12 x^T Q x + w \min(1 + q_{\text{ref}}^T q,\; 1 - q_{\text{ref}}^T q)$$

$$\nabla_q \ell = \begin{cases} +w\, q_{\text{ref}} & q_{\text{ref}}^T q < 0 \\ -w\, q_{\text{ref}} & q_{\text{ref}}^T q \ge 0 \end{cases}$$

The inner product $q_{\text{ref}}^T q$ is **convention-invariant**: converting both operands
to JPL negates both vector parts, leaving the dot product unchanged. Only the index layout
changes — `q_ind = [0, 1, 2, 3]` scalar-last, against Julia's `SA[4,5,6,7]`.

### Model structure declaration

Two base classes in v1:

- `EuclideanModel` — `ne == n`, identity error map.
- `RigidBody` — layout fixed to $[r, q, v, \omega]$, `n = 13`, `ne = 12`.

The only v1 model requiring SO(3) is the quadrotor, whose layout is exactly that. The
generalization to `Rotations.jl`-style `LieState{R,P}` partitioning — arbitrary interleaving
of Euclidean blocks and rotations, supporting multi-body systems — is the documented
expansion path. Because the interface is the `state_diff` / `errstate_jacobian` pair, that
generalization is additive and does not break existing models.

### Cross-test operand ordering

`RobotDynamics.state_diff` builds `δq = q0 \ q`, that is
$q_{\text{ref}}^{-1} \otimes_{\text{Ham}} q$ (`liestate.jl:216`), which is the **opposite
operand order** from `Quaternion.error_to`'s $q \otimes_{\text{JPL}} q_{\text{ref}}^{-1}$.

The relation is **derived in full in [`quaternion_operand_ordering.md`](quaternion_operand_ordering.md)**.
Summarised: conjugation is an isomorphism from the JPL product to the Hamilton product, so
$\text{to\_hamilton}(q_{\text{err}}) = \bar q \otimes_{\text{Ham}} \bar q_{\text{ref}}^{-1}$,
and that differs from the Julia quantity by a similarity transform,

$$\delta q_{\text{Julia}} = \text{to\_hamilton}(q_{\text{ref}})^{-1} \otimes_{\text{Ham}} \text{to\_hamilton}(q_{\text{err}}) \otimes_{\text{Ham}} \text{to\_hamilton}(q_{\text{ref}})$$

which at the level of the error vector is a rotation by the reference attitude:

$$\operatorname{vec}(\delta q_{\text{Julia}}) = -\,R(q_{\text{ref}})^{T}\operatorname{vec}(q_{\text{err}}), \qquad \operatorname{scalar}(\delta q_{\text{Julia}}) = \operatorname{scalar}(q_{\text{err}})$$

with $R(q_{\text{ref}})$ the JPL (passive) rotation matrix of the reference attitude. The two
error vectors are the same relative rotation resolved in two different frames — the
left-versus-right multiplicative error distinction — which is why no global sign relates them.

Because the two share a scalar part and their vector parts share a norm, every error map in use
here ($\delta\theta = 2v$, Cayley, exponential) rescales both by the same factor, so the same
relation holds after the map:

$$\delta\theta_{\text{Julia}} = -\,R(q_{\text{ref}})^{T}\,\delta\theta_{\text{Python}}$$

**No convention change is required.** The conventions reconcile exactly, in closed form, using
only quantities both sides already compute.

Consequences for the cross-tests, which are binding:

1. That relation is what the test asserts. It is **not** discovered by flipping signs until the
   test passes: a sign error in `to_hamilton` can cancel a sign error in the kernel and produce
   a green test over a wrong implementation. `to_hamilton` is cross-tested independently against
   known Hamilton values first, before any conjugated comparison exists.
2. The test pair must have a **nonzero cross product between the quaternion vector parts** — the
   two orderings differ by exactly $-2\,(x \times y)$ and by nothing else. A pure-`x` against a
   pure-`y` rotation is the case of record.
3. Coaxial pairs and an identity reference are **degenerate**: the orderings coincide there, so
   such a pair would pass against a reversed implementation. Both are pinned as labelled
   negative controls in `test/unit/test_quaternion_ordering.py` so they are never mistaken for
   coverage.

The derivation is verified symbolically for a general quaternion pair, numerically over the
`x`/`y` pair plus 200 random pairs, and against a live `Rotations.jl` evaluation, agreeing to
`1.1e-16`.

---

## 8. Cost functions and objectives

### Form

$$\ell(x, u) = \tfrac12 x^T Q x + \tfrac12 u^T R u + u^T H x + q^T x + r^T u + c$$

### Objective structure

`Objective` holds **one stage cost and one terminal cost**, homogeneous in type, with
parameters stacked over the horizon:

| Field | Shape |
| :--- | :--- |
| `Q` | `(N-1, n, n)`, or `(N-1, n)` if diagonal |
| `R` | `(N-1, m, m)`, or `(N-1, m)` if diagonal |
| `H` | `(N-1, m, n)` |
| `q` | `(N-1, n)` |
| `r` | `(N-1, m)` |
| `c` | `(N-1,)` |
| terminal `Q_f`, `q_f`, `c_f` | `(n, n)`, `(n,)`, `()` |

This is a single `vmap` over `k`. A list of `N` heterogeneous Python cost objects — the Julia
design — is the polymorphic-dispatch pitfall in its purest form, and it buys a capability
(different cost *types* at different knot points) that nothing in the benchmark set needs.
Stacked parameters preserve everything the list actually provided:

- `LQRObjective(Q, R, Qf, N)` — stacked-constant parameters carrying shape only; the target arrives as a reference window through `BoundaryConditions`.
- `TrackingObjective(Q, R, Z_ref)` — stacked time-varying parameters, with
  $q_k = -Q x_{\text{ref},k}$ and $r_k = -R u_{\text{ref},k}$.
- `update_reference(obj, Z_ref, start=0)` — rebuilds the stacked `q` and `r` arrays.

### Cost variants

| Variant | Structure |
| :--- | :--- |
| Diagonal | `Q`, `R` stored as `(N-1, n)`, `(N-1, m)`; `H = 0`. Hessian inversion is $O(n+m)$. |
| Dense quadratic | Full `Q`, `R`, `H`. |
| LQR tracking | $q = -Qx_f$, $r = -Ru_f$, $c = \tfrac12 x_f^T Q x_f + \tfrac12 u_f^T R u_f$. |
| Quaternion geodesic | Adds the section 7 geodesic penalty. |
| Error-state quadratic | $\tfrac12 (x \ominus x_d)^T Q (x \ominus x_d)$ using the section 7 error map. |
| Generic | Arbitrary user callable, differentiated by AD. |

`GenericCost` is a plain Python function of `(x, u, t)` and is traced like anything else; it
carries no special machinery.

### Methods

- `evaluate(x, u, t) -> float`
- `gradient(x, u, t) -> Array[n+m]`
- `hessian(x, u, t) -> Array[n+m, n+m]`
- `invert(...)` — analytic inverse using the diagonal or block-diagonal shortcut
- `cost(obj, traj) -> float` — $\sum_k \ell_k$, one `vmap` plus a sum

---

## 9. Cones and projections

Every constraint is expressed as $c(x,u) \in \mathcal{K}$.

| Cone | Set | Projection $\Pi_{\mathcal{K}}(x)$ |
| :--- | :--- | :--- |
| `ZeroCone` (equality) | $x = 0$ | $0$ |
| `NegativeOrthant` (inequality) | $x \le 0$ | $\min(0, x)$ |
| `PositiveOrthant` | $x \ge 0$ | $\max(0, x)$ |
| `SecondOrderCone` | $\lVert v\rVert_2 \le s$, $x = [v; s]$ | piecewise, below |

### Second-order cone

Let $x = [v; s]$, $v \in \mathbb{R}^{p-1}$, $a = \lVert v \rVert_2$.

$$\Pi_{\mathcal{K}}(x) = \begin{cases}
0 & a \le -s \quad (\text{below dual cone}) \\
x & a \le s \quad (\text{inside}) \\
\tfrac12\left(1 + \tfrac{s}{a}\right) \begin{bmatrix} v \\ a \end{bmatrix} & a > |s| \quad (\text{outside})
\end{cases}$$

$$\nabla \Pi_{\mathcal{K}}(x) = \begin{cases}
0 & a \le -s \\
I & a \le s \\
\tfrac12\begin{bmatrix} \left(1+\tfrac{s}{a}\right)I - \tfrac{s}{a^3} v v^T & \tfrac{v}{a} \\ \tfrac{v^T}{a} & 1 \end{bmatrix} & a > |s|
\end{cases}$$

$\nabla^2 \Pi_{\mathcal{K}}(x)[b]$ is obtained by AD, not hand-derived.

### Branchless implementation

The three-way branch is implemented with `jnp.where`, never Python `if`/`elif`, and the
division is guarded as $\max(a, 10^{-10})$. Python control flow on traced values either fails
outright or silently bakes one branch into the trace.

The existing pure-NumPy `src/trajopt/cones.py` predates this specification and is replaced
wholesale.

---

## 10. Constraint catalog and ConstraintList

### Base classes

- `StageConstraint` — depends on $x_k$ and $u_k$.
- `StateConstraint` — depends on $x_k$ only; control Jacobian is zero.
- `ControlConstraint` — depends on $u_k$ only; state Jacobian is zero.

### Catalog

| Constraint | Formula | Sense |
| :--- | :--- | :--- |
| `GoalConstraint` | $x_k[\text{inds}] - x_f = 0$ | Equality |
| `StateBound` | $x_k - x_{\max} \le 0$, $x_{\min} - x_k \le 0$ | Inequality |
| `ControlBound` | $u_k - u_{\max} \le 0$, $u_{\min} - u_k \le 0$ | Inequality |
| `BoundConstraint` | combined state and control box bounds | Inequality |
| `LinearConstraint` | $A[x_k; u_k] - b \le 0$, or $= 0$ | Either |
| `CircleConstraint` | $-(x - x_c)^2 - (y - y_c)^2 + r^2 \le 0$ | Inequality |
| `SphereConstraint` | $-(x-x_c)^2 - (y-y_c)^2 - (z-z_c)^2 + r^2 \le 0$ | Inequality |
| `CollisionConstraint` | $r^2 - \lVert x[b_1] - x[b_2]\rVert^2 \le 0$ | Inequality |
| `NormConstraint` | $\lVert y\rVert_2^2 - a^2 \le 0$, or $[y; a] \in \mathcal{K}_{SOC}$ | Inequality / SOC |
| `QuatVecEq` | quaternion attitude equality | Equality |
| `IndexedConstraint` | wraps a sub-system constraint onto a slice of $(x, u)$ | inherited |
| `DynamicsConstraint` | $x_{k+1} - f_d(x_k, u_k) = 0$ (explicit) | Equality |

Implicit collocation:
$x_k - x_{k+1} + \Delta t\, f\!\left(\tfrac{x_k+x_{k+1}}{2}, u_k\right) = 0$.

Jacobians are AD-derived. The analytic forms are retained in this document as documentation
and as expected values for cross-tests, not as implementation.

`BoundConstraint`, `StateBound`, and `ControlBound` additionally expose a `primal_bounds()`
path that maps directly onto solver variable limits $z_L \le z \le z_U$ rather than becoming
rows of $c(Z)$.

### ConstraintList

Constraints and their active knot-point ranges are **fused at build time** into a single
concatenated $c_k(x_k, u_k)$ per knot point. One `vmap` over `k`; no runtime batching logic,
no grouping by type.

Fields:

- `n`, `m` — dimensions (static).
- `constraints` — the registered constraint objects.
- `inds` — knot-point index range per constraint.
- `p` — array of length `N`, total constraint dimension at each knot point.

Operations:

- `add_constraint(con, inds)` — dimension check and registration.
- `num_constraints() -> p`
- `primal_bounds() -> (zL, zU)`
- `build()` — trace and fuse into per-knot functions.

Two consequences of fusing at build time. `IndexedConstraint` — a Python-level composition
that would otherwise fight `vmap` — is resolved during tracing, so `vmap` never sees it. And
trace time scales with total constraint count, while every structural change invalidates the
trace. The latter is acceptable precisely because section 4 already forbids structural change
inside the control loop.

The Julia `sigs` (in-place versus return-new signature) and `diffs` (autodiff versus finite
difference versus analytic) fields are **removed**. `sigs` is meaningless when arrays are
immutable. `diffs` is meaningless when AD of, say, $-(x-x_c)^2-(y-y_c)^2+r^2$ compiles to
exactly the analytic expression — an analytic override would be a second place to be wrong
for no gain.

---

## 11. Expansions

`expansions.py` is the shared seam between NLP transcription (v1) and native iLQR/ALTRO (v2).

### Interface

Three composable methods on the owning objects, each producing an `Expansion` module of
stacked arrays:

```python
problem.dynamics_expansion(traj) -> Expansion
problem.cost_expansion(traj) -> Expansion
problem.augmented_lagrangian_expansion(traj, expansion, lam, mu) -> Expansion
```

| Field | Shape |
| :--- | :--- |
| `A` | `(N-1, ne, ne)` |
| `B` | `(N-1, ne, m)` |
| `q` | `(N, ne)` |
| `r` | `(N-1, m)` |
| `Q` | `(N, ne, ne)` |
| `R` | `(N-1, m, m)` |
| `H` | `(N-1, m, ne)` |

### Error coordinates

Expansions are returned in **error coordinates** ($n_e$), with $G_k$ applied inside. Two
reasons this is not negotiable:

1. The augmented-Lagrangian terms below "add directly into $q_k, r_k, Q_k, R_k, H_k$", which
   is only true if every contribution already lives in one coordinate system.
2. In state coordinates the quaternion's unit-norm direction is a null direction of the
   Hessian, which will wreck a KKT factorization.

For Euclidean models $n_e = n$ and $G = I$, which `jit` folds away entirely. Returning state
coordinates instead would mean every future consumer re-derives the same $G$ sandwich, and
eventually one of them gets it wrong.

### Augmented Lagrangian

For $c(x,u) \in \mathcal{K}$ with multiplier $\lambda$ and penalty $\mu > 0$:

$$\mathcal{L}_A = \ell(x,u) + \lambda^T \Pi_{\mathcal{K}^*}\!\left(c + \tfrac{\lambda}{\mu}\right) + \frac{\mu}{2}\left\lVert \Pi_{\mathcal{K}^*}\!\left(c + \tfrac{\lambda}{\mu}\right)\right\rVert^2$$

Its gradient and Hessian add into the `Expansion` fields rather than into a separate
structure.

---

## 12. NLP transcription

### Primal vector

$$Z = [x_1^T, u_1^T, x_2^T, u_2^T, \dots, x_{N-1}^T, u_{N-1}^T, x_N^T]^T \in \mathbb{R}^{Nn + (N-1)m}$$

### Constraint vector

$$c(Z) = \begin{bmatrix} x_1 - x_0 \\ x_2 - f_d(x_1, u_1) \\ c_1(x_1, u_1) \\ \vdots \\ x_N - f_d(x_{N-1}, u_{N-1}) \\ c_N(x_N) \end{bmatrix} \in \mathbb{R}^P$$

### Sparse assembly

Ipopt's `eval_jac_g` has a structure callback invoked once and a values callback invoked every
iteration, returning a flat array in the structure's order. Accordingly:

1. **Build time.** Compute the `(row, col)` COO pattern from `N`, `n`, `m`, `p` using plain
   NumPy. The pattern is a pure function of the dimensions.
2. **Runtime.** A `vmap` produces dense per-knot blocks; a reshape and concatenate place their
   values into the `data` array in the pattern's order. No gather, no per-iteration
   allocation.

Each per-knot block is treated as **dense**; structural zeros inside a block are not
exploited. Explicit zeros cost Ipopt almost nothing at these problem sizes, whereas
per-constraint sparsity masks would make the pattern depend on constraint *types*,
reintroducing exactly the heterogeneity that build-time fusion (section 10) eliminates.

Building a `scipy.sparse` matrix per call is the memory-blow-up pitfall wearing a disguise: it
allocates inside the iteration loop.

### Structure

- $\nabla c(Z)$ — block bidiagonal from dynamics, block diagonal from stage constraints.
- $\nabla^2 \mathcal{L}(Z)$ — block diagonal, one block per knot point. Block-diagonality
  holds only because of the section 1 invariant.

---

## 13. Problem, MPC, and the control loop

Structure and per-step data are **separate types**. This turns "never let `x0` become static"
from a discipline into a property of the type system: nothing in `Problem` can change per
step, and nothing in `BoundaryConditions` is ever a trace constant.

### `Problem` — structure, the compilation cache key

| Field | Kind |
| :--- | :--- |
| `model` | `eqx.Module`; parameters are leaves, `n`/`m`/`ne` static |
| `obj` | `Objective` with stacked parameters |
| `constraints` | built `ConstraintList` |
| `N` | static |
| `dt` | leaf, shape `(N - 1,)` |
| `integrator` | static |

`Problem` is itself an `eqx.Module` passed as a traced pytree. The compilation cache key is
its treedef plus its static fields; model parameters flow through as leaves, which is what
makes "change the mass without recompiling" actually true.

### `BoundaryConditions` — per-step data, always traced

| Field | Kind |
| :--- | :--- |
| `x0`, `t0` | leaves |
| `X_ref`, `U_ref` | leaves (the reference window the objective tracks) |
| `xf` | leaf (the terminal goal constraints bind; separate from the window) |

Zero static fields, so a boundary update can never key a recompile.

### `Program` — a Problem compiled and allocated for one solver

Mutable, eager-side, and the single `jax.jit` call site: it caches the traced cores keyed by
function and shape, and holds any live backend handle across receding-horizon steps.

### `WarmStart` — the primal/dual iterates carried between steps

| Field | Kind |
| :--- | :--- |
| `Z` | flat primal vector |
| `lam`, `mu` | transcription duals |
| `al` | AL duals and penalties, or `None` |

### `MPC` — the driver

Holds one `Program`, the current `BoundaryConditions`, and the warm start.

- `mpc.solve()` — solve this step, folding the result into the warm start and returning the
  backend's `SolverResult`.
- `mpc.cost()` — scalar objective at the current warm start.
- `mpc.states`, `mpc.controls`, `mpc.trajectory()` — the current plan.
- `mpc.measure(x, t)` — inject the measurement into $x_0$, $t_0$.
- `mpc.set_goal(xf)` — replace the reference with a constant window.
- `mpc.set_reference(window)` / `mpc.push_reference(x_ref, u_ref)` — replace the tracked window
  wholesale, or stage the point that enters it at the far end on the next shift.
- `mpc.shift(dt)` — advance a knot: warm start and reference window both slide.

### Control loop

```text
problem = Problem(model, obj, constraints, N, dt)   # built once, compiled once
mpc     = MPC(problem, Ipopt(), x0=x0, t0=t0, xf=xf)

loop:
    mpc.measure(x_measured, t_current)
    mpc.set_goal(xf)                 # optional
    mpc.solve()                      # Ipopt() / OSQP() / Clarabel() / ALTRO() / ...
    u_command = mpc.controls[0]
    mpc.shift(dt)                    # warm start next solve
```

The Julia `set_goal_state!` mutates the problem, the objective, and the goal constraints in
sync. Under this split, $x_f$ lives in the boundary conditions alone and both the objective and
the goal constraint read it as an argument, so there is nothing to keep in sync.

---

## 14. Models

`models/` contains the four cross-test targets and the model transforms that this
specification's own decisions have made mandatory.

### Benchmark models

`Cartpole`, `Pendulum`, `Quadrotor`, `DubinsCar` — parameters matched **bit-for-bit** to
`RobotZoo.jl` (gravity, masses, lengths, and the quadrotor's motor mixing matrix). That
matching is what makes the dynamics row of section 15 meaningful; a model that differs by a
parameter produces a cross-test that silently tests nothing.

`Quadrotor` is a `RigidBody` with state $[r, q, v, \omega]$, $n = 13$, $n_e = 12$. RobotZoo
stores its quaternion Hamilton scalar-first, so its cross-test converts both the state **and**
the Jacobian.

### Model transforms

- `with_control_rate_penalty(model, R_delta)` — augments the state with $u_{k-1}$ so that a
  control-rate penalty becomes an ordinary stage cost, preserving the section 1 invariant.
- `model.linearize(trajectory)` — trajectory linearization for MPC.

---

## 15. Verification strategy

Two independent oracles, used for different things.

- **`juliacall`, in-process, component level.** Exact numerical parity on mathematical
  operations where indexing, signs, coordinate frames, or differentiation paths can diverge
  silently.
- **A standalone pure-CasADi implementation (`casadi.Opti`), end-to-end.** Full OCP and MPC
  solves.

Julia is deliberately **not** used as an end-to-end oracle; see
[Appendix A](#appendix-a-divergences-from-trajectoryoptimizationjl) for why.

### Cross-test harness

```python
# test/cross_verification/test_cross_cones.py
import numpy as np
import pytest

from trajopt.cones import SecondOrderCone


@pytest.mark.julia
def test_soc_projection(jl_to) -> None:
    TO = jl_to.TO
    # Every Julia name here contains "!" or a Unicode nabla, so none of them is a valid
    # Python identifier. They must be reached with getattr, not attribute syntax.
    jl_project = getattr(TO, "projection!")
    jl_jacobian = getattr(TO, "∇projection!")

    py_cone = SecondOrderCone()
    jl_cone = TO.SecondOrderCone()

    x = np.array([2.0, 3.0, 1.0, 1.0])

    px_py = py_cone.project(x)
    px_jl = np.zeros_like(x)
    jl_project(jl_cone, px_jl, x)
    np.testing.assert_allclose(px_py, px_jl, rtol=1e-14, atol=1e-14)

    J_py = py_cone.jacobian(x)
    J_jl = np.zeros((4, 4))
    jl_jacobian(jl_cone, J_jl, x)
    np.testing.assert_allclose(J_py, J_jl, rtol=1e-12, atol=1e-12)
```

The `jl_to` session fixture activates `trajopt_jl/` and skips when no Julia runtime is
present. All such tests carry `@pytest.mark.julia`, deselectable with `-m "not julia"`.

The Julia cone API is `projection!`, `∇projection!`, and `∇²projection!` — each takes the cone
first and writes into a preallocated output. There is no `grad_projection!`. An earlier draft of
this document named one, and the test written against that name failed at attribute lookup
rather than at a numerical comparison, which is why the API is pinned here explicitly.

### What must be cross-tested against Julia

| Component | Targets | Tolerance |
| :--- | :--- | :--- |
| Cones and projections | $\Pi$, $\nabla\Pi$, $\nabla^2\Pi[b]$ for all four cones, across every region (inside, outside, below dual cone) | `1e-14` values, `1e-12` derivatives |
| Dynamics and integrators | continuous $f$, `discrete_dynamics`, $[A_k, B_k]$ for RK4, Euler, implicit midpoint, on all four benchmark models | `1e-14` steps, `1e-12` Jacobians |
| Rotations | $\Xi(q)$, $G_k$, error state, $\bar A_k = G_{k+1}^T A_k G_k$, geodesic cost and its double-cover branches | `1e-12` |
| Costs and objectives | `evaluate`, `gradient`, `hessian`, `invert` for diagonal, dense, LQR, tracking | `1e-14` values, `1e-12` derivatives |
| Constraint catalog | $c(x,u)$ and $[\nabla_x c, \nabla_u c]$ for the whole catalog, across active knot points | `1e-12` |

The rotations row is a **conjugated** comparison, through `to_hamilton`, which is strictly
weaker than a direct one: a convention error in the bridge can cancel a convention error in
the kernel. Two mitigations are mandatory. `to_hamilton` is cross-tested independently against
known Hamilton values before any conjugated comparison exists, and the operand-ordering
relation of section 7 is derived symbolically rather than fitted. Both are discharged in
`test/unit/test_quaternion_ordering.py`.

Where Python uses a different error map from the Julia default, the cross-test passes the
matching map explicitly — `Rotations.QuatVecMap()`, scaling `1.0` — rather than relying on
`state_diff`'s `CayleyMap` default.

### What is not cross-tested

1. **Storage and memory layout** — `KnotPoint` accessors, view semantics, container mechanics,
   row-major versus column-major buffers.
2. **Allocation metrics** — Julia's `@allocated` zero-allocation assertions have no analogue.
   JAX memory behaviour is verified with XLA profiling and the zero-recompile test instead.
3. **Type hierarchy and dispatch** — Julia's `AbstractConstraint{S,D}` tree and multiple
   dispatch signatures. Python uses `eqx.Module` and explicit composition.
4. **Line-search step sequences** — intermediate backtracking $\alpha$ values. FMA and
   associativity differences between XLA and LLVM perturb them without changing the converged
   solution.
5. **Builders and sugar** — chaining helpers, plotting, docstrings, CLI wrappers.

### End-to-end validation against pure CasADi

Each benchmark is formulated identically in `trajopt` and in a standalone `casadi.Opti`
transcription, matching discretization, cost matrices, boundary conditions, constraint sets,
and Ipopt options.

- Primal parity: $\lVert X^*_{\text{trajopt}} - X^*_{\text{CasADi}}\rVert_\infty \le 10^{-5}$,
  and likewise for $U^*$.
- Objective parity:
  $|J^*_{\text{trajopt}} - J^*_{\text{CasADi}}| / J^*_{\text{CasADi}} \le 10^{-5}$.
- Constraint satisfaction: $\lVert c(Z^*)\rVert_\infty \le \epsilon_{\text{feas}}$, with dual
  multiplier parity under identical solver settings.

### Benchmarking

Timing is broken down into transcription latency (assembling sparsity patterns, bounds, and
structures), derivative evaluation per iteration, pure solver runtime, and sustained
closed-loop MPC rate with latency jitter and warm-start speedup.

Benchmark problems:

- **Cartpole swing-up** — underactuated, bounded actuation, state limits.
- **Quadrotor obstacle avoidance** — SO(3) attitude tracking through spherical keep-out zones.
- **Dubins car** — nonholonomic navigation with corridor constraints and a tracking objective.

### Standing risks

- **Sparse assembly from `vmap`ed dense blocks is the largest chunk of genuinely novel work in
  v1.** No library does it, and the COO ordering must line up exactly with Ipopt's structure
  callback or the failure mode is a wrong answer rather than an error.
- **The $\delta\theta$ map's 180° singularity is a real constraint on expressible problems.**
  It is unreachable for all three benchmarks, but it rules out large-angle reorientation until
  the exponential map lands.

---

## 16. Deferred work

Documented expansion paths, each designed to be additive rather than breaking:

| Deferred | Enabled by |
| :--- | :--- |
| `LieState{R,P}` partitioning for multi-rotation states | the `state_diff` / `errstate_jacobian` interface pair |
| Time-varying $n_k$, $m_k$ for hybrid and contact problems | requires abandoning fixed scalars; ragged buffers are incompatible with `vmap` |
| Exponential/log error map for large-angle maneuvers | error map is a named constant behind one interface |
| Native iLQR and ALTRO | `expansions.py`, cut before its first consumer |
| Analytic Jacobian overrides | none planned; AD compiles to the analytic form |

---

## Appendix A: divergences from TrajectoryOptimization.jl

### Structural

| Julia | `trajopt` | Reason |
| :--- | :--- | :--- |
| `SampledTrajectory` as a list of `KnotPoint` | struct-of-arrays; `KnotPoint` is a view | a list of Python objects cannot be traced or `vmap`ed |
| `Objective` as `N` heterogeneous cost objects | one stage + one terminal, parameters stacked | same reason; stacked parameters preserve every used capability |
| `ConstraintList.sigs` | removed | in-place versus return-new is meaningless with immutable arrays |
| `ConstraintList.diffs` | removed | AD compiles to the analytic form; an override is a second place to be wrong |
| In-place `!` mutators | value-returning methods | arrays are immutable |
| `Problem` holds `x0`, `xf` | split into `Problem`, `BoundaryConditions` and the `MPC` driver | makes the zero-recompile invariant structural |
| Time-varying $n_k$, $m_k$ | fixed `n`, `m` | ragged buffers are incompatible with `vmap`; no benchmark needs it |
| Multiple dispatch on abstract type trees | `eqx.Module` and explicit composition | no Python equivalent worth emulating |

### Conventions

| Julia | `trajopt` |
| :--- | :--- |
| Hamilton quaternions, scalar-first `[w,x,y,z]` | JPL, scalar-last `[x,y,z,w]` |
| Active rotation matrix | passive (frame transformation) |
| `CayleyMap` error map (default) | $\delta\theta = 2\,\text{vec}(q_{\text{err}})$ |
| `state_diff` uses $q_{\text{ref}}^{-1}\otimes q$ | `error_to` uses $q \otimes q_{\text{ref}}^{-1}$; related by $\delta\theta_{\text{Julia}} = -R(q_{\text{ref}})^{T}\delta\theta$, section 7 |
| `q_ind = SA[4,5,6,7]` | `q_ind = [0,1,2,3]` |

### Gaps in the Julia reference

Two rows of an earlier draft of the verification table could not be built, because the
corresponding Julia functionality does not exist:

- **No solver.** `trajopt_jl/Project.toml` depends on `RobotDynamics`, `RobotZoo`,
  `Rotations`, `StaticArrays`, `ForwardDiff`, and `FiniteDiff` — but not on `Altro.jl`. There
  is no ALTRO or iLQR implementation to compare against. Adding `Altro.jl` as a test
  dependency was considered and rejected: it would pin the comparison to another solver's
  parameter defaults and iteration schedule, for agreement that is only meaningful to `1e-5`
  anyway. The pure-CasADi baseline is a better oracle because it is *independent* rather than
  a sibling implementation.
- **`TrajOptNLP` is exported but never defined.** It appears in the export list at
  `trajopt_jl/src/TrajectoryOptimization.jl:38`, but no `nlp.jl` exists and no file defines
  it. There is no Julia NLP transcription against which to cross-test $Z$, $c(Z)$, or the
  sparsity patterns. These are validated against CasADi instead.

Additionally, much of what an earlier draft attributed to `TrajectoryOptimization.jl` actually
lives in its dependencies: `KnotPoint`, `SampledTrajectory`, `state_diff`,
`state_diff_jacobian!`, the integrators, and `rollout!` are all imported from
`RobotDynamics.jl`, and the benchmark models are `RobotZoo.jl`. They remain reachable through
`juliacall` for cross-testing, but the Python port's scope is correspondingly broader than
"port `TrajectoryOptimization.jl`".

### Corrections to the earlier draft

- The attitude Jacobian was given as
  $G_k = \partial(x \ominus x_d)/\partial x \in \mathbb{R}^{13\times12}$. That derivative is
  $12\times13$. The shape was right and the formula inverted:
  $G_k = \partial x/\partial(\delta x)$, which is what makes $\bar A_k = G_{k+1}^T A_k G_k$
  dimensionally consistent.
- The "NumPy view aliasing and silent mutation" pitfall and its `.copy()` remedy are void.
  JAX arrays are immutable and the failure mode cannot occur. The real JAX pitfall in its
  place is buffer donation and `.at[].set()` producing full copies inside hot loops.
- The quaternion mathematics throughout was transcribed from Hamilton scalar-first Julia and
  has been rewritten for JPL scalar-last.
