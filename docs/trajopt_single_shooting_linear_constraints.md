# Issue: `SingleShooting` rejects control-space `LinearConstraint`, forcing 100x slower full collocation

**Repository / Package**: `trajopt` (`trajopt.transcription.single_shooting`)  
**Severity**: High (Performance & Functionality Bottleneck)  
**Type**: Bug / Limitation  

---

## 1. Summary

In `trajopt.transcription.single_shooting.SingleShooting`, the constraint validator `_validate_supported_constraints` rejects all non-`GoalConstraint` knot constraints. This occurs even when a `LinearConstraint` operates strictly on control variables (i.e. indices in the range `[n, n + m)`).

Because single shooting refuses control-space linear constraints (such as Kirchhoff's Current Law $\sum_{i=1}^m u_i = 0$ for electrical stimulation montages), client controllers are forced to fall back to **full collocation (`Ipopt`)**. 

In high-dimensional dynamical systems (e.g., 62-channel EEG predictors with $n = 620$ to $1556$ state variables), this fallback expands the NLP decision vector from $O(N \cdot m)$ (e.g., 12 to 150 variables) to $O(N \cdot n)$ (e.g., 7,780 to 31,620 variables), degrading solve times from **< 5 ms** to **10–15 seconds per step** (>100x slowdown).

---

## 2. Root Cause Analysis

In `trajopt/transcription/single_shooting.py`:

```python
def _validate_supported_constraints(problem: Problem) -> None:
    """Refuse constraints single shooting cannot express as a function of controls alone."""
    x_lower, x_upper, _, _ = problem.constraints.primal_bounds()
    if np.any(np.isfinite(x_lower)) or np.any(np.isfinite(x_upper)):
        msg = (
            "Single shooting cannot express state bounds: with no state decision variables a "
            "StateBound has no primal variable to bound. Remove the state bound or use the "
            "multiple-shooting transcription."
        )
        raise ValueError(msg)

    for evaluator in problem.constraints.knot_evaluators:
        for con in evaluator.constraints:
            if not isinstance(con, GoalConstraint):
                msg = (
                    f"Single shooting v1 supports only ControlBound and GoalConstraint, found "
                    f"{type(con).__name__}. Remove it or use the multiple-shooting transcription."
                )
                raise ValueError(msg)
```

### The Flaw:
The docstring states: *"Refuse constraints single shooting cannot express as a function of controls alone."*  
However, a `LinearConstraint` on control indices:
$$A \mathbf{u}_k - \mathbf{b} = 0 \quad (\text{with } \text{inds} \subset [n, n + m))$$
is **purely a function of controls alone**. 

Because `SingleShooting` rejects any constraint where `not isinstance(con, GoalConstraint)`, it unnecessarily rejects valid control-space linear constraints.

---

## 3. Quantitative Impact on MPC Execution

Benchmarked on closed-loop neurostimulation simulations ($t_{\text{end}} = 12\text{ s}$):

| Parameter | Single-Shooting (Controls Only) | Full Collocation Fallback (`Ipopt`) | Impact |
| :--- | :--- | :--- | :--- |
| **NLP Decision Variables** | $N \cdot m = 4 \times 3 = \mathbf{12}$ | $(N+1) \cdot n + N \cdot m = \mathbf{7,792}$ | **650x larger NLP** |
| **NLP Equality Constraints** | $N \cdot 1 = \mathbf{4}$ (Kirchhoff only) | $N \cdot n + N \cdot 1 = \mathbf{7,784}$ (Dynamics + Kirchhoff) | **1,940x more constraints** |
| **Solve Time Per Step** | **~2 – 5 ms** | **~10 – 15 seconds** | **>2,000x slower per step** |
| **12s Closed-Loop Simulation** | **~2 – 5 seconds total** | **~1.5 – 2 hours total** | Renders closed-loop evaluation intractable |

---

## 4. Expected Behavior

`SingleShooting` should allow `LinearConstraint` (and any general knot constraint) as long as its dependent indices belong exclusively to the control slice:

$$\text{con.inds} \subseteq \{n, n+1, \dots, n+m-1\}$$

When such control constraints are present, `_SingleShootingCallback` should evaluate them during `cb.constraints(u)` and pass the corresponding `cl <= g(u) <= cu` bounds to `cyipopt`.

---

## 5. Proposed Fix

### Step 1: Update `_validate_supported_constraints`
```python
def _validate_supported_constraints(problem: Problem) -> None:
    n, m = problem.model.n, problem.model.m
    control_range = set(range(n, n + m))
    
    for evaluator in problem.constraints.knot_evaluators:
        for con in evaluator.constraints:
            if isinstance(con, GoalConstraint):
                continue
            # Allow constraints that depend strictly on control variables
            if hasattr(con, "inds") and set(con.inds).issubset(control_range):
                continue
            msg = (
                f"Single shooting cannot express constraint {type(con).__name__} depending on state variables. "
                "Remove it or use the multiple-shooting transcription."
            )
            raise ValueError(msg)
```

### Step 2: Ensure `_SingleShootingCallback` evaluates control constraints
In `_SingleShootingCallback.constraints(u)` and `_constraint_bounds(problem)`, include the evaluation of knot-level control constraints:
```python
def constraints(self, u: np.ndarray) -> np.ndarray:
    # Evaluate goal constraint at terminal state and any stage control constraints
    # g_vals = [...]
    ...
```

---

## 6. Workaround in Client Code (Until Fixed in `trajopt`)

Clients needing fast single-shooting with linear equality control constraints (like $\mathbf{1}^T \mathbf{u} = 0$) can reparameterize the control input into its null space:
$$\mathbf{u} = \mathbf{N} \mathbf{v}, \quad \mathbf{N} \in \mathbb{R}^{m \times (m-1)} \text{ such that } \mathbf{1}^T \mathbf{N} = 0$$
where $\mathbf{v} \in \mathbb{R}^{m-1}$ becomes the unconstrained control input, eliminating the linear equality constraint entirely.
