# TrajectoryOptimization (`trajopt`)

A high-performance framework for formulating, evaluating, and solving discrete-time optimal control and trajectory optimization problems in **Python**, powered by JAX and Equinox.

Inspired by the Julia library [`TrajectoryOptimization.jl`](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl), `trajopt` is built from scratch around JAX's compilation model and vectorized differentiation.

---

## Features

- **Fast Automatic Differentiation & Expansion Engine:** Dedicated expansion engine (`trajopt.expansions`) powered by JAX and Equinox (`eqx.Module`, `eqx.filter_jit`) computing analytical-accuracy first- and second-order derivatives (cost Hessians, dynamics Jacobians, and constraint Jacobians) with zero manual derivative bookkeeping.
- **Conic Constraint Formulation:** Unifies equality, inequality (orthant), and second-order cone (SOC) constraints via projection operations and explicit Jacobians.
- **Manifold Kinematics on $\mathrm{SO}(3)$:** Native 3D rotation and attitude dynamics using JPL quaternions and error-state representations ($\delta\theta = 2\operatorname{vec}(q_{\text{err}})$) for rigid bodies and quadrotors.
- **Multiple Solver Integrations:** Direct NLP transcriptions and solver adapters for **Ipopt** (`cyipopt`), **OSQP**, and **Clarabel**.
- **Cross-Verification & Parity:** Rigorous test suite validating numerical parity against the Julia reference implementation and an independent CasADi baseline.

---

## Installation (Python / `uv`)

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible Python environment and dependency management.

### 1. Prerequisites (Ipopt & Build Tools)

Because `cyipopt` compiles C++ extensions against the underlying COIN-OR Ipopt solver, ensure system prerequisites are installed:

#### Windows

1. **C++ Compiler:** Install [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select *Desktop development with C++*).
2. **`pkg-config`:**

   ```powershell
   winget install pkg-config
   # or: choco install pkgconfiglite
   ```

3. **Ipopt C++ Binaries:**
   - Download the latest precompiled MSVC release (e.g. `Ipopt-3.14.x-win64-msvs2022-md.zip`) from [COIN-OR Ipopt Releases](https://github.com/coin-or/Ipopt/releases).
   - Extract to a directory (e.g., `C:\Ipopt\Ipopt-3.14.19-win64-msvs2022-md`).
   - Copy or rename `lib/ipopt.dll.lib` to `lib/ipopt.lib` (and `lib/coinmumps.dll.lib` to `lib/coinmumps.lib`).

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential pkg-config coinor-libipopt-dev coinor-libipopt1v5 gfortran
```

#### macOS

```bash
brew install ipopt pkg-config
export PKG_CONFIG_PATH="$(brew --prefix ipopt)/lib/pkgconfig:$PKG_CONFIG_PATH"
```

---

### 2. Configure Environment (`.env`)

Copy `.env.example` to `.env` (or create `.env`) and set the path to your Ipopt installation:

```ini
# Ipopt installation directory (Windows / Custom source builds)
IPOPT_DIR="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md"
IPOPT_BIN_DIR="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md/bin"
PKG_CONFIG_PATH="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md/lib/pkgconfig"
```

> **Note for Windows:** `trajopt` automatically reads `.env` on import and registers the DLL directory with `os.add_dll_directory()`, so you do not need manual DLL loading code in your scripts.

---

### 3. Sync & Install with `uv`

Run `uv sync` to create the virtual environment and install dependencies:

```powershell
# On Windows (ensures Cython build tools are available before compiling cyipopt):
uv venv
uv pip install Cython setuptools wheel
uv sync --all-extras
```

```bash
# On Linux / macOS:
uv sync --all-extras
```

#### Dependency Groups & Extras

Optional extras in `pyproject.toml` (`[project.optional-dependencies]`, installed with `--all-extras` or `--extra <name>`):

- `solvers`: `cyipopt`, `osqp`, `clarabel`.
- `test`: `juliacall` (for cross-verification against the Julia reference).
- `dev`: `ruff`, `ty`, `pytest`, `pytest-benchmark`, `pytest-xdist`, `marimo`, `validate-pyproject`.

Default development tools (`[dependency-groups] dev`, installed by default with `uv sync`):

- `casadi` (independent baseline for cross-verification), `pre-commit`.

---

## Quick Start (Python)

### Trajectory Optimization with Ipopt

The following example solves a swing-up problem for a non-linear pendulum with torque bounds and a terminal goal state:

```python
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem

# 1. Define model and problem dimensions
model = Pendulum()
n, m, N = model.n, model.m, 21
dt = 0.05

# 2. Define objective (LQR tracking towards upright equilibrium [pi, 0])
xf = jnp.array([np.pi, 0.0])
Q = jnp.diag(jnp.array([10.0, 1.0]))
R = jnp.diag(jnp.array([0.1]))
Qf = jnp.diag(jnp.array([100.0, 10.0]))
obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

# 3. Add constraints (torque limits and terminal goal)
constraints = ConstraintList(n=n, m=m, N=N)
constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
constraints.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)

# 4. Build problem and initialize MPC state
prob = Problem(model=model, obj=obj, constraints=constraints, N=N, integrator=RK4())
x0 = jnp.array([0.0, 0.0])
state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf, dt=dt)

# 5. Solve the trajectory optimization problem with Ipopt
opt_state = prob.solve(state)
X = opt_state.states
U = opt_state.controls

print(f"Optimal final state: {np.asarray(X[-1])}")
```

---

## Testing & Quality Checks

Run the targeted unit tests with `pytest`:

```bash
# Run unit tests
uv run pytest test/unit -x

# Run all tests excluding Julia cross-verification (if Julia runtime is not installed)
uv run pytest -m "not julia"

# Run full test suite including Julia cross-verification
uv run pytest
```

Run type checking, formatting, and pre-commit checks:

```bash
# Type checking (via ty)
uv run ty check

# Code formatting and linting
uv run ruff check --fix

# Pre-commit gate
uv run pre-commit run --all-files
```

---

## Roadmap

The core v1 architecture establishes the expansion engine, dynamics integration, conic constraint catalog, and external solver transcriptions (Ipopt, OSQP, Clarabel). Future milestones include:

- **Native iLQR & DDP:** Iterative Linear Quadratic Regulator and Differential Dynamic Programming algorithms consuming the shared expansion engine.
- **Native ALTRO:** Augmented Lagrangian Trajectory Optimizer for fast constrained trajectory optimization.
- **Sequential Quadratic Programming (SQP):** Direct SQP solver with line-search filter and active-set strategy.
- **Closed-Loop Simulation Harness:** Environment wrappers for receding-horizon MPC simulation and hardware-in-the-loop testing.

---

## Architecture & Documentation

For a detailed mathematical and technical specification covering discrete optimal control representations, expansion engines, rotation manifolds on $\mathrm{SO}(3)$, and NLP transcriptions, refer to:

- [Trajectory Optimization Technical Specification](docs/TRAJECTORY_OPTIMIZATION_SPEC.md)
- [TrajectoryOptimization.jl Reference Documentation](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/stable)
