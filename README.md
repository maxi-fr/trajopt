# TrajectoryOptimization (`trajopt`)

A high-performance framework for formulating, evaluating, and solving discrete-time optimal control and trajectory optimization problems in **Python** (powered by JAX) and **Julia** ([`TrajectoryOptimization.jl`](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl)).

---

## Features

- **Conic Constraint Formulation:** Unifies equality, inequality (orthant), and second-order cone (SOC) constraints via projection operations and explicit Jacobians.
- **Fast Automatic Differentiation:** Uses JAX in Python and ForwardDiff in Julia for zero-overhead first- and second-order expansions of dynamics, costs, and constraints.
- **Solver Integrations:** Direct NLP transcriptions and interfaces for **Ipopt** (`cyipopt`), **OSQP**, and **Clarabel**, alongside specialized Augmented Lagrangian trajectory solvers (ALTRO).
- **Cross-Verification:** Built-in test suite cross-verifying Python implementations against Julia reference implementations.

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

Copy `.env.example` to `.env` and set the path to your Ipopt installation:

```bash
cp .env.example .env
```

Edit `.env` for your system (e.g. on Windows):

```ini
# Ipopt installation directory (Windows / Custom source builds)
IPOPT_DIR="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md"
IPOPT_BIN_DIR="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md/bin"
PKG_CONFIG_PATH="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md/lib/pkgconfig"
```

> **Note for Windows:** `trajopt` automatically reads `.env` on import and registers the DLL directory with `os.add_dll_directory()`, so you do not need manual DLL loading code in your scripts.

---

### 3. Sync & Install with `uv`

Run `uv sync` to create the virtual environment and build `cyipopt`:

```powershell
# On Windows:
uv pip install Cython setuptools wheel
uv sync --all-extras
```

```bash
# On Linux / macOS:
uv sync --all-extras
```

Available dependency groups in `pyproject.toml`:

- `solvers`: Includes `cyipopt`, `osqp`, and `clarabel`.
- `dev`: Includes development tools (`ruff`, `ty`, `pytest`, `pytest-benchmark`, `pytest-xdist`, `marimo`, `casadi`, `pre-commit`).
- `test`: Includes `juliacall` for cross-verification.

---

## Installation (Julia)

To use the Julia implementation:

```julia
using Pkg
Pkg.add("TrajectoryOptimization")
```

Or for local development:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## Quick Start (Python)

### Trajectory Optimization with Ipopt

```python
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem, controls, solve, states

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
opt_state = solve(prob, state)
X = states(opt_state)
U = controls(opt_state)

print(f"Optimal final state: {np.asarray(X[-1])}")
```

---

## Testing & Quality Checks

Run the test suite with `pytest`:

```bash
# Run unit tests
uv run pytest test/unit

# Run all tests (deselecting Julia cross-verification if Julia runtime is absent)
uv run pytest -m "not julia"

# Run full test suite including Julia cross-verification
uv run pytest
```

Run type checking and linting:

```bash
# Type checking
uv run ty check

# Code formatting and linting
uv run ruff check .
```

---

## Architecture & Documentation

For a detailed technical overview of discrete optimal control representations, expansion engines, rotation groups on $\mathrm{SO}(3)$, and NLP transcriptions, refer to:

- [Trajectory Optimization Technical Specification](docs/TRAJECTORY_OPTIMIZATION_SPEC.md)
- [TrajectoryOptimization.jl Documentation](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/stable)

---

## References

- Howell, T. A., Jackson, B. E., and Manchester, Z. (2019). *ALTRO: A Fast Solver for Constrained Trajectory Optimization*. IROS 2019. [[PDF]](https://rexlab.stanford.edu/papers/altro-iros.pdf)
- Wächter, A., and Biegler, L. T. (2006). *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*. Mathematical Programming.
