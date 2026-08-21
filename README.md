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
- `dev`: Includes `ruff`, `mypy`, `pytest`, and `pytest-benchmark`.
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

### Conic Projections & Constraints

```python
import numpy as np
from trajopt.cones import SecondOrderCone, NegativeOrthant

# Create a 3D Second-Order Cone (Lorentz cone: ||x[0:2]||_2 <= x[2])
soc = SecondOrderCone(dim=3)
v = np.array([3.0, 4.0, 2.0])

# Project point onto cone
proj = soc.project(v)
print("Projection:", proj)

# Compute projection Jacobian / derivative
jac = soc.jacobian(v)
print("Jacobian shape:", jac.shape)
```

### Solving with Ipopt

```python
import trajopt  # Automatically configures Ipopt DLL paths on Windows
import cyipopt
import numpy as np


# Minimize 2D Rosenbrock function
def objective(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def gradient(x):
    return np.array([-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2), 200.0 * (x[1] - x[0] ** 2)])


res = cyipopt.minimize_ipopt(objective, [0.0, 0.0], jac=gradient)
print("Converged:", res.success)
print("Optimal Solution:", res.x)
```

---

## Testing & Quality Checks

Run the test suite with `pytest`:

```bash
# Run unit tests
uv run pytest tests_py/unit

# Run all tests (including Julia cross-verification if Julia runtime is present)
uv run pytest
```

Run code formatting and type checking:

```bash
uv run ruff check .
uv run mypy python/trajopt
```

---

## Architecture & Documentation

For a detailed technical overview of discrete optimal control representations, expansion engines, rotation groups on $\mathrm{SO}(3)$, and NLP transcriptions, refer to:

- [Trajectory Optimization Technical Specification](TRAJECTORY_OPTIMIZATION_SPEC.md)
- [TrajectoryOptimization.jl Documentation](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/stable)

---

## References

- Howell, T. A., Jackson, B. E., and Manchester, Z. (2019). *ALTRO: A Fast Solver for Constrained Trajectory Optimization*. IROS 2019. [[PDF]](https://rexlab.stanford.edu/papers/altro-iros.pdf)
- Wächter, A., and Biegler, L. T. (2006). *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*. Mathematical Programming.
