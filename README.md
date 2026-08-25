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

## Installation

### Using the package in your own project

`trajopt` is installed from git. To use the external solvers (Ipopt, OSQP, Clarabel), install the `solvers` extra:

```bash
# uv
uv add Cython setuptools wheel
uv add "trajopt[solvers]" --git https://github.com/maxi-fr/trajopt.git

# pip
pip install Cython setuptools wheel
pip install "trajopt[solvers] @ git+https://github.com/maxi-fr/trajopt.git"
```

- **OSQP and Clarabel** install as prebuilt wheels — they work immediately, with no system dependencies.
- **cyipopt (the Ipopt backend)** compiles from source against the underlying COIN-OR Ipopt solver, so ensure these system prerequisites are installed:

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

#### Runtime: finding Ipopt on Windows

`trajopt` registers the Ipopt DLL directory with `os.add_dll_directory()` at import on Windows, so no manual DLL loading code is needed in your scripts. Set one of these environment variables to your extracted Ipopt install:

```ini
IPOPT_DIR="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md"
# or, to point directly at the binaries:
IPOPT_BIN_DIR="C:/Ipopt/Ipopt-3.14.19-win64-msvs2022-md/bin"
```

On Linux and macOS no runtime variable is needed — Ipopt is found through the system library search.

`Problem.solve(state)` defaults to the Ipopt backend. Install the `solvers` extra above (a bare install raises a guided `ImportError`), or pass a backend explicitly — `OSQP()` from `trajopt.transcription.osqp`, `Clarabel()` from `trajopt.transcription.clarabel`, or the native `ALTRO()` from `trajopt.solvers.altro`, which needs no external installs at all.

---

### Developing this repository

Clone the repo and sync with [`uv`](https://docs.astral.sh/uv/) for fast, reproducible environment and dependency management:

```powershell
# On Windows (ensures Cython build tools are available before compiling cyipopt):
uv venv
uv add Cython setuptools wheel
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

### Native iLQR / ALTRO solvers

Alongside the Ipopt/OSQP/Clarabel transcription adapters, `trajopt.solvers` ports iLQR, an
augmented-Lagrangian (AL) outer loop, a control-limited DDP backward pass, Projected Newton (PN)
polish, and the combined ALTRO driver from
[`Altro.jl`](https://github.com/RoboticExplorationLab/Altro.jl) as native JAX solvers. Every
solver loop is a `lax.while_loop`/`lax.scan`, so a whole solve is one jittable, vmappable function
-- the intended reason to reach for one of these over the Ipopt adapter is speed on a
repeatedly-solved path (e.g. receding-horizon MPC via `jax.jit`/`jax.vmap` over the traced core),
not feature parity: the native solvers are not reverse-mode differentiable, and constraints go
through shooting (`ILQR`/`AL`/`ALTRO`) or PN's own multiple-shooting layout, not the general NLP
transcription `transcription/` builds for Ipopt. `ALTRO().solve()` is a *thin eager wrapper* around
the traced core, but does call `jax.jit` on it internally, caching the compiled closure per solver
instance and problem identity -- so a bare first call pays a one-off compile, and every repeated
`.solve()` call on the same instance and problem (e.g. an MPC loop) reuses that compilation; see
`docs/adr/0001-altro-port-divergences.md`'s benchmark section for measured numbers against Ipopt.

- **`ILQR`** -- unconstrained (or box-only, via the control-limited variant below) iterative LQR.
  Cheapest of the three; reach for it when the problem has no general constraints, or when a
  caller is doing its own AL-style penalty wrapping.
- **`AL`** -- augmented-Lagrangian outer loop wrapping `ILQR`'s inner solve, for problems with
  general (in)equality and conic constraints. `MPCState.al` carries the per-knot duals and
  penalties so they warm-start across MPC steps, which is most of the reason AL suits MPC over a
  cold-started NLP solve.
- **`ALTRO`** -- `AL` followed by a Projected Newton polish phase (`PN`) that drives the
  constraint violation the rest of the way to tolerance where AL alone plateaus. Takes an
  unconstrained-problem shortcut straight to `ILQR` when `problem` has no constraints at all.
- **Control-limited DDP** -- a Tassa box-QP backward pass (`trajopt.solvers.boxqp`), not part of
  Altro.jl, verified independently against Clarabel. Enforces control bounds exactly in the
  backward pass rather than through an AL penalty; pass its `solve_kd_builder` into `AL`/`ALTRO`
  to route `ControlBound` rows through it while other constraints still go through the AL loop.

Swapping backends is a one-word change, since every solver satisfies the same `Solver` protocol:

```python3
from trajopt.solvers.altro import ALTRO

opt_state = prob.solve(state, solver=ALTRO())  # was: prob.solve(state), the Ipopt default
```

See `docs/altro-jl-reference.md` (corrected by `docs/altro-port/00-overview.md`'s findings) for
the algorithm this port targets, and `docs/adr/0001-altro-port-divergences.md` for where the
Python port deliberately diverges from it.

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

The core v1 architecture establishes the expansion engine, dynamics integration, conic constraint catalog, external solver transcriptions (Ipopt, OSQP, Clarabel), and native `ILQR`/`AL`/`ALTRO` solvers (see [Native iLQR / ALTRO solvers](#native-ilqr--altro-solvers) above). Future milestones include:

- **Sequential Quadratic Programming (SQP):** Direct SQP solver with line-search filter and active-set strategy.
- **Closed-Loop Simulation Harness:** Environment wrappers for receding-horizon MPC simulation and hardware-in-the-loop testing.
- **Reverse-mode differentiable native solves:** a `custom_vjp` on the native solvers' fixed point, layered on top of the traced `lax.while_loop` cores without changing them.

---

## Architecture & Documentation

For a detailed mathematical and technical specification covering discrete optimal control representations, expansion engines, rotation manifolds on $\mathrm{SO}(3)$, and NLP transcriptions, refer to:

- [Trajectory Optimization Technical Specification](docs/TRAJECTORY_OPTIMIZATION_SPEC.md)
- [TrajectoryOptimization.jl Reference Documentation](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/stable)
