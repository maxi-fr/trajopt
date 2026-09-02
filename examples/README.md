# Examples Gallery

This directory contains introductory and benchmark examples for `trajopt` implemented as [marimo](https://marimo.io) notebooks.

Each notebook is a self-contained, reproducible optimal control problem written using `trajopt` primitives (`AbstractModel`, `LQRObjective` / `TrackingObjective`, `ConstraintList`, `Problem`, `MPCState`, and solvers).

---

## Running the Notebooks

### Interactive Mode (Marimo Web App)

To open and interactively edit or explore a notebook in your browser:

```bash
# Inverted Pendulum swing-up
uv run marimo edit examples/01_pendulum.py

# Underactuated Cartpole swing-up with bounds
uv run marimo edit examples/02_cartpole.py

# Nonholonomic Dubins Car corridor tracking
uv run marimo edit examples/03_dubins_car.py

# 6-DOF Quadrotor attitude on SO(3) with obstacle avoidance
uv run marimo edit examples/04_quadrotor.py

# Receding-Horizon Closed-Loop MPC with Disturbance Rejection
uv run marimo edit examples/05_closed_loop_mpc.py
```

To run a notebook as a read-only interactive dashboard:

```bash
uv run marimo run examples/01_pendulum.py
```

### Standalone Script Mode

Every marimo notebook is also a standard Python script and can be executed directly from the command line:

```bash
uv run python examples/01_pendulum.py
```

---

## Notebook Catalog

| Notebook | Physical System | Key Concepts Demonstrated | Solvers Compared |
| :--- | :--- | :--- | :--- |
| [`01_pendulum.py`](01_pendulum.py) | Inverted Pendulum | Swing-up dynamics, `LQRObjective`, `ControlBound` torque limits, terminal `GoalConstraint`, phase portraits | `ALTRO`, `Ipopt`, `ILQR` |
| [`02_cartpole.py`](02_cartpole.py) | Cartpole | Underactuated dynamics, cart track limits (`StateBound`), actuation force bounds (`ControlBound`), trajectory keyframes | `ALTRO`, `Ipopt` |
| [`03_dubins_car.py`](03_dubins_car.py) | Dubins Car | Nonholonomic kinematics, `TrackingObjective`, lateral corridor constraints (`StateBound`), linear/angular velocity bounds | `ALTRO`, `Ipopt` |
| [`04_quadrotor.py`](04_quadrotor.py) | Quadrotor | 6-DOF rigid body, JPL quaternions on $\mathrm{SO}(3)$, `QuatGeodesicCost`, `SphereConstraint` 3D obstacle avoidance | `ALTRO`, `Ipopt` |
| [`05_closed_loop_mpc.py`](05_closed_loop_mpc.py) | Cartpole (MPC) | Receding-horizon feedback, `MPCState.with_measurement()`, `MPCState.shift()`, impulse disturbance rejection, predicted trajectory fans | `ALTRO`, `Ipopt` |

---

## Solvers

- **`ALTRO` (`trajopt.solvers.altro.ALTRO`)**: Fast native JAX solver combining an Augmented Lagrangian outer loop, iLQR inner iterations, and Projected Newton (PN) multiplier polish.
- **`Ipopt` (`trajopt.transcription.ipopt.Ipopt`)**: Interior-point NLP solver backend consuming the sparse transcribed nonlinear program.
- **`OSQP` / `Clarabel` (`trajopt.transcription.osqp.OSQP`, `trajopt.transcription.clarabel.Clarabel`)**: QP and conic solvers solving convex subproblems about an Operating Point.
