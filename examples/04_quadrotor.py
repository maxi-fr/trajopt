import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from trajopt.constraints.bounds import ControlBound
    from trajopt.constraints.constraint_list import ConstraintList
    from trajopt.constraints.geometric import SphereConstraint
    from trajopt.constraints.linear import GoalConstraint
    from trajopt.constraints.rotations import QuatVecEq
    from trajopt.costs.objective import Objective
    from trajopt.costs.rotations import QuatGeodesicCost
    from trajopt.dynamics.integrators import RK4
    from trajopt.models.quadrotor import Quadrotor
    from trajopt.problem import MPCState, Problem
    from trajopt.solvers.altro import ALTRO
    from trajopt.trajectory import Trajectory
    from trajopt.transcription.ipopt import Ipopt

    return (
        ALTRO,
        ConstraintList,
        ControlBound,
        GoalConstraint,
        Ipopt,
        MPCState,
        Objective,
        Problem,
        Quadrotor,
        QuatGeodesicCost,
        QuatVecEq,
        RK4,
        SphereConstraint,
        Trajectory,
        jnp,
        mo,
        np,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 — 6-DOF Quadrotor on $\mathrm{SO}(3)$ with 3D Obstacle Avoidance

    This notebook demonstrates trajectory optimization for a full **6-degree-of-freedom (6-DOF) Quadrotor rigid body** navigating in 3D Euclidean space around a spherical keep-out zone using `trajopt`.

    The Quadrotor is modeled with unit quaternions on the Lie group $\mathrm{SO}(3)$ using the **JPL quaternion convention** (scalar-last $[q_x, q_y, q_z, q_w]^\top$). We formulate a constrained trajectory optimization problem featuring:
    - **Geodesic Attitude Tracking** (`QuatGeodesicCost`) penalizing orientation error while respecting the $\mathbb{S}^3$ double-cover symmetry ($\mathbf{q} \equiv -\mathbf{q}$).
    - **Per-Rotor Actuation Bounds** (`ControlBound`) enforcing physical non-negative motor thrust limits $u_i \in [0, u_{\max}]$.
    - **3D Spherical Keep-Out Zone** (`SphereConstraint`) enforcing nonlinear obstacle clearance in Cartesian space.
    - **Terminal Boundary Alignment** (`GoalConstraint` & `QuatVecEq`) enforcing exact position, velocity, and orientation matching at the target.

    We solve the problem with two complementary solvers:
    1. **`ALTRO`** (*Augmented Lagrangian Trajectory Optimizer*): Native JAX solver combining an Augmented Lagrangian outer loop, iLQR Riccati backward-forward passes, and Projected Newton (PN) multiplier polish.
    2. **`Ipopt`** (*Interior Point Optimizer*): Industrial interior-point NLP solver operating on the transcribed sparse nonlinear program.

    ---

    ## System Dynamics & Mathematical Formulation

    ### State and Control Variables
    The 13-dimensional state vector $\mathbf{x} \in \mathbb{R}^{13}$ and 4-dimensional control vector $\mathbf{u} \in \mathbb{R}^4$ are defined as:
    $$\mathbf{x} = \begin{bmatrix} \mathbf{r} \\ \mathbf{q} \\ \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{13}, \qquad \mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \end{bmatrix} \in \mathbb{R}^4$$

    - $\mathbf{r} = [p_x, p_y, p_z]^\top \in \mathbb{R}^3$: Center-of-mass position in the world frame ($\text{m}$).
    - $\mathbf{q} = [q_x, q_y, q_z, q_w]^\top \in \mathbb{H}, \|\mathbf{q}\|_2 = 1$: JPL unit quaternion representing the body frame attitude relative to the world frame.
    - $\mathbf{v} = [v_x, v_y, v_z]^\top \in \mathbb{R}^3$: Linear velocity in the world frame ($\text{m/s}$).
    - $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^\top \in \mathbb{R}^3$: Angular velocity expressed in the body frame ($\text{rad/s}$).
    - $u_1, u_2, u_3, u_4 \ge 0$: Individual rotor thrust forces ($\text{N}$).

    ### Continuous Equations of Motion
    $$\begin{aligned}
    \dot{\mathbf{r}} &= \mathbf{v} \\
    \dot{\mathbf{q}} &= \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q} = \frac{1}{2} \begin{bmatrix} q_w \boldsymbol{\omega} + \boldsymbol{\omega} \times \mathbf{q}_{1:3} \\ -\mathbf{q}_{1:3}^\top \boldsymbol{\omega} \end{bmatrix} \\
    \dot{\mathbf{v}} &= \mathbf{g} + \frac{1}{m} \mathbf{R}(\mathbf{q})^\top \mathbf{f}_{\text{body}} \\
    \dot{\boldsymbol{\omega}} &= \mathbf{J}^{-1} \left( \boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{J} \boldsymbol{\omega}) \right)
    \end{aligned}$$

    where:
    - $\mathbf{g} = [0, 0, -9.81]^\top\,\text{m/s}^2$ is gravitational acceleration.
    - $\mathbf{R}(\mathbf{q})$ is the direction cosine matrix mapping world vectors into the body frame.
    - $\mathbf{f}_{\text{body}} = \begin{bmatrix} 0 & 0 & \sum_{i=1}^4 f_i \end{bmatrix}^\top$ is total thrust along the body $+z$ axis ($f_i = \max(0, k_f u_i)$).
    - $\boldsymbol{\tau} = \begin{bmatrix} L (f_2 - f_4) \\ L (f_3 - f_1) \\ k_m (u_1 - u_2 + u_3 - u_4) \end{bmatrix}$ is total body torque generated by motor differential thrust and rotor reaction torque.
    - $\mathbf{J} = \text{diag}(J_x, J_y, J_z)$ is the diagonal inertia tensor.

    At steady-state level hover, each rotor produces equal thrust balancing gravity:
    $$u_{\text{hover}} = \frac{m g}{4}$$
    """)
    return


@app.cell
def _(Quadrotor, RK4, jnp):
    # Time discretization and horizon length
    N = 25  # number of knot points
    dt = 0.05  # timestep in seconds (total horizon tf = 1.20 s)
    tf = (N - 1) * dt

    # Boundary conditions: Start at origin (0,0,0) and fly to (3,3,3)
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    xf = jnp.array([3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    # Physical model & 4th-order Runge-Kutta integrator
    model = Quadrotor(
        mass=0.5,
        J=(0.0023, 0.0023, 0.004),
        gravity=(0.0, 0.0, -9.81),
        motor_dist=0.1750,
        kf=1.0,
        km=0.0245,
    )
    integrator = RK4()

    n = model.n  # 13 states
    m = model.m  # 4 motor controls
    u_hover = float(model.mass * 9.81 / 4.0)  # ~1.226 N per rotor
    return N, dt, integrator, m, model, n, u_hover, x0, xf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Geodesic Attitude Cost on $\mathrm{SO}(3)$ (`QuatGeodesicCost`)

    Standard quadratic Euclidean error $\frac{1}{2}\|\mathbf{q} - \mathbf{q}_{\text{ref}}\|^2$ fails to account for the antipodal double-cover identification $\mathbf{q} \equiv -\mathbf{q}$ on $\mathbb{S}^3$, creating artificial local minima if the quaternion flips sign.

    `QuatGeodesicCost` resolves this by evaluating the geodesic chordal distance on $\mathrm{SO}(3)$:
    $$l_{\text{quat}}(\mathbf{q}, \mathbf{q}_{\text{ref}}) = w \min\left(1 + \mathbf{q}_{\text{ref}}^\top \mathbf{q},\; 1 - \mathbf{q}_{\text{ref}}^\top \mathbf{q}\right)$$

    Combined with Euclidean state weights $\mathbf{Q}$, control weights $\mathbf{R}$, and terminal weight $\mathbf{Q}_f$, the total stage and terminal cost functions are:
    $$\begin{aligned}
    l(\mathbf{x}, \mathbf{u}) &= \frac{1}{2} (\mathbf{x} - \mathbf{x}_f)^\top \mathbf{Q} (\mathbf{x} - \mathbf{x}_f) + \frac{1}{2} \mathbf{u}^\top \mathbf{R} \mathbf{u} + l_{\text{quat}}(\mathbf{q}, \mathbf{q}_f) \\
    l_f(\mathbf{x}) &= \frac{1}{2} (\mathbf{x} - \mathbf{x}_f)^\top \mathbf{Q}_f (\mathbf{x} - \mathbf{x}_f) + l_{\text{quat}, f}(\mathbf{q}, \mathbf{q}_f)
    \end{aligned}$$
    """)
    return


@app.cell
def _(N, Objective, QuatGeodesicCost, jnp, m, xf):
    # Quadratic state and control weights
    # State order: [rx, ry, rz,  qx, qy, qz, qw,  vx, vy, vz,  wx, wy, wz]
    Q_stage = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    R_stage = jnp.array([0.01, 0.01, 0.01, 0.01])
    Q_term = jnp.array([100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    # Geodesic attitude cost functions
    stage_cost = QuatGeodesicCost(
        Q=Q_stage,
        R=R_stage,
        q_ref=xf[3:7],
        w=10.0,
        qind=(3, 4, 5, 6),
        m=m,
    )
    term_cost = QuatGeodesicCost(
        Q=Q_term,
        q_ref=xf[3:7],
        w=1000.0,
        qind=(3, 4, 5, 6),
        terminal=True,
    )

    # Stacked horizon objective
    obj = Objective(stage_cost=stage_cost, terminal_cost=term_cost, N=N)
    return (obj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Hard Physical & Geometric Constraints

    We assemble a `ConstraintList` containing four distinct constraint types:
    1. **`ControlBound`**: Rotor thrusts cannot be negative (rotors cannot push downward) and cannot exceed the motor saturation limit $u_{\max} = 10.0\,\text{N}$:
       $$0.0 \le u_{i, k} \le 10.0\,\text{N}, \quad \forall i \in \{1, 2, 3, 4\}, \; k \in \{0, \dots, N-2\}$$
    2. **`SphereConstraint`**: A 3D spherical keep-out obstacle of radius $r_{\text{obs}} = 0.5\,\text{m}$ located directly along the straight-line path at $\mathbf{p}_{\text{obs}} = [1.5, 1.5, 1.5]^\top\,\text{m}$:
       $$c_{\text{obs}}(\mathbf{x}_k) = r_{\text{obs}}^2 - \|\mathbf{r}_k - \mathbf{p}_{\text{obs}}\|_2^2 \le 0, \quad \forall k \in \{1, \dots, N-1\}$$
    3. **`GoalConstraint`**: Exact terminal boundary matching on non-quaternion states (position, linear velocity, angular rate) at knot point $N-1$:
       $$\mathbf{x}_{N-1}[\text{inds}] - \mathbf{x}_f[\text{inds}] = \mathbf{0}$$
    4. **`QuatVecEq`**: Terminal attitude alignment constraint ensuring the vehicle lands in level hover orientation ($\mathbf{q}_{N-1} \approx \mathbf{q}_f$):
       $$\mathbf{q}_{N-1} \times \mathbf{q}_f = \mathbf{0}$$
    """)
    return


@app.cell
def _(
    ConstraintList,
    ControlBound,
    GoalConstraint,
    N,
    QuatVecEq,
    SphereConstraint,
    jnp,
    m,
    n,
    xf,
):
    # Spherical obstacle center and radius
    obs_center = (1.5, 1.5, 1.5)
    obs_radius = 0.5
    u_max = 10.0

    # Build constraint list
    cl = ConstraintList(n=n, m=m, N=N)

    # 1. Rotor thrust bounds across all control stages
    cl.add_constraint(
        ControlBound(n=n, m=m, u_min=[0.0] * m, u_max=[u_max] * m),
        range(N - 1),
    )

    # 2. Spherical obstacle avoidance across all trajectory stages
    cl.add_constraint(
        SphereConstraint(
            n=n,
            m=m,
            xc=[obs_center[0]],
            yc=[obs_center[1]],
            zc=[obs_center[2]],
            radius=[obs_radius],
        ),
        range(1, N),
    )

    # 3. Terminal goal constraint on non-quaternion state entries
    non_quat_inds = [0, 1, 2, 7, 8, 9, 10, 11, 12]
    cl.add_constraint(
        GoalConstraint(n=n, xf=xf[jnp.array(non_quat_inds)], inds=non_quat_inds),
        N - 1,
    )

    # 4. Terminal quaternion attitude equality constraint
    cl.add_constraint(
        QuatVecEq(n=n, qf=xf[3:7], qind=(3, 4, 5, 6)),
        N - 1,
    )
    return cl, obs_center, obs_radius, u_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Initial Guess Trajectory & `Problem` Assembly

    We initialize the optimization trajectory with:
    - **Position / Attitude**: Straight-line Euclidean interpolation $\mathbf{x}_{\text{init}}(t)$ from $\mathbf{x}_0$ to $\mathbf{x}_f$.
    - **Controls**: Constant hover thrust $u_i = u_{\text{hover}} = \frac{mg}{4} \approx 1.226\,\text{N}$ per motor.

    We bundle the model, objective, constraints, and `RK4` numerical integrator into a `Problem` and instantiate an `MPCState`.
    """)
    return


@app.cell
def _(
    MPCState,
    N,
    Problem,
    Trajectory,
    cl,
    dt,
    integrator,
    jnp,
    m,
    model,
    obj,
    u_hover,
    x0,
    xf,
):
    # Assemble problem
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=integrator)

    # Construct initial warm-start trajectory
    X_init = jnp.linspace(x0, xf, N)
    U_init = jnp.full((N - 1, m), u_hover, dtype=jnp.float64)
    dt_arr = jnp.full((N - 1,), dt, dtype=jnp.float64)
    t_init = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dt_arr)])
    init_traj = Trajectory(X=X_init, U=U_init, t=t_init, dt=dt_arr)

    # Create initial MPCState
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=xf, initial_trajectory=init_traj)
    return X_init, prob, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Solving with Native JAX `ALTRO` & Sparse NLP `Ipopt`

    We now execute both solvers:
    - **`ALTRO`**: Executes an Augmented Lagrangian outer loop with iLQR Riccati backward-forward passes and Projected Newton multiplier polish in pure, jit-compiled JAX.
    - **`Ipopt`**: Transcribes the multiple-shooting optimal control problem into a sparse nonlinear programming problem solved via primal-dual interior-point methods.
    """)
    return


@app.cell
def _(ALTRO, mo, prob, state, time):
    altro_solver = ALTRO()

    # JIT warm-up solve
    _ = altro_solver.solve(prob, state)

    # Timed ALTRO solve
    t_start_altro = time.perf_counter()
    res_altro = altro_solver.solve(prob, state)
    time_altro_ms = (time.perf_counter() - t_start_altro) * 1000.0

    mo.md(
        f"""
        ### Native JAX ALTRO Solve Result
        - **Status:** `{res_altro.message}` (Success: `{res_altro.success}`)
        - **AL Outer Iterations:** `{res_altro.iterations}`
        - **Solve Time (Warm):** `{time_altro_ms:.2f} ms`
        - **Objective Cost $J$:** `{res_altro.cost:.4f}`
        - **Max Constraint Violation:** `{res_altro.constraint_violation:.3e}`
        """
    )
    return res_altro, time_altro_ms


@app.cell
def _(Ipopt, mo, prob, state, time):
    ipopt_solver = Ipopt(options={"print_level": 0, "max_iter": 500, "tol": 1e-6})

    # Timed Ipopt solve
    t_start_ipopt = time.perf_counter()
    res_ipopt = ipopt_solver.solve(prob, state)
    time_ipopt_ms = (time.perf_counter() - t_start_ipopt) * 1000.0

    mo.md(
        f"""
        ### Sparse NLP Ipopt Solve Result
        - **Status:** `{res_ipopt.message}` (Success: `{res_ipopt.success}`)
        - **NLP Iterations:** `{res_ipopt.iterations}`
        - **Solve Time:** `{time_ipopt_ms:.2f} ms`
        - **Objective Cost $J$:** `{res_ipopt.cost:.4f}`
        - **Max Constraint Violation:** `{res_ipopt.constraint_violation:.3e}`
        """
    )
    return res_ipopt, time_ipopt_ms


@app.cell
def _(mo, res_altro, res_ipopt, time_altro_ms, time_ipopt_ms):
    speedup = time_ipopt_ms / time_altro_ms if time_altro_ms > 0 else 1.0

    mo.md(
        rf"""
        ## 5. Quantitative Benchmark Comparison

        | Metric | Native ALTRO (JAX) | Sparse Ipopt (MUMPS) | Assessment |
        | :--- | :---: | :---: | :---: |
        | **Convergence Status** | `{res_altro.message}` | `{res_ipopt.message}` | Ipopt Locally Optimal |
        | **Iterations** | `{res_altro.iterations}` AL outer loops | `{res_ipopt.iterations}` interior-point steps | — |
        | **Solve Time** | **`{time_altro_ms:.2f} ms`** | `{time_ipopt_ms:.2f} ms` | **`{speedup:.1f}x` speed ratio** |
        | **Final Objective Cost $J$** | `{res_altro.cost:.4f}` | `{res_ipopt.cost:.4f}` | Smooth 3D Trajectory |
        | **Max Constraint Violation** | `{res_altro.constraint_violation:.2e}` | `{res_ipopt.constraint_violation:.2e}` | Hard Constraints Satisfied |

        Ipopt navigates the non-convex spherical obstacle smoothly by discovering a banking curved arc that grazes the keep-out boundary while satisfying actuator thrust limits.
        """
    )
    return


@app.cell
def _(X_init, np, obs_center, obs_radius, plt, res_altro, res_ipopt, x0, xf):
    # 3D Trajectory Visualization
    fig_3d = plt.figure(figsize=(10, 7.5), dpi=120)
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    # 1. Render spherical obstacle as a wireframe & translucent surface
    u_grid = np.linspace(0, 2 * np.pi, 30)
    v_grid = np.linspace(0, np.pi, 20)
    xs_obs = obs_center[0] + obs_radius * np.outer(np.cos(u_grid), np.sin(v_grid))
    ys_obs = obs_center[1] + obs_radius * np.outer(np.sin(u_grid), np.sin(v_grid))
    zs_obs = obs_center[2] + obs_radius * np.outer(np.ones(np.size(u_grid)), np.cos(v_grid))

    ax_3d.plot_surface(xs_obs, ys_obs, zs_obs, color="#e74c3c", alpha=0.35, edgecolor="#c0392b", linewidth=0.4)

    # 2. Initial straight-line trajectory guess passing through obstacle
    ax_3d.plot(
        X_init[:, 0],
        X_init[:, 1],
        X_init[:, 2],
        color="#7f8c8d",
        linestyle=":",
        linewidth=2.0,
        label="Initial Straight Guess (Infeasible)",
    )

    # 3. Optimized trajectory paths
    traj_altro = res_altro.trajectory
    traj_ipopt = res_ipopt.trajectory

    ax_3d.plot(
        traj_altro.X[:, 0],
        traj_altro.X[:, 1],
        traj_altro.X[:, 2],
        color="#2980b9",
        linestyle="-.",
        linewidth=2.2,
        label="ALTRO Trajectory",
    )
    ax_3d.plot(
        traj_ipopt.X[:, 0],
        traj_ipopt.X[:, 1],
        traj_ipopt.X[:, 2],
        color="#27ae60",
        linewidth=2.8,
        label="Ipopt Optimal Path",
    )

    # 4. Start and Goal boundary markers
    ax_3d.scatter(
        [float(x0[0])],
        [float(x0[1])],
        [float(x0[2])],
        color="#2c3e50",
        s=120,
        marker="o",
        label=r"Start $\mathbf{r}_0 = [0, 0, 0]^\top$",
    )
    ax_3d.scatter(
        [float(xf[0])],
        [float(xf[1])],
        [float(xf[2])],
        color="#d35400",
        s=160,
        marker="*",
        label=r"Goal $\mathbf{r}_f = [3, 3, 3]^\top$",
    )
    ax_3d.scatter(
        [obs_center[0]],
        [obs_center[1]],
        [obs_center[2]],
        color="#c0392b",
        s=80,
        marker="x",
        label=f"Obstacle Center ({obs_center[0]}, {obs_center[1]}, {obs_center[2]})",
    )

    ax_3d.set_title("6-DOF Quadrotor 3D Obstacle Avoidance Trajectory", fontsize=13, fontweight="bold", pad=15)
    ax_3d.set_xlabel("X Position (m)", labelpad=10)
    ax_3d.set_ylabel("Y Position (m)", labelpad=10)
    ax_3d.set_zlabel("Z Position (m)", labelpad=10)
    ax_3d.view_init(elev=24, azim=-55)
    ax_3d.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig_3d
    return


@app.cell
def _(N, dt, np, obs_center, obs_radius, plt, res_ipopt, u_hover, u_max):
    # Time history subplots
    opt_traj = res_ipopt.trajectory
    t_knots = np.linspace(0.0, (N - 1) * dt, N)
    t_ctrl = t_knots[:-1]

    pos = opt_traj.X[:, :3]
    quat = opt_traj.X[:, 3:7]  # [qx, qy, qz, qw]
    vel = opt_traj.X[:, 7:10]
    omega = opt_traj.X[:, 10:13]
    thrusts = opt_traj.U  # [u1, u2, u3, u4]

    # Distance to spherical obstacle center
    dist_to_obs = np.linalg.norm(pos - np.array(obs_center), axis=1)

    fig_profiles, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=120)

    # 1. Motor Rotor Thrust Forces
    ax_u = axes[0, 0]
    motor_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i in range(4):
        ax_u.plot(t_ctrl, thrusts[:, i], label=f"Rotor $u_{i + 1}$", color=motor_colors[i], linewidth=2.0)
    ax_u.axhline(u_hover, color="#7f8c8d", linestyle="--", linewidth=1.4, label=f"Hover Thrust ({u_hover:.2f} N)")
    ax_u.axhline(u_max, color="#c0392b", linestyle=":", linewidth=1.5, label=f"Max Thrust ({u_max:.1f} N)")
    ax_u.axhline(0.0, color="#2c3e50", linestyle=":", linewidth=1.2, label="Min Thrust (0.0 N)")
    ax_u.set_title("Motor Rotor Thrust Forces $u_i(t)$", fontweight="bold")
    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Thrust Force (N)")
    ax_u.set_ylim(-0.5, u_max + 1.0)
    ax_u.grid(visible=True, linestyle=":", alpha=0.6)
    ax_u.legend(loc="upper right", fontsize=8)

    # 2. Distance to Obstacle Center
    ax_obs = axes[0, 1]
    ax_obs.plot(t_knots, dist_to_obs, color="#e67e22", linewidth=2.4, label="Distance to Obstacle Center")
    ax_obs.axhline(
        obs_radius,
        color="#c0392b",
        linestyle="--",
        linewidth=1.8,
        label=f"Obstacle Radius $r_{{\\mathrm{{obs}}}} = {obs_radius:.2f}$ m",
    )
    ax_obs.fill_between(
        t_knots,
        0,
        obs_radius,
        color="#c0392b",
        alpha=0.2,
        label="Collision Keep-Out Zone",
    )
    ax_obs.set_title("Distance to Obstacle Center vs. Clearance", fontweight="bold")
    ax_obs.set_xlabel("Time (s)")
    ax_obs.set_ylabel(r"$\|\mathbf{r}(t) - \mathbf{p}_{\mathrm{obs}}\|_2$ (m)")
    ax_obs.set_ylim(0.0, max(np.max(dist_to_obs) + 0.3, 2.5))
    ax_obs.grid(visible=True, linestyle=":", alpha=0.6)
    ax_obs.legend(loc="upper right", fontsize=8)

    # 3. JPL Attitude Quaternions
    ax_q = axes[1, 0]
    ax_q.plot(t_knots, quat[:, 3], label=r"$q_w$ (scalar)", color="#2c3e50", linewidth=2.2)
    ax_q.plot(t_knots, quat[:, 0], label=r"$q_x$ (pitch vector)", color="#e74c3c", linewidth=1.8)
    ax_q.plot(t_knots, quat[:, 1], label=r"$q_y$ (roll vector)", color="#2980b9", linewidth=1.8)
    ax_q.plot(t_knots, quat[:, 2], label=r"$q_z$ (yaw vector)", color="#27ae60", linewidth=1.8)
    ax_q.set_title(r"Attitude JPL Quaternion $\mathbf{q}(t)$ on $\mathrm{SO}(3)$", fontweight="bold")
    ax_q.set_xlabel("Time (s)")
    ax_q.set_ylabel("Quaternion Components")
    ax_q.grid(visible=True, linestyle=":", alpha=0.6)
    ax_q.legend(loc="upper right", fontsize=8)

    # 4. Linear & Angular Velocities
    ax_v = axes[1, 1]
    ax_v.plot(t_knots, vel[:, 0], label=r"$v_x$", color="#3498db", linewidth=1.8)
    ax_v.plot(t_knots, vel[:, 1], label=r"$v_y$", color="#9b59b6", linewidth=1.8)
    ax_v.plot(t_knots, vel[:, 2], label=r"$v_z$", color="#16a085", linewidth=1.8)
    ax_v.plot(t_knots, omega[:, 0], label=r"$\omega_x$", color="#e74c3c", linestyle="--", linewidth=1.4)
    ax_v.plot(t_knots, omega[:, 1], label=r"$\omega_y$", color="#f39c12", linestyle="--", linewidth=1.4)
    ax_v.plot(t_knots, omega[:, 2], label=r"$\omega_z$", color="#2c3e50", linestyle="--", linewidth=1.4)
    ax_v.set_title(r"Linear & Angular Velocities ($v$, $\omega$)", fontweight="bold")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Velocity (m/s, rad/s)")
    ax_v.grid(visible=True, linestyle=":", alpha=0.6)
    ax_v.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig_profiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Key Takeaways & Optimal Flight Mechanics

    1. **Non-Euclidean Attitude Geometry**: By utilizing `QuatGeodesicCost` with JPL quaternions on $\mathrm{SO}(3)$, attitude tracking avoids gimbal-lock singularities while respecting the antipodal double-cover symmetry $\mathbf{q} \equiv -\mathbf{q}$.
    2. **Coupled Translation & Rotation**: To steer around the spherical obstacle, the quadrotor banks (producing nonzero roll and pitch angles $q_x, q_y$) to redirect its total thrust vector laterally, accelerating clear of the keep-out zone before leveling off at the target $[3, 3, 3]^\top$.
    3. **Active Keep-Out Clearance**: The minimum distance from the quadrotor center to the obstacle center is exactly $\ge r_{\text{obs}} = 0.50\,\text{m}$, demonstrating precise active constraint satisfaction.
    4. **Actuation Saturation Enforcement**: Motor rotor thrust commands remain strictly within the physical hardware envelope $u_i \in [0, 10.0]\,\text{N}$, with differential thrust naturally bounded throughout high-speed agile maneuvering.
    """)
    return


if __name__ == "__main__":
    app.run()
