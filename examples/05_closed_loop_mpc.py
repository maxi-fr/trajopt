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

    from trajopt.constraints.bounds import ControlBound, StateBound
    from trajopt.constraints.constraint_list import ConstraintList
    from trajopt.costs.objective import LQRObjective
    from trajopt.dynamics.integrators import RK4
    from trajopt.models.cartpole import Cartpole
    from trajopt.problem import MPCState, Problem
    from trajopt.solvers.altro import ALTRO
    from trajopt.transcription.ipopt import Ipopt

    return (
        ALTRO,
        Cartpole,
        ConstraintList,
        ControlBound,
        Ipopt,
        LQRObjective,
        MPCState,
        Problem,
        RK4,
        StateBound,
        jnp,
        mo,
        np,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 — Receding-Horizon Model Predictive Control (MPC) with ALTRO

    This notebook demonstrates **closed-loop Model Predictive Control (MPC)** on an underactuated **Cartpole** system using `trajopt`.

    We explore the fundamentals of receding-horizon optimal control:
    1. **Receding Horizon Control**: Formulating and solving a constrained finite-horizon optimal control problem at every sampling instant $t_k$.
    2. **State Feedback & Multiplier Warm-Starting**: Injecting real-time state feedback via `MPCState.with_measurement()` and warm-starting the primal-dual trajectory with `MPCState.shift()`.
    3. **Disturbance Rejection & Hard Constraint Enforcement**: Rejecting an unmodeled angular velocity impulse kick at $t = 1.0\,\text{s}$ while strictly respecting control saturation ($|u| \le 20.0\,\text{N}$) and cart track boundaries ($|p| \le 0.8\,\text{m}$).
    4. **Solver Comparison & Real-Time Performance**: Benchmarking native JAX **ALTRO** against **Ipopt** and evaluating sub-millisecond warm-started execution latencies.

    ---

    ## 1. Receding-Horizon MPC Mathematical Formulation

    At each discrete sampling instant $t_k = k \Delta t$, the controller measures the true plant state $x(t_k)$ and solves a finite-horizon trajectory optimization problem over $N$ knot points (prediction horizon $T = (N-1)\Delta t$):

    $$\begin{aligned}
    \min_{X, U} \quad & \frac{1}{2} (x_{N-1} - x_f)^\top Q_f (x_{N-1} - x_f) + \sum_{i=0}^{N-2} \left[ \frac{1}{2} (x_i - x_f)^\top Q (x_i - x_f) + \frac{1}{2} u_i^\top R u_i \right] \\
    \text{subject to} \quad & x_0 = x(t_k), \\
    & x_{i+1} = f_d(x_i, u_i, t_k + i\Delta t, \Delta t), \quad i \in \{0, \dots, N-2\}, \\
    & u_{\min} \le u_i \le u_{\max}, \quad i \in \{0, \dots, N-2\}, \\
    & x_{\min} \le x_i \le x_{\max}, \quad i \in \{0, \dots, N-1\}.
    \end{aligned}$$

    ### The Receding Horizon Principle
    Once the optimal trajectory $(X^*, U^*)$ is computed:
    1. **Apply Control**: Only the first control action $u_0^* = u^*(t_k)$ is applied to the physical system over the interval $[t_k, t_k + \Delta t)$.
    2. **Advance Time**: The system evolves to $x(t_{k+1}) = f_d(x(t_k), u_0^*)$.
    3. **Shift Horizon**: The warm-start trajectory is shifted forward by $\Delta t$:
       $$\tilde{X} = [x_1^*, x_2^*, \dots, x_{N-1}^*, x_{N-1}^*]^\top, \qquad \tilde{U} = [u_1^*, u_2^*, \dots, u_{N-2}^*, u_{N-2}^*]^\top$$
    4. **Repeat**: The measurement is updated $x_0 \leftarrow x(t_{k+1})$, and the optimization is repeated with the shifted warm start.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. System Dynamics: Cartpole

    The cartpole model consists of a cart of mass $m_c$ sliding on a horizontal track, with a pendulum pole of mass $m_p$ and length $l$ attached via an unactuated revolute joint.

    ### State and Control Variables
    - **State vector**: $x = [p, \theta, \dot{p}, \dot{\theta}]^\top \in \mathbb{R}^4$
      - $p$: Horizontal cart position ($\text{m}$).
      - $\theta$: Pole angle from downward vertical ($\text{rad}$, with $\theta = 0$ hanging down and $\theta = \pi$ upright).
      - $\dot{p}$: Cart linear velocity ($\text{m/s}$).
      - $\dot{\theta}$: Pole angular velocity ($\text{rad/s}$).
    - **Control vector**: $u = [F] \in \mathbb{R}^1$
      - $F$: Horizontal actuator force applied to the cart ($\text{N}$).

    ### Equations of Motion
    The continuous-time dynamics are given in standard manipulator form:
    $$H(q) \ddot{q} + C(q, \dot{q})\dot{q} + G(q) = B u$$
    where $q = [p, \theta]^\top$ and:
    $$H(q) = \begin{bmatrix} m_c + m_p & m_p l \cos\theta \\ m_p l \cos\theta & m_p l^2 \end{bmatrix}, \quad C(q, \dot{q}) = \begin{bmatrix} 0 & -m_p \dot{\theta} l \sin\theta \\ 0 & 0 \end{bmatrix}, \quad G(q) = \begin{bmatrix} 0 \\ m_p g l \sin\theta \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
    """)
    return


@app.cell
def _(
    Cartpole,
    ConstraintList,
    ControlBound,
    LQRObjective,
    MPCState,
    Problem,
    RK4,
    StateBound,
    jnp,
    np,
):
    # System dimensions & horizon configuration
    n = 4  # state dimension [p, theta, p_dot, theta_dot]
    m = 1  # control dimension [F]
    N = 20  # horizon knot points
    dt = 0.05  # discretization step (50 ms -> 1.0 s prediction horizon)
    tf_horizon = (N - 1) * dt
    n_steps = 40  # total closed-loop simulation steps (2.0 s total)

    # Initial perturbed state and upright stabilization target
    # Cart at center, pole slightly tilted (-14.3 deg), small linear and angular velocities
    x0 = jnp.array([0.0, np.pi - 0.25, 0.1, -0.2], dtype=jnp.float64)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0], dtype=jnp.float64)

    # LQR Tracking Objective weights
    Q = jnp.diag(jnp.array([5.0, 20.0, 1.0, 2.0])) * dt
    R = jnp.diag(jnp.array([0.05])) * dt
    Qf = jnp.diag(jnp.array([50.0, 200.0, 10.0, 20.0]))

    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    # Constraints: Actuator limits & Cart track safety envelope
    u_max = 20.0  # Force limits [-20 N, +20 N]
    x_track_max = 0.8  # Track position limits [-0.8 m, +0.8 m]

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(
        ControlBound(n=n, m=m, u_min=[-u_max], u_max=[u_max]),
        range(N - 1),
    )
    clist.add_constraint(
        StateBound(
            n=n,
            m=m,
            x_min=[-x_track_max, -np.inf, -np.inf, -np.inf],
            x_max=[x_track_max, np.inf, np.inf, np.inf],
        ),
        range(N),
    )

    # Physical model & 4th-order Runge-Kutta integrator
    model = Cartpole(mc=1.0, mp=0.2, l=0.5, g=9.81)
    integrator = RK4()
    dmodel = model.discretize(integrator)

    # Assemble the Trajectory Optimization Problem
    prob = Problem(
        model=model,
        obj=obj,
        constraints=clist,
        N=N,
        integrator=integrator,
    )

    # Initialize MPC state container
    state_init = MPCState.initial(prob, x0=x0, dt=dt)

    return (
        N,
        clist,
        dmodel,
        dt,
        integrator,
        m,
        model,
        n,
        n_steps,
        obj,
        prob,
        state_init,
        tf_horizon,
        u_max,
        x0,
        x_track_max,
        xf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Initial Solve Benchmark: ALTRO vs. Ipopt

    Before starting closed-loop simulation, we solve the initial optimization problem at $t = 0$ using both:
    1. **`ALTRO`**: JAX-native Augmented Lagrangian Trajectory Optimizer using iLQR Riccati backward-forward passes and Projected Newton polish.
    2. **`Ipopt`**: General-purpose sparse interior-point NLP solver across the transcribed direct collocation system.
    """)
    return


@app.cell
def _(ALTRO, Ipopt, prob, state_init, time):
    altro_solver = ALTRO()
    ipopt_solver = Ipopt()

    # Warm up JIT compilation for ALTRO
    _ = prob.solve(state_init, solver=altro_solver)

    # Benchmark ALTRO initial solve
    t0_altro = time.perf_counter()
    res_altro_init = prob.solve(state_init, solver=altro_solver)
    altro_init_ms = (time.perf_counter() - t0_altro) * 1000.0

    # Benchmark Ipopt initial solve
    t0_ipopt = time.perf_counter()
    res_ipopt_init = prob.solve(state_init, solver=ipopt_solver)
    ipopt_init_ms = (time.perf_counter() - t0_ipopt) * 1000.0

    return (
        altro_init_ms,
        altro_solver,
        ipopt_init_ms,
        ipopt_solver,
        res_altro_init,
        res_ipopt_init,
    )


@app.cell(hide_code=True)
def _(altro_init_ms, ipopt_init_ms, mo, res_altro_init, res_ipopt_init):
    mo.md(rf"""
    ### Initial Step Benchmark Results ($t = 0.0\,\text{{s}}$)

    | Metric | ALTRO (JAX Native) | Ipopt (MUMPS Sparse NLP) |
    | :--- | :--- | :--- |
    | **Solver Status** | `{res_altro_init.status}` | `{res_ipopt_init.status}` |
    | **Solve Latency** | **`{altro_init_ms:.2f} ms`** | `{ipopt_init_ms:.2f} ms` |
    | **Initial Control $u_0^*$** | `{float(res_altro_init.controls[0, 0]):.4f} N` | `{float(res_ipopt_init.controls[0, 0]):.4f} N` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Closed-Loop Simulation with Unmodeled Disturbance Injection

    We now execute the full closed-loop MPC feedback loop for **40 steps ($T_{\text{total}} = 2.0\,\text{s}$)**:
    - At each step $k$, the controller measures the state $x(t_k)$ via `state.with_measurement(x_curr, t_curr)`.
    - `prob.solve(state, solver=ALTRO())` optimizes the next control trajectory.
    - The first control input $u_0^*$ is applied to simulate the true plant dynamics:
      $$x(t_{k+1}) = f_d(x(t_k), u_0^*, t_k, \Delta t)$$
    - `state.shift(dt)` rolls the trajectory and Augmented Lagrangian dual multipliers forward by $\Delta t$, warm-starting the next solve.
    - **Disturbance Kick**: At $t = 1.0\,\text{s}$ (step 20), an unmodeled angular velocity impulse of **$+1.5\,\text{rad/s}$** is injected into the pole. The MPC loop must immediately detect the state deviation and steer the cartpole back to upright equilibrium without violating track limits.
    """)
    return


@app.cell
def _(altro_solver, dmodel, dt, n_steps, np, prob, state_init, time, x0):
    # Closed-loop state and telemetry logs
    x_curr = x0
    t_curr = 0.0
    state = state_init

    x_history = [np.asarray(x_curr)]
    u_history = []
    t_history = [0.0]
    solve_times_ms = []

    # Snapshot prediction horizons at specific steps to visualize receding horizon fans
    snapshot_steps = [0, 10, 20, 30]
    horizon_plans = {}

    for step in range(n_steps):
        # Unmodeled impulse disturbance kick at t = 1.0 s (step 20)
        if step == 20:
            x_curr = x_curr.at[3].add(1.5)

        # 1. State Feedback: update initial condition & timestamp
        state = state.with_measurement(x_curr, t_curr)

        # 2. Optimize Trajectory with ALTRO
        t_solve_start = time.perf_counter()
        state = prob.solve(state, solver=altro_solver)
        dt_solve = (time.perf_counter() - t_solve_start) * 1000.0
        solve_times_ms.append(dt_solve)

        # Save snapshot prediction horizon
        if step in snapshot_steps:
            pred_traj = state.to_trajectory()
            horizon_plans[step] = {
                "t": np.asarray(pred_traj.t),
                "X": np.asarray(pred_traj.X),
                "U": np.asarray(pred_traj.U),
            }

        # 3. Apply first control command to the simulated plant
        u_cmd = state.controls[0]
        u_history.append(float(np.asarray(u_cmd)[0]))

        # 4. Integrate plant dynamics forward by dt
        x_curr = dmodel.discrete_dynamics(x_curr, u_cmd, t_curr, dt)
        x_history.append(np.asarray(x_curr))

        # 5. Shift state forward for warm-starting the next iteration
        state = state.shift(dt)
        t_curr += dt
        t_history.append(float(t_curr))

    x_history_arr = np.array(x_history)
    u_history_arr = np.array(u_history)
    t_history_arr = np.array(t_history)
    solve_times_arr = np.array(solve_times_ms)

    return (
        horizon_plans,
        snapshot_steps,
        solve_times_arr,
        t_history_arr,
        u_history_arr,
        x_history_arr,
    )


@app.cell(hide_code=True)
def _(dt, mo, np, solve_times_arr):
    mean_lat = np.mean(solve_times_arr)
    median_lat = np.median(solve_times_arr)
    p95_lat = np.percentile(solve_times_arr, 95)
    min_lat = np.min(solve_times_arr)
    max_lat = np.max(solve_times_arr)
    freq_hz = 1000.0 / mean_lat
    budget_ms = dt * 1000.0

    mo.md(rf"""
    ## 5. Telemetry & Real-Time Performance Analysis

    | Metric | Value | Target / Budget |
    | :--- | :--- | :--- |
    | **Sampling Period $\Delta t$** | **`{budget_ms:.1f} ms`** | $50.0\,\text{{ms}}$ |
    | **Mean Solve Latency** | **`{mean_lat:.2f} ms`** | $< 50.0\,\text{{ms}}$ |
    | **Median Solve Latency** | **`{median_lat:.2f} ms`** | $< 50.0\,\text{{ms}}$ |
    | **95th Percentile (p95)** | **`{p95_lat:.2f} ms`** | $< 50.0\,\text{{ms}}$ |
    | **Min / Max Latency** | **`{min_lat:.2f} ms` / `{max_lat:.2f} ms`** | — |
    | **Sustained Control Rate** | **`{freq_hz:.1f} Hz`** | $\ge 20.0\,\text{{Hz}}$ |

    > [!TIP]
    > **Warm-Starting Efficacy**: Once JIT-compiled by XLA, `MPCState.shift()` initializes both the primal trajectory and the Augmented Lagrangian dual multipliers close to the optimal manifold. Between steps $k$ and $k+1$, ALTRO requires only a few Riccati sweeps to re-converge.
    """)
    return (
        budget_ms,
        freq_hz,
        max_lat,
        mean_lat,
        median_lat,
        min_lat,
        p95_lat,
    )


@app.cell
def _(
    budget_ms,
    horizon_plans,
    np,
    p95_lat,
    plt,
    solve_times_arr,
    t_history_arr,
    u_history_arr,
    u_max,
    x_history_arr,
    x_track_max,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), dpi=120)

    # Palette colors
    fan_colors = {
        0: "#2980b9",  # Blue (Initial stabilization)
        10: "#27ae60",  # Green (Steady upright)
        20: "#e74c3c",  # Red (Disturbance kick reaction)
        30: "#8e44ad",  # Purple (Recovery)
    }

    # -------------------------------------------------------------
    # Subplot 1: Cart Position & Pole Angle with Overlaid Horizon Fans
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    # Closed-loop actual trajectories
    t_states = t_history_arr
    p_actual = x_history_arr[:, 0]
    theta_actual = x_history_arr[:, 1]

    # Cart track bounds
    ax1.axhline(x_track_max, color="#c0392b", linestyle=":", linewidth=1.4, label=r"Track Limit $(\pm 0.8\,$m$)$")
    ax1.axhline(-x_track_max, color="#c0392b", linestyle=":", linewidth=1.4)
    ax1.fill_between(t_states, x_track_max, 1.2, color="#c0392b", alpha=0.08)
    ax1.fill_between(t_states, -1.2, -x_track_max, color="#c0392b", alpha=0.08)

    # Plot actual closed-loop curves
    line_p = ax1.plot(t_states, p_actual, color="#1f77b4", linewidth=2.5, label="Cart Position $p(t)$ [m]")[0]
    line_th = ax1_twin.plot(
        t_states, theta_actual, color="#d35400", linewidth=2.5, linestyle="-", label=r"Pole Angle $\theta(t)$ [rad]"
    )[0]
    ax1_twin.axhline(np.pi, color="#7f8c8d", linestyle="--", linewidth=1.2, label=r"Upright Goal $\theta = \pi$")

    # Overlaid predicted horizon curves ("receding horizon fans")
    for step_idx, hplan in horizon_plans.items():
        c = fan_colors.get(step_idx, "#95a5a6")
        label_p = f"Pred. Horizon (step {step_idx}, $t={step_idx*0.05:.1f}\\,$s)"
        ax1.plot(hplan["t"], hplan["X"][:, 0], color=c, linestyle="--", linewidth=1.6, alpha=0.85, label=label_p)
        ax1_twin.plot(hplan["t"], hplan["X"][:, 1], color=c, linestyle=":", linewidth=1.6, alpha=0.85)

    # Disturbance annotation
    ax1.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.8, alpha=0.9)
    ax1.annotate(
        "Impulse Disturbance\n$(\\Delta\\dot{\\theta} = +1.5\\,\\text{rad/s})$",
        xy=(1.0, 0.2),
        xytext=(1.08, 0.45),
        arrowprops={"arrowstyle": "->", "color": "#c0392b", "lw": 1.8},
        fontsize=9,
        fontweight="bold",
        color="#c0392b",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#fbeee6", "ec": "#c0392b", "lw": 1.2},
    )

    ax1.set_xlim(0.0, 2.0)
    ax1.set_ylim(-0.95, 0.95)
    ax1_twin.set_ylim(2.5, 3.6)
    ax1.set_ylabel("Cart Position $p$ (m)", color="#1f77b4", fontweight="bold")
    ax1_twin.set_ylabel(r"Pole Angle $\theta$ (rad)", color="#d35400", fontweight="bold")
    ax1.set_title("Closed-Loop States with Overlaid Receding-Horizon Prediction Fans", fontweight="bold", fontsize=12)
    ax1.grid(visible=True, linestyle=":", alpha=0.6)

    lines_ax1 = [line_p, line_th]
    labels_ax1 = [line_item.get_label() for line_item in lines_ax1]
    ax1.legend(lines_ax1, labels_ax1, loc="upper left", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Subplot 2: Commanded Force u(t) with Actuation Bounds
    # -------------------------------------------------------------
    ax2 = axes[1]
    t_ctrl = t_history_arr[:-1]

    ax2.axhline(u_max, color="#c0392b", linestyle=":", linewidth=1.5, label=r"Force Bound $(\pm 20\,$N$)$")
    ax2.axhline(-u_max, color="#c0392b", linestyle=":", linewidth=1.5)
    ax2.fill_between(t_ctrl, u_max, 25.0, color="#c0392b", alpha=0.08)
    ax2.fill_between(t_ctrl, -25.0, -u_max, color="#c0392b", alpha=0.08)

    ax2.step(t_ctrl, u_history_arr, where="post", color="#2c3e50", linewidth=2.2, label="Commanded Force $u_0(t)$")

    # Overlay control horizon plans
    for step_idx, hplan in horizon_plans.items():
        c = fan_colors.get(step_idx, "#95a5a6")
        t_u = hplan["t"][:-1]
        ax2.step(
            t_u,
            hplan["U"][:, 0],
            where="post",
            color=c,
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            label=f"Plan $U^*$ (step {step_idx})",
        )

    ax2.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.8, alpha=0.9)
    ax2.annotate(
        "Reactive Force Spike",
        xy=(1.0, u_history_arr[20]),
        xytext=(1.12, u_history_arr[20] - 6.0),
        arrowprops={"arrowstyle": "->", "color": "#2c3e50", "lw": 1.6},
        fontsize=9,
        fontweight="bold",
        color="#2c3e50",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ecf0f1", "ec": "#2c3e50", "lw": 1.0},
    )

    ax2.set_xlim(0.0, 2.0)
    ax2.set_ylim(-24.0, 24.0)
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel("Force $F$ (N)", fontweight="bold")
    ax2.set_title("Commanded Control Input $u(t)$ and Actuation Saturation Enforcement", fontweight="bold", fontsize=12)
    ax2.grid(visible=True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    # -------------------------------------------------------------
    # Subplot 3: Per-Step Solve Latency & Sustained Frequency
    # -------------------------------------------------------------
    ax3 = axes[2]
    steps = np.arange(len(solve_times_arr))
    step_times = steps * 0.05

    ax3.bar(step_times, solve_times_arr, width=0.035, color="#3498db", alpha=0.75, edgecolor="#2980b9", label="Solve Time (ms)")
    ax3.axhline(budget_ms, color="#c0392b", linestyle="--", linewidth=1.8, label=f"Timestep Budget $\\Delta t = {budget_ms:.0f}\\,$ms")
    ax3.axhline(p95_lat, color="#e67e22", linestyle=":", linewidth=1.6, label=f"p95 Latency ({p95_lat:.1f} ms)")

    ax3.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.8, alpha=0.9)

    ax3.set_xlim(-0.05, 2.05)
    ax3.set_ylim(0.0, max(np.max(solve_times_arr) * 1.25, budget_ms * 1.2))
    ax3.set_xlabel("Simulation Time $t$ (s)")
    ax3.set_ylabel("Latency (ms)", fontweight="bold")
    ax3.set_title("Per-Step Optimization Latency Telemetry", fontweight="bold", fontsize=12)
    ax3.grid(visible=True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig
    return (
        ax1,
        ax1_twin,
        ax2,
        ax3,
        axes,
        c,
        fan_colors,
        fig,
        hplan,
        label_p,
        labels_ax1,
        line_p,
        line_th,
        lines_ax1,
        p_actual,
        step_idx,
        step_times,
        steps,
        t_ctrl,
        t_states,
        t_u,
        theta_actual,
    )


@app.cell
def _(model, np, plt, x_history_arr):
    # Visual keyframe snapshots of the cartpole physical geometry
    fig_geom, ax_geom = plt.subplots(figsize=(12, 4.5), dpi=120)

    key_steps = [0, 10, 20, 22, 30, 40]
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(key_steps)))
    pole_len = float(model.l)
    cart_w, cart_h = 0.22, 0.10

    # Draw track rail
    ax_geom.axhline(0, color="#7f8c8d", linewidth=3.0, zorder=1)
    ax_geom.axvline(-0.8, color="#c0392b", linestyle=":", linewidth=1.5, label="Track Bounds")
    ax_geom.axvline(0.8, color="#c0392b", linestyle=":", linewidth=1.5)

    for idx, col in zip(key_steps, palette, strict=True):
        st = x_history_arr[idx]
        p_pos = float(st[0])
        th_ang = float(st[1])
        t_val = idx * 0.05

        # Cart body
        cart_rect = plt.Rectangle(
            (p_pos - cart_w / 2, -cart_h / 2),
            cart_w,
            cart_h,
            facecolor=col,
            edgecolor="black",
            alpha=0.8,
            linewidth=1.2,
            zorder=3,
        )
        ax_geom.add_patch(cart_rect)

        # Pendulum bob position
        # Note theta is from downward vertical, so tip is at (p + l*sin(th), -l*cos(th))
        bob_x = p_pos + pole_len * np.sin(th_ang)
        bob_y = -pole_len * np.cos(th_ang)

        # Draw pole rod and tip mass
        ax_geom.plot([p_pos, bob_x], [0, bob_y], color=col, linewidth=2.8, zorder=4)
        ax_geom.plot(
            bob_x,
            bob_y,
            "o",
            color=col,
            markeredgecolor="black",
            markersize=9,
            zorder=5,
            label=f"$t={t_val:.2f}\\,$s",
        )

    ax_geom.set_xlim(-1.0, 1.0)
    ax_geom.set_ylim(-0.25, 0.75)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.set_xlabel("Track Position $p$ (m)", fontweight="bold")
    ax_geom.set_ylabel("Height $y$ (m)", fontweight="bold")
    ax_geom.set_title("Cartpole Motion Snapshots Across Closed-Loop Recovery", fontweight="bold", fontsize=12)
    ax_geom.grid(visible=True, linestyle=":", alpha=0.6)
    ax_geom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.tight_layout()
    fig_geom
    return (
        ax_geom,
        bob_x,
        bob_y,
        cart_h,
        cart_rect,
        cart_w,
        col,
        fig_geom,
        idx,
        key_steps,
        palette,
        p_pos,
        pole_len,
        st,
        t_val,
        th_ang,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Key Takeaways & Practical Insights

    1. **Receding Horizon Robustness**: Even with an aggressive unmodeled angular velocity impulse ($\Delta\dot{\theta} = +1.5\,\text{rad/s}$ at $t = 1.0\,\text{s}$), receding-horizon replanning at $20\,\text{Hz}$ naturally rejects the disturbance and stabilizes the pole to upright vertical without requiring manual gain scheduling.
    2. **State Feedback & Multiplier Shifting**: `MPCState.with_measurement(x_meas, t_meas)` and `MPCState.shift(dt)` enable zero-overhead state updating and primal-dual warm-starting across iterations.
    3. **Constraint Fidelity**: Actuator commands remain strictly within $[-20\,\text{N}, +20\,\text{N}]$, with the controller exploiting the full force envelope during the reactive recovery spike without violating track safety limits ($|p| \le 0.8\,\text{m}$).
    4. **JAX ALTRO Acceleration**: ALTRO's Riccati-based backward-forward sweep delivers fast convergence suitable for real-time robotic deployment.
    """)
    return


if __name__ == "__main__":
    app.run()
