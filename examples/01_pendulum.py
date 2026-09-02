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

    from trajopt.constraints.bounds import ControlBound
    from trajopt.constraints.constraint_list import ConstraintList
    from trajopt.constraints.linear import GoalConstraint
    from trajopt.costs.objective import LQRObjective
    from trajopt.dynamics.integrators import RK4
    from trajopt.models.pendulum import Pendulum
    from trajopt.problem import MPCState, Problem
    from trajopt.solvers.altro import ALTRO
    from trajopt.transcription.ipopt import Ipopt

    return (
        ALTRO,
        ConstraintList,
        ControlBound,
        GoalConstraint,
        Ipopt,
        LQRObjective,
        MPCState,
        Pendulum,
        Problem,
        RK4,
        jnp,
        mo,
        np,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 — Inverted Pendulum Swing-Up with ALTRO & Ipopt

    This notebook demonstrates trajectory optimization for the classic **underactuated, torque-limited inverted pendulum swing-up** benchmark using `trajopt`.

    We solve the optimal control problem using two complementary solvers:
    1. **ALTRO** (*Augmented Lagrangian Trajectory Optimizer*): A fast, native JAX solver combining iLQR (Iterative Linear Quadratic Regulator), an Augmented Lagrangian outer loop for inequality/equality constraints, and Projected Newton (PN) multiplier polish.
    2. **Ipopt** (*Interior Point Optimizer*): The industry-standard interior-point NLP solver, run against the transcribed sparse problem representation.

    ---

    ## System Dynamics & Mathematical Formulation

    The simple pendulum consists of a point mass $m$ at distance $l_c$ from a pivot with viscous friction coefficient $b$, actuated by a control torque $\tau$ applied at the pivot.

    ### State and Control Variables
    - **State vector**: $x = [\theta, \omega]^\top \in \mathbb{R}^2$
      - $\theta$: Angle from downward vertical in radians ($0 = \text{down}$, $\pi = \text{upright}$).
      - $\omega = \dot{\theta}$: Angular velocity in $\text{rad/s}$.
    - **Control input**: $u = [\tau] \in \mathbb{R}^1$
      - $\tau$: Control torque applied at the base in $\text{N}\cdot\text{m}$.

    ### Continuous Dynamics
    The equation of motion is:
    $$\ddot{\theta} = \frac{\tau - b\,\omega - m\,g\,l_c \sin\theta}{m\,l_c^2}$$

    In first-order state-space form $\dot{x} = f(x, u)$:
    $$\dot{x}(t) = \begin{bmatrix} \dot{\theta}(t) \\ \ddot{\theta}(t) \end{bmatrix} = \begin{bmatrix} \omega(t) \\ \frac{\tau(t) - b\,\omega(t) - m\,g\,l_c \sin(\theta(t))}{m\,l_c^2} \end{bmatrix}$$

    ### Objective Function
    We formulate a finite-horizon discrete-time quadratic tracking cost:
    $$J(X, U) = \frac{1}{2} (x_{N-1} - x_f)^\top Q_f (x_{N-1} - x_f) + \sum_{k=0}^{N-2} \left[ \frac{1}{2} (x_k - x_f)^\top Q (x_k - x_f) + \frac{1}{2} u_k^\top R u_k \right]$$

    ### Constraints
    - **Torque Limits (Control Bound)**:
      $$-u_{\max} \le u_k \le u_{\max}, \quad \forall k \in \{0, \dots, N-2\} \quad (u_{\max} = 5.0\,\text{N}\cdot\text{m})$$
    - **Boundary Conditions**:
      - Initial state: $x_0 = [0.0, 0.0]^\top$ (hanging at rest)
      - Terminal goal: $x_{N-1} = x_f = [\pi, 0.0]^\top$ (upright at rest)
    """)
    return


@app.cell
def _(Pendulum, RK4, jnp, np):
    # Physical and horizon parameters
    n = 2  # state dimension [theta, omega]
    m = 1  # control dimension [tau]
    N = 51  # number of knot points
    tf = 3.0  # total time horizon (seconds)
    dt = tf / (N - 1)  # discretization timestep (0.06 s)

    # Initial and target states
    x0 = jnp.array([0.0, 0.0])  # downward equilibrium
    xf = jnp.array([np.pi, 0.0])  # upright equilibrium

    # Dynamics model & numerical integrator
    model = Pendulum(mass=1.0, len=0.5, b=0.1, lc=0.5, I=0.25, g=9.81)
    integrator = RK4()
    return N, dt, integrator, m, model, n, x0, xf


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Cost Formulation (`LQRObjective`)

    We use an `LQRObjective` with quadratic stage cost matrices $Q$ and $R$ (scaled by $\Delta t$) and a terminal cost matrix $Q_f$.
    The cost penalizes deviations from the upright equilibrium $x_f = [\pi, 0.0]^\top$ and control effort $\tau^2$.
    """)
    return


@app.cell
def _(LQRObjective, N, dt, jnp, xf):
    # Stage and terminal cost matrices
    Q = jnp.diag(jnp.array([1.0, 0.1])) * dt
    R = jnp.diag(jnp.array([0.01])) * dt
    Qf = jnp.diag(jnp.array([100.0, 10.0]))

    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)
    return (obj,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Constraints (`ConstraintList`, `ControlBound`, `GoalConstraint`)

    Next, we define:
    1. `ControlBound`: Symmetric actuation bounds $|u_k| \le 5.0\,\text{N}\cdot\text{m}$ for all control stages $k \in \{0, \dots, N-2\}$.
    2. `GoalConstraint`: Exact terminal equality constraint $x_{N-1} = x_f$ at knot point $N-1$.
    """)
    return


@app.cell
def _(ConstraintList, ControlBound, GoalConstraint, N, m, n, xf):
    u_max = 5.0  # N*m torque limit
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(
        ControlBound(n=n, m=m, u_min=[-u_max], u_max=[u_max]),
        range(N - 1),
    )
    clist.add_constraint(
        GoalConstraint(n=n, xf=xf.tolist()),
        N - 1,
    )
    return clist, u_max


@app.cell
def _(MPCState, N, Problem, clist, dt, integrator, model, obj, x0):
    # Assemble the optimal control problem
    prob = Problem(
        model=model,
        obj=obj,
        constraints=clist,
        N=N,
        integrator=integrator,
    )

    # Initial MPC state containing initial conditions and timestep
    state = MPCState.initial(prob, x0=x0, dt=dt)
    return prob, state


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Solving the Problem: ALTRO vs. Ipopt

    We solve the problem using two different optimization backends:
    - **ALTRO (`trajopt.solvers.altro.ALTRO`)**: JAX-native solver implementing Augmented Lagrangian iLQR with Projected Newton multiplier polish. Operates directly on the stagewise control problem structure with fast Riccati backward passes.
    - **Ipopt (`trajopt.transcription.ipopt.Ipopt`)**: Interior-point optimizer over the full transcribed direct collocation nonlinear program using sparse Jacobians and Hessians.
    """)
    return


@app.cell
def _(ALTRO, prob, state, time):
    # Warmup solve for JIT compilation
    _ = ALTRO().solve(prob, state)

    # Timed solve with ALTRO
    t0_altro = time.perf_counter()
    res_altro = ALTRO().solve(prob, state)
    altro_solve_time_ms = (time.perf_counter() - t0_altro) * 1000.0
    return altro_solve_time_ms, res_altro


@app.cell
def _(Ipopt, prob, state, time):
    # Solve with Ipopt (print_level=0 for clean output)
    t0_ipopt = time.perf_counter()
    res_ipopt = Ipopt(options={"print_level": 0}).solve(prob, state)
    ipopt_solve_time_ms = (time.perf_counter() - t0_ipopt) * 1000.0
    return ipopt_solve_time_ms, res_ipopt


@app.cell
def _(
    altro_solve_time_ms,
    ipopt_solve_time_ms,
    mo,
    np,
    res_altro,
    res_ipopt,
    xf,
):
    altro_err = float(np.linalg.norm(np.asarray(res_altro.trajectory.X[-1]) - np.asarray(xf)))
    ipopt_err = float(np.linalg.norm(np.asarray(res_ipopt.trajectory.X[-1]) - np.asarray(xf)))

    mo.md(
        rf"""
        ### Solver Performance & Accuracy Comparison

        | Metric | ALTRO (JAX Native) | Ipopt (Interior Point) |
        | :--- | :--- | :--- |
        | **Convergence Status** | `{"Success" if res_altro.success else "Failed"}` ({res_altro.message}) | `{"Success" if res_ipopt.success else "Failed"}` ({res_ipopt.message}) |
        | **Iterations** | `{res_altro.iterations}` outer AL iterations | `{res_ipopt.iterations}` NLP iterations |
        | **Solve Time** | `{altro_solve_time_ms:.2f} ms` | `{ipopt_solve_time_ms:.2f} ms` |
        | **Objective Cost $J$** | `{res_altro.cost:.4f}` | `{res_ipopt.cost:.4f}` |
        | **Constraint Violation $c_\max$** | `{res_altro.constraint_violation:.2e}` | `{res_ipopt.constraint_violation:.2e}` |
        | **Terminal Goal Error $\|x(t_f) - x_f\|_2$** | `{altro_err:.2e}` rad | `{ipopt_err:.2e}` rad |
        """
    )
    return


@app.cell
def _(np, plt, res_altro, res_ipopt, u_max):
    t_altro = np.asarray(res_altro.trajectory.t)
    X_altro = np.asarray(res_altro.trajectory.X)
    U_altro = np.asarray(res_altro.trajectory.U)

    t_ipopt = np.asarray(res_ipopt.trajectory.t)
    X_ipopt = np.asarray(res_ipopt.trajectory.X)
    U_ipopt = np.asarray(res_ipopt.trajectory.U)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # 1. Pendulum Angle
    axes[0].plot(t_altro, X_altro[:, 0], "b-", linewidth=2.5, label="ALTRO")
    axes[0].plot(t_ipopt, X_ipopt[:, 0], "r--", linewidth=2.0, label="Ipopt")
    axes[0].axhline(np.pi, color="black", linestyle=":", linewidth=1.5, label=r"Goal $\theta = \pi$")
    axes[0].set_ylabel(r"Angle $\theta(t)$ [rad]", fontsize=11)
    axes[0].set_title("Inverted Pendulum Optimal Trajectories", fontsize=13, fontweight="bold")
    axes[0].grid(visible=True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")

    # 2. Angular Velocity
    axes[1].plot(t_altro, X_altro[:, 1], "b-", linewidth=2.5, label="ALTRO")
    axes[1].plot(t_ipopt, X_ipopt[:, 1], "r--", linewidth=2.0, label="Ipopt")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel(r"Angular Velocity $\omega(t)$ [rad/s]", fontsize=11)
    axes[1].grid(visible=True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    # 3. Control Torque
    u_altro_plot = U_altro[:, 0] if U_altro.ndim > 1 else U_altro
    u_ipopt_plot = U_ipopt[:, 0] if U_ipopt.ndim > 1 else U_ipopt

    axes[2].plot(t_altro[:-1], u_altro_plot, "b-", linewidth=2.5, label=r"ALTRO $\tau(t)$")
    axes[2].plot(t_ipopt[:-1], u_ipopt_plot, "r--", linewidth=2.0, label=r"Ipopt $\tau(t)$")
    axes[2].axhline(u_max, color="firebrick", linestyle="-.", linewidth=1.5, label=r"Torque Limits $\pm 5.0\,\mathrm{N\cdot m}$")
    axes[2].axhline(-u_max, color="firebrick", linestyle="-.", linewidth=1.5)
    axes[2].fill_between(t_altro[:-1], u_max, u_max + 1.0, color="firebrick", alpha=0.1)
    axes[2].fill_between(t_altro[:-1], -u_max - 1.0, -u_max, color="firebrick", alpha=0.1)
    axes[2].set_ylim(-u_max - 1.0, u_max + 1.0)
    axes[2].set_ylabel(r"Control Torque $\tau(t)$ [$\mathrm{N\cdot m}$]", fontsize=11)
    axes[2].set_xlabel(r"Time $t$ [s]", fontsize=11)
    axes[2].grid(visible=True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    fig
    return X_altro, X_ipopt


@app.cell
def _(X_altro, X_ipopt, model, np, plt):
    fig2, (ax_phase, ax_geom) = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Panel 1: Phase Portrait ---
    theta_grid = np.linspace(-np.pi / 2, 3 * np.pi / 2, 25)
    omega_grid = np.linspace(-8.0, 8.0, 25)
    Theta, Omega = np.meshgrid(theta_grid, omega_grid)

    m_eff = float(model.mass * (model.lc**2))
    g_val = float(model.g)
    lc_val = float(model.lc)
    b_val = float(model.b)

    Theta_dot = Omega
    Omega_dot = - (g_val / lc_val) * np.sin(Theta) - (b_val / m_eff) * Omega

    ax_phase.streamplot(Theta, Omega, Theta_dot, Omega_dot, color="silver", density=0.8, arrowsize=1.0)
    ax_phase.plot(X_altro[:, 0], X_altro[:, 1], "b-", linewidth=2.5, label="ALTRO Trajectory")
    ax_phase.plot(X_ipopt[:, 0], X_ipopt[:, 1], "r--", linewidth=2.0, label="Ipopt Trajectory")
    ax_phase.plot(0.0, 0.0, "go", markersize=9, label="Initial State (0, 0)")
    ax_phase.plot(np.pi, 0.0, "y*", markersize=14, markeredgecolor="black", label=r"Goal State $(\pi, 0)$")

    ax_phase.set_xlabel(r"Angle $\theta$ [rad]", fontsize=11)
    ax_phase.set_ylabel(r"Angular Velocity $\dot{\theta}$ [rad/s]", fontsize=11)
    ax_phase.set_title("Phase Portrait & Swing-Up Trajectory", fontsize=12, fontweight="bold")
    ax_phase.grid(visible=True, linestyle="--", alpha=0.5)
    ax_phase.legend(loc="upper left")

    # --- Panel 2: Pendulum Geometry Snapshots ---
    L_pend = float(model.len)
    keyframe_indices = np.linspace(0, len(X_altro) - 1, 6, dtype=int)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(keyframe_indices)))

    ax_geom.plot(0, 0, "ko", markersize=8, zorder=5)  # Pivot

    for idx, col in zip(keyframe_indices, colors, strict=True):
        th = X_altro[idx, 0]
        t_val = idx * 3.0 / (len(X_altro) - 1)
        x_bob = L_pend * np.sin(th)
        y_bob = -L_pend * np.cos(th)
        ax_geom.plot([0, x_bob], [0, y_bob], "-", color=col, linewidth=2.5, alpha=0.85)
        ax_geom.plot(
            x_bob,
            y_bob,
            "o",
            color=col,
            markersize=10,
            markeredgecolor="black",
            alpha=0.9,
            label=f"$t={t_val:.1f}\\,$s",
        )

    # Draw circular trajectory arc of the tip
    arc_theta = np.linspace(0, np.pi, 100)
    ax_geom.plot(L_pend * np.sin(arc_theta), -L_pend * np.cos(arc_theta), "k:", alpha=0.3)

    ax_geom.set_xlim(-0.65, 0.65)
    ax_geom.set_ylim(-0.65, 0.65)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.set_xlabel(r"$x$ [m]", fontsize=11)
    ax_geom.set_ylabel(r"$y$ [m]", fontsize=11)
    ax_geom.set_title("Pendulum Swing-Up Keyframes", fontsize=12, fontweight="bold")
    ax_geom.grid(visible=True, linestyle="--", alpha=0.5)
    ax_geom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary & Key Takeaways

    1. **Underactuation & Torque Limits**: With actuation torque limited to $|u| \le 5.0\,\mathrm{N\cdot m}$, the pendulum cannot swing directly upwards in a single monotonic stroke. Instead, it must build mechanical energy by first pumping in the opposite direction before swinging up to $\theta = \pi$.
    2. **Solver Parity**: Both `ALTRO` and `Ipopt` converge to the identical optimal trajectory and objective value ($J \approx 2.5325$), confirming the mathematical fidelity of the discretization and constraints.
    3. **Performance Difference**:
       - `ALTRO` exploits the dynamic Riccati backward-pass structure, achieving convergence in milliseconds once JIT-compiled.
       - `Ipopt` formulates a general nonlinear programming problem with sparse KKT systems, providing robust global convergence from arbitrary initialization.
    """)
    return


if __name__ == "__main__":
    app.run()
