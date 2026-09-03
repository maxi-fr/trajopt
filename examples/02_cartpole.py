import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patches

    from trajopt.constraints.bounds import ControlBound, StateBound
    from trajopt.constraints.constraint_list import ConstraintList
    from trajopt.constraints.linear import GoalConstraint
    from trajopt.costs.objective import LQRObjective
    from trajopt.dynamics.integrators import RK4
    from trajopt.models.cartpole import Cartpole
    from trajopt.mpc import MPC
    from trajopt.problem import Problem
    from trajopt.solvers.altro import ALTRO
    from trajopt.transcription.ipopt import Ipopt

    return (
        ALTRO,
        Cartpole,
        ConstraintList,
        ControlBound,
        GoalConstraint,
        Ipopt,
        LQRObjective,
        MPC,
        Problem,
        RK4,
        StateBound,
        jnp,
        np,
        patches,
        plt,
        time,
    )


@app.cell
def _(jnp, np):
    # Horizon and discretization
    N = 25
    dt = 0.05
    tf = (N - 1) * dt

    # Initial state (hanging down with slight perturbation) and target upright state
    x0 = jnp.array([0.0, 0.01, 0.0, 0.0])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])

    # Physical limits
    x_pos_bound = 0.4  # Track limit |x| <= 0.4 m
    u_bound = 20.0  # Control limit |u| <= 20.0 N

    # Time grid
    time_grid = np.linspace(0.0, tf, N)
    time_grid_u = time_grid[:-1]
    return N, dt, time_grid, time_grid_u, u_bound, x0, x_pos_bound, xf


@app.cell
def _(Cartpole):
    model = Cartpole()
    return (model,)


@app.cell
def _(LQRObjective, N, jnp):
    # Stage cost matrices
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))

    # Terminal cost matrix
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))

    obj = LQRObjective(Q=Q, R=R, Qf=Qf, N=N)
    return (obj,)


@app.cell
def _(
    ConstraintList,
    ControlBound,
    GoalConstraint,
    N,
    StateBound,
    model,
    np,
    u_bound,
    x_pos_bound,
    xf,
):
    n, m = model.n, model.m

    # Track position bounds: -x_pos_bound <= p <= x_pos_bound, unconstrained angles & velocities
    x_min = [-x_pos_bound, -np.inf, -np.inf, -np.inf]
    x_max = [x_pos_bound, np.inf, np.inf, np.inf]

    # Actuator bounds: -u_bound <= u <= u_bound
    u_min = [-u_bound]
    u_max = [u_bound]

    # Assemble constraints into ConstraintList
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(StateBound(n=n, m=m, x_min=x_min, x_max=x_max), range(N))
    cl.add_constraint(ControlBound(n=n, m=m, u_min=u_min, u_max=u_max), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    return (cl,)


@app.cell
def _(N, Problem, RK4, cl, dt, model, obj):
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, dt=dt, integrator=RK4())
    return (prob,)


@app.cell
def _(ALTRO, MPC, prob, time, x0, xf):
    altro_mpc = MPC(prob, ALTRO(), x0=x0, xf=xf)

    t_start_altro = time.perf_counter()
    altro_res = altro_mpc.solve()
    time.perf_counter() - t_start_altro
    return (altro_res,)


@app.cell
def _(Ipopt, MPC, prob, time, x0, xf):
    ipopt_mpc = MPC(prob, Ipopt(options={"print_level": 0, "tol": 1e-6, "max_iter": 300}), x0=x0, xf=xf)

    t_start_ipopt = time.perf_counter()
    ipopt_res = ipopt_mpc.solve()
    time.perf_counter() - t_start_ipopt
    return (ipopt_res,)


@app.cell
def _(altro_res, ipopt_res, np, plt, time_grid, x_pos_bound):
    fig_states, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # Extract state profiles
    X_ipopt = np.asarray(ipopt_res.trajectory.X)
    X_altro = np.asarray(altro_res.trajectory.X)

    # Cart position x(t)
    ax = axes[0, 0]
    ax.axhline(x_pos_bound, color="crimson", linestyle="--", alpha=0.7, label=r"Track bounds $\pm 0.4\,\text{m}$")
    ax.axhline(-x_pos_bound, color="crimson", linestyle="--", alpha=0.7)
    ax.fill_between(time_grid, -x_pos_bound, x_pos_bound, color="crimson", alpha=0.08)
    ax.plot(time_grid, X_ipopt[:, 0], "b-", linewidth=2.0, label="Ipopt")
    ax.plot(time_grid, X_altro[:, 0], "g--", linewidth=1.5, label="ALTRO")
    ax.set_ylabel("Cart Position $x$ (m)", fontsize=11)
    ax.set_title("Cart Position with Track Limits", fontsize=12, fontweight="bold")
    ax.grid(visible=True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9)

    # Pole angle theta(t)
    ax = axes[0, 1]
    ax.axhline(np.pi, color="purple", linestyle=":", alpha=0.7, label=r"Upright Goal $\theta = \pi$")
    ax.plot(time_grid, X_ipopt[:, 1], "b-", linewidth=2.0, label="Ipopt")
    ax.plot(time_grid, X_altro[:, 1], "g--", linewidth=1.5, label="ALTRO")
    ax.set_ylabel(r"Pole Angle $\theta$ (rad)", fontsize=11)
    ax.set_title("Pendulum Angle", fontsize=12, fontweight="bold")
    ax.grid(visible=True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    # Cart velocity x_dot(t)
    ax = axes[1, 0]
    ax.plot(time_grid, X_ipopt[:, 2], "b-", linewidth=2.0, label="Ipopt")
    ax.plot(time_grid, X_altro[:, 2], "g--", linewidth=1.5, label="ALTRO")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"Cart Velocity $\dot{x}$ (m/s)", fontsize=11)
    ax.set_title("Cart Velocity", fontsize=12, fontweight="bold")
    ax.grid(visible=True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    # Pole angular velocity theta_dot(t)
    ax = axes[1, 1]
    ax.plot(time_grid, X_ipopt[:, 3], "b-", linewidth=2.0, label="Ipopt")
    ax.plot(time_grid, X_altro[:, 3], "g--", linewidth=1.5, label="ALTRO")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"Angular Velocity $\dot{\theta}$ (rad/s)", fontsize=11)
    ax.set_title("Pole Angular Velocity", fontsize=12, fontweight="bold")
    ax.grid(visible=True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    fig_states.suptitle("Optimal State Trajectories: Underactuated Cartpole", fontsize=14, fontweight="bold", y=0.99)
    fig_states.tight_layout()
    return (X_ipopt,)


@app.cell
def _(altro_res, ipopt_res, np, plt, time_grid_u, u_bound):
    fig_ctrl, ax_ctrl = plt.subplots(figsize=(9, 4.5))

    U_ipopt = np.asarray(ipopt_res.trajectory.U).squeeze()
    U_altro = np.asarray(altro_res.trajectory.U).squeeze()

    ax_ctrl.axhline(
        u_bound, color="crimson", linestyle="--", linewidth=1.5, label=r"Actuator Bounds $\pm 20.0\,\text{N}$"
    )
    ax_ctrl.axhline(-u_bound, color="crimson", linestyle="--", linewidth=1.5)
    ax_ctrl.fill_between(time_grid_u, -u_bound, u_bound, color="crimson", alpha=0.08)

    ax_ctrl.step(time_grid_u, U_ipopt, where="post", color="royalblue", linewidth=2.0, label="Ipopt Control $u(t)$")
    ax_ctrl.step(
        time_grid_u,
        U_altro,
        where="post",
        color="forestgreen",
        linestyle="--",
        linewidth=1.5,
        label="ALTRO Control $u(t)$",
    )

    ax_ctrl.set_xlabel("Time (s)", fontsize=11)
    ax_ctrl.set_ylabel("Applied Force $F$ (N)", fontsize=11)
    ax_ctrl.set_title("Actuator Force Trajectory with Bounds", fontsize=13, fontweight="bold")
    ax_ctrl.set_ylim(-u_bound * 1.25, u_bound * 1.25)
    ax_ctrl.grid(visible=True, linestyle=":", alpha=0.6)
    ax_ctrl.legend(loc="lower right", framealpha=0.9)

    fig_ctrl.tight_layout()
    return


@app.cell
def _(N, X_ipopt, model, np, patches, plt, x_pos_bound):
    fig_anim, ax_anim = plt.subplots(figsize=(10, 5))

    cart_w, cart_h = 0.12, 0.06
    pole_len = float(model.l)
    wheel_r = 0.015

    # Select key knot point snapshots across time horizon
    n_frames = 9
    sample_indices = np.linspace(0, N - 1, n_frames, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, n_frames))

    # Track line and bounds
    ax_anim.axhline(0.0, color="gray", linewidth=1.5, zorder=1)
    ax_anim.axvline(
        x_pos_bound, color="crimson", linestyle="--", linewidth=1.5, label=r"Track Limits $\pm 0.4\,\text{m}$"
    )
    ax_anim.axvline(-x_pos_bound, color="crimson", linestyle="--", linewidth=1.5)
    ax_anim.fill_betweenx([-0.6, 0.8], -x_pos_bound, x_pos_bound, color="crimson", alpha=0.04)

    # Draw cartpole snapshots
    for i, idx in enumerate(sample_indices):
        t_val = idx * 0.05
        p_val = float(X_ipopt[idx, 0])
        th_val = float(X_ipopt[idx, 1])
        alpha = 0.35 + 0.65 * (i / (n_frames - 1))
        col = colors[i]

        # Cart body
        cart_x = p_val - cart_w / 2.0
        cart_y = -cart_h / 2.0
        rect = patches.FancyBboxPatch(
            (cart_x, cart_y),
            cart_w,
            cart_h,
            boxstyle="round,pad=0.005",
            facecolor=col,
            edgecolor="black",
            linewidth=1.0,
            alpha=alpha,
            zorder=3,
        )
        ax_anim.add_patch(rect)

        # Wheels
        for wx in [cart_x + 0.02, cart_x + cart_w - 0.02]:
            ax_anim.add_patch(patches.Circle((wx, cart_y - wheel_r), wheel_r, facecolor="black", alpha=alpha, zorder=2))

        # Pendulum pole (theta measured from downward vertical)
        tip_x = p_val + pole_len * np.sin(th_val)
        tip_y = -pole_len * np.cos(th_val)
        ax_anim.plot([p_val, tip_x], [0.0, tip_y], color="black", linewidth=2.0, alpha=alpha, zorder=4)
        ax_anim.add_patch(
            patches.Circle(
                (tip_x, tip_y), 0.022, facecolor=col, edgecolor="black", linewidth=1.0, alpha=alpha, zorder=5
            )
        )

        # Timestamp label
        ax_anim.text(
            p_val, cart_y - 0.06, f"{t_val:.2f}s", fontsize=8, ha="center", va="top", color=col, fontweight="bold"
        )

    ax_anim.set_xlim(-0.6, 0.6)
    ax_anim.set_ylim(-0.65, 0.75)
    ax_anim.set_aspect("equal")
    ax_anim.set_xlabel("Track Position $x$ (m)", fontsize=11)
    ax_anim.set_ylabel("Vertical Position $y$ (m)", fontsize=11)
    ax_anim.set_title("Cartpole Swing-Up Motion Sequence (0.00s to 1.20s)", fontsize=13, fontweight="bold")
    ax_anim.grid(visible=True, linestyle=":", alpha=0.5)
    ax_anim.legend(loc="upper right", framealpha=0.9)

    fig_anim.tight_layout()
    return


if __name__ == "__main__":
    app.run()
