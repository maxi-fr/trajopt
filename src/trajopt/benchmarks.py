import dataclasses
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import SphereConstraint
from trajopt.constraints.linear import GoalConstraint
from trajopt.constraints.rotations import QuatVecEq
from trajopt.costs.objective import LQRObjective, Objective, TrackingObjective
from trajopt.costs.rotations import QuatGeodesicCost
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.dubins import DubinsCar
from trajopt.models.quadrotor import Quadrotor
from trajopt.problem import MPCState, Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.clarabel import Clarabel
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.layout import (
    compute_constraint_violation,
    constraint_bounds,
    primal_bounds,
)
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.result import Solver, SolverResult
from trajopt.transcription.transcription import (
    eval_f,
    eval_grad_f,
    eval_h,
    eval_jac_g,
)

# Backends that solve one convex subproblem about the Operating Point rather than the nonlinear
# problem itself. Their rows are flagged, since a faster time bought by answering an easier
# question is not a faster solve.
_LINEARIZING = (OSQP, Clarabel)


class ClosedLoopStats(NamedTuple):
    """Statistical measurement of a closed-loop MPC simulation.

    Parameters
    ----------
    num_steps : int
        Number of receding horizon MPC steps executed.
    durations_s : np.ndarray
        Array of solve durations for each MPC step in seconds, shape (num_steps,).
    mean_latency_s : float
        Mean solve latency across receding horizon steps in seconds.
    std_latency_s : float
        Standard deviation of solve latency (latency jitter) in seconds.
    min_latency_s : float
        Minimum solve latency in seconds.
    max_latency_s : float
        Maximum solve latency in seconds.
    median_latency_s : float
        Median solve latency in seconds.
    p95_latency_s : float
        95th percentile solve latency in seconds.
    p99_latency_s : float
        99th percentile solve latency in seconds.
    sustained_frequency_hz : float
        Sustained closed-loop MPC control frequency in Hz (1 / mean_latency_s).
    warmstart_speedup : float
        Ratio of cold-start to warm-start solve time, both measured at the same MPC step so the
        two solves face the same problem and differ only in their initial guess.
    total_duration_s : float
        Total elapsed wall-clock duration of the closed-loop run in seconds.
    """

    num_steps: int
    durations_s: np.ndarray
    mean_latency_s: float
    std_latency_s: float
    min_latency_s: float
    max_latency_s: float
    median_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    sustained_frequency_hz: float
    warmstart_speedup: float
    total_duration_s: float


def cartpole_swingup_benchmark(  # noqa: PLR0913 -- benchmark problem factory parameters
    N: int = 25,
    dt: float = 0.05,
    *,
    x0: Sequence[float] | jax.Array = (0.0, 0.01, 0.0, 0.0),
    xf: Sequence[float] | jax.Array = (0.0, np.pi, 0.0, 0.0),
    u_bound: float = 20.0,
    x_pos_bound: float = 0.4,
) -> tuple[Problem, MPCState, dict[str, Any]]:
    """Build the underactuated Cartpole swing-up benchmark problem with state and control limits.

    Parameters
    ----------
    N : int, optional
        Horizon length in knot points. Defaults to 25.
    dt : float, optional
        Time step duration in seconds. Defaults to 0.05.
    x0 : Sequence[float] | jax.Array, optional
        Initial state [p, theta, p_dot, theta_dot]. Defaults to (0.0, 0.01, 0.0, 0.0).
    xf : Sequence[float] | jax.Array, optional
        Goal state. Defaults to (0.0, np.pi, 0.0, 0.0).
    u_bound : float, optional
        Control actuator limit |u| <= u_bound. Defaults to 20.0.
    x_pos_bound : float, optional
        Cart position limit |p| <= x_pos_bound. Defaults to 0.4, tight enough that the swing-up
        drives the cart onto the limit rather than merely satisfying it.

    Returns
    -------
    tuple[Problem, MPCState, dict[str, Any]]
        Transcribed Problem instance, initial MPCState, and benchmark metadata dictionary.
    """
    model = Cartpole()
    n, m = model.n, model.m

    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    xf_arr = jnp.asarray(xf, dtype=jnp.float64)

    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, N=N)

    x_min = [-x_pos_bound, -np.inf, -np.inf, -np.inf]
    x_max = [x_pos_bound, np.inf, np.inf, np.inf]

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(StateBound(n=n, m=m, x_min=x_min, x_max=x_max), range(N))
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-u_bound], u_max=[u_bound]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf_arr), N - 1)

    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())
    state = MPCState.initial(prob, x0=x0_arr, dt=dt, xf=xf_arr)

    info = {
        "name": "cartpole_swingup",
        "description": "Underactuated Cartpole swing-up with bounded actuation and cart position limits",
        "x0": x0_arr,
        "xf": xf_arr,
        "dt": dt,
        "N": N,
        "x_pos_bound": x_pos_bound,
    }
    return prob, state, info


def quadrotor_obstacle_benchmark(  # noqa: PLR0913 -- benchmark problem factory parameters
    N: int = 25,
    dt: float = 0.05,
    *,
    x0: Sequence[float] | jax.Array = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    xf: Sequence[float] | jax.Array = (3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    obstacles: Sequence[tuple[float, float, float, float]] = ((1.5, 1.5, 1.5, 0.5),),
    u_max: float = 10.0,
) -> tuple[Problem, MPCState, dict[str, Any]]:
    """Build the Quadrotor obstacle avoidance benchmark problem with attitude tracking on SO(3).

    Parameters
    ----------
    N : int, optional
        Horizon length in knot points. Defaults to 25.
    dt : float, optional
        Time step duration in seconds. Defaults to 0.05.
    x0 : Sequence[float] | jax.Array, optional
        Initial state [r(3), q(4), v(3), omega(3)]. Defaults to origin with identity attitude.
    xf : Sequence[float] | jax.Array, optional
        Goal state. Defaults to [3, 3, 3] position with identity attitude.
    obstacles : Sequence[tuple[float, float, float, float]], optional
        Spherical keep-out zones (xc, yc, zc, radius). Defaults to ((1.5, 1.5, 1.5, 0.5),).
    u_max : float, optional
        Maximum motor rotor thrust force limit. Defaults to 10.0.

    Returns
    -------
    tuple[Problem, MPCState, dict[str, Any]]
        Transcribed Problem instance, initial MPCState, and benchmark metadata dictionary.
    """
    model = Quadrotor()
    n, m = model.n, model.m

    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    xf_arr = jnp.asarray(xf, dtype=jnp.float64)

    Q_stage = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    R_stage = jnp.array([0.01, 0.01, 0.01, 0.01])
    Q_term = jnp.array([100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    stage_cost = QuatGeodesicCost(Q=Q_stage, R=R_stage, q_ref=xf_arr[3:7], w=10.0, qind=(3, 4, 5, 6), m=m)
    term_cost = QuatGeodesicCost(Q=Q_term, q_ref=xf_arr[3:7], w=1000.0, qind=(3, 4, 5, 6), terminal=True)
    obj = Objective(stage_cost=stage_cost, terminal_cost=term_cost, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[0.0] * m, u_max=[u_max] * m), range(N - 1))
    if obstacles:
        obs_xc = [obs[0] for obs in obstacles]
        obs_yc = [obs[1] for obs in obstacles]
        obs_zc = [obs[2] for obs in obstacles]
        obs_r = [obs[3] for obs in obstacles]
        cl.add_constraint(SphereConstraint(n=n, m=m, xc=obs_xc, yc=obs_yc, zc=obs_zc, radius=obs_r), range(1, N))

    non_quat_inds = [0, 1, 2, 7, 8, 9, 10, 11, 12]
    cl.add_constraint(GoalConstraint(n=n, xf=xf_arr[jnp.array(non_quat_inds)], inds=non_quat_inds), N - 1)
    cl.add_constraint(QuatVecEq(n=n, qf=xf_arr[3:7], qind=(3, 4, 5, 6)), N - 1)

    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    # Build sensible initial trajectory guess
    X_init = jnp.linspace(x0_arr, xf_arr, N)
    u_hover = float(model.mass * 9.81 / 4.0)
    U_init = jnp.full((N - 1, m), u_hover, dtype=jnp.float64)
    dt_arr = jnp.full((N - 1,), dt, dtype=jnp.float64)
    t_init = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dt_arr)])
    init_traj = Trajectory(X=X_init, U=U_init, t=t_init, dt=dt_arr)

    state = MPCState.initial(prob, x0=x0_arr, dt=dt, xf=xf_arr, initial_trajectory=init_traj)

    info = {
        "name": "quadrotor_obstacle_avoidance",
        "description": "SO(3) attitude tracking Quadrotor navigating around spherical keep-out zones",
        "x0": x0_arr,
        "xf": xf_arr,
        "dt": dt,
        "N": N,
        "obstacles": obstacles,
    }
    return prob, state, info


def dubins_corridor_benchmark(  # noqa: PLR0913 -- benchmark problem factory parameters
    N: int = 25,
    dt: float = 0.1,
    *,
    x0: Sequence[float] | jax.Array = (0.0, 0.0, 0.0),
    xf: Sequence[float] | jax.Array = (2.0, 0.0, 0.0),
    y_corridor_bound: float = 0.5,
    y_ref_bulge: float = 1.0,
    v_max: float = 2.0,
    omega_max: float = 1.5,
) -> tuple[Problem, MPCState, dict[str, Any]]:
    """Build the nonholonomic Dubins car benchmark problem with corridor constraints and tracking objective.

    Parameters
    ----------
    N : int, optional
        Horizon length in knot points. Defaults to 25.
    dt : float, optional
        Time step duration in seconds. Defaults to 0.1.
    x0 : Sequence[float] | jax.Array, optional
        Initial state [x, y, theta]. Defaults to (0.0, 0.0, 0.0).
    xf : Sequence[float] | jax.Array, optional
        Goal state. Defaults to (2.0, 0.0, 0.0).
    y_corridor_bound : float, optional
        Corridor lateral bound |y| <= y_corridor_bound. Defaults to 0.5.
    y_ref_bulge : float, optional
        Peak lateral offset of the tracking reference, y_ref = y_ref_bulge * sin(pi * s).
        Defaults to 1.0, which exceeds the corridor and so pulls the tracked trajectory onto
        the lateral bound instead of leaving it inert. Zero gives a reference straight down
        the corridor centreline, which no lateral bound can bind on.
    v_max : float, optional
        Maximum linear velocity. Defaults to 2.0.
    omega_max : float, optional
        Maximum angular turning rate. Defaults to 1.5.

    Returns
    -------
    tuple[Problem, MPCState, dict[str, Any]]
        Transcribed Problem instance, initial MPCState, and benchmark metadata dictionary.
    """
    model = DubinsCar()
    n, m = model.n, model.m

    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    xf_arr = jnp.asarray(xf, dtype=jnp.float64)

    t_arr = jnp.linspace(0.0, (N - 1) * dt, N)
    dt_arr = jnp.full((N - 1,), dt, dtype=jnp.float64)

    # A half sine vanishing at both ends, so the reference still starts at x0 and ends at xf
    # while bulging past the corridor in between.
    s_arc = jnp.linspace(0.0, 1.0, N)
    y_ref = y_ref_bulge * jnp.sin(jnp.pi * s_arc)
    X_ref = (
        jnp.zeros((N, n), dtype=jnp.float64)
        .at[:, 0]
        .set(jnp.linspace(float(x0_arr[0]), float(xf_arr[0]), N))
        .at[:, 1]
        .set(y_ref)
    )
    v_ref = float((xf_arr[0] - x0_arr[0]) / ((N - 1) * dt))
    U_ref = jnp.ones((N - 1, m), dtype=jnp.float64) * jnp.array([v_ref, 0.0])
    ref_traj = Trajectory(X=X_ref, U=U_ref, t=t_arr, dt=dt_arr)

    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Qf = jnp.diag(jnp.array([100.0, 100.0, 10.0]))
    obj = TrackingObjective(Q=Q, R=R, trajectory=ref_traj, Qf=Qf)

    x_min = [-0.5, -y_corridor_bound, -np.inf]
    x_max = [float(xf_arr[0]) + 1.0, y_corridor_bound, np.inf]

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(StateBound(n=n, m=m, x_min=x_min, x_max=x_max), range(N))
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[0.0, -omega_max], u_max=[v_max, omega_max]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf_arr), N - 1)

    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())
    state = MPCState.initial(prob, x0=x0_arr, dt=dt, reference=ref_traj, initial_trajectory=ref_traj)

    info = {
        "name": "dubins_corridor_tracking",
        "description": "Nonholonomic Dubins car with corridor bounds and trajectory tracking objective",
        "x0": x0_arr,
        "xf": xf_arr,
        "dt": dt,
        "N": N,
        "corridor_bound": y_corridor_bound,
        "y_ref_bulge": y_ref_bulge,
    }
    return prob, state, info


def measure_transcription_setup(
    problem: Problem,
    x0: jax.Array,
    *,
    dt: float | jax.Array = 0.05,
    num_runs: int = 20,
) -> float:
    """Measure the time in seconds to assemble sparsity patterns, bound vectors, and transcription structures."""
    N = int(problem.N)
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N - 1,))
    ipopt = Ipopt()

    # Warm up
    _ = primal_bounds(problem)
    _ = constraint_bounds(problem)
    _ = ipopt.transcription_callback(problem, x0_arr, 0.0, dt_arr)

    t_start = time.perf_counter()
    for _ in range(num_runs):
        _ = primal_bounds(problem)
        _ = constraint_bounds(problem)
        _ = ipopt.transcription_callback(problem, x0_arr, 0.0, dt_arr)
    t_end = time.perf_counter()

    return (t_end - t_start) / num_runs


def measure_derivative_evaluations(
    problem: Problem,
    state: MPCState,
    *,
    num_evals: int = 30,
) -> dict[str, float]:
    """Measure per-iteration derivative evaluation times in seconds."""
    z = state.Z
    t0 = state.t0
    dt = state.dt
    xf = state.xf
    bc = state.bc
    x0 = state.x0
    lam = state.lam

    # Warmup JIT compilation
    _ = eval_grad_f(problem, z, t0, dt, bc).block_until_ready()
    _ = eval_jac_g(problem, z, x0, t0, dt, xf=xf).block_until_ready()
    _ = eval_h(problem, z, t0=t0, dt=dt, obj_factor=1.0, lam=lam, bc=bc).block_until_ready()

    # Gradient timing
    t0_time = time.perf_counter()
    for _ in range(num_evals):
        _ = eval_grad_f(problem, z, t0, dt, bc).block_until_ready()
    t_grad = (time.perf_counter() - t0_time) / num_evals

    # Jacobian timing
    t0_time = time.perf_counter()
    for _ in range(num_evals):
        _ = eval_jac_g(problem, z, x0, t0, dt, xf=xf).block_until_ready()
    t_jac = (time.perf_counter() - t0_time) / num_evals

    # Hessian timing
    t0_time = time.perf_counter()
    for _ in range(num_evals):
        _ = eval_h(problem, z, t0=t0, dt=dt, obj_factor=1.0, lam=lam, bc=bc).block_until_ready()
    t_hess = (time.perf_counter() - t0_time) / num_evals

    t_total_deriv = t_grad + t_jac + t_hess

    return {
        "grad_f": t_grad,
        "jac_g": t_jac,
        "hess_l": t_hess,
        "total_derivative": t_total_deriv,
    }


class SolveTiming(NamedTuple):
    """Wall-clock timings of one solver on one problem over repeated warm calls.

    Parameters
    ----------
    first_call_time_s : float
        Duration of the first, discarded call in seconds. It includes `jax.jit` compilation for a
        Native Solver and extension warmup for a Backend, so it is not a cold-start-from-default-
        guess measurement -- `ClosedLoopStats.warmstart_speedup` is the one that means that.
    median_time_s : float
        Median duration of the timed calls in seconds.
    min_time_s : float
        Shortest timed call in seconds, the cleanest signal for a compute-bound solve.
    """

    first_call_time_s: float
    median_time_s: float
    min_time_s: float


def measure_solver_runtime(
    problem: Problem,
    state: MPCState,
    solver: Solver,
    *,
    n_repeats: int = 5,
) -> tuple[SolverResult, SolveTiming]:
    """Time `n_repeats` warm `solver.solve` calls after one discarded call, returning the last result.

    The discarded call is reported as `SolveTiming.first_call_time_s` rather than thrown away:
    it is free, and for a Native Solver it is the compile the jit cache exists to amortize.
    """
    t_start = time.perf_counter()
    res = solver.solve(problem, state)
    t_first = time.perf_counter() - t_start

    durations: list[float] = []
    for _ in range(n_repeats):
        t_start = time.perf_counter()
        res = solver.solve(problem, state)
        durations.append(time.perf_counter() - t_start)

    return res, SolveTiming(
        first_call_time_s=t_first,
        median_time_s=statistics.median(durations),
        min_time_s=min(durations),
    )


def _measure_warmstart_pair(problem: Problem, state: MPCState, solver: Solver) -> tuple[float, float]:
    """Time one MPC step solved warm-started and cold, returning (t_warm, t_cold) in seconds.

    Both solves face the same problem at the same step and differ only in the initial guess, so
    the ratio isolates warm starting. The cold guess is the default one, x0 repeated with zero
    controls.
    """
    cold_state = MPCState.initial(
        problem,
        x0=state.x0,
        t0=state.t0,
        xf=state.xf,
        dt=state.dt,
    )

    t_start = time.perf_counter()
    _ = solver.solve(problem, state)
    t_warm = time.perf_counter() - t_start

    t_start = time.perf_counter()
    _ = solver.solve(problem, cold_state)
    t_cold = time.perf_counter() - t_start

    return t_warm, t_cold


def measure_closed_loop_mpc(
    problem: Problem,
    initial_state: MPCState,
    solver: Solver,
    *,
    num_steps: int = 20,
) -> ClosedLoopStats:
    """Measure closed-loop MPC sustained frequency, latency jitter, and warm-start speedup.

    Every timed section is preceded by a discarded solve, so the statistics describe the solver
    rather than JIT compilation: the first step of an uncompiled loop runs two orders of
    magnitude slower than the rest and would otherwise set the mean, the jitter and the p95.
    """
    # 1. Seed the loop, discarding a compilation solve first
    _ = solver.solve(problem, initial_state)
    cold_res = solver.solve(problem, initial_state)

    # 2. Receding horizon warm-started loop
    dt_val = float(initial_state.dt[0]) if initial_state.dt.ndim > 0 else float(initial_state.dt)
    model = problem.model

    curr_state = dataclasses.replace(initial_state, Z=cold_res.Z)

    durations: list[float] = []
    t_loop_start = time.perf_counter()

    for _ in range(num_steps):
        curr_state = curr_state.shift(dt_val)
        t_step_start = time.perf_counter()
        solved_state = problem.solve(curr_state, solver=solver)
        t_step_dur = time.perf_counter() - t_step_start
        durations.append(t_step_dur)

        u0 = solved_state.controls[0]
        t0_curr = float(curr_state.t0)
        x_next = model.discrete_dynamics(curr_state.x0, u0, t0_curr, dt_val)
        curr_state = solved_state.with_measurement(x_next, t0_curr + dt_val)

    t_total_loop = time.perf_counter() - t_loop_start
    durations_arr = np.asarray(durations, dtype=np.float64)

    t_warm_step, t_cold_step = _measure_warmstart_pair(problem, curr_state, solver)

    mean_lat = float(np.mean(durations_arr))
    std_lat = float(np.std(durations_arr))
    min_lat = float(np.min(durations_arr))
    max_lat = float(np.max(durations_arr))
    med_lat = float(np.median(durations_arr))
    p95_lat = float(np.percentile(durations_arr, 95))
    p99_lat = float(np.percentile(durations_arr, 99))
    freq = 1.0 / mean_lat if mean_lat > 0 else 0.0
    speedup = t_cold_step / t_warm_step if t_warm_step > 0 else 1.0

    return ClosedLoopStats(
        num_steps=num_steps,
        durations_s=durations_arr,
        mean_latency_s=mean_lat,
        std_latency_s=std_lat,
        min_latency_s=min_lat,
        max_latency_s=max_lat,
        median_latency_s=med_lat,
        p95_latency_s=p95_lat,
        p99_latency_s=p99_lat,
        sustained_frequency_hz=freq,
        warmstart_speedup=speedup,
        total_duration_s=t_total_loop,
    )


def _model_name(problem: Problem) -> str:
    """Name the problem's model for a table header, unwrapping the integrator around it.

    `problem.model` is usually a `DiscretizedDynamics` holding the model the caller recognises,
    and a header reading "DiscretizedDynamics" names every problem equally badly.
    """
    model = problem.model
    inner = getattr(model, "continuous_dynamics", None)
    return type(inner if inner is not None else model).__name__


def _labelled(solvers: Sequence[Solver] | Mapping[str, Solver]) -> list[tuple[str, Solver]]:
    """Pair each solver with its row label: a mapping's key, or the class name for a sequence."""
    if isinstance(solvers, Mapping):
        return list(solvers.items())
    return [(type(s).__name__, s) for s in solvers]


def _score(problem: Problem, state: MPCState, Z: jax.Array) -> tuple[float, float]:
    """Recompute (cost, maximum constraint violation) of a solved `Z` under the transcription.

    Every row is scored here rather than read off the solver's own result, because the solvers do
    not agree on what either number means: ALTRO reports its augmented Lagrangian `c_max`, PN its
    active-set residual, the Backends the transcription's violation including Defects, and iLQR
    evaluates a retargeted objective. One definition applied to each returned Primal Vector is
    what makes a column comparable down its length.
    """
    N = int(problem.N)
    Z_arr = jnp.asarray(Z, dtype=jnp.float64)
    dt_arr = jnp.broadcast_to(jnp.asarray(state.dt, dtype=jnp.float64), (N - 1,))

    cost = float(eval_f(problem, Z_arr, state.t0, dt_arr, state.bc))
    viol = compute_constraint_violation(problem, Z_arr, state.x0, t0=state.t0, dt=dt_arr, xf=state.xf)
    return cost, viol


class SolverRow(NamedTuple):
    """One solver's outcome and timing in a `SolverComparison`.

    Parameters
    ----------
    solver : str
        Row label: the solver's class name, or the caller's key when solvers were passed as a
        mapping.
    success : bool
        The solver's own convergence flag.
    iterations : int
        Iterations the solver reported.
    cost : float
        Objective value recomputed by the harness at the returned Primal Vector, not the solver's
        self-reported cost.
    constraint_violation : float
        Maximum violation recomputed by the harness at the returned Primal Vector, not the
        solver's self-reported violation. A solver that ignores constraints shows it here.
    timing : SolveTiming
        First-call, median, and minimum warm solve durations.
    linearizing : bool
        Whether this solver solved a single convex subproblem about the Operating Point rather
        than the nonlinear problem, which makes its time incomparable to the other rows'.
    result : SolverResult
        The last returned result, kept so a caller can read the solver's own numbers.
    """

    solver: str
    success: bool
    iterations: int
    cost: float
    constraint_violation: float
    timing: SolveTiming
    linearizing: bool
    result: SolverResult


class SolverComparison(NamedTuple):
    """The table `compare_solvers` returns: one `SolverRow` per solver, in the order given.

    No winner is declared. Which of speed, feasibility and cost matters is the caller's problem,
    and a solver that is fastest because it ignored a constraint is not the answer to it.

    Parameters
    ----------
    model : str
        Class name of the problem's dynamics model, for the table header.
    n_repeats : int
        Number of timed warm calls behind each row's median and minimum.
    rows : tuple[SolverRow, ...]
        One row per solver, in the order the solvers were given.
    """

    model: str
    n_repeats: int
    rows: tuple[SolverRow, ...]

    def format_table(self) -> str:
        """Render the comparison as a plain-text table, returning it rather than printing it."""
        header = (
            f"{'solver':<12} {'ok':>5} {'iters':>6} {'cost':>14} {'violation':>12} "
            f"{'first (ms)':>12} {'median (ms)':>12} {'min (ms)':>10}"
        )
        lines = [f"{self.model}, {self.n_repeats} warm calls", header, "-" * len(header)]
        for r in self.rows:
            label = r.solver + (" *" if r.linearizing else "")
            lines.append(
                f"{label:<12} {r.success!s:>5} {r.iterations:>6} {r.cost:>14.6g} "
                f"{r.constraint_violation:>12.3e} {r.timing.first_call_time_s * 1e3:>12.1f} "
                f"{r.timing.median_time_s * 1e3:>12.2f} {r.timing.min_time_s * 1e3:>10.2f}"
            )
        if any(r.linearizing for r in self.rows):
            lines.append("* one convex solve about the Operating Point, not the nonlinear problem")
        return "\n".join(lines)


def compare_solvers(
    problem: Problem,
    state: MPCState,
    solvers: Sequence[Solver] | Mapping[str, Solver],
    *,
    n_repeats: int = 5,
) -> SolverComparison:
    """Solve one problem with each solver and tabulate their cost, feasibility, and warm solve time.

    The solvers arrive already configured and their options are never rewritten: mapping one
    tolerance onto `SolverOptions.constraint_tolerance`, Ipopt's `tol` and OSQP's `eps_abs` would
    be wrong in a different way for each. Solvers compared at different tolerances compare
    tolerances, which is the caller's to get right.

    Errors propagate: a solver that raises on this problem stops the comparison rather than
    becoming a failed row.

    Parameters
    ----------
    solvers : Sequence[Solver] | Mapping[str, Solver]
        Configured solver instances. Pass a mapping to label rows yourself, which is what
        distinguishes two differently configured instances of the same solver.
    n_repeats : int, optional
        Timed warm calls per solver, after one discarded call. Defaults to 5.
    """
    rows = []
    for label, solver in _labelled(solvers):
        res, timing = measure_solver_runtime(problem, state, solver, n_repeats=n_repeats)
        cost, viol = _score(problem, state, res.Z)
        rows.append(
            SolverRow(
                solver=label,
                success=res.success,
                iterations=res.iterations,
                cost=cost,
                constraint_violation=viol,
                timing=timing,
                linearizing=isinstance(solver, _LINEARIZING),
                result=res,
            )
        )

    return SolverComparison(model=_model_name(problem), n_repeats=n_repeats, rows=tuple(rows))


class ClosedLoopRow(NamedTuple):
    """One solver's closed-loop MPC statistics in a `ClosedLoopComparison`.

    Parameters
    ----------
    solver : str
        Row label, as in `SolverRow.solver`.
    linearizing : bool
        As in `SolverRow.linearizing`.
    stats : ClosedLoopStats
        Latency, jitter, and warm-start statistics over the receding horizon run.
    """

    solver: str
    linearizing: bool
    stats: ClosedLoopStats


class ClosedLoopComparison(NamedTuple):
    """The table `compare_solvers_closed_loop` returns: one `ClosedLoopRow` per solver.

    Parameters
    ----------
    model : str
        Class name of the problem's dynamics model, for the table header.
    num_steps : int
        Receding horizon steps each solver ran.
    rows : tuple[ClosedLoopRow, ...]
        One row per solver, in the order the solvers were given.
    """

    model: str
    num_steps: int
    rows: tuple[ClosedLoopRow, ...]

    def format_table(self) -> str:
        """Render the closed-loop comparison as a plain-text table, returning it rather than printing it."""
        header = (
            f"{'solver':<12} {'mean (ms)':>10} {'median (ms)':>12} {'p95 (ms)':>10} "
            f"{'p99 (ms)':>10} {'Hz':>8} {'warmstart':>10}"
        )
        lines = [f"{self.model}, {self.num_steps} closed-loop steps", header, "-" * len(header)]
        for r in self.rows:
            st = r.stats
            label = r.solver + (" *" if r.linearizing else "")
            lines.append(
                f"{label:<12} {st.mean_latency_s * 1e3:>10.2f} {st.median_latency_s * 1e3:>12.2f} "
                f"{st.p95_latency_s * 1e3:>10.2f} {st.p99_latency_s * 1e3:>10.2f} "
                f"{st.sustained_frequency_hz:>8.1f} {st.warmstart_speedup:>9.2f}x"
            )
        if any(r.linearizing for r in self.rows):
            lines.append("* one convex solve about the Operating Point, not the nonlinear problem")
        return "\n".join(lines)


def compare_solvers_closed_loop(
    problem: Problem,
    state: MPCState,
    solvers: Sequence[Solver] | Mapping[str, Solver],
    *,
    num_steps: int = 15,
) -> ClosedLoopComparison:
    """Run a receding horizon MPC loop per solver and tabulate latency, jitter, and warm-start gain.

    Separate from `compare_solvers` rather than a mode of it: the two share no columns, since
    latency percentiles over a moving problem and repeated solves of one fixed problem answer
    different questions. This one costs `num_steps` solves per solver.
    """
    rows = [
        ClosedLoopRow(
            solver=label,
            linearizing=isinstance(solver, _LINEARIZING),
            stats=measure_closed_loop_mpc(problem, state, solver, num_steps=num_steps),
        )
        for label, solver in _labelled(solvers)
    ]
    return ClosedLoopComparison(model=_model_name(problem), num_steps=num_steps, rows=tuple(rows))
