from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.benchmarks import (
    BenchmarkSuiteResult,
    ClosedLoopStats,
    TimingBreakdown,
    cartpole_swingup_benchmark,
    dubins_corridor_benchmark,
    measure_closed_loop_mpc,
    measure_derivative_evaluations,
    measure_solver_runtime,
    measure_transcription_setup,
    quadrotor_obstacle_benchmark,
    run_all_benchmarks,
    run_benchmark,
)
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.layout import _z_to_trajectory

# Every test here runs at least one full benchmark solve, and the file is the slowest in
# the suite by a wide margin. The dev loop is `-m "not slow and not benchmark"`.
pytestmark = pytest.mark.slow


def _bound_activity(
    res: IpoptResult,
    N: int,
    n: int,
    m: int,
    index: int,
    bound: float,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the values and bound duals of state `index` at the knots pressed onto +/- `bound`.

    A constraint that is merely satisfied and one that shapes the solution look identical in the
    trajectory alone, so the multiplier is what separates them: `res.mu` carries the signed
    variable-bound duals ``mult_x_U - mult_x_L`` laid out like Z, and only an active bound
    carries a nonzero one.
    """
    X = np.asarray(res.trajectory.X)
    mu_X, _ = _z_to_trajectory(jnp.asarray(res.mu), N, n, m)
    on_bound = np.abs(np.abs(X[:, index]) - bound) <= tol
    return X[on_bound, index], np.asarray(mu_X)[on_bound, index]


def test_cartpole_swingup_benchmark_drives_the_cart_onto_its_position_limit() -> None:
    """Verify the Cartpole swing-up solves to optimality with the cart position limit active."""
    pytest.importorskip("cyipopt")

    prob, state, info = cartpole_swingup_benchmark(N=25, dt=0.05, u_bound=20.0)
    assert info["name"] == "cartpole_swingup"
    assert prob.model.n == 4
    assert prob.model.m == 1

    p_bound = float(info["x_pos_bound"])
    opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    res = Ipopt(options=opts).solve(prob, state)

    assert res.success is True
    assert res.status in {0, 1}
    assert res.cost > 0.0

    X = np.asarray(res.trajectory.X)
    U = np.asarray(res.trajectory.U)

    # Cart position limit
    assert np.all(X[:, 0] >= -p_bound - 1e-4)
    assert np.all(X[:, 0] <= p_bound + 1e-4)

    # ... and the swing-up is shaped by it rather than merely permitted by it: at least one knot
    # sits on the limit carrying a nonzero multiplier, so relaxing it would change the optimum.
    on_bound, duals = _bound_activity(res, prob.N, prob.model.n, prob.model.m, index=0, bound=p_bound)
    assert len(on_bound) > 0, f"cart never reaches |p| = {p_bound}; max |p| was {np.max(np.abs(X[:, 0])):.4f}"
    assert np.max(np.abs(duals)) > 1.0, f"cart position limit is inert: bound duals {duals}"

    # Actuation limits
    assert np.all(U >= -20.0 - 1e-4)
    assert np.all(U <= 20.0 + 1e-4)

    # Goal reaching
    xf_target = np.asarray(info["xf"])
    np.testing.assert_allclose(X[-1], xf_target, atol=1e-3)


def test_quadrotor_obstacle_benchmark_solves_to_optimality() -> None:
    """Verify Quadrotor navigates around spherical keep-out zones while tracking attitude reference."""
    pytest.importorskip("cyipopt")

    obs = ((1.5, 1.5, 1.5, 0.5),)
    prob, state, info = quadrotor_obstacle_benchmark(
        N=25,
        dt=0.05,
        obstacles=obs,
        u_max=10.0,
    )
    assert info["name"] == "quadrotor_obstacle_avoidance"
    assert prob.model.n == 13
    assert prob.model.m == 4

    opts = {"max_iter": 500, "tol": 1e-6, "print_level": 0}
    res = Ipopt(options=opts).solve(prob, state)

    assert res.success is True
    assert res.status in {0, 1}
    assert res.cost > 0.0

    X = np.asarray(res.trajectory.X)
    U = np.asarray(res.trajectory.U)

    # Control bounds
    assert np.all(U >= 0.0 - 1e-4)
    assert np.all(U <= 10.0 + 1e-4)

    # Spherical obstacle clearance at every knot point
    pos = X[:, :3]
    for xc, yc, zc, r in obs:
        center = np.array([xc, yc, zc])
        dists = np.linalg.norm(pos - center, axis=1)
        # Distance must be at least obstacle radius (within small numerical tolerance)
        assert np.all(dists >= r - 1e-3), f"Obstacle violation detected: min distance {np.min(dists)} < {r}"

    # Attitude tracking at goal
    xf_target = np.asarray(info["xf"])
    np.testing.assert_allclose(X[-1, :3], xf_target[:3], atol=1e-3)  # Position reached
    # Quaternion alignment: |q_opt . q_ref| ~= 1
    q_opt = X[-1, 3:7]
    q_ref = xf_target[3:7]
    assert abs(np.dot(q_opt, q_ref)) >= 0.99


def test_dubins_corridor_benchmark_pins_the_tracked_trajectory_to_the_corridor_wall() -> None:
    """Verify the Dubins car rides the lateral corridor bound its tracking reference bulges past."""
    pytest.importorskip("cyipopt")

    prob, state, info = dubins_corridor_benchmark(
        N=25,
        dt=0.1,
        y_corridor_bound=0.5,
        v_max=2.0,
        omega_max=1.5,
    )
    assert info["name"] == "dubins_corridor_tracking"
    assert prob.model.n == 3
    assert prob.model.m == 2
    y_bound = float(info["corridor_bound"])
    assert float(info["y_ref_bulge"]) > y_bound, "reference must leave the corridor for it to bind"

    opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    res = Ipopt(options=opts).solve(prob, state)

    assert res.success is True
    assert res.status in {0, 1}

    X = np.asarray(res.trajectory.X)
    U = np.asarray(res.trajectory.U)

    # Lateral corridor bounds |y| <= 0.5
    assert np.all(X[:, 1] >= -y_bound - 1e-4)
    assert np.all(X[:, 1] <= y_bound + 1e-4)

    # The tracking objective pulls y out to the bulge and the corridor stops it, so the wall is
    # load-bearing: several knots sit on it with nonzero multipliers.
    on_bound, duals = _bound_activity(res, prob.N, prob.model.n, prob.model.m, index=1, bound=y_bound)
    assert len(on_bound) >= 2, f"corridor never binds; max |y| was {np.max(np.abs(X[:, 1])):.4f}"
    assert np.max(np.abs(duals)) > 1.0, f"corridor bound is inert: bound duals {duals}"

    # Control bounds
    assert np.all(U[:, 0] >= 0.0 - 1e-4)
    assert np.all(U[:, 0] <= 2.0 + 1e-4)
    assert np.all(U[:, 1] >= -1.5 - 1e-4)
    assert np.all(U[:, 1] <= 1.5 + 1e-4)

    # Goal reached
    xf_target = np.asarray(info["xf"])
    np.testing.assert_allclose(X[-1], xf_target, atol=1e-3)


def test_timing_breakdown_measurements() -> None:
    """Verify timing measurement functions report positive durations for setup, derivatives, and solver."""
    pytest.importorskip("cyipopt")

    prob, state, _ = cartpole_swingup_benchmark(N=25, dt=0.05)

    # 1. Setup timing
    t_setup = measure_transcription_setup(prob, state.x0, num_runs=5)
    assert isinstance(t_setup, float)
    assert t_setup > 0.0

    # 2. Derivative timing
    deriv_times = measure_derivative_evaluations(prob, state, num_evals=5)
    assert "grad_f" in deriv_times
    assert "jac_g" in deriv_times
    assert "hess_l" in deriv_times
    assert "total_derivative" in deriv_times
    assert deriv_times["grad_f"] > 0.0
    assert deriv_times["jac_g"] > 0.0
    assert deriv_times["hess_l"] > 0.0
    assert deriv_times["total_derivative"] >= deriv_times["grad_f"] + deriv_times["jac_g"]

    # 3. Solver runtime
    res, t_solve = measure_solver_runtime(prob, state, options={"max_iter": 50, "print_level": 0})
    assert res.success is True
    assert isinstance(t_solve, float)
    assert t_solve > 0.0


def test_closed_loop_mpc_measurement_and_jitter() -> None:
    """Verify closed-loop MPC reports sustained frequency, latency jitter, and warm-start speedup."""
    pytest.importorskip("cyipopt")

    prob, state, _ = cartpole_swingup_benchmark(N=25, dt=0.05)
    num_steps = 10

    stats: ClosedLoopStats = measure_closed_loop_mpc(
        prob,
        state,
        num_steps=num_steps,
        solver_options={"max_iter": 50, "tol": 1e-4, "print_level": 0},
    )

    assert stats.num_steps == num_steps
    assert len(stats.durations_s) == num_steps
    assert stats.mean_latency_s > 0.0
    assert stats.std_latency_s >= 0.0  # Latency jitter is non-negative
    assert stats.min_latency_s <= stats.median_latency_s <= stats.max_latency_s
    assert stats.p95_latency_s >= stats.median_latency_s
    assert stats.p99_latency_s >= stats.p95_latency_s
    assert stats.sustained_frequency_hz > 0.0
    assert stats.warmstart_speedup > 1.0  # Warm-start speedup is quantified and > 1
    assert stats.total_duration_s > 0.0


def test_run_benchmark_suite() -> None:
    """Verify run_benchmark executes end-to-end and returns structured BenchmarkSuiteResult."""
    pytest.importorskip("cyipopt")

    result: BenchmarkSuiteResult = run_benchmark(
        dubins_corridor_benchmark,
        num_closed_loop_steps=5,
        solver_options={"max_iter": 50, "tol": 1e-4, "print_level": 0},
    )

    assert result.name == "dubins_corridor_tracking"
    assert result.solve_result.success is True
    assert isinstance(result.timing, TimingBreakdown)
    assert result.timing.transcription_setup_time_s > 0.0
    assert result.timing.solver_runtime_s > 0.0
    assert isinstance(result.closed_loop, ClosedLoopStats)
    assert result.closed_loop.sustained_frequency_hz > 0.0


def test_run_all_benchmarks() -> None:
    """Verify run_all_benchmarks executes all three problems successfully."""
    pytest.importorskip("cyipopt")

    results = run_all_benchmarks(num_closed_loop_steps=5)
    assert set(results.keys()) == {"cartpole", "quadrotor", "dubins"}
    for name, res in results.items():
        assert res.solve_result.success is True, f"Benchmark {name} solve failed"
        assert res.closed_loop.sustained_frequency_hz > 0.0


@pytest.mark.benchmark
def test_benchmark_cartpole_solve(benchmark: Any) -> None:
    """Pytest-benchmark performance test for Cartpole swing-up solve."""
    pytest.importorskip("cyipopt")

    prob, state, _ = cartpole_swingup_benchmark(N=25, dt=0.05)
    opts = {"max_iter": 100, "tol": 1e-6, "print_level": 0}

    res = benchmark(Ipopt(options=opts).solve, prob, state)
    assert res.success is True


@pytest.mark.benchmark
def test_benchmark_quadrotor_solve(benchmark: Any) -> None:
    """Pytest-benchmark performance test for Quadrotor obstacle avoidance solve."""
    pytest.importorskip("cyipopt")

    prob, state, _ = quadrotor_obstacle_benchmark(N=25, dt=0.05)
    opts = {"max_iter": 500, "tol": 1e-6, "print_level": 0}

    res = benchmark(Ipopt(options=opts).solve, prob, state)
    assert res.success is True


@pytest.mark.benchmark
def test_benchmark_dubins_solve(benchmark: Any) -> None:
    """Pytest-benchmark performance test for Dubins car corridor tracking solve."""
    pytest.importorskip("cyipopt")

    prob, state, _ = dubins_corridor_benchmark(N=25, dt=0.1)
    opts = {"max_iter": 100, "tol": 1e-6, "print_level": 0}

    res = benchmark(Ipopt(options=opts).solve, prob, state)
    assert res.success is True
