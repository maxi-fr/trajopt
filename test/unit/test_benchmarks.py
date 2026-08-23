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
from trajopt.transcription.ipopt import solve_ipopt


def test_cartpole_swingup_benchmark_solves_to_optimality() -> None:
    """Verify underactuated Cartpole swing-up benchmark solves to optimality respecting bounds."""
    pytest.importorskip("cyipopt")

    prob, state, info = cartpole_swingup_benchmark(
        N=25,
        dt=0.05,
        u_bound=20.0,
        x_pos_bound=2.0,
    )
    assert info["name"] == "cartpole_swingup"
    assert prob.model.n == 4
    assert prob.model.m == 1

    opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    res = solve_ipopt(prob, state, options=opts)

    assert res.success is True
    assert res.status in {0, 1}
    assert res.cost > 0.0

    X = np.asarray(res.trajectory.X)
    U = np.asarray(res.trajectory.U)

    # Cart position limit
    assert np.all(X[:, 0] >= -2.0 - 1e-4)
    assert np.all(X[:, 0] <= 2.0 + 1e-4)

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
    res = solve_ipopt(prob, state, options=opts)

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


def test_dubins_corridor_benchmark_solves_to_optimality() -> None:
    """Verify nonholonomic Dubins car enforces corridor constraints alongside tracking objective."""
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

    opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    res = solve_ipopt(prob, state, options=opts)

    assert res.success is True
    assert res.status in {0, 1}

    X = np.asarray(res.trajectory.X)
    U = np.asarray(res.trajectory.U)

    # Lateral corridor bounds |y| <= 0.5
    assert np.all(X[:, 1] >= -0.5 - 1e-4)
    assert np.all(X[:, 1] <= 0.5 + 1e-4)

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

    prob, state, _ = cartpole_swingup_benchmark(N=20, dt=0.05)

    # 1. Setup timing
    t_setup = measure_transcription_setup(prob, state, num_runs=5)
    assert isinstance(t_setup, float)
    assert t_setup > 0.0

    # 2. Derivative timing
    deriv_times = measure_derivative_evaluations(prob, state, num_evals=5)
    assert "grad_f" in deriv_times
    assert "jac_g" in deriv_times
    assert "hess_L" in deriv_times
    assert "total_derivative" in deriv_times
    assert deriv_times["grad_f"] > 0.0
    assert deriv_times["jac_g"] > 0.0
    assert deriv_times["hess_L"] > 0.0
    assert deriv_times["total_derivative"] >= deriv_times["grad_f"] + deriv_times["jac_g"]

    # 3. Solver runtime
    res, t_solve = measure_solver_runtime(prob, state, options={"max_iter": 50, "print_level": 0})
    assert res.success is True
    assert isinstance(t_solve, float)
    assert t_solve > 0.0


def test_closed_loop_mpc_measurement_and_jitter() -> None:
    """Verify closed-loop MPC reports sustained frequency, latency jitter, and warm-start speedup."""
    pytest.importorskip("cyipopt")

    prob, state, _ = cartpole_swingup_benchmark(N=20, dt=0.05)
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

    res = benchmark(solve_ipopt, prob, state, options=opts)
    assert res.success is True


@pytest.mark.benchmark
def test_benchmark_quadrotor_solve(benchmark: Any) -> None:
    """Pytest-benchmark performance test for Quadrotor obstacle avoidance solve."""
    pytest.importorskip("cyipopt")

    prob, state, _ = quadrotor_obstacle_benchmark(N=25, dt=0.05)
    opts = {"max_iter": 500, "tol": 1e-6, "print_level": 0}

    res = benchmark(solve_ipopt, prob, state, options=opts)
    assert res.success is True


@pytest.mark.benchmark
def test_benchmark_dubins_solve(benchmark: Any) -> None:
    """Pytest-benchmark performance test for Dubins car corridor tracking solve."""
    pytest.importorskip("cyipopt")

    prob, state, _ = dubins_corridor_benchmark(N=25, dt=0.1)
    opts = {"max_iter": 100, "tol": 1e-6, "print_level": 0}

    res = benchmark(solve_ipopt, prob, state, options=opts)
    assert res.success is True
