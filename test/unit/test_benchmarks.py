import pytest

from trajopt.benchmarks import (
    ClosedLoopStats,
    NativeVsIpoptTiming,
    cartpole_swingup_benchmark,
    measure_altro_vs_ipopt,
    measure_closed_loop_mpc,
)

# Every test here builds at least one benchmark problem, and the file is the slowest in the
# suite by a wide margin. The dev loop is `-m "not slow"`.
pytestmark = pytest.mark.slow


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


def test_altro_vs_ipopt_timing_reports_both_solve_durations() -> None:
    """Verify the native-vs-Ipopt comparison times both solvers on the same problem."""
    pytest.importorskip("cyipopt")

    prob, state, _ = cartpole_swingup_benchmark(N=25, dt=0.05)

    timing: NativeVsIpoptTiming = measure_altro_vs_ipopt(
        prob, state, ipopt_options={"max_iter": 200, "tol": 1e-6, "print_level": 0}
    )

    assert timing.ipopt_time_s > 0.0
    assert timing.altro_time_s > 0.0
    assert timing.speedup > 0.0
