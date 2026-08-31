"""Time trajopt's Ipopt transcription against an equivalent CasADi Opti formulation.

Not wired into pytest: run manually with `uv run python scripts/compare_casadi_timing.py`.
Prints setup and solve timing for the three benchmark problems (cartpole, quadrotor, dubins)
side by side; there is no stored baseline or pass/fail threshold, it's a manual timing readout.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "test"))

import jax  # noqa: E402 -- must follow the sys.path setup above
import numpy as np  # noqa: E402 -- must follow the sys.path setup above
from cross_verification.casadi_baseline import (  # noqa: E402 -- must follow the sys.path setup above  # ty: ignore[unresolved-import]
    CasadiProblem,
    build_casadi_from_problem,
)

from trajopt.benchmarks import (  # noqa: E402 -- must follow the sys.path setup above
    cartpole_swingup_benchmark,
    dubins_corridor_benchmark,
    measure_solver_runtime,
    measure_transcription_setup,
    quadrotor_obstacle_benchmark,
)
from trajopt.problem import MPCState, Problem  # noqa: E402 -- must follow the sys.path setup above
from trajopt.transcription.ipopt import Ipopt  # noqa: E402 -- must follow the sys.path setup above

BenchmarkFactory = Callable[..., tuple[Problem, MPCState, dict[str, Any]]]


class ProblemSpec(NamedTuple):
    """One benchmark problem's factory, solver options, and CasADi comparison settings."""

    name: str
    factory: BenchmarkFactory
    factory_kwargs: dict[str, Any]
    solver_opts: dict[str, Any]
    use_state_as_init_guess: bool


PROBLEMS = [
    ProblemSpec(
        "cartpole_swingup",
        cartpole_swingup_benchmark,
        {"N": 25, "dt": 0.05, "u_bound": 20.0},
        {"max_iter": 500, "tol": 1e-10, "print_level": 0},
        use_state_as_init_guess=False,
    ),
    ProblemSpec(
        "quadrotor_obstacle",
        quadrotor_obstacle_benchmark,
        {"N": 25, "dt": 0.05, "obstacles": ((1.5, 1.5, 1.5, 0.5),), "u_max": 10.0},
        {"max_iter": 500, "tol": 1e-8, "print_level": 0},
        use_state_as_init_guess=True,
    ),
    ProblemSpec(
        "dubins_corridor",
        dubins_corridor_benchmark,
        {"N": 25, "dt": 0.1, "y_corridor_bound": 0.5},
        {"max_iter": 500, "tol": 1e-10, "print_level": 0},
        use_state_as_init_guess=True,
    ),
]


def _measure_casadi_setup(problem: Problem, x0: jax.Array, dt: float, *, num_runs: int = 20) -> float:
    """Measure the average time in seconds to build a CasadiProblem from a trajopt Problem."""
    _ = build_casadi_from_problem(problem, x0=x0, dt=dt)  # warm up (JIT of CasADi's own graph build)

    t_start = time.perf_counter()
    for _ in range(num_runs):
        _ = build_casadi_from_problem(problem, x0=x0, dt=dt)
    t_end = time.perf_counter()
    return (t_end - t_start) / num_runs


def _measure_casadi_solve(
    casadi_prob: CasadiProblem,
    *,
    options: dict[str, Any],
    initial_X: np.ndarray | None,
    initial_U: np.ndarray | None,
) -> tuple[float, bool, Any]:
    """Measure CasADi solve wall-clock time in seconds, discarding a first solve to match trajopt's timing."""
    _ = casadi_prob.solve(options=options, initial_X=initial_X, initial_U=initial_U)

    t_start = time.perf_counter()
    res = casadi_prob.solve(options=options, initial_X=initial_X, initial_U=initial_U)
    t_duration = time.perf_counter() - t_start
    return t_duration, res.success, res.info.get("iter_count")


def compare_one(spec: ProblemSpec) -> None:
    """Build, time, and print a trajopt-vs-CasADi comparison for one benchmark problem."""
    prob, state, info = spec.factory(**spec.factory_kwargs)
    x0 = state.x0
    dt = float(info["dt"])

    trajopt_setup_s = measure_transcription_setup(prob, x0, dt=dt)
    trajopt_solve_res, trajopt_timing = measure_solver_runtime(prob, state, Ipopt(options=spec.solver_opts))

    casadi_setup_s = _measure_casadi_setup(prob, x0, dt)
    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)
    initial_X = np.asarray(state.states) if spec.use_state_as_init_guess else None
    initial_U = np.asarray(state.controls) if spec.use_state_as_init_guess else None
    casadi_solve_s, casadi_success, casadi_iters = _measure_casadi_solve(
        casadi_prob, options=spec.solver_opts, initial_X=initial_X, initial_U=initial_U
    )

    print(f"\n{spec.name}")
    print(f"  {'':<10} {'setup (ms)':>12} {'solve (ms)':>12} {'iters':>8} {'success':>8}")
    print(
        f"  {'trajopt':<10} {trajopt_setup_s * 1e3:>12.3f} {trajopt_timing.median_time_s * 1e3:>12.3f} "
        f"{trajopt_solve_res.iterations:>8} {trajopt_solve_res.success!s:>8}"
    )
    print(
        f"  {'casadi':<10} {casadi_setup_s * 1e3:>12.3f} {casadi_solve_s * 1e3:>12.3f} "
        f"{casadi_iters!s:>8} {casadi_success!s:>8}"
    )
    print(f"  solve speedup (casadi / trajopt): {casadi_solve_s / trajopt_timing.median_time_s:.2f}x")


def main() -> None:
    """Run the trajopt-vs-CasADi timing comparison for all registered benchmark problems."""
    for spec in PROBLEMS:
        compare_one(spec)


if __name__ == "__main__":
    main()
