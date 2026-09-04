from trajopt.benchmarks import (
    cartpole_swingup_benchmark,
    compare_solvers,
    dubins_corridor_benchmark,
    quadrotor_obstacle_benchmark,
)
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt


def run_benchmarks() -> None:
    """Run comparative benchmark of ALTRO vs IPOPT on standard problem suites."""
    problems = [
        ("Cartpole Swingup", lambda: cartpole_swingup_benchmark(N=25, dt=0.05)),
        ("Quadrotor Obstacle", lambda: quadrotor_obstacle_benchmark(N=25, dt=0.05)),
        ("Dubins Corridor", lambda: dubins_corridor_benchmark(N=25, dt=0.05)),
    ]

    for name, factory in problems:
        print(f"\n==================== {name} ====================")
        prob, state, _ = factory()
        solvers = {
            "IPOPT": Ipopt(options={"print_level": 0, "tol": 1e-4, "max_iter": 500}),
            "ALTRO": ALTRO(options=SolverOptions()),
        }
        comparison = compare_solvers(prob, state, solvers, n_repeats=5)
        print(comparison.format_table())

        res_altro = comparison.rows[1].result
        pn_stats = res_altro.info.get("pn_stats")
        pn_iter = getattr(pn_stats, "iterations", 0) if pn_stats is not None else 0
        ran_pn = res_altro.info.get("ran_pn")
        print(f"  [ALTRO details] AL iter: {res_altro.iterations}, PN iter: {pn_iter}, Ran PN: {ran_pn}")


if __name__ == "__main__":
    run_benchmarks()
