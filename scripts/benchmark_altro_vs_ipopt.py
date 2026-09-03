import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import jax
import numpy as np

from trajopt.benchmarks import (
    cartpole_swingup_benchmark,
    dubins_corridor_benchmark,
    quadrotor_obstacle_benchmark,
)
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt


def run_benchmarks() -> None:
    problems = [
        ("Cartpole Swingup", cartpole_swingup_benchmark, {"N": 25, "dt": 0.05}),
        ("Quadrotor Obstacle", quadrotor_obstacle_benchmark, {"N": 25, "dt": 0.05}),
        ("Dubins Corridor", dubins_corridor_benchmark, {"N": 25, "dt": 0.05}),
    ]

    for name, factory, kwargs in problems:
        print(f"\n==================== {name} ====================")
        prob, state, _ = factory(**kwargs)

        # 1. Ipopt
        ipopt = Ipopt(options={"print_level": 0, "tol": 1e-4, "max_iter": 500})
        t0 = time.perf_counter()
        res_ipopt = ipopt.solve(prob, state)
        t_ipopt_first = time.perf_counter() - t0

        times_ipopt = []
        for _ in range(5):
            t0 = time.perf_counter()
            res_ipopt = ipopt.solve(prob, state)
            times_ipopt.append(time.perf_counter() - t0)

        print("IPOPT:")
        print(f"  First call: {t_ipopt_first * 1000:.2f} ms")
        print(f"  Median warm: {np.median(times_ipopt) * 1000:.2f} ms (min: {np.min(times_ipopt) * 1000:.2f} ms)")
        print(
            f"  Cost: {res_ipopt.cost:.4f}, Viol: {res_ipopt.constraint_violation:.2e}, Iter: {res_ipopt.iterations}, Success: {res_ipopt.success}"
        )

        # 2. ALTRO default
        opts = SolverOptions()
        altro = ALTRO(options=opts)
        t0 = time.perf_counter()
        res_altro = altro.solve(prob, state)
        jax.block_until_ready(res_altro.trajectory.X)
        t_altro_first = time.perf_counter() - t0

        times_altro = []
        for _ in range(5):
            t0 = time.perf_counter()
            res_altro = altro.solve(prob, state)
            jax.block_until_ready(res_altro.trajectory.X)
            times_altro.append(time.perf_counter() - t0)

        print("ALTRO (default):")
        print(f"  First call: {t_altro_first * 1000:.2f} ms")
        print(f"  Median warm: {np.median(times_altro) * 1000:.2f} ms (min: {np.min(times_altro) * 1000:.2f} ms)")
        al_iter = res_altro.iterations
        pn_iter = res_altro.info.get("pn_stats").iterations if res_altro.info.get("pn_stats") is not None else 0
        ran_pn = res_altro.info.get("ran_pn")
        print(
            f"  Cost: {res_altro.cost:.4f}, Viol: {res_altro.constraint_violation:.2e}, AL iter: {al_iter}, PN iter: {pn_iter}, Ran PN: {ran_pn}, Success: {res_altro.success}"
        )


if __name__ == "__main__":
    run_benchmarks()
