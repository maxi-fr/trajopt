import jax.numpy as jnp
import pytest

from trajopt.benchmarks import (
    ClosedLoopStats,
    SolverComparison,
    SolverRow,
    cartpole_swingup_benchmark,
    compare_solvers,
    compare_solvers_closed_loop,
    measure_closed_loop_mpc,
)
from trajopt.problem import MPCState, Problem
from trajopt.solvers.ilqr import ILQR
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt, IpoptResult


class StubSolver:
    """Solver returning `state`'s own guess while claiming a cost and violation of its choosing.

    Lets the harness be tested without a real solve, and pins down that the table's numbers are
    recomputed rather than copied out of the result.
    """

    def __init__(self, *, claimed_cost: float, claimed_violation: float) -> None:
        self.claimed_cost = claimed_cost
        self.claimed_violation = claimed_violation

    def solve(self, problem: Problem, state: MPCState) -> IpoptResult:
        """Return `state`'s guess dressed up as a converged solve."""
        del problem
        t = jnp.concatenate([jnp.zeros(1), jnp.cumsum(state.dt)])
        return IpoptResult(
            trajectory=Trajectory(X=state.states, U=state.controls, t=t, dt=state.dt),
            success=True,
            status=0,
            message="stub",
            cost=self.claimed_cost,
            Z=state.Z,
            info={},
            constraint_violation=self.claimed_violation,
            iterations=7,
        )


def test_compare_solvers_scores_rows_itself_rather_than_believing_the_solver() -> None:
    """Verify cost and violation are recomputed from the returned Z, not copied from the result."""
    prob, state, _ = cartpole_swingup_benchmark(N=5, dt=0.05)
    stub = StubSolver(claimed_cost=-1.0, claimed_violation=0.0)

    comparison = compare_solvers(prob, state, [stub], n_repeats=2)

    (row,) = comparison.rows
    assert row.solver == "StubSolver"
    assert row.iterations == 7  # reported values still come straight through
    assert row.success is True
    # The stub returned the initial guess, which sits at theta=0.01 against a goal of pi.
    assert row.cost > 0.0
    assert row.constraint_violation > 1.0
    assert row.result.cost == -1.0  # the solver's own claim is preserved on the result


def test_compare_solvers_labels_rows_from_a_mapping() -> None:
    """Verify a mapping's keys become row labels, so two configurations of one solver stay apart."""
    prob, state, _ = cartpole_swingup_benchmark(N=5, dt=0.05)
    solvers = {
        "loose": StubSolver(claimed_cost=1.0, claimed_violation=0.0),
        "tight": StubSolver(claimed_cost=2.0, claimed_violation=0.0),
    }

    comparison = compare_solvers(prob, state, solvers, n_repeats=1)

    assert [row.solver for row in comparison.rows] == ["loose", "tight"]
    assert comparison.model == "Cartpole"
    assert comparison.n_repeats == 1


def test_compare_solvers_timing_orders_first_call_median_and_min() -> None:
    """Verify each row carries a first-call duration alongside the median and minimum warm call."""
    prob, state, _ = cartpole_swingup_benchmark(N=5, dt=0.05)

    comparison = compare_solvers(prob, state, [StubSolver(claimed_cost=0.0, claimed_violation=0.0)], n_repeats=3)

    timing = comparison.rows[0].timing
    assert timing.first_call_time_s > 0.0
    assert timing.min_time_s <= timing.median_time_s


def _row(name: str, *, linearizing: bool) -> SolverRow:
    """Build a table row with placeholder numbers, for rendering tests."""
    stub = StubSolver(claimed_cost=0.0, claimed_violation=0.0)
    prob, state, _ = cartpole_swingup_benchmark(N=5, dt=0.05)
    result = stub.solve(prob, state)
    return SolverRow(
        solver=name,
        success=True,
        iterations=3,
        cost=1.0,
        constraint_violation=1e-9,
        timing=compare_solvers(prob, state, [stub], n_repeats=1).rows[0].timing,
        linearizing=linearizing,
        result=result,
    )


def test_format_table_flags_linearizing_rows_and_footnotes_them() -> None:
    """Verify a Backend that solved one linearization is marked in the table and explained below it."""
    table = SolverComparison(
        model="Cartpole",
        n_repeats=5,
        rows=(_row("Ipopt", linearizing=False), _row("OSQP", linearizing=True)),
    ).format_table()

    assert "OSQP *" in table
    assert "Ipopt *" not in table
    assert "Operating Point" in table.splitlines()[-1]


def test_format_table_omits_the_footnote_when_no_row_linearizes() -> None:
    """Verify the linearization footnote appears only when it applies to a row."""
    table = SolverComparison(model="Cartpole", n_repeats=5, rows=(_row("Ipopt", linearizing=False),)).format_table()

    assert "*" not in table


@pytest.mark.slow
def test_ilqr_warns_about_ignored_constraints_and_reports_the_violation_it_leaves() -> None:
    """Verify iLQR announces that it dropped the constraints and measures how far it ends up outside them."""
    prob, state, _ = cartpole_swingup_benchmark(N=15, dt=0.05)

    with pytest.warns(UserWarning, match="ignores constraints and box bounds"):
        result = ILQR().solve(prob, state)

    # The cartpole's cart is bounded to |p| <= 0.4 and its actuator to |u| <= 20; an unconstrained
    # swing-up respects neither, and the violation is now measured rather than reported as zero.
    assert result.constraint_violation > 0.0


@pytest.mark.slow
def test_closed_loop_mpc_measurement_and_jitter() -> None:
    """Verify closed-loop MPC reports sustained frequency, latency jitter, and warm-start speedup."""
    pytest.importorskip("cyipopt")

    prob, state, _ = cartpole_swingup_benchmark(N=25, dt=0.05)
    num_steps = 10
    solver = Ipopt(options={"max_iter": 50, "tol": 1e-4, "print_level": 0})

    stats: ClosedLoopStats = measure_closed_loop_mpc(prob, state, solver, num_steps=num_steps)

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


@pytest.mark.slow
def test_compare_solvers_closed_loop_gives_every_solver_its_own_latency_row() -> None:
    """Verify the closed-loop comparison runs a receding horizon per solver and keeps their order."""
    prob, state, _ = cartpole_swingup_benchmark(N=10, dt=0.05)
    solvers = {"stub_a": StubSolver(claimed_cost=0.0, claimed_violation=0.0), "ilqr": ILQR()}

    comparison = compare_solvers_closed_loop(prob, state, solvers, num_steps=3)

    assert [row.solver for row in comparison.rows] == ["stub_a", "ilqr"]
    assert comparison.num_steps == 3
    assert all(row.stats.num_steps == 3 for row in comparison.rows)
    assert not any(row.linearizing for row in comparison.rows)
