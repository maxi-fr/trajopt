import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

from trajopt.solvers.options import SolverOptions, SolverStats, TerminationStatus, to_solver_status

if TYPE_CHECKING:
    from trajopt.transcription.result import SolverStatus

_DEAD_OPTION_NAMES = {
    "iterations_inner",
    "bp_reg",
    "square_root",
    "save_S",
    "static_bp",
    "reuse_jacobians",
    "active_set_tolerance_al",
    "rho_chol",
    "rho_dual",
    "solve_type",
    "show_summary",
    "trim_stats",
    "bp_reg_type",
    "closed_loop_initial_rollout",
    "dynamics_funsig",
    "dynamics_diffmethod",
}

_EXPECTED_DEFAULTS = {
    "constraint_tolerance": 1e-6,
    "cost_tolerance": 1e-4,
    "cost_tolerance_intermediate": 1e-4,
    "gradient_tolerance": 10.0,
    "gradient_tolerance_intermediate": 1.0,
    "expected_decrease_tolerance": 1e-10,
    "dJ_counter_limit": 10,
    "line_search_lower_bound": 1e-8,
    "line_search_upper_bound": 10.0,
    "line_search_decrease_factor": 0.5,
    "iterations_linesearch": 20,
    "max_cost_value": 1.0e8,
    "max_state_value": 1.0e8,
    "max_control_value": 1.0e8,
    "bp_reg_initial": 0.0,
    "bp_reg_increase_factor": 1.6,
    "bp_reg_max": 1.0e8,
    "bp_reg_min": 1.0e-8,
    "bp_reg_fp": 10.0,
    "use_conic_cost": False,
    "penalty_initial": 1.0,
    "penalty_scaling": 10.0,
    "penalty_max": 1e8,
    "dual_max": 1e8,
    "iterations_outer": 30,
    "kickout_max_penalty": False,
    "reset_duals": True,
    "reset_penalties": True,
    "force_pn": False,
    "verbose_pn": False,
    "n_steps": 2,
    "projected_newton_tolerance": 1e-3,
    "active_set_tolerance_pn": 1e-3,
    "multiplier_projection": False,
    "rho_primal": 1.0e-8,
    "r_threshold": 1.1,
    "projected_newton": True,
    "iterations": 1000,
    "verbose": 0,
}


def test_solver_options_defaults_match_altro() -> None:
    """Every SolverOptions default matches altro_jl/src/solver_opts.jl literally."""
    opts = SolverOptions()
    for name, expected in _EXPECTED_DEFAULTS.items():
        actual = getattr(opts, name)
        assert actual == expected, f"{name}: expected {expected}, got {actual}"


def test_solver_options_field_set_matches_expected_defaults_exactly() -> None:
    """SolverOptions carries exactly the expected live fields, no more and no fewer."""
    field_names = {f.name for f in dataclasses.fields(SolverOptions)}
    assert field_names == set(_EXPECTED_DEFAULTS)


def test_solver_options_excludes_dead_options() -> None:
    """None of the eleven dead options, bp_reg_type, or the Julia-only dispatch knobs are fields."""
    field_names = {f.name for f in dataclasses.fields(SolverOptions)}
    assert field_names.isdisjoint(_DEAD_OPTION_NAMES)


def test_solver_options_is_frozen() -> None:
    """SolverOptions is immutable, matching the Ipopt/OSQP/Clarabel static-config pattern."""
    opts = SolverOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.iterations = 5  # ty: ignore[invalid-assignment]


def test_bp_reg_max_documents_the_altro_divergence() -> None:
    """bp_reg_max's docstring records that it is live here and dead upstream."""
    docstring = SolverOptions.__doc__ or ""
    assert "bp_reg_max" in docstring
    assert "live" in docstring.lower()
    assert "dead" in docstring.lower()
    assert "Altro" in docstring


def test_termination_status_ordinals_match_altro_enum_declaration() -> None:
    """TerminationStatus member order matches Altro's @enum(...) declaration exactly."""
    expected_order = [
        "UNSOLVED",
        "LINESEARCH_FAIL",
        "SOLVE_SUCCEEDED",
        "MAX_ITERATIONS",
        "MAX_ITERATIONS_OUTER",
        "MAXIMUM_COST",
        "STATE_LIMIT",
        "CONTROL_LIMIT",
        "NO_PROGRESS",
        "COST_INCREASE",
    ]
    for ordinal, name in enumerate(expected_order):
        assert TerminationStatus[name] == ordinal


def test_termination_status_ordering_is_load_bearing() -> None:
    """SOLVE_SUCCEEDED sits strictly above LINESEARCH_FAIL and strictly below MAX_ITERATIONS.

    Reproduces the two Altro comparisons (finding C) that decide AL outer-loop break and
    ALTRO's polish phase: `status > SOLVE_SUCCEEDED` and `status <= SOLVE_SUCCEEDED`.
    """
    assert TerminationStatus.LINESEARCH_FAIL < TerminationStatus.SOLVE_SUCCEEDED
    assert TerminationStatus.MAX_ITERATIONS > TerminationStatus.SOLVE_SUCCEEDED


def test_solver_stats_roundtrips_through_jit_with_fixed_shapes() -> None:
    """Constructing, writing into, and reading a SolverStats survives jax.jit unchanged in shape."""
    opts = SolverOptions(iterations=5)

    @jax.jit
    def step(stats: SolverStats, i: jax.Array) -> SolverStats:
        return dataclasses.replace(
            stats,
            iterations=stats.iterations + 1,
            cost=stats.cost.at[i].set(1.0),
            dJ=stats.dJ.at[i].set(2.0),
            c_max=stats.c_max.at[i].set(3.0),
            gradient=stats.gradient.at[i].set(4.0),
            penalty_max=stats.penalty_max.at[i].set(5.0),
            dJ_zero_counter=stats.dJ_zero_counter,
            ls_failed=stats.ls_failed,
        )

    stats = SolverStats.create(opts)
    assert stats.cost.shape == (5,)

    stats = step(stats, jnp.asarray(0))

    assert stats.cost.shape == (5,)
    assert stats.dJ.shape == (5,)
    assert stats.c_max.shape == (5,)
    assert stats.gradient.shape == (5,)
    assert stats.penalty_max.shape == (5,)
    assert int(stats.iterations) == 1
    assert float(stats.cost[0]) == 1.0
    assert float(stats.dJ[0]) == 2.0
    assert float(stats.c_max[0]) == 3.0
    assert float(stats.gradient[0]) == 4.0
    assert float(stats.penalty_max[0]) == 5.0


def test_solver_stats_is_a_pytree() -> None:
    """SolverStats survives a tree_map without a host callback or shape change."""
    opts = SolverOptions(iterations=3)
    stats = SolverStats.create(opts)
    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, stats)
    assert doubled.cost.shape == stats.cost.shape


def test_to_solver_status_covers_every_termination_status_exhaustively() -> None:
    """Every TerminationStatus member maps onto a valid public SolverStatus."""
    valid_statuses: set[SolverStatus] = {"converged", "infeasible", "iteration_limit", "error"}
    for member in TerminationStatus:
        result = to_solver_status(member)
        assert result in valid_statuses, f"{member.name} mapped to invalid status {result!r}"


def test_to_solver_status_mapping_matches_reference_table() -> None:
    """The mapping matches reference doc §2's suggested table exactly."""
    expected: dict[TerminationStatus, SolverStatus] = {
        TerminationStatus.SOLVE_SUCCEEDED: "converged",
        TerminationStatus.MAX_ITERATIONS_OUTER: "infeasible",
        TerminationStatus.MAXIMUM_COST: "infeasible",
        TerminationStatus.UNSOLVED: "infeasible",
        TerminationStatus.MAX_ITERATIONS: "iteration_limit",
        TerminationStatus.NO_PROGRESS: "iteration_limit",
        TerminationStatus.LINESEARCH_FAIL: "iteration_limit",
        TerminationStatus.STATE_LIMIT: "error",
        TerminationStatus.CONTROL_LIMIT: "error",
        TerminationStatus.COST_INCREASE: "error",
    }
    assert set(expected) == set(TerminationStatus)
    for member, status in expected.items():
        assert to_solver_status(member) == status
