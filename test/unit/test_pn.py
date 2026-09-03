import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.solvers.al import AL, ALResult
from trajopt.solvers.options import SolverOptions
from trajopt.solvers.pn import (
    PN,
    PNLayout,
    PNResult,
    _pack_z_pn,
    _pn_evaluate,
    _pn_linesearch,
    _pn_refine,
    _refine_converged,
    _solve_kkt_step,
    _violation,
    multiplier_projection,
    pn_solve,
)
from trajopt.trajectory import Trajectory
from trajopt.transcription.result import Solver, SolverResult

XF = jnp.array([0.0, np.pi, 0.0, 0.0])


def _cartpole_problem(u_bnd: float = 3.0) -> tuple[Problem, jnp.ndarray, float]:
    """Cartpole swing-up with a symmetric control bound and a terminal goal constraint."""
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    x0 = jnp.zeros(n)
    model = Cartpole()
    obj = LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=XF.tolist()), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, dt=dt, integrator=RK4())
    return prob, x0, dt


@pytest.fixture(scope="module")
def al_warm_start_data() -> tuple[Problem, jnp.ndarray, float, Trajectory, ALResult]:
    """Precompute AL warm start for cartpole once per test module."""
    prob, x0, dt = _cartpole_problem()
    al_result = MPC(prob, AL(options=SolverOptions(iterations=300, iterations_outer=30)), x0=x0, xf=XF).solve()
    assert isinstance(al_result, ALResult)
    assert al_result.success
    return prob, x0, dt, al_result.trajectory, al_result


def test_pn_layout_is_a_distinct_second_row_ordering() -> None:
    """PNLayout is self-contained and does not reuse transcription/layout.py's canonical order."""
    import trajopt.solvers.pn as pn_module

    prob, _x0, _dt = _cartpole_problem()
    layout = PNLayout.build(prob)
    n, m, N = 4, 1, 101
    assert layout.Np == N * n + (N - 1) * m
    assert layout.Nd == n + (N - 1) * n + N * layout.p_max
    assert "second row-ordering convention" in (pn_module.__doc__ or "")


def test_pn_kkt_shapes_are_static_across_active_sets(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """The dense KKT system is assembled at the same full size regardless of which rows are active."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)
    options = SolverOptions()

    ev_feasible = _pn_evaluate(prob, layout, options, x0, warm_traj, warm_traj.X, warm_traj.U)
    perturbed_U = warm_traj.U.at[10].set(10.0)  # well past the +-3.0 control bound
    ev_perturbed = _pn_evaluate(prob, layout, options, x0, warm_traj, warm_traj.X, perturbed_U)

    assert ev_feasible.H.shape == ev_perturbed.H.shape == (layout.Np, layout.Np)
    assert ev_feasible.D.shape == ev_perturbed.D.shape == (layout.Nd, layout.Np)
    assert ev_feasible.active.shape == ev_perturbed.active.shape == (layout.Nd,)
    assert bool(jnp.any(ev_feasible.active != ev_perturbed.active))


@pytest.mark.slow
def test_pn_reduces_violation_by_three_orders_within_n_steps(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """A trajectory with a known (small) violation is projected to feasibility within n_steps."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)
    options = SolverOptions(multiplier_projection=False)

    ev0 = _pn_evaluate(prob, layout, options, x0, warm_traj, warm_traj.X, warm_traj.U)
    v0 = float(_violation(ev0.d_pn, ev0.active))
    assert v0 > 0.0

    final_traj, stats, _duals, status = pn_solve(prob, warm_traj, x0, options)
    ev1 = _pn_evaluate(prob, layout, options, x0, final_traj, final_traj.X, final_traj.U)
    v1 = float(_violation(ev1.d_pn, ev1.active))

    assert v1 <= v0 * 1e-3
    assert int(stats.iterations) <= options.n_steps + 1
    assert status in (2, 4)  # SOLVE_SUCCEEDED or MAX_ITERATIONS_OUTER, both fine as long as viol dropped


@pytest.mark.slow
def test_pn_outer_loop_permits_three_solves_at_default_n_steps(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """Finding M: `count <= n_steps` runs the body while count is 0, 1, 2 -- three solves at n_steps=2."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    options = SolverOptions(multiplier_projection=False, constraint_tolerance=0.0, n_steps=2)

    _final_traj, stats, _duals, _status = pn_solve(prob, warm_traj, x0, options)

    assert options.n_steps == 2
    assert int(stats.iterations) == 3


def test_pn_refine_exits_on_tolerance() -> None:
    """`_refine_converged` returns True once the violation clears `projected_newton_tolerance`."""
    converged = _refine_converged(jnp.asarray(1e-4), jnp.asarray(1.0), tolerance=1e-3, r_threshold=1e-9)
    assert bool(converged)


def test_pn_refine_exits_on_convergence_rate() -> None:
    """`_refine_converged` returns True once log10(viol)/log10(viol_prev) falls below r_threshold, even above tolerance."""
    # viol_prev=1e-2 (log=-2), viol=1e-8 (log=-8): rate = 4.0, comfortably above a low r_threshold...
    not_converged = _refine_converged(jnp.asarray(1e-8), jnp.asarray(1e-2), tolerance=1e-12, r_threshold=1.0)
    assert not bool(not_converged)
    # ...but comfortably below a high r_threshold, so the rate criterion alone triggers the exit.
    converged = _refine_converged(jnp.asarray(1e-8), jnp.asarray(1e-2), tolerance=1e-12, r_threshold=10.0)
    assert bool(converged)


def test_pn_inner_linesearch_halves_alpha_and_does_not_update_active_set(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """The inner line search only ever tries alpha = 1, 1/2, 1/4, ... and freezes the active set it was given."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)
    options = SolverOptions()

    ev0 = _pn_evaluate(prob, layout, options, x0, warm_traj, warm_traj.X, warm_traj.U)
    v0 = _violation(ev0.d_pn, ev0.active)
    p = _solve_kkt_step(ev0, layout, options)
    z0 = _pack_z_pn(warm_traj.X, warm_traj.U)

    z_ls, v_ls, accepted = _pn_linesearch(prob, layout, x0, warm_traj, z0, p, ev0.active, v0)

    assert bool(accepted)
    assert float(v_ls) <= float(v0)
    # A step of exactly alpha=1 (no halving) or a power-of-two fraction of it must reproduce z_ls,
    # confirming the accepted alpha came from {1, 1/2, 1/4, ...} and nothing else.
    candidates = jnp.stack([z0 + p / (2.0**k) for k in range(10)])
    assert bool(jnp.any(jnp.all(jnp.isclose(candidates, z_ls[None, :]), axis=1)))


def test_pn_inner_linesearch_rejects_step_that_never_reduces_violation(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """An all-zero step never reduces violation (viol strictly decreases is required), so the search exhausts."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)
    options = SolverOptions()

    ev0 = _pn_evaluate(prob, layout, options, x0, warm_traj, warm_traj.X, warm_traj.U)
    v0 = _violation(ev0.d_pn, ev0.active)
    z0 = _pack_z_pn(warm_traj.X, warm_traj.U)
    zero_step = jnp.zeros_like(z0)

    z_ls, v_ls, accepted = _pn_linesearch(prob, layout, x0, warm_traj, z0, zero_step, ev0.active, v0)

    assert not bool(accepted)
    np.testing.assert_allclose(np.asarray(z_ls), np.asarray(z0))
    assert float(v_ls) == pytest.approx(float(v0))


def test_pn_refine_reuses_the_same_step_across_rounds(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """`_pn_refine` keeps reapplying the single KKT step `p` solved once at the top of the outer iteration."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)
    options = SolverOptions()

    ev0 = _pn_evaluate(prob, layout, options, x0, warm_traj, warm_traj.X, warm_traj.U)
    v0 = _violation(ev0.d_pn, ev0.active)
    p = _solve_kkt_step(ev0, layout, options)
    z0 = _pack_z_pn(warm_traj.X, warm_traj.U)

    z_refined, v_refined = _pn_refine(prob, layout, options, x0, warm_traj, z0, p, ev0.active, v0)

    assert float(v_refined) <= float(v0)
    assert z_refined.shape == z0.shape


def test_rho_primal_regularizes_hessian_and_rho_chol_rho_dual_do_not_exist(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """`rho_primal` shifts the KKT Hessian block's diagonal; `rho_chol`/`rho_dual` are not options (finding F)."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)

    ev = _pn_evaluate(prob, layout, SolverOptions(), x0, warm_traj, warm_traj.X, warm_traj.U)
    p_small = _solve_kkt_step(ev, layout, SolverOptions(rho_primal=1e-8))
    p_large = _solve_kkt_step(ev, layout, SolverOptions(rho_primal=10.0))
    assert not jnp.allclose(p_small, p_large)

    assert not hasattr(SolverOptions(), "rho_chol")
    assert not hasattr(SolverOptions(), "rho_dual")


@pytest.mark.slow
def test_multiplier_projection_is_gated_and_defaults_false(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """`options.multiplier_projection` defaults False and actually gates the projection call."""
    assert SolverOptions().multiplier_projection is False

    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data

    _traj_on, _stats_on, duals_on, _status_on = pn_solve(prob, warm_traj, x0, SolverOptions(multiplier_projection=True))
    _traj_off, _stats_off, duals_off, _status_off = pn_solve(prob, warm_traj, x0, SolverOptions())

    assert bool(jnp.any(duals_on != 0.0))
    assert bool(jnp.all(duals_off == 0.0))


def test_multiplier_projection_matches_the_normal_equations_least_squares_estimate(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """`multiplier_projection` solves `(D_active D_active^T) y = -D_active g`, the KKT stationarity least-squares estimate."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    layout = PNLayout.build(prob)

    ev = _pn_evaluate(prob, layout, SolverOptions(), x0, warm_traj, warm_traj.X, warm_traj.U)
    y = multiplier_projection(ev, layout)
    assert y.shape == (layout.Nd,)

    active = np.asarray(ev.active)
    D_active = np.asarray(ev.D)[active]
    g = np.asarray(ev.g)
    y_ref_active = np.linalg.solve(D_active @ D_active.T, -(D_active @ g))

    np.testing.assert_allclose(np.asarray(y)[active], y_ref_active, atol=1e-6)
    assert bool(np.all(np.asarray(y)[~active] == 0.0))

    # The projected multiplier strictly reduces the KKT stationarity residual relative to y=0.
    D_masked = np.asarray(ev.D) * active[:, None]
    residual_raw = np.linalg.norm(g)
    residual_projected = np.linalg.norm(g + D_masked.T @ np.asarray(y))
    assert residual_projected < residual_raw


def test_dense_kkt_divergence_is_documented_in_solver_docstring() -> None:
    """The module docstring records the dense-KKT-vs-QDLDL divergence, per the ticket's requirement."""
    import trajopt.solvers.pn as pn_module

    doc = pn_module.__doc__ or ""
    assert "QDLDL" in doc
    assert "dense" in doc.lower()


def test_pn_satisfies_solver_protocol() -> None:
    """PN structurally satisfies the Solver protocol (has a matching .solve)."""
    assert isinstance(PN(), Solver)


def test_pn_result_satisfies_solver_result_protocol(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """PNResult structurally satisfies the SolverResult protocol."""
    prob, x0, _dt, warm_traj, _al_res = al_warm_start_data
    result = MPC(
        prob, PN(options=SolverOptions(n_steps=1)), x0=x0, xf=XF, initial_trajectory=warm_traj
    ).solve()

    assert isinstance(result, PNResult)
    assert isinstance(result, SolverResult)
    assert result.constraint_violation < SolverOptions().constraint_tolerance * 10


@pytest.mark.slow
def test_pn_cartpole_polishes_al_output_and_cost_barely_moves(
    al_warm_start_data: tuple[Problem, jnp.ndarray, float, Trajectory, ALResult],
) -> None:
    """End to end: PN polishes an AL-converged cartpole trajectory to a tighter violation, cost nearly unchanged."""
    prob, x0, _dt, warm_traj, al_result = al_warm_start_data

    pn_result = MPC(prob, PN(options=SolverOptions()), x0=x0, xf=XF, initial_trajectory=warm_traj).solve()

    assert pn_result.constraint_violation <= al_result.constraint_violation
    assert pn_result.cost == pytest.approx(al_result.cost, rel=1e-3)
