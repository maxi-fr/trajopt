from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models import Cartpole
from trajopt.problem import Problem, retarget_to_goal
from trajopt.solvers.al import ALConstraints, al_solve
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory

# Ticket 29, reference §8.2 row 15: a full al_solve outer loop against Altro.ALSolver's solve!.
# Built manually (not via Altro.Problems.Cartpole(), which uses RK3 -- ticket 27's own cross test
# hits the same mismatch and reconstructs the problem with TO.Problem's default RK4 instead, so
# every dynamics call, not just the AL bookkeeping, lines up between the two sides) with the same
# bound + goal constraints ticket 28's cross test already verified in isolation.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trajopt_ticket29_setup(model, Q, R, Qf, x0, xf, N, dt, u_bnd, U0, opts)
    tf = dt * (N - 1)
    n, m = RD.dims(model)
    obj = TO.LQRObjective(Diagonal(Q), Diagonal(R), Diagonal(Qf), xf, N)

    conSet = TO.ConstraintList(n, m, N)
    bnd = TO.BoundConstraint(n, m, u_min=-u_bnd, u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    TO.add_constraint!(conSet, bnd, 1:N-1)
    TO.add_constraint!(conSet, goal, N:N)

    prob = TO.Problem(model, obj, x0, tf; xf=xf, constraints=conSet, U0=[copy(u) for u in U0])
    return Altro.ALSolver(prob, opts)
end

function trajopt_ticket29_run_solve(solver)
    Altro.solve!(solver)

    N = solver.ilqr.N
    Z = solver.ilqr.Z
    X = cat([Vector(RD.state(Z[k])) for k = 1:N]..., dims=2)
    U = cat([Vector(RD.control(Z[k])) for k = 1:N-1]..., dims=2)

    stats = solver.stats
    n_al = stats.iterations_outer
    cost_hist = zeros(n_al)
    c_max_hist = zeros(n_al)
    mu_max_hist = zeros(n_al)
    for al_i in 1:n_al
        idx = findfirst(==(al_i), stats.iteration_outer)
        cost_hist[al_i] = stats.cost[idx]
        c_max_hist[al_i] = stats.c_max[idx]
        mu_max_hist[al_i] = stats.penalty_max[idx]
    end

    conset = TO.get_constraints(solver)
    bndcon = conset.constraints[1]
    goalcon = conset.constraints[2]
    lam_bnd = cat([Vector(bndcon.λ[i]) for i in 1:length(bndcon.inds)]..., dims=2)
    mu_bnd = cat([Vector(bndcon.μ[i]) for i in 1:length(bndcon.inds)]..., dims=2)
    lam_goal = Vector(goalcon.λ[1])
    mu_goal = Vector(goalcon.μ[1])

    return X, U, cost_hist, c_max_hist, mu_max_hist, Int(stats.status), n_al, lam_bnd, mu_bnd, lam_goal, mu_goal
end
"""


def _build_jl_al_solver(jl: Any, options: SolverOptions, u_bnd: float, x0: np.ndarray, xf: np.ndarray) -> Any:
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Qv, Rv, Qfv = 1e-2, 1e-1, 1e2
    Q = Qv * np.ones(n) * dt
    R = Rv * np.ones(m) * dt
    Qf = Qfv * np.ones(n)
    U0 = np.full((N - 1, m), 0.01)

    jl.seval(_ALTRO_SETUP)
    setup_fn = jl.seval("trajopt_ticket29_setup")
    jl_model = jl.seval("RobotZoo.Cartpole()")
    jl_opts = jl.Altro.SolverOptions(
        constraint_tolerance=float(options.constraint_tolerance),
        cost_tolerance=float(options.cost_tolerance),
        cost_tolerance_intermediate=float(options.cost_tolerance_intermediate),
        gradient_tolerance=float(options.gradient_tolerance),
        gradient_tolerance_intermediate=float(options.gradient_tolerance_intermediate),
        max_state_value=float(options.max_state_value),
        max_control_value=float(options.max_control_value),
        iterations_linesearch=int(options.iterations_linesearch),
        line_search_lower_bound=float(options.line_search_lower_bound),
        line_search_upper_bound=float(options.line_search_upper_bound),
        line_search_decrease_factor=float(options.line_search_decrease_factor),
        expected_decrease_tolerance=float(options.expected_decrease_tolerance),
        bp_reg_fp=float(options.bp_reg_fp),
        bp_reg_min=float(options.bp_reg_min),
        bp_reg_initial=float(options.bp_reg_initial),
        bp_reg_increase_factor=float(options.bp_reg_increase_factor),
        penalty_initial=float(options.penalty_initial),
        penalty_scaling=float(options.penalty_scaling),
        penalty_max=float(options.penalty_max),
        dual_max=float(options.dual_max),
        iterations_outer=int(options.iterations_outer),
        kickout_max_penalty=bool(options.kickout_max_penalty),
        reset_duals=bool(options.reset_duals),
        reset_penalties=bool(options.reset_penalties),
        max_cost_value=float(options.max_cost_value),
        iterations=int(options.iterations),
    )
    return setup_fn(jl_model, Q, R, Qf, x0, xf, N, dt, u_bnd, list(U0), jl_opts)


def _python_problem(u_bnd: float, xf: np.ndarray) -> tuple[Problem, Trajectory]:
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    model = Cartpole()
    obj = retarget_to_goal(LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N), jnp.asarray(xf))

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=xf.tolist()), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())
    U0 = jnp.full((N - 1, m), 0.01)
    t = jnp.arange(N) * dt
    dt_arr = jnp.full(N - 1, dt)
    guess = Trajectory(X=jnp.zeros((N, n)), U=U0, t=t, dt=dt_arr)
    return prob, guess


def test_cross_al_solve_cartpole_matches_altro(jl_altro: Any) -> None:
    u_bnd = 3.0
    x0 = np.zeros(4)
    xf = np.array([0.0, np.pi, 0.0, 0.0])
    options = SolverOptions()

    solver = _build_jl_al_solver(jl_altro, options, u_bnd, x0, xf)
    run_solve = jl_altro.seval("trajopt_ticket29_run_solve")
    X_jl, U_jl, cost_jl, c_max_jl, mu_max_jl, status_jl, n_al_jl, lam_bnd_jl, mu_bnd_jl, lam_goal_jl, mu_goal_jl = (
        run_solve(solver)
    )
    X_jl = np.moveaxis(np.asarray(X_jl), -1, 0)
    U_jl = np.moveaxis(np.asarray(U_jl), -1, 0)

    prob, guess = _python_problem(u_bnd, xf)
    al0 = ALConstraints.build(prob.constraints, penalty_initial=options.penalty_initial)
    final_traj, final_al, stats, status = al_solve(prob, guess, al0, options)

    n_iter = int(stats.iterations)
    assert n_iter == int(n_al_jl)
    assert int(status) == int(status_jl)
    assert int(status) == int(TerminationStatus.SOLVE_SUCCEEDED)

    np.testing.assert_allclose(np.asarray(final_traj.X), X_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(final_traj.U), U_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(stats.cost[:n_iter]), np.asarray(cost_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(stats.c_max[:n_iter]), np.asarray(c_max_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(stats.penalty_max[:n_iter]), np.asarray(mu_max_jl), atol=1e-8)

    n, m, N = 4, 1, 101
    u_start = al0.p_cons_max + 2 * n
    np.testing.assert_allclose(
        np.asarray(final_al.lam[: N - 1, u_start : u_start + 2 * m]), np.asarray(lam_bnd_jl).T, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(final_al.mu[: N - 1, u_start : u_start + 2 * m]), np.asarray(mu_bnd_jl).T, atol=1e-8
    )
    np.testing.assert_allclose(np.asarray(final_al.lam[N - 1, :n]), np.asarray(lam_goal_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(final_al.mu[N - 1, :n]), np.asarray(mu_goal_jl), atol=1e-8)
