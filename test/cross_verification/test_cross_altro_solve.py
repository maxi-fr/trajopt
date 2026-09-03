from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models import Cartpole
from trajopt.problem import Problem, retarget_to_goal
from trajopt.solvers.al import ALConstraints
from trajopt.solvers.altro import altro_solve
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory

# Ticket 33, reference §8.2's "Box" scenario: the full two-phase ALTRO driver (AL, then PN's
# polish) against Altro.ALTROSolver's own solve!, on the same bound + goal constrained cartpole
# ticket 29's and ticket 32's own cross tests already verify each phase against in isolation. Built
# manually (not via Altro.Problems.Cartpole(), which uses RK3) with TO.Problem's default RK4.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trajopt_ticket33_setup(model, Q, R, Qf, x0, xf, N, dt, u_bnd, U0, opts)
    tf = dt * (N - 1)
    obj = TO.LQRObjective(Diagonal(Q), Diagonal(R), Diagonal(Qf), xf, N)

    conSet = TO.ConstraintList(size(Q, 1), size(R, 1), N)
    bnd = TO.BoundConstraint(size(Q, 1), size(R, 1), u_min=-u_bnd, u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    TO.add_constraint!(conSet, bnd, 1:N-1)
    TO.add_constraint!(conSet, goal, N:N)

    prob = TO.Problem(model, obj, x0, tf; xf=xf, constraints=conSet, U0=[copy(u) for u in U0])
    return Altro.ALTROSolver(prob, opts)
end

function trajopt_ticket33_run_solve(solver)
    Altro.solve!(solver)

    Z = solver.solver_al.ilqr.Z
    N = length(Z)
    X = cat([Vector(RD.state(Z[k])) for k = 1:N]..., dims=2)
    U = cat([Vector(RD.control(Z[k])) for k = 1:N-1]..., dims=2)

    st = Altro.stats(solver)
    cost_final = st.cost[st.iterations]
    viol = Altro.max_violation(solver)
    return X, U, cost_final, viol, Int(st.status), st.iterations_outer
end
"""


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


def _build_jl_altro_solver(jl: Any, options: SolverOptions, u_bnd: float, x0: np.ndarray, xf: np.ndarray) -> Any:
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    U0 = np.full((N - 1, m), 0.01)

    jl.seval(_ALTRO_SETUP)
    setup_fn = jl.seval("trajopt_ticket33_setup")
    jl_model = jl.seval("RobotZoo.Cartpole()")
    jl_opts = jl.Altro.SolverOptions(
        constraint_tolerance=float(options.constraint_tolerance),
        cost_tolerance=float(options.cost_tolerance),
        cost_tolerance_intermediate=float(options.cost_tolerance_intermediate),
        gradient_tolerance=float(options.gradient_tolerance),
        gradient_tolerance_intermediate=float(options.gradient_tolerance_intermediate),
        iterations_linesearch=int(options.iterations_linesearch),
        line_search_lower_bound=float(options.line_search_lower_bound),
        line_search_upper_bound=float(options.line_search_upper_bound),
        line_search_decrease_factor=float(options.line_search_decrease_factor),
        bp_reg_fp=float(options.bp_reg_fp),
        bp_reg_min=float(options.bp_reg_min),
        bp_reg_initial=float(options.bp_reg_initial),
        bp_reg_increase_factor=float(options.bp_reg_increase_factor),
        penalty_initial=float(options.penalty_initial),
        penalty_scaling=float(options.penalty_scaling),
        penalty_max=float(options.penalty_max),
        dual_max=float(options.dual_max),
        iterations_outer=int(options.iterations_outer),
        iterations=int(options.iterations),
        projected_newton=bool(options.projected_newton),
        projected_newton_tolerance=float(options.projected_newton_tolerance),
        n_steps=int(options.n_steps),
    )
    return setup_fn(jl_model, Q, R, Qf, x0, xf, N, dt, u_bnd, list(U0), jl_opts)


def test_cross_altro_solve_cartpole_matches_altro(jl_altro: Any) -> None:
    """End-to-end ALTRO driver (AL then PN) matches Altro.ALTROSolver's own solve! on the cartpole Box scenario."""
    u_bnd = 3.0
    x0 = np.zeros(4)
    xf = np.array([0.0, np.pi, 0.0, 0.0])
    options = SolverOptions(iterations=300, iterations_outer=30)

    solver = _build_jl_altro_solver(jl_altro, options, u_bnd, x0, xf)
    run_solve = jl_altro.seval("trajopt_ticket33_run_solve")
    X_jl, U_jl, cost_jl, viol_jl, status_jl, n_al_jl = run_solve(solver)
    X_jl = np.moveaxis(np.asarray(X_jl), -1, 0)
    U_jl = np.moveaxis(np.asarray(U_jl), -1, 0)

    prob, guess = _python_problem(u_bnd, xf)
    al0 = ALConstraints.build(prob.constraints, penalty_initial=options.penalty_initial)
    result = altro_solve(prob, guess, al0, jnp.asarray(x0), options)

    assert int(result.status) == int(status_jl)
    assert int(result.status) == int(TerminationStatus.SOLVE_SUCCEEDED)
    assert int(result.al_stats.iterations) == int(n_al_jl)

    np.testing.assert_allclose(np.asarray(result.trajectory.X), X_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.trajectory.U), U_jl, atol=1e-8)
    # `Altro.max_violation(solver)` is `max` of the AL- and PN-side trackers separately, evaluated
    # at slightly different points in Altro's own post-solve bookkeeping than our single
    # recomputation on the final trajectory (both sides' state/control trajectories already agree
    # to 1e-8 above); both violations are nonetheless comfortably inside constraint_tolerance.
    assert float(result.c_max) < options.constraint_tolerance
    assert float(viol_jl) < options.constraint_tolerance

    py_cost = float(prob.obj.cost(result.trajectory))
    np.testing.assert_allclose(py_cost, float(cost_jl), atol=1e-6)
