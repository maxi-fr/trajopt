from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective, Objective
from trajopt.models import Cartpole, Pendulum
from trajopt.problem import Problem, retarget_to_goal
from trajopt.solvers.ilqr import ILQR, ilqr_solve
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory

# Ticket 27: a full ilqr_solve against Altro.iLQRSolver's solve!. Ticket 25/26 already
# cross-verify backward_pass and forward_pass/rollout_closed_loop in isolation; this closes the
# loop by comparing the whole `initialize! + solve!` iteration against ours end to end.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trojopt_ticket27_setup(model, Q, R, Qf, x0, xf, N, dt, U0, opts)
    tf = dt * (N - 1)
    obj = TO.LQRObjective(Diagonal(Q), Diagonal(R), Diagonal(Qf), xf, N)
    prob = TO.Problem(model, obj, x0, tf; xf=xf, U0=[copy(u) for u in U0])
    solver = Altro.iLQRSolver(prob, opts)
    return solver
end

function trojopt_ticket27_run_solve(solver)
    Altro.solve!(solver)
    N = solver.N
    X = cat([Vector(RD.state(solver.Z[k])) for k = 1:N]..., dims=2)
    U = cat([Vector(RD.control(solver.Z[k])) for k = 1:N-1]..., dims=2)
    J = TO.cost(solver.obj, solver.Z)
    return X, U, J, Int(solver.stats.status), solver.stats.iterations
end
"""


class _BenchmarkSetup(NamedTuple):
    model: Pendulum | Cartpole
    obj: Objective
    N: int
    dt: float
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    x0: np.ndarray
    xf: np.ndarray
    U0: np.ndarray


def _pendulum_setup() -> _BenchmarkSetup:
    """Ported verbatim from altro_jl/problems/pendulum.jl, but with default (RK4) integration."""
    n, m, N, tf = 2, 1, 51, 3.0
    dt = tf / (N - 1)
    Q = 1e-3 * np.ones(n) * dt
    R = 1e-3 * np.ones(m) * dt
    Qf = np.ones(n)
    x0 = np.zeros(n)
    xf = np.array([np.pi, 0.0])
    model = Pendulum()
    obj = retarget_to_goal(LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N), jnp.asarray(xf))
    U0 = np.full((N - 1, m), 0.1)
    return _BenchmarkSetup(model=model, obj=obj, N=N, dt=dt, Q=Q, R=R, Qf=Qf, x0=x0, xf=xf, U0=U0)


def _cartpole_setup() -> _BenchmarkSetup:
    """Ported verbatim from altro_jl/problems/cartpole.jl, but with default (RK4) integration."""
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Qv, Rv, Qfv = 1e-2, 1e-1, 1e2
    Q = Qv * np.ones(n) * dt
    R = Rv * np.ones(m) * dt
    Qf = Qfv * np.ones(n)
    x0 = np.zeros(n)
    xf = np.array([0.0, np.pi, 0.0, 0.0])
    model = Cartpole()
    obj = retarget_to_goal(LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N), jnp.asarray(xf))
    U0 = np.full((N - 1, m), 0.01)
    return _BenchmarkSetup(model=model, obj=obj, N=N, dt=dt, Q=Q, R=R, Qf=Qf, x0=x0, xf=xf, U0=U0)


def _build_jl_solver(jl: Any, setup: _BenchmarkSetup, jl_model_expr: str, options: SolverOptions) -> Any:
    jl.seval(_ALTRO_SETUP)
    setup_fn = jl.seval("trojopt_ticket27_setup")
    jl_model = jl.seval(jl_model_expr)
    jl_opts = jl.Altro.SolverOptions(
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
        cost_tolerance=float(options.cost_tolerance),
        gradient_tolerance=float(options.gradient_tolerance),
        constraint_tolerance=float(options.constraint_tolerance),
        iterations=int(options.iterations),
        max_cost_value=float(options.max_cost_value),
    )
    return setup_fn(
        jl_model,
        setup.Q,
        setup.R,
        setup.Qf,
        setup.x0,
        setup.xf,
        setup.N,
        setup.dt,
        list(setup.U0),
        jl_opts,
    )


def _assert_solve_matches_altro(jl: Any, setup: _BenchmarkSetup, jl_model_expr: str) -> None:
    options = SolverOptions()
    solver = _build_jl_solver(jl, setup, jl_model_expr, options)

    prob = Problem(model=setup.model, obj=setup.obj, N=setup.N, dt=setup.dt)
    t = jnp.arange(setup.N) * setup.dt
    dt = jnp.full(setup.N - 1, setup.dt)
    guess = Trajectory(X=jnp.zeros((setup.N, setup.x0.shape[0])), U=jnp.asarray(setup.U0), t=t, dt=dt)
    init_traj = Trajectory(X=guess.X.at[0].set(jnp.asarray(setup.x0)), U=guess.U, t=t, dt=dt)

    final_traj, stats, status = ilqr_solve(prob, init_traj, options)

    run_solve = jl.seval("trojopt_ticket27_run_solve")
    X_jl, U_jl, J_jl, status_jl, iters_jl = run_solve(solver)
    X_jl = np.moveaxis(np.asarray(X_jl), -1, 0)
    U_jl = np.moveaxis(np.asarray(U_jl), -1, 0)

    assert int(status) == int(status_jl)
    assert int(status) == int(TerminationStatus.SOLVE_SUCCEEDED)
    assert int(stats.iterations) == int(iters_jl)
    np.testing.assert_allclose(np.asarray(final_traj.X), X_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(final_traj.U), U_jl, atol=1e-8)
    np.testing.assert_allclose(float(prob.obj.cost(final_traj)), float(J_jl), atol=1e-8)


def test_cross_ilqr_solve_pendulum(jl_altro: Any) -> None:
    _assert_solve_matches_altro(jl_altro, _pendulum_setup(), "RobotZoo.Pendulum()")


def test_cross_ilqr_solve_cartpole(jl_altro: Any) -> None:
    _assert_solve_matches_altro(jl_altro, _cartpole_setup(), "RobotZoo.Cartpole()")
