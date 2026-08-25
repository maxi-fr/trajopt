from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models import Cartpole
from trajopt.problem import MPCState, Problem
from trajopt.solvers.al import AL
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.solvers.pn import pn_solve
from trajopt.trajectory import Trajectory

# Ticket 32, reference §8.2 row 16: the active set, the KKT step, and the per-iteration violation
# reduction against Altro.ProjectedNewtonSolver, with multiplier_projection off on both sides
# (finding: Altro's own multiplier_projection! is dead code -- see pn.py's module docstring).
# Both solvers start from the same AL-converged warm-start trajectory, matching the ticket's own
# framing of PN as a post-AL polishing phase, not a general-purpose solver.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trajopt_ticket32_setup(model, Q, R, Qf, x0, xf, N, dt, u_bnd, X0, U0, opts)
    tf = dt * (N - 1)
    obj = TO.LQRObjective(Diagonal(Q), Diagonal(R), Diagonal(Qf), xf, N)

    conSet = TO.ConstraintList(size(Q, 1), size(R, 1), N)
    bnd = TO.BoundConstraint(size(Q, 1), size(R, 1), u_min=-u_bnd, u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    TO.add_constraint!(conSet, bnd, 1:N-1)
    TO.add_constraint!(conSet, goal, N:N)

    prob = TO.Problem(model, obj, x0, tf; xf=xf, constraints=conSet,
                       X0=[copy(x) for x in X0], U0=[copy(u) for u in U0])
    pn = Altro.ProjectedNewtonSolver(prob, opts)
    # ProjectedNewtonSolver's own _data starts zeroed (finding: its constructor never seeds Zdata
    # from prob.Z); ALTROSolver's solve! seeds it via `copyto!(Zpn, Zal)` after the AL phase, so we
    # replicate that seeding step here from the X0/U0 warm start instead.
    copyto!(TO.get_trajectory(pn), prob.Z)
    return pn
end

# Mirrors pn_solve.jl's projection_solve! / _qdldl_solve! exactly, but snapshots the active set
# right after update_active_set! -- the same point _qdldl_solve! itself uses it from, before the
# refinement line search runs -- so it lines up with our own PNStats.active (`ev.active`, computed
# once per outer iteration before the KKT solve/refine).
function trajopt_ticket32_run(pn)
    Np = Altro.num_primals(pn)
    copyto!(pn.Z̄data, pn.Zdata)

    eps_feas = pn.opts.constraint_tolerance
    Altro.evaluate_constraints!(pn)
    viol = Altro.max_violation(pn, nothing)
    max_projection_iters = pn.opts.n_steps
    count = 0

    c_max_hist = Float64[]
    n_active_hist = Int[]

    while count <= max_projection_iters && viol > eps_feas
        Altro.evaluate_constraints!(pn)
        Altro.update_active_set!(pn)
        push!(n_active_hist, sum(pn.active[Np+1:end]))

        Altro._qdldl_solve!(pn)
        viol = Altro.max_violation(pn)  # re-evaluates constraints + active set on the refined Z
        push!(c_max_hist, viol)
        count += 1
    end

    N = length(pn.ix)
    Z = pn.Z
    X = cat([Vector(RD.state(Z[k])) for k = 1:N]..., dims=2)
    U = cat([Vector(RD.control(Z[k])) for k = 1:N-1]..., dims=2)
    return X, U, c_max_hist, n_active_hist, count
end
"""


def _cartpole_setup() -> tuple[int, int, int, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Shared problem constants for both sides: n, m, N, dt, Q, R, Qf, u_bnd, xf."""
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    u_bnd = 3.0
    xf = np.array([0.0, np.pi, 0.0, 0.0])
    return n, m, N, dt, Q, R, Qf, u_bnd, xf


def _python_problem() -> tuple[Problem, float, np.ndarray]:
    n, m, N, dt, Q, R, Qf, u_bnd, xf = _cartpole_setup()
    model = Cartpole()
    obj = LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), xf=jnp.asarray(xf), N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=xf.tolist()), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())
    return prob, dt, xf


def _al_warm_start(prob: Problem, dt: float) -> Trajectory:
    """Run AL to convergence and return the resulting trajectory, a realistic PN warm start."""
    n = int(prob.model.n)
    x0 = jnp.zeros(n)
    state = MPCState.initial(prob, x0=x0, dt=dt, initial_trajectory=None)
    al_result = AL(options=SolverOptions(iterations=300, iterations_outer=30)).solve(prob, state)
    assert al_result.success
    return al_result.trajectory


def _build_jl_pn_solver(
    jl: Any, options: SolverOptions, u_bnd: float, x0: np.ndarray, xf: np.ndarray, X0: np.ndarray, U0: np.ndarray
) -> Any:
    _n, _m, N, dt, Q, R, Qf, _u_bnd, _xf = _cartpole_setup()
    jl.seval(_ALTRO_SETUP)
    setup_fn = jl.seval("trajopt_ticket32_setup")
    jl_model = jl.seval("RobotZoo.Cartpole()")
    jl_opts = jl.Altro.SolverOptions(
        constraint_tolerance=float(options.constraint_tolerance),
        n_steps=int(options.n_steps),
        projected_newton_tolerance=float(options.projected_newton_tolerance),
        active_set_tolerance_pn=float(options.active_set_tolerance_pn),
        multiplier_projection=bool(options.multiplier_projection),
        ρ_primal=float(options.rho_primal),
        r_threshold=float(options.r_threshold),
    )
    return setup_fn(jl_model, Q, R, Qf, x0, xf, N, dt, u_bnd, list(X0), list(U0), jl_opts)


def test_cross_pn_solve_cartpole_matches_altro(jl_altro: Any) -> None:
    prob, dt, xf = _python_problem()
    n = int(prob.model.n)
    x0 = np.zeros(n)
    u_bnd = 3.0
    options = SolverOptions(multiplier_projection=False)

    warm_traj = _al_warm_start(prob, dt)

    solver = _build_jl_pn_solver(jl_altro, options, u_bnd, x0, xf, np.asarray(warm_traj.X), np.asarray(warm_traj.U))
    run_pn = jl_altro.seval("trajopt_ticket32_run")
    X_jl, U_jl, c_max_jl, n_active_jl, n_iter_jl = run_pn(solver)
    X_jl = np.moveaxis(np.asarray(X_jl), -1, 0)
    U_jl = np.moveaxis(np.asarray(U_jl), -1, 0)
    c_max_jl = np.asarray(c_max_jl)
    n_active_jl = np.asarray(n_active_jl)

    final_traj, stats, _duals, status = pn_solve(prob, warm_traj, jnp.asarray(x0), options)

    n_iter = int(stats.iterations)
    assert n_iter == int(n_iter_jl)

    # Finding M: n_steps=2 permits three projection solves (count <= n_steps).
    assert n_iter <= options.n_steps + 1

    np.testing.assert_allclose(np.asarray(stats.c_max[:n_iter]), c_max_jl, atol=1e-6)

    # Row order differs between PN's own layout and Altro's (finding L, a second row-ordering
    # convention) -- compare active-set size per iteration rather than a row-by-row bitmask.
    n_active_py = np.asarray(stats.active[:n_iter]).sum(axis=1)
    np.testing.assert_array_equal(n_active_py, n_active_jl)

    np.testing.assert_allclose(np.asarray(final_traj.X), X_jl, atol=1e-6)
    np.testing.assert_allclose(np.asarray(final_traj.U), U_jl, atol=1e-6)

    # Demonstration criterion: three-plus orders of magnitude violation reduction within n_steps.
    assert float(stats.c_max[n_iter - 1]) < float(stats.c_max[0]) * 1e-3 or float(stats.c_max[n_iter - 1]) < 1e-8
    assert int(status) == int(TerminationStatus.SOLVE_SUCCEEDED)
