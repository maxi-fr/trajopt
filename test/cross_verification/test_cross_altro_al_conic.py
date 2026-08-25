from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import SecondOrderCone
from trajopt.constraints import ConstraintList, GoalConstraint, NormConstraint
from trajopt.solvers.al import (
    ALConstraints,
    conic_al_cost,
    conic_al_grad_hess,
    conic_dual_update,
    evaluate_al_constraints,
)
from trajopt.solvers.options import SolverOptions
from trajopt.trajectory import Trajectory

# Ticket 31: cross-verifies the generic conic path (`options.use_conic_cost=True`) -- Altro's
# `alcost`/`algrad!`/`alhess!`/`dualupdate!` dispatched through their `use_conic` branch rather
# than the cone-special-cased one test_cross_altro_al.py exercises -- against our
# `conic_al_cost`/`conic_al_grad_hess`/`conic_dual_update`. Two cases, matching the ticket's
# "Julia parity" section: a second-order-cone `NormConstraint` (the case the special-cased path
# cannot even express) and the terminal `GoalConstraint` (an equality/ZeroCone constraint, which
# is the case that catches finding E's sign flip -- both the special-cased and generic paths
# exist for it and must disagree in exactly the documented way).
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, StaticArrays, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trajopt_ticket31_problem(con)
    n, m, N, tf = 4, 1, 41, 2.0
    dt = tf / (N - 1)
    Q = 1e-2 * Diagonal(@SVector ones(n)) * dt
    Qf = 1e2 * Diagonal(@SVector ones(n))
    R = 1e-1 * Diagonal(@SVector ones(m)) * dt
    x0 = @SVector zeros(n)
    xf = @SVector [0.0, pi, 0.0, 0.0]
    model = RobotZoo.Cartpole()
    obj = TO.LQRObjective(Q, R, Qf, xf, N)

    conSet = TO.ConstraintList(n, m, N)
    TO.add_constraint!(conSet, con.fn(n, m, N), con.knots(N))

    X0 = [@SVector fill(NaN, n) for k in 1:N]
    U0 = [@SVector fill(0.01, m) for k in 1:N-1]
    Z = TO.SampledTrajectory(X0, U0, dt=dt * ones(N - 1))
    prob = TO.Problem(model, obj, conSet, x0, xf, Z, N, 0.0, tf, integration=RD.RK3(model))
    TO.rollout!(prob)
    opts = Altro.SolverOptions(penalty_scaling=10.0, penalty_initial=1.0)
    return prob, opts
end

function trajopt_ticket31_run_al(con, X, U, lam, mu)
    prob, opts = trajopt_ticket31_problem(con)
    solver = Altro.ALSolver(prob, opts)
    Z = TO.get_trajectory(solver)
    N = length(Z)
    for k in 1:N
        RD.setstate!(Z[k], X[k, :])
        k < N && RD.setcontrol!(Z[k], U[k, :])
    end

    conset = TO.get_constraints(solver)
    alcon = conset.constraints[1]
    alcon.opts.use_conic_cost = true
    P = length(alcon.inds)
    for i in 1:P
        alcon.λ[i] .= lam[i, :]
        alcon.μ[i] .= mu[i, :]
    end

    TO.evaluate_constraints!(alcon)
    TO.constraint_jacobians!(alcon)
    alcon.cost .= 0
    Altro.alcost(alcon)
    J = sum(alcon.cost[k] for k in alcon.inds)
    Altro.algrad!(alcon)
    Altro.alhess!(alcon)

    vals = cat([Vector(alcon.vals[i]) for i in 1:P]..., dims=2)
    grads = cat([Vector(Altro.getgrad(alcon, i)) for i in 1:P]..., dims=2)
    hesses = cat([Matrix(Altro.gethess(alcon, i)) for i in 1:P]..., dims=3)

    Altro.dualupdate!(alcon)
    lam_new = cat([Vector(alcon.λ[i]) for i in 1:P]..., dims=2)

    return J, vals, grads, hesses, lam_new
end

trajopt_ticket31_soc_con = (fn=(n, m, N) -> TO.NormConstraint(n, m, 1.0, TO.SecondOrderCone(), :state), knots=(N) -> 1:N)
trajopt_ticket31_goal_con = (fn=(n, m, N) -> TO.GoalConstraint(@SVector [0.0, pi, 0.0, 0.0]), knots=(N) -> N:N)
"""


def _run_altro_al_conic(
    jl: Any,
    con_name: str,
    X: np.ndarray,
    U: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
) -> tuple[Any, ...]:
    jl.seval(_ALTRO_SETUP)
    run_al = jl.seval("trajopt_ticket31_run_al")
    con = jl.seval(con_name)
    return run_al(con, X, U, lam, mu)


def test_cross_conic_soc_norm_constraint_matches_altro(jl_altro: Any) -> None:
    """Generic conic alcost/algrad!/alhess!/dualupdate! match Altro for a second-order-cone constraint."""
    n, m, N = 4, 1, 41

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(NormConstraint(n=n, m=m, val=1.0, sense=SecondOrderCone(), inds="state"), range(N))
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0, use_conic_cost=True)

    rng = np.random.default_rng(310)
    X = rng.normal(size=(N, n))
    U = rng.normal(size=(N - 1, m))
    traj = Trajectory(X=jnp.asarray(X), U=jnp.asarray(U), t=jnp.zeros(N), dt=jnp.ones(N - 1))

    p = n + 1
    lam_active = rng.normal(size=(N, p))
    mu_active = rng.uniform(0.5, 2.0, size=(N, p))
    lam = al.lam.at[:, :p].set(jnp.asarray(lam_active))
    mu = al.mu.at[:, :p].set(jnp.asarray(mu_active))
    al = ALConstraints(
        lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max, is_conic=True
    )

    options = SolverOptions()
    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = conic_al_cost(al, C, constraints)
    grad_x, _, Hxx, _, _ = conic_al_grad_hess(al, C, Jx, Ju, constraints)
    al_dual = conic_dual_update(al, C, constraints, options)

    J_jl, vals_jl, grads_jl, hesses_jl, lam_new_jl = _run_altro_al_conic(
        jl_altro, "trajopt_ticket31_soc_con", X, U, lam_active, mu_active
    )

    # NormConstraint is a generic StageConstraint (not narrowed to StateOnly), so Altro's
    # getgrad/gethess return the full (n+m)-wide state+control block even though this
    # constraint's Jacobian is structurally zero in the control columns (inds="state").
    np.testing.assert_allclose(float(cost), float(J_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(C[:, :p]), np.asarray(vals_jl).T, atol=1e-8)
    np.testing.assert_allclose(np.asarray(grad_x), np.asarray(grads_jl)[:n, :].T, atol=1e-8)
    np.testing.assert_allclose(np.asarray(Hxx), np.moveaxis(np.asarray(hesses_jl)[:n, :n, :], -1, 0), atol=1e-8)
    np.testing.assert_allclose(np.asarray(al_dual.lam[:, :p]), np.asarray(lam_new_jl).T, atol=1e-8)


def test_cross_conic_goal_constraint_matches_altro(jl_altro: Any) -> None:
    """Generic conic path matches Altro on the equality/ZeroCone GoalConstraint, the finding-E sign-flip case."""
    n, m, N = 4, 1, 41
    xf = [0.0, np.pi, 0.0, 0.0]

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0, use_conic_cost=True)

    rng = np.random.default_rng(311)
    X = rng.normal(size=(N, n))
    U = rng.normal(size=(N - 1, m))
    traj = Trajectory(X=jnp.asarray(X), U=jnp.asarray(U), t=jnp.zeros(N), dt=jnp.ones(N - 1))

    lam_active = rng.uniform(-1.0, 1.0, size=(1, n))
    mu_active = rng.uniform(0.5, 2.0, size=(1, n))
    lam = al.lam.at[N - 1, :n].set(jnp.asarray(lam_active[0]))
    mu = al.mu.at[N - 1, :n].set(jnp.asarray(mu_active[0]))
    al = ALConstraints(
        lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max, is_conic=True
    )

    options = SolverOptions()
    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = conic_al_cost(al, C, constraints)
    grad_x, _, Hxx, _, _ = conic_al_grad_hess(al, C, Jx, Ju, constraints)
    al_dual = conic_dual_update(al, C, constraints, options)

    J_jl, vals_jl, grads_jl, hesses_jl, lam_new_jl = _run_altro_al_conic(
        jl_altro, "trajopt_ticket31_goal_con", X, U, lam_active, mu_active
    )

    np.testing.assert_allclose(float(cost), float(J_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(C[N - 1, :n]), np.asarray(vals_jl)[:, 0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(grad_x[N - 1]), np.asarray(grads_jl)[:, 0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(Hxx[N - 1]), np.asarray(hesses_jl)[:, :, 0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(al_dual.lam[N - 1, :n]), np.asarray(lam_new_jl)[:, 0], atol=1e-8)

    # Finding E: the same constraint's non-conic dual update (lam <- lam + mu*c, no clamp needed
    # here) sign-flips relative to the conic one (lam <- Pi_{K*}(lam - mu*c) = lam - mu*c for the
    # identity-projecting dual of ZeroCone) whenever c != 0 -- verified directly, not just implied.
    c_np = np.asarray(C[N - 1, :n])
    nonconic_dual = lam_active[0] + mu_active[0] * c_np
    conic_dual = lam_active[0] - mu_active[0] * c_np
    assert not np.allclose(nonconic_dual, conic_dual, atol=1e-6)
    np.testing.assert_allclose(np.asarray(al_dual.lam[N - 1, :n]), conic_dual, atol=1e-8)
