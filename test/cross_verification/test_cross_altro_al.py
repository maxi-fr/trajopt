from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.solvers.al import (
    ALConstraints,
    al_cost,
    al_grad_hess,
    dual_update,
    evaluate_al_constraints,
    max_penalty,
    max_violation,
    penalty_update,
)
from trajopt.solvers.options import SolverOptions
from trajopt.trajectory import Trajectory

# Ticket 28: cross-verifies alcost/algrad!/alhess!/dualupdate!/penaltyupdate!/max_violation/
# max_penalty against a single Altro.ALConstraint pulled out of Altro.Problems.Cartpole()'s
# ALConstraintSet (bnd = control BoundConstraint, inequality; goal = GoalConstraint, equality).
# SolverOptions() defaults (penalty_initial=1, penalty_scaling=10, penalty_max=dual_max=1e8)
# match both Cartpole's own opts (penalty_scaling=10) and Altro's ConstraintOptions defaults, so
# no explicit option overrides are needed on either side.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trajopt_ticket28_run_al(problem_fn, con_index, X, U, lam, mu)
    prob, opts = problem_fn()
    solver = Altro.ALSolver(prob, opts)
    Z = TO.get_trajectory(solver)
    N = length(Z)
    for k in 1:N
        RD.setstate!(Z[k], X[k, :])
        k < N && RD.setcontrol!(Z[k], U[k, :])
    end

    conset = TO.get_constraints(solver)
    alcon = conset.constraints[con_index]
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

    c_max = zeros(N)
    Altro.normviolation!(alcon, Inf, c_max)
    viol = norm(c_max, Inf)
    mp = Altro.max_penalty(alcon)

    Altro.dualupdate!(alcon)
    Altro.penaltyupdate!(alcon)
    lam_new = cat([Vector(alcon.λ[i]) for i in 1:P]..., dims=2)
    mu_new = cat([Vector(alcon.μ[i]) for i in 1:P]..., dims=2)

    return J, vals, grads, hesses, viol, mp, lam_new, mu_new
end
"""


def _run_altro_al(
    jl: Any,
    problem_fn_name: str,
    con_index: int,
    X: np.ndarray,
    U: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
) -> tuple[Any, ...]:
    jl.seval(_ALTRO_SETUP)
    run_al = jl.seval("trajopt_ticket28_run_al")
    problem_fn = jl.seval(f"Altro.Problems.{problem_fn_name}")
    return run_al(problem_fn, con_index, X, U, lam, mu)


def test_cross_al_control_bound_matches_altro(jl_altro: Any) -> None:
    n, m, N, u_bnd = 4, 1, 101, 3.0

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)

    rng = np.random.default_rng(28)
    X = rng.normal(size=(N, n))
    U = rng.normal(size=(N - 1, m)) * u_bnd
    traj = Trajectory(X=jnp.asarray(X), U=jnp.asarray(U), t=jnp.zeros(N), dt=jnp.ones(N - 1))

    # Row layout: [constraint block (empty here) | x_upper(n) | x_lower(n) | u_upper(m) | u_lower(m)].
    u_start = al.p_cons_max + 2 * n
    lam_active = rng.uniform(0.0, 2.0, size=(N - 1, 2 * m))
    mu_active = rng.uniform(0.5, 2.0, size=(N - 1, 2 * m))
    lam = jnp.zeros_like(al.lam).at[: N - 1, u_start : u_start + 2 * m].set(jnp.asarray(lam_active))
    mu = al.mu.at[: N - 1, u_start : u_start + 2 * m].set(jnp.asarray(mu_active))
    al = ALConstraints(lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max)

    options = SolverOptions()
    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = al_cost(al, C)
    _, grad_u, _, Huu, _ = al_grad_hess(al, C, Jx, Ju)
    viol = max_violation(al, C)
    mp = max_penalty(al)
    al_dual = dual_update(al, C, options)
    al_pen = penalty_update(al, options)

    J_jl, vals_jl, grads_jl, hesses_jl, viol_jl, mp_jl, lam_new_jl, mu_new_jl = _run_altro_al(
        jl_altro, "Cartpole", 1, X, U, lam_active, mu_active
    )

    np.testing.assert_allclose(float(cost), float(J_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(C[: N - 1, u_start : u_start + 2 * m]), np.asarray(vals_jl).T, atol=1e-8)
    # getgrad/gethess restrict to the control-only input block, whose sole entry is index n (0-based).
    np.testing.assert_allclose(np.asarray(grad_u[: N - 1, 0]), np.asarray(grads_jl)[n, :], atol=1e-8)
    np.testing.assert_allclose(np.asarray(Huu[: N - 1, 0, 0]), np.asarray(hesses_jl)[n, n, :], atol=1e-8)
    np.testing.assert_allclose(float(viol), float(viol_jl), atol=1e-8)
    np.testing.assert_allclose(float(mp), float(mp_jl), atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(al_dual.lam[: N - 1, u_start : u_start + 2 * m]), np.asarray(lam_new_jl).T, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(al_pen.mu[: N - 1, u_start : u_start + 2 * m]), np.asarray(mu_new_jl).T, atol=1e-8
    )


def test_cross_al_goal_constraint_matches_altro(jl_altro: Any) -> None:
    n, m, N = 4, 1, 101
    xf = [0.0, np.pi, 0.0, 0.0]

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)

    rng = np.random.default_rng(29)
    X = rng.normal(size=(N, n))
    U = rng.normal(size=(N - 1, m))
    traj = Trajectory(X=jnp.asarray(X), U=jnp.asarray(U), t=jnp.zeros(N), dt=jnp.ones(N - 1))

    lam_active = rng.uniform(-1.0, 1.0, size=(1, n))
    mu_active = rng.uniform(0.5, 2.0, size=(1, n))
    lam = al.lam.at[N - 1, :n].set(jnp.asarray(lam_active[0]))
    mu = al.mu.at[N - 1, :n].set(jnp.asarray(mu_active[0]))
    al = ALConstraints(lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max)

    options = SolverOptions()
    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = al_cost(al, C)
    grad_x, _, Hxx, _, _ = al_grad_hess(al, C, Jx, Ju)
    viol = max_violation(al, C)
    mp = max_penalty(al)
    al_dual = dual_update(al, C, options)
    al_pen = penalty_update(al, options)

    J_jl, vals_jl, grads_jl, hesses_jl, viol_jl, mp_jl, lam_new_jl, mu_new_jl = _run_altro_al(
        jl_altro, "Cartpole", 2, X, U, lam_active, mu_active
    )

    np.testing.assert_allclose(float(cost), float(J_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(C[N - 1, :n]), np.asarray(vals_jl)[:, 0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(grad_x[N - 1]), np.asarray(grads_jl)[:, 0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(Hxx[N - 1]), np.asarray(hesses_jl)[:, :, 0], atol=1e-8)
    np.testing.assert_allclose(float(viol), float(viol_jl), atol=1e-8)
    np.testing.assert_allclose(float(mp), float(mp_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(al_dual.lam[N - 1, :n]), np.asarray(lam_new_jl)[:, 0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(al_pen.mu[N - 1, :n]), np.asarray(mu_new_jl)[:, 0], atol=1e-8)
