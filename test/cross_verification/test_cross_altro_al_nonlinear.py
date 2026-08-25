from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import CircleConstraint, ConstraintList
from trajopt.solvers.al import ALConstraints, al_cost, al_grad_hess, evaluate_al_constraints, max_violation
from trajopt.trajectory import Trajectory

# Ticket 28 gap (mid-point review): the two existing AL cross-verification tests
# (test_cross_altro_al.py) only cover a ControlBound and a GoalConstraint, both affine, so
# Gauss-Newton is exact for them by construction and they cannot catch a sign/index/chain-rule
# error in the Jacobian machinery that only shows up when the constraint Jacobian itself depends
# on the state. CircleConstraint is genuinely nonlinear (its Jacobian varies with x, y), so this
# cross-checks alcost/algrad!/alhess! against Julia for that path specifically.
#
# `Altro.Problems.car_escape` cannot supply this: `altro_jl/problems/car_escape.jl` builds its
# CircleConstraint on the old `ConstraintVals`/`ConstraintSet` API and is not wired into the
# `Altro.Problems` module ALSolver expects (`Altro.Problems.car_escape` raises `UndefVarError`).
# Instead this builds a small self-contained ALSolver-compatible problem inline, following the
# same `ConstraintList` + `Problem` + `ALSolver` shape `altro_jl/problems/cartpole.jl` uses,
# swapping its bound/goal constraints for a single `CircleConstraint`.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, StaticArrays, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trajopt_ticket28_circle_problem()
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Q = 1e-2 * Diagonal(@SVector ones(n)) * dt
    Qf = 1e2 * Diagonal(@SVector ones(n))
    R = 1e-1 * Diagonal(@SVector ones(m)) * dt
    x0 = @SVector zeros(n)
    xf = @SVector [0.0, pi, 0.0, 0.0]
    model = RobotZoo.Cartpole()
    obj = TO.LQRObjective(Q, R, Qf, xf, N)

    conSet = TO.ConstraintList(n, m, N)
    circ = TO.CircleConstraint(n, [0.5], [1.5], [0.3])
    TO.add_constraint!(conSet, circ, 2:N-1)

    X0 = [@SVector fill(NaN, n) for k in 1:N]
    U0 = [@SVector fill(0.01, m) for k in 1:N-1]
    Z = TO.SampledTrajectory(X0, U0, dt=dt * ones(N - 1))
    prob = TO.Problem(model, obj, conSet, x0, xf, Z, N, 0.0, tf, integration=RD.RK3(model))
    TO.rollout!(prob)
    opts = Altro.SolverOptions(penalty_scaling=10.0, penalty_initial=1.0)
    return prob, opts
end

function trajopt_ticket28_run_al_circle(X, U, lam, mu)
    prob, opts = trajopt_ticket28_circle_problem()
    solver = Altro.ALSolver(prob, opts)
    Z = TO.get_trajectory(solver)
    N = length(Z)
    for k in 1:N
        RD.setstate!(Z[k], X[k, :])
        k < N && RD.setcontrol!(Z[k], U[k, :])
    end

    conset = TO.get_constraints(solver)
    alcon = conset.constraints[1]
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

    return J, vals, grads, hesses, viol
end
"""


def _run_altro_al_circle(
    jl: Any,
    X: np.ndarray,
    U: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
) -> tuple[Any, ...]:
    jl.seval(_ALTRO_SETUP)
    run_al = jl.seval("trajopt_ticket28_run_al_circle")
    return run_al(X, U, lam, mu)


def test_cross_al_circle_constraint_matches_altro(jl_altro: Any) -> None:
    """Gauss-Newton alcost/algrad!/alhess! match Altro for a genuinely nonlinear CircleConstraint.

    Unlike the affine bound/goal cross tests, this exercises a constraint whose Jacobian depends
    on the state, so it can actually catch a chain-rule/index/sign error in the Gauss-Newton
    machinery that an affine-only check cannot.
    """
    n, m, N = 4, 1, 101

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(CircleConstraint(n=n, xc=[0.5], yc=[1.5], radius=[0.3], m=m), range(1, N - 1))
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)

    rng = np.random.default_rng(281)
    X = rng.normal(loc=0.0, scale=1.0, size=(N, n))
    U = rng.normal(size=(N - 1, m))
    traj = Trajectory(X=jnp.asarray(X), U=jnp.asarray(U), t=jnp.zeros(N), dt=jnp.ones(N - 1))

    P = N - 2
    lam_active = rng.uniform(-1.0, 1.0, size=(P, 1))
    mu_active = rng.uniform(0.5, 2.0, size=(P, 1))
    lam = al.lam.at[1 : N - 1, 0].set(jnp.asarray(lam_active[:, 0]))
    mu = al.mu.at[1 : N - 1, 0].set(jnp.asarray(mu_active[:, 0]))
    al = ALConstraints(lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max)

    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = al_cost(al, C)
    grad_x, _, Hxx, _, _ = al_grad_hess(al, C, Jx, Ju)
    viol = max_violation(al, C)

    J_jl, vals_jl, grads_jl, hesses_jl, viol_jl = _run_altro_al_circle(jl_altro, X, U, lam_active, mu_active)

    np.testing.assert_allclose(float(cost), float(J_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(C[1 : N - 1, 0]), np.asarray(vals_jl)[0, :], atol=1e-8)
    np.testing.assert_allclose(np.asarray(grad_x[1 : N - 1]), np.asarray(grads_jl).T, atol=1e-8)
    np.testing.assert_allclose(np.asarray(Hxx[1 : N - 1]), np.moveaxis(np.asarray(hesses_jl), -1, 0), atol=1e-8)
    np.testing.assert_allclose(float(viol), float(viol_jl), atol=1e-8)
