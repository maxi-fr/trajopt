import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, SecondOrderCone, ZeroCone
from trajopt.constraints import ConstraintList, GoalConstraint, NormConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.problem import MPCState, Problem
from trajopt.solvers.al import (
    AL,
    ALConstraints,
    conic_al_cost,
    conic_al_grad_hess,
    conic_dual_update,
    dual_update,
    evaluate_al_constraints,
)
from trajopt.solvers.options import SolverOptions
from trajopt.trajectory import Trajectory


def _random_trajectory(n: int, m: int, N: int, seed: int) -> Trajectory:
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(N, n)))
    U = jnp.asarray(rng.normal(size=(N - 1, m)))
    t = jnp.arange(N, dtype=jnp.float64)
    dt = jnp.ones(N - 1, dtype=jnp.float64)
    return Trajectory(X=X, U=U, t=t, dt=dt)


def _soc_problem(n: int, m: int, N: int, val: float = 1.0) -> ConstraintList:
    """ConstraintList with a state-only second-order-cone norm constraint at every knot."""
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(NormConstraint(n=n, m=m, val=val, sense=SecondOrderCone(), inds="state"), range(N))
    return clist


def test_conic_al_cost_matches_manual_soc_formula() -> None:
    """conic_al_cost matches Altro's generic `alcost`, computed independently via `cone.dual().project`."""
    n, m, N = 2, 1, 3
    clist = _soc_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0, use_conic_cost=True)

    traj = _random_trajectory(n, m, N, seed=10)
    rng = np.random.default_rng(11)
    lam = jnp.asarray(rng.normal(size=al.lam.shape)) * al.row_mask
    mu = jnp.where(al.row_mask, jnp.asarray(rng.uniform(0.5, 2.0, size=al.mu.shape)), 0.0)
    al = ALConstraints(
        lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max, is_conic=True
    )

    C, _Jx, _Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = conic_al_cost(al, C, constraints)

    cone = SecondOrderCone()
    dual_cone = cone.dual()
    p = n + 1
    expected = 0.0
    for k in range(N):
        lam_k = np.asarray(al.lam[k, :p])
        mu_k = np.asarray(al.mu[k, :p])
        c_k = np.asarray(C[k, :p])
        lam_bar = lam_k - mu_k * c_k
        lam_p = np.asarray(dual_cone.project(jnp.asarray(lam_bar)))
        mu_inv = 1.0 / mu_k
        expected += 0.5 * (np.sum(lam_p * mu_inv * lam_p) - np.sum(lam_k * mu_inv * lam_k))

    np.testing.assert_allclose(float(cost), expected, atol=1e-10)


def test_conic_hessian_matches_finite_difference() -> None:
    """conic_al_grad_hess's Hessian, including the second-order projection term, matches jax.hessian exactly."""
    n, m, N = 2, 1, 3
    clist = _soc_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0, use_conic_cost=True)

    traj = _random_trajectory(n, m, N, seed=12)
    rng = np.random.default_rng(13)
    lam = jnp.asarray(rng.normal(size=al.lam.shape)) * al.row_mask
    mu = jnp.where(al.row_mask, jnp.asarray(rng.uniform(0.5, 2.0, size=al.mu.shape)), 0.0)
    al = ALConstraints(
        lam=lam, mu=mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max, is_conic=True
    )

    knot = 1

    def penalty_at_knot_x(x_k: jax.Array) -> jax.Array:
        traj_pert = Trajectory(X=traj.X.at[knot].set(x_k), U=traj.U, t=traj.t, dt=traj.dt)
        C, _, _ = evaluate_al_constraints(al, constraints, model=None, traj=traj_pert)
        return conic_al_cost(al, C, constraints)

    exact_hess = jax.hessian(penalty_at_knot_x)(traj.X[knot])
    exact_grad = jax.grad(penalty_at_knot_x)(traj.X[knot])

    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    grad_x, _, Hxx, _, _ = conic_al_grad_hess(al, C, Jx, Ju, constraints)

    np.testing.assert_allclose(np.asarray(grad_x[knot]), np.asarray(exact_grad), atol=1e-8)
    np.testing.assert_allclose(np.asarray(Hxx[knot]), np.asarray(exact_hess), atol=1e-8)

    # And the second-order term is not a no-op here: the pure Gauss-Newton (first-order-only)
    # Hessian J' diag(...) J would be rank-(n+1) at best and generally differs from the exact one.
    gn_only = jnp.einsum("pe,p,pf->ef", Jx[knot, : n + 1], jnp.ones(n + 1), Jx[knot, : n + 1])
    assert not np.allclose(np.asarray(Hxx[knot]), np.asarray(gn_only), atol=1e-6)


def test_conic_dual_update_matches_altro_formula() -> None:
    """conic_dual_update reproduces `lam <- Pi_{K*}(lam - mu*c)`, clamped, for a mixed SOC/box-bound layout."""
    n, m, N = 2, 1, 3
    clist = _soc_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0, use_conic_cost=True)
    options = SolverOptions(dual_max=0.5)

    traj = _random_trajectory(n, m, N, seed=14)
    rng = np.random.default_rng(15)
    lam = jnp.asarray(rng.normal(size=al.lam.shape)) * al.row_mask
    al = ALConstraints(
        lam=lam, mu=al.mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max, is_conic=True
    )

    C, _, _ = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    updated = conic_dual_update(al, C, constraints, options)

    cone = SecondOrderCone()
    dual_cone = cone.dual()
    p = n + 1
    expected = np.zeros_like(np.asarray(al.lam))
    for k in range(N):
        lam_bar = np.asarray(al.lam[k, :p]) - np.asarray(al.mu[k, :p]) * np.asarray(C[k, :p])
        expected[k, :p] = np.asarray(dual_cone.project(jnp.asarray(lam_bar)))

    box_cone = NegativeOrthant().dual()
    box_rows = slice(al.p_cons_max, al.p_cons_max + 2 * n + 2 * m)
    lam_bar_box = np.asarray(al.lam[:, box_rows]) - np.asarray(al.mu[:, box_rows]) * np.asarray(C[:, box_rows])
    expected[:, box_rows] = np.asarray(jax.vmap(box_cone.project)(jnp.asarray(lam_bar_box)))

    expected = np.clip(expected, -options.dual_max, options.dual_max)
    expected = np.where(np.asarray(al.row_mask), expected, 0.0)

    np.testing.assert_allclose(np.asarray(updated.lam), expected, atol=1e-10)
    assert bool(updated.is_conic)

    # dual_update dispatches identically when options.use_conic_cost is set.
    dispatched = dual_update(al, C, SolverOptions(dual_max=0.5, use_conic_cost=True), constraints)
    np.testing.assert_allclose(np.asarray(dispatched.lam), np.asarray(updated.lam), atol=1e-12)


def test_dual_update_requires_constraints_when_conic() -> None:
    """dual_update raises rather than silently ignoring options.use_conic_cost=True without `constraints`."""
    n, m, N = 2, 1, 2
    clist = ConstraintList(n=n, m=m, N=N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0, use_conic_cost=True)
    traj = _random_trajectory(n, m, N, seed=16)
    C, _, _ = evaluate_al_constraints(al, constraints, model=None, traj=traj)

    with pytest.raises(ValueError, match="requires `constraints`"):
        dual_update(al, C, SolverOptions(use_conic_cost=True))


def _small_goal_only_problem() -> tuple[Problem, jax.Array, float, jax.Array]:
    """Small cartpole swing-up with only a terminal (equality) goal constraint, for fast AL solves."""
    n, m, N, tf = 4, 1, 41, 2.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    x0 = jnp.zeros(n)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    model = Cartpole()
    obj = LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(GoalConstraint(n=n, xf=xf.tolist()), N - 1)

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())
    return prob, x0, dt, xf


def test_use_conic_cost_switch_on_warm_started_duals_raises_unless_reset() -> None:
    """Switching options.use_conic_cost with a prior state.al raises (finding E), unless reset_duals discards it."""
    prob, x0, dt, xf = _small_goal_only_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=xf, initial_trajectory=None)

    non_conic_result = AL(options=SolverOptions(iterations=20, iterations_outer=3, use_conic_cost=False)).solve(
        prob, state
    )
    warm_state = MPCState.initial(prob, x0=x0, dt=dt, xf=xf, initial_trajectory=non_conic_result.trajectory)
    state_with_duals = dataclasses.replace(warm_state, al=non_conic_result.al)

    with pytest.raises(ValueError, match="use_conic_cost"):
        AL(options=SolverOptions(iterations=20, iterations_outer=3, use_conic_cost=True, reset_duals=False)).solve(
            prob, state_with_duals
        )

    # reset_duals=True discards the mismatched duals instead of raising.
    result = AL(options=SolverOptions(iterations=20, iterations_outer=3, use_conic_cost=True, reset_duals=True)).solve(
        prob, state_with_duals
    )
    assert result.al is not None
    assert bool(result.al.is_conic)


@pytest.mark.slow
def test_equality_constraint_conic_and_nonconic_converge_to_same_kkt_point() -> None:
    """Both penalty paths reach the same primal optimum from the same start; duals negate (finding E).

    At convergence (c ~ 0) the non-conic gradient contribution is `J' * lam_nonconic` while the
    conic one is `-J' * lam_conic` (finding D's Iu factor collapses `dproj` to identity for the
    equality/ZeroCone rows here); both solves stopping at the same stationary point therefore
    requires `lam_conic ~ -lam_nonconic`, not merely that both happen to be small.
    """
    prob, x0, dt, xf = _small_goal_only_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=xf, initial_trajectory=None)

    result_nonconic = AL(options=SolverOptions(use_conic_cost=False, iterations=150, iterations_outer=25)).solve(
        prob, state
    )
    result_conic = AL(options=SolverOptions(use_conic_cost=True, iterations=150, iterations_outer=25)).solve(
        prob, state
    )

    assert result_nonconic.success
    assert result_conic.success

    np.testing.assert_allclose(
        np.asarray(result_nonconic.trajectory.X), np.asarray(result_conic.trajectory.X), atol=1e-2
    )
    np.testing.assert_allclose(np.asarray(result_nonconic.trajectory.X[-1]), np.asarray(xf), atol=1e-2)

    n = prob.model.n
    assert result_nonconic.al is not None
    assert result_conic.al is not None
    lam_nonconic = np.asarray(result_nonconic.al.lam[-1, :n])
    lam_conic = np.asarray(result_conic.al.lam[-1, :n])

    # The sign relationship, not just closeness to zero: negating one must match the other.
    np.testing.assert_allclose(lam_conic, -lam_nonconic, atol=5e-2)
    assert np.max(np.abs(lam_nonconic)) > 1e-2, "test is vacuous if the multiplier never grows"


def test_zero_cone_is_dual_of_identity_cone_used_by_equality_rows() -> None:
    """Sanity check underlying the equality-row derivation above: ZeroCone's dual is the identity cone."""
    x = jnp.array([0.3, -1.2, 5.0])
    zero_dual = ZeroCone().dual()
    np.testing.assert_allclose(np.asarray(zero_dual.project(x)), np.asarray(x), atol=1e-12)
