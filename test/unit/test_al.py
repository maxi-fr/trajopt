import jax
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


def _random_trajectory(n: int, m: int, N: int, seed: int) -> Trajectory:
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(N, n)))
    U = jnp.asarray(rng.normal(size=(N - 1, m)))
    t = jnp.arange(N, dtype=jnp.float64)
    dt = jnp.ones(N - 1, dtype=jnp.float64)
    return Trajectory(X=X, U=U, t=t, dt=dt)


def _bound_and_goal_problem(n: int, m: int, N: int) -> tuple[ConstraintList, jax.Array]:
    """ConstraintList with a control bound at every stage and a terminal goal constraint."""
    xf = np.arange(n, dtype=float)
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-1.0] * m, u_max=[1.0] * m, n=n), range(N - 1))
    clist.add_constraint(GoalConstraint(n=n, xf=xf.tolist()), N - 1)
    return clist, jnp.asarray(xf)


def test_al_constraints_build_row_mask_and_is_equality() -> None:
    n, m, N = 3, 2, 5
    clist, _ = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=2.0)

    # ControlBound is a box bound, hoisted by `ConstraintList.build` out of the knot evaluators
    # and into `u_upper`/`u_lower` -- so the transcribed constraint-only block holds just the
    # terminal GoalConstraint (p=n=3), and the control bound rows live in the padded box block.
    assert al.p_cons_max == n
    assert not bool(jnp.any(al.row_mask[: N - 1, :n]))
    assert bool(jnp.all(al.row_mask[N - 1, :n]))
    assert bool(jnp.all(al.is_equality[N - 1, :n]))

    u_bound_start = al.p_cons_max + 2 * n
    u_bound_end = u_bound_start + 2 * m
    assert bool(jnp.all(al.row_mask[: N - 1, u_bound_start:u_bound_end]))
    assert not bool(jnp.any(al.is_equality[: N - 1, u_bound_start:u_bound_end]))
    # Terminal knot has no control, so its u-bound rows are masked out.
    assert not bool(jnp.any(al.row_mask[N - 1, u_bound_start:u_bound_end]))

    # lambda starts at zero, mu starts at penalty_initial on real rows only.
    assert bool(jnp.all(al.lam == 0.0))
    assert bool(jnp.all(al.mu[al.row_mask] == 2.0))
    assert bool(jnp.all(al.mu[~al.row_mask] == 0.0))


def test_al_cost_matches_manual_equality_and_inequality() -> None:
    n, m, N = 3, 2, 4
    clist, xf = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.5)

    traj = _random_trajectory(n, m, N, seed=0)
    rng = np.random.default_rng(1)
    lam = jnp.asarray(rng.normal(size=al.lam.shape)) * al.row_mask
    al = ALConstraints(lam=lam, mu=al.mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max)

    C, _Jx, _Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = al_cost(al, C)

    # Manual reference: inequality rows are active when c >= 0 or lambda > 0, equality rows always.
    lam_np, mu_np, c_np = np.asarray(al.lam), np.asarray(al.mu), np.asarray(C)
    is_eq_np, mask_np = np.asarray(al.is_equality), np.asarray(al.row_mask)
    active = is_eq_np | (c_np >= 0.0) | (lam_np > 0.0)
    a_np = np.where(active & mask_np, mu_np, 0.0)
    expected = np.sum(np.where(mask_np, lam_np * c_np + 0.5 * a_np * c_np**2, 0.0))

    np.testing.assert_allclose(float(cost), expected, atol=1e-10)

    # And the goal-constraint rows' residual is exactly x[-1] - xf (affine, equality).
    np.testing.assert_allclose(np.asarray(C[-1, :n]), np.asarray(traj.X[-1] - xf), atol=1e-10)


def test_gn_hessian_matches_exact_hessian_for_affine_constraint() -> None:
    """Reference sec 7.1: GN Hessian J' diag(a) J is exact for affine constraints."""
    n, m, N = 3, 2, 4
    clist, _xf = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.5)

    traj = _random_trajectory(n, m, N, seed=2)
    rng = np.random.default_rng(3)
    lam = jnp.asarray(rng.normal(size=al.lam.shape)) * al.row_mask
    al = ALConstraints(lam=lam, mu=al.mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max)

    def penalty_at_terminal_x(x_term: jax.Array) -> jax.Array:
        traj_pert = Trajectory(X=traj.X.at[-1].set(x_term), U=traj.U, t=traj.t, dt=traj.dt)
        C, _, _ = evaluate_al_constraints(al, constraints, model=None, traj=traj_pert)
        return al_cost(al, C)

    exact_hess = jax.hessian(penalty_at_terminal_x)(traj.X[-1])

    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    _, _, Hxx, _, _ = al_grad_hess(al, C, Jx, Ju)

    np.testing.assert_allclose(np.asarray(Hxx[-1]), np.asarray(exact_hess), atol=1e-12)


def test_jaxpr_size_independent_of_horizon() -> None:
    """The traced eval+cost jaxpr must not grow with N: `groups` is bounded by structure, not N."""
    n, m = 3, 2

    def jaxpr_eqn_count(nn: int) -> int:
        clist, _ = _bound_and_goal_problem(n, m, nn)
        constraints = clist.build()
        al = ALConstraints.build(constraints)
        traj = _random_trajectory(n, m, nn, seed=0)

        def evaluate(traj: Trajectory) -> jax.Array:
            C, _Jx, _Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
            return al_cost(al, C)

        return len(jax.make_jaxpr(evaluate)(traj).eqns)

    assert jaxpr_eqn_count(5) == jaxpr_eqn_count(50)


def test_masked_rows_are_inert() -> None:
    """Extra masked padding columns, even carrying adversarial garbage, affect no reduction or update."""
    n, m, N = 2, 1, 3
    clist, _xf = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)
    traj = _random_trajectory(n, m, N, seed=4)
    options = SolverOptions()

    C, Jx, Ju = evaluate_al_constraints(al, constraints, model=None, traj=traj)
    cost = al_cost(al, C)
    _, _, Hxx, Huu, Hux = al_grad_hess(al, C, Jx, Ju)
    viol = max_violation(al, C)

    pad = 2

    def pad2d(arr: jax.Array, fill: float) -> jax.Array:
        extra = jnp.full((arr.shape[0], pad), fill, dtype=arr.dtype)
        return jnp.concatenate([arr, extra], axis=1)

    def pad3d(arr: jax.Array, fill: float) -> jax.Array:
        extra = jnp.full((arr.shape[0], pad, arr.shape[2]), fill, dtype=arr.dtype)
        return jnp.concatenate([arr, extra], axis=1)

    C_p = pad2d(C, 123.0)
    Jx_p = pad3d(Jx, 7.0)
    Ju_p = pad3d(Ju, -3.0)
    al_p = ALConstraints(
        lam=pad2d(al.lam, 456.0),
        mu=pad2d(al.mu, 789.0),
        row_mask=pad2d(al.row_mask, 0.0),
        is_equality=pad2d(al.is_equality, 1.0),
        p_cons_max=al.p_cons_max,
    )

    cost_p = al_cost(al_p, C_p)
    _, _, Hxx_p, Huu_p, Hux_p = al_grad_hess(al_p, C_p, Jx_p, Ju_p)
    viol_p = max_violation(al_p, C_p)

    np.testing.assert_allclose(float(cost_p), float(cost), atol=1e-10)
    np.testing.assert_allclose(np.asarray(Hxx_p), np.asarray(Hxx), atol=1e-10)
    np.testing.assert_allclose(np.asarray(Huu_p), np.asarray(Huu), atol=1e-10)
    np.testing.assert_allclose(np.asarray(Hux_p), np.asarray(Hux), atol=1e-10)
    np.testing.assert_allclose(float(viol_p), float(viol), atol=1e-10)

    # Updates must also leave the garbage-filled masked rows inert: dual clamps to 0, penalty freezes.
    dual_p = dual_update(al_p, C_p, options)
    pen_p = penalty_update(al_p, options)
    assert bool(jnp.all(dual_p.lam[:, -pad:] == 0.0))
    assert bool(jnp.all(pen_p.mu[:, -pad:] == al_p.mu[:, -pad:]))


def test_dual_update_matches_altro_formula() -> None:
    n, m, N = 3, 2, 4
    clist, _ = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)
    options = SolverOptions(dual_max=0.1)

    traj = _random_trajectory(n, m, N, seed=5)
    C, _, _ = evaluate_al_constraints(al, constraints, model=None, traj=traj)

    updated = dual_update(al, C, options)

    raw = np.asarray(al.lam) + np.asarray(al.mu) * np.asarray(C)
    is_eq = np.asarray(al.is_equality)
    expected = np.where(is_eq, raw, np.maximum(0.0, raw))
    expected = np.clip(expected, -options.dual_max, options.dual_max)
    expected = np.where(np.asarray(al.row_mask), expected, 0.0)

    np.testing.assert_allclose(np.asarray(updated.lam), expected, atol=1e-10)
    # Inequality rows never go negative.
    assert bool(jnp.all(updated.lam[~al.is_equality] >= 0.0))


def test_penalty_update_matches_altro_formula() -> None:
    n, m, N = 3, 2, 4
    clist, _ = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)
    options = SolverOptions(penalty_scaling=10.0, penalty_max=5.0)

    updated = penalty_update(al, options)

    expected = np.clip(np.asarray(al.mu) * options.penalty_scaling, 0.0, options.penalty_max)
    expected = np.where(np.asarray(al.row_mask), expected, np.asarray(al.mu))

    np.testing.assert_allclose(np.asarray(updated.mu), expected, atol=1e-10)
    assert bool(jnp.all(updated.mu <= options.penalty_max))


def test_max_violation_and_max_penalty() -> None:
    n, m, N = 3, 2, 4
    clist, _xf = _bound_and_goal_problem(n, m, N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=3.0)

    traj = _random_trajectory(n, m, N, seed=6)
    C, _, _ = evaluate_al_constraints(al, constraints, model=None, traj=traj)

    viol = max_violation(al, C)
    mu_max = max_penalty(al)

    is_eq = np.asarray(al.is_equality)
    mask = np.asarray(al.row_mask)
    c_np = np.asarray(C)
    expected_viol_rows = np.where(is_eq, -c_np, -np.maximum(0.0, c_np))
    expected_viol = np.max(np.abs(np.where(mask, expected_viol_rows, 0.0)))

    np.testing.assert_allclose(float(viol), expected_viol, atol=1e-10)
    assert float(mu_max) == pytest.approx(3.0)


def test_max_penalty_no_rows_is_zero() -> None:
    n, m, N = 2, 1, 3
    clist = ConstraintList(n=n, m=m, N=N)
    constraints = clist.build()
    al = ALConstraints.build(constraints, penalty_initial=1.0)

    assert not bool(jnp.any(al.row_mask))
    assert float(max_penalty(al)) == 0.0
