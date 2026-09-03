import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.expansions import Expansion
from trajopt.models.cartpole import Cartpole
from trajopt.problem import MPCState, Problem
from trajopt.solvers.al import AL
from trajopt.solvers.boxqp import (
    BoxQP,
    box_qp_solve,
    extract_uniform_control_bounds,
    make_control_bound_solve_kd,
)
from trajopt.solvers.ilqr import DynamicRegularization, backward_pass
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory

XF = jnp.array([0.0, np.pi, 0.0, 0.0])


def _lqr_expansion(N: int, ne: int, m: int, *, seed: int = 0) -> Expansion:
    """Build a time-varying, regulation-about-origin (q=r=0) LQ expansion for testing.

    Mirrors `test_ilqr_backward_pass._lqr_expansion`'s random construction (not imported directly
    since `test/` is not an importable package); only the resulting `Expansion` is needed here.
    """
    rng = np.random.default_rng(seed)
    As, Bs, Qs, Rs = [], [], [], []
    for _ in range(N - 1):
        A = rng.normal(size=(ne, ne)) * 0.3 + np.eye(ne)
        B = rng.normal(size=(ne, m)) * 0.3
        Q = rng.normal(size=(ne, ne))
        Q = Q @ Q.T + np.eye(ne)
        R = rng.normal(size=(m, m))
        R = R @ R.T + np.eye(m)
        As.append(A)
        Bs.append(B)
        Qs.append(Q)
        Rs.append(R)
    Qf = rng.normal(size=(ne, ne))
    Qf = Qf @ Qf.T + np.eye(ne)

    return Expansion(
        A=jnp.asarray(np.stack(As)),
        B=jnp.asarray(np.stack(Bs)),
        q=jnp.zeros((N, ne)),
        r=jnp.zeros((N - 1, m)),
        Q=jnp.asarray(np.concatenate([np.stack(Qs), Qf[None]], axis=0)),
        R=jnp.asarray(np.stack(Rs)),
        H=jnp.zeros((N - 1, m, ne)),
    )


def _cartpole_problem(u_bnd: float = 6.0) -> tuple[Problem, jnp.ndarray, float]:
    """Cartpole swing-up with a symmetric control bound and a terminal goal constraint.

    `u_bnd` defaults to 6.0: the unconstrained swing-up needs ~7.56 in magnitude, so 6.0 is
    genuinely bound-active (forces the solver off the unconstrained optimum) while still leaving
    enough control authority for the swing-up to be feasible at all.
    """
    n, m, N, tf = 4, 1, 61, 3.0
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

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())
    return prob, x0, dt


def _control_only_problem(u_bnd: float = 3.0) -> tuple[Problem, jnp.ndarray, float]:
    """Cartpole swing-up with only a control bound, no other constraint."""
    n, m, N, tf = 4, 1, 61, 3.0
    dt = tf / (N - 1)
    Q = 1e-2 * np.ones(n) * dt
    R = 1e-1 * np.ones(m) * dt
    Qf = 1e2 * np.ones(n)
    x0 = jnp.zeros(n)
    model = Cartpole()
    obj = LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N)

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(m=m, u_min=[-u_bnd], u_max=[u_bnd], n=n), range(N - 1))

    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=RK4())
    return prob, x0, dt


def _solve_boxqp_clarabel(Quu: np.ndarray, Qu: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Solve the same box-QP with Clarabel's conic interior point method, as an independent oracle."""
    import clarabel

    m = Qu.shape[0]
    P_triu = sp.triu(sp.csc_matrix(Quu)).tocsc()
    q = np.asarray(Qu, dtype=np.float64)
    A = sp.vstack([sp.eye(m), -sp.eye(m)]).tocsc()
    b = np.concatenate([np.asarray(hi, dtype=np.float64), -np.asarray(lo, dtype=np.float64)])
    cones = [clarabel.NonnegativeConeT(2 * m)]  # ty: ignore[unresolved-attribute]

    settings = clarabel.DefaultSettings()  # ty: ignore[unresolved-attribute]
    settings.verbose = False
    solver = clarabel.DefaultSolver(P_triu, q, A, b, cones, settings)  # ty: ignore[unresolved-attribute]
    result = solver.solve()
    return np.asarray(result.x, dtype=np.float64)


@pytest.mark.parametrize("seed", range(8))
def test_box_qp_solve_matches_clarabel(seed: int) -> None:
    """box_qp_solve matches Clarabel's minimizer to 1e-8 across randomized Quu, Qu, and bounds."""
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 4))
    R = rng.normal(size=(m, m))
    Quu = R @ R.T + np.eye(m)
    Qu = rng.normal(size=m) * 3.0
    # bounds scaled to sometimes clamp every control and sometimes none, by construction.
    width = rng.choice([0.05, 0.5, 5.0])
    center = rng.normal(size=m) * 0.5
    lo = center - width
    hi = center + width

    result = box_qp_solve(jnp.asarray(Quu), jnp.asarray(Qu), jnp.asarray(lo), jnp.asarray(hi))
    x_ref = _solve_boxqp_clarabel(Quu, Qu, lo, hi)

    np.testing.assert_allclose(np.asarray(result.x), x_ref, atol=1e-8)
    free_ref = (x_ref > lo + 1e-7) & (x_ref < hi - 1e-7)
    np.testing.assert_array_equal(np.asarray(result.free), free_ref)
    assert not bool(result.failed)


def test_box_qp_solve_all_clamped() -> None:
    """Bounds so tight every control clamps: box_qp_solve matches Clarabel with an empty free set."""
    m = 3
    rng = np.random.default_rng(42)
    R = rng.normal(size=(m, m))
    Quu = R @ R.T + np.eye(m)
    Qu = np.array([5.0, -5.0, 5.0])
    lo = np.full(m, -0.01)
    hi = np.full(m, 0.01)

    result = box_qp_solve(jnp.asarray(Quu), jnp.asarray(Qu), jnp.asarray(lo), jnp.asarray(hi))
    x_ref = _solve_boxqp_clarabel(Quu, Qu, lo, hi)

    np.testing.assert_allclose(np.asarray(result.x), x_ref, atol=1e-8)
    assert not bool(jnp.any(result.free))


def test_box_qp_solve_none_clamped() -> None:
    """Bounds wide enough that nothing clamps: box_qp_solve matches Clarabel with a full free set."""
    m = 3
    rng = np.random.default_rng(7)
    R = rng.normal(size=(m, m))
    Quu = R @ R.T + np.eye(m)
    Qu = rng.normal(size=m)
    lo = np.full(m, -1e6)
    hi = np.full(m, 1e6)

    result = box_qp_solve(jnp.asarray(Quu), jnp.asarray(Qu), jnp.asarray(lo), jnp.asarray(hi))
    x_ref = _solve_boxqp_clarabel(Quu, Qu, lo, hi)

    np.testing.assert_allclose(np.asarray(result.x), x_ref, atol=1e-8)
    assert bool(jnp.all(result.free))


def test_box_qp_backward_pass_matches_unconstrained_with_wide_bounds() -> None:
    """With bounds wide enough to be inactive, the box-QP path reproduces the unconstrained backward pass to 1e-10."""
    N, ne, m = 6, 3, 2
    exp = _lqr_expansion(N, ne, m, seed=0)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    result_unconstrained = backward_pass(exp, reg, options)

    lo = jnp.full((m,), -1e8)
    hi = jnp.full((m,), 1e8)
    traj = Trajectory(X=jnp.zeros((N, ne)), U=jnp.zeros((N - 1, m)), t=jnp.zeros(N), dt=jnp.ones(N - 1))
    solve_kd = make_control_bound_solve_kd(lo, hi)(traj)
    result_boxqp = backward_pass(exp, reg, options, solve_kd)

    np.testing.assert_allclose(np.asarray(result_boxqp.K), np.asarray(result_unconstrained.K), atol=1e-10)
    np.testing.assert_allclose(np.asarray(result_boxqp.d), np.asarray(result_unconstrained.d), atol=1e-10)
    assert not bool(result_boxqp.failed)


def test_box_qp_solve_kd_zeros_k_rows_for_clamped_controls() -> None:
    """Rows of K for clamped controls are exactly zero -- a clamped control does not respond to state deviation."""
    m, ne = 2, 3
    rng = np.random.default_rng(11)
    # Quu = I decouples the two controls, so each one's unconstrained optimum is just -Qu[i]:
    # component 0's is 10 (clamps hard at the tiny upper bound), component 1's is -0.1 (free).
    Quu = jnp.eye(m)
    Qux = jnp.asarray(rng.normal(size=(m, ne)))
    Qu = jnp.asarray([-10.0, 0.1])
    lo = jnp.asarray([-1e6, -1e6])
    hi = jnp.asarray([0.01, 1e6])

    traj = Trajectory(X=jnp.zeros((2, ne)), U=jnp.zeros((1, m)), t=jnp.zeros(2), dt=jnp.ones(1))
    solve_kd = make_control_bound_solve_kd(lo, hi)(traj)
    K_k, d_k, failed = solve_kd(jnp.int32(0), Quu, Qux, Qu)

    assert not bool(failed)
    np.testing.assert_array_equal(np.asarray(K_k[0]), np.zeros(ne))
    assert bool(jnp.any(K_k[1] != 0.0))
    assert float(d_k[0]) == pytest.approx(0.01, abs=1e-8)


def test_box_qp_backward_pass_indefinite_quu_terminates() -> None:
    """The rho-retry loop and bp_reg_max bound still wrap the box-QP backward pass on an indefinite Quu."""
    N, ne, m = 3, 2, 1
    exp = Expansion(
        A=jnp.asarray(np.stack([np.eye(ne)] * (N - 1))),
        B=jnp.zeros((N - 1, ne, m)),
        q=jnp.zeros((N, ne)),
        r=jnp.zeros((N - 1, m)),
        Q=jnp.asarray(np.stack([np.eye(ne)] * N)),
        R=jnp.asarray(np.stack([-1.0e12 * np.eye(m)] * (N - 1))),
        H=jnp.zeros((N - 1, m, ne)),
    )
    options = SolverOptions(bp_reg_max=1.0)
    reg = DynamicRegularization.initial(options)

    lo = jnp.full((m,), -3.0)
    hi = jnp.full((m,), 3.0)
    traj = Trajectory(X=jnp.zeros((N, ne)), U=jnp.zeros((N - 1, m)), t=jnp.zeros(N), dt=jnp.ones(N - 1))
    solve_kd = make_control_bound_solve_kd(lo, hi)(traj)

    jitted = jax.jit(functools.partial(backward_pass, options=options, solve_kd=solve_kd))
    result = jitted(exp, reg)

    assert bool(result.failed)
    assert float(result.regularization.rho) > options.bp_reg_max


def test_box_qp_backward_pass_compile_time_recorded() -> None:
    """Compile time for the box-QP's nested traced while_loop (inside backward_pass's scan) is measured and printed."""
    N, ne, m = 6, 3, 2
    exp = _lqr_expansion(N, ne, m, seed=1)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)
    lo = jnp.full((m,), -3.0)
    hi = jnp.full((m,), 3.0)
    traj = Trajectory(X=jnp.zeros((N, ne)), U=jnp.zeros((N - 1, m)), t=jnp.zeros(N), dt=jnp.ones(N - 1))
    solve_kd = make_control_bound_solve_kd(lo, hi)(traj)

    jitted = jax.jit(functools.partial(backward_pass, options=options, solve_kd=solve_kd))
    start = time.perf_counter()
    lowered = jitted.lower(exp, reg)
    compiled = lowered.compile()
    elapsed = time.perf_counter() - start
    print(f"\nbox-QP backward_pass compile time: {elapsed:.2f}s")  # noqa: T201 -- ticket 30 requires recording this

    result = compiled(exp, reg)
    assert not bool(result.failed)
    # generous soft bound: the point is to record the number, not to gate the build on a machine-specific budget.
    assert elapsed < 300.0


def test_extract_uniform_control_bounds_raises_on_per_knot_variation() -> None:
    """Requesting box-QP for a problem whose control bounds vary per knot raises at build time, naming the problem."""
    n, m, N = 4, 1, 21
    clist = ConstraintList(n=n, m=m, N=N)
    half = (N - 1) // 2
    clist.add_constraint(ControlBound(m=m, u_min=[-3.0], u_max=[3.0], n=n), range(half))
    clist.add_constraint(ControlBound(m=m, u_min=[-5.0], u_max=[5.0], n=n), range(half, N - 1))
    built = clist.build()

    with pytest.raises(ValueError, match="uniform across the whole horizon"):
        extract_uniform_control_bounds(built)


@pytest.mark.slow
def test_boxqp_control_only_solves_within_bounds_even_early_not_only_at_convergence() -> None:
    """A control-bounded cartpole solved by BoxQP keeps every rolled-out control inside bounds, even far from convergence."""
    u_bnd = 3.0
    prob, x0, dt = _control_only_problem(u_bnd=u_bnd)
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)

    for n_iters in (1, 2, 5, 300):
        options = SolverOptions(iterations=n_iters, iterations_outer=1)
        result = BoxQP(options=options).solve(prob, state)
        assert bool(jnp.all(jnp.abs(result.trajectory.U) <= u_bnd + 1e-6)), f"violated at iterations={n_iters}"


@pytest.mark.slow
def test_boxqp_with_goal_constraint_converges_and_respects_bounds() -> None:
    """Control bounds route to box-QP while the goal constraint still goes through the AL outer loop."""
    u_bnd = 6.0
    prob, x0, dt = _cartpole_problem(u_bnd=u_bnd)
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=30)

    result = BoxQP(options=options).solve(prob, state)

    assert result.success
    assert result.status == int(TerminationStatus.SOLVE_SUCCEEDED)
    np.testing.assert_allclose(np.asarray(result.trajectory.X[-1]), np.asarray(XF), atol=1e-3)
    assert bool(jnp.all(jnp.abs(result.trajectory.U) <= u_bnd + 1e-6))


@pytest.mark.slow
def test_boxqp_and_al_reach_comparable_final_cost() -> None:
    """Documentation test: the same bound-and-goal-constrained cartpole via AL (ticket 29) and box-QP (ticket 30).

    Both approaches enforce the same control bound by different means -- AL by penalizing
    violations down over outer iterations, box-QP exactly at every backward pass -- so their
    final base (non-AL-augmented) trajectory costs should agree closely once both have converged.
    """
    prob, x0, dt = _cartpole_problem()
    state = MPCState.initial(prob, x0=x0, dt=dt, xf=XF, initial_trajectory=None)
    options = SolverOptions(iterations=300, iterations_outer=30)

    al_result = AL(options=options).solve(prob, state)
    boxqp_result = BoxQP(options=options).solve(prob, state)

    assert al_result.success
    assert boxqp_result.success

    al_base_cost = float(prob.obj.cost(al_result.trajectory))
    boxqp_base_cost = float(prob.obj.cost(boxqp_result.trajectory))
    print(f"\nAL base cost: {al_base_cost:.6f}, box-QP base cost: {boxqp_base_cost:.6f}")  # noqa: T201 -- ticket 30 asks this comparison be recorded

    rel_diff = abs(al_base_cost - boxqp_base_cost) / max(abs(al_base_cost), 1e-8)
    assert rel_diff < 0.1
