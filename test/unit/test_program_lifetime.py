import jax.numpy as jnp
import pytest

import trajopt._env
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.models.affine import AffineModel
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import (
    SingleShooting,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_h,
    eval_jac_g,
)
from trajopt.transcription.transcription import (
    nlp_eval_f,
    nlp_eval_g,
    nlp_eval_grad_f,
    nlp_eval_h,
    nlp_eval_jac_g,
)


def _make_problem(
    N: int = 7,
    b: int = 1,
    *,
    binary: bool = False,
    with_goal: bool = False,
) -> Problem:
    """Build a linear AffineModel Problem with specified Horizon and control blocking."""
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    model = AffineModel(A=A, B=B)
    Q = jnp.eye(2)
    R = jnp.eye(1)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    binary_indices = (0,) if binary else ()
    cl = None
    if with_goal:
        clist = ConstraintList(n=2, m=1, N=N)
        clist.add_constraint(GoalConstraint(2, jnp.zeros(2)), N - 1)
        cl = clist
    return Problem(
        model=model,
        obj=obj,
        constraints=cl,
        N=N,
        dt=0.1,
        control_block_steps=b,
        binary_control_indices=binary_indices,
    )


def test_program_retains_single_ipopt_handle_across_mpc_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Program with Ipopt constructs one cyipopt Problem and reuses it across MPC cycles."""
    cyipopt = pytest.importorskip("cyipopt")

    allocations = 0
    orig_problem_cls = cyipopt.Problem

    def counted_problem(*args: object, **kwargs: object) -> object:
        nonlocal allocations
        allocations += 1
        return orig_problem_cls(*args, **kwargs)

    monkeypatch.setattr(cyipopt, "Problem", counted_problem)

    problem = _make_problem(N=7, b=1)
    solver = Ipopt(options={"print_level": 0})
    x0 = jnp.array([1.0, 0.5])
    mpc = MPC(problem, solver, x0=x0)

    res1 = mpc.solve()
    assert res1.success
    assert allocations == 1
    assert "ipopt" in mpc.program.handles

    handle = mpc.program.handles["ipopt"]
    initial_nlp = handle.nlp

    for step in range(1, 4):
        mpc.advance(1)
        mpc.measure(x0 + float(step) * 0.1)
        res = mpc.solve()
        assert res.success
        assert allocations == 1
        assert mpc.program.handles["ipopt"].nlp is initial_nlp


def test_program_retains_single_single_shooting_handle_across_mpc_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Program with SingleShooting constructs one cyipopt Problem and reuses it across MPC cycles."""
    cyipopt = pytest.importorskip("cyipopt")

    allocations = 0
    orig_problem_cls = cyipopt.Problem

    def counted_problem(*args: object, **kwargs: object) -> object:
        nonlocal allocations
        allocations += 1
        return orig_problem_cls(*args, **kwargs)

    monkeypatch.setattr(cyipopt, "Problem", counted_problem)

    problem = _make_problem(N=7, b=2)
    solver = SingleShooting(Ipopt(options={"print_level": 0}), hessian="dense")
    x0 = jnp.array([1.0, 0.5])
    mpc = MPC(problem, solver, x0=x0)

    res1 = mpc.solve()
    assert res1.success
    assert allocations == 1
    assert "single_shooting" in mpc.program.handles or "ipopt" in mpc.program.handles

    handle = mpc.program.handles.get("single_shooting") or mpc.program.handles.get("ipopt")
    assert handle is not None
    initial_nlp = handle.nlp

    for step in range(1, 4):
        mpc.advance(2)
        mpc.measure(x0 + float(step) * 0.1)
        res = mpc.solve()
        assert res.success
        assert allocations == 1
        current_handle = mpc.program.handles.get("single_shooting") or mpc.program.handles.get("ipopt")
        assert current_handle is not None
        assert current_handle.nlp is initial_nlp


def test_binary_continuation_reuses_single_handle_across_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary continuation passes execute through the single retained cyipopt handle."""
    cyipopt = pytest.importorskip("cyipopt")

    allocations = 0
    orig_problem_cls = cyipopt.Problem

    def counted_problem(*args: object, **kwargs: object) -> object:
        nonlocal allocations
        allocations += 1
        return orig_problem_cls(*args, **kwargs)

    monkeypatch.setattr(cyipopt, "Problem", counted_problem)

    problem = _make_problem(N=7, b=2, binary=True)
    solver = Ipopt(
        options={"print_level": 0},
        beta_initial=1.0,
        beta_growth=10.0,
        max_passes=3,
    )
    x0 = jnp.array([1.0, 0.0])
    mpc = MPC(problem, solver, x0=x0)

    res1 = mpc.solve()
    assert res1.info.get("homotopy_passes", 1) >= 1
    assert allocations == 1

    handle = mpc.program.handles["ipopt"]
    initial_nlp = handle.nlp

    mpc.measure(x0 + 0.1)
    res2 = mpc.solve()
    assert allocations == 1
    assert mpc.program.handles["ipopt"].nlp is initial_nlp
    assert res2.info.get("homotopy_passes", 1) >= 1


def test_failed_solve_followed_by_successful_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-solve state is initialized afresh and the handle survives a failed solve."""
    cyipopt = pytest.importorskip("cyipopt")

    allocations = 0
    orig_problem_cls = cyipopt.Problem

    def counted_problem(*args: object, **kwargs: object) -> object:
        nonlocal allocations
        allocations += 1
        return orig_problem_cls(*args, **kwargs)

    monkeypatch.setattr(cyipopt, "Problem", counted_problem)

    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=7)
    problem = Problem(model=model, obj=obj, N=7, dt=0.1)
    solver = Ipopt(options={"print_level": 0, "max_iter": 1})
    x0 = jnp.array([3.14, 0.0])
    mpc = MPC(problem, solver, x0=x0)

    res_fail = mpc.solve()
    assert not res_fail.success
    assert allocations == 1
    assert "ipopt" in mpc.program.handles

    handle = mpc.program.handles["ipopt"]
    initial_nlp = handle.nlp

    # Relax max_iter on the same external handle and solve from a mild perturbation
    handle.nlp.add_option("max_iter", 100)
    mpc.measure(jnp.array([0.1, 0.0]))
    res_success = mpc.solve()

    assert res_success.success
    assert allocations == 1
    assert mpc.program.handles["ipopt"].nlp is initial_nlp
    assert res_success.status == 0
    assert res_success.iterations > 0


def _cache_size(kernel: object) -> int:
    """Read the compilation cache size of a filter_jit wrapped kernel."""
    cached = getattr(kernel, "_cached")  # noqa: B009 -- wrapper hides dynamic attributes from static checker
    return int(cached._cache_size())  # noqa: SLF001 -- inspect JAX compilation cache directly


def test_zero_additional_jax_traces_or_compiles_after_warmup() -> None:
    """Varying boundary data, targets, and continuation beta causes zero recompilation after warm-up."""
    pytest.importorskip("cyipopt")

    problem = _make_problem(N=7, b=2, binary=True, with_goal=True)
    solver = Ipopt(
        options={"print_level": 0},
        beta_initial=1.0,
        beta_growth=5.0,
        max_passes=2,
    )
    x0 = jnp.array([1.0, 0.5])
    xf = jnp.array([0.0, 0.0])
    ref_x = jnp.ones((7, 2)) * 0.5
    ref_u = jnp.ones((6, 1)) * 0.1
    ref = Trajectory(X=ref_x, U=ref_u, t=jnp.linspace(0, 0.6, 7), dt=jnp.ones(6) * 0.1)

    mpc = MPC(problem, solver, x0=x0, xf=xf, reference=ref)

    # Warm-up solve
    mpc.solve()

    kernels = [nlp_eval_f, nlp_eval_grad_f, nlp_eval_g, nlp_eval_jac_g, nlp_eval_h]
    warmup_caches = [_cache_size(k) for k in kernels]
    assert all(c >= 1 for c in warmup_caches)

    # Vary dynamic data: x0, t0, X_ref, U_ref, xf, beta across multiple cycles
    for i in range(1, 4):
        mpc.advance(2)
        mpc.measure(x0 + float(i) * 0.2, t=float(i) * 0.2)
        mpc.set_goal(xf + float(i) * 0.05)
        mpc.push_reference(jnp.array([0.2 * i, -0.1 * i]), jnp.array([0.05 * i]))
        mpc.solve()

    post_caches = [_cache_size(k) for k in kernels]
    assert post_caches == warmup_caches


def test_zero_additional_jax_compiles_single_shooting() -> None:
    """SingleShooting causes zero recompilation after warm-up when varying boundary data and goal."""
    pytest.importorskip("cyipopt")

    problem = _make_problem(N=7, b=2, binary=False, with_goal=True)
    solver = SingleShooting(Ipopt(options={"print_level": 0}), hessian="dense")
    x0 = jnp.array([1.0, 0.5])
    xf = jnp.array([0.0, 0.0])
    ref_x = jnp.ones((7, 2)) * 0.5
    ref_u = jnp.ones((6, 1)) * 0.1
    ref = Trajectory(X=ref_x, U=ref_u, t=jnp.linspace(0, 0.6, 7), dt=jnp.ones(6) * 0.1)

    mpc = MPC(problem, solver, x0=x0, xf=xf, reference=ref)

    # Warm-up solve
    mpc.solve()

    kernels = [eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h]
    warmup_caches = [_cache_size(k) for k in kernels]
    assert all(c >= 1 for c in warmup_caches)

    for i in range(1, 4):
        mpc.advance(2)
        mpc.measure(x0 + float(i) * 0.2, t=float(i) * 0.2)
        mpc.set_goal(xf + float(i) * 0.05)
        mpc.push_reference(jnp.array([0.2 * i, -0.1 * i]), jnp.array([0.05 * i]))
        mpc.solve()

    post_caches = [_cache_size(k) for k in kernels]
    assert post_caches == warmup_caches
