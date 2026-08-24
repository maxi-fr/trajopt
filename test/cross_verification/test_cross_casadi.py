import jax.numpy as jnp
import numpy as np
import pytest
from cross_verification.casadi_baseline import (
    assert_dual_block_parity,
    assert_parity,
    assert_setups_match,
    build_cartpole_casadi,
    build_casadi_from_problem,
    build_dubins_casadi,
)

from trajopt.benchmarks import (
    cartpole_swingup_benchmark,
    dubins_corridor_benchmark,
    quadrotor_obstacle_benchmark,
)
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import CircleConstraint, SphereConstraint
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective, TrackingObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.dubins import DubinsCar
from trajopt.models.pendulum import Pendulum
from trajopt.models.quadrotor import Quadrotor
from trajopt.problem import Problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import solve_ipopt

# Every test here solves the same problem twice, once per formulation. Unlike the Julia
# cross-verification files these carry no `julia` marker to deselect them by.
pytestmark = pytest.mark.slow


def test_cartpole_swingup_casadi_parity() -> None:
    """Verify end-to-end parity against independent CasADi baseline on Cartpole swing-up."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.01, 0.0, 0.0])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_cartpole_casadi(
        N=N,
        dt=dt,
        x0=x0,
        xf=xf,
        Q=np.diag(np.array(Q)),
        R=np.diag(np.array(R)),
        Qf=np.diag(np.array(Qf)),
        u_min=-20.0,
        u_max=20.0,
    )

    # 1. Assert setup agreement
    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    # 2. Solve both under identical Ipopt options
    solver_opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts)

    # 3. Assert full parity
    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )


def test_cartpole_with_state_limits_casadi_parity() -> None:
    """Verify parity on Cartpole swing-up with both state position limits and control bounds."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.01, 0.0, 0.0])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    x_min = [-2.0, -np.inf, -np.inf, -np.inf]
    x_max = [2.0, np.inf, np.inf, np.inf]

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(StateBound(n=n, m=m, x_min=x_min, x_max=x_max), range(N))
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_cartpole_casadi(
        N=N,
        dt=dt,
        x0=x0,
        xf=xf,
        Q=np.diag(np.array(Q)),
        R=np.diag(np.array(R)),
        Qf=np.diag(np.array(Qf)),
        u_min=-20.0,
        u_max=20.0,
        x_min=x_min,
        x_max=x_max,
    )

    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    solver_opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts)

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )


def test_dubins_car_casadi_parity() -> None:
    """Verify end-to-end parity against independent CasADi baseline on Dubins car navigation."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.1
    model = DubinsCar()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.0, 0.0])
    xf = jnp.array([2.0, 1.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Qf = jnp.diag(jnp.array([100.0, 100.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[0.0, -1.5], u_max=[2.0, 1.5]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_dubins_casadi(
        N=N,
        dt=dt,
        x0=x0,
        xf=xf,
        Q=np.diag(np.array(Q)),
        R=np.diag(np.array(R)),
        Qf=np.diag(np.array(Qf)),
        u_min=[0.0, -1.5],
        u_max=[2.0, 1.5],
    )

    # 1. Assert setup agreement
    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    # 2. Solve both
    solver_opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts)

    # 3. Assert full parity
    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )


def test_dubins_car_with_corridor_and_obstacles_casadi_parity() -> None:
    """Verify parity on Dubins car navigation with corridor bounds and circular obstacle keep-out."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 30
    dt = 0.1
    model = DubinsCar()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.0, 0.0])
    xf = jnp.array([2.0, 1.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Qf = jnp.diag(jnp.array([100.0, 100.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    x_min = [-1.0, -1.0, -np.inf]
    x_max = [3.0, 3.0, np.inf]

    obs_xc = [1.0]
    obs_yc = [-0.5]
    obs_r = [0.2]

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(StateBound(n=n, m=m, x_min=x_min, x_max=x_max), range(N))
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[0.0, -1.5], u_max=[2.0, 1.5]), range(N - 1))
    cl.add_constraint(CircleConstraint(n=n, m=m, xc=obs_xc, yc=obs_yc, radius=obs_r), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_dubins_casadi(
        N=N,
        dt=dt,
        x0=x0,
        xf=xf,
        Q=np.diag(np.array(Q)),
        R=np.diag(np.array(R)),
        Qf=np.diag(np.array(Qf)),
        u_min=[0.0, -1.5],
        u_max=[2.0, 1.5],
        x_min=x_min,
        x_max=x_max,
        obstacles=[(1.0, -0.5, 0.2)],
    )

    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    X_init = jnp.linspace(x0, xf, N)
    U_init = jnp.ones((N - 1, m)) * jnp.array([1.0, 0.0])
    dt_arr = jnp.full((N - 1,), dt)
    t_init = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dt_arr)])
    init_traj = Trajectory(X=X_init, U=U_init, t=t_init, dt=dt_arr)

    solver_opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, initial_trajectory=init_traj, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts, initial_X=np.asarray(X_init), initial_U=np.asarray(U_init))

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )


def test_dual_multipliers_parity_under_identical_solver_settings() -> None:
    """Verify that dual multipliers agree under identical constraint representations and solver settings."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 20
    dt = 0.1
    model = DubinsCar()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.0, 0.0])
    xf = jnp.array([1.5, 1.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Qf = jnp.diag(jnp.array([100.0, 100.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)

    # Solve both with tight tolerance
    solver_opts = {"max_iter": 500, "tol": 1e-9, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts)

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-6,
        tol_control=1e-6,
        tol_cost=1e-6,
        check_duals=True,
        tol_dual=1e-5,
    )


def test_cartpole_dual_multipliers_parity() -> None:
    """Verify dual multiplier agreement on Cartpole equality-constrained swing-up."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.01, 0.0, 0.0])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)

    solver_opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts)

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        check_duals=True,
        tol_dual=1e-4,
    )


def test_setup_assertion_detects_mismatch() -> None:
    """Verify that assert_setups_match raises AssertionError when problem setups differ."""
    pytest.importorskip("casadi")

    N = 20
    dt = 0.1
    model = DubinsCar()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.0, 0.0])
    xf = jnp.array([2.0, 1.0, 0.0])
    Q = jnp.eye(3)
    R = jnp.eye(2)
    Qf = jnp.eye(3)
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)
    cl = ConstraintList(n=n, m=m, N=N)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    # Mismatched horizon
    casadi_prob_diff_N = build_dubins_casadi(N=25, dt=dt, x0=x0, xf=xf)
    with pytest.raises(AssertionError, match="Horizon mismatch"):
        assert_setups_match(prob, casadi_prob_diff_N, x0=x0, dt=dt)

    # Mismatched dt
    casadi_prob_diff_dt = build_dubins_casadi(N=N, dt=0.05, x0=x0, xf=xf)
    with pytest.raises(AssertionError, match="Step duration dt mismatch"):
        assert_setups_match(prob, casadi_prob_diff_dt, x0=x0, dt=dt)

    # Mismatched initial state
    casadi_prob_diff_x0 = build_dubins_casadi(N=N, dt=dt, x0=[1.0, 0.0, 0.0], xf=xf)
    with pytest.raises(AssertionError, match="Initial state x0 mismatch"):
        assert_setups_match(prob, casadi_prob_diff_x0, x0=x0, dt=dt)


def test_automated_builder_matches_standalone_builders() -> None:
    """Verify that build_casadi_from_problem produces identical results to handwritten builders."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.01, 0.0, 0.0])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    cas_standalone = build_cartpole_casadi(
        N=N,
        dt=dt,
        x0=x0,
        xf=xf,
        Q=np.diag(np.array(Q)),
        R=np.diag(np.array(R)),
        Qf=np.diag(np.array(Qf)),
        u_min=-20.0,
        u_max=20.0,
    )
    cas_auto = build_casadi_from_problem(prob, x0=x0, dt=dt)

    opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    res_standalone = cas_standalone.solve(opts)
    res_auto = cas_auto.solve(opts)

    np.testing.assert_allclose(res_standalone.trajectory.X, res_auto.trajectory.X, atol=1e-6)
    np.testing.assert_allclose(res_standalone.trajectory.U, res_auto.trajectory.U, atol=1e-6)
    assert abs(res_standalone.cost - res_auto.cost) / res_auto.cost <= 1e-6


def test_pendulum_casadi_parity() -> None:
    """Verify parity on Pendulum swing-up model."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    model = Pendulum()
    n, m = model.n, model.m

    x0 = jnp.array([0.0, 0.0])
    xf = jnp.array([np.pi, 0.0])
    Q = jnp.diag(jnp.array([10.0, 1.0]))
    R = jnp.diag(jnp.array([0.1]))
    Qf = jnp.diag(jnp.array([100.0, 100.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)
    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, x0=x0, dt=dt, options=opts)
    casadi_res = casadi_prob.solve(opts)

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )


def test_quadrotor_obstacle_benchmark_casadi_parity() -> None:
    """Verify end-to-end parity on Quadrotor obstacle avoidance benchmark with spherical keep-out zones."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    prob, state, info = quadrotor_obstacle_benchmark(
        N=25,
        dt=0.05,
        obstacles=((1.5, 1.5, 1.5, 0.5),),
        u_max=10.0,
    )
    x0 = state.x0
    dt = float(info["dt"])

    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)
    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    X_init = state.states()
    U_init = state.controls()

    # The quaternion block of the terminal Hessian is exactly singular, so Ipopt leans on inertia
    # correction and cannot drive this problem past roughly 1e-8; the duals are compared at the
    # accuracy it does reach rather than at one it does not.
    solver_opts = {"max_iter": 500, "tol": 1e-8, "print_level": 0}
    trajopt_res = solve_ipopt(prob, state, options=solver_opts)
    casadi_res = casadi_prob.solve(
        options=solver_opts,
        initial_X=np.asarray(X_init),
        initial_U=np.asarray(U_init),
    )

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        # The two dual vectors have different lengths (361 against 457) because the formulations
        # split constraint rows from variable bounds differently, so the whole-vector comparison
        # cannot run. The block comparison below is the one that applies.
        check_duals=False,
    )
    assert_dual_block_parity(prob, trajopt_res, casadi_res, tol_dual=1e-4)


def test_dubins_corridor_benchmark_casadi_parity() -> None:
    """Verify end-to-end parity on Dubins car corridor benchmark with trajectory tracking objective."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    prob, state, info = dubins_corridor_benchmark(
        N=25,
        dt=0.1,
        y_corridor_bound=0.5,
    )
    x0 = state.x0
    dt = float(info["dt"])

    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)
    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    X_init = state.states()
    U_init = state.controls()

    # Duals converge later than primals, and at 1e-8 the costates still differ in the fourth
    # significant figure while the trajectories agree to 1e-10. Tightening the solve is what
    # makes a dual comparison measure the formulations rather than the stopping rule.
    solver_opts = {"max_iter": 500, "tol": 1e-10, "print_level": 0}
    trajopt_res = solve_ipopt(prob, state, options=solver_opts)
    casadi_res = casadi_prob.solve(
        options=solver_opts,
        initial_X=np.asarray(X_init),
        initial_U=np.asarray(U_init),
    )

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )
    assert_dual_block_parity(prob, trajopt_res, casadi_res, tol_dual=1e-4)


def test_cartpole_benchmark_casadi_parity() -> None:
    """Verify end-to-end parity on underactuated Cartpole swing-up benchmark with state position limits."""
    pytest.importorskip("casadi")
    pytest.importorskip("cyipopt")

    # The benchmark's own cart position limit, which binds, rather than a slack one that would
    # leave the whole bound path out of the comparison.
    prob, state, info = cartpole_swingup_benchmark(N=25, dt=0.05, u_bound=20.0)
    x0 = state.x0
    dt = float(info["dt"])

    casadi_prob = build_casadi_from_problem(prob, x0=x0, dt=dt)
    assert_setups_match(prob, casadi_prob, x0=x0, dt=dt)

    solver_opts = {"max_iter": 500, "tol": 1e-10, "print_level": 0}
    trajopt_res = solve_ipopt(prob, state, options=solver_opts)
    casadi_res = casadi_prob.solve(options=solver_opts)

    assert_parity(
        trajopt_res,
        casadi_res,
        tol_state=1e-5,
        tol_control=1e-5,
        tol_cost=1e-5,
        tol_feas=1e-4,
        check_duals=False,
    )
    assert_dual_block_parity(prob, trajopt_res, casadi_res, tol_dual=1e-4)
