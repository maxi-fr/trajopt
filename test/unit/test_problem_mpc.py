import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective, Objective, TrackingObjective
from trajopt.costs.rotations import QuatGeodesicCost
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.dubins import DubinsCar
from trajopt.models.pendulum import Pendulum
from trajopt.models.quadrotor import Quadrotor
from trajopt.mpc import MPC
from trajopt.problem import (
    BoundaryConditions,
    Problem,
)
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.transcription import (
    constraints_and_jac,
    eval_f,
    eval_grad_f,
    eval_h,
)


def _window(mpc: MPC) -> tuple[jax.Array, jax.Array]:
    """Return the driver's reference window, asserting it tracks one at all."""
    X_ref, U_ref = mpc.bc.X_ref, mpc.bc.U_ref
    assert X_ref is not None
    assert U_ref is not None
    return X_ref, U_ref


def _tracking_problem() -> tuple[Problem, Trajectory]:
    """Build a Dubins-style problem whose objective tracks a reference and whose goal is a constraint."""
    model = DubinsCar()
    n, m, N, dt = model.n, model.m, 8, 0.1
    xf = jnp.array([2.0, 0.0, 0.0])

    t_arr = jnp.linspace(0.0, (N - 1) * dt, N)
    dt_arr = jnp.full((N - 1,), dt)
    X_ref = jnp.zeros((N, n)).at[:, 0].set(jnp.linspace(0.0, float(xf[0]), N))
    U_ref = jnp.ones((N - 1, m)) * jnp.array([float(xf[0]) / ((N - 1) * dt), 0.0])
    ref = Trajectory(X=X_ref, U=U_ref, t=t_arr, dt=dt_arr)

    obj = TrackingObjective(Q=jnp.diag(jnp.array([1.0, 10.0, 0.1])), R=jnp.diag(jnp.array([0.1, 0.1])), trajectory=ref)
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    return Problem(model=model, obj=obj, constraints=cl, N=N, dt=dt, integrator=RK4()), ref


def test_problem_structure_driver_split() -> None:
    """Verify Problem holds static structure and the driver holds the dynamic per-step data."""
    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    N = 10
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    cl = ConstraintList(n=2, m=1, N=N)
    cl.add_constraint(GoalConstraint(n=2, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    # Problem fields
    assert hasattr(prob, "model")
    assert hasattr(prob, "obj")
    assert hasattr(prob, "constraints")
    assert hasattr(prob, "N")
    assert hasattr(prob, "dt")
    assert not hasattr(prob, "integrator")
    assert not hasattr(prob, "x0")
    assert not hasattr(prob, "t0")
    assert not hasattr(prob, "solve")
    assert not hasattr(prob, "cost")

    # Boundary conditions and warm start, split across the driver
    x0 = jnp.array([0.1, 0.2])
    mpc = MPC(prob, Ipopt(), x0=x0, t0=0.0, xf=xf)
    assert isinstance(mpc.bc, BoundaryConditions)
    assert hasattr(mpc.bc, "x0")
    assert hasattr(mpc.bc, "t0")
    assert mpc.bc.xf is not None
    assert hasattr(mpc.warm_start, "lam")
    assert hasattr(mpc.warm_start, "mu")
    assert hasattr(mpc.warm_start, "Z")

    # Boundary conditions are all traced leaves, with no static field to retrace on
    leaves, _ = jax.tree.flatten(mpc.bc)
    assert len(leaves) == 4
    assert mpc.states.shape == (N, 2)
    assert mpc.controls.shape == (N - 1, 1)


def test_driver_per_step_operations_update_boundary_and_warm_start() -> None:
    """Verify per-step measurement update, goal update, and trajectory shift move the driver's state."""
    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    N = 5
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    prob = Problem(model=model, obj=obj, N=N, dt=0.1, integrator=RK4())

    x0 = jnp.array([0.0, 0.0])
    mpc = MPC(prob, Ipopt(), x0=x0, t0=0.0, xf=xf)

    # 1. Update measurement
    new_x = jnp.array([0.5, -0.1])
    new_t = 0.1
    mpc.measure(new_x, new_t)
    np.testing.assert_allclose(mpc.x0, new_x)
    np.testing.assert_allclose(mpc.t0, new_t)
    np.testing.assert_allclose(mpc.states[0], new_x)

    # 2. Update goal
    new_xf = jnp.array([0.0, 0.0])
    mpc.set_goal(new_xf)
    assert mpc.xf is not None
    np.testing.assert_allclose(mpc.xf, new_xf)

    # 3. Shift trajectory forward
    X_init = jnp.arange(N * 2, dtype=jnp.float64).reshape((N, 2))
    U_init = jnp.arange((N - 1) * 1, dtype=jnp.float64).reshape((N - 1, 1))
    mpc._ws = mpc.warm_start.with_primal(prob, X=X_init, U=U_init)  # noqa: SLF001 -- seeding a known primal

    t_before = mpc.t0
    mpc.shift(dt=0.1)
    X_shifted, U_shifted = mpc.states, mpc.controls

    # Shift drops index 0, shifts remaining forward, and duplicates the final element
    np.testing.assert_allclose(X_shifted[:-1], X_init[1:])
    np.testing.assert_allclose(X_shifted[-1], X_init[-1])
    np.testing.assert_allclose(U_shifted[:-1], U_init[1:])
    np.testing.assert_allclose(U_shifted[-1], U_init[-1])
    np.testing.assert_allclose(mpc.t0, t_before + 0.1)
    np.testing.assert_allclose(mpc.x0, X_init[1])


def test_pushed_reference_window_shifts_and_appends() -> None:
    """Verify the reference window advances one knot per shift, appending a pushed point or holding the last."""
    model = Pendulum()
    N = 5
    obj = LQRObjective(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), N=N)
    prob = Problem(model=model, obj=obj, N=N, dt=0.1, integrator=RK4())

    goal = jnp.array([np.pi, 0.0])
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(2), xf=goal)
    X_ref_0 = mpc.bc.X_ref
    assert X_ref_0 is not None

    # Nothing pushed: a constant window is shift-invariant, which is why a fixed goal needs no
    # special case at all.
    mpc.shift()
    np.testing.assert_allclose(_window(mpc)[0], X_ref_0)

    # Pushed: the point enters at the far end, one knot per shift, and the window slides under it.
    entering = jnp.array([1.0, 2.0])
    mpc.push_reference(entering, jnp.array([0.5]))
    mpc.shift()
    X_ref, U_ref = _window(mpc)
    np.testing.assert_allclose(X_ref[:-1], X_ref_0[1:])
    np.testing.assert_allclose(X_ref[-1], entering)
    np.testing.assert_allclose(U_ref[-1], jnp.array([0.5]))

    # The push is consumed, so the next shift holds that last point rather than repeating it.
    mpc.shift()
    X_ref, _ = _window(mpc)
    np.testing.assert_allclose(X_ref[-1], entering)
    np.testing.assert_allclose(X_ref[-2], entering)


def test_set_reference_replaces_the_window_wholesale() -> None:
    """Verify set_reference swaps the whole tracked window in one call."""
    prob, ref = _tracking_problem()
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(3), reference=ref)

    shifted = Trajectory(X=ref.X + 1.0, U=ref.U, t=ref.t, dt=ref.dt)
    mpc.set_reference(shifted)
    X_ref, U_ref = _window(mpc)
    np.testing.assert_allclose(X_ref, ref.X + 1.0)
    np.testing.assert_allclose(U_ref, ref.U)


def test_goal_state_single_source_of_truth() -> None:
    """Verify the goal state lives in the boundary conditions alone, read by objective and goal constraint."""
    model = Pendulum()
    n, m, N = 2, 1, 4
    Q = jnp.eye(n)
    R = jnp.eye(m)
    xf_initial = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=xf_initial), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    x0 = jnp.array([0.1, 0.0])
    mpc = MPC(prob, Ipopt(), x0=x0, t0=0.0, xf=xf_initial)
    X_state = mpc.states

    # Initial cost and constraint evaluation
    c1 = mpc.cost()
    con1, _ = constraints_and_jac(prob, mpc.Z, mpc.x0, mpc.t0, prob.dt, xf=mpc.xf)

    # Update goal on the driver alone
    xf_new = jnp.array([0.0, 0.0])
    mpc.set_goal(xf_new)

    c2 = mpc.cost()
    con2, _ = constraints_and_jac(prob, mpc.Z, mpc.x0, mpc.t0, prob.dt, xf=mpc.xf)

    # Goal constraint residual at terminal index must reflect xf_new directly
    np.testing.assert_allclose(con1[-n:], X_state[-1] - xf_initial)
    np.testing.assert_allclose(con2[-n:], X_state[-1] - xf_new)
    assert not np.allclose(c1, c2)


@pytest.mark.slow
def test_zero_recompile_across_100_mpc_iterations() -> None:
    """Assert the compilation counter remains at zero across 100 consecutive MPC steps with changing data."""
    model = Cartpole()
    n, m, N = model.n, model.m, 10
    Q = jnp.eye(n)
    R = jnp.eye(m)
    xf_init = jnp.array([0.0, np.pi, 0.0, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    prob = Problem(model=model, obj=obj, N=N, dt=0.05, integrator=RK4())

    x0 = jnp.zeros(n)
    mpc = MPC(prob, Ipopt(), x0=x0, t0=0.0, xf=xf_init)

    compile_count_cost = 0
    compile_count_jac = 0
    compile_count_hess = 0

    # The cost path takes the boundary conditions and the constraint path the goal point, exactly
    # as the real call sites do, so this measures those rather than a narrowed version of them.
    def cost_target(p: Problem, z: jax.Array, t0: jax.Array, dt: jax.Array, bc: BoundaryConditions | None) -> jax.Array:
        nonlocal compile_count_cost
        compile_count_cost += 1
        return eval_f(p, z, t0, dt, bc)

    def jac_target(
        p: Problem, z: jax.Array, x_init: jax.Array, t0: jax.Array, dt: jax.Array, xf: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        nonlocal compile_count_jac
        compile_count_jac += 1
        return constraints_and_jac(p, z, x_init, t0, dt, xf=xf)

    def hess_target(
        p: Problem,
        z: jax.Array,
        t0: jax.Array,
        dt: jax.Array,
        obj_factor: float,
        lam: jax.Array,
        bc: BoundaryConditions | None,
    ) -> jax.Array:
        nonlocal compile_count_hess
        compile_count_hess += 1
        return eval_h(p, z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam, bc=bc)

    jit_cost = eqx.filter_jit(cost_target)
    jit_jac = eqx.filter_jit(jac_target)
    jit_hess = eqx.filter_jit(hess_target)

    # Initial warmup compile (iteration 0)
    _ = jit_cost(prob, mpc.Z, mpc.t0, prob.dt, mpc.bc)
    _ = jit_jac(prob, mpc.Z, mpc.x0, mpc.t0, prob.dt, mpc.xf)
    _ = jit_hess(prob, mpc.Z, mpc.t0, prob.dt, 1.0, mpc.warm_start.lam, mpc.bc)

    assert compile_count_cost == 1
    assert compile_count_jac == 1
    assert compile_count_hess == 1

    # Run 100 consecutive MPC iterations with varying x0, t0, xf
    for i in range(100):
        t_curr = (i + 1) * 0.05
        x_meas = jnp.sin(jnp.arange(n, dtype=jnp.float64) + i * 0.1)
        xf_step = jnp.cos(jnp.arange(n, dtype=jnp.float64) + i * 0.05)

        mpc.measure(x_meas, t_curr)
        mpc.set_goal(xf_step)

        _ = jit_cost(prob, mpc.Z, mpc.t0, prob.dt, mpc.bc)
        _ = jit_jac(prob, mpc.Z, mpc.x0, mpc.t0, prob.dt, mpc.xf)
        _ = jit_hess(prob, mpc.Z, mpc.t0, prob.dt, 1.0, mpc.warm_start.lam, mpc.bc)

    # Assert exactly zero new compilations occurred across all 100 steps
    assert compile_count_cost == 1
    assert compile_count_jac == 1
    assert compile_count_hess == 1


def test_model_parameters_traced_zero_recompile() -> None:
    """Verify changing model parameters does not trigger recompilation."""
    model = Cartpole(mc=1.0, mp=0.2)
    n, m, N = model.n, model.m, 5
    Q = jnp.eye(n)
    R = jnp.eye(m)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    prob1 = Problem(model=model, obj=obj, N=N, integrator=RK4())

    mpc = MPC(prob1, Ipopt(), x0=jnp.zeros(n), t0=0.0, xf=xf)

    compile_count = 0

    def jac_fn(p: Problem, z: jax.Array, x0: jax.Array, t0: jax.Array, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        nonlocal compile_count
        compile_count += 1
        return constraints_and_jac(p, z, x0, t0, dt)

    jit_jac = eqx.filter_jit(jac_fn)

    # First call compiles
    _c1, j1 = jit_jac(prob1, mpc.Z, mpc.x0, mpc.t0, prob1.dt)
    assert compile_count == 1

    # Change mass parameter in model
    prob2 = eqx.tree_at(lambda p: p.model.continuous_dynamics.mp, prob1, jnp.asarray(0.35, dtype=jnp.float64))
    _c2, j2 = jit_jac(prob2, mpc.Z, mpc.x0, mpc.t0, prob2.dt)

    # Must NOT recompile (compile_count stays 1)
    assert compile_count == 1
    assert not np.allclose(j1, j2)


@pytest.mark.slow
def test_cartpole_warm_start_reduces_iterations() -> None:
    """Assert warm-starting from shifted previous solution reduces solver iterations vs cold start."""
    pytest.importorskip("cyipopt")

    N = 25
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])

    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, dt=dt, integrator=RK4())

    options = {"max_iter": 200, "tol": 1e-4, "print_level": 0}
    mpc = MPC(prob, Ipopt(options=options), x0=x0, t0=0.0, xf=xf)

    # Initial solve
    mpc.solve()
    u0 = mpc.controls[0]

    # Advance 1 step
    dmodel = RK4(model)
    x1 = dmodel.discrete_dynamics(x0, u0, 0.0, dt)

    # 1. Warm start: shift previous optimal solution
    mpc.measure(x1, dt)
    mpc.shift(dt)
    res_warm = mpc.solve()

    # 2. Cold start: reset trajectory to constant x1 and zero controls
    res_cold = MPC(prob, Ipopt(options=options), x0=x1, t0=dt, xf=xf).solve()

    # Warm start measurably reduces solver iterations vs cold start
    assert res_warm.iterations < res_cold.iterations
    np.testing.assert_allclose(res_warm.trajectory.U[0], res_cold.trajectory.U[0], atol=1e-2)


@pytest.mark.slow
def test_closed_loop_cartpole_mpc() -> None:
    """Verify closed-loop cartpole MPC stabilizes to upright goal from a perturbed initial state."""
    pytest.importorskip("cyipopt")

    N = 20
    dt = 0.05
    model = Cartpole()
    n, m = model.n, model.m

    # Perturbed initial state near upright: [x=0.0, theta=pi - 0.25, x_dot=0.1, theta_dot=-0.2]
    x_curr = jnp.array([0.0, np.pi - 0.25, 0.1, -0.2])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])

    Q = jnp.diag(jnp.array([5.0, 20.0, 1.0, 2.0]))
    R = jnp.diag(jnp.array([0.05]))
    Qf = jnp.diag(jnp.array([50.0, 200.0, 10.0, 20.0]))
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, dt=dt, integrator=RK4())

    mpc = MPC(prob, Ipopt(options={"max_iter": 50, "tol": 1e-4, "print_level": 0}), x0=x_curr, t0=0.0, xf=xf)
    dmodel = RK4(model)

    sim_steps = 20
    t_curr = 0.0

    for _ in range(sim_steps):
        mpc.measure(x_curr, t_curr)
        mpc.solve()
        u_cmd = mpc.controls[0]

        # Simulate system forward with applied control
        x_curr = dmodel.discrete_dynamics(x_curr, u_cmd, t_curr, dt)
        t_curr += dt
        mpc.shift(dt)

    # After 20 steps (1.0 second), cartpole should be stabilized close to upright [0, pi, 0, 0]
    np.testing.assert_allclose(x_curr[1], np.pi, atol=0.1)
    np.testing.assert_allclose(x_curr[0], 0.0, atol=0.5)
    np.testing.assert_allclose(x_curr[2:], 0.0, atol=0.5)


def test_rollout_problem_state() -> None:
    """Verify model.rollout(trajectory) simulates dynamics and returns a Trajectory."""
    model = Cartpole()
    n, m, N = model.n, model.m, 10
    Q = jnp.eye(n)
    R = jnp.eye(m)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)
    prob = Problem(model=model, obj=obj, N=N, dt=0.05, integrator=RK4())

    x0 = jnp.array([0.1, 0.2, 0.0, 0.0])
    mpc = MPC(prob, Ipopt(), x0=x0, t0=0.0, xf=xf)
    U_const = jnp.full((N - 1, m), 0.5)
    mpc._ws = mpc.warm_start.with_primal(prob, U=U_const)  # noqa: SLF001 -- seeding a known control guess

    traj = prob.model.rollout(mpc.trajectory())
    assert isinstance(traj, Trajectory)
    assert traj.N == N
    assert traj.n == n
    assert traj.m == m
    np.testing.assert_allclose(traj.X[0], x0)
    np.testing.assert_allclose(traj.U, U_const)

    # Verify dynamics defects along trajectory are zero
    dmodel = RK4(model)
    for k in range(N - 1):
        x_next = dmodel.discrete_dynamics(traj.X[k], traj.U[k], traj.t[k], traj.dt[k])
        np.testing.assert_allclose(traj.X[k + 1], x_next)


def test_runtime_goal_retargets_a_goal_regulating_objective() -> None:
    """Assert a run-time xf moves a shape-only LQRObjective exactly as baking that goal in would."""
    model = Cartpole()
    N = 8
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    xf_build = jnp.array([0.0, np.pi, 0.0, 0.0])
    xf_new = jnp.array([0.3, 2.0, -0.1, 0.4])

    prob = Problem(model=model, obj=LQRObjective(Q=Q, R=R, Qf=Qf, N=N), N=N, dt=0.05, integrator=RK4())
    obj_rebuilt = LQRObjective(Q=Q, R=R, Qf=Qf, N=N).with_reference(
        jnp.broadcast_to(xf_new, (N, model.n)),
        jnp.zeros((N - 1, model.m)),
    )
    prob_rebuilt = Problem(model=model, obj=obj_rebuilt, N=N, dt=0.05, integrator=RK4())

    mpc = MPC(prob, Ipopt(), x0=jnp.array([0.1, 0.2, 0.0, 0.0]), xf=xf_build)
    Z = mpc.Z + 0.05 * jnp.arange(len(mpc.Z), dtype=mpc.Z.dtype)

    # The retarget rebuilds q, r and c from the new target, so a retargeted objective and one
    # rebuilt at that target agree by a constant offset (here zero) and their gradients outright.
    mpc.set_goal(xf_new)
    bc_new = mpc.bc
    np.testing.assert_allclose(
        eval_grad_f(prob, Z, mpc.t0, prob.dt, bc_new),
        eval_grad_f(prob_rebuilt, Z, mpc.t0, prob.dt, None),
        rtol=1e-12,
        atol=1e-12,
    )
    j_retargeted = eval_f(prob, Z, mpc.t0, prob.dt, bc_new)
    j_rebuilt = eval_f(prob_rebuilt, Z, mpc.t0, prob.dt, None)
    offset = j_retargeted - j_rebuilt
    Z2 = Z * 0.5
    np.testing.assert_allclose(
        eval_f(prob, Z2, mpc.t0, prob.dt, bc_new) - eval_f(prob_rebuilt, Z2, mpc.t0, prob.dt, None),
        offset,
        rtol=1e-12,
        atol=1e-12,
    )


def test_runtime_reference_window_tracks_while_the_goal_constrains() -> None:
    """Assert a run-time reference window aims the cost while xf still drives the goal constraint."""
    prob, ref = _tracking_problem()
    mpc = MPC(prob, Ipopt(), x0=jnp.zeros(3), reference=ref, initial_trajectory=ref)

    # At the reference the tracking cost is zero, and retargeting to the run-time window, which
    # here is that same reference, reproduces it exactly.
    np.testing.assert_allclose(prob.obj.cost(ref), 0.0, atol=1e-12)
    np.testing.assert_allclose(mpc.bc.retarget(prob.obj).cost(ref), 0.0, atol=1e-12)

    # The goal constraint follows the run-time goal, which a pushed point moves.
    xf_new = jnp.array([1.0, 0.25, 0.0])
    mpc.push_reference(xf_new)
    mpc.shift()
    assert mpc.xf is not None
    np.testing.assert_allclose(mpc.xf, xf_new, atol=1e-12)
    con, _ = constraints_and_jac(prob, mpc.Z, mpc.x0, mpc.t0, prob.dt, xf=mpc.xf)
    np.testing.assert_allclose(con[-3:], mpc.states[-1] - xf_new, atol=1e-12)


def test_runtime_goal_rejected_when_nothing_reads_it() -> None:
    """Assert a goal that neither the objective nor a constraint consumes is refused at construction."""
    model = Quadrotor()
    N = 8
    q_ref = jnp.array([1.0, 0.0, 0.0, 0.0])
    Q = jnp.ones(model.n).at[3:7].set(0.0)
    stage = QuatGeodesicCost(Q=Q, R=jnp.full(model.m, 0.01), q_ref=q_ref, w=10.0, m=model.m)
    term = QuatGeodesicCost(Q=Q, q_ref=q_ref, w=100.0, terminal=True)
    prob = Problem(model=model, obj=Objective(stage_cost=stage, terminal_cost=term, N=N), N=N, dt=0.1, integrator=RK4())
    x0 = jnp.zeros(model.n).at[3].set(1.0)

    with pytest.raises(ValueError, match="nothing in the problem reads it"):
        MPC(prob, Ipopt(), x0=x0, xf=x0)

    mpc = MPC(prob, Ipopt(), x0=x0)
    assert mpc.xf is None
    with pytest.raises(ValueError, match="built without a goal"):
        mpc.set_goal(x0)


def test_constant_goal_rejected_against_an_objective_that_already_tracks() -> None:
    """Assert xf is refused on a TrackingObjective, whose per-knot reference it would flatten.

    A constant goal window overwrites q, r and c at every knot, so it silently replaces the
    tracked trajectory rather than adding to it. The window form is the way to move that target.
    """
    prob, ref = _tracking_problem()

    with pytest.raises(ValueError, match="already tracks a build-time reference"):
        MPC(prob, Ipopt(), x0=jnp.zeros(3), xf=jnp.array([2.0, 0.0, 0.0]))

    MPC(prob, Ipopt(), x0=jnp.zeros(3), reference=ref)
