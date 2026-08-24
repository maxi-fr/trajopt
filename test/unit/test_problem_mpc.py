import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective, TrackingObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.models.dubins import DubinsCar
from trajopt.models.pendulum import Pendulum
from trajopt.problem import (
    MPCState,
    Problem,
    controls,
    cost,
    initial_controls,
    initial_states,
    states,
)
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.transcription import (
    constraints_and_jac,
    eval_f,
    eval_grad_f,
    eval_h,
)


def test_problem_structure_mpcstate_split() -> None:
    """Verify Problem holds static structure and MPCState holds dynamic per-step data."""
    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    N = 10
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    cl = ConstraintList(n=2, m=1, N=N)
    cl.add_constraint(GoalConstraint(n=2, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    # Problem fields
    assert hasattr(prob, "model")
    assert hasattr(prob, "obj")
    assert hasattr(prob, "constraints")
    assert hasattr(prob, "N")
    assert not hasattr(prob, "integrator")
    assert not hasattr(prob, "x0")
    assert not hasattr(prob, "t0")

    # MPCState fields and pytree leaves
    x0 = jnp.array([0.1, 0.2])
    t0 = 0.0
    state = MPCState.initial(prob, x0=x0, t0=t0, xf=xf)
    assert isinstance(state, MPCState)
    assert hasattr(state, "x0")
    assert hasattr(state, "t0")
    assert hasattr(state, "xf")
    assert hasattr(state, "lam")
    assert hasattr(state, "mu")
    assert hasattr(state, "Z")

    # Static metadata vs leaves
    leaves, _ = jax.tree.flatten(state)
    assert len(leaves) > 0
    assert state.n == 2
    assert state.m == 1
    assert state.N == N


def test_mpcstate_per_step_operations_return_new_values() -> None:
    """Verify per-step measurement update, goal update, and trajectory shift return new instances."""
    model = Pendulum()
    Q = jnp.eye(2)
    R = jnp.eye(1)
    xf = jnp.array([np.pi, 0.0])
    N = 5
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, N=N, integrator=RK4())

    x0 = jnp.array([0.0, 0.0])
    state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf, dt=0.1)

    # 1. Update measurement
    new_x = jnp.array([0.5, -0.1])
    new_t = 0.1
    s_meas = state.with_measurement(new_x, new_t)
    assert s_meas is not state
    np.testing.assert_allclose(s_meas.x0, new_x)
    np.testing.assert_allclose(s_meas.t0, new_t)
    np.testing.assert_allclose(states(s_meas)[0], new_x)

    # 2. Update goal
    new_xf = jnp.array([0.0, 0.0])
    s_goal = state.with_goal(new_xf)
    assert s_goal is not state
    assert s_goal.xf is not None
    np.testing.assert_allclose(s_goal.xf, new_xf)

    # 3. Shift trajectory forward
    X_init = jnp.arange(N * 2, dtype=jnp.float64).reshape((N, 2))
    U_init = jnp.arange((N - 1) * 1, dtype=jnp.float64).reshape((N - 1, 1))
    s_custom = state.initial_states(X_init).initial_controls(U_init)

    s_shifted = s_custom.shift(dt=0.1)
    assert s_shifted is not s_custom
    X_shifted = states(s_shifted)
    U_shifted = controls(s_shifted)

    # Shift drops index 0, shifts remaining forward, and duplicates the final element
    np.testing.assert_allclose(X_shifted[:-1], X_init[1:])
    np.testing.assert_allclose(X_shifted[-1], X_init[-1])
    np.testing.assert_allclose(U_shifted[:-1], U_init[1:])
    np.testing.assert_allclose(U_shifted[-1], U_init[-1])
    np.testing.assert_allclose(s_shifted.t0, state.t0 + 0.1)


def test_goal_state_single_source_of_truth() -> None:
    """Verify goal state lives in MPCState alone and is read by both objective and goal constraint."""
    model = Pendulum()
    n, m, N = 2, 1, 4
    Q = jnp.eye(n)
    R = jnp.eye(m)
    xf_initial = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf_initial, N=N)
    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(GoalConstraint(n=n, xf=xf_initial), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    x0 = jnp.array([0.1, 0.0])
    state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf_initial)

    # Initial cost and constraint evaluation
    c1 = cost(prob, state)
    con1, _ = constraints_and_jac(prob, state.Z, state.x0, state.t0, state.dt, xf=state.xf)

    # Update goal on state alone
    xf_new = jnp.array([0.0, 0.0])
    state_new = state.with_goal(xf_new)
    assert state_new.xf is not state.xf

    c2 = cost(prob, state_new)
    con2, _ = constraints_and_jac(prob, state_new.Z, state_new.x0, state_new.t0, state_new.dt, xf=state_new.xf)

    # Goal constraint residual at terminal index must reflect xf_new directly
    terminal_con1 = con1[-n:]
    terminal_con2 = con2[-n:]
    X_state = states(state)
    np.testing.assert_allclose(terminal_con1, X_state[-1] - xf_initial)
    np.testing.assert_allclose(terminal_con2, X_state[-1] - xf_new)
    assert not np.allclose(c1, c2)


def test_zero_recompile_across_100_mpc_iterations() -> None:
    """Assert the compilation counter remains at zero across 100 consecutive MPC steps with changing data."""
    model = Cartpole()
    n, m, N = model.n, model.m, 10
    Q = jnp.eye(n)
    R = jnp.eye(m)
    xf_init = jnp.array([0.0, np.pi, 0.0, 0.0])
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf_init, N=N)
    prob = Problem(model=model, obj=obj, N=N, integrator=RK4())

    x0 = jnp.zeros(n)
    state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf_init, dt=0.05)

    compile_count_cost = 0
    compile_count_jac = 0
    compile_count_hess = 0

    # xf is Array | None on MPCState, and forwarding it as such is what lets these mirror the
    # real call sites rather than a narrowed version of them.
    def cost_target(p: Problem, z: jax.Array, t0: jax.Array, dt: jax.Array, xf: jax.Array | None) -> jax.Array:
        nonlocal compile_count_cost
        compile_count_cost += 1
        return eval_f(p, z, t0, dt, xf)

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
        xf: jax.Array | None,
    ) -> jax.Array:
        nonlocal compile_count_hess
        compile_count_hess += 1
        return eval_h(p, z, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam, xf=xf)

    jit_cost = eqx.filter_jit(cost_target)
    jit_jac = eqx.filter_jit(jac_target)
    jit_hess = eqx.filter_jit(hess_target)

    # Initial warmup compile (iteration 0)
    _ = jit_cost(prob, state.Z, state.t0, state.dt, state.xf)
    _ = jit_jac(prob, state.Z, state.x0, state.t0, state.dt, state.xf)
    _ = jit_hess(prob, state.Z, state.t0, state.dt, 1.0, state.lam, state.xf)

    assert compile_count_cost == 1
    assert compile_count_jac == 1
    assert compile_count_hess == 1

    # Run 100 consecutive MPC iterations with varying x0, t0, xf
    for i in range(100):
        t_curr = (i + 1) * 0.05
        x_meas = jnp.sin(jnp.arange(n, dtype=jnp.float64) + i * 0.1)
        xf_step = jnp.cos(jnp.arange(n, dtype=jnp.float64) + i * 0.05)

        state = state.with_measurement(x_meas, t_curr).with_goal(xf_step)

        _ = jit_cost(prob, state.Z, state.t0, state.dt, state.xf)
        _ = jit_jac(prob, state.Z, state.x0, state.t0, state.dt, state.xf)
        _ = jit_hess(prob, state.Z, state.t0, state.dt, 1.0, state.lam, state.xf)

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
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob1 = Problem(model=model, obj=obj, N=N, integrator=RK4())

    state = MPCState.initial(prob1, x0=jnp.zeros(n), t0=0.0, xf=xf)

    compile_count = 0

    def jac_fn(p: Problem, z: jax.Array, x0: jax.Array, t0: jax.Array, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        nonlocal compile_count
        compile_count += 1
        return constraints_and_jac(p, z, x0, t0, dt)

    jit_jac = eqx.filter_jit(jac_fn)

    # First call compiles
    _c1, j1 = jit_jac(prob1, state.Z, state.x0, state.t0, state.dt)
    assert compile_count == 1

    # Change mass parameter in model
    prob2 = eqx.tree_at(lambda p: p.model.continuous_dynamics.mp, prob1, jnp.asarray(0.35, dtype=jnp.float64))
    _c2, j2 = jit_jac(prob2, state.Z, state.x0, state.t0, state.dt)

    # Must NOT recompile (compile_count stays 1)
    assert compile_count == 1
    assert not np.allclose(j1, j2)


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
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf, dt=dt)

    # Initial solve
    state_opt = prob.solve(state, solver=Ipopt(options={"max_iter": 200, "tol": 1e-4, "print_level": 0}))
    u0 = controls(state_opt)[0]

    # Advance 1 step
    dmodel = RK4(model)
    x1 = dmodel.discrete_dynamics(x0, u0, 0.0, dt)

    # 1. Warm start: shift previous optimal solution
    state_warm = state_opt.with_measurement(x1, dt).shift(dt)
    res_warm = Ipopt(options={"max_iter": 200, "tol": 1e-4, "print_level": 0}).solve(prob, state_warm)

    # 2. Cold start: reset trajectory to constant x1 and zero controls
    state_cold = MPCState.initial(prob, x0=x1, t0=dt, xf=xf, dt=dt)
    res_cold = Ipopt(options={"max_iter": 200, "tol": 1e-4, "print_level": 0}).solve(prob, state_cold)

    # Warm start measurably reduces solver iterations vs cold start
    assert res_warm.iterations < res_cold.iterations
    np.testing.assert_allclose(res_warm.trajectory.U[0], res_cold.trajectory.U[0], atol=1e-2)


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
    obj = LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-20.0], u_max=[20.0]), range(N - 1))
    prob = Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4())

    state = MPCState.initial(prob, x0=x_curr, t0=0.0, xf=xf, dt=dt)
    dmodel = RK4(model)

    sim_steps = 20
    t_curr = 0.0

    for _ in range(sim_steps):
        state = state.with_measurement(x_curr, t_curr)
        state = prob.solve(state, solver=Ipopt(options={"max_iter": 50, "tol": 1e-4, "print_level": 0}))
        u_cmd = controls(state)[0]

        # Simulate system forward with applied control
        x_curr = dmodel.discrete_dynamics(x_curr, u_cmd, t_curr, dt)
        t_curr += dt
        state = state.shift(dt)

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
    obj = LQRObjective(Q=Q, R=R, Qf=Q, xf=xf, N=N)
    prob = Problem(model=model, obj=obj, N=N, integrator=RK4())

    x0 = jnp.array([0.1, 0.2, 0.0, 0.0])
    state = MPCState.initial(prob, x0=x0, t0=0.0, xf=xf, dt=0.05)
    U_const = jnp.full((N - 1, m), 0.5)
    state = state.initial_controls(U_const)

    traj = prob.model.rollout(state.to_trajectory())
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
    return Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4()), ref


def test_runtime_goal_retargets_a_goal_regulating_objective() -> None:
    """Assert a run-time xf moves an LQRObjective exactly as rebuilding it at the new goal would."""
    model = Cartpole()
    N = 8
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = jnp.diag(jnp.array([100.0, 1000.0, 10.0, 10.0]))
    xf_build = jnp.array([0.0, np.pi, 0.0, 0.0])
    xf_new = jnp.array([0.3, 2.0, -0.1, 0.4])

    prob = Problem(model=model, obj=LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf_build, N=N), N=N, integrator=RK4())
    prob_rebuilt = Problem(model=model, obj=LQRObjective(Q=Q, R=R, Qf=Qf, xf=xf_new, N=N), N=N, integrator=RK4())

    state = MPCState.initial(prob, x0=jnp.array([0.1, 0.2, 0.0, 0.0]), dt=0.05, xf=xf_build)
    Z = state.Z + 0.05 * jnp.arange(len(state.Z), dtype=state.Z.dtype)

    # The retarget rewrites q = -Q xf and leaves c at its build value, so the two costs agree up
    # to that constant; the gradient, which is what the solver sees, agrees outright.
    np.testing.assert_allclose(
        eval_grad_f(prob, Z, state.t0, state.dt, xf_new),
        eval_grad_f(prob_rebuilt, Z, state.t0, state.dt, None),
        rtol=1e-12,
        atol=1e-12,
    )
    j_retargeted = eval_f(prob, Z, state.t0, state.dt, xf_new)
    j_rebuilt = eval_f(prob_rebuilt, Z, state.t0, state.dt, None)
    offset = j_retargeted - j_rebuilt
    Z2 = Z * 0.5
    np.testing.assert_allclose(
        eval_f(prob, Z2, state.t0, state.dt, xf_new) - eval_f(prob_rebuilt, Z2, state.t0, state.dt, None),
        offset,
        rtol=1e-12,
        atol=1e-12,
    )


def test_runtime_goal_leaves_a_tracking_objective_alone() -> None:
    """Assert xf reaches the goal constraint without displacing a tracking objective's reference."""
    prob, ref = _tracking_problem()
    xf = jnp.array([2.0, 0.0, 0.0])
    state = MPCState.initial(prob, x0=jnp.zeros(3), dt=0.1, xf=xf, initial_trajectory=ref)

    # At the reference the tracking cost is zero; regulating to xf instead would not be.
    np.testing.assert_allclose(prob.obj.cost(ref), 0.0, atol=1e-12)
    np.testing.assert_allclose(eval_f(prob, state.Z, state.t0, state.dt, state.xf), 0.0, atol=1e-12)

    # The goal constraint still follows the run-time goal.
    xf_new = jnp.array([1.0, 0.25, 0.0])
    con, _ = constraints_and_jac(prob, state.Z, state.x0, state.t0, state.dt, xf=state.with_goal(xf_new).xf)
    np.testing.assert_allclose(con[-3:], states(state)[-1] - xf_new, atol=1e-12)


def test_runtime_goal_rejected_when_nothing_reads_it() -> None:
    """Assert a goal that neither the objective nor a constraint consumes is refused at construction."""
    prob, ref = _tracking_problem()
    unconstrained = Problem(model=prob.model, obj=prob.obj, N=prob.N, integrator=RK4())

    with pytest.raises(ValueError, match="nothing in the problem reads it"):
        MPCState.initial(unconstrained, x0=jnp.zeros(3), dt=0.1, xf=jnp.array([2.0, 0.0, 0.0]))

    state = MPCState.initial(unconstrained, x0=jnp.zeros(3), dt=0.1, initial_trajectory=ref)
    assert state.xf is None
    with pytest.raises(ValueError, match="built without a goal"):
        state.with_goal(jnp.array([2.0, 0.0, 0.0]))


def test_tracking_objective_cannot_be_retargeted_by_a_goal() -> None:
    """Assert retargeting a tracking objective is refused rather than silently discarding its reference."""
    prob, _ = _tracking_problem()
    assert prob.obj.regulates_to_goal is False
    with pytest.raises(TypeError, match="does not regulate to a goal state"):
        prob.obj.with_goal(jnp.array([2.0, 0.0, 0.0]))
