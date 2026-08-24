import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.dynamics import (
    RK4,
    DiscretizedDynamics,
    rollout_states,
)
from trajopt.models import Cartpole
from trajopt.trajectory import Trajectory


def test_rollout_propagates_states_correctly() -> None:
    model = Cartpole()
    disc_model = RK4(model)

    N = 21
    dt_val = 0.05
    x0 = jnp.array([0.1, -0.2, 0.05, -0.1])
    U = jnp.ones((N - 1, 1)) * 0.5
    t = jnp.linspace(0.0, (N - 1) * dt_val, N)
    dt = jnp.full((N - 1,), dt_val)

    # 1. Rollout via scan
    traj_init = Trajectory(X=jnp.zeros((N, 4)), U=U, t=t, dt=dt)
    traj_sim = disc_model.rollout(traj_init, x0=x0)
    assert isinstance(traj_sim, Trajectory)
    assert traj_sim.N == N
    assert traj_sim.X.shape == (N, 4)
    np.testing.assert_allclose(traj_sim.X[0], x0)

    # 2. Compare step-by-step with direct simulation
    x_curr = x0
    X_expected = [x_curr]
    for k in range(N - 1):
        x_curr = disc_model.discrete_dynamics(x_curr, U[k], t[k], dt[k])
        X_expected.append(x_curr)
    X_expected_arr = jnp.stack(X_expected, axis=0)

    np.testing.assert_allclose(traj_sim.X, X_expected_arr, rtol=1e-14, atol=1e-14)


def test_rollout_with_trajectory_instance_and_x0_override() -> None:
    model = Cartpole()
    disc_model = RK4(model)

    N = 15
    dt_val = 0.02
    X_zeros = jnp.zeros((N, 4))
    U = jnp.linspace(-1.0, 1.0, N - 1).reshape(N - 1, 1)
    t = jnp.linspace(0.0, (N - 1) * dt_val, N)
    dt = jnp.full((N - 1,), dt_val)

    traj_init = Trajectory(X=X_zeros, U=U, t=t, dt=dt)

    # Rollout using initial trajectory state (which is zero)
    traj_res = disc_model.rollout(traj_init)
    np.testing.assert_allclose(traj_res.X[0], jnp.zeros(4))
    assert traj_res.X.shape == (N, 4)

    # Rollout with initial condition override
    x0_new = jnp.array([1.0, 0.5, -0.5, 0.2])
    traj_res_override = disc_model.rollout(traj_init, x0=x0_new)
    np.testing.assert_allclose(traj_res_override.X[0], x0_new)

    # Verify rollout_states helper
    X_states = rollout_states(disc_model, x0_new, U, t, dt)
    np.testing.assert_allclose(X_states, traj_res_override.X, rtol=1e-14, atol=1e-14)


def test_rollout_with_continuous_model_auto_discretization() -> None:
    model = Cartpole()
    N = 10
    x0 = jnp.array([0.0, 0.1, 0.0, 0.0])
    U = jnp.zeros((N - 1, 1))

    # Passing continuous model should default to RK4 discretized dynamics
    X = rollout_states(model, x0, U, dt=0.05)
    assert X.shape == (N, 4)
    np.testing.assert_allclose(X[0], x0)
    np.testing.assert_allclose(X, rollout_states(RK4(model), x0, U, dt=0.05), rtol=1e-14, atol=1e-14)


def test_rollout_jit_compilation() -> None:
    model = Cartpole()
    disc_model = RK4(model)

    N = 51
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    U = jnp.ones((N - 1, 1)) * 0.2
    t = jnp.linspace(0.0, 1.0, N)
    dt = jnp.full((N - 1,), 1.0 / (N - 1))

    traj_in = Trajectory(X=jnp.zeros((N, 4)), U=U, t=t, dt=dt)

    @eqx.filter_jit
    def run_sim(m: DiscretizedDynamics, tr: Trajectory) -> Trajectory:
        return m.rollout(tr, x0=x0)

    # Compile and run
    res1 = run_sim(disc_model, traj_in)
    res2 = run_sim(disc_model, traj_in)

    np.testing.assert_allclose(res1.X, res2.X)


def test_horizon_length_does_not_multiply_interpreter_overhead() -> None:
    """Demonstrates that jax.lax.scan tracing/compilation time does not scale linearly with horizon length.

    With Python unrolling (for-loop), tracing time and HLO size scale as O(N).
    With jax.lax.scan, the body function is traced exactly once regardless of N,
    keeping tracing overhead constant O(1).
    """
    model = Cartpole()
    disc_model = RK4(model)
    x0 = jnp.zeros(4)

    # Define Python unrolled loop rollout for comparison
    def python_loop_rollout(m, x_init, U_arr, t_arr, dt_arr):
        n_pts = U_arr.shape[0] + 1
        X_list = [x_init]
        x_curr = x_init
        for k in range(n_pts - 1):
            x_curr = m.discrete_dynamics(x_curr, U_arr[k], t_arr[k], dt_arr[k])
            X_list.append(x_curr)
        return jnp.stack(X_list, axis=0)

    # Measure Jaxpr expression size / trace complexity for scan vs loop
    def make_data(N):
        U = jnp.ones((N - 1, 1)) * 0.1
        t = jnp.linspace(0.0, 1.0, N)
        dt = jnp.full((N - 1,), 1.0 / (N - 1))
        return U, t, dt

    # 1. Scan Jaxpr size remains constant across horizon lengths
    U_short, t_short, dt_short = make_data(10)
    U_long, t_long, dt_long = make_data(500)

    jaxpr_scan_short = jax.make_jaxpr(lambda u, t, dt: rollout_states(disc_model, x0, u, t, dt))(
        U_short, t_short, dt_short
    )
    jaxpr_scan_long = jax.make_jaxpr(lambda u, t, dt: rollout_states(disc_model, x0, u, t, dt))(U_long, t_long, dt_long)

    # The scan representation has exactly 1 scan equation regardless of horizon length
    num_eqs_scan_short = len(jaxpr_scan_short.jaxpr.eqns)
    num_eqs_scan_long = len(jaxpr_scan_long.jaxpr.eqns)
    assert num_eqs_scan_short == num_eqs_scan_long

    # 2. In contrast, Python unrolled loop equation count scales linearly with N
    jaxpr_loop_short = jax.make_jaxpr(lambda u, t, dt: python_loop_rollout(disc_model, x0, u, t, dt))(
        U_short, t_short, dt_short
    )
    jaxpr_loop_long = jax.make_jaxpr(lambda u, t, dt: python_loop_rollout(disc_model, x0, u, t, dt))(
        U_long, t_long, dt_long
    )

    num_eqs_loop_short = len(jaxpr_loop_short.jaxpr.eqns)
    num_eqs_loop_long = len(jaxpr_loop_long.jaxpr.eqns)

    # Python loop equation count grows proportional to N (~50x larger for N=500 vs N=10)
    assert num_eqs_loop_long > 30 * num_eqs_loop_short
    # Scan has a tiny constant equation count
    assert num_eqs_scan_long < num_eqs_loop_short
