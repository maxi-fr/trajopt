"""Unit tests for trajectory storage and KnotPoint views."""

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.trajectory import KnotPoint, Trajectory


def test_trajectory_creation_and_properties() -> None:
    N, n, m = 11, 4, 2
    X = jnp.zeros((N, n))
    U = jnp.ones((N - 1, m))
    t = jnp.linspace(0.0, 1.0, N)
    dt = jnp.full((N - 1,), 0.1)

    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    assert traj.N == N
    assert traj.n == n
    assert traj.m == m
    assert len(traj) == N

    np.testing.assert_allclose(traj.states(), X)
    np.testing.assert_allclose(traj.controls(), U)
    np.testing.assert_allclose(traj.times(), t)
    np.testing.assert_allclose(traj.dt, dt)


def test_trajectory_shape_validation() -> None:
    N, n, m = 11, 4, 2
    X = jnp.zeros((N, n))
    U = jnp.ones((N - 1, m))
    t = jnp.linspace(0.0, 1.0, N)
    dt = jnp.full((N - 1,), 0.1)

    # Mismatched control length
    with pytest.raises(ValueError, match="Controls U length"):
        Trajectory(X=X, U=jnp.ones((N, m)), t=t, dt=dt)

    # Mismatched times length
    with pytest.raises(ValueError, match="Times t length"):
        Trajectory(X=X, U=U, t=jnp.linspace(0.0, 1.0, N + 1), dt=dt)

    # Mismatched dt length
    with pytest.raises(ValueError, match="Step durations dt length"):
        Trajectory(X=X, U=U, t=t, dt=jnp.full((N,), 0.1))

    # Horizon too short (< 2)
    with pytest.raises(ValueError, match="Horizon N must be at least 2"):
        Trajectory(X=jnp.zeros((1, n)), U=jnp.zeros((0, m)), t=jnp.zeros((1,)), dt=jnp.zeros((0,)))


def test_knot_point_views() -> None:
    N, n, m = 5, 3, 2
    X = jnp.arange(N * n, dtype=jnp.float64).reshape((N, n))
    U = jnp.arange((N - 1) * m, dtype=jnp.float64).reshape((N - 1, m))
    t = jnp.linspace(0.0, 0.4, N)
    dt = jnp.full((N - 1,), 0.1)

    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    # Intermediate knot point
    kp0 = traj[0]
    assert isinstance(kp0, KnotPoint)
    assert not kp0.is_terminal
    np.testing.assert_allclose(kp0.x, X[0])
    assert kp0.u is not None
    np.testing.assert_allclose(kp0.u, U[0])
    np.testing.assert_allclose(kp0.t, t[0])
    np.testing.assert_allclose(kp0.dt, dt[0])

    kp2 = traj[2]
    assert not kp2.is_terminal
    np.testing.assert_allclose(kp2.x, X[2])
    assert kp2.u is not None
    np.testing.assert_allclose(kp2.u, U[2])
    np.testing.assert_allclose(kp2.t, t[2])
    np.testing.assert_allclose(kp2.dt, dt[2])

    # Terminal knot point (positive index)
    kp_term = traj[N - 1]
    assert kp_term.is_terminal
    np.testing.assert_allclose(kp_term.x, X[N - 1])
    assert kp_term.u is None
    np.testing.assert_allclose(kp_term.t, t[N - 1])
    np.testing.assert_allclose(kp_term.dt, 0.0)

    # Terminal knot point (negative index)
    kp_neg = traj[-1]
    assert kp_neg.is_terminal
    np.testing.assert_allclose(kp_neg.x, X[-1])
    assert kp_neg.u is None

    # Index error out of bounds
    with pytest.raises(IndexError):
        _ = traj[N]
    with pytest.raises(IndexError):
        _ = traj[-N - 1]

    # Iteration
    kps = list(traj)
    assert len(kps) == N
    for k, kp in enumerate(kps):
        assert kp.is_terminal == (k == N - 1)
        np.testing.assert_allclose(kp.x, X[k])
        if not kp.is_terminal:
            assert kp.u is not None
            np.testing.assert_allclose(kp.u, U[k])


def test_trajectory_immutability_and_setters() -> None:
    N, n, m = 5, 2, 1
    X = jnp.zeros((N, n))
    U = jnp.zeros((N - 1, m))
    t = jnp.linspace(0.0, 1.0, N)
    dt = jnp.full((N - 1,), 0.25)

    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    # set_states returns new instance
    new_X = jnp.ones((N, n)) * 2.0
    traj2 = traj.set_states(new_X)
    assert traj2 is not traj
    np.testing.assert_allclose(traj.states(), X)
    np.testing.assert_allclose(traj2.states(), new_X)
    np.testing.assert_allclose(traj2.controls(), U)

    # set_controls returns new instance
    new_U = jnp.ones((N - 1, m)) * 3.0
    traj3 = traj.set_controls(new_U)
    assert traj3 is not traj
    np.testing.assert_allclose(traj.controls(), U)
    np.testing.assert_allclose(traj3.controls(), new_U)
    np.testing.assert_allclose(traj3.states(), X)


def test_trajectory_with_initial_time() -> None:
    N, n, m = 4, 2, 1
    X = jnp.zeros((N, n))
    U = jnp.zeros((N - 1, m))
    dt_val = 0.1
    t = jnp.array([0.0, 0.1, 0.2, 0.3])
    dt = jnp.full((N - 1,), dt_val)

    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    t0_new = 5.0
    traj_shifted = traj.with_initial_time(t0_new)

    assert traj_shifted is not traj
    np.testing.assert_allclose(traj.times(), t)
    expected_t = jnp.array([5.0, 5.1, 5.2, 5.3])
    np.testing.assert_allclose(traj_shifted.times(), expected_t)


def test_trajectory_shift_for_warmstarting() -> None:
    N = 4
    X = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    U = jnp.array([[10.0], [20.0], [30.0]])
    t = jnp.array([0.0, 0.1, 0.2, 0.3])
    dt = jnp.array([0.1, 0.1, 0.1])

    traj = Trajectory(X=X, U=U, t=t, dt=dt)

    # Shift forward one step
    shifted = traj.shift()
    assert shifted is not traj
    assert shifted.N == N

    # States shifted: X[1], X[2], X[3], duplicated X[3]
    expected_X = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0]])
    np.testing.assert_allclose(shifted.states(), expected_X)

    # Controls shifted: U[1], U[2], duplicated U[2]
    expected_U = jnp.array([[20.0], [30.0], [30.0]])
    np.testing.assert_allclose(shifted.controls(), expected_U)

    # Step durations shifted: dt[1], dt[2], duplicated dt[2]
    expected_dt = jnp.array([0.1, 0.1, 0.1])
    np.testing.assert_allclose(shifted.dt, expected_dt)

    # Times shifted: starts at t[1] (0.1) up to t[3] + dt[2] (0.4)
    expected_t = jnp.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(shifted.times(), expected_t)

    # Shift with explicit dt argument
    shifted_custom_dt = traj.shift(dt=0.2)
    expected_t_custom = jnp.array([0.1, 0.2, 0.3, 0.5])
    expected_dt_custom = jnp.array([0.1, 0.1, 0.2])
    np.testing.assert_allclose(shifted_custom_dt.times(), expected_t_custom)
    np.testing.assert_allclose(shifted_custom_dt.dt, expected_dt_custom)
