import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics import (
    RK4,
    DiscretizedDynamics,
    Euler,
    ImplicitMidpoint,
    rollout_states,
)
from trajopt.models import Cartpole, DubinsCar, Pendulum, Quadrotor
from trajopt.models.transforms import (
    ControlRateModel,
    LinearTrajectoryModel,
    control_rate_cost,
    with_control_rate_penalty,
)
from trajopt.rotations.quaternion import Quaternion
from trajopt.trajectory import Trajectory


def test_with_control_rate_penalty_dimensions_euclidean() -> None:
    """Assert with_control_rate_penalty correctly augments Euclidean model dimensions."""
    cartpole = Cartpole()
    assert cartpole.n == 4
    assert cartpole.m == 1
    assert cartpole.ne == 4
    R_delta = jnp.array([1.0])

    # 1. Continuous dynamics input (discretized with default RK4)
    aug_cont, cost_cont = with_control_rate_penalty(cartpole, R_delta)
    assert isinstance(aug_cont, ControlRateModel)
    assert isinstance(cost_cont, QuadraticCost)
    assert aug_cont.n == 5
    assert aug_cont.m == 1
    assert aug_cont.ne == 5
    assert cost_cont.n == 5
    assert cost_cont.m == 1

    # 2. Discrete dynamics input
    disc_cartpole = DiscretizedDynamics(cartpole, RK4())
    aug_disc, cost_disc = with_control_rate_penalty(disc_cartpole, R_delta)
    assert isinstance(aug_disc, ControlRateModel)
    assert isinstance(cost_disc, QuadraticCost)
    assert aug_disc.n == 5
    assert aug_disc.m == 1
    assert aug_disc.ne == 5

    # 3. DubinsCar (m=2)
    car = DubinsCar()
    assert car.n == 3
    assert car.m == 2
    assert car.ne == 3
    R_delta_car = jnp.array([1.0, 2.0])
    aug_car, cost_car = with_control_rate_penalty(car, R_delta_car)
    assert aug_car.n == 5
    assert aug_car.m == 2
    assert aug_car.ne == 5
    assert cost_car.n == 5
    assert cost_car.m == 2


def test_with_control_rate_penalty_dimensions_rigidbody() -> None:
    """Assert with_control_rate_penalty correctly augments RigidBody model dimensions."""
    quad = Quadrotor()
    assert quad.n == 13
    assert quad.m == 4
    assert quad.ne == 12
    R_delta = jnp.ones(4)

    # Continuous input
    aug_quad, cost_quad = with_control_rate_penalty(quad, R_delta)
    assert isinstance(aug_quad, ControlRateModel)
    assert isinstance(cost_quad, QuadraticCost)
    assert aug_quad.n == 17
    assert aug_quad.m == 4
    assert aug_quad.ne == 16
    assert cost_quad.n == 17
    assert cost_quad.m == 4

    # Discrete input
    disc_quad = DiscretizedDynamics(quad, RK4())
    aug_disc_quad, cost_disc_quad = with_control_rate_penalty(disc_quad, R_delta)
    assert isinstance(aug_disc_quad, ControlRateModel)
    assert isinstance(cost_disc_quad, QuadraticCost)
    assert aug_disc_quad.n == 17
    assert aug_disc_quad.m == 4
    assert aug_disc_quad.ne == 16


def test_with_control_rate_penalty_state_diff_and_error_jacobian() -> None:
    """Assert state_diff and errstate_jacobian for both Euclidean and RigidBody models."""
    # 1. Euclidean model: Cartpole (n=4, m=1)
    cartpole = Cartpole()
    R_delta = jnp.array([1.0])
    aug_cartpole, _ = with_control_rate_penalty(cartpole, R_delta)

    x1 = jnp.array([1.0, 0.5, -0.2, 0.3, 2.0])
    x0 = jnp.array([0.5, 0.2, 0.0, -0.1, 1.5])
    dx = aug_cartpole.state_diff(x1, x0)
    assert dx.shape == (5,)
    np.testing.assert_allclose(dx, x1 - x0)

    G_euc = aug_cartpole.errstate_jacobian(x1)
    assert G_euc.shape == (5, 5)
    np.testing.assert_allclose(G_euc, np.eye(5))

    # 2. RigidBody model: Quadrotor (n=13, m=4, ne=12 -> n_aug=17, ne_aug=16)
    quad = Quadrotor()
    aug_quad, _ = with_control_rate_penalty(quad, jnp.ones(4))

    q1 = Quaternion.from_array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])
    q0 = Quaternion.from_array([0.0, np.sin(-0.2), 0.0, np.cos(-0.2)])
    x_quad1 = jnp.array([1.0, 2.0, 3.0, *q1.to_array(), 0.5, -0.5, 1.0, 0.1, 0.2, -0.3, 1.0, 1.1, 1.2, 1.3])
    x_quad0 = jnp.array([0.5, 1.5, 2.0, *q0.to_array(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 1.0, 1.1])

    dx_quad = aug_quad.state_diff(x_quad1, x_quad0)
    assert dx_quad.shape == (16,)
    np.testing.assert_allclose(dx_quad[:12], quad.state_diff(x_quad1[:13], x_quad0[:13]))
    np.testing.assert_allclose(dx_quad[12:], x_quad1[13:] - x_quad0[13:])

    G_quad = aug_quad.errstate_jacobian(x_quad1)
    assert G_quad.shape == (17, 16)
    np.testing.assert_allclose(G_quad[:13, :12], quad.errstate_jacobian(x_quad1[:13]))
    np.testing.assert_allclose(G_quad[:13, 12:], np.zeros((13, 4)))
    np.testing.assert_allclose(G_quad[13:, :12], np.zeros((4, 12)))
    np.testing.assert_allclose(G_quad[13:, 12:], np.eye(4))


def test_control_rate_cost_parity_with_hand_calculation() -> None:
    """Assert control-rate penalty produces identical trajectory cost to coupled hand calculation."""
    N = 10
    m = 2
    n = 3
    rng = np.random.default_rng(42)

    U = rng.standard_normal((N - 1, m))
    u_prev_0 = rng.standard_normal(m)

    # 1. Diagonal R_delta
    R_delta_diag = np.array([2.5, 1.8])
    cost_diag = control_rate_cost(jnp.array(R_delta_diag), n=n, m=m)
    assert isinstance(cost_diag, QuadraticCost)
    assert cost_diag.n == n + m
    assert cost_diag.m == m
    assert cost_diag.has_cross_coupling

    # Hand-computed coupled penalty
    cost_hand_diag = 0.0
    u_prev = u_prev_0
    for k in range(N - 1):
        u_k = U[k]
        du = u_k - u_prev
        cost_hand_diag += 0.5 * np.sum(R_delta_diag * (du**2))
        u_prev = u_k

    # Augmented trajectory cost
    cost_aug_diag = 0.0
    u_prev = u_prev_0
    for k in range(N - 1):
        x_aug = jnp.concatenate([jnp.zeros(n), jnp.array(u_prev)])
        u_k = jnp.array(U[k])
        cost_aug_diag += float(cost_diag.evaluate(x_aug, u_k))
        u_prev = U[k]

    np.testing.assert_allclose(cost_aug_diag, cost_hand_diag, rtol=1e-14, atol=1e-14)

    # 2. Dense R_delta
    R_delta_dense = np.array([[3.0, 0.5], [0.5, 2.0]])
    cost_dense = control_rate_cost(jnp.array(R_delta_dense), n=n, m=m)
    assert isinstance(cost_dense, QuadraticCost)

    cost_hand_dense = 0.0
    u_prev = u_prev_0
    for k in range(N - 1):
        u_k = U[k]
        du = u_k - u_prev
        cost_hand_dense += 0.5 * float(du @ R_delta_dense @ du)
        u_prev = u_k

    cost_aug_dense = 0.0
    u_prev = u_prev_0
    for k in range(N - 1):
        x_aug = jnp.concatenate([jnp.zeros(n), jnp.array(u_prev)])
        u_k = jnp.array(U[k])
        cost_aug_dense += float(cost_dense.evaluate(x_aug, u_k))
        u_prev = U[k]

    np.testing.assert_allclose(cost_aug_dense, cost_hand_dense, rtol=1e-14, atol=1e-14)


def test_cost_hessian_block_diagonal_verification() -> None:
    """Assert state augmentation turns a knot-coupling control-rate penalty into a block-diagonal one at equal cost."""
    N = 5
    n = 2
    m = 1
    R_delta = np.array([4.0])
    stage_cost = control_rate_cost(jnp.array(R_delta), n=n, m=m)

    rng = np.random.default_rng(7)
    X = rng.standard_normal((N, n))
    U = rng.standard_normal((N - 1, m))
    u_prev_0 = rng.standard_normal(m)

    # Layout A, unaugmented: knot k is [x_k, u_k], terminal knot is [x_{N-1}].
    # The penalty at knot k reads u_{k-1} out of knot k-1, so knots couple.
    plain_slices = [(k * (n + m), k * (n + m) + n + m) for k in range(N - 1)]
    plain_slices.append(((N - 1) * (n + m), (N - 1) * (n + m) + n))
    Z_plain = jnp.array(np.concatenate([*(np.concatenate([X[k], U[k]]) for k in range(N - 1)), X[-1]]))

    def coupled_cost(Z: jax.Array) -> jax.Array:
        cost = 0.0
        u_prev = jnp.array(u_prev_0)
        for k in range(N - 1):
            u_k = Z[k * (n + m) + n : k * (n + m) + n + m]
            du = u_k - u_prev
            cost = cost + 0.5 * jnp.sum(jnp.asarray(R_delta) * (du**2))
            u_prev = u_k
        return cost

    # Layout B, augmented: knot k is [x_k, u_{k-1}, u_k]. The same penalty is now a stage cost.
    aug_slices = [(k * (n + 2 * m), k * (n + 2 * m) + n + 2 * m) for k in range(N - 1)]
    aug_slices.append(((N - 1) * (n + 2 * m), (N - 1) * (n + 2 * m) + n + m))
    u_prev_of = [u_prev_0, *[U[k] for k in range(N - 2)]]
    Z_aug = jnp.array(
        np.concatenate(
            [
                *(np.concatenate([X[k], u_prev_of[k], U[k]]) for k in range(N - 1)),
                np.concatenate([X[-1], U[-1]]),
            ]
        )
    )

    def augmented_cost(Z: jax.Array) -> jax.Array:
        cost = 0.0
        for k in range(N - 1):
            base = k * (n + 2 * m)
            x_aug_k = Z[base : base + n + m]
            u_k = Z[base + n + m : base + n + 2 * m]
            cost = cost + stage_cost.evaluate(x_aug_k, u_k)
        return cost

    # 1. Both layouts express the same penalty, so the comparison below is like for like.
    np.testing.assert_allclose(float(augmented_cost(Z_aug)), float(coupled_cost(Z_plain)), rtol=1e-14, atol=1e-14)

    def max_offdiag(H: jax.Array, slices: list[tuple[int, int]]) -> float:
        return max(
            float(jnp.max(jnp.abs(H[a:b, c:d])))
            for i, (a, b) in enumerate(slices)
            for j, (c, d) in enumerate(slices)
            if i != j
        )

    # 2. Unaugmented, the penalty couples adjacent knots: block tridiagonal.
    H_coupled = jax.hessian(coupled_cost)(Z_plain)
    np.testing.assert_allclose(float(H_coupled[n, n + m + n]), -R_delta[0], rtol=1e-14)
    assert max_offdiag(H_coupled, plain_slices) > 0.0, "Coupled penalty must produce off-diagonal knot blocks"

    # 3. Augmented, the identical penalty is stage separable: block diagonal.
    H_aug = jax.hessian(augmented_cost)(Z_aug)
    assert max_offdiag(H_aug, aug_slices) == 0.0, "Augmented penalty must have no off-diagonal knot blocks"


def test_composition_with_integrators_and_rollout() -> None:
    """Assert augmented model seamlessly composes with Euler, RK4, ImplicitMidpoint and rollout."""
    cartpole = Cartpole()
    N = 8
    dt = 0.05
    t = 0.0
    rng = np.random.default_rng(123)

    U = jnp.array(rng.uniform(-2.0, 2.0, size=(N - 1, 1)))
    x0_orig = jnp.array([0.1, -0.2, 0.3, -0.1])
    u_prev_0 = jnp.array([0.5])
    x0_aug = jnp.concatenate([x0_orig, u_prev_0])
    R_delta = jnp.array([1.0])

    # 1. RK4
    disc_rk4 = RK4(cartpole)
    aug_rk4, _ = with_control_rate_penalty(disc_rk4, R_delta)

    X_sim_aug = rollout_states(aug_rk4, x0_aug, U, dt=dt)
    assert X_sim_aug.shape == (N, 5)

    for k in range(N - 1):
        x_next_expected = disc_rk4.discrete_dynamics(X_sim_aug[k, :4], U[k], t + k * dt, dt)
        np.testing.assert_allclose(X_sim_aug[k + 1, :4], x_next_expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(X_sim_aug[k + 1, 4:], U[k], rtol=1e-14, atol=1e-14)

    # 2. Euler
    disc_euler = Euler(cartpole)
    aug_euler, _ = with_control_rate_penalty(disc_euler, R_delta)
    X_sim_euler = rollout_states(aug_euler, x0_aug, U, dt=dt)
    assert X_sim_euler.shape == (N, 5)
    for k in range(N - 1):
        x_next_expected = disc_euler.discrete_dynamics(X_sim_euler[k, :4], U[k], t + k * dt, dt)
        np.testing.assert_allclose(X_sim_euler[k + 1, :4], x_next_expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(X_sim_euler[k + 1, 4:], U[k], rtol=1e-14, atol=1e-14)

    # 3. Implicit Midpoint
    disc_mid = ImplicitMidpoint(cartpole, iters=10)
    aug_mid, _ = with_control_rate_penalty(disc_mid, R_delta)
    X_sim_mid = rollout_states(aug_mid, x0_aug, U, dt=dt)
    assert X_sim_mid.shape == (N, 5)
    for k in range(N - 1):
        x_next_expected = disc_mid.discrete_dynamics(X_sim_mid[k, :4], U[k], t + k * dt, dt)
        np.testing.assert_allclose(X_sim_mid[k + 1, :4], x_next_expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(X_sim_mid[k + 1, 4:], U[k], rtol=1e-14, atol=1e-14)

    # 4. Trajectory rollout object
    traj = Trajectory(X=jnp.zeros((N, 5)), U=U, t=jnp.arange(N) * dt, dt=jnp.full(N - 1, dt))
    traj_rolled = aug_rk4.rollout(traj, x0=x0_aug)
    np.testing.assert_allclose(traj_rolled.X, X_sim_aug)


def test_linearize_about_euclidean_models() -> None:
    """Assert model.linearize produces stacked state and control Jacobians along reference trajectory."""
    # 1. Cartpole with RK4
    cartpole = Cartpole()
    disc_cartpole = DiscretizedDynamics(cartpole, RK4())
    N = 10
    dt = 0.05

    rng = np.random.default_rng(200)
    U_ref = jnp.array(rng.uniform(-1.5, 1.5, size=(N - 1, 1)))
    x0 = jnp.array([0.0, 0.1, 0.0, 0.0])
    X_ref = rollout_states(disc_cartpole, x0, U_ref, dt=dt)
    traj_ref = Trajectory(X=X_ref, U=U_ref, t=jnp.arange(N) * dt, dt=jnp.full(N - 1, dt))

    lin = disc_cartpole.linearize(traj_ref)
    assert isinstance(lin, LinearTrajectoryModel)
    assert lin.A.shape == (N - 1, 4, 4)
    assert lin.B.shape == (N - 1, 4, 1)
    assert lin.n == 4
    assert lin.m == 1
    assert lin.ne == 4
    assert lin.N == N

    for k in range(N - 1):
        xk = X_ref[k]
        uk = U_ref[k]
        tk = k * dt
        Ak_expected = disc_cartpole.state_jacobian(xk, uk, tk, dt)
        Bk_expected = disc_cartpole.control_jacobian(xk, uk, tk, dt)
        np.testing.assert_allclose(lin.A[k], Ak_expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(lin.B[k], Bk_expected, rtol=1e-14, atol=1e-14)

    # 2. Continuous model without explicit discretization (defaults to RK4)
    lin_cont = cartpole.linearize(traj_ref)
    np.testing.assert_allclose(lin_cont.A, lin.A)
    np.testing.assert_allclose(lin_cont.B, lin.B)


def test_linearize_about_rigidbody_quadrotor() -> None:
    """Assert model.linearize applies error-state sandwich for RigidBody Quadrotor."""
    quad = Quadrotor()
    disc_quad = DiscretizedDynamics(quad, RK4())
    N = 8
    dt = 0.05

    rng = np.random.default_rng(300)
    U_ref = jnp.array(rng.uniform(1.0, 2.5, size=(N - 1, 4)))
    q0 = Quaternion.from_array([0.0, 0.0, 0.0, 1.0])
    x0 = jnp.array([0.0, 0.0, 1.0, *q0.to_array(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_ref = rollout_states(disc_quad, x0, U_ref, dt=dt)
    traj = Trajectory(X=X_ref, U=U_ref, t=jnp.arange(N) * dt, dt=jnp.full(N - 1, dt))

    lin = disc_quad.linearize(traj)
    assert lin.A.shape == (N - 1, 12, 12)
    assert lin.B.shape == (N - 1, 12, 4)
    assert lin.n == 13
    assert lin.m == 4
    assert lin.ne == 12

    # Cross-verify with dynamics_expansion
    exp = disc_quad.dynamics_expansion(traj)
    np.testing.assert_allclose(lin.A, exp.A, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(lin.B, exp.B, rtol=1e-14, atol=1e-14)


def test_linearize_about_taylor_prediction() -> None:
    """Assert linear model predicts first-order perturbations with second-order error."""
    cartpole = Cartpole()
    disc_cartpole = DiscretizedDynamics(cartpole, RK4())
    dt = 0.05

    x_ref = jnp.array([0.5, 0.2, -0.1, 0.3])
    u_ref = jnp.array([1.2])
    X_ref = jnp.stack([x_ref, disc_cartpole.discrete_dynamics(x_ref, u_ref, 0.0, dt)], axis=0)
    U_ref = jnp.expand_dims(u_ref, axis=0)
    traj_ref = Trajectory(X=X_ref, U=U_ref, t=jnp.array([0.0, dt]), dt=jnp.array([dt]))

    lin = disc_cartpole.linearize(traj_ref)
    A0 = lin.A[0]
    B0 = lin.B[0]

    dx = jnp.array([0.05, -0.03, 0.02, 0.04])
    du = jnp.array([0.08])

    epsilons = [1e-1, 1e-2, 1e-3, 1e-4]
    errors = []
    for eps in epsilons:
        x_pert = x_ref + eps * dx
        u_pert = u_ref + eps * du
        x_next_nonlinear = disc_cartpole.discrete_dynamics(x_pert, u_pert, 0.0, dt)
        x_next_linear = X_ref[1] + eps * (A0 @ dx + B0 @ du)
        err = float(jnp.linalg.norm(x_next_nonlinear - x_next_linear))
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 50.0
    assert ratio2 > 50.0


def test_transforms_equinox_jit_compatibility() -> None:
    """Assert model transforms are valid Equinox pytrees and compatible with filter_jit."""
    cartpole = Cartpole()
    disc = RK4(cartpole)
    R_delta = jnp.array([1.0])
    aug, _ = with_control_rate_penalty(disc, R_delta)

    @eqx.filter_jit
    def step_aug(m: ControlRateModel, x: jax.Array, u: jax.Array) -> jax.Array:
        return m.discrete_dynamics(x, u, 0.0, 0.05)

    x0 = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    u0 = jnp.array([1.0])
    x1 = step_aug(aug, x0, u0)
    assert x1.shape == (5,)

    @eqx.filter_jit
    def lin_traj(m: DiscretizedDynamics, X: jax.Array, U: jax.Array) -> LinearTrajectoryModel:
        traj = Trajectory(X=X, U=U, t=jnp.arange(6) * 0.05, dt=jnp.full(5, 0.05))
        return m.linearize(traj)

    X_ref = jnp.zeros((6, 4))
    U_ref = jnp.zeros((5, 1))
    lin = lin_traj(disc, X_ref, U_ref)
    assert lin.A.shape == (5, 4, 4)
    assert lin.B.shape == (5, 4, 1)
