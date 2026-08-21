"""Unit tests for Euler, RK4, and ImplicitMidpoint integrators and their analytical properties."""

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.dynamics import (
    RK4,
    ContinuousDynamics,
    DiscretizedDynamics,
    EuclideanModel,
    Euler,
    ImplicitMidpoint,
    euler_step,
    implicit_midpoint_step,
    rk4_step,
)


class ExponentialDecay(EuclideanModel):
    """Linear decay system: xdot = -lambda * x."""

    lam: float = 2.0

    def __init__(self, lam: float = 2.0) -> None:
        super().__init__(n=1, m=1)
        self.lam = lam

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del u, t
        return -self.lam * x


class HarmonicOscillator(EuclideanModel):
    """Simple harmonic oscillator: xdot = [x2, -omega^2 * x1]."""

    omega: float = 1.0

    def __init__(self, omega: float = 1.0) -> None:
        super().__init__(n=2, m=1)
        self.omega = omega

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del t
        return jnp.array([x[1], -(self.omega**2) * x[0] + u[0]])


class TimeDependentSystem(EuclideanModel):
    """Time-dependent ODE: xdot = 3 * t^2 + u."""

    def __init__(self) -> None:
        super().__init__(n=1, m=1)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del x
        t_val = jnp.asarray(t)
        return jnp.array([3.0 * (t_val**2) + u[0]])


class LinearSystem(EuclideanModel):
    """Linear continuous system xdot = A x + B u."""

    A: jax.Array
    B: jax.Array

    def __init__(self, A: jax.Array, B: jax.Array) -> None:
        super().__init__(n=A.shape[0], m=B.shape[1])
        self.A = A
        self.B = B

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del t
        return self.A @ x + self.B @ u


def test_integrator_instantiation_and_composition() -> None:
    model = HarmonicOscillator(omega=2.0)

    # 1. Instantiation as pure integrators
    euler = Euler()
    rk4 = RK4()
    mid = ImplicitMidpoint(iters=8)
    assert mid.iters == 8

    # 2. Composition via DiscretizedDynamics(model, integrator)
    d_euler = DiscretizedDynamics(model, euler)
    d_rk4 = DiscretizedDynamics(model, rk4)
    d_mid = DiscretizedDynamics(model, mid)

    assert isinstance(d_euler, DiscretizedDynamics)
    assert isinstance(d_rk4, DiscretizedDynamics)
    assert isinstance(d_mid, DiscretizedDynamics)

    # 3. Direct constructor composition RK4(model), Euler(model), ImplicitMidpoint(model)
    d_euler_direct = Euler(model)
    d_rk4_direct = RK4(model)
    d_mid_direct = ImplicitMidpoint(model, iters=12)

    assert isinstance(d_euler_direct, DiscretizedDynamics)
    assert isinstance(d_rk4_direct, DiscretizedDynamics)
    assert isinstance(d_mid_direct, DiscretizedDynamics)
    assert isinstance(d_mid_direct.integrator, ImplicitMidpoint)
    assert d_mid_direct.integrator.iters == 12

    # 4. Functional step callables in DiscretizedDynamics
    d_rk4_func = DiscretizedDynamics(model, rk4_step)
    assert isinstance(d_rk4_func, DiscretizedDynamics)

    x = jnp.array([1.0, 0.0])
    u = jnp.array([0.0])
    dt = 0.01

    np.testing.assert_allclose(
        d_rk4.discrete_dynamics(x, u, 0.0, dt),
        d_rk4_direct.discrete_dynamics(x, u, 0.0, dt),
    )
    np.testing.assert_allclose(
        d_rk4.discrete_dynamics(x, u, 0.0, dt),
        d_rk4_func.discrete_dynamics(x, u, 0.0, dt),
    )


def test_exponential_decay_analytical_convergence_rates() -> None:
    """Verifies that Euler is 1st order, ImplicitMidpoint is 2nd order, and RK4 is 4th order."""
    lam = 1.5
    model = ExponentialDecay(lam=lam)
    T = 1.0
    x0 = jnp.array([2.0])
    u0 = jnp.array([0.0])
    x_exact = float(x0[0] * np.exp(-lam * T))

    dts = [0.1, 0.05, 0.025, 0.0125]

    def simulate(integrator_step, dt_val, iters_arg=None):
        n_steps = round(T / dt_val)

        def _step_fn(carry, _):
            x_curr, t_curr = carry
            if iters_arg is not None:
                x_next = integrator_step(model, x_curr, u0, t_curr, dt_val, iters=iters_arg)
            else:
                x_next = integrator_step(model, x_curr, u0, t_curr, dt_val)
            return (x_next, t_curr + dt_val), None

        (x_final, _), _ = jax.lax.scan(_step_fn, (x0, 0.0), None, length=n_steps)
        return float(x_final[0])

    errors_euler = [abs(simulate(euler_step, dt) - x_exact) for dt in dts]
    errors_mid = [abs(simulate(implicit_midpoint_step, dt, iters_arg=10) - x_exact) for dt in dts]
    errors_rk4 = [abs(simulate(rk4_step, dt) - x_exact) for dt in dts]

    # Compute convergence rates = log2(error[k] / error[k+1])
    rates_euler = [np.log2(errors_euler[k] / errors_euler[k + 1]) for k in range(len(dts) - 1)]
    rates_mid = [np.log2(errors_mid[k] / errors_mid[k + 1]) for k in range(len(dts) - 1)]
    rates_rk4 = [np.log2(errors_rk4[k] / errors_rk4[k + 1]) for k in range(len(dts) - 1)]

    # Euler should converge with rate ~ 1.0
    for r in rates_euler:
        assert 0.9 <= r <= 1.1

    # Implicit midpoint should converge with rate ~ 2.0
    for r in rates_mid:
        assert 1.9 <= r <= 2.1

    # RK4 should converge with rate ~ 4.0
    for r in rates_rk4:
        assert 3.9 <= r <= 4.1


def test_harmonic_oscillator_symplectic_energy_preservation() -> None:
    """Verifies that Implicit Midpoint conserves quadratic Hamiltonian/energy exactly."""
    omega = 3.0
    model = HarmonicOscillator(omega=omega)
    x0 = jnp.array([1.0, 0.5])
    u0 = jnp.array([0.0])
    dt = 0.05
    n_steps = 200

    def energy(x):
        return 0.5 * (omega**2) * (x[0] ** 2) + 0.5 * (x[1] ** 2)

    e0 = energy(x0)

    # 1. Implicit midpoint energy preservation via scan
    def _mid_step(carry, _):
        x_curr, t_curr = carry
        x_next = implicit_midpoint_step(model, x_curr, u0, t_curr, dt, iters=10)
        return (x_next, t_curr + dt), None

    (x_final_mid, _), _ = jax.lax.scan(_mid_step, (x0, 0.0), None, length=n_steps)

    e_final_mid = energy(x_final_mid)
    # Energy error should be at machine precision (< 1e-11)
    np.testing.assert_allclose(e_final_mid, e0, rtol=1e-11, atol=1e-11)

    # 2. Compare to Euler which blows up energy
    def _euler_step(carry, _):
        x_curr, t_curr = carry
        x_next = euler_step(model, x_curr, u0, t_curr, dt)
        return (x_next, t_curr + dt), None

    (x_final_euler, _), _ = jax.lax.scan(_euler_step, (x0, 0.0), None, length=n_steps)

    e_final_euler = energy(x_final_euler)
    # Euler has massive energy growth
    assert e_final_euler > 10.0 * e0


def test_time_dependent_dynamics_stepping() -> None:
    """Verifies that time t and half-step times t + dt/2 are passed correctly."""
    model = TimeDependentSystem()
    x0 = jnp.array([0.0])
    u = jnp.array([1.0])
    t0 = 2.0
    dt = 0.5

    # Exact integral of xdot = 3 t^2 + 1 from t0 to t0+dt:
    # x(t0+dt) - x(t0) = [(t0+dt)^3 + (t0+dt)] - [t0^3 + t0]
    x_exact = ((t0 + dt) ** 3 + (t0 + dt)) - (t0**3 + t0)

    # RK4 integrates polynomials of degree <= 3 EXACTLY!
    x_rk4 = rk4_step(model, x0, u, t0, dt)
    np.testing.assert_allclose(x_rk4[0], x_exact, rtol=1e-14, atol=1e-14)


def test_linear_system_analytical_discrete_jacobians() -> None:
    """Verifies discrete state and control Jacobians [A_d, B_d] against exact analytical matrix formulas."""
    A = jnp.array([[0.0, 1.0], [-2.0, -1.5]])
    B = jnp.array([[0.0], [1.0]])
    model = LinearSystem(A, B)

    x = jnp.array([0.5, -0.2])
    u = jnp.array([1.0])
    t = 0.0
    dt = 0.1
    I2 = jnp.eye(2)

    # 1. Euler
    d_euler = Euler(model)
    J_euler = d_euler.jacobian(x, u, t, dt)
    expected_Ad_euler = I2 + dt * A
    expected_Bd_euler = dt * B
    np.testing.assert_allclose(J_euler[:, :2], expected_Ad_euler, rtol=1e-12)
    np.testing.assert_allclose(J_euler[:, 2:], expected_Bd_euler, rtol=1e-12)

    # 2. RK4
    d_rk4 = RK4(model)
    J_rk4 = d_rk4.jacobian(x, u, t, dt)
    dtA = dt * A
    expected_Ad_rk4 = I2 + dtA + (dtA @ dtA) / 2.0 + (dtA @ dtA @ dtA) / 6.0 + (dtA @ dtA @ dtA @ dtA) / 24.0
    expected_Bd_rk4 = (dt * I2 + (dt**2 * A) / 2.0 + (dt**3 * (A @ A)) / 6.0 + (dt**4 * (A @ A @ A)) / 24.0) @ B
    np.testing.assert_allclose(J_rk4[:, :2], expected_Ad_rk4, rtol=1e-12)
    np.testing.assert_allclose(J_rk4[:, 2:], expected_Bd_rk4, rtol=1e-12)

    # 3. Implicit Midpoint (Cayley transform)
    d_mid = ImplicitMidpoint(model, iters=10)
    J_mid = d_mid.jacobian(x, u, t, dt)
    inv_mid = jnp.linalg.inv(I2 - 0.5 * dt * A)
    expected_Ad_mid = inv_mid @ (I2 + 0.5 * dt * A)
    expected_Bd_mid = inv_mid @ (dt * B)
    np.testing.assert_allclose(J_mid[:, :2], expected_Ad_mid, rtol=1e-12)
    np.testing.assert_allclose(J_mid[:, 2:], expected_Bd_mid, rtol=1e-12)
