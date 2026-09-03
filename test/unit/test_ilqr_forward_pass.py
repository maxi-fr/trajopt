import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective, Objective
from trajopt.dynamics import DiscreteDynamics
from trajopt.solvers.ilqr import (
    DynamicRegularization,
    forward_pass,
    increase_regularization,
    rollout_closed_loop,
)
from trajopt.solvers.options import SolverOptions, TerminationStatus
from trajopt.trajectory import Trajectory


class _LinearDiscreteModel(DiscreteDynamics):
    """Scalar discrete linear model x_{k+1} = Ad*x + Bd*u, for controlled unit tests."""

    Ad: jax.Array
    Bd: jax.Array

    def __init__(self, Ad: float, Bd: float) -> None:
        super().__init__(n=1, m=1, ne=1)
        self.Ad = jnp.asarray([[Ad]])
        self.Bd = jnp.asarray([[Bd]])

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        del t, dt
        return self.Ad @ x + self.Bd @ u


def _trajectory(X: list[float], U: list[float]) -> Trajectory:
    X_arr = jnp.asarray(X, dtype=jnp.float64).reshape(-1, 1)
    U_arr = jnp.asarray(U, dtype=jnp.float64).reshape(-1, 1)
    N = X_arr.shape[0]
    t = jnp.arange(N, dtype=jnp.float64)
    dt = jnp.ones(N - 1, dtype=jnp.float64)
    return Trajectory(X=X_arr, U=U_arr, t=t, dt=dt)


# --- rollout_closed_loop -----------------------------------------------------------------


def test_rollout_closed_loop_matches_manual_computation() -> None:
    """u_k = ubar_k + K_k @ dx_k + alpha*d_k, dx_k from state_diff against the nominal."""
    model = _LinearDiscreteModel(Ad=1.0, Bd=1.0)
    nominal = _trajectory([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    K = jnp.zeros((3, 1, 1))
    d = jnp.asarray([[1.0], [2.0], [-1.0]])
    alpha = 0.5
    options = SolverOptions()

    result = rollout_closed_loop(model, nominal, K, d, alpha, options)

    # u0 = 0.5, x1 = 0.5; u1 = 1.0, x2 = 1.5; u2 = -0.5, x3 = 1.0
    np.testing.assert_allclose(np.asarray(result.X).ravel(), [0.0, 0.5, 1.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(result.U).ravel(), [0.5, 1.0, -0.5], atol=1e-12)
    assert not bool(result.failed)
    assert int(result.status) == int(TerminationStatus.UNSOLVED)


def test_rollout_closed_loop_is_jittable() -> None:
    """rollout_closed_loop runs unchanged under jax.jit with SolverOptions held static."""
    model = _LinearDiscreteModel(Ad=1.0, Bd=1.0)
    nominal = _trajectory([0.0, 0.0, 0.0], [0.0, 0.0])
    K = jnp.zeros((2, 1, 1))
    d = jnp.asarray([[1.0], [-2.0]])
    options = SolverOptions()

    jitted = jax.jit(functools.partial(rollout_closed_loop, options=options))
    eager = rollout_closed_loop(model, nominal, K, d, 0.3, options)
    traced = jitted(model, nominal, K, d, 0.3)

    np.testing.assert_allclose(np.asarray(traced.X), np.asarray(eager.X))
    np.testing.assert_allclose(np.asarray(traced.U), np.asarray(eager.U))


def test_rollout_closed_loop_zero_alpha_reproduces_nominal_exactly() -> None:
    """alpha = 0 (any K, d) reproduces a dynamically consistent nominal trajectory exactly."""
    model = _LinearDiscreteModel(Ad=1.2, Bd=0.8)
    x0 = jnp.asarray([2.0])
    traj_skeleton = _trajectory([0.0, 0.0, 0.0, 0.0], [0.3, -0.4, 0.1])
    nominal = model.rollout(traj_skeleton, x0=x0)

    K = jnp.ones((3, 1, 1)) * 5.0
    d = jnp.asarray([[10.0], [-7.0], [3.0]])
    options = SolverOptions()

    result = rollout_closed_loop(model, nominal, K, d, 0.0, options)

    np.testing.assert_allclose(np.asarray(result.X), np.asarray(nominal.X), atol=1e-12)
    np.testing.assert_allclose(np.asarray(result.U), np.asarray(nominal.U), atol=1e-12)
    assert not bool(result.failed)


def test_rollout_closed_loop_control_only_guard_fires() -> None:
    """A control-only guard trip (state unaffected) is reported as CONTROL_LIMIT."""
    model = _LinearDiscreteModel(Ad=1.0, Bd=0.0)  # control has no effect on the next state
    nominal = _trajectory([0.0, 0.0, 0.0], [0.0, 0.0])
    K = jnp.zeros((2, 1, 1))
    d = jnp.asarray([[1e9], [0.0]])
    options = SolverOptions()

    result = rollout_closed_loop(model, nominal, K, d, 1.0, options)

    assert bool(result.failed)
    assert int(result.status) == int(TerminationStatus.CONTROL_LIMIT)


def test_rollout_closed_loop_state_takes_precedence_within_a_knot() -> None:
    """When state and control both trip at the same knot, STATE_LIMIT wins (checked first)."""
    model = _LinearDiscreteModel(Ad=1.0, Bd=1.0)  # control directly drives the next state
    nominal = _trajectory([0.0, 0.0, 0.0], [0.0, 0.0])
    K = jnp.zeros((2, 1, 1))
    d = jnp.asarray([[1e9], [0.0]])
    options = SolverOptions()

    result = rollout_closed_loop(model, nominal, K, d, 1.0, options)

    assert bool(result.failed)
    assert int(result.status) == int(TerminationStatus.STATE_LIMIT)


def test_rollout_closed_loop_first_failure_across_knots_wins() -> None:
    """A CONTROL_LIMIT trip at knot 0 is reported even though a later knot trips STATE_LIMIT."""
    model = _LinearDiscreteModel(Ad=2.0, Bd=0.0)  # state grows on its own; control has no effect
    N = 9
    nominal = _trajectory([1.0] * N, [0.0] * (N - 1))
    K = jnp.zeros((N - 1, 1, 1))
    d = jnp.zeros((N - 1, 1)).at[0, 0].set(1e9)  # only knot 0's control trips
    options = SolverOptions(max_state_value=100.0)  # x = 2**k crosses 100 well before knot N-1

    result = rollout_closed_loop(model, nominal, K, d, 1.0, options)

    assert bool(result.failed)
    assert int(result.status) == int(TerminationStatus.CONTROL_LIMIT)


# --- forward_pass -------------------------------------------------------------------------

_Q = jnp.asarray([1.0])
_R = jnp.asarray([1.0])


def _lqr_setup(x0: float, N: int) -> tuple[_LinearDiscreteModel, Trajectory, Objective]:
    model = _LinearDiscreteModel(Ad=1.0, Bd=1.0)
    nominal = _trajectory([x0] * N, [0.0] * (N - 1))
    obj = LQRObjective(_Q, _R, _Q, N)
    return model, nominal, obj


def test_forward_pass_accepts_within_two_sided_interval() -> None:
    """A step with z = 1 (exact dV match) is accepted immediately at alpha = 1."""
    model, nominal, obj = _lqr_setup(x0=10.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[-5.0]])
    dV = jnp.asarray([-50.0, 25.0])  # exact quadratic match: z(alpha) == 1 for all alpha
    J_prev = jnp.asarray(100.0)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    result = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)

    assert float(result.alpha) == pytest.approx(1.0)
    assert float(result.J) == pytest.approx(75.0)
    assert not bool(result.ls_failed)
    assert int(result.status) == int(TerminationStatus.UNSOLVED)
    np.testing.assert_allclose(float(result.regularization.rho), float(reg.rho))


def test_forward_pass_guard_retry_then_accepts() -> None:
    """A control-limit guard trip at alpha=1 retries with a halved alpha and then accepts."""
    model, nominal, obj = _lqr_setup(x0=10.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[-5.0]])
    dV = jnp.asarray([-50.0, 25.0])  # z(alpha) == 1 for all alpha once the rollout succeeds
    J_prev = jnp.asarray(100.0)
    options = SolverOptions(max_control_value=4.0)  # |u0| = 5 at alpha=1 trips, 2.5 at alpha=0.5 doesn't
    reg = DynamicRegularization.initial(options)

    result = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)

    assert float(result.alpha) == pytest.approx(0.5)
    assert not bool(result.ls_failed)
    assert int(result.status) == int(TerminationStatus.UNSOLVED)
    np.testing.assert_allclose(float(result.regularization.rho), float(reg.rho))


def test_forward_pass_no_step_when_expected_decrease_too_small() -> None:
    """0 < expected < expected_decrease_tolerance takes no step and bumps regularization once."""
    model, nominal, obj = _lqr_setup(x0=10.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[-5.0]])
    dV = jnp.asarray([-1e-11, 0.0])  # expected(alpha=1) = 1e-11, in (0, 1e-10)
    J_prev = jnp.asarray(200.0)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    result = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)

    assert float(result.alpha) == pytest.approx(0.0)
    assert float(result.J) == pytest.approx(200.0)
    np.testing.assert_allclose(np.asarray(result.trajectory.X), np.asarray(nominal.X))
    assert not bool(result.ls_failed)
    assert int(result.status) == int(TerminationStatus.UNSOLVED)

    expected_reg = increase_regularization(reg, options)
    np.testing.assert_allclose(float(result.regularization.rho), float(expected_reg.rho))
    np.testing.assert_allclose(float(result.regularization.drho), float(expected_reg.drho))


def test_forward_pass_rejects_via_upper_bound_then_accepts() -> None:
    """z(alpha=1) = 50 > z_ub = 10 is rejected by the upper bound; alpha=0.5 then accepts."""
    model, nominal, obj = _lqr_setup(x0=10.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[-5.0]])
    dV = jnp.asarray([-74.5, 74.0])  # expected(1) = 0.5 -> z(1) = 50; expected(0.5) = 18.75 -> z(0.5) = 1
    J_prev = jnp.asarray(100.0)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    result = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)

    assert float(result.alpha) == pytest.approx(0.5)
    assert float(result.J) == pytest.approx(81.25)  # x0=10, u0=-2.5, x1=7.5 -> 0.5*(100+6.25+56.25)
    assert not bool(result.ls_failed)
    assert int(result.status) == int(TerminationStatus.UNSOLVED)


def test_forward_pass_iteration_exhaustion_sets_ls_failed_and_extra_regularization() -> None:
    """z <= 0 for every attempt exhausts the search: ls_failed, alpha=0, rho gets +bp_reg_fp."""
    model, nominal, obj = _lqr_setup(x0=10.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[-5.0]])
    dV = jnp.asarray([1.0, 0.0])  # expected(alpha) = -alpha < 0 for every alpha -> z = -1 always
    J_prev = jnp.asarray(200.0)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    result = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)

    assert float(result.alpha) == pytest.approx(0.0)
    assert float(result.J) == pytest.approx(200.0)
    assert bool(result.ls_failed)

    expected_reg = increase_regularization(reg, options)
    expected_rho = float(expected_reg.rho) + options.bp_reg_fp
    np.testing.assert_allclose(float(result.regularization.rho), expected_rho)
    np.testing.assert_allclose(float(result.regularization.drho), float(expected_reg.drho))


def test_forward_pass_guard_exhaustion_exits_through_cost_increase() -> None:
    """Every rollout guard-trips: ls_failed stays false, J is NaN, status is COST_INCREASE."""
    model, nominal, obj = _lqr_setup(x0=0.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[1e12]])  # even alpha = 0.5**19 keeps |u0| far above max_control_value
    dV = jnp.asarray([-1.0, 0.0])
    J_prev = jnp.asarray(0.0)
    options = SolverOptions(max_control_value=1.0)
    reg = DynamicRegularization.initial(options)

    result = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)

    assert not bool(result.ls_failed)
    assert bool(jnp.isnan(result.J))
    assert int(result.status) == int(TerminationStatus.COST_INCREASE)
    assert float(result.alpha) == pytest.approx(options.line_search_decrease_factor**options.iterations_linesearch)
    np.testing.assert_allclose(float(result.regularization.rho), float(reg.rho))


def test_forward_pass_is_jittable() -> None:
    """forward_pass runs unchanged under jax.jit with SolverOptions held static."""
    model, nominal, obj = _lqr_setup(x0=10.0, N=2)
    K = jnp.zeros((1, 1, 1))
    d = jnp.asarray([[-5.0]])
    dV = jnp.asarray([-50.0, 25.0])
    J_prev = jnp.asarray(100.0)
    options = SolverOptions()
    reg = DynamicRegularization.initial(options)

    jitted = jax.jit(functools.partial(forward_pass, model, obj, options=options))
    eager = forward_pass(model, obj, nominal, K, d, dV, J_prev, reg, options)
    traced = jitted(nominal, K, d, dV, J_prev, reg)

    np.testing.assert_allclose(float(traced.alpha), float(eager.alpha))
    np.testing.assert_allclose(float(traced.J), float(eager.J))
