# ruff: noqa: RUF001 -- embedded Julia source uses Altro's own field names rho/drho (Greek in Julia)
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective, Objective
from trajopt.models import Cartpole, Pendulum
from trajopt.problem import retarget_to_goal
from trajopt.solvers.ilqr import (
    DynamicRegularization,
    backward_pass,
    forward_pass,
    rollout_closed_loop,
)
from trajopt.solvers.options import SolverOptions
from trajopt.trajectory import Trajectory

# Ticket 26: rollout_closed_loop/forward_pass against Altro.rollout!/Altro.forwardpass!.
# Both engines drive real Pendulum/Cartpole dynamics and a real LQR objective built to match
# altro_jl/problems/{pendulum,cartpole}.jl exactly (default RK4 integration, so no dependence on
# the RK3-integrated Problems.Cartpole benchmark). K, d, dV come from our own backward_pass,
# already cross-verified against Altro.backwardpass! in ticket 25, so this isolates the forward
# pass / rollout machinery itself rather than re-deriving a policy by hand.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

function trojopt_ticket26_setup(model, Q, R, Qf, x0, xf, N, dt, U0, opts)
    n = length(x0)
    m = length(U0[1])
    tf = dt * (N - 1)
    obj = TO.LQRObjective(Diagonal(Q), Diagonal(R), Diagonal(Qf), xf, N)
    prob = TO.Problem(model, obj, x0, tf; xf=xf, U0=[copy(u) for u in U0])
    TO.rollout!(prob)
    solver = Altro.iLQRSolver(prob, opts)
    return solver
end

function trojopt_ticket26_run_rollout(solver, K, d, alpha)
    N = solver.N
    for k in 1:N-1
        solver.K[k] .= K[k, :, :]
        solver.d[k] .= d[k, :]
    end
    ok = Altro.rollout!(solver, alpha)
    X = cat([Vector(RD.state(solver.Z̄[k])) for k = 1:N]..., dims=2)
    U = cat([Vector(RD.control(solver.Z̄[k])) for k = 1:N-1]..., dims=2)
    return X, U, ok, Int(solver.stats.status)
end

# Local copy of Altro.forwardpass!'s loop body (verbatim from altro_jl/src/ilqr/forwardpass.jl),
# extended to also return alpha/expected/z, which the public function discards.
function trojopt_ticket26_forwardpass!(solver, J_prev)
    Z = solver.Z; Z̄ = solver.Z̄
    ΔV = solver.ΔV
    ϕ = solver.opts.line_search_decrease_factor
    z_lb = solver.opts.line_search_lower_bound
    z_ub = solver.opts.line_search_upper_bound

    α = 1.0
    J = Inf
    z = Inf
    expected = Inf

    solver.stats.ls_failed = false
    max_iters = solver.opts.iterations_linesearch
    exit_linesearch = false
    for i = 1:max_iters
        isrolloutgood = Altro.rollout!(solver, α)

        if !isrolloutgood
            α *= ϕ
            continue
        end

        J = TO.cost(solver.obj, Z̄)
        expected = -α*(ΔV[1] + α*ΔV[2])

        if 0.0 < expected < solver.opts.expected_decrease_tolerance
            α = 0.0
            z = Inf
            copyto!(Z̄, Z)
            J = J_prev
            Altro.increaseregularization!(solver)
            exit_linesearch = true
        elseif expected > 0.0
            z = (J_prev - J) / expected
        else
            z = -1.0
        end

        if (z_lb ≤ z ≤ z_ub)
            exit_linesearch = true
            break
        end

        if i == max_iters
            α = 0.0
            copyto!(Z̄, Z)
            J = J_prev
            Altro.increaseregularization!(solver)
            solver.reg.ρ += solver.opts.bp_reg_fp
            solver.stats.ls_failed = true
            exit_linesearch = true
        end

        exit_linesearch && break
        α *= ϕ
    end

    if J > J_prev
        return NaN, α, expected, z, solver.stats.ls_failed
    end
    return J, α, expected, z, solver.stats.ls_failed
end

function trojopt_ticket26_run_forward_pass(solver, K, d, dV, rho, drho)
    N = solver.N
    for k in 1:N-1
        solver.K[k] .= K[k, :, :]
        solver.d[k] .= d[k, :]
    end
    solver.ΔV .= dV
    solver.reg.ρ = rho
    solver.reg.dρ = drho
    J_prev = TO.cost(solver.obj, solver.Z)
    J, alpha, expected, z, ls_failed = trojopt_ticket26_forwardpass!(solver, J_prev)
    X = cat([Vector(RD.state(solver.Z̄[k])) for k = 1:N]..., dims=2)
    U = cat([Vector(RD.control(solver.Z̄[k])) for k = 1:N-1]..., dims=2)
    return J, alpha, expected, z, ls_failed, X, U, solver.reg.ρ, solver.reg.dρ, J_prev
end
"""


class _BenchmarkSetup(NamedTuple):
    model: Pendulum | Cartpole
    obj: Objective
    nominal: Trajectory
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    x0: np.ndarray
    xf: np.ndarray


def _pendulum_setup() -> _BenchmarkSetup:
    """Ported verbatim from altro_jl/problems/pendulum.jl, but with default (RK4) integration."""
    n, m, N, tf = 2, 1, 51, 3.0
    dt = tf / (N - 1)
    Q = 1e-3 * np.ones(n) * dt
    R = 1e-3 * np.ones(m) * dt
    Qf = np.ones(n)
    x0 = np.zeros(n)
    xf = np.array([np.pi, 0.0])
    model = Pendulum()
    obj = retarget_to_goal(LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N), jnp.asarray(xf))
    U0 = np.full((N - 1, m), 0.1)
    nominal = model.rollout(
        Trajectory(
            X=jnp.zeros((N, n)),
            U=jnp.asarray(U0),
            t=jnp.linspace(0.0, tf, N),
            dt=jnp.full(N - 1, dt),
        ),
        x0=jnp.asarray(x0),
    )
    return _BenchmarkSetup(model=model, obj=obj, nominal=nominal, Q=Q, R=R, Qf=Qf, x0=x0, xf=xf)


def _cartpole_setup() -> _BenchmarkSetup:
    """Ported verbatim from altro_jl/problems/cartpole.jl, but with default (RK4) integration."""
    n, m, N, tf = 4, 1, 101, 5.0
    dt = tf / (N - 1)
    Qv, Rv, Qfv = 1e-2, 1e-1, 1e2
    Q = Qv * np.ones(n) * dt
    R = Rv * np.ones(m) * dt
    Qf = Qfv * np.ones(n)
    x0 = np.zeros(n)
    xf = np.array([0.0, np.pi, 0.0, 0.0])
    model = Cartpole()
    obj = retarget_to_goal(LQRObjective(Q=jnp.asarray(Q), R=jnp.asarray(R), Qf=jnp.asarray(Qf), N=N), jnp.asarray(xf))
    U0 = np.full((N - 1, m), 0.01)
    nominal = model.rollout(
        Trajectory(
            X=jnp.zeros((N, n)),
            U=jnp.asarray(U0),
            t=jnp.linspace(0.0, tf, N),
            dt=jnp.full(N - 1, dt),
        ),
        x0=jnp.asarray(x0),
    )
    return _BenchmarkSetup(model=model, obj=obj, nominal=nominal, Q=Q, R=R, Qf=Qf, x0=x0, xf=xf)


def _build_jl_solver(jl: Any, setup: _BenchmarkSetup, jl_model_expr: str, options: SolverOptions) -> Any:
    jl.seval(_ALTRO_SETUP)
    setup_fn = jl.seval("trojopt_ticket26_setup")
    jl_model = jl.seval(jl_model_expr)
    jl_opts = jl.Altro.SolverOptions(
        max_state_value=float(options.max_state_value),
        max_control_value=float(options.max_control_value),
        iterations_linesearch=int(options.iterations_linesearch),
        line_search_lower_bound=float(options.line_search_lower_bound),
        line_search_upper_bound=float(options.line_search_upper_bound),
        line_search_decrease_factor=float(options.line_search_decrease_factor),
        expected_decrease_tolerance=float(options.expected_decrease_tolerance),
        bp_reg_fp=float(options.bp_reg_fp),
        bp_reg_min=float(options.bp_reg_min),
        bp_reg_initial=float(options.bp_reg_initial),
        bp_reg_increase_factor=float(options.bp_reg_increase_factor),
    )
    U0 = np.asarray(setup.nominal.U)
    return setup_fn(
        jl_model,
        setup.Q,
        setup.R,
        setup.Qf,
        setup.x0,
        setup.xf,
        setup.nominal.N,
        float(setup.nominal.dt[0]),
        list(U0),
        jl_opts,
    )


def _lqr_policy(setup: _BenchmarkSetup, options: SolverOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run our own (Altro-cross-verified) backward pass to get a realistic, real-cost policy."""
    exp = setup.model.dynamics_expansion(setup.nominal) + setup.obj.cost_expansion(setup.nominal, setup.model)
    reg = DynamicRegularization.initial(options)
    bp = backward_pass(exp, reg, options)
    assert not bool(bp.failed)
    return np.asarray(bp.K), np.asarray(bp.d), np.asarray(bp.dV)


def _assert_rollout_matches_altro(
    jl: Any,
    setup: _BenchmarkSetup,
    jl_model_expr: str,
    *,
    alpha: float,
) -> None:
    options = SolverOptions()
    K, d, _dV = _lqr_policy(setup, options)
    solver = _build_jl_solver(jl, setup, jl_model_expr, options)

    result = rollout_closed_loop(setup.model, setup.nominal, jnp.asarray(K), jnp.asarray(d), alpha, options)

    run_rollout = jl.seval("trojopt_ticket26_run_rollout")
    X_jl, U_jl, ok_jl, status_jl = run_rollout(solver, K, d, alpha)
    X_jl = np.moveaxis(np.asarray(X_jl), -1, 0)
    U_jl = np.moveaxis(np.asarray(U_jl), -1, 0)

    np.testing.assert_allclose(np.asarray(result.X), X_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.U), U_jl, atol=1e-8)
    assert bool(result.failed) == (not bool(ok_jl))
    assert int(result.status) == int(status_jl)


def _assert_forward_pass_matches_altro(
    jl: Any,
    setup: _BenchmarkSetup,
    jl_model_expr: str,
    *,
    options: SolverOptions | None = None,
    K: np.ndarray | None = None,
    d: np.ndarray | None = None,
    dV: np.ndarray | None = None,
    check_trajectory: bool = True,
) -> None:
    opts = SolverOptions() if options is None else options
    if K is None or d is None or dV is None:
        K, d, dV = _lqr_policy(setup, opts)
    solver = _build_jl_solver(jl, setup, jl_model_expr, opts)

    reg = DynamicRegularization(rho=jnp.asarray(0.0), drho=jnp.asarray(0.0))
    J_prev = setup.obj.cost(setup.nominal)
    result = forward_pass(
        setup.model, setup.obj, setup.nominal, jnp.asarray(K), jnp.asarray(d), jnp.asarray(dV), J_prev, reg, opts
    )

    run_fp = jl.seval("trojopt_ticket26_run_forward_pass")
    J_jl, alpha_jl, expected_jl, z_jl, ls_failed_jl, X_jl, U_jl, rho_jl, drho_jl, J_prev_jl = run_fp(
        solver, K, d, np.asarray(dV), 0.0, 0.0
    )
    X_jl = np.moveaxis(np.asarray(X_jl), -1, 0)
    U_jl = np.moveaxis(np.asarray(U_jl), -1, 0)

    np.testing.assert_allclose(float(J_prev), float(J_prev_jl), atol=1e-8)
    if np.isnan(J_jl):
        assert bool(np.isnan(np.asarray(result.J)))
    else:
        np.testing.assert_allclose(np.asarray(result.J), J_jl, atol=1e-8)
    np.testing.assert_allclose(float(result.alpha), float(alpha_jl), atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.expected), expected_jl, atol=1e-8)
    if np.isinf(z_jl):
        assert bool(np.isinf(np.asarray(result.z)))
    else:
        np.testing.assert_allclose(np.asarray(result.z), z_jl, atol=1e-8)
    assert bool(result.ls_failed) == bool(ls_failed_jl)
    np.testing.assert_allclose(float(result.regularization.rho), float(rho_jl), atol=1e-8)
    np.testing.assert_allclose(float(result.regularization.drho), float(drho_jl), atol=1e-8)
    if check_trajectory:
        np.testing.assert_allclose(np.asarray(result.trajectory.X), X_jl, atol=1e-8)
        np.testing.assert_allclose(np.asarray(result.trajectory.U), U_jl, atol=1e-8)


def test_cross_rollout_pendulum_full_step(jl_altro: Any) -> None:
    _assert_rollout_matches_altro(jl_altro, _pendulum_setup(), "RobotZoo.Pendulum()", alpha=1.0)


def test_cross_rollout_pendulum_partial_step(jl_altro: Any) -> None:
    _assert_rollout_matches_altro(jl_altro, _pendulum_setup(), "RobotZoo.Pendulum()", alpha=0.3)


def test_cross_rollout_cartpole_full_step(jl_altro: Any) -> None:
    _assert_rollout_matches_altro(jl_altro, _cartpole_setup(), "RobotZoo.Cartpole()", alpha=1.0)


def test_cross_forward_pass_accepts_pendulum(jl_altro: Any) -> None:
    """A real LQR backward-pass policy should be accepted, matching Altro's chosen alpha/J/z."""
    _assert_forward_pass_matches_altro(jl_altro, _pendulum_setup(), "RobotZoo.Pendulum()")


def test_cross_forward_pass_accepts_cartpole(jl_altro: Any) -> None:
    _assert_forward_pass_matches_altro(jl_altro, _cartpole_setup(), "RobotZoo.Cartpole()")


def test_cross_forward_pass_line_search_exhaustion(jl_altro: Any) -> None:
    """K = d = 0 keeps every rollout at the nominal, so expected decrease is always 0 (z = -1
    forever): the search runs every iteration without ever accepting, matching Altro's
    iteration-exhaustion exit (regularization bumped twice, ls_failed set).
    """
    setup = _pendulum_setup()
    K = np.zeros((setup.nominal.N - 1, 1, 2))
    d = np.zeros((setup.nominal.N - 1, 1))
    dV = np.zeros(2)
    _assert_forward_pass_matches_altro(jl_altro, setup, "RobotZoo.Pendulum()", K=K, d=d, dV=dV)


def test_cross_forward_pass_guard_exhaustion_cost_increase(jl_altro: Any) -> None:
    """An unreachably tight max_state_value trips the rollout guard at every alpha, so J never
    leaves Inf (finding J): Altro's forwardpass! exits the loop normally with ls_failed still
    False and reports COST_INCREASE / NaN, which this checks on both engines.

    The trajectory field itself is not compared here: with every rollout attempt failing, our
    `should_exit` never fires so `forward_pass` leaves it at `nominal`, while Altro's `Z̄` buffer
    keeps the garbage from the last (also-failing) rollout attempt at a near-zero alpha -- neither
    side's value is meaningful on this path (see `forward_pass`'s docstring), so they are only
    guaranteed to be close, not equal, and are excluded from this otherwise 1e-8 comparison.
    """
    setup = _pendulum_setup()
    options = SolverOptions(max_state_value=1e-6)
    _assert_forward_pass_matches_altro(jl_altro, setup, "RobotZoo.Pendulum()", options=options, check_trajectory=False)
