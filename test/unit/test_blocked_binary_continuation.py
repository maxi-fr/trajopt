from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.models.affine import AffineModel
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.transcription.homotopy import solve_continuation_passes
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting


def _make_binary_problem(
    *,
    N: int = 5,
    b: int = 1,
    budget: float | None = None,
    r_val: float = -0.8,
) -> Problem:
    """Build a test binary Problem with optional blocking and Horizon budget."""
    n = 1
    m = 1
    H = N - 1
    model = AffineModel(A=jnp.ones((1, 1)), B=jnp.zeros((1, 1)))
    stage = DiagonalCost(Q=jnp.zeros(1), R=jnp.ones(1), r=jnp.array([r_val]))
    cl = ConstraintList(n=n, m=m, N=N)
    if budget is not None:
        u_inds = [j * (n + m) + n for j in range(H)]
        cl.add_horizon_constraint(
            LinearHorizonConstraint(
                A=jnp.ones((1, H)),
                b=jnp.array([budget]),
                sense=NegativeOrthant(),
                inds=u_inds,
            )
        )
    terminal = DiagonalCost(Q=jnp.zeros(n), terminal=True, m=m)
    return Problem(
        model,
        Objective(stage, terminal_cost=terminal, N=N),
        cl,
        control_block_steps=b,
        binary_control_indices=(0,),
    )


def _make_blocked_budget_problem(*, budget: float) -> Problem:
    """Build a deterministic blocked binary Problem with native Horizon activation budget."""
    N = 13
    H = 12
    b = 6
    n = 1
    m = 1
    model = AffineModel(A=jnp.ones((1, 1)), B=jnp.zeros((1, 1)))

    r_stages = jnp.concatenate([jnp.full((6, 1), -0.9), jnp.full((6, 1), -0.6)], axis=0)
    stage = DiagonalCost(
        Q=jnp.zeros((H, n)),
        R=jnp.ones((H, m)),
        r=r_stages,
    )

    cl = ConstraintList(n=n, m=m, N=N)
    u_inds = [j * (n + m) + n for j in range(H)]
    cl.add_horizon_constraint(
        LinearHorizonConstraint(
            A=jnp.ones((1, H)),
            b=jnp.array([budget]),
            sense=NegativeOrthant(),
            inds=u_inds,
        )
    )

    terminal = DiagonalCost(Q=jnp.zeros(n), terminal=True, m=m)
    return Problem(
        model,
        Objective(stage, terminal_cost=terminal, N=N),
        cl,
        control_block_steps=b,
        binary_control_indices=(0,),
    )


def test_homotopy_parameter_validation_and_forwarding() -> None:
    """SingleShooting reads continuation settings from its solver and validates identically to Ipopt."""
    pytest.importorskip("cyipopt")

    solver = Ipopt(beta_initial=2.0, beta_growth=5.0, max_passes=6, binary_tolerance=1e-5)
    ss = SingleShooting(solver)

    assert ss.beta_initial == 2.0
    assert ss.beta_growth == 5.0
    assert ss.max_passes == 6
    assert ss.binary_tolerance == 1e-5

    prob_bin = _make_binary_problem(N=3, b=1)
    bc = BoundaryConditions(x0=jnp.zeros(1), t0=jnp.array(0.0))

    ws = WarmStart.cold(prob_bin, jnp.zeros(1))

    bad_solvers = [
        Ipopt(beta_initial=0.0),
        Ipopt(beta_initial=-1.0),
        Ipopt(beta_growth=1.0),
        Ipopt(beta_growth=0.5),
        Ipopt(max_passes=1),
        Ipopt(max_passes=0),
        Ipopt(binary_tolerance=0.0),
        Ipopt(binary_tolerance=-1e-4),
    ]

    expected_msg = "Binary homotopy needs positive beta/tolerance, growth > 1, and at least two passes."
    for bad_ipopt in bad_solvers:
        bad_ss = SingleShooting(bad_ipopt)

        with pytest.raises(ValueError, match=expected_msg):
            Program(prob_bin, bad_ipopt)
        with pytest.raises(ValueError, match=expected_msg):
            Program(prob_bin, bad_ss)

        with pytest.raises(ValueError, match=expected_msg):
            bad_ipopt.solve(Program(prob_bin, solver), bc, ws)
        with pytest.raises(ValueError, match=expected_msg):
            bad_ss.solve(Program(prob_bin, solver), bc, ws)

    # Continuous problem without binary controls does not raise
    prob_cont = Problem(prob_bin.model, prob_bin.obj, prob_bin.constraints)
    prog_cont_ipopt = Program(prob_cont, Ipopt(beta_initial=-1.0))
    prog_cont_ss = Program(prob_cont, SingleShooting(Ipopt(beta_initial=-1.0)))
    assert prog_cont_ipopt is not None
    assert prog_cont_ss is not None


def test_single_shooting_penalty_finite_differences_unblocked_and_blocked() -> None:
    """SingleShooting reduced gradient and Hessian diagonal match finite differences for b = 1 and b > 1."""
    pytest.importorskip("cyipopt")

    for b in (1, 3):
        N = 7  # H = 6 native transitions, B = 6 // b blocks
        prob = _make_binary_problem(N=N, b=b)
        n_u = (N - 1) // b

        ss = SingleShooting(Ipopt(), hessian="dense")
        cb = ss.transcription_callback(prob, jnp.zeros(1), jnp.array(0.0), jnp.array(0.1))

        beta = 3.5
        cb.beta = beta
        obj_factor = 0.8

        u_frac = np.linspace(0.2, 0.8, n_u, dtype=np.float64)
        grad = cb.gradient(u_frac)
        hess_vals = cb.hessian(u_frac, np.zeros(0, dtype=np.float64), obj_factor=obj_factor)

        # Gradient finite differences
        eps = 1e-6
        fd_grad = np.zeros(n_u, dtype=np.float64)
        for i in range(n_u):
            u_plus = u_frac.copy()
            u_plus[i] += eps
            u_minus = u_frac.copy()
            u_minus[i] -= eps
            fd_grad[i] = (cb.objective(u_plus) - cb.objective(u_minus)) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, atol=1e-5)

        # Analytical penalty gradient contribution
        cb.beta = 0.0
        grad_unpenalized = cb.gradient(u_frac)
        cb.beta = beta
        expected_grad_pen = b * beta * (1 - 2 * u_frac)
        np.testing.assert_allclose(grad - grad_unpenalized, expected_grad_pen, atol=1e-8)

        # Hessian diagonal finite differences
        fd_diag = np.zeros(n_u, dtype=np.float64)
        for i in range(n_u):
            u_plus = u_frac.copy()
            u_plus[i] += eps
            u_minus = u_frac.copy()
            u_minus[i] -= eps
            fd_diag[i] = obj_factor * (cb.gradient(u_plus)[i] - cb.gradient(u_minus)[i]) / (2 * eps)

        # Extract diagonal from lower-triangular packed Hessian
        diag_mask = cb.hess_rows == cb.hess_cols
        hess_diag = hess_vals[diag_mask]
        np.testing.assert_allclose(hess_diag, fd_diag, atol=1e-4)

        # Analytical Hessian penalty contribution
        cb.beta = 0.0
        hess_unpenalized = cb.hessian(u_frac, np.zeros(0, dtype=np.float64), obj_factor=obj_factor)
        cb.beta = beta
        expected_hess_pen = -2 * b * beta * obj_factor
        np.testing.assert_allclose(hess_diag - hess_unpenalized[diag_mask], expected_hess_pen, atol=1e-8)


def test_continuation_schedule_dual_carry_and_abort_on_failure() -> None:
    """Continuation carries dual iterates, follows schedule, and aborts immediately on failed pass."""
    pytest.importorskip("cyipopt")

    class MockNLP:
        def __init__(self, fail_on_pass: int | None = None) -> None:
            self.options: dict[str, Any] = {}
            self.pass_calls: list[dict[str, Any]] = []
            self.fail_on_pass = fail_on_pass

        def add_option(self, key: str, value: Any) -> None:
            self.options[key] = value

        def solve(self, z0: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            call_idx = len(self.pass_calls)
            self.pass_calls.append({"z0": z0.copy(), "kwargs": dict(kwargs), "options": dict(self.options)})
            status = -1 if self.fail_on_pass is not None and call_idx == self.fail_on_pass else 0
            # Return fractional iterate for pass 0, integer for pass 1+
            z_ret = np.array([0.5]) if call_idx == 0 else np.array([1.0])
            info = {
                "status": status,
                "status_msg": "OK" if status == 0 else "Failed",
                "mult_g": np.array([0.2 * (call_idx + 1)]),
                "mult_x_L": np.array([0.0]),
                "mult_x_U": np.array([0.4 * (call_idx + 1)]),
            }
            return z_ret, info

    class MockCB:
        def __init__(self) -> None:
            self.beta = 0.0

    # Test 1: Immediate abort on pass 0 failure
    cb_fail0 = MockCB()
    nlp_fail0 = MockNLP(fail_on_pass=0)
    _z_opt, info, pass_count, _dist = solve_continuation_passes(
        nlp_fail0,
        cb_fail0,
        np.array([0.5]),
        None,
        None,
        binary_indices=np.array([0]),
        beta_initial=2.0,
        beta_growth=3.0,
        max_passes=5,
    )
    assert pass_count == 1
    assert len(nlp_fail0.pass_calls) == 1
    assert cb_fail0.beta == 0.0
    assert info["status"] == -1

    # Test 2: Pass 0 succeeds, pass 1 succeeds and terminates (binary_distance <= tolerance)
    cb_succ = MockCB()
    nlp_succ = MockNLP()
    _z_opt, _info, pass_count, dist = solve_continuation_passes(
        nlp_succ,
        cb_succ,
        np.array([0.5]),
        None,
        None,
        binary_indices=np.array([0]),
        beta_initial=2.0,
        beta_growth=3.0,
        max_passes=5,
        binary_tolerance=1e-4,
    )
    assert pass_count == 2
    assert len(nlp_succ.pass_calls) == 2
    # Pass 0 beta was 0.0, pass 1 beta was 2.0
    assert cb_succ.beta == 2.0
    assert dist == 0.0
    # Check that pass 1 received dual warm start from pass 0
    p1_kwargs = nlp_succ.pass_calls[1]["kwargs"]
    assert "lagrange" in p1_kwargs
    np.testing.assert_allclose(p1_kwargs["lagrange"], [0.2])
    assert "zu" in p1_kwargs
    np.testing.assert_allclose(p1_kwargs["zu"], [0.4])

    # Test 3: Pass 0 succeeds, pass 1 fails -> immediate abort
    cb_fail1 = MockCB()
    nlp_fail1 = MockNLP(fail_on_pass=1)
    _z_opt, info, pass_count, _dist = solve_continuation_passes(
        nlp_fail1,
        cb_fail1,
        np.array([0.5]),
        None,
        None,
        binary_indices=np.array([0]),
        beta_initial=2.0,
        beta_growth=3.0,
        max_passes=5,
    )
    assert pass_count == 2
    assert len(nlp_fail1.pass_calls) == 2
    assert info["status"] == -1


def test_failure_reporting_and_no_rounding() -> None:
    """Reaching max_passes returns success=False without rounding primal, preserving diagnostics."""
    pytest.importorskip("cyipopt")

    prob = _make_binary_problem(N=3, b=1)
    # Low beta and max_passes=2 so continuation cannot force integrality
    opts = {"print_level": 0}
    solver = Ipopt(options=opts, beta_initial=1e-4, beta_growth=1.1, max_passes=2, binary_tolerance=1e-5)

    res_ipopt = MPC(prob, solver, x0=jnp.zeros(1)).solve()
    assert not res_ipopt.success
    assert res_ipopt.info["homotopy_passes"] == 2
    assert res_ipopt.info["binary_distance"] > 1e-4
    # Primal is NOT rounded: fractional controls preserved
    u_ipopt = np.asarray(res_ipopt.trajectory.U)
    assert np.all((u_ipopt > 0.05) & (u_ipopt < 0.95))
    assert res_ipopt.cost == pytest.approx(float(prob.obj.cost(res_ipopt.trajectory)))

    res_ss = MPC(prob, SingleShooting(solver, hessian="dense"), x0=jnp.zeros(1)).solve()
    assert not res_ss.success
    assert res_ss.info["homotopy_passes"] == 2
    assert res_ss.info["binary_distance"] > 1e-4
    u_ss = np.asarray(res_ss.trajectory.U)
    assert np.all((u_ss > 0.05) & (u_ss < 0.95))
    assert res_ss.cost == pytest.approx(float(prob.obj.cost(res_ss.trajectory)))


def test_blocked_binary_activation_budget_matches_oracle() -> None:
    """Blocked binary Problem with Horizon budget matches enumerated oracle in both formulations."""
    pytest.importorskip("cyipopt")

    opts = {"print_level": 0, "tol": 1e-6, "constr_viol_tol": 1e-5}
    solver_ipopt = Ipopt(options=opts, beta_initial=0.5, beta_growth=5.0, max_passes=6)
    solver_ss_dense = SingleShooting(solver_ipopt, hessian="dense")
    solver_ss_lbfgs = SingleShooting(solver_ipopt, hessian="lbfgs")

    x0 = jnp.zeros(1)

    # Budget cases:
    # 1. budget = 6.0: Allows 1 active block. Block 0 is cheaper (-2.4 vs -0.6). Oracle: block 0 active.
    # 2. budget = 5.0 (< 6): Cannot activate any block. Oracle: neither block active.
    # 3. budget = 8.0 (non-multiple): Allows at most 1 active block. Oracle: block 0 active.
    cases = [
        (6.0, np.array([1.0, 0.0])),
        (2.0, np.array([0.0, 0.0])),
        (8.0, np.array([1.0, 0.0])),
    ]

    for budget, expected_blocks in cases:
        prob = _make_blocked_budget_problem(budget=budget)

        for name, solver in [
            ("Ipopt", solver_ipopt),
            ("SingleShooting(dense)", solver_ss_dense),
            ("SingleShooting(lbfgs)", solver_ss_lbfgs),
        ]:
            res = MPC(prob, solver, x0=x0).solve()
            assert res.success, f"{name} failed on budget={budget}"
            u_native = np.asarray(res.trajectory.U).ravel()
            # Verify block constancy: block 0 (indices 0..5), block 1 (indices 6..11)
            np.testing.assert_allclose(u_native[:6], expected_blocks[0], atol=1e-3, err_msg=f"{name} blk 0")
            np.testing.assert_allclose(u_native[6:], expected_blocks[1], atol=1e-3, err_msg=f"{name} blk 1")
            assert np.sum(u_native) <= budget + 1e-4
            assert res.info["binary_distance"] <= solver_ipopt.binary_tolerance
