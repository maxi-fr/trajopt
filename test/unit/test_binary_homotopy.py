import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.models.affine import AffineModel
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.layout import compute_constraint_violation, constraint_bounds
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.sparsity import jacobian_sparsity_pattern
from trajopt.transcription.transcription import eval_g, eval_jac_g


def _problem(*, budget: float | None = None) -> Problem:
    """Build a small binary OCP with a relaxed fractional optimum."""
    N = 3
    model = AffineModel(A=jnp.ones((1, 1)), B=jnp.zeros((1, 1)))
    stage = DiagonalCost(Q=jnp.zeros(1), R=jnp.ones(1), r=jnp.array([-0.8]))
    cl = ConstraintList(n=1, m=1, N=N)
    if budget is not None:
        A = jnp.array([[0.0, 1.0, 0.0, 1.0, 0.0]])
        cl.add_horizon_constraint(LinearHorizonConstraint(A, jnp.array([budget]), sense=NegativeOrthant()))
    return Problem(model, Objective(stage, N=N), cl, binary_control_indices=(0,))


def test_binary_bounds_and_horizon_rows() -> None:
    """Check binary limits and the trailing Horizon residual and Jacobian."""
    problem = _problem(budget=1.0)
    _, _, lo, hi = problem.constraints.primal_bounds()
    np.testing.assert_array_equal(lo, 0.0)
    np.testing.assert_array_equal(hi, 1.0)
    z = jnp.array([0.0, 0.8, 0.0, 0.7, 0.0])
    assert float(eval_g(problem, z, jnp.zeros(1))[-1]) == pytest.approx(0.5)
    assert len(eval_jac_g(problem, z, jnp.zeros(1))) == 12
    np.testing.assert_array_equal(np.asarray(eval_jac_g(problem, z, jnp.zeros(1)))[-5:], [0, 1, 0, 1, 0])
    rows, cols = jacobian_sparsity_pattern(3, 1, 1, problem.constraints.p, ((0, 1, 2, 3, 4),), (1,))
    np.testing.assert_array_equal(rows[-5:], 3)
    np.testing.assert_array_equal(cols[-5:], np.arange(5))
    assert len(constraint_bounds(problem)[0]) == 4
    assert compute_constraint_violation(problem, z, jnp.zeros(1)) == pytest.approx(0.5)


def test_indexed_horizon_equality() -> None:
    """Evaluate a selected Primal Vector equality with its chosen columns."""
    cl = ConstraintList(n=1, m=1, N=3)
    cl.add_horizon_constraint(
        LinearHorizonConstraint(jnp.array([[1.0, 1.0]]), jnp.array([1.0]), sense=ZeroCone(), inds=(1, 3))
    )
    base = _problem()
    problem = Problem(base.model, base.obj, cl)
    z = jnp.array([0.0, 0.2, 0.0, 0.7, 0.0])
    assert float(eval_g(problem, z, jnp.zeros(1))[-1]) == pytest.approx(-0.1)
    assert compute_constraint_violation(problem, z, jnp.zeros(1)) == pytest.approx(0.1)


def test_invalid_binary_declaration() -> None:
    """Reject repeated or out-of-range binary coordinates at construction."""
    problem = _problem()
    with pytest.raises(ValueError, match="Binary control indices"):
        Problem(problem.model, problem.obj, binary_control_indices=(0, 0))
    with pytest.raises(ValueError, match="Binary control indices"):
        Problem(problem.model, problem.obj, binary_control_indices=(1,))
    cl = ConstraintList(n=1, m=1, N=3)
    cl.add_constraint(ControlBound(m=1, n=1, u_min=jnp.array([1.1])), range(2))
    with pytest.raises(ValueError, match="contradict"):
        Problem(problem.model, problem.obj, cl, binary_control_indices=(0,))


def test_mpc_binary_homotopy_and_budget() -> None:
    """Solve a binary OCP through MPC and enforce the Horizon budget."""
    pytest.importorskip("cyipopt")
    solver = Ipopt(options={"print_level": 0}, beta_initial=1.0)
    unconstrained = MPC(_problem(), solver, x0=jnp.zeros(1)).solve()
    limited = MPC(_problem(budget=1.0), solver, x0=jnp.zeros(1)).solve()
    assert unconstrained.success
    assert limited.success
    np.testing.assert_allclose(np.asarray(unconstrained.trajectory.U), 1.0, atol=1e-4)
    assert float(jnp.sum(limited.trajectory.U)) <= 1.0 + 1e-4
    assert limited.info["binary_distance"] <= solver.binary_tolerance
    assert limited.cost == pytest.approx(float(_problem(budget=1.0).obj.cost(limited.trajectory)))


def test_pass_limit_keeps_fractional_primal() -> None:
    """Report failure without rounding when continuation ends before integrality."""
    pytest.importorskip("cyipopt")
    result = MPC(
        _problem(), Ipopt(options={"print_level": 0}, beta_initial=0.01, max_passes=2), x0=jnp.zeros(1)
    ).solve()
    assert not result.success
    assert result.info["binary_distance"] > 1e-4
    np.testing.assert_allclose(np.asarray(result.trajectory.U), 0.8, atol=0.05)


def test_reference_keeps_linear_control_cost_and_shift() -> None:
    """Keep the independent linear cost while moving the MPC reference and warm start."""
    problem = _problem()
    bc = BoundaryConditions(
        x0=jnp.zeros(1),
        t0=jnp.array(0.0),
        X_ref=jnp.ones((3, 1)),
        U_ref=jnp.zeros((2, 1)),
    )
    np.testing.assert_allclose(retarget_problem(problem, bc).obj.r, -0.8)
    pytest.importorskip("cyipopt")
    mpc = MPC(problem, Ipopt(options={"print_level": 0}), x0=jnp.zeros(1))
    first = mpc.solve()
    mpc.shift()
    second = mpc.solve()
    assert first.success
    assert second.success
    assert mpc.warm_start.Z.shape == first.Z.shape


def test_unsupported_solver_rejects_binary_problem() -> None:
    """Reject binary declarations before an unsupported Backend solves a relaxation."""
    mpc = MPC(_problem(), OSQP(), x0=jnp.zeros(1))
    with pytest.raises(ValueError, match="does not support"):
        mpc.solve()
