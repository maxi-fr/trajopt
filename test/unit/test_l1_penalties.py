from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.constraints.output import OutputConstraint
from trajopt.costs import GenericCost, LQRObjective, Objective, QuadraticCost
from trajopt.costs.l1 import L1ControlCost
from trajopt.costs.pseudo_huber import PseudoHuberControlCost
from trajopt.models.affine import AffineModel
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.program import Program
from trajopt.trajectory import Trajectory
from trajopt.transcription import penalties
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting


@pytest.mark.filterwarnings("ignore:.*single convex subproblem:UserWarning")
@pytest.mark.parametrize(("backend", "block_steps"), [("osqp", 1), ("ipopt", 2), ("single_shooting", 2)])
def test_l1_cost_mpc_matches_exact_optimum(backend: str, block_steps: int) -> None:
    """Check the exact L1 optimum and physical layout with and without Control Blocking."""
    pytest.importorskip("osqp" if backend == "osqp" else "cyipopt", exc_type=ImportError)
    osqp = OSQP(options={"eps_abs": 1e-9, "eps_rel": 1e-9})
    ipopt = Ipopt(options={"tol": 1e-10})
    solver = {"osqp": osqp, "ipopt": ipopt, "single_shooting": SingleShooting(ipopt)}[backend]
    model = AffineModel(A=jnp.eye(2), B=jnp.eye(2))
    base = LQRObjective(jnp.ones(2), 0.1 * jnp.ones(2), jnp.ones(2), N=5)
    weight = 0.5
    l1 = L1ControlCost(n=2, m=2, weight=weight)
    objective = Objective(base.stage_cost + l1, base.terminal_cost, N=base.N)
    problem = Problem(model, objective, control_block_steps=block_steps)
    reference = Trajectory(
        X=jnp.linspace(jnp.zeros(2), jnp.array([4.0, -4.0]), base.N),
        U=jnp.zeros((base.N - 1, 2)),
        t=jnp.arange(base.N) * 0.1,
        dt=jnp.full(base.N - 1, 0.1),
    )
    result = MPC(problem, solver, x0=[0.0, 0.0], reference=reference, penalty_diagnostics=True).solve()

    assert result.success, result.message
    # Solve the stationarity equations on the positive branch of the first control.
    controls = np.repeat(np.eye(4 // block_steps), block_steps, axis=0)
    states = np.cumsum(controls, axis=0)
    target = np.arange(1.0, 5.0)
    block_u = np.linalg.solve(states.T @ states + 0.1 * controls.T @ controls, states.T @ target - weight * block_steps)
    u = controls @ block_u
    assert np.all(u > 0)
    expected_u = np.column_stack([u, -u])
    expected_x = np.vstack([np.zeros(2), np.cumsum(expected_u, axis=0)])
    np.testing.assert_allclose(result.trajectory.X, expected_x, atol=1e-6)
    np.testing.assert_allclose(result.trajectory.U, expected_u, atol=1e-6)
    np.testing.assert_allclose(result.info["penalties"][0].stage_slack, np.abs(expected_u), atol=1e-6)
    expected_cost = np.sum((np.cumsum(u) - target) ** 2 + 0.1 * u**2 + 2 * weight * u)
    assert result.cost == pytest.approx(expected_cost, abs=1e-6)


@pytest.mark.parametrize(
    ("slack", "expected_violation", "physical_checks"), [(0.75, 0.25, 0), (0.125, 0.625, 0), (np.nan, 0.25, 1)]
)
def test_blocked_l1_reuses_transformed_feasibility(
    monkeypatch, slack: float, expected_violation: float, physical_checks: int
) -> None:
    """Reuse finite feasibility metrics and recheck the physical Trajectory if slacks make them nonfinite."""
    pytest.importorskip("cyipopt", exc_type=ImportError)
    model = AffineModel(A=jnp.eye(1), B=jnp.zeros((1, 1)))
    objective = Objective(stage_cost=L1ControlCost(n=1, m=1, weight=2.0), N=3)
    constraints = ConstraintList(n=1, m=1, N=3)
    constraints.add_constraint(ControlBound(m=1, u_min=[-0.5], u_max=[0.5], n=1), range(2))
    problem = Problem(model, objective, constraints, control_block_steps=2)

    def solve_passes(*args, **kwargs):
        """Supply one deliberately infeasible block of physical control and slack."""
        return np.array([0.75, slack]), {"status": -1}, 1, 0.0

    check_physical = Mock(wraps=penalties.compute_constraint_violation)
    monkeypatch.setattr(SingleShooting, "_solve_passes", solve_passes)
    monkeypatch.setattr(penalties, "compute_constraint_violation", check_physical)
    result = MPC(problem, SingleShooting(Ipopt()), x0=[0.0], penalty_diagnostics=True).solve()

    assert check_physical.call_count == physical_checks
    assert result.constraint_violation == pytest.approx(expected_violation)
    np.testing.assert_array_equal(result.trajectory.U, [[0.75], [0.75]])
    np.testing.assert_array_equal(result.info["penalties"][0].stage_slack, [[slack], [slack]])
    assert result.cost == pytest.approx(3.0)


def test_projected_cost_reuses_one_trace_with_moving_reference_and_clock(monkeypatch) -> None:
    """Project the exact physical cost using each solve's reference and timestamps."""
    pytest.importorskip("cyipopt", exc_type=ImportError)
    model = AffineModel(A=jnp.eye(1), B=jnp.ones((1, 1)))
    base = LQRObjective(jnp.ones(1), jnp.ones(1), jnp.ones(1), N=3)
    stage = base.stage_cost + L1ControlCost(n=1, m=1, weight=0.2) + GenericCost(lambda _x, _u, t: t**2, n=1, m=1)
    objective = Objective(stage, base.terminal_cost, N=3)
    problem = Problem(model, objective, control_block_steps=2)
    mpc = MPC(problem, SingleShooting(Ipopt(options={"print_level": 0})), x0=[0.0])
    evaluate = Objective.cost
    physical_evaluations = 0

    def count_physical_evaluation(self, traj):
        nonlocal physical_evaluations
        if type(self) is Objective:
            physical_evaluations += 1
        return evaluate(self, traj)

    monkeypatch.setattr(Objective, "cost", count_physical_evaluation)
    for i in range(3):
        mpc.measure([0.0], t=float(i))
        mpc.set_reference(
            Trajectory(
                X=jnp.full((3, 1), float(i)),
                U=jnp.zeros((2, 1)),
                t=mpc.t0 + jnp.arange(3) * 0.05,
                dt=jnp.full(2, 0.05),
            )
        )
        result = mpc.solve()
        assert result.success, result.message
        expected = evaluate(mpc.bc.retarget(objective), result.trajectory)
        assert result.cost == pytest.approx(float(expected), rel=1e-12)

    assert physical_evaluations == 1


def test_pseudo_huber_control_cost_is_smooth_l1_stage_cost() -> None:
    cost = PseudoHuberControlCost(n=2, m=2, weight=3.0, delta=0.5)
    x = jnp.array([7.0, -4.0])
    u = jnp.array([0.0, 1.5])

    np.testing.assert_allclose(cost.evaluate(x, u), 3.0 * (np.sqrt(2.5) - 0.5))
    np.testing.assert_allclose(cost.gradient(x, u), [0.0, 0.0, 0.0, 3.0 * 1.5 / np.sqrt(2.5)])
    np.testing.assert_allclose(cost.hessian(x, u)[3, 3], 3.0 * 0.25 / 2.5**1.5)
    assert cost.evaluate(x, jnp.zeros(2)) == 0.0

    obj = Objective(stage_cost=cost, N=3)
    assert obj.terminal_cost.evaluate(x) == 0.0


@pytest.mark.parametrize(("weight", "delta"), [(-1.0, 0.1), (1.0, 0.0)])
def test_pseudo_huber_control_cost_rejects_invalid_parameters(weight: float, delta: float) -> None:
    with pytest.raises(ValueError, match="weight must be finite"):
        PseudoHuberControlCost(n=1, m=1, weight=weight, delta=delta)


def _lowered_problem(problem: Problem, weight: float) -> Problem:
    """Prepare the L1 Objective and return the Problem consumed by the Backend."""
    obj = problem.obj
    stage = obj.stage_cost + L1ControlCost(n=obj.n, m=obj.m, weight=weight)
    physical = Problem(
        problem.model,
        Objective(stage, obj.terminal_cost, N=obj.N),
        problem.constraints,
        dt=problem.dt,
        binary_control_indices=problem.binary_control_indices,
    )
    program = Program(physical, Ipopt())
    transformed = program._transformed  # noqa: SLF001 -- inspect the solver layout in structural assertions
    assert transformed is not None
    return transformed.problem


def test_l1_epigraph_adds_exact_control_penalty_without_changing_dynamics() -> None:
    """Check the slack price and inequalities alongside the physical dynamics."""
    model = AffineModel(A=jnp.eye(1), B=jnp.array([[2.0, -1.0]]))
    problem = Problem(model, LQRObjective(jnp.array([1.0]), jnp.array([0.5, 0.25]), jnp.array([2.0]), N=3))
    augmented = _lowered_problem(problem, 4.0)
    x = jnp.array([0.7])
    u = jnp.array([-1.5, 0.5])
    s = jnp.array([1.5, 0.5])

    assert problem.model.m == 2
    assert augmented.model.m == 4
    assert augmented.obj.m == 4
    np.testing.assert_allclose(augmented.model.discrete_dynamics(x, jnp.concatenate([u, s]), 0.0, 0.1), [0.7 - 3.5])
    np.testing.assert_allclose(
        augmented.obj[0].evaluate(x, jnp.concatenate([u, s])),
        problem.obj[0].evaluate(x, u) + L1ControlCost(n=1, m=2, weight=4.0).evaluate(x, u),
    )
    residual = augmented.constraints.evaluate_knot(0, x, jnp.concatenate([u, s]))[-4:]
    assert bool(jnp.all(residual <= 0.0))
    assert bool(jnp.all(jnp.max(jnp.stack([residual[:2], residual[2:]]), axis=0) == 0.0))


def test_l1_epigraph_preserves_constraints_bounds_and_horizon_rows() -> None:
    """Check the remapping of original rows, bounds, and output maps."""
    model = AffineModel(A=jnp.eye(1), B=jnp.ones((1, 1)), C=jnp.ones((1, 1)), D=2 * jnp.ones((1, 1)))
    constraints = ConstraintList(n=1, m=1, N=3)
    constraints.add_constraint(StateBound(n=1, x_min=[-2.0], x_max=[2.0], m=1), range(3))
    constraints.add_constraint(ControlBound(m=1, u_min=[-0.5], u_max=[0.8], n=1), range(2))
    constraints.add_constraint(LinearConstraint(n=1, m=1, A=jnp.array([[0.0, 1.0]]), b=jnp.array([0.7])), range(2))
    constraints.add_constraint(
        OutputConstraint(model, LinearConstraint(n=1, m=0, A=jnp.array([[1.0]]), b=jnp.array([1.0]), inds=[0])),
        range(2),
    )
    constraints.add_constraint(GoalConstraint(n=1, xf=[0.0], m=1), 2)
    constraints.add_horizon_constraint(LinearHorizonConstraint(jnp.array([[1.0, 1.0]]), jnp.array([1.0]), inds=[1, 3]))
    constraints.add_horizon_constraint(LinearHorizonConstraint(jnp.ones((1, 5)), jnp.array([2.0])))
    problem = Problem(
        model, LQRObjective(jnp.ones(1), jnp.ones(1), jnp.ones(1), 3), constraints, binary_control_indices=[0]
    )
    augmented = _lowered_problem(problem, 2.0)
    X = jnp.array([[0.0], [0.3], [0.7]])
    U = jnp.array([[0.3], [0.4]])
    S = jnp.array([[0.3], [0.4]])
    old = Trajectory(X=X, U=U, t=jnp.array([0.0, 0.1, 0.2]), dt=jnp.array([0.1, 0.1]))
    lifted = Trajectory(X=X, U=jnp.concatenate([U, S], axis=1), t=old.t, dt=old.dt)

    assert augmented.binary_control_indices == (0,)
    assert augmented.constraints.has_goal_constraint()
    np.testing.assert_allclose(augmented.model.output(X[0], lifted.U[0]), problem.model.output(X[0], U[0]))
    np.testing.assert_allclose(augmented.model.output_control_jacobian(X[0], lifted.U[0]), [[2.0, 0.0]])
    for k in range(2):
        original_rows = problem.constraints.evaluate_knot(k, X[k], U[k])
        augmented_rows = augmented.constraints.evaluate_knot(k, X[k], lifted.U[k])
        np.testing.assert_allclose(augmented_rows[: len(original_rows)], original_rows)
    np.testing.assert_allclose(augmented.constraints.evaluate_knot(2, X[2], xf=jnp.array([0.7])), [0.0])
    np.testing.assert_allclose(
        augmented.constraints.horizon_constraints[0].evaluate(jnp.array([0, 0.3, 0.3, 0.3, 0.4, 0.4, 0.7])), -0.3
    )
    np.testing.assert_allclose(
        augmented.constraints.horizon_constraints[1].evaluate(jnp.array([0, 0.3, 0.3, 0.3, 0.4, 0.4, 0.7])),
        -0.3,
    )
    xL, xU, uL, uU = augmented.constraints.primal_bounds()
    np.testing.assert_allclose(xL[:, 0], -2.0)
    np.testing.assert_allclose(xU[:, 0], 2.0)
    np.testing.assert_allclose(uL[:, 0], 0.0)
    np.testing.assert_allclose(uU[:, 0], 0.8)
    assert np.all(np.isneginf(uL[:, 1]))
    assert np.all(np.isposinf(uU[:, 1]))
    np.testing.assert_allclose(augmented.obj.cost(lifted), problem.obj.cost(old) + 1.4)


def test_l1_epigraph_preserves_generic_cost_and_quadratic_reference() -> None:
    """Keep the original costs and the L1 slack price when retargeting."""
    model = AffineModel(A=jnp.eye(1), B=jnp.ones((1, 1)))
    generic = Objective(
        stage_cost=GenericCost(lambda x, u: (x[0] - 1.0) ** 4 + u[0] ** 4, n=1, m=1),
        terminal_cost=QuadraticCost(Q=jnp.ones((1, 1)), terminal=True, m=1),
        N=3,
    )
    transformed = _lowered_problem(Problem(model, generic), 3.0)
    np.testing.assert_allclose(transformed.obj[0].evaluate(jnp.array([2.0]), jnp.array([-2.0, 2.0])), 23.0)
    assert not transformed.obj.is_quadratic
    assert transformed.obj[0].m == 2

    quadratic = _lowered_problem(Problem(model, LQRObjective(jnp.ones(1), jnp.ones(1), jnp.ones(1), 3)), 3.0)
    assert not quadratic.obj.carries_reference
    X_ref = jnp.ones((3, 1))
    U_ref = jnp.array([[0.5, 0.0], [0.5, 0.0]])
    aimed = quadratic.obj.with_reference(X_ref, U_ref)
    np.testing.assert_allclose(aimed[0].evaluate(X_ref[0], U_ref[0]), 0.0)
    np.testing.assert_allclose(aimed[0].evaluate(X_ref[0], jnp.array([0.5, 2.0])), 6.0)


@pytest.mark.parametrize("weight", [0.0, -1.0, np.inf, np.nan])
def test_l1_control_cost_requires_positive_scalar_weight(weight: float) -> None:
    """Require a strictly positive, finite L1 weight."""
    with pytest.raises(ValueError, match="weight must be a finite positive scalar"):
        L1ControlCost(n=1, m=1, weight=weight)
