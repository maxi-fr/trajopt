import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.base import CostFunction
from trajopt.costs.generic import GenericCost
from trajopt.costs.objective import LQRObjective, Objective
from trajopt.costs.one_sided_quadratic import OneSidedQuadratic
from trajopt.costs.reformulation import CostReformulation
from trajopt.models.affine import AffineModel
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.solvers.ilqr import ILQR
from trajopt.trajectory import Trajectory
from trajopt.transcription.clarabel import Clarabel
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting
from trajopt.transcription.sqp import SQP


def test_objective_owns_penalty_cost_and_keeps_limits_when_retargeted():
    term = OneSidedQuadratic(jnp.array([[0.0, 1.0]]), jnp.array([0.5]), 2.0, n=1)
    base = LQRObjective(jnp.zeros(1), jnp.ones(1), jnp.ones(1), 2)
    obj = Objective(base.stage_cost + term, base.terminal_cost, N=2)
    reference = Trajectory(X=jnp.zeros((2, 1)), U=jnp.ones((1, 1)), t=jnp.array([0.0, 0.1]), dt=jnp.array([0.1]))
    assert not obj.is_quadratic
    assert obj.can_retarget
    assert obj.cost(reference) == pytest.approx(1.0)
    assert obj.with_reference(reference.X, reference.U).cost(reference) == pytest.approx(0.5)
    assert obj.update_reference(reference).cost(reference) == pytest.approx(0.5)


def _problem(formulation="epigraph", *, terminal=True):
    model = AffineModel(A=jnp.ones((1, 1)), B=jnp.ones((1, 1)))
    stage = OneSidedQuadratic(
        jnp.array([[0.0, 1.0], [0.0, -1.0]]), jnp.zeros(2), [1.0, 2.0], n=1, formulation=formulation
    )
    end = OneSidedQuadratic(jnp.ones((1, 1)), jnp.array([0.5]), 2.0, n=1, terminal=True, formulation=formulation)
    base = LQRObjective(jnp.zeros(1), jnp.ones(1), jnp.ones(1), 2)
    return Problem(
        model,
        Objective(
            base.stage_cost + stage,
            base.terminal_cost + end if terminal else base.terminal_cost,
            N=base.N,
            linear_control_cost=base.linear_control_cost,
        ),
    )


@pytest.mark.parametrize("backend", [OSQP, Clarabel])
@pytest.mark.parametrize("terminal", [False, True])
def test_epigraph_mpc_preserves_physical_layout_and_finds_exact_optimum(backend, terminal):
    pytest.importorskip(backend.__name__.lower())
    initial = Trajectory(X=jnp.zeros((2, 1)), U=jnp.zeros((1, 1)), t=jnp.array([0.0, 0.05]), dt=jnp.array([0.05]))
    reference = Trajectory(X=jnp.array([[0.0], [4.0]]), U=jnp.ones((1, 1)), t=initial.t, dt=initial.dt)
    mpc = MPC(
        _problem(terminal=terminal),
        backend(operating_point=initial),
        x0=[0.0],
        reference=reference,
        penalty_diagnostics=True,
    )
    with pytest.warns(UserWarning, match="single convex subproblem"):
        result = mpc.solve()
    assert result.success
    u = 7.0 / 8.0 if terminal else 5.0 / 4.0
    np.testing.assert_allclose(result.trajectory.U, [[u]], atol=1e-3)
    assert result.Z.shape == result.mu.shape == (3,)
    assert result.lam.shape == (2,)
    np.testing.assert_allclose(mpc.controls, [[u]], atol=1e-3)
    np.testing.assert_allclose(result.cost, mpc.cost(), atol=1e-9)
    stage = result.info["penalties"][0]
    np.testing.assert_allclose(stage.stage_violation, [[u, 0.0]], atol=1e-3)
    np.testing.assert_allclose(stage.stage_slack, stage.stage_violation, atol=1e-3)
    if terminal:
        end = result.info["penalties"][1]
        np.testing.assert_allclose(end.terminal_slack, [u - 0.5], atol=1e-3)
    mpc.advance()
    with pytest.warns(UserWarning, match="single convex subproblem"):
        assert mpc.solve().success
    if backend is OSQP:
        assert mpc.program.handles["osqp"].regime == "vector"


def test_ilqr_requires_direct_form_and_preserves_reference_tracking():
    with pytest.raises(ValueError, match=r"ILQR.*epigraph.*direct"):
        MPC(_problem(), ILQR(), x0=[0.0], xf=[4.0])
    mpc = MPC(_problem("direct"), ILQR(), x0=[0.0], xf=[4.0])
    result = mpc.solve()
    assert result.success
    np.testing.assert_allclose(result.trajectory.U, [[0.75]], atol=1e-4)
    np.testing.assert_allclose(result.cost, mpc.cost())
    assert "penalties" not in result.info


@pytest.mark.parametrize("backend", [Ipopt(), SingleShooting(Ipopt()), SQP()])
def test_shared_epigraph_solves_with_other_constrained_backends(backend):
    if isinstance(backend, (Ipopt, SingleShooting)):
        pytest.importorskip("cyipopt", exc_type=ImportError)
    else:
        pytest.importorskip("osqp")
    mpc = MPC(_problem(), backend, x0=[0.0], xf=[4.0])
    result = mpc.solve()
    assert result.success, result.message
    np.testing.assert_allclose(result.trajectory.U, [[0.75]], atol=1e-3)
    assert "penalties" not in result.info


def test_mixed_forms_keep_diagnostics_in_declaration_order():
    pytest.importorskip("osqp")
    original = _problem(terminal=False)
    end = OneSidedQuadratic(jnp.ones((1, 1)), jnp.array([0.5]), 2.0, n=1, terminal=True, formulation="direct")
    problem = Problem(original.model, Objective(original.obj.stage_cost, original.obj.terminal_cost + end, N=2))
    result = MPC(problem, SQP(), x0=[0.0], xf=[4.0], penalty_diagnostics=True).solve()
    assert result.success, result.message
    np.testing.assert_allclose(result.trajectory.U, [[0.75]], atol=1e-3)
    stage, terminal = result.info["penalties"]
    np.testing.assert_allclose(stage.stage_slack, [[0.75, 0.0]], atol=1e-3)
    np.testing.assert_allclose(terminal.terminal_violation, [0.25], atol=1e-3)
    assert terminal.terminal_slack is None


def test_stage_and_terminal_penalties_preserve_bounds_and_goal_multipliers():
    pytest.importorskip("osqp")
    original = _problem()
    constraints = ConstraintList(n=1, m=1, N=3)
    constraints.add_constraint(ControlBound(m=1, u_max=[0.2], n=1), 0)
    constraints.add_constraint(GoalConstraint(n=1, m=1), 2)
    _, stage = original.obj.stage_cost.terms
    _, terminal = original.obj.terminal_cost.terms
    base = LQRObjective(jnp.zeros(1), jnp.ones(1), jnp.ones(1), 3)
    problem = Problem(
        original.model, Objective(base.stage_cost + stage, base.terminal_cost + terminal, N=3), constraints
    )
    mpc = MPC(problem, OSQP(options={"eps_abs": 1e-8, "eps_rel": 1e-8}), x0=[0.0], xf=[1.0], penalty_diagnostics=True)
    with pytest.warns(UserWarning, match="single convex subproblem"):
        result = mpc.solve()
    assert result.success
    np.testing.assert_allclose(result.trajectory.U, [[0.2], [0.8]], atol=1e-6)
    assert result.mu[1] > 0.0
    assert result.lam.shape == (4,)
    assert result.lam[-1] < 0.0
    assert result.constraint_violation < 1e-6
    np.testing.assert_allclose(result.info["penalties"][1].terminal_slack, [0.5], atol=1e-6)


class _AbsoluteControlCost(CostFunction):
    def evaluate(self, x, u=None, t=0.0):
        assert u is not None
        return jnp.abs(u[0])

    def reformulate(self):
        # Price an auxiliary linearly. The Program has no knowledge of this cost type.
        cost = GenericCost(lambda _x, u: u[1], n=1, m=2)
        rows = LinearConstraint(1, 2, jnp.array([[0.0, 1.0, -1.0], [0.0, -1.0, -1.0]]), jnp.zeros(2))
        return CostReformulation(cost, (rows,), auxiliary_dim=1)


def test_user_defined_cost_can_supply_a_different_epigraph():
    pytest.importorskip("osqp")
    model = AffineModel(A=jnp.ones((1, 1)), B=jnp.ones((1, 1)))
    base = LQRObjective(jnp.zeros(1), jnp.ones(1), jnp.ones(1), 2)
    limit = OneSidedQuadratic(jnp.array([[0.0, 1.0]]), jnp.zeros(1), 1.0, n=1)
    objective = Objective(
        base.stage_cost + _AbsoluteControlCost(1, 1, terminal=False) + limit,
        base.terminal_cost,
        N=2,
        linear_control_cost=base.linear_control_cost,
    )
    mpc = MPC(Problem(model, objective), OSQP(), x0=[0.0], xf=[4.0], penalty_diagnostics=True)
    with pytest.warns(UserWarning, match="single convex subproblem"):
        result = mpc.solve()
    assert result.success
    np.testing.assert_allclose(result.trajectory.U, [[0.75]], atol=1e-3)
    entries = result.info["penalties"]
    assert len(entries) == 2
    for entry in entries:
        np.testing.assert_allclose(entry.stage_slack, [[0.75]], atol=1e-3)
