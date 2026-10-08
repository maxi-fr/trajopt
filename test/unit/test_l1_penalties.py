import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.constraints.output import OutputConstraint
from trajopt.costs import GenericCost, LQRObjective, Objective, QuadraticCost
from trajopt.costs.pseudo_huber import PseudoHuberControlCost
from trajopt.models.affine import AffineModel
from trajopt.problem import Problem, add_l1_epigraph
from trajopt.trajectory import Trajectory


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


def test_l1_epigraph_adds_exact_control_penalty_without_changing_dynamics() -> None:
    model = AffineModel(A=jnp.eye(1), B=jnp.array([[2.0, -1.0]]))
    problem = Problem(model, LQRObjective(jnp.array([1.0]), jnp.array([0.5, 0.25]), jnp.array([2.0]), N=3))
    augmented = add_l1_epigraph(problem, weight=4.0)
    x = jnp.array([0.7])
    u = jnp.array([-1.5, 0.5])
    s = jnp.array([1.5, 0.5])

    assert problem.model.m == 2
    assert augmented.model.m == 4
    assert augmented.obj.m == 4
    np.testing.assert_allclose(augmented.model.discrete_dynamics(x, jnp.concatenate([u, s]), 0.0, 0.1), [0.7 - 3.5])
    np.testing.assert_allclose(
        augmented.obj.stage_cost.evaluate(x, jnp.concatenate([u, s])),
        problem.obj.stage_cost.evaluate(x, u) + 8.0,
    )
    residual = augmented.constraints.evaluate_knot(0, x, jnp.concatenate([u, s]))[-4:]
    assert bool(jnp.all(residual <= 0.0))
    assert bool(jnp.all(jnp.max(jnp.stack([residual[:2], residual[2:]]), axis=0) == 0.0))
    assert augmented.obj.is_quadratic


def test_l1_epigraph_preserves_constraints_bounds_and_horizon_rows() -> None:
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
    augmented = add_l1_epigraph(problem, 2.0)
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
    model = AffineModel(A=jnp.eye(1), B=jnp.ones((1, 1)))
    generic = Objective(
        stage_cost=GenericCost(lambda x, u: (x[0] - 1.0) ** 4 + u[0] ** 4, n=1, m=1),
        terminal_cost=QuadraticCost(Q=jnp.ones((1, 1)), terminal=True, m=1),
        N=3,
    )
    transformed = add_l1_epigraph(Problem(model, generic), 3.0)
    np.testing.assert_allclose(transformed.obj.stage_cost.evaluate(jnp.array([2.0]), jnp.array([-2.0, 2.0])), 23.0)
    assert not transformed.obj.is_quadratic
    assert transformed.obj[0].m == 2

    quadratic = add_l1_epigraph(Problem(model, LQRObjective(jnp.ones(1), jnp.ones(1), jnp.ones(1), 3)), 3.0)
    assert not quadratic.obj.carries_reference
    X_ref = jnp.ones((3, 1))
    U_ref = jnp.array([[0.5, 0.0], [0.5, 0.0]])
    aimed = quadratic.obj.with_reference(X_ref, U_ref)
    np.testing.assert_allclose(aimed[0].evaluate(X_ref[0], U_ref[0]), 0.0)
    np.testing.assert_allclose(aimed[0].evaluate(X_ref[0], jnp.array([0.5, 2.0])), 6.0)


def test_l1_epigraph_requires_positive_scalar_weight() -> None:
    problem = Problem(
        AffineModel(A=jnp.eye(1), B=jnp.ones((1, 1))), LQRObjective(jnp.ones(1), jnp.ones(1), jnp.ones(1), 2)
    )
    for weight in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="weight must be a finite positive scalar"):
            add_l1_epigraph(problem, weight)


def test_l1_epigraph_solve_matches_absolute_value_optimum() -> None:
    model = AffineModel(A=jnp.eye(1), B=jnp.ones((1, 1)))
    objective = LQRObjective(jnp.zeros(1), jnp.array([0.1]), jnp.ones(1), 2).with_reference(
        jnp.array([[0.0], [2.0]]), jnp.zeros((1, 1))
    )
    augmented = add_l1_epigraph(Problem(model, objective), 0.5)
    times = jnp.array([0.0, 0.1])

    def trajectory(z: np.ndarray) -> Trajectory:
        X = jnp.array([[z[0]], [z[3]]])
        U = jnp.array([[z[1], z[2]]])
        return Trajectory(X=X, U=U, t=times, dt=jnp.array([0.1]))

    result = minimize(
        lambda z: float(augmented.obj.cost(trajectory(z))),
        x0=np.zeros(4),
        method="SLSQP",
        bounds=[(0.0, 0.0), (None, None), (None, None), (None, None)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda z: np.asarray(
                    [z[3] - augmented.model.discrete_dynamics(jnp.array([z[0]]), jnp.array([z[1], z[2]]), 0.0, 0.1)[0]]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda z: (
                    -np.asarray(augmented.constraints.evaluate_knot(0, jnp.array([z[0]]), jnp.array([z[1], z[2]])))
                ),
            },
        ],
        options={"ftol": 1e-10},
    )

    assert result.success
    u, s = result.x[1:3]
    np.testing.assert_allclose(u, 1.5 / 1.1, atol=1e-4)
    np.testing.assert_allclose(s, abs(u), atol=1e-5)
    np.testing.assert_allclose(result.fun, 0.5 * (u - 2.0) ** 2 + 0.05 * u**2 + 0.5 * abs(u), atol=1e-6)
