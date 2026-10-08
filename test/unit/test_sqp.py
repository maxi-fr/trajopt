from typing import Any

import jax.numpy as jnp
import osqp
import pytest

from trajopt.cones import SecondOrderCone
from trajopt.constraints.bounds import StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.models.affine import AffineModel
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.transcription.sqp import SQP, SQPResult


@pytest.mark.parametrize("mode", ["bfgs", "gauss_newton"])
def test_sqp_solves_affine_control_problem(mode: str) -> None:
    """SQP returns a feasible Trajectory and cost for a small affine problem."""
    model = AffineModel(A=jnp.array([[1.0]]), B=jnp.array([[0.1]]))
    problem = Problem(
        model=model,
        obj=LQRObjective(Q=jnp.eye(1), R=jnp.eye(1), Qf=jnp.eye(1), N=4),
        constraints=ConstraintList(1, 1, 4),
        N=4,
        dt=0.1,
    )
    result = MPC(problem, SQP(hessian=mode), x0=jnp.array([1.0])).solve()
    assert isinstance(result, SQPResult)
    assert result.success, result.message
    assert result.constraint_violation < 1e-6
    assert result.iterations > 0
    assert result.cost > 0.0


def test_sqp_accepts_second_order_cone() -> None:
    """SQP handles a conic control limit through linearized supporting planes."""
    model = AffineModel(A=jnp.array([[1.0]]), B=jnp.array([[1.0]]))
    constraints = ConstraintList(1, 1, 4)
    constraints.add_constraint(NormConstraint(n=1, m=1, val=0.25, sense=SecondOrderCone(), inds="control"), range(3))
    problem = Problem(
        model=model,
        obj=LQRObjective(Q=jnp.eye(1), R=jnp.eye(1) * 0.01, Qf=jnp.eye(1), N=4),
        constraints=constraints,
        N=4,
        dt=1.0,
    )
    result = MPC(problem, SQP(), x0=jnp.array([1.0])).solve()
    assert result.success, result.message
    assert result.constraint_violation < 1e-6
    assert jnp.max(jnp.abs(result.trajectory.U)) <= 0.25 + 1e-6


def test_sqp_reports_infeasible_linearized_subproblem() -> None:
    """Conflicting initial state and state bounds report infeasibility."""
    model = AffineModel(A=jnp.array([[1.0]]), B=jnp.array([[1.0]]))
    constraints = ConstraintList(1, 1, 3)
    constraints.add_constraint(StateBound(n=1, m=1, x_min=[-1.0], x_max=[0.0]), range(3))
    problem = Problem(
        model=model,
        obj=LQRObjective(Q=jnp.eye(1), R=jnp.eye(1), Qf=jnp.eye(1), N=3),
        constraints=constraints,
        N=3,
        dt=1.0,
    )
    result = MPC(problem, SQP(), x0=jnp.array([1.0])).solve()
    assert result.status == "infeasible"
    assert not result.success


def test_sqp_hessian_stays_sparse_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    """OSQP receives only knot-local Hessian entries after BFGS updates."""
    horizon = 20
    model = AffineModel(A=jnp.array([[1.0]]), B=jnp.array([[0.1]]))
    problem = Problem(
        model=model,
        obj=LQRObjective(Q=jnp.eye(1), R=jnp.eye(1) * 0.1, Qf=jnp.eye(1), N=horizon),
        constraints=ConstraintList(1, 1, horizon),
        N=horizon,
        dt=0.1,
    )
    nonzeros: list[tuple[int, int]] = []
    updates: list[tuple[int, int]] = []
    original_setup = osqp.OSQP.setup
    original_update = osqp.OSQP.update

    def record_setup(self: osqp.OSQP, *args: object, **kwargs: Any) -> None:
        """Record the actual QP Hessian pattern passed to OSQP."""
        nonzeros.append((kwargs["P"].nnz, kwargs["A"].nnz))
        original_setup(self, *args, **kwargs)

    def record_update(self: osqp.OSQP, *args: object, **kwargs: Any) -> None:
        """Record OSQP matrix value updates on the fixed sparse pattern."""
        updates.append((len(kwargs["Px"]), len(kwargs["Ax"])))
        original_update(self, *args, **kwargs)

    monkeypatch.setattr(osqp.OSQP, "setup", record_setup)
    monkeypatch.setattr(osqp.OSQP, "update", record_update)
    result = MPC(problem, SQP(), x0=jnp.array([1.0])).solve()
    assert result.success, result.message
    assert len(nonzeros) == 1
    assert updates
    assert max(p for p, _ in nonzeros) <= 3 * (horizon - 1) + 1
    assert max(a for _, a in nonzeros) <= 5 * horizon
    assert all(p <= 3 * (horizon - 1) + 1 and a <= 5 * horizon for p, a in updates)
