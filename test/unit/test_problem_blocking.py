from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import IdentityCone, NegativeOrthant, PositiveOrthant, SecondOrderCone, ZeroCone
from trajopt.constraints.bounds import BoundConstraint, ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program
from trajopt.solvers.al import AL
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.ilqr import ILQR
from trajopt.transcription.clarabel import Clarabel
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.layout import _cone_bounds, blocked_control_bounds
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting
from trajopt.transcription.sqp import SQP


def _make_problem(
    N: int = 7,
    b: Any = 1,
    binary: tuple[int, ...] = (),
    constraints: ConstraintList | None = None,
) -> Problem:
    model = Pendulum()
    n, m = model.n, model.m
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    return Problem(
        model=model,
        obj=obj,
        constraints=constraints,
        N=N,
        dt=0.05,
        integrator=RK4(),
        binary_control_indices=binary,
        control_block_steps=b,
    )


def test_problem_blocking_construction_defaults() -> None:
    """Problem defaults to unit control blocking and computes canonical dimensions."""
    prob = _make_problem(N=7)
    assert prob.control_block_steps == 1
    assert prob.N == 7
    assert prob.model.n == 2
    assert prob.model.m == 1
    assert len(prob.dt) == 6
    canonical_len = prob.N * prob.model.n + (prob.N - 1) * prob.model.m
    assert canonical_len == 7 * 2 + 6 * 1


def test_problem_blocking_validation_invalid_types() -> None:
    """Problem rejects boolean, fractional, and non-positive control_block_steps."""
    with pytest.raises(TypeError, match="control_block_steps"):
        _make_problem(N=7, b=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="control_block_steps"):
        _make_problem(N=7, b=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="control_block_steps"):
        _make_problem(N=7, b=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control_block_steps"):
        _make_problem(N=7, b=0)
    with pytest.raises(ValueError, match="control_block_steps"):
        _make_problem(N=7, b=-2)


def test_problem_blocking_validation_horizon_divisibility() -> None:
    """Problem requires H = N - 1 >= 1 and H divisible by control_block_steps."""
    model = Pendulum()
    obj = LQRObjective(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), N=1)
    with pytest.raises(ValueError, match="H = N - 1"):
        Problem(model=model, obj=obj, N=1, control_block_steps=1)

    with pytest.raises(ValueError, match="divisible"):
        _make_problem(N=6, b=2)  # H = 5 not divisible by 2

    prob2 = _make_problem(N=7, b=2)  # H = 6 divisible by 2
    assert prob2.control_block_steps == 2
    prob3 = _make_problem(N=7, b=3)  # H = 6 divisible by 3
    assert prob3.control_block_steps == 3
    prob6 = _make_problem(N=7, b=6)  # H = 6 divisible by 6
    assert prob6.control_block_steps == 6


def test_problem_blocking_preserved_through_retargeting() -> None:
    """Retargeting preserves Control Blocking."""
    prob = _make_problem(N=7, b=2)
    bc = BoundaryConditions(x0=jnp.zeros(2), t0=jnp.array(0.0))
    retargeted = retarget_problem(prob, bc)
    assert retargeted.control_block_steps == 2


def test_blocked_control_bounds_computation() -> None:
    """blocked_control_bounds computes intersected bounds across each block."""
    # H = 4, b = 2, B = 2
    cl = ConstraintList(n=2, m=1, N=5)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[-1.0], u_max=[2.0]), 0)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[-0.5], u_max=[1.5]), 1)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[0.2], u_max=[0.2]), 2)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[0.2], u_max=[0.2]), 3)
    prob = _make_problem(N=5, b=2, constraints=cl)

    b_uL, b_uU = blocked_control_bounds(prob)
    assert b_uL.shape == (2, 1)
    assert b_uU.shape == (2, 1)
    np.testing.assert_allclose(b_uL[0, 0], -0.5)
    np.testing.assert_allclose(b_uU[0, 0], 1.5)
    np.testing.assert_allclose(b_uL[1, 0], 0.2)
    np.testing.assert_allclose(b_uU[1, 0], 0.2)


def test_blocked_control_bounds_infeasible_rejection() -> None:
    """blocked_control_bounds raises ValueError when block lower bound exceeds upper bound."""
    # Empty intersection in block 0: [-1.0, 0.0] and [1.0, 2.0]
    cl = ConstraintList(n=2, m=1, N=5)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[-1.0], u_max=[0.0]), 0)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[1.0], u_max=[2.0]), 1)
    prob = _make_problem(N=5, b=2, constraints=cl)
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        blocked_control_bounds(prob)


def test_blocked_control_bounds_binary_validation() -> None:
    """blocked_control_bounds validates that binary control coordinates contain 0.0 or 1.0."""
    # Binary coordinate with [0.2, 0.8] contains neither 0.0 nor 1.0
    cl = ConstraintList(n=2, m=1, N=5)
    cl.add_constraint(ControlBound(n=2, m=1, u_min=[0.2], u_max=[0.8]), range(4))
    prob = _make_problem(N=5, b=2, binary=(0,), constraints=cl)
    with pytest.raises(ValueError, match=r"contains neither 0\.0 nor 1\.0"):
        blocked_control_bounds(prob)

    # Valid binary subsets: [0.0, 0.5] (contains 0.0), [0.5, 1.0] (contains 1.0)
    cl_valid = ConstraintList(n=2, m=1, N=5)
    cl_valid.add_constraint(ControlBound(n=2, m=1, u_min=[0.0], u_max=[0.5]), (0, 1))
    cl_valid.add_constraint(ControlBound(n=2, m=1, u_min=[0.5], u_max=[1.0]), (2, 3))
    prob_valid = _make_problem(N=5, b=2, binary=(0,), constraints=cl_valid)
    b_uL, b_uU = blocked_control_bounds(prob_valid)
    np.testing.assert_allclose(b_uL[0, 0], 0.0)
    np.testing.assert_allclose(b_uU[0, 0], 0.5)
    np.testing.assert_allclose(b_uL[1, 0], 0.5)
    np.testing.assert_allclose(b_uU[1, 0], 1.0)


def test_program_pairing_guards_ipopt_and_single_shooting() -> None:
    """Ipopt and SingleShooting(Ipopt) accept blocked Problems and LinearHorizonConstraints."""
    prob = _make_problem(N=7, b=2, binary=(0,))
    prog_ipopt = Program(prob, Ipopt())
    assert prog_ipopt.solver is not None

    prog_ss = Program(prob, SingleShooting(solver=Ipopt()))
    assert prog_ss.solver is not None

    # LinearHorizonConstraint
    cl = ConstraintList(n=2, m=1, N=7)
    hz = LinearHorizonConstraint(A=jnp.ones((1, 20)), b=jnp.array([1.0]))
    cl.add_horizon_constraint(hz)
    prob_hz = _make_problem(N=7, b=2, constraints=cl)
    assert Program(prob_hz, Ipopt()).solver is not None
    assert Program(prob_hz, SingleShooting(solver=Ipopt())).solver is not None


def test_program_pairing_guards_single_shooting_rejects_state_bounds() -> None:
    """SingleShooting rejects Problem configurations with state bounds."""
    cl_stage = ConstraintList(n=2, m=1, N=7)
    cl_stage.add_constraint(StateBound(n=2, x_min=[-1.0, -1.0], x_max=[1.0, 1.0]), 0)
    prob_stage = _make_problem(N=7, b=2, constraints=cl_stage)
    with pytest.raises(ValueError, match="state bound"):
        Program(prob_stage, SingleShooting(solver=Ipopt()))

    cl_term = ConstraintList(n=2, m=1, N=7)
    cl_term.add_constraint(StateBound(n=2, x_min=[-1.0, -1.0], x_max=[1.0, 1.0]), 6)
    prob_term = _make_problem(N=7, b=2, constraints=cl_term)
    with pytest.raises(ValueError, match="state bound"):
        Program(prob_term, SingleShooting(solver=Ipopt()))

    # Ipopt accepts state bounds
    assert Program(prob_stage, Ipopt()).solver is not None


def test_program_pairing_guards_unsupported_solvers_reject_blocking() -> None:
    """Unsupported Backends and Native Solvers reject control_block_steps > 1."""
    prob_blocked = _make_problem(N=7, b=2)
    unsupported_solvers = [ALTRO(), ILQR(), AL(), SQP(), OSQP(), Clarabel()]

    for s in unsupported_solvers:
        with pytest.raises(ValueError, match="does not support control_block_steps > 1"):
            Program(prob_blocked, s)

    # Unit blocking succeeds on continuous problems
    prob_unit = _make_problem(N=7, b=1)
    for s in unsupported_solvers:
        assert Program(prob_unit, s).solver is not None

    # Unsupported solvers reject binary controls even with b = 1
    prob_bin = _make_problem(N=7, b=1, binary=(0,))
    for s in unsupported_solvers:
        with pytest.raises(ValueError, match="does not support binary controls or Horizon constraints"):
            Program(prob_bin, s)

    # Unsupported solvers reject Horizon constraints even with b = 1
    cl = ConstraintList(n=2, m=1, N=7)
    hz = LinearHorizonConstraint(A=jnp.ones((1, 20)), b=jnp.array([1.0]))
    cl.add_horizon_constraint(hz)
    prob_hz = _make_problem(N=7, b=1, constraints=cl)
    for s in unsupported_solvers:
        with pytest.raises(ValueError, match="does not support binary controls or Horizon constraints"):
            Program(prob_hz, s)


def test_program_pairing_guards_rejects_non_interval_cones() -> None:
    """Pairing guards and _cone_bounds reject non-interval Cones such as SecondOrderCone."""
    cl = ConstraintList(n=2, m=1, N=7)
    cl.add_constraint(NormConstraint(n=2, m=1, inds=(0, 1), val=1.0, sense=SecondOrderCone()), 0)
    prob = _make_problem(N=7, b=1, constraints=cl)

    with pytest.raises(ValueError, match=r"IPOPT|Ipopt|Cannot express non-interval cone"):
        Program(prob, Ipopt())
    with pytest.raises(ValueError, match=r"IPOPT|Ipopt|Cannot express non-interval cone"):
        Program(prob, SingleShooting(solver=Ipopt()))

    # _cone_bounds directly rejects SecondOrderCone
    with pytest.raises(ValueError, match="Cannot express non-interval cone"):
        _cone_bounds(SecondOrderCone(), p=3)

    # Valid interval cones succeed in _cone_bounds
    gL, gU = _cone_bounds(ZeroCone(), p=2)
    assert np.all(gL == 0.0)
    assert np.all(gU == 0.0)
    gL, gU = _cone_bounds(NegativeOrthant(), p=2)
    assert np.all(np.isneginf(gL))
    assert np.all(gU == 0.0)
    gL, gU = _cone_bounds(PositiveOrthant(), p=2)
    assert np.all(gL == 0.0)
    assert np.all(np.isposinf(gU))
    gL, gU = _cone_bounds(IdentityCone(), p=2)
    assert np.all(np.isneginf(gL))
    assert np.all(np.isposinf(gU))
