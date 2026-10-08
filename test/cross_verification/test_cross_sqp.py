import jax.numpy as jnp
import numpy as np
import pytest
from cross_verification.casadi_baseline import assert_parity, build_casadi_from_problem

from trajopt.constraints.constraint_list import ConstraintList
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.base import DiscretizedDynamics
from trajopt.dynamics.integrators import Euler
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.transcription.sqp import SQP


@pytest.mark.slow
def test_sqp_agrees_with_casadi_sqpmethod() -> None:
    """Compare converged trajectories and costs on a nonlinear pendulum problem."""
    pytest.importorskip("casadi")
    model = DiscretizedDynamics(Pendulum(), Euler())
    problem = Problem(
        model=model,
        obj=LQRObjective(Q=jnp.eye(2), R=jnp.eye(1), Qf=jnp.eye(2), N=6),
        constraints=ConstraintList(2, 1, 6),
        N=6,
        dt=0.1,
    )
    x0 = jnp.array([0.2, 0.0])
    ours = MPC(problem, SQP(), x0=x0).solve()
    reference = build_casadi_from_problem(problem, np.asarray(x0), dt=0.1, integrator="euler").solve(solver="sqpmethod")
    assert_parity(ours, reference, tol_state=1e-3, tol_control=1e-3, tol_cost=1e-4, check_duals=False)
