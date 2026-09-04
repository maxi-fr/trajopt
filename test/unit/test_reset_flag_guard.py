import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints import ConstraintList, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.solvers.al import AL
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions

MATCH = "reset_duals=False with options.reset_penalties=True"


def _small_goal_only_problem() -> tuple[Problem, jax.Array, jax.Array]:
    """Small cartpole swing-up with only a terminal (equality) goal constraint, for fast AL solves."""
    n, m, N, tf = 4, 1, 41, 2.0
    dt = tf / (N - 1)
    x0 = jnp.zeros(n)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    obj = LQRObjective(
        Q=jnp.asarray(1e-2 * np.ones(n) * dt),
        R=jnp.asarray(1e-1 * np.ones(m) * dt),
        Qf=jnp.asarray(1e2 * np.ones(n)),
        N=N,
    )
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(GoalConstraint(n=n, xf=xf.tolist()), N - 1)
    return Problem(model=Cartpole(), obj=obj, constraints=clist, N=N, dt=dt, integrator=RK4()), x0, xf


def _warm_state() -> tuple[Problem, BoundaryConditions, WarmStart]:
    """A problem plus the boundary conditions and populated `ws.al` left by one short AL solve."""
    prob, x0, xf = _small_goal_only_problem()
    mpc = MPC(prob, AL(options=SolverOptions(iterations=2, iterations_outer=1)), x0=x0, xf=xf)
    mpc.solve()
    return prob, mpc.bc, mpc.warm_start


@pytest.mark.parametrize("solver_cls", [AL, ALTRO, BoxQP])
def test_stale_duals_with_fresh_penalties_raises(solver_cls: type) -> None:
    """A carried `ws.al` with reset_duals=False and reset_penalties=True is a corrupted objective, not a warm start."""
    prob, bc, ws = _warm_state()
    options = SolverOptions(iterations=2, iterations_outer=1, reset_duals=False, reset_penalties=True)
    assert ws.al is not None

    with pytest.raises(ValueError, match=MATCH):
        Program(prob, solver_cls(options=options)).solve(bc, ws)


def test_no_carried_al_state_is_not_rejected() -> None:
    """With `ws.al` None there is nothing to corrupt, so the mixed flags pass, matching the conic guard's style."""
    prob, bc, ws = _warm_state()
    cold = WarmStart(Z=ws.Z, lam=ws.lam, mu=ws.mu)
    options = SolverOptions(iterations=2, iterations_outer=1, reset_duals=False, reset_penalties=True)

    Program(prob, AL(options=options)).solve(bc, cold)
