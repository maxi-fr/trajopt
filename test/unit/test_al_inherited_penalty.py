import jax.numpy as jnp
import numpy as np

from trajopt.constraints import ConstraintList, GoalConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.solvers.al import AL
from trajopt.solvers.options import SolverOptions


def _converged_warm_state() -> tuple[Problem, BoundaryConditions, WarmStart]:
    """A solved cartpole swing-up plus the boundary conditions and converged `ws.al` it leaves behind."""
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
    prob = Problem(model=Cartpole(), obj=obj, constraints=clist, N=N, dt=dt, integrator=RK4())

    mpc = MPC(prob, AL(options=SolverOptions()), x0=x0, xf=xf)
    mpc.solve()
    return prob, mpc.bc, mpc.warm_start


def test_inherited_penalty_cannot_end_the_outer_loop_on_iteration_one() -> None:
    """With reset_penalties=False a feasible first inner solve must still take one dual update."""
    prob, bc, ws = _converged_warm_state()
    carried = SolverOptions(reset_duals=False, reset_penalties=False)

    result = Program(prob, AL(options=carried)).solve(bc, ws)
    assert result.iterations > 1


def test_reset_penalties_still_exits_after_one_outer_iteration() -> None:
    """The guard is targeted: with reset_penalties=True a feasible first inner solve still ends the loop at once."""
    prob, bc, ws = _converged_warm_state()
    loose = SolverOptions(constraint_tolerance=1e3)

    assert Program(prob, AL(options=loose)).solve(bc, ws).iterations == 1
    carried = SolverOptions(constraint_tolerance=1e3, reset_duals=False, reset_penalties=False)
    assert Program(prob, AL(options=carried)).solve(bc, ws).iterations > 1
