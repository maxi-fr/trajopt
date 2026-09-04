import jax.numpy as jnp
import numpy as np

from trajopt.constraints import ConstraintList, ControlBound
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import Problem
from trajopt.program import WarmStart
from trajopt.solvers.al import ALConstraints

N = 6
DT = 0.05


def _bounded_problem() -> Problem:
    """Pendulum swing-up with a control bound at every stage knot."""
    clist = ConstraintList(n=2, m=1, N=N)
    clist.add_constraint(ControlBound(m=1, u_min=[-1.0], u_max=[1.0], n=2), range(N - 1))
    obj = LQRObjective(Q=jnp.eye(2) * DT, R=jnp.eye(1) * DT, Qf=jnp.eye(2) * 10.0, N=N)
    return Problem(model=Pendulum(), obj=obj, constraints=clist, N=N, dt=DT, integrator=RK4())


def test_shift_keeps_the_last_control_bound_multiplier() -> None:
    """The +-inf control-bound padding at knot N - 1 must not zero knot N - 2's live multiplier."""
    problem = _bounded_problem()
    al = ALConstraints.build(problem.constraints, penalty_initial=1.0)

    u_start = al.p_cons_max + 2 * problem.model.n
    assert bool(jnp.all(al.row_mask[N - 2, u_start:]))
    assert not bool(jnp.any(al.row_mask[N - 1, u_start:]))

    lam = al.lam.at[N - 2, u_start:].set(7.5)
    ws = WarmStart.cold(problem, x0=jnp.zeros(2))
    ws = WarmStart(
        Z=ws.Z,
        lam=ws.lam,
        mu=ws.mu,
        al=ALConstraints(lam=lam, mu=al.mu, row_mask=al.row_mask, is_equality=al.is_equality, p_cons_max=al.p_cons_max),
    )

    shifted = ws.shift(problem)
    assert shifted.al is not None
    # Knot N - 2 holds the old last knot's rows, so its multiplier is held, not zeroed.
    np.testing.assert_allclose(np.asarray(shifted.al.lam[N - 2, u_start:]), 7.5)
    # Padded rows stay at zero lambda and keep their own penalty.
    np.testing.assert_allclose(np.asarray(shifted.al.lam[N - 1, u_start:]), 0.0)
    np.testing.assert_allclose(np.asarray(shifted.al.mu[N - 1, u_start:]), 0.0)
    np.testing.assert_allclose(np.asarray(shifted.al.mu[N - 2, u_start:]), 1.0)
