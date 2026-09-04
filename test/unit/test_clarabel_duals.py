import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import SecondOrderCone
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import ContinuousDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import Euler
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.transcription.clarabel import Clarabel
from trajopt.transcription.layout import operating_point_z
from trajopt.transcription.subproblem import quadratic_subproblem


class PlanarDoubleIntegrator(ContinuousDynamics):
    """Planar double integrator with `m` force channels driving the last `m` states."""

    def __init__(self, m: int) -> None:
        super().__init__(n=2 * m, m=m, ne=2 * m)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate continuous dynamics."""
        del t
        return jnp.concatenate([x[self.m :], u])


def _problem(clist: ConstraintList, n: int, m: int, N: int) -> Problem:
    """Double-integrator tracking problem carrying `clist`."""
    obj = Objective(
        stage_cost=QuadraticCost(Q=jnp.eye(n), R=jnp.eye(m) * 0.1, r=jnp.zeros(m), c=0.0),
        terminal_cost=QuadraticCost(Q=10.0 * jnp.eye(n), R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0),
        N=N,
    )
    model = DiscretizedDynamics(PlanarDoubleIntegrator(m), Euler())
    return Problem(model=model, obj=obj, constraints=clist, N=N, dt=0.1)


def _stationarity_residual(clist: ConstraintList, n: int, m: int, N: int) -> float:
    """Max abs KKT stationarity residual of the solve's own subproblem at Clarabel's reported duals."""
    problem = _problem(clist, n, m, N)
    x0 = jnp.asarray(np.linspace(2.0, 1.0, n), dtype=jnp.float64)
    bc = BoundaryConditions(x0=x0, t0=jnp.asarray(0.0))
    ws = WarmStart.cold(problem, x0)

    backend = Clarabel()
    with pytest.warns(UserWarning, match="single convex subproblem"):
        res = backend.solve(Program(problem, backend), bc, ws)
    assert res.success

    retargeted = retarget_problem(problem, bc)
    qp = quadratic_subproblem(retargeted, operating_point_z(retargeted, None), bc)

    P = qp.P.toarray()
    P_full = P + P.T - np.diag(np.diag(P))
    z = np.asarray(res.Z, dtype=np.float64)
    grad = P_full @ z + qp.q + qp.A.T @ res.lam + res.mu
    return float(np.max(np.abs(grad)))


def test_clarabel_duals_satisfy_stationarity_without_soc() -> None:
    """Box- and orthant-only duals reported by Clarabel satisfy the subproblem's KKT stationarity."""
    n, m, N = 4, 2, 11
    clist = ConstraintList(n, m, N)
    clist.add_constraint(ControlBound(m=m, u_min=-0.5 * jnp.ones(m), u_max=0.5 * jnp.ones(m), n=n), range(N - 1))
    clist.add_constraint(StateBound(n=n, x_min=-5.0 * jnp.ones(n), x_max=5.0 * jnp.ones(n), m=m), range(N))

    assert _stationarity_residual(clist, n, m, N) < 1e-6


def test_clarabel_soc_duals_satisfy_stationarity() -> None:
    """SecondOrderCone row duals are reported in canonical row order, not Clarabel's permuted one."""
    n, m, N = 4, 2, 11
    clist = ConstraintList(n, m, N)
    clist.add_constraint(NormConstraint(n=n, m=m, val=0.4, sense=SecondOrderCone(), inds="control"), range(N - 1))

    assert _stationarity_residual(clist, n, m, N) < 1e-6
