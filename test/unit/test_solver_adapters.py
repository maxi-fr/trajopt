import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import NegativeOrthant, SecondOrderCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import ContinuousDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import Euler
from trajopt.problem import MPCState, Problem, solve
from trajopt.transcription.clarabel import ClarabelResult, solve_clarabel
from trajopt.transcription.ipopt import IpoptResult, solve_ipopt
from trajopt.transcription.osqp import OSQPResult, solve_osqp


class PlanarDoubleIntegrator(ContinuousDynamics):
    """Planar double integrator: 2D pos, 2D vel, 2D force."""

    def __init__(self) -> None:
        super().__init__(n=4, m=2, ne=4)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        """Evaluate continuous dynamics."""
        del t
        return jnp.array([x[2], x[3], u[0], u[1]])


def test_clarabel_soc_vs_ipopt_quadratic_norm_parity() -> None:
    """Verify Clarabel second-order cone solve against Ipopt quadratic norm formulation."""
    model = DiscretizedDynamics(PlanarDoubleIntegrator(), Euler())
    n, m, N = 4, 2, 16
    dt = 0.1
    max_thrust = 1.5

    Q = jnp.diag(jnp.array([1.0, 1.0, 0.5, 0.5]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    Qf = jnp.diag(jnp.array([50.0, 50.0, 10.0, 10.0]))
    cost = QuadraticCost(Q=Q, R=R, r=jnp.zeros(m), c=0.0)
    term_cost = QuadraticCost(Q=Qf, R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=term_cost, N=N)

    # 1. Clarabel problem with native SecondOrderCone
    clist_clarabel = ConstraintList(n, m, N)
    clist_clarabel.add_constraint(
        NormConstraint(n=n, m=m, val=max_thrust, sense=SecondOrderCone(), inds="control"),
        range(N - 1),
    )
    prob_clarabel = Problem(model=model, obj=obj, constraints=clist_clarabel, N=N)

    # 2. Ipopt problem with NegativeOrthant (quadratic norm <= val^2)
    clist_ipopt = ConstraintList(n, m, N)
    clist_ipopt.add_constraint(
        NormConstraint(n=n, m=m, val=max_thrust, sense=NegativeOrthant(), inds="control"),
        range(N - 1),
    )
    prob_ipopt = Problem(model=model, obj=obj, constraints=clist_ipopt, N=N)

    x0 = jnp.array([3.0, 2.0, -1.0, 0.5])

    res_clarabel = solve_clarabel(prob_clarabel, x0, dt=dt, options={"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8})
    res_ipopt = solve_ipopt(prob_ipopt, x0, dt=dt, options={"tol": 1e-8, "print_level": 0})

    assert res_clarabel.success is True
    assert res_ipopt.success is True

    # Validate cost parity
    assert np.isclose(res_clarabel.cost, res_ipopt.cost, rtol=1e-4, atol=1e-4)

    # Validate trajectory parity
    assert np.allclose(res_clarabel.trajectory.X, res_ipopt.trajectory.X, atol=1e-3)
    assert np.allclose(res_clarabel.trajectory.U, res_ipopt.trajectory.U, atol=1e-3)

    # Validate that norm constraint was respected
    for k in range(N - 1):
        u_norm_clarabel = float(np.linalg.norm(res_clarabel.trajectory.U[k]))
        u_norm_ipopt = float(np.linalg.norm(res_ipopt.trajectory.U[k]))
        assert u_norm_clarabel <= max_thrust + 1e-4
        assert u_norm_ipopt <= max_thrust + 1e-4


def test_problem_definition_invariance_across_solvers() -> None:
    """Verify that solver selection does not change how a problem is defined."""
    model = DiscretizedDynamics(PlanarDoubleIntegrator(), Euler())
    n, m, N = 4, 2, 11
    dt = 0.1

    Q = jnp.eye(n)
    R = jnp.eye(m) * 0.1
    Qf = jnp.eye(n) * 10.0
    cost = QuadraticCost(Q=Q, R=R, r=jnp.zeros(m), c=0.0)
    term_cost = QuadraticCost(Q=Qf, R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=term_cost, N=N)

    clist = ConstraintList(n, m, N)
    clist.add_constraint(
        ControlBound(m=m, u_min=jnp.array([-1.0, -1.0]), u_max=jnp.array([1.0, 1.0]), n=n),
        range(N - 1),
    )

    # Define one single Problem instance
    prob = Problem(model=model, obj=obj, constraints=clist, N=N)
    x0 = jnp.array([1.0, 0.5, 0.0, 0.0])

    state = MPCState.initial(prob, x0=x0, dt=dt)

    state_ipopt = solve(prob, state, solver="ipopt", options={"print_level": 0})
    state_osqp = solve(prob, state, solver="osqp", options={"eps_abs": 1e-7, "eps_rel": 1e-7})
    state_clarabel = solve(prob, state, solver="clarabel", options={"tol_gap_abs": 1e-7})

    assert isinstance(state_ipopt, MPCState)
    assert isinstance(state_osqp, MPCState)
    assert isinstance(state_clarabel, MPCState)

    # Trajectories across all three solvers match for convex QP
    assert np.allclose(state_osqp.Z, state_clarabel.Z, atol=1e-3)
    assert np.allclose(state_ipopt.Z, state_osqp.Z, atol=1e-3)

    # Dual multipliers propagated
    assert state_ipopt.lam is not None
    assert len(state_ipopt.lam) > 0
    assert state_osqp.lam is not None
    assert len(state_osqp.lam) > 0
    assert state_clarabel.lam is not None
    assert len(state_clarabel.lam) > 0


def test_common_adapter_interface() -> None:
    """Verify common interface reporting convergence, iteration count, and constraint violation."""
    model = DiscretizedDynamics(PlanarDoubleIntegrator(), Euler())
    n, m, N = 4, 2, 11
    dt = 0.1

    cost = QuadraticCost(Q=jnp.eye(n), R=jnp.eye(m) * 0.1, r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=cost, N=N)
    prob = Problem(model=model, obj=obj, constraints=ConstraintList(n, m, N), N=N)
    x0 = jnp.array([1.0, 0.5, 0.0, 0.0])

    res_ipopt = solve_ipopt(prob, x0, dt=dt, options={"print_level": 0})
    res_osqp = solve_osqp(prob, x0, dt=dt)
    res_clarabel = solve_clarabel(prob, x0, dt=dt)

    results = [res_ipopt, res_osqp, res_clarabel]
    expected_types = [IpoptResult, OSQPResult, ClarabelResult]

    for res, exp_type in zip(results, expected_types, strict=True):
        assert isinstance(res, exp_type)
        assert hasattr(res, "trajectory")
        assert hasattr(res, "success")
        assert hasattr(res, "status")
        assert hasattr(res, "message")
        assert hasattr(res, "cost")
        assert hasattr(res, "Z")
        assert hasattr(res, "info")
        assert hasattr(res, "iterations")
        assert hasattr(res, "constraint_violation")

        assert isinstance(res.success, bool)
        assert res.success is True
        assert isinstance(res.iterations, int)
        assert res.iterations >= 0
        assert isinstance(res.constraint_violation, float)
        assert res.constraint_violation < 1e-4
