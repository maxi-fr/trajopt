import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, SecondOrderCone
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.geometric import NormConstraint
from trajopt.constraints.linear import GoalConstraint
from trajopt.costs.objective import LQRObjective, Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import ContinuousDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import RK4, Euler
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.transcription.clarabel import Clarabel, ClarabelResult
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.osqp import OSQP, OSQPResult
from trajopt.transcription.result import SolverResult, split_bound_duals


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
    state_clarabel = MPCState.initial(prob_clarabel, x0=x0, dt=dt)
    state_ipopt = MPCState.initial(prob_ipopt, x0=x0, dt=dt)

    res_clarabel = Clarabel(options={"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8}).solve(prob_clarabel, state_clarabel)
    res_ipopt = Ipopt(options={"tol": 1e-8, "print_level": 0}).solve(prob_ipopt, state_ipopt)

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

    state_ipopt = prob.solve(state, solver=Ipopt(options={"print_level": 0}))
    state_osqp = prob.solve(state, solver=OSQP(options={"eps_abs": 1e-7, "eps_rel": 1e-7}))
    state_clarabel = prob.solve(state, solver=Clarabel(options={"tol_gap_abs": 1e-7}))

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


# Each backend runs at the tolerance it needs to have converged rather than merely stopped: OSQP
# is a first-order method and does not reach the interior-point solvers' accuracy at its defaults.
_IPOPT_OPTS: dict[str, Any] = {"print_level": 0, "tol": 1e-10}
_OSQP_OPTS: dict[str, Any] = {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 20000}
_CLARABEL_OPTS: dict[str, Any] = {}

_BACKENDS = [
    ("ipopt", Ipopt, IpoptResult, _IPOPT_OPTS),
    ("osqp", OSQP, OSQPResult, _OSQP_OPTS),
    ("clarabel", Clarabel, ClarabelResult, _CLARABEL_OPTS),
]


def _bound_active_double_integrator() -> tuple[Problem, MPCState]:
    """Build a problem all three backends solve to the same optimum, with bounds active both ways.

    Euler-discretised double integrator dynamics are affine and the cost quadratic, so the QP
    backends approximate nothing here: any disagreement in the duals is a row-layout or sign bug
    rather than a modelling difference. The velocity and thrust limits are tight enough to be
    active at both their upper and their lower ends, which is what exercises the signed
    mult_x_U - mult_x_L convention rather than only half of it.
    """
    model = DiscretizedDynamics(PlanarDoubleIntegrator(), Euler())
    n, m, N = 4, 2, 20
    dt = 0.1

    x0 = jnp.array([1.0, -0.8, 0.0, 0.0])
    xf = jnp.zeros(n)
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m) * 0.1, Qf=jnp.eye(n) * 10.0, xf=xf, N=N)

    cl = ConstraintList(n, m, N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-1.4, -1.4], u_max=[1.4, 1.4]), range(N - 1))
    cl.add_constraint(
        StateBound(n=n, m=m, x_min=[-5.0, -5.0, -0.95, -0.95], x_max=[5.0, 5.0, 0.95, 0.95]),
        range(N),
    )
    cl.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)

    prob = Problem(model=model, obj=obj, constraints=cl, N=N)
    return prob, MPCState.initial(prob, x0=x0, dt=dt, xf=xf)


def _with_state(state: MPCState, Z, lam, mu) -> MPCState:
    """Return a copy of state carrying the given primal vector and multipliers."""
    return dataclasses.replace(
        state,
        lam=jnp.asarray(lam, dtype=state.Z.dtype),
        mu=jnp.asarray(mu, dtype=state.Z.dtype),
        Z=jnp.asarray(Z, dtype=state.Z.dtype),
    )


def test_common_adapter_interface() -> None:
    """Verify every backend's result conforms to SolverResult and reports convergence honestly."""
    prob, state = _bound_active_double_integrator()

    for name, solver_cls, exp_type, opts in _BACKENDS:
        res = solver_cls(options=opts).solve(prob, state)

        assert isinstance(res, exp_type)
        # Structural conformance rather than a checklist of hasattr: a field renamed or dropped
        # from one backend stops satisfying the Protocol the whole layer is written against.
        assert isinstance(res, SolverResult), f"{name} result does not satisfy SolverResult"

        assert isinstance(res.success, bool)
        assert res.success is True, f"{name} failed: {res.message}"
        assert isinstance(res.iterations, int)
        assert res.iterations >= 0
        assert isinstance(res.constraint_violation, float)
        assert res.constraint_violation < 1e-4
        assert res.lam.shape == state.lam.shape
        assert res.mu.shape == state.mu.shape


def test_backends_agree_on_the_duals_of_a_shared_optimum() -> None:
    """Verify all three adapters report the same duals, in the same rows, with the same signs."""
    prob, state = _bound_active_double_integrator()

    results = {name: solver_cls(options=opts).solve(prob, state) for name, solver_cls, _, opts in _BACKENDS}
    for name, res in results.items():
        assert res.success is True, f"{name} failed: {res.message}"

    ref = results["ipopt"]
    lower, upper = split_bound_duals(np.asarray(ref.mu))
    assert np.sum(upper > 1e-6) >= 2, "no upper bound is active, so a sign error would go unseen"
    assert np.sum(lower > 1e-6) >= 2, "no lower bound is active, so a sign error would go unseen"
    assert not np.any((upper > 1e-6) & (lower > 1e-6)), "a variable cannot press both its limits"

    for name, res in results.items():
        np.testing.assert_allclose(res.Z, ref.Z, atol=1e-6, err_msg=f"{name} found a different optimum")
        np.testing.assert_allclose(res.lam, ref.lam, atol=1e-6, err_msg=f"{name} constraint duals disagree")
        np.testing.assert_allclose(res.mu, ref.mu, atol=1e-6, err_msg=f"{name} bound duals disagree")


@pytest.mark.parametrize(
    ("name", "solver_cls", "opts", "warm_startable"),
    [
        ("ipopt", Ipopt, _IPOPT_OPTS, True),
        ("osqp", OSQP, _OSQP_OPTS, True),
        # Clarabel exposes no warm-start API, so handing it duals is a documented no-op. Asserting
        # the count is unchanged is what keeps that a deliberate gap rather than a silent one.
        ("clarabel", Clarabel, _CLARABEL_OPTS, False),
    ],
)
def test_dual_warm_start_cuts_iterations(name, solver_cls, opts, warm_startable) -> None:
    """Verify handing a backend the converged duals costs it fewer iterations than the primal alone."""
    prob, state = _bound_active_double_integrator()
    solver = solver_cls(options=opts)
    solved = solver.solve(prob, state)
    assert solved.success is True, f"{name} failed: {solved.message}"

    # Both re-solves start from the optimal trajectory, so the only difference between them is
    # whether the multipliers come along.
    both = _with_state(state, solved.Z, solved.lam, solved.mu)
    primal_only = _with_state(state, solved.Z, jnp.zeros_like(state.lam), jnp.zeros_like(state.mu))

    res_primal = solver.solve(prob, primal_only)
    res_both = solver.solve(prob, both)

    assert res_primal.success is True
    assert res_both.success is True
    np.testing.assert_allclose(res_both.cost, res_primal.cost, rtol=1e-6)

    if warm_startable:
        assert res_both.iterations < res_primal.iterations, (
            f"{name} took {res_both.iterations} iterations warm-started against "
            f"{res_primal.iterations} from the primal alone"
        )
    else:
        assert res_both.iterations == res_primal.iterations


def _pendulum_swingup_problem() -> tuple[Problem, jax.Array, float]:
    """Build a bounded Pendulum swing-up, whose RK4 dynamics are genuinely nonlinear."""
    model = DiscretizedDynamics(Pendulum(), RK4())
    n, m, N, dt = 2, 1, 21, 0.05
    obj = LQRObjective(
        Q=jnp.diag(jnp.array([10.0, 1.0])),
        R=jnp.diag(jnp.array([0.1])),
        Qf=jnp.diag(jnp.array([100.0, 10.0])),
        xf=jnp.array([np.pi, 0.0]),
        N=N,
    )
    cl = ConstraintList(n, m, N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    return Problem(model=model, obj=obj, constraints=cl, N=N, integrator=RK4()), jnp.array([0.1, 0.0]), dt


# Eight re-expansions per backend, and the two parameters together are 60 seconds.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("solver_cls", "result_type", "qp_options"),
    [
        # OSQP is a first-order method, so its own tolerances have to be tightened before the
        # residual left over is the linearization's rather than the solver's.
        (OSQP, OSQPResult, {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 40000}),
        (Clarabel, ClarabelResult, {}),
    ],
)
def test_operating_point_drives_a_qp_adapter_onto_the_nonlinear_solution(solver_cls, result_type, qp_options) -> None:
    """Assert re-expanding about the previous solution converges the linearized solve onto Ipopt's."""
    problem, x0, dt = _pendulum_swingup_problem()
    state = MPCState.initial(problem, x0=x0, dt=dt)
    ref = Ipopt(options={"print_level": 0, "tol": 1e-8}).solve(problem, state)

    # Expanded about the origin the QP is a poor model of the pendulum, and says so: the
    # trajectory it returns is far from satisfying the true nonlinear dynamics.
    res_origin = solver_cls(options=qp_options).solve(problem, state)
    assert isinstance(res_origin, result_type)
    assert res_origin.success is True
    assert res_origin.constraint_violation > 1e-2

    z_op = res_origin.Z
    for _ in range(8):
        res = solver_cls(operating_point=z_op, options=qp_options).solve(problem, state)
        z_op = res.Z

    assert res.constraint_violation < 1e-4
    np.testing.assert_allclose(res.cost, ref.cost, rtol=1e-3)
    np.testing.assert_allclose(res.trajectory.X, ref.trajectory.X, atol=5e-3)


@pytest.mark.parametrize("solver_cls", [OSQP, Clarabel])
def test_operating_point_does_not_move_the_solution_of_an_affine_problem(solver_cls) -> None:
    """Assert the expansion point is irrelevant when the dynamics are affine and the cost quadratic."""
    model = DiscretizedDynamics(PlanarDoubleIntegrator(), Euler())
    n, m, N, dt = 4, 2, 12, 0.1
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.5, 0.5]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    cost = QuadraticCost(Q=Q, R=R, r=jnp.zeros(m), c=0.0)
    term = QuadraticCost(Q=Q * 20.0, R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    cl = ConstraintList(n, m, N)
    cl.add_constraint(ControlBound(n=n, m=m, u_min=[-1.0, -1.0], u_max=[1.0, 1.0]), range(N - 1))
    problem = Problem(model=model, obj=Objective(stage_cost=cost, terminal_cost=term, N=N), constraints=cl, N=N)
    x0 = jnp.array([2.0, -1.0, 0.0, 0.0])
    state = MPCState.initial(problem, x0=x0, dt=dt)

    nz = N * n + (N - 1) * m
    z_op = jnp.asarray(np.random.default_rng(11).standard_normal(nz))

    res_origin = solver_cls().solve(problem, state)
    res_shifted = solver_cls(operating_point=z_op).solve(problem, state)

    np.testing.assert_allclose(res_shifted.Z, res_origin.Z, atol=1e-5)
    np.testing.assert_allclose(res_shifted.cost, res_origin.cost, rtol=1e-6)
