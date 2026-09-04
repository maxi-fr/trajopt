import equinox as eqx
import jax.numpy as jnp
import numpy as np

from trajopt.constraints import ConstraintList, ControlBound, GoalConstraint, StateBound
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.solvers.al import AL, ALConstraints, evaluate_al_residuals, penalty_update
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.options import SolverOptions, TerminationStatus


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


def test_penalty_cap_pulls_an_inherited_penalty_back_down() -> None:
    """The cap governs a carried mu, not only a freshly scaled one: an overflowing row comes down."""
    n, m, N = 2, 1, 3
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(GoalConstraint(n=n, xf=[0.0, 0.0]), N - 1)
    prob = Problem(
        model=Cartpole(),
        obj=LQRObjective(Q=jnp.ones(4), R=jnp.ones(1), Qf=jnp.ones(4), N=N),
        constraints=clist,
        N=N,
        dt=0.1,
        integrator=RK4(),
    )
    options = SolverOptions()
    al = ALConstraints.build(prob.constraints, penalty_initial=1.0)
    inherited = eqx.tree_at(lambda a: a.mu, al, jnp.where(al.row_mask, options.penalty_max, al.mu))

    C = jnp.where(al.row_mask, 10.0, 0.0)
    updated = penalty_update(inherited, options, C=C)

    live = np.asarray(al.row_mask)
    mu = np.asarray(updated.mu)[live]
    assert mu.max() < options.penalty_max, "an inherited penalty at penalty_max must be capped, not held"
    np.testing.assert_allclose(mu, 2.0 * options.max_cost_value / 100.0)
    assert (0.5 * mu * 100.0).max() <= options.max_cost_value


def test_capped_penalties_stay_bounded_when_inherited_across_horizon_steps() -> None:
    """A receding-horizon run carrying both duals and penalties stays finite and non-fatal.

    The combination is new: the penalty cap was added against a solve that starts from
    `penalty_initial`, while `reset_penalties=False` hands the next step whatever the last one
    ratcheted to. Neither branch covered a capped penalty arriving as an inheritance.

    The cap holds against the residuals of the update that applied it, not against the trajectory
    the next step goes on to find, so the invariant checked per step is the one that survives the
    seam: penalties are genuinely inherited, nothing is fatal, nothing is infinite, and nothing sits
    above `penalty_max`. The cap binding is not asserted here and should not be -- gated on the
    active set it converges out of the picture, since a row that is actually in the cost has its
    residual driven to zero. `test_penalty_cap_pulls_an_inherited_penalty_back_down` is where the
    cap meets an inherited penalty head on.
    """
    n, m, N = 4, 1, 15
    dt = 0.05
    x0 = jnp.array([0.0, np.pi - 0.25, 0.1, -0.2])
    xf = jnp.array([0.0, np.pi, 0.0, 0.0])
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(n=n, m=m, u_min=[-8.0], u_max=[8.0]), range(N - 1))
    clist.add_constraint(
        StateBound(n=n, m=m, x_min=[-0.8, -np.inf, -np.inf, -np.inf], x_max=[0.8, np.inf, np.inf, np.inf]),
        range(N),
    )
    obj = LQRObjective(
        Q=jnp.diag(jnp.array([5.0, 20.0, 1.0, 2.0])) * dt,
        R=jnp.diag(jnp.array([0.05])) * dt,
        Qf=jnp.diag(jnp.array([50.0, 200.0, 10.0, 20.0])),
        N=N,
    )
    prob = Problem(model=Cartpole(), obj=obj, constraints=clist, N=N, dt=dt, integrator=RK4())
    options = SolverOptions(reset_duals=False, reset_penalties=False)

    dmodel = prob.model.discretize(RK4())
    mpc = MPC(prob, ALTRO(options=options), x0=x0, xf=xf)
    x, t = x0, 0.0
    fatal = {TerminationStatus.MAXIMUM_COST, TerminationStatus.STATE_LIMIT, TerminationStatus.CONTROL_LIMIT}
    inherited = False

    for _ in range(8):
        mpc.measure(x, t)
        result = mpc.solve()

        assert TerminationStatus[result.message] not in fatal, result.message
        assert np.isfinite(result.cost)

        assert mpc.warm_start.al is not None
        live = np.asarray(mpc.warm_start.al.row_mask)
        mu = np.asarray(mpc.warm_start.al.mu)[live]
        assert np.all(np.isfinite(mu))
        assert np.all(mu <= options.penalty_max)

        inherited |= bool(mu.max() > options.penalty_initial)

        x = dmodel.discrete_dynamics(x, mpc.controls[0], t, dt)
        mpc.shift(dt)
        t += dt

    assert inherited, "penalties never crossed a step boundary, so this run does not exercise the carry"


def test_penalty_cap_leaves_a_satisfied_inequality_on_altros_ladder() -> None:
    """A comfortably satisfied inequality contributes no penalty cost, so the cap must not throttle it.

    This is what the whole-solve Julia cross-verification turns on: capping against `c^2` on rows
    outside `_active_penalty` diverges from Altro's unconditional ladder on rows where there is no
    overflow to prevent.
    """
    n, m, N = 4, 1, 3
    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(n=n, m=m, u_min=[-3.0], u_max=[3.0]), range(N - 1))
    prob = Problem(
        model=Cartpole(),
        obj=LQRObjective(Q=jnp.ones(n), R=jnp.ones(m), Qf=jnp.ones(n), N=N),
        constraints=clist,
        N=N,
        dt=0.1,
        integrator=RK4(),
    )
    options = SolverOptions()
    al = ALConstraints.build(prob.constraints, penalty_initial=1.0)
    mu0 = 1.0e6
    inherited = eqx.tree_at(lambda a: a.mu, al, jnp.where(al.row_mask, mu0, al.mu))

    # A satisfied inequality row: c well below zero with a zero multiplier, so `a` is 0 in `al_cost`.
    slack = -50.0
    C = jnp.where(al.row_mask, slack, 0.0)
    assert 0.5 * mu0 * options.penalty_scaling * slack**2 > options.max_cost_value

    updated = penalty_update(inherited, options, C=C)
    mu = np.asarray(updated.mu)[np.asarray(al.row_mask)]
    np.testing.assert_allclose(mu, mu0 * options.penalty_scaling)
