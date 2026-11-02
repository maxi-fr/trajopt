import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.layout import _trajectory_to_z, _z_to_trajectory
from trajopt.transcription.result import constraint_row_count
from trajopt.transcription.single_shooting import (
    SingleShooting,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    single_shooting_dimensions,
)
from trajopt.transcription.transcription import (
    eval_f as eval_f_trans,
)
from trajopt.transcription.transcription import (
    eval_g as eval_g_trans,
)


def test_single_shooting_dimensions() -> None:
    """Solver sees controls only, with exactly the registered user constraints."""
    model = Pendulum()
    n, m, N = model.n, model.m, 21
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(
        Q=jnp.diag(jnp.array([10.0, 1.0])), R=jnp.diag(jnp.array([0.1])), Qf=jnp.diag(jnp.array([100.0, 10.0])), N=N
    )
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    constraints.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())

    n_u, p_user = single_shooting_dimensions(prob)
    assert n_u == (N - 1) * m
    assert p_user == n

    x0 = jnp.array([0.0, 0.0])
    u = jnp.zeros(n_u)
    assert eval_g(prob, u, x0, t0=0.0, dt=dt, xf=xf).shape == (p_user,)
    assert eval_grad_f(prob, u, x0, t0=0.0, dt=dt).shape == (n_u,)
    assert eval_jac_g(prob, u, x0, t0=0.0, dt=dt, xf=xf).shape == (p_user * n_u,)


def test_single_shooting_rejects_state_bound() -> None:
    """State bounds have no primal variable in single shooting and must be refused."""
    model = Pendulum()
    n, m, N = model.n, model.m, 10
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(StateBound(n=n, x_min=[-1.0, -1.0], x_max=[1.0, 1.0]), range(N))
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=0.05, integrator=RK4())
    with pytest.raises(ValueError, match="state bound"):
        MPC(prob, SingleShooting(Ipopt()), x0=jnp.zeros(n), xf=xf).solve()


def test_single_shooting_accepts_general_stage_constraint() -> None:
    """Non-goal stage constraints are solved, not refused: states are a differentiable function of u."""
    model = Pendulum()
    n, m, N = model.n, model.m, 10
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([10.0])),
        range(N - 1),
    )
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=0.05, integrator=RK4())
    result = MPC(prob, SingleShooting(Ipopt()), x0=jnp.zeros(n), xf=xf).solve()

    assert result.success


def test_single_shooting_rejects_control_constraint_at_terminal_knot() -> None:
    """A control-using constraint at the terminal knot is refused by ConstraintList itself.

    Single shooting no longer validates this: `ConstraintList.add_constraint` already
    refuses it at problem-construction time, for every transcription.
    """
    model = Pendulum()
    n, m, N = model.n, model.m, 10
    constraints = ConstraintList(n=n, m=m, N=N)

    with pytest.raises(ValueError, match="terminal knot"):
        constraints.add_constraint(
            LinearConstraint(n=n, m=m, A=jnp.array([[1.0]]), b=jnp.array([0.0]), inds=[n]),
            N - 1,
        )


@pytest.mark.slow
def test_single_shooting_matches_multiple_shooting() -> None:
    """Both transcriptions solve the same NLP, so their optimal controls agree."""
    pytest.importorskip("cyipopt")

    model = Pendulum()
    n, m, N = model.n, model.m, 21
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(
        Q=jnp.diag(jnp.array([10.0, 1.0])), R=jnp.diag(jnp.array([0.1])), Qf=jnp.diag(jnp.array([100.0, 10.0])), N=N
    )
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    constraints.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())
    x0 = jnp.array([0.0, 0.0])
    options = {"max_iter": 200, "tol": 1e-8, "print_level": 0}
    ms = MPC(prob, Ipopt(options=options), x0=x0, xf=xf).solve()
    ss_mpc = MPC(prob, SingleShooting(Ipopt(options=options), hessian="dense"), x0=x0, xf=xf)
    ss = ss_mpc.solve()

    np.testing.assert_allclose(ss.trajectory.U, ms.trajectory.U, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ss.trajectory.X, ms.trajectory.X, atol=1e-5, rtol=1e-5)

    # Single shooting satisfies dynamics by construction, not merely to solver tolerance.
    rolled = prob.model.rollout(ss_mpc.trajectory())
    np.testing.assert_allclose(rolled.X, ss.trajectory.X, atol=1e-12)


def test_single_shooting_canonical_multiplier_dimensions() -> None:
    """SingleShooting returns canonical-length lam (P) and mu (N*n + H*m)."""
    pytest.importorskip("cyipopt")

    model = Pendulum()
    n, m, N = model.n, model.m, 10
    dt = 0.05
    xf = jnp.zeros(n)
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    constraints.add_constraint(GoalConstraint(n=n, xf=xf), N - 1)
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())
    x0 = jnp.array([0.1, -0.1])

    p_canon = constraint_row_count(prob)
    nz_canon = N * n + (N - 1) * m

    mpc = MPC(prob, SingleShooting(Ipopt(options={"print_level": 0})), x0=x0, xf=xf)
    res = mpc.solve()
    assert res.success
    assert res.lam.shape == (p_canon,)
    assert res.mu.shape == (nz_canon,)

    # Verify zero entries in state positions and control bounds in native control positions
    x_mu, u_mu = _z_to_trajectory(jnp.asarray(res.mu), N, n, m)
    np.testing.assert_allclose(x_mu, 0.0)
    expected_u_mu = np.asarray(res.info["mult_x_U"]) - np.asarray(res.info["mult_x_L"])
    np.testing.assert_allclose(np.asarray(u_mu).reshape(-1), expected_u_mu)


def test_single_shooting_canonical_lagrangian_stationarity() -> None:
    """Canonical Lagrangian gradient vanishes within solver tolerance on a smooth Problem."""
    pytest.importorskip("cyipopt")

    model = Pendulum()
    n, m, N = model.n, model.m, 10
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([10.0])),
        range(N - 1),
    )
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())
    x0 = jnp.zeros(n)
    bc = BoundaryConditions(x0=x0, t0=jnp.array(0.0), xf=xf)
    ws = WarmStart.cold(prob, x0)
    ss = SingleShooting(Ipopt(options={"max_iter": 200, "tol": 1e-8, "print_level": 0}), hessian="dense")
    prog = Program(prob, ss)
    res = ss.solve(prog, bc, ws)
    assert res.success

    retargeted = retarget_problem(prob, bc)

    def canonical_lagrangian(z: jax.Array) -> jax.Array:
        cost = eval_f_trans(retargeted, z, bc.t0, retargeted.dt, bc)
        c = eval_g_trans(retargeted, z, bc.x0, bc.t0, retargeted.dt, xf=bc.xf)
        return cost + jnp.dot(jnp.asarray(res.lam), c) + jnp.dot(jnp.asarray(res.mu), z)

    grad_l = jax.grad(canonical_lagrangian)(res.Z)
    np.testing.assert_allclose(grad_l, 0.0, atol=1e-6)


def test_single_shooting_dual_warm_start() -> None:
    """SingleShooting extracts retained Multipliers from a canonical WarmStart."""
    pytest.importorskip("cyipopt")

    model = Pendulum()
    n, m, N = model.n, model.m, 10
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([10.0])),
        range(N - 1),
    )
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())
    x0 = jnp.zeros(n)
    bc = BoundaryConditions(x0=x0, t0=jnp.array(0.0), xf=xf)
    ws = WarmStart.cold(prob, x0)
    ss = SingleShooting(Ipopt(options={"max_iter": 200, "tol": 1e-8, "print_level": 0}), hessian="dense")
    prog = Program(prob, ss)
    res1 = ss.solve(prog, bc, ws)
    assert res1.success

    ws_warm = WarmStart(Z=res1.Z, lam=jnp.asarray(res1.lam), mu=jnp.asarray(res1.mu))
    res2 = ss.solve(prog, bc, ws_warm)
    assert res2.success
    assert res2.iterations < res1.iterations


def test_single_shooting_cone_feasibility_negative_orthant() -> None:
    """SingleShooting constraint_violation uses Cone projection so satisfied inequalities contribute 0."""
    pytest.importorskip("cyipopt")

    model = Pendulum()
    n, m, N = model.n, model.m, 10
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    # Satisfied inequality with margin: x[0] <= 10.0
    constraints.add_constraint(
        LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.0, 0.0]]), b=jnp.array([10.0])),
        range(N - 1),
    )
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())
    x0 = jnp.zeros(n)
    ss = SingleShooting(Ipopt(options={"print_level": 0}), hessian="dense")
    prog = Program(prob, ss)
    bc = BoundaryConditions(x0=x0, t0=jnp.array(0.0), xf=xf)
    ws = WarmStart.cold(prob, x0)
    res = ss.solve(prog, bc, ws)
    assert res.success
    # The old metric took abs(c) ~ 10.0; canonical Cone projection evaluates to <= 1e-4
    assert res.constraint_violation < 1e-4


def test_mpc_solve_validates_multiplier_dimensions() -> None:
    """MPC.solve raises ValueError on malformed Multiplier dimensions."""
    model = Pendulum()
    n, m, N = model.n, model.m, 10
    dt = 0.05
    xf = jnp.array([np.pi, 0.0])
    obj = LQRObjective(Q=jnp.eye(n), R=jnp.eye(m), Qf=jnp.eye(n), N=N)
    constraints = ConstraintList(n=n, m=m, N=N)
    prob = Problem(model=model, obj=obj, constraints=constraints, N=N, dt=dt, integrator=RK4())
    x0 = jnp.zeros(n)

    class BadMultiplierSolver:
        def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
            """Return result with malformed Multiplier lam."""
            del program, bc, ws
            x_arr = jnp.zeros((N, n))
            u_arr = jnp.zeros((N - 1, m))
            t_arr = jnp.zeros(N)
            dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
            traj = Trajectory(X=x_arr, U=u_arr, t=t_arr, dt=dt_arr)
            z_arr = _trajectory_to_z(x_arr, u_arr)
            return IpoptResult(
                trajectory=traj,
                success=True,
                status=0,
                message="",
                cost=0.0,
                Z=z_arr,
                info={},
                constraint_violation=0.0,
                lam=np.ones(3),  # Malformed length!
                mu=np.zeros(len(z_arr)),
            )

    mpc = MPC(prob, BadMultiplierSolver(), x0=x0, xf=xf)
    with pytest.raises(ValueError, match="Multiplier lam has length 3, expected canonical constraint count"):
        mpc.solve()

    class BadBoundSolver:
        def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
            """Return result with malformed bound Multiplier mu."""
            del program, bc, ws
            x_arr = jnp.zeros((N, n))
            u_arr = jnp.zeros((N - 1, m))
            t_arr = jnp.zeros(N)
            dt_arr = jnp.broadcast_to(jnp.asarray(dt), (N - 1,))
            traj = Trajectory(X=x_arr, U=u_arr, t=t_arr, dt=dt_arr)
            z_arr = _trajectory_to_z(x_arr, u_arr)
            return IpoptResult(
                trajectory=traj,
                success=True,
                status=0,
                message="",
                cost=0.0,
                Z=z_arr,
                info={},
                constraint_violation=0.0,
                lam=np.zeros(constraint_row_count(prob)),
                mu=np.ones(5),  # Malformed length!
            )

    mpc2 = MPC(prob, BadBoundSolver(), x0=x0, xf=xf)
    with pytest.raises(ValueError, match="Multiplier mu has length 5, expected canonical Primal Vector length"):
        mpc2.solve()

