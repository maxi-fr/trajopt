import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simulate.sensor import GaussianSensor

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.output import OutputConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.output import OutputCost, pullback_output_cost
from trajopt.costs.quadratic import DiagonalCost, QuadraticCost
from trajopt.dynamics.base import AbstractModel, ContinuousDynamics, DiscretizedDynamics
from trajopt.dynamics.integrators import RK4
from trajopt.models.affine import AffineModel
from trajopt.models.transforms import _linearize_about
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.simulate import TrajOptDynamics, TrajOptMeasurement
from trajopt.solvers.altro import ALTRO
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt


class _DummyModelWithoutOutput(AbstractModel):
    """Model that does not define an output dimension p."""

    def __init__(self) -> None:
        super().__init__(n=2, m=1)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        del t, args
        return x + u[0]

    def discretize(self, integrator=None):
        del integrator
        return self


class _NonlinearOutputModel(ContinuousDynamics):
    """Continuous model with nonlinear dynamics and nonlinear output function."""

    def __init__(self) -> None:
        super().__init__(n=2, m=1, ne=2, p=2)

    def dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array = 0.0) -> jax.Array:
        del t
        return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        del t
        u_val = 0.0 if u is None else u[0]
        return jnp.array([x[0] ** 2 + u_val, jnp.sin(x[1])])


class _ManifoldDummyModel(AbstractModel):
    """Model with non-trivial error-state map G != I for verifying coordinate projection."""

    scale_g: float

    def __init__(self, scale_g: float = 2.5) -> None:
        super().__init__(n=2, m=1, ne=2, p=2)
        self.scale_g = scale_g

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
        *args: float | jax.Array,
    ) -> jax.Array:
        del t, args
        return x + u[0]

    def discretize(self, integrator=None):
        del integrator
        return self

    def errstate_jacobian(self, x: jax.Array) -> jax.Array:
        del x
        return self.scale_g * jnp.eye(2)

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        del t
        u_val = 0.0 if u is None else u[0]
        return jnp.array([x[0] + 3.0 * x[1], 2.0 * u_val])


def test_abstract_model_output_contract() -> None:
    """Verify AbstractModel output contract raises NotImplementedError when p is None."""
    model = _DummyModelWithoutOutput()
    assert model.p is None

    x = jnp.array([1.0, 2.0])
    u = jnp.array([0.5])
    traj = Trajectory(
        X=jnp.ones((4, 2)),
        U=jnp.ones((3, 1)),
        t=jnp.linspace(0.0, 0.3, 4),
        dt=jnp.full(3, 0.1),
    )

    with pytest.raises(NotImplementedError, match="does not define an output dimension p"):
        model.output(x, u)

    with pytest.raises(NotImplementedError, match="does not define an output dimension p"):
        model.output_state_jacobian(x, u)

    with pytest.raises(NotImplementedError, match="does not define an output dimension p"):
        model.output_control_jacobian(x, u)

    with pytest.raises(NotImplementedError, match="does not define an output dimension p"):
        model.evaluate_output(traj)


def test_nonlinear_model_output_and_jacobians() -> None:
    """Verify output evaluation and AD Jacobians on a model with nonlinear output map."""
    model = _NonlinearOutputModel()
    assert model.p == 2

    x = jnp.array([1.5, 0.5])
    u = jnp.array([2.0])

    y = model.output(x, u, 0.0)
    expected_y = jnp.array([1.5**2 + 2.0, jnp.sin(0.5)])
    np.testing.assert_allclose(y, expected_y, atol=1e-12)

    y_term = model.output(x, None, 0.0)
    expected_y_term = jnp.array([1.5**2, jnp.sin(0.5)])
    np.testing.assert_allclose(y_term, expected_y_term, atol=1e-12)

    cx = model.output_state_jacobian(x, u, 0.0)
    expected_cx = jnp.array([[2.0 * 1.5, 0.0], [0.0, jnp.cos(0.5)]])
    np.testing.assert_allclose(cx, expected_cx, atol=1e-12)

    cu = model.output_control_jacobian(x, u, 0.0)
    expected_cu = jnp.array([[1.0], [0.0]])
    np.testing.assert_allclose(cu, expected_cu, atol=1e-12)

    traj = Trajectory(
        X=jnp.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4]]),
        U=jnp.array([[0.5], [1.0]]),
        t=jnp.array([0.0, 0.1, 0.2]),
        dt=jnp.array([0.1, 0.1]),
    )
    Y = model.evaluate_output(traj)
    assert Y.shape == (3, 2)
    np.testing.assert_allclose(Y[0], model.output(traj.X[0], traj.U[0], traj.t[0]))
    np.testing.assert_allclose(Y[1], model.output(traj.X[1], traj.U[1], traj.t[1]))
    np.testing.assert_allclose(Y[2], model.output(traj.X[2], None, traj.t[2]))


def test_discretized_dynamics_output_forwarding() -> None:
    """Verify DiscretizedDynamics forwards output dimension, output, and Jacobians."""
    cont = _NonlinearOutputModel()
    discrete = cont.discretize(RK4())

    assert isinstance(discrete, DiscretizedDynamics)
    assert discrete.p == cont.p == 2

    x = jnp.array([0.7, -0.3])
    u = jnp.array([1.2])

    np.testing.assert_allclose(discrete.output(x, u, 0.1), cont.output(x, u, 0.1))
    np.testing.assert_allclose(discrete.output_state_jacobian(x, u, 0.1), cont.output_state_jacobian(x, u, 0.1))
    np.testing.assert_allclose(discrete.output_control_jacobian(x, u, 0.1), cont.output_control_jacobian(x, u, 0.1))


def test_affine_model_output_and_jacobians() -> None:
    """Verify AffineModel output functions and analytic Jacobians."""
    A = jnp.array([[1.0, 0.1], [0.0, 0.9]])
    B = jnp.array([[0.0], [0.2]])
    d = jnp.array([0.01, -0.02])
    C = jnp.array([[2.0, 1.0], [0.0, -1.0], [1.0, 0.5]])
    D = jnp.array([[0.5], [1.5], [0.0]])
    dy = jnp.array([0.1, 0.2, -0.3])

    model = AffineModel(A=A, B=B, d=d, C=C, D=D, dy=dy)
    assert model.p == 3

    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])

    expected_y = C @ x + D @ u + dy
    np.testing.assert_allclose(model.output(x, u), expected_y, atol=1e-12)

    expected_y_term = C @ x + dy
    np.testing.assert_allclose(model.output(x, None), expected_y_term, atol=1e-12)

    np.testing.assert_allclose(model.output_state_jacobian(x, u), C, atol=1e-12)
    np.testing.assert_allclose(model.output_control_jacobian(x, u), D, atol=1e-12)


def test_linear_trajectory_model_and_error_coordinates() -> None:
    """Verify LinearTrajectoryModel stores C, D and projects state Jacobians via G."""
    model = _ManifoldDummyModel(scale_g=3.0)
    traj = Trajectory(
        X=jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        U=jnp.array([[0.1], [0.2]]),
        t=jnp.array([0.0, 0.1, 0.2]),
        dt=jnp.array([0.1, 0.1]),
    )

    lin_model = _linearize_about(model, traj)
    assert lin_model.p == 2
    assert lin_model.C is not None
    assert lin_model.D is not None
    assert lin_model.C.shape == (3, 2, 2)
    assert lin_model.D.shape == (2, 2, 1)

    # Expected error-coordinate output matrix C_k = (dg/dx) @ G with scale_g = 3.0
    expected_C_k = jnp.array([[3.0, 9.0], [0.0, 0.0]])
    expected_D_k = jnp.array([[0.0], [2.0]])

    for k in range(3):
        np.testing.assert_allclose(lin_model.C[k], expected_C_k, atol=1e-12)
    for k in range(2):
        np.testing.assert_allclose(lin_model.D[k], expected_D_k, atol=1e-12)

    no_output_model = _DummyModelWithoutOutput()
    lin_no_output = _linearize_about(no_output_model, traj)
    assert lin_no_output.p is None
    assert lin_no_output.C is None
    assert lin_no_output.D is None


def test_output_cost_derivatives() -> None:
    """Verify OutputCost gradient and hessian match automatic differentiation."""
    model = _NonlinearOutputModel()
    inner_cost = DiagonalCost(
        Q=jnp.array([3.0, 2.0]),
        R=jnp.array([0.5]),
        q=jnp.array([0.1, -0.2]),
        r=jnp.array([0.05]),
        c=1.2,
    )
    cost = OutputCost(model, inner_cost)
    assert cost.n == model.n == 2
    assert cost.m == model.m == 1
    assert not cost.terminal

    x = jnp.array([1.2, -0.4])
    u = jnp.array([0.8])

    y = model.output(x, u)
    val = cost.evaluate(x, u)
    np.testing.assert_allclose(val, inner_cost.evaluate(y, u))

    grad = cost.gradient(x, u)

    def composite(z):
        return inner_cost.evaluate(model.output(z[:2], z[2:]), z[2:])

    grad_ad = jax.grad(composite)(jnp.concatenate([x, u]))
    np.testing.assert_allclose(grad, grad_ad, atol=1e-12)

    hess = cost.hessian(x, u)
    hess_ad = jax.hessian(composite)(jnp.concatenate([x, u]))
    np.testing.assert_allclose(hess, hess_ad, atol=1e-12)

    term_inner = DiagonalCost(Q=jnp.array([5.0, 1.0]), q=jnp.array([0.2, 0.0]), c=0.5, terminal=True, m=1)
    cost_term = OutputCost(model, term_inner)
    assert cost_term.terminal

    grad_term = cost_term.gradient(x, None)
    grad_term_ad = jax.grad(lambda x_: term_inner.evaluate(model.output(x_, None), None))(x)
    np.testing.assert_allclose(grad_term, grad_term_ad, atol=1e-12)


def test_pullback_output_cost_analytic_matches_numeric() -> None:
    """Verify pullback_output_cost produces exact QuadraticCost matching numeric evaluations."""
    A = jnp.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 0.9]])
    B = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2]])
    d = jnp.array([0.05, -0.02, 0.01])
    C = jnp.array([[1.0, 0.5, -0.2], [0.0, 1.2, 0.8]])
    D = jnp.array([[0.3, -0.1], [0.0, 0.4]])
    dy = jnp.array([-0.2, 0.1])

    model = AffineModel(A=A, B=B, d=d, C=C, D=D, dy=dy)

    Q_y = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    R_u = jnp.array([[2.0, 0.5], [0.5, 1.5]])
    H_yu = jnp.array([[0.2, -0.1], [0.1, 0.3]])
    q_y = jnp.array([0.4, -0.6])
    r_u = jnp.array([0.1, 0.2])
    c_val = 2.5

    quad_cost = QuadraticCost(Q=Q_y, R=R_u, H=H_yu, q=q_y, r=r_u, c=c_val, terminal=False)
    analytic_cost = pullback_output_cost(model, quad_cost)
    ad_cost = OutputCost(model, quad_cost)

    rng = np.random.default_rng(42)
    for _ in range(5):
        x_pt = jnp.asarray(rng.standard_normal(3))
        u_pt = jnp.asarray(rng.standard_normal(2))

        val_analytic = analytic_cost.evaluate(x_pt, u_pt)
        val_ad = ad_cost.evaluate(x_pt, u_pt)
        np.testing.assert_allclose(val_analytic, val_ad, atol=1e-12)

        grad_analytic = analytic_cost.gradient(x_pt, u_pt)
        grad_ad = ad_cost.gradient(x_pt, u_pt)
        np.testing.assert_allclose(grad_analytic, grad_ad, atol=1e-12)

        hess_analytic = analytic_cost.hessian(x_pt, u_pt)
        hess_ad = ad_cost.hessian(x_pt, u_pt)
        np.testing.assert_allclose(hess_analytic, hess_ad, atol=1e-12)

    quad_term = QuadraticCost(Q=Q_y, q=q_y, c=c_val, terminal=True, m=2)
    analytic_term = pullback_output_cost(model, quad_term)
    ad_term = OutputCost(model, quad_term)

    for _ in range(5):
        x_pt = jnp.asarray(rng.standard_normal(3))
        np.testing.assert_allclose(analytic_term.evaluate(x_pt, None), ad_term.evaluate(x_pt, None), atol=1e-12)
        np.testing.assert_allclose(analytic_term.gradient(x_pt, None), ad_term.gradient(x_pt, None), atol=1e-12)
        np.testing.assert_allclose(analytic_term.hessian(x_pt, None), ad_term.hessian(x_pt, None), atol=1e-12)


def test_output_constraint_with_state_bound() -> None:
    """Verify OutputConstraint evaluate and Jacobians wrapping a StateBound."""
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    C = jnp.array([[1.0, 0.0], [0.5, 0.5]])
    dy = jnp.array([0.1, -0.1])
    model = AffineModel(A=A, B=B, C=C, dy=dy)

    bound = StateBound(n=2, x_min=[-1.0, -0.5], x_max=[1.0, 0.5])
    out_constraint = OutputConstraint(model, bound)

    assert out_constraint.n == 2
    assert out_constraint.m == 1
    assert out_constraint.p == bound.p == 4
    assert not out_constraint.uses_control()

    x = jnp.array([0.2, 0.4])
    u = jnp.array([1.0])

    y = model.output(x, u)
    np.testing.assert_allclose(out_constraint.evaluate(x, u), bound.evaluate(y, None))

    jx = out_constraint.jacobian_x(x, u)
    jx_ad = jax.jacobian(lambda x_: out_constraint.evaluate(x_, u))(x)
    np.testing.assert_allclose(jx, jx_ad, atol=1e-12)

    ju = out_constraint.jacobian_u(x, u)
    np.testing.assert_allclose(ju, jnp.zeros((4, 1)), atol=1e-12)


def test_solve_problem_with_output_cost_and_constraint() -> None:
    """Solve an optimal control problem with output cost and output constraint via Ipopt and ALTRO."""
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    C = jnp.array([[1.0, 0.0]])
    model = AffineModel(A=A, B=B, C=C)

    N = 15
    dt = 0.1
    x0 = jnp.array([0.0, 0.0])

    # Target position y = 1.0, penalize control effort
    q_output_stage = QuadraticCost(
        Q=jnp.array([[10.0]]),
        R=jnp.array([[0.01]]),
        q=jnp.array([-10.0]),
        terminal=False,
    )
    q_output_term = QuadraticCost(
        Q=jnp.array([[100.0]]),
        q=jnp.array([-100.0]),
        terminal=True,
        m=1,
    )

    constraints = ConstraintList(n=2, m=1, N=N)
    constraints.add_constraint(ControlBound(n=2, m=1, u_min=[-5.0], u_max=[5.0]), range(N - 1))
    out_bound = OutputConstraint(model, StateBound(n=1, x_min=[-0.1], x_max=[1.2]))
    constraints.add_constraint(out_bound, range(N))

    # Solve with OutputCost via Ipopt
    obj_ad = Objective(
        stage_cost=OutputCost(model, q_output_stage),
        terminal_cost=OutputCost(model, q_output_term),
        N=N,
    )
    prob_ipopt = Problem(model=model, obj=obj_ad, constraints=constraints, N=N, dt=dt)
    res_ipopt = MPC(prob_ipopt, Ipopt(options={"print_level": 0, "tol": 1e-6}), x0=x0).solve()
    assert res_ipopt.success
    Y_opt = model.evaluate_output(res_ipopt.trajectory)
    assert np.all(np.asarray(Y_opt) <= 1.2 + 1e-4)
    assert np.all(np.asarray(Y_opt) >= -0.1 - 1e-4)
    np.testing.assert_allclose(Y_opt[-1, 0], 1.0, atol=0.05)

    # Solve with pullback_output_cost via ALTRO
    stage_pb = pullback_output_cost(model, q_output_stage)
    term_pb = pullback_output_cost(model, q_output_term)
    obj_pb = Objective(stage_cost=stage_pb, terminal_cost=term_pb, N=N)
    prob_altro = Problem(model=model, obj=obj_pb, constraints=constraints, N=N, dt=dt)
    res_altro = MPC(prob_altro, ALTRO(), x0=x0).solve()
    assert res_altro.success
    Y_altro = model.evaluate_output(res_altro.trajectory)
    assert np.all(np.asarray(Y_altro) <= 1.2 + 1e-3)
    assert np.all(np.asarray(Y_altro) >= -0.1 - 1e-3)
    np.testing.assert_allclose(Y_altro[-1, 0], 1.0, atol=0.05)


def test_simulation_trajopt_measurement_and_dynamics() -> None:
    """Verify TrajOptMeasurement and TrajOptDynamics output in simulation bridge."""
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    C = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    D = jnp.array([[0.0], [0.5]])
    dy = jnp.array([0.1, -0.2])
    model = AffineModel(A=A, B=B, C=C, D=D, dy=dy)

    plant = TrajOptDynamics(dt=0.1, model=model, x0=np.array([1.0, 2.0]))
    u = np.array([3.0])

    y_plant = plant.output(0.0, plant.x, u)
    assert isinstance(y_plant, np.ndarray)
    assert y_plant.dtype == np.float64
    np.testing.assert_allclose(y_plant, [1.0 + 0.1, 2.0 + 1.5 - 0.2])

    measurement = TrajOptMeasurement(model)
    y_meas = measurement(0.0, plant.x, u)
    assert isinstance(y_meas, np.ndarray)
    np.testing.assert_allclose(y_meas, y_plant)

    sensor = GaussianSensor(dt=0.1, measurement=measurement, std_dev=0.0)
    y_sensor, _ = sensor.update(0.0, plant.x, u)
    np.testing.assert_allclose(y_sensor, y_plant)

    cfg = {"model": {"class_path": "trajopt.models.pendulum.Pendulum"}}
    meas_from_cfg = TrajOptMeasurement.from_config(cfg)
    assert isinstance(meas_from_cfg, TrajOptMeasurement)


def test_output_constraint_feedthrough_and_dimension_validation() -> None:
    """Verify OutputConstraint feedthrough detection and control dimension validation."""
    A = jnp.eye(2)
    B = jnp.ones((2, 1))
    C = jnp.eye(2)
    D_zero = jnp.zeros((2, 1))
    D_feed = jnp.ones((2, 1))

    model_no_feed = AffineModel(A=A, B=B, C=C, D=D_zero)
    model_feed = AffineModel(A=A, B=B, C=C, D=D_feed)

    state_bound = StateBound(n=2, x_min=[-1.0, -1.0], x_max=[1.0, 1.0])
    con_no_feed = OutputConstraint(model_no_feed, state_bound)
    con_feed = OutputConstraint(model_feed, state_bound)

    assert not con_no_feed.uses_control()
    assert con_feed.uses_control()

    con_override = OutputConstraint(model_no_feed, state_bound, has_feedthrough=True)
    assert con_override.uses_control()

    mismatched_con = ControlBound(n=2, m=3, u_min=[-1.0, -1.0, -1.0], u_max=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"Inner constraint m.*must match model control dimension"):
        OutputConstraint(model_no_feed, mismatched_con)


def test_output_cost_control_dimension_validation() -> None:
    """Verify OutputCost and pullback_output_cost reject mismatched control dimensions."""
    A = jnp.eye(2)
    B = jnp.ones((2, 1))
    C = jnp.eye(2)
    model = AffineModel(A=A, B=B, C=C)

    # Cost with m=2 when model has m=1
    mismatched_quad = QuadraticCost(
        Q=jnp.eye(2),
        R=jnp.eye(2),
        terminal=False,
    )

    with pytest.raises(ValueError, match=r"Cost control dimension.*must be 0 or match"):
        OutputCost(model, mismatched_quad)

    with pytest.raises(ValueError, match=r"Cost control dimension.*must be 0 or match"):
        pullback_output_cost(model, mismatched_quad)
