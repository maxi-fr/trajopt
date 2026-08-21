"""Unit tests for constraint catalog and fused ConstraintList."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, SecondOrderCone, ZeroCone
from trajopt.constraints import (
    BoundConstraint,
    CircleConstraint,
    CollisionConstraint,
    ConstraintList,
    ControlBound,
    ControlConstraint,
    DynamicsConstraint,
    GoalConstraint,
    ImplicitDynamicsConstraint,
    IndexedConstraint,
    LinearConstraint,
    NormConstraint,
    QuatVecEq,
    SphereConstraint,
    StageConstraint,
    StateBound,
    StateConstraint,
)
from trajopt.dynamics import RK4
from trajopt.models import Cartpole


def test_base_constraint_classes() -> None:
    """Test StateConstraint, ControlConstraint, and StageConstraint Jacobians."""
    n, m = 4, 2

    # 1. StateConstraint: control Jacobian block is implied zeros
    class CustomStateCon(StateConstraint):
        def evaluate(
            self,
            x: jax.Array | None = None,
            u: jax.Array | None = None,
            t: float | jax.Array = 0.0,
        ) -> jax.Array:
            del u, t
            assert x is not None
            return jnp.array([x[0] ** 2 + x[1], x[2] * x[3]])

    scon = CustomStateCon(n=n, m=m, p=2, cone=ZeroCone())
    assert scon.n == n
    assert scon.m == m
    assert scon.p == 2
    assert isinstance(scon.cone, ZeroCone)

    x = jnp.array([1.5, 2.0, -1.0, 3.0])
    u = jnp.array([0.5, -0.5])
    c_val = scon.evaluate(x, u)
    np.testing.assert_allclose(c_val, np.array([1.5**2 + 2.0, -1.0 * 3.0]))

    jx, ju = scon.jacobian(x, u)
    assert jx.shape == (2, n)
    assert ju.shape == (2, m)
    np.testing.assert_allclose(ju, np.zeros((2, m)))
    expected_jx = np.array(
        [
            [2 * 1.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, -1.0],
        ]
    )
    np.testing.assert_allclose(jx, expected_jx, atol=1e-12)

    # 2. ControlConstraint: state Jacobian block is implied zeros
    class CustomControlCon(ControlConstraint):
        def evaluate(
            self,
            x: jax.Array | None = None,
            u: jax.Array | None = None,
            t: float | jax.Array = 0.0,
        ) -> jax.Array:
            del x, t
            assert u is not None
            return jnp.array([u[0] ** 2 + u[1] ** 2])

    ccon = CustomControlCon(n=n, m=m, p=1, cone=NegativeOrthant())
    assert ccon.n == n
    assert ccon.m == m
    assert ccon.p == 1
    assert isinstance(ccon.cone, NegativeOrthant)

    jx_c, ju_c = ccon.jacobian(x, u)
    assert jx_c.shape == (1, n)
    assert ju_c.shape == (1, m)
    np.testing.assert_allclose(jx_c, np.zeros((1, n)))
    expected_ju = np.array([[2 * 0.5, 2 * (-0.5)]])
    np.testing.assert_allclose(ju_c, expected_ju, atol=1e-12)

    # 3. StageConstraint: both state and control Jacobians are AD derived
    class CustomStageCon(StageConstraint):
        def evaluate(
            self,
            x: jax.Array | None = None,
            u: jax.Array | None = None,
            t: float | jax.Array = 0.0,
        ) -> jax.Array:
            del t
            assert x is not None
            assert u is not None
            return jnp.array([x[0] * u[0] + x[1] * u[1]])

    stgcon = CustomStageCon(n=n, m=m, p=1, cone=ZeroCone())
    jx_stg, ju_stg = stgcon.jacobian(x, u)
    expected_jx_stg = np.array([[0.5, -0.5, 0.0, 0.0]])
    expected_ju_stg = np.array([[1.5, 2.0]])
    np.testing.assert_allclose(jx_stg, expected_jx_stg, atol=1e-12)
    np.testing.assert_allclose(ju_stg, expected_ju_stg, atol=1e-12)


def test_goal_constraint() -> None:
    """Test GoalConstraint on full and partial state."""
    n, m = 4, 2
    xf_full = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Full state goal
    gcon_full = GoalConstraint(n=n, xf=xf_full, m=m)
    assert gcon_full.p == 4
    assert isinstance(gcon_full.cone, ZeroCone)

    x = jnp.array([1.2, 2.0, 2.8, 4.1])
    val = gcon_full.evaluate(x)
    np.testing.assert_allclose(val, np.array([0.2, 0.0, -0.2, 0.1]), atol=1e-14)

    jx, ju = gcon_full.jacobian(x)
    np.testing.assert_allclose(jx, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(ju, np.zeros((4, m)), atol=1e-14)

    # Partial state goal (indices 0 and 2)
    xf_partial = jnp.array([1.0, 3.0])
    gcon_part = GoalConstraint(n=n, xf=xf_partial, inds=(0, 2), m=m)
    assert gcon_part.p == 2
    val_part = gcon_part.evaluate(x)
    np.testing.assert_allclose(val_part, np.array([0.2, -0.2]), atol=1e-14)

    jx_part, ju_part = gcon_part.jacobian(x)
    expected_jx_part = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(jx_part, expected_jx_part, atol=1e-14)
    np.testing.assert_allclose(ju_part, np.zeros((2, m)), atol=1e-14)


def test_state_and_control_bounds() -> None:
    """Test StateBound, ControlBound, and BoundConstraint."""
    n, m = 3, 2

    # 1. StateBound with mixed finite and infinite bounds
    x_min = jnp.array([-2.0, -np.inf, 0.0])
    x_max = jnp.array([2.0, 5.0, np.inf])
    sbnd = StateBound(n=n, x_min=x_min, x_max=x_max, m=m)
    assert sbnd.p == 4
    assert isinstance(sbnd.cone, NegativeOrthant)

    x = jnp.array([1.0, 6.0, -0.5])
    val = sbnd.evaluate(x)
    np.testing.assert_allclose(val, np.array([-1.0, 1.0, -3.0, 0.5]), atol=1e-14)

    jx, ju = sbnd.jacobian(x)
    expected_jx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    np.testing.assert_allclose(jx, expected_jx, atol=1e-14)
    np.testing.assert_allclose(ju, np.zeros((4, m)), atol=1e-14)

    # primal_bounds path
    zL_x, zU_x = sbnd.primal_bounds()
    np.testing.assert_allclose(zL_x, np.array([-2.0, -np.inf, 0.0]))
    np.testing.assert_allclose(zU_x, np.array([2.0, 5.0, np.inf]))

    # 2. ControlBound
    u_min = jnp.array([-1.0, -2.0])
    u_max = jnp.array([1.0, 2.0])
    cbnd = ControlBound(m=m, u_min=u_min, u_max=u_max, n=n)
    assert cbnd.p == 4
    u = jnp.array([1.5, -3.0])
    val_u = cbnd.evaluate(u=u)
    np.testing.assert_allclose(val_u, np.array([0.5, -5.0, -2.5, 1.0]), atol=1e-14)
    jx_u, ju_u = cbnd.jacobian(u=u)
    np.testing.assert_allclose(jx_u, np.zeros((4, n)))
    expected_ju = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    np.testing.assert_allclose(ju_u, expected_ju, atol=1e-14)

    # 3. BoundConstraint (combined)
    bbnd = BoundConstraint(n=n, m=m, x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max)
    assert bbnd.p == 8
    val_b = bbnd.evaluate(x, u)
    assert val_b.shape == (8,)
    jx_b, ju_b = bbnd.jacobian(x, u)
    assert jx_b.shape == (8, n)
    assert ju_b.shape == (8, m)
    zL_full, zU_full = bbnd.primal_bounds()
    assert len(zL_full) == n + m
    assert len(zU_full) == n + m


def test_linear_constraint() -> None:
    """Test LinearConstraint for inequality and equality forms."""
    n, m = 3, 2
    A = jnp.array(
        [
            [1.0, 0.0, -1.0, 0.5, 0.0],
            [0.0, 2.0, 0.0, -1.0, 1.0],
        ]
    )
    b = jnp.array([0.5, -0.2])

    lcon_ineq = LinearConstraint(n=n, m=m, A=A, b=b, sense=NegativeOrthant())
    assert lcon_ineq.p == 2
    assert isinstance(lcon_ineq.cone, NegativeOrthant)

    x = jnp.array([1.0, 0.5, -0.5])
    u = jnp.array([2.0, -1.0])
    val = lcon_ineq.evaluate(x, u)
    np.testing.assert_allclose(val, np.array([2.0, -1.8]), atol=1e-14)

    jx, ju = lcon_ineq.jacobian(x, u)
    np.testing.assert_allclose(jx, np.array(A[:, :n]), atol=1e-14)
    np.testing.assert_allclose(ju, np.array(A[:, n:]), atol=1e-14)


def test_circle_and_sphere_constraints() -> None:
    """Test CircleConstraint and SphereConstraint obstacle avoidance."""
    n, m = 4, 1

    # 1. CircleConstraint with 2 obstacles
    xc = jnp.array([1.0, 3.0])
    yc = jnp.array([2.0, 4.0])
    radius = jnp.array([0.5, 1.0])

    circle_con = CircleConstraint(n=n, xc=xc, yc=yc, radius=radius, xi=0, yi=1, m=m)
    assert circle_con.p == 2
    assert isinstance(circle_con.cone, NegativeOrthant)

    x = jnp.array([1.5, 2.0, 0.0, 0.0])
    val = circle_con.evaluate(x)
    np.testing.assert_allclose(val, np.array([0.0, -5.25]), atol=1e-14)

    jx, ju = circle_con.jacobian(x)
    expected_jx = np.array(
        [
            [-2 * (1.5 - 1.0), -2 * (2.0 - 2.0), 0.0, 0.0],
            [-2 * (1.5 - 3.0), -2 * (2.0 - 4.0), 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(jx, expected_jx, atol=1e-12)
    np.testing.assert_allclose(ju, np.zeros((2, m)), atol=1e-14)

    # 2. SphereConstraint with 1 obstacle
    zc = jnp.array([1.0])
    sphere_con = SphereConstraint(
        n=n,
        xc=jnp.array([1.0]),
        yc=jnp.array([2.0]),
        zc=zc,
        radius=jnp.array([0.5]),
        xi=0,
        yi=1,
        zi=2,
        m=m,
    )
    assert sphere_con.p == 1
    x_sph = jnp.array([1.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(sphere_con.evaluate(x_sph), np.array([0.25]), atol=1e-14)
    jx_sph, _ = sphere_con.jacobian(x_sph)
    np.testing.assert_allclose(jx_sph, np.zeros((1, 4)), atol=1e-12)


def test_collision_constraint() -> None:
    """Test pairwise collision avoidance constraint."""
    n, m = 6, 2
    col_con = CollisionConstraint(n=n, x1=(0, 1), x2=(2, 3), radius=1.0, m=m)
    assert col_con.p == 1
    assert isinstance(col_con.cone, NegativeOrthant)

    x = jnp.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    val = col_con.evaluate(x)
    np.testing.assert_allclose(val, np.array([0.75]), atol=1e-14)

    jx, ju = col_con.jacobian(x)
    expected_jx = np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    np.testing.assert_allclose(jx, expected_jx, atol=1e-12)
    np.testing.assert_allclose(ju, np.zeros((1, m)), atol=1e-14)


def test_norm_constraint() -> None:
    """Test NormConstraint in quadratic inequality, equality, and second-order cone forms."""
    n, m = 3, 2

    # 1. Quadratic inequality: ||u||_2^2 - a^2 <= 0
    norm_ineq = NormConstraint(n=n, m=m, val=2.0, sense=NegativeOrthant(), inds="control")
    assert norm_ineq.p == 1
    assert isinstance(norm_ineq.cone, NegativeOrthant)

    x = jnp.zeros(n)
    u = jnp.array([1.0, 1.0])
    np.testing.assert_allclose(norm_ineq.evaluate(x, u), np.array([-2.0]), atol=1e-14)
    jx, ju = norm_ineq.jacobian(x, u)
    np.testing.assert_allclose(jx, np.zeros((1, n)))
    np.testing.assert_allclose(ju, np.array([[2.0, 2.0]]), atol=1e-12)

    # 2. Quadratic equality: ||x||_2^2 - a^2 = 0
    norm_eq = NormConstraint(n=n, m=m, val=3.0, sense=ZeroCone(), inds="state")
    assert norm_eq.p == 1
    assert isinstance(norm_eq.cone, ZeroCone)

    # 3. Second-order cone: [y; a] in K_soc
    norm_soc = NormConstraint(n=n, m=m, val=2.5, sense=SecondOrderCone(), inds=(0, 1))
    assert norm_soc.p == 3
    assert isinstance(norm_soc.cone, SecondOrderCone)
    x_soc = jnp.array([1.5, -2.0, 0.0])
    val_soc = norm_soc.evaluate(x_soc, u)
    np.testing.assert_allclose(val_soc, np.array([1.5, -2.0, 2.5]), atol=1e-14)
    jx_soc, ju_soc = norm_soc.jacobian(x_soc, u)
    expected_jx_soc = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(jx_soc, expected_jx_soc, atol=1e-12)
    np.testing.assert_allclose(ju_soc, np.zeros((3, m)), atol=1e-14)


def test_quat_vec_eq() -> None:
    """Test QuatVecEq attitude equality constraint."""
    n, m = 7, 3
    qf = jnp.array([0.0, 0.0, 0.0, 1.0])
    qcon = QuatVecEq(n=n, qf=qf, qind=(3, 4, 5, 6), m=m)
    assert qcon.p == 3
    assert isinstance(qcon.cone, ZeroCone)

    # Same attitude
    x_same = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(qcon.evaluate(x_same), np.zeros(3), atol=1e-14)

    # Antipodal identical attitude
    x_anti = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(qcon.evaluate(x_anti), np.zeros(3), atol=1e-14)

    # Small perturbation in vector part
    x_pert = jnp.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.0, 0.9746794])
    val_pert = qcon.evaluate(x_pert)
    assert val_pert.shape == (3,)


def test_indexed_constraint() -> None:
    """Test IndexedConstraint subsystem wrapping."""
    n_full, m_full = 8, 4
    norm_sub = NormConstraint(n=3, m=1, val=1.5, sense=NegativeOrthant(), inds="state")
    idx_con = IndexedConstraint(n=n_full, m=m_full, constraint=norm_sub, ix=(2, 3, 4), iu=(1,))
    assert idx_con.p == 1
    assert isinstance(idx_con.cone, NegativeOrthant)

    x_full = jnp.array([10.0, 20.0, 1.0, -1.0, 1.0, 30.0, 40.0, 50.0])
    u_full = jnp.array([5.0, 0.5, 6.0, 7.0])

    val = idx_con.evaluate(x_full, u_full)
    np.testing.assert_allclose(val, np.array([0.75]), atol=1e-14)

    jx, ju = idx_con.jacobian(x_full, u_full)
    assert jx.shape == (1, n_full)
    assert ju.shape == (1, m_full)
    expected_jx = np.array([[0.0, 0.0, 2.0, -2.0, 2.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(jx, expected_jx, atol=1e-12)
    np.testing.assert_allclose(ju, np.zeros((1, m_full)), atol=1e-14)


def test_dynamics_constraints() -> None:
    """Test explicit and implicit dynamics constraints."""
    model = Cartpole()
    n, m = model.n, model.m
    dt = 0.1

    # 1. Explicit dynamics constraint with RK4
    dmodel = RK4(model)
    dyn_con = DynamicsConstraint(model=dmodel)
    assert dyn_con.n == n
    assert dyn_con.m == m
    assert dyn_con.p == n
    assert isinstance(dyn_con.cone, ZeroCone)

    x_k = jnp.array([0.1, 0.2, -0.1, 0.05])
    u_k = jnp.array([1.2])
    x_next = dmodel.discrete_dynamics(x_k, u_k, 0.0, dt)

    # Feasible step evaluates to zero
    val = dyn_con.evaluate(x_k, u_k, t=0.0, x_next=x_next, dt=dt)
    np.testing.assert_allclose(val, np.zeros(n), atol=1e-14)

    # Infeasible step evaluates to defect
    x_bad = x_next + jnp.array([0.01, -0.02, 0.03, -0.04])
    val_bad = dyn_con.evaluate(x_k, u_k, t=0.0, x_next=x_bad, dt=dt)
    np.testing.assert_allclose(val_bad, np.array([0.01, -0.02, 0.03, -0.04]), atol=1e-14)

    # Jacobians: ∇x_k = -A_k, ∇u_k = -B_k, ∇x_{k+1} = I_n
    jx_k, ju_k, jx_next = dyn_con.jacobian(x_k, u_k, t=0.0, x_next=x_next, dt=dt)
    A_k = dmodel.state_jacobian(x_k, u_k, 0.0, dt)
    B_k = dmodel.control_jacobian(x_k, u_k, 0.0, dt)
    np.testing.assert_allclose(jx_k, -A_k, atol=1e-12)
    np.testing.assert_allclose(ju_k, -B_k, atol=1e-12)
    np.testing.assert_allclose(jx_next, np.eye(n), atol=1e-14)

    # 2. Implicit dynamics constraint (implicit midpoint collocation)
    imp_con = ImplicitDynamicsConstraint(model=model)
    assert imp_con.n == n
    assert imp_con.m == m
    assert imp_con.p == n
    assert isinstance(imp_con.cone, ZeroCone)

    val_imp = imp_con.evaluate(x_k, u_k, t=0.0, x_next=x_next, dt=dt)
    assert val_imp.shape == (n,)
    jx_k_imp, ju_k_imp, jx_next_imp = imp_con.jacobian(x_k, u_k, t=0.0, x_next=x_next, dt=dt)
    assert jx_k_imp.shape == (n, n)
    assert ju_k_imp.shape == (n, m)
    assert jx_next_imp.shape == (n, n)


def test_constraint_list_registration_and_dimension_check() -> None:
    """Test ConstraintList registration, bounds checking, and queryable dimensions."""
    n, m, N = 4, 2, 10
    clist = ConstraintList(n=n, m=m, N=N)

    assert clist.n == n
    assert clist.m == m
    assert clist.N == N
    assert len(clist) == 0

    # Add control bound over knot points 0..N-2
    cbnd = ControlBound(m=m, u_min=-jnp.ones(m), u_max=jnp.ones(m), n=n)
    clist.add_constraint(cbnd, inds=range(N - 1))
    assert len(clist) == 1

    # Add goal constraint at terminal knot point N-1
    gcon = GoalConstraint(n=n, xf=jnp.zeros(n), m=m)
    clist.add_constraint(gcon, inds=N - 1)
    assert len(clist) == 2

    # Query total constraint dimension per knot point
    p_arr = clist.num_constraints()
    assert len(p_arr) == N
    assert p_arr[0] == 4
    assert p_arr[N - 2] == 4
    assert p_arr[N - 1] == 4

    # Dimension mismatch checks
    bad_state_con = StateBound(n=n + 1, x_min=-jnp.ones(n + 1), x_max=jnp.ones(n + 1))
    with pytest.raises(ValueError, match="State dimension mismatch"):
        clist.add_constraint(bad_state_con, inds=0)

    with pytest.raises(ValueError, match="Index out of horizon"):
        clist.add_constraint(gcon, inds=N)

    with pytest.raises(ValueError, match="Control constraint cannot be applied at terminal knot"):
        clist.add_constraint(cbnd, inds=N - 1)


def test_deterministic_concatenation_order() -> None:
    """Test that multiple constraints at a single knot point concatenate deterministically."""
    n, m, N = 4, 2, 5
    clist = ConstraintList(n=n, m=m, N=N)

    c1 = CircleConstraint(n=n, xc=jnp.array([1.0, 2.0]), yc=jnp.array([0.0, 0.0]), radius=jnp.array([0.5, 0.5]), m=m)
    A = jnp.ones((3, n + m))
    b = jnp.zeros(3)
    c2 = LinearConstraint(n=n, m=m, A=A, b=b)
    c3 = NormConstraint(n=n, m=m, val=1.0, sense=NegativeOrthant(), inds="control")

    clist.add_constraint(c1, inds=2)
    clist.add_constraint(c2, inds=2)
    clist.add_constraint(c3, inds=2)

    assert clist.num_constraints()[2] == 2 + 3 + 1

    built = clist.build()
    x = jnp.array([1.0, 0.5, -0.5, 2.0])
    u = jnp.array([0.2, -0.3])

    val_fused = built.evaluate_knot(2, x, u)
    val1 = c1.evaluate(x, u)
    val2 = c2.evaluate(x, u)
    val3 = c3.evaluate(x, u)
    expected_val = jnp.concatenate([val1, val2, val3])

    np.testing.assert_allclose(val_fused, expected_val, atol=1e-14)

    # Verify fused Jacobians
    jx_fused, ju_fused = built.jacobian_knot(2, x, u)
    jx1, ju1 = c1.jacobian(x, u)
    jx2, ju2 = c2.jacobian(x, u)
    jx3, ju3 = c3.jacobian(x, u)
    expected_jx = jnp.vstack([jx1, jx2, jx3])
    expected_ju = jnp.vstack([ju1, ju2, ju3])

    np.testing.assert_allclose(jx_fused, expected_jx, atol=1e-12)
    np.testing.assert_allclose(ju_fused, expected_ju, atol=1e-12)


def test_built_constraint_list_batched_evaluation() -> None:
    """Test full horizon batched evaluation of built ConstraintList."""
    n, m, N = 4, 1, 6
    clist = ConstraintList(n=n, m=m, N=N)

    # Obstacle avoidance on stages 1..4
    circle_con = CircleConstraint(
        n=n,
        xc=jnp.array([1.0]),
        yc=jnp.array([0.0]),
        radius=jnp.array([0.5]),
        m=m,
    )
    clist.add_constraint(circle_con, inds=range(1, N - 1))

    # Control bound on stages 0..N-2
    cbnd = ControlBound(m=m, u_min=-3.0 * jnp.ones(m), u_max=3.0 * jnp.ones(m), n=n)
    clist.add_constraint(cbnd, inds=range(N - 1))

    # Goal constraint on terminal stage N-1
    gcon = GoalConstraint(n=n, xf=jnp.array([0.0, np.pi, 0.0, 0.0]), m=m)
    clist.add_constraint(gcon, inds=N - 1)

    built = clist.build()

    X = jnp.zeros((N, n))
    U = jnp.zeros((N - 1, m))

    vals = built.evaluate(X, U)
    assert len(vals) == N
    assert vals[0].shape == (2,)
    assert vals[1].shape == (3,)
    assert vals[N - 1].shape == (4,)

    jacs = built.jacobian(X, U)
    assert len(jacs) == N
    for k in range(N):
        jx, ju = jacs[k]
        pk = built.p[k]
        assert jx.shape == (pk, n)
        if k < N - 1:
            assert ju.shape == (pk, m)
        else:
            assert ju.shape == (pk, 0)
