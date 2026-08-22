from typing import Any

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
    DynamicsConstraint,
    GoalConstraint,
    ImplicitDynamicsConstraint,
    IndexedConstraint,
    LinearConstraint,
    NormConstraint,
    QuatVecEq,
    SphereConstraint,
    StateBound,
)
from trajopt.dynamics import RK4
from trajopt.models import Cartpole


@pytest.mark.julia
def test_cross_goal_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 4, 2
    xf_np = np.array([1.0, 2.5, -0.5, 3.2])
    gcon_py = GoalConstraint(n=n, xf=jnp.array(xf_np), m=m)

    jl_gcon = jl.seval("xf -> TO.GoalConstraint(MVector{4,Float64}(xf...))")(xf_np)

    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.5, -0.5, 3.2]),
        np.array([-1.2, 0.8, 3.4, -2.1]),
    ]

    jl_eval = jl.seval("function (con, x) RD.evaluate(con, SVector{4,Float64}(x...)) end")
    jl_jac = jl.seval("""
    function (con, x)
        J = zeros(4, 4)
        RD.jacobian!(con, J, zeros(4), SVector{4,Float64}(x...))
        J
    end
    """)

    for x_np in test_states:
        x_jax = jnp.array(x_np)
        val_py = np.array(gcon_py.evaluate(x_jax))
        val_jl = np.array(jl_eval(jl_gcon, x_np))
        np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

        jx_py, ju_py = gcon_py.jacobian(x_jax)
        jx_jl = np.array(jl_jac(jl_gcon, x_np))
        np.testing.assert_allclose(np.array(jx_py), jx_jl, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(ju_py), np.zeros((4, m)), rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_state_and_control_bounds(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 3, 2
    x_min = np.array([-2.0, -np.inf, 0.0])
    x_max = np.array([2.0, 5.0, np.inf])

    sbnd_py = StateBound(n=n, x_min=jnp.array(x_min), x_max=jnp.array(x_max), m=m)
    jl_sbnd = jl.seval("function (xmin, xmax) TO.StateBound(3, x_min=xmin, x_max=xmax) end")(x_min, x_max)

    test_states = [
        np.array([0.0, 2.0, 1.0]),
        np.array([-3.0, 6.0, -1.0]),
        np.array([1.5, -4.0, 2.5]),
    ]

    jl_eval_s = jl.seval("function (con, x) RD.evaluate(con, SVector{3,Float64}(x...)) end")
    jl_jac_s = jl.seval("""
    function (con, x)
        p = RD.output_dim(con)
        J = zeros(p, 3)
        RD.jacobian!(con, J, zeros(p), SVector{3,Float64}(x...))
        J
    end
    """)

    for x_np in test_states:
        x_jax = jnp.array(x_np)
        val_py = np.array(sbnd_py.evaluate(x_jax))
        val_jl = np.array(jl_eval_s(jl_sbnd, x_np))
        np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

        jx_py, ju_py = sbnd_py.jacobian(x_jax)
        jx_jl = np.array(jl_jac_s(jl_sbnd, x_np))
        np.testing.assert_allclose(np.array(jx_py), jx_jl, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(ju_py), np.zeros((sbnd_py.p, m)), rtol=1e-12, atol=1e-12)

    # Control bound
    u_min = np.array([-1.5, -3.0])
    u_max = np.array([1.5, 3.0])
    cbnd_py = ControlBound(m=m, u_min=jnp.array(u_min), u_max=jnp.array(u_max), n=n)
    jl_cbnd = jl.seval("function (umin, umax) TO.ControlBound(2, u_min=umin, u_max=umax) end")(u_min, u_max)

    jl_eval_u = jl.seval("function (con, u) RD.evaluate(con, SVector{2,Float64}(u...)) end")
    jl_jac_u = jl.seval("""
    function (con, u)
        p = RD.output_dim(con)
        J = zeros(p, 2)
        RD.jacobian!(con, J, zeros(p), SVector{2,Float64}(u...))
        J
    end
    """)

    test_controls = [
        np.array([0.0, 0.0]),
        np.array([2.0, -4.0]),
        np.array([-1.0, 1.5]),
    ]

    for u_np in test_controls:
        u_jax = jnp.array(u_np)
        val_py = np.array(cbnd_py.evaluate(u=u_jax))
        val_jl = np.array(jl_eval_u(jl_cbnd, u_np))
        np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

        jx_py, ju_py = cbnd_py.jacobian(u=u_jax)
        ju_jl = np.array(jl_jac_u(jl_cbnd, u_np))
        np.testing.assert_allclose(np.array(jx_py), np.zeros((cbnd_py.p, n)), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(ju_py), ju_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_bound_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 3, 2
    x_min = np.array([-2.0, -np.inf, 0.0])
    x_max = np.array([2.0, 5.0, np.inf])
    u_min = np.array([-1.0, -2.0])
    u_max = np.array([1.0, 2.0])

    bbnd_py = BoundConstraint(
        n=n, m=m, x_min=jnp.array(x_min), x_max=jnp.array(x_max), u_min=jnp.array(u_min), u_max=jnp.array(u_max)
    )
    jl_bbnd = jl.seval(
        "function (xmin, xmax, umin, umax) TO.BoundConstraint(3, 2, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax) end"
    )(x_min, x_max, u_min, u_max)

    x_np = np.array([1.0, 6.0, -0.5])
    u_np = np.array([1.5, -3.0])

    jl_eval = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        RD.evaluate(con, z)
    end
    """)
    jl_jac = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        p = RD.output_dim(con)
        J = zeros(p, 5)
        RD.jacobian!(con, J, zeros(p), z)
        J
    end
    """)

    val_py = np.array(bbnd_py.evaluate(jnp.array(x_np), jnp.array(u_np)))
    val_jl = np.array(jl_eval(jl_bbnd, x_np, u_np))
    np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

    jx_py, ju_py = bbnd_py.jacobian(jnp.array(x_np), jnp.array(u_np))
    J_jl = np.array(jl_jac(jl_bbnd, x_np, u_np))
    np.testing.assert_allclose(np.array(jx_py), J_jl[:, :n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(ju_py), J_jl[:, n:], rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_circle_and_sphere_constraints(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 4, 2
    xc = np.array([1.0, 3.0])
    yc = np.array([2.0, -1.0])
    radius = np.array([0.8, 1.2])

    circ_py = CircleConstraint(n=n, xc=jnp.array(xc), yc=jnp.array(yc), radius=jnp.array(radius), xi=0, yi=1, m=m)
    jl_circ = jl.seval("""
    function (n, xc, yc, r)
        TO.CircleConstraint(n, SVector{2,Float64}(xc...), SVector{2,Float64}(yc...), SVector{2,Float64}(r...), 1, 2)
    end
    """)(n, xc, yc, radius)

    test_states = [
        np.array([0.0, 0.0, 1.0, -1.0]),
        np.array([1.5, 2.2, 0.0, 0.5]),
        np.array([2.8, -0.5, -2.0, 3.0]),
    ]

    jl_eval = jl.seval("function (con, x) RD.evaluate(con, SVector{4,Float64}(x...)) end")
    jl_jac = jl.seval("""
    function (con, x)
        z = TO.KnotPoint(SVector{4,Float64}(x...), SVector{2,Float64}(0.0, 0.0), 0.0, 0.1)
        J = zeros(2, 6)
        RD.jacobian!(con, J, zeros(2), z)
        J
    end
    """)

    for x_np in test_states:
        x_jax = jnp.array(x_np)
        val_py = np.array(circ_py.evaluate(x_jax))
        val_jl = np.array(jl_eval(jl_circ, x_np))
        np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

        jx_py, ju_py = circ_py.jacobian(x_jax)
        J_jl = np.array(jl_jac(jl_circ, x_np))
        np.testing.assert_allclose(np.array(jx_py), J_jl[:, :n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.array(ju_py), np.zeros((2, m)), rtol=1e-12, atol=1e-12)

    # Sphere constraint
    zc = np.array([0.5, 1.5])
    sph_py = SphereConstraint(
        n=n,
        xc=jnp.array(xc),
        yc=jnp.array(yc),
        zc=jnp.array(zc),
        radius=jnp.array(radius),
        xi=0,
        yi=1,
        zi=2,
        m=m,
    )
    jl_sph = jl.seval("""
    function (n, xc, yc, zc, r)
        TO.SphereConstraint(n, SVector{2,Float64}(xc...), SVector{2,Float64}(yc...), SVector{2,Float64}(zc...), SVector{2,Float64}(r...), 1, 2, 3)
    end
    """)(n, xc, yc, zc, radius)

    jl_eval_sph = jl.seval("function (con, x) RD.evaluate(con, SVector{4,Float64}(x...)) end")
    jl_jac_sph = jl.seval("""
    function (con, x)
        z = TO.KnotPoint(SVector{4,Float64}(x...), SVector{2,Float64}(0.0, 0.0), 0.0, 0.1)
        J = zeros(2, 6)
        RD.jacobian!(con, J, zeros(2), z)
        J
    end
    """)

    for x_np in test_states:
        x_jax = jnp.array(x_np)
        val_py = np.array(sph_py.evaluate(x_jax))
        val_jl = np.array(jl_eval_sph(jl_sph, x_np))
        np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

        jx_py, ju_py = sph_py.jacobian(x_jax)
        J_jl = np.array(jl_jac_sph(jl_sph, x_np))
        np.testing.assert_allclose(np.array(jx_py), J_jl[:, :n], rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_collision_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 6, 2
    r = 1.5
    col_py = CollisionConstraint(n=n, x1=(0, 1), x2=(2, 3), radius=r, m=m)
    jl_col = jl.seval("""
    function (n, r)
        TO.CollisionConstraint(n, SVector{2,Int}(1, 2), SVector{2,Int}(3, 4), r)
    end
    """)(n, r)

    test_states = [
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        np.array([2.0, -1.0, 0.5, 0.5, 1.0, -1.0]),
        np.array([0.1, 0.2, 0.15, 0.25, -2.0, 3.0]),
    ]

    jl_eval = jl.seval("function (con, x) RD.evaluate(con, SVector{6,Float64}(x...)) end")
    jl_jac = jl.seval("""
    function (con, x)
        J = zeros(1, 6)
        RD.jacobian!(con, J, zeros(1), SVector{6,Float64}(x...))
        J
    end
    """)

    for x_np in test_states:
        x_jax = jnp.array(x_np)
        val_py = np.array(col_py.evaluate(x_jax))
        val_jl = np.array(jl_eval(jl_col, x_np))
        np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

        jx_py, _ = col_py.jacobian(x_jax)
        jx_jl = np.array(jl_jac(jl_col, x_np))
        np.testing.assert_allclose(np.array(jx_py), jx_jl, rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_norm_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 3, 2
    val_a = 2.5

    # 1. Inequality norm constraint on control
    norm_py = NormConstraint(n=n, m=m, val=val_a, sense=NegativeOrthant(), inds="control")
    jl_norm = jl.seval("function (n, m, a) TO.NormConstraint(n, m, a, TO.Inequality(), :control) end")(n, m, val_a)

    jl_eval = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        RD.evaluate(con, z)
    end
    """)
    jl_jac = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        J = zeros(1, 5)
        RD.jacobian!(con, J, zeros(1), z)
        J
    end
    """)

    x_np = np.array([1.0, -1.0, 0.5])
    u_np = np.array([1.2, -0.8])

    val_py = np.array(norm_py.evaluate(jnp.array(x_np), jnp.array(u_np)))
    val_jl = np.array(jl_eval(jl_norm, x_np, u_np))
    np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

    jx_py, ju_py = norm_py.jacobian(jnp.array(x_np), jnp.array(u_np))
    J_jl = np.array(jl_jac(jl_norm, x_np, u_np))
    np.testing.assert_allclose(np.array(jx_py), J_jl[:, :n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(ju_py), J_jl[:, n:], rtol=1e-12, atol=1e-12)

    # 2. SecondOrderCone norm constraint on state
    norm_soc_py = NormConstraint(n=n, m=m, val=val_a, sense=SecondOrderCone(), inds=(0, 1))
    jl_norm_soc = jl.seval("""
    function (n, m, a)
        TO.NormConstraint(n, m, a, TO.SecondOrderCone(), SVector{2,Int}(1, 2))
    end
    """)(n, m, val_a)

    jl_eval_soc = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        RD.evaluate(con, z)
    end
    """)
    jl_jac_soc = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        p = RD.output_dim(con)
        J = zeros(p, 5)
        RD.jacobian!(con, J, zeros(p), z)
        J
    end
    """)

    val_soc_py = np.array(norm_soc_py.evaluate(jnp.array(x_np), jnp.array(u_np)))
    val_soc_jl = np.array(jl_eval_soc(jl_norm_soc, x_np, u_np))
    np.testing.assert_allclose(val_soc_py, val_soc_jl, rtol=1e-12, atol=1e-12)

    jx_soc_py, ju_soc_py = norm_soc_py.jacobian(jnp.array(x_np), jnp.array(u_np))
    J_soc_jl = np.array(jl_jac_soc(jl_norm_soc, x_np, u_np))
    np.testing.assert_allclose(np.array(jx_soc_py), J_soc_jl[:, :n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(ju_soc_py), J_soc_jl[:, n:], rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_linear_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 3, 2
    A = np.array(
        [
            [1.0, 0.0, -1.0, 0.5, 0.0],
            [0.0, 2.0, 0.0, -1.0, 1.0],
        ]
    )
    b = np.array([0.5, -0.2])

    lcon_py = LinearConstraint(n=n, m=m, A=jnp.array(A), b=jnp.array(b), sense=NegativeOrthant())
    jl_lcon = jl.seval(
        "function (A, b) TO.LinearConstraint(3, 2, Matrix{Float64}(A), Vector{Float64}(b), TO.Inequality()) end"
    )(A, b)

    x_np = np.array([1.0, 0.5, -0.5])
    u_np = np.array([2.0, -1.0])

    jl_eval = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{3,Float64}(x...), SVector{2,Float64}(u...), 0.0, 0.1)
        RD.evaluate(con, z)
    end
    """)

    val_py = np.array(lcon_py.evaluate(jnp.array(x_np), jnp.array(u_np)))
    val_jl = np.array(jl_eval(jl_lcon, x_np, u_np))
    np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

    jx_py, ju_py = lcon_py.jacobian(jnp.array(x_np), jnp.array(u_np))
    np.testing.assert_allclose(np.array(jx_py), A[:, :n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(ju_py), A[:, n:], rtol=1e-12, atol=1e-12)


@pytest.mark.julia
def test_cross_indexed_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 6, 3
    # Inner constraint on 2 state dims and 1 control dim
    norm_sub = NormConstraint(n=2, m=1, val=2.0, sense=NegativeOrthant(), inds="state")
    idx_py = IndexedConstraint(n=n, m=m, constraint=norm_sub, ix=(1, 2), iu=(0,))

    jl_norm_sub = jl.seval("TO.NormConstraint(2, 1, 2.0, TO.Inequality(), :state)")
    jl_idx = jl.seval("function (inner) TO.IndexedConstraint(6, 3, inner, 2:3, 1:1) end")(jl_norm_sub)

    x_np = np.array([0.5, 1.2, -0.3, 0.8, 2.1, -1.5])
    u_np = np.array([0.4, -0.2, 1.0])

    jl_eval = jl.seval("""
    function (con, x, u)
        z = TO.KnotPoint(SVector{6,Float64}(x...), SVector{3,Float64}(u...), 0.0, 0.1)
        RD.evaluate(con, z)
    end
    """)

    val_py = np.array(idx_py.evaluate(jnp.array(x_np), jnp.array(u_np)))
    val_jl = np.array(jl_eval(jl_idx, x_np, u_np))
    np.testing.assert_allclose(val_py, val_jl, rtol=1e-12, atol=1e-12)

    jx_py, ju_py = idx_py.jacobian(jnp.array(x_np), jnp.array(u_np))
    assert jx_py.shape == (1, n)
    assert ju_py.shape == (1, m)


@pytest.mark.julia
def test_cross_quat_vec_eq(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m = 7, 3
    qf = np.array([0.0, 0.0, 0.0, 1.0])
    qcon_py = QuatVecEq(n=n, qf=jnp.array(qf), qind=(3, 4, 5, 6), m=m)

    x_np = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    val_py = np.array(qcon_py.evaluate(jnp.array(x_np)))
    np.testing.assert_allclose(val_py, np.zeros(3), atol=1e-14)

    # Inverted sign quaternion (antipodal)
    x_anti = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    val_anti = np.array(qcon_py.evaluate(jnp.array(x_anti)))
    np.testing.assert_allclose(val_anti, np.zeros(3), atol=1e-14)


@pytest.mark.julia
def test_cross_dynamics_constraint(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, RobotZoo, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    model = Cartpole()
    dmodel = RK4(model)
    dyn_py = DynamicsConstraint(model=dmodel)

    jl_dmodel = jl.seval("RD.DiscretizedDynamics{RD.RK4}(RobotZoo.Cartpole())")

    x_np = np.array([0.1, 0.2, -0.1, 0.05])
    u_np = np.array([1.2])
    dt = 0.1

    jl_disc = jl.seval(
        "function (m, x, u, dt) z = TO.KnotPoint(SVector{4,Float64}(x...), SVector{1,Float64}(u...), 0.0, dt); RD.discrete_dynamics(m, z) end"
    )
    x_next_jl = np.array(jl_disc(jl_dmodel, x_np, u_np, dt))

    # Python discrete dynamics defect
    x_next_jax = jnp.array(x_next_jl)
    val_py = np.array(dyn_py.evaluate(jnp.array(x_np), jnp.array(u_np), x_next_jax, t=0.0, dt=dt))
    np.testing.assert_allclose(val_py, np.zeros(4), atol=1e-12)

    # Infeasible defect
    x_bad = x_next_jl + np.array([0.05, -0.05, 0.02, -0.01])
    val_bad_py = np.array(dyn_py.evaluate(jnp.array(x_np), jnp.array(u_np), jnp.array(x_bad), t=0.0, dt=dt))
    np.testing.assert_allclose(val_bad_py, np.array([0.05, -0.05, 0.02, -0.01]), atol=1e-12)

    # Implicit dynamics constraint
    imp_py = ImplicitDynamicsConstraint(model=model)
    val_imp = imp_py.evaluate(jnp.array(x_np), jnp.array(u_np), x_next_jax, t=0.0, dt=dt)
    assert val_imp.shape == (4,)


@pytest.mark.julia
def test_cross_constraint_list(jl_to: Any) -> None:
    jl = jl_to
    jl.seval(
        "using TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra; const TO = TrajectoryOptimization; const RD = RobotDynamics"
    )

    n, m, N = 4, 1, 5
    clist_py = ConstraintList(n=n, m=m, N=N)
    jl_clist = jl.seval("TO.ConstraintList(4, 1, 5)")

    # 1. Control bound
    u_min, u_max = -2.5, 2.5
    cbnd_py = ControlBound(m=m, u_min=jnp.array([u_min]), u_max=jnp.array([u_max]), n=n)
    clist_py.add_constraint(cbnd_py, inds=range(N - 1))
    jl.seval(
        "function (clist, umin, umax, N) TO.add_constraint!(clist, TO.BoundConstraint(4, 1, u_min=umin, u_max=umax), 1:N-1) end"
    )(jl_clist, u_min, u_max, N)

    # 2. Goal constraint
    xf = np.array([0.0, np.pi, 0.0, 0.0])
    gcon_py = GoalConstraint(n=n, xf=jnp.array(xf), m=m)
    clist_py.add_constraint(gcon_py, inds=N - 1)
    jl.seval(
        "function (clist, xf, N) TO.add_constraint!(clist, TO.GoalConstraint(MVector{4,Float64}(xf...)), N:N) end"
    )(jl_clist, xf, N)

    p_py = clist_py.num_constraints()
    p_jl = np.array(jl.seval("clist -> TO.num_constraints(clist)")(jl_clist))
    np.testing.assert_allclose(p_py, p_jl)
