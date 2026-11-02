from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.cones import NegativeOrthant, ZeroCone
from trajopt.constraints.base import Constraint, ControlConstraint, StateConstraint
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.indexed import IndexedConstraint
from trajopt.constraints.linear import LinearConstraint
from trajopt.constraints.output import OutputConstraint
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.base import ContinuousDynamics
from trajopt.dynamics.integrators import RK4
from trajopt.models.affine import AffineModel
from trajopt.mpc import MPC
from trajopt.problem import Problem, add_l1_epigraph
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.layout import (
    nlp_constraint_bounds,
    nlp_constraint_row_count,
    retained_knot_evaluators,
)
from trajopt.transcription.result import constraint_row_count
from trajopt.transcription.single_shooting import (
    SingleShooting,
    _cost_fn,
    eval_g,
    eval_h,
    eval_jac_g,
    single_shooting_dimensions,
)


class LinearContinuous2D(ContinuousDynamics):
    """Linear continuous test model on 2D state and 3D control."""

    def __init__(self) -> None:
        super().__init__(n=2, m=3, ne=2, p=3)

    def dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate xdot = A x + B u."""
        del t
        A = jnp.array([[0.0, 1.0], [-1.0, -0.5]])
        B = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
        return A @ x + B @ u

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate output y = u."""
        del x, t
        return jnp.zeros(3) if u is None else u

    def has_control_feedthrough(self) -> bool:
        """Model output directly maps control."""
        return True

    def output_uses_state(self) -> bool:
        """Model output does not depend on state."""
        return False

    def output_is_time_independent(self) -> bool:
        """Model output has no timestamp dependence."""
        return True


class CustomTimeDependentControl(ControlConstraint):
    """Custom control constraint that reads timestamp t."""

    def __init__(self, m: int) -> None:
        super().__init__(n=2, m=m, p=1, cone=NegativeOrthant())

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate u[0] - t."""
        del x
        u_val = jnp.zeros(self.m) if u is None else u
        return jnp.array([u_val[0] - t])


class CustomTimeIndependentControl(ControlConstraint):
    """Custom control constraint declaring time-invariance."""

    def __init__(self, m: int) -> None:
        super().__init__(n=2, m=m, p=1, cone=NegativeOrthant())

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate u[0] - 1.0."""
        del x, t
        u_val = jnp.zeros(self.m) if u is None else u
        return jnp.array([u_val[0] - 1.0])

    def is_time_independent(self) -> bool:
        """Declare constraint is time-invariant."""
        return True


def _make_prob(
    *,
    N: int = 13,
    b: int = 6,
    m: int = 3,
    n: int = 2,
    constraints: Sequence[tuple[Constraint, Sequence[int]]] = (),
    horizon_constraints: Sequence[LinearHorizonConstraint] = (),
) -> Problem:
    """Construct deterministic test Problem."""
    model = LinearContinuous2D() if m == 3 else AffineModel(A=jnp.eye(n), B=jnp.ones((n, m)))
    Q = jnp.eye(n)
    R = jnp.eye(m)
    obj = LQRObjective(Q=Q, R=R, Qf=Q, N=N)

    cl = ConstraintList(n=n, m=m, N=N)
    for con, inds in constraints:
        cl.add_constraint(con, inds)
    for h_con in horizon_constraints:
        cl.add_horizon_constraint(h_con)

    return Problem(
        model=model,
        obj=obj,
        constraints=cl,
        N=N,
        dt=0.05,
        integrator=RK4(),
        control_block_steps=b,
    )


def test_linear_constraint_classification() -> None:
    """LinearConstraint accurately classifies state usage and time independence."""
    n, m = 2, 2

    # 1. Pure control constraint: inds in [n, n+m)
    con_u = LinearConstraint(n=n, m=m, A=jnp.array([[1.0, -1.0]]), b=jnp.zeros(1), inds=[2, 3])
    assert not con_u.uses_state()
    assert con_u.uses_control()
    assert con_u.is_time_independent()
    assert con_u.is_time_independent_input_constraint()

    # 2. Mixed state/control constraint
    con_xu = LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 2.0]]), b=jnp.zeros(1), inds=[0, 2])
    assert con_xu.uses_state()
    assert con_xu.uses_control()
    assert not con_xu.is_time_independent_input_constraint()

    # 3. Pure state constraint
    con_x = LinearConstraint(n=n, m=m, A=jnp.array([[1.0, 0.5]]), b=jnp.zeros(1), inds=[0, 1])
    assert con_x.uses_state()
    assert not con_x.uses_control()
    assert not con_x.is_time_independent_input_constraint()

    # 4. Zero column on state coordinate: mathematically does not read state
    con_zero_x = LinearConstraint(
        n=n,
        m=m,
        A=jnp.array([[0.0, 1.0]]),
        b=jnp.zeros(1),
        inds=[0, 2],
    )
    assert not con_zero_x.uses_state()
    assert con_zero_x.uses_control()
    assert con_zero_x.is_time_independent_input_constraint()


def test_indexed_and_output_constraint_classification() -> None:
    """IndexedConstraint and OutputConstraint accurately classify input dependence."""
    n, m = 4, 3

    # Inner control constraint on 2 controls
    inner_u = LinearConstraint(n=0, m=2, A=jnp.array([[1.0, 1.0]]), b=jnp.zeros(1), inds=[0, 1])
    indexed_u = IndexedConstraint(n=n, m=m, constraint=inner_u, ix=[], iu=[1, 2])
    assert not indexed_u.uses_state()
    assert indexed_u.uses_control()
    assert indexed_u.is_time_independent_input_constraint()

    # Inner state constraint
    inner_x = LinearConstraint(n=2, m=0, A=jnp.array([[1.0, -1.0]]), b=jnp.zeros(1), inds=[0, 1])
    indexed_x = IndexedConstraint(n=n, m=m, constraint=inner_x, ix=[0, 1], iu=[])
    assert indexed_x.uses_state()
    assert not indexed_x.uses_control()
    assert not indexed_x.is_time_independent_input_constraint()

    # OutputConstraint with AffineModel: C=0, D nonzero
    aff_model_pure_u = AffineModel(
        A=jnp.eye(2),
        B=jnp.ones((2, 2)),
        C=jnp.zeros((2, 2)),
        D=jnp.eye(2),
    )
    out_con_u = OutputConstraint(
        aff_model_pure_u,
        LinearConstraint(n=2, m=0, A=jnp.eye(2), b=jnp.zeros(2), inds=[0, 1]),
    )
    assert not out_con_u.uses_state()
    assert out_con_u.uses_control()
    assert out_con_u.is_time_independent_input_constraint()

    # OutputConstraint with C nonzero
    aff_model_with_x = AffineModel(
        A=jnp.eye(2),
        B=jnp.ones((2, 2)),
        C=jnp.eye(2),
        D=jnp.eye(2),
    )
    out_con_x = OutputConstraint(
        aff_model_with_x,
        LinearConstraint(n=2, m=0, A=jnp.eye(2), b=jnp.zeros(2), inds=[0, 1]),
    )
    assert out_con_x.uses_state()
    assert not out_con_x.is_time_independent_input_constraint()


def test_custom_and_epigraph_constraint_classification() -> None:
    """Custom constraints default conservatively and epigraph constraints classify as input-only."""
    # Custom constraint defaults to uses_state() -> True
    class ArbitraryConstraint(Constraint):
        def __init__(self, n: int, m: int, p: int) -> None:
            super().__init__(n=n, m=m, p=p, cone=NegativeOrthant())

        def evaluate(
            self,
            x: jax.Array | None = None,
            u: jax.Array | None = None,
            t: float | jax.Array = 0.0,
        ) -> jax.Array:
            del x, t
            u_val = jnp.zeros(1) if u is None else u
            return u_val[:1]

    custom = ArbitraryConstraint(n=2, m=1, p=1)
    assert custom.uses_state()
    assert not custom.is_time_independent_input_constraint()

    # Custom control constraint without explicit time invariance defaults conservatively
    time_dep = CustomTimeDependentControl(m=3)
    assert not time_dep.uses_state()
    assert not time_dep.is_time_independent_input_constraint()

    # Custom control constraint with explicit time invariance
    time_indep = CustomTimeIndependentControl(m=3)
    assert not time_indep.uses_state()
    assert time_indep.is_time_independent_input_constraint()

    # Epigraph constraint via add_l1_epigraph
    prob = _make_prob(N=7, b=2, m=2, n=2)
    epigraph_prob = add_l1_epigraph(prob, weight=1.0)
    # The epigraph constraint is at index 0 of knot_evaluators[0]
    epigraph_con = epigraph_prob.constraints.knot_evaluators[0].constraints[-1]
    assert not epigraph_con.uses_state()
    assert epigraph_con.uses_control()
    assert epigraph_con.is_time_independent_input_constraint()


def test_kirchhoff_deduplication_in_both_shooting_formulations() -> None:
    """Kirchhoff equality across H Knot Points collapses to B rows without rank deficiency."""
    pytest.importorskip("cyipopt")

    N, b, m, n = 13, 6, 3, 2
    H = N - 1
    B = H // b

    # Kirchhoff equality: sum(u) == 0 registered across all H Knot Points
    A_kirch = jnp.ones((1, m))
    b_kirch = jnp.zeros(1)
    kirchhoff = LinearConstraint(
        n=n,
        m=m,
        A=A_kirch,
        b=b_kirch,
        sense=ZeroCone(),
        inds=range(n, n + m),
    )

    prob = _make_prob(N=N, b=b, m=m, n=n, constraints=[(kirchhoff, range(H))])

    # 1. Check retained Knot Point evaluators: exactly 1 Kirchhoff row per block
    retained = retained_knot_evaluators(prob)
    assert len(retained) == N
    # In each block of 6 intervals: first interval has p=1, other 5 have p=0
    for blk in range(B):
        assert retained[blk * b].p == 1
        for s in range(1, b):
            assert retained[blk * b + s].p == 0
    assert retained[N - 1].p == 0

    # 2. Check NLP constraint row count
    nlp_rows = nlp_constraint_row_count(prob)
    # n (init) + H * n (defects) + B (retained kirchhoff)
    assert nlp_rows == n + H * n + B
    gL, gU = nlp_constraint_bounds(prob)
    assert len(gL) == nlp_rows
    assert len(gU) == nlp_rows

    # 3. Check SingleShooting dimensions
    n_u, p_user = single_shooting_dimensions(prob)
    assert n_u == B * m
    assert p_user == B

    x0 = jnp.array([0.5, -0.2])
    opts = {"tol": 1e-8, "constr_viol_tol": 1e-8, "print_level": 0, "max_iter": 200}

    # 4. Multiple shooting solve
    res_ms = MPC(prob, Ipopt(options=opts), x0=x0).solve()
    assert res_ms.success
    # Native feasibility on all H original native rows
    assert res_ms.constraint_violation <= 1e-6
    # All 12 native controls satisfy Kirchhoff equality exactly
    u_sums = np.sum(np.asarray(res_ms.trajectory.U), axis=1)
    np.testing.assert_allclose(u_sums, 0.0, atol=1e-6)

    # 5. Single shooting solve
    res_ss = MPC(prob, SingleShooting(Ipopt(options=opts), hessian="dense"), x0=x0).solve()
    assert res_ss.success
    assert res_ss.constraint_violation <= 1e-6
    u_sums_ss = np.sum(np.asarray(res_ss.trajectory.U), axis=1)
    np.testing.assert_allclose(u_sums_ss, 0.0, atol=1e-6)

    # Agree on optimal cost
    np.testing.assert_allclose(res_ms.cost, res_ss.cost, atol=1e-5)


def test_first_applicable_occurrence_not_at_block_start() -> None:
    """First applicable occurrence of an input constraint mid-block is retained."""
    N, b, m, n = 13, 6, 3, 2
    H = N - 1
    B = H // b

    # Kirchhoff constraint registered starting at knot 1 (not knot 0)
    kirchhoff = LinearConstraint(
        n=n,
        m=m,
        A=jnp.ones((1, m)),
        b=jnp.zeros(1),
        sense=ZeroCone(),
        inds=range(n, n + m),
    )
    prob = _make_prob(N=N, b=b, m=m, n=n, constraints=[(kirchhoff, range(1, H))])

    retained = retained_knot_evaluators(prob)
    # Block 0: knot 0 has no Kirchhoff; knot 1 is first occurrence and retained
    assert retained[0].p == 0
    assert retained[1].p == 1
    for s in range(2, b):
        assert retained[s].p == 0
    # Block 1: knot 6 is first occurrence in block 1 and retained
    assert retained[6].p == 1
    for s in range(7, 12):
        assert retained[s].p == 0

    assert sum(k.p for k in retained) == B


def test_state_dependent_and_time_dependent_constraints_retained() -> None:
    """State-dependent, mixed, and time-dependent constraints retain every native evaluation."""
    N, b, m, n = 7, 3, 2, 2
    H = N - 1

    # Mixed state/control constraint
    mixed = LinearConstraint(
        n=n,
        m=m,
        A=jnp.array([[1.0, 0.0, 1.0, 0.0]]),
        b=jnp.array([1.0]),
        inds=[0, 1, 2, 3],
    )
    # Time-dependent custom constraint
    time_dep = CustomTimeDependentControl(m=m)

    prob = _make_prob(N=N, b=b, m=m, n=n, constraints=[(mixed, range(H)), (time_dep, range(H))])
    retained = retained_knot_evaluators(prob)

    # Every native stage retains both constraints (p = 1 + 1 = 2)
    for j in range(H):
        assert retained[j].p == 2


def test_linear_horizon_constraint_activation_budget_and_accumulation() -> None:
    """LinearHorizonConstraint accumulates coefficients for repeated block control columns."""
    pytest.importorskip("cyipopt")

    N, b, m, n = 7, 3, 2, 2
    H = N - 1

    # Budget constraint on native control u_0: sum_{j=0}^{5} u_j[0] <= 4.0
    # Canonical indices for u_j[0] are j * (n + m) + n
    u0_inds = [j * (n + m) + n for j in range(H)]
    budget_con = LinearHorizonConstraint(
        A=jnp.ones((1, H)),
        b=jnp.array([4.0]),
        sense=NegativeOrthant(),
        inds=u0_inds,
    )

    prob = _make_prob(N=N, b=b, m=m, n=n, horizon_constraints=[budget_con])

    x0 = jnp.array([0.0, 0.0])
    opts = {"tol": 1e-8, "constr_viol_tol": 1e-8, "print_level": 0, "max_iter": 200}

    # Solve multiple shooting
    res_ms = MPC(prob, Ipopt(options=opts), x0=x0).solve()
    assert res_ms.success
    u_opt_ms = np.asarray(res_ms.trajectory.U)
    # Verify native activation sum <= 4.0
    assert np.sum(u_opt_ms[:, 0]) <= 4.0 + 1e-6

    # Solve single shooting
    res_ss = MPC(prob, SingleShooting(Ipopt(options=opts), hessian="dense"), x0=x0).solve()
    assert res_ss.success
    u_opt_ss = np.asarray(res_ss.trajectory.U)
    assert np.sum(u_opt_ss[:, 0]) <= 4.0 + 1e-6

    np.testing.assert_allclose(res_ms.cost, res_ss.cost, atol=1e-5)


def test_single_shooting_horizon_constraint_finite_differences() -> None:
    """SingleShooting Horizon rows reading state coordinates match finite differences for Jacobian and Hessian."""
    pytest.importorskip("cyipopt")

    N, b, m, n = 5, 2, 1, 2
    H = N - 1
    B = H // b

    # Horizon constraint reading state x_2[0] and control u_0[0]
    # x_2[0] is canonical index 2 * (n + m)
    # u_0[0] is canonical index 0 * (n + m) + n
    idx_x2 = 2 * (n + m)
    idx_u0 = n
    h_con = LinearHorizonConstraint(
        A=jnp.array([[2.0, -1.5]]),
        b=jnp.array([0.5]),
        sense=NegativeOrthant(),
        inds=[idx_x2, idx_u0],
    )

    prob = _make_prob(N=N, b=b, m=m, n=n, horizon_constraints=[h_con])
    x0 = jnp.array([0.3, -0.2])
    t0 = jnp.array(0.0)
    dt = jnp.array(0.05)

    n_u, p_user = single_shooting_dimensions(prob)
    assert n_u == B * m
    assert p_user == 1  # 1 horizon constraint row

    u = jnp.array([0.4, -0.6])

    # 1. Constraint Jacobian vs central finite differences
    jac_ad = eval_jac_g(prob, u, x0, t0=t0, dt=dt)
    eps = 1e-6
    jac_fd = []
    for i in range(len(u)):
        e = jnp.zeros(len(u)).at[i].set(eps)
        g_plus = eval_g(prob, u + e, x0, t0=t0, dt=dt)
        g_minus = eval_g(prob, u - e, x0, t0=t0, dt=dt)
        jac_fd.append((g_plus - g_minus) / (2 * eps))
    jac_fd_arr = jnp.column_stack(jac_fd).reshape(-1)
    np.testing.assert_allclose(jac_ad, jac_fd_arr, rtol=1e-4, atol=1e-5)

    # 2. Lagrangian Hessian vs central finite differences of gradient
    lam = jnp.array([0.7])
    obj_factor = 1.2
    h_ad = eval_h(prob, u, x0, t0=t0, dt=dt, obj_factor=obj_factor, lam=lam)

    # Gradient of Lagrangian
    def lagrangian(u_vec: jax.Array) -> jax.Array:
        cost = _cost_fn(prob, u_vec, x0, t0, dt)
        cons = eval_g(prob, u_vec, x0, t0=t0, dt=dt)
        return obj_factor * cost + jnp.dot(lam, cons)

    grad_lag = jax.grad(lagrangian)
    hess_fd = []
    for i in range(len(u)):
        e = jnp.zeros(len(u)).at[i].set(eps)
        g_plus = grad_lag(u + e)
        g_minus = grad_lag(u - e)
        hess_fd.append((g_plus - g_minus) / (2 * eps))
    hess_fd_mat = jnp.column_stack(hess_fd)
    tril_r, tril_c = np.tril_indices(len(u))
    np.testing.assert_allclose(h_ad, hess_fd_mat[tril_r, tril_c], rtol=1e-4, atol=1e-5)


def test_multiplier_recovery_and_zeroing_with_horizon_constraints() -> None:
    """Unit solves return recovered canonical Multipliers; blocked solves return zero canonical Multipliers."""
    pytest.importorskip("cyipopt")

    # 1. b = 1 single shooting with Horizon constraint
    N, b, m, n = 5, 1, 1, 2
    H = N - 1
    h_con = LinearHorizonConstraint(
        A=jnp.ones((1, H)),
        b=jnp.array([3.0]),
        sense=NegativeOrthant(),
        inds=[j * (n + m) + n for j in range(H)],
    )
    prob_unit = _make_prob(N=N, b=b, m=m, n=n, horizon_constraints=[h_con])
    x0 = jnp.array([0.1, -0.1])
    opts = {"tol": 1e-8, "constr_viol_tol": 1e-8, "print_level": 0}

    res_ss_unit = MPC(prob_unit, SingleShooting(Ipopt(options=opts)), x0=x0).solve()
    assert res_ss_unit.success
    # Canonical lam length matches constraint_row_count
    assert len(res_ss_unit.lam) == constraint_row_count(prob_unit)
    assert len(res_ss_unit.mu) == N * n + H * m

    # 2. b = 2 single shooting and multiple shooting with Horizon constraint
    prob_blk = _make_prob(N=N, b=2, m=m, n=n, horizon_constraints=[h_con])
    res_ss_blk = MPC(prob_blk, SingleShooting(Ipopt(options=opts)), x0=x0).solve()
    assert res_ss_blk.success
    assert len(res_ss_blk.lam) == constraint_row_count(prob_blk)
    assert len(res_ss_blk.mu) == N * n + H * m
    # For b > 1, canonical duals are reset to exact zeros
    assert np.all(res_ss_blk.lam == 0.0)
    assert np.all(res_ss_blk.mu == 0.0)

    res_ms_blk = MPC(prob_blk, Ipopt(options=opts), x0=x0).solve()
    assert res_ms_blk.success
    assert len(res_ms_blk.lam) == constraint_row_count(prob_blk)
    assert len(res_ms_blk.mu) == N * n + H * m
    assert np.all(res_ms_blk.lam == 0.0)
    assert np.all(res_ms_blk.mu == 0.0)
