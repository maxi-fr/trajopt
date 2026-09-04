import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from trajopt.cones import NegativeOrthant, ZeroCone
from trajopt.constraints.base import Constraint
from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import DiscretizedDynamics
from trajopt.dynamics.integrators import Euler
from trajopt.models.cartpole import Cartpole
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.transcription.layout import _z_to_trajectory
from trajopt.transcription.subproblem import quadratic_subproblem
from trajopt.transcription.transcription import eval_f, eval_g

N, n, m = 7, 4, 1
DT = 0.05


def _problem() -> Problem:
    """Cartpole over a short horizon with box bounds, a linear inequality and a terminal goal."""
    cost = QuadraticCost(Q=jnp.eye(n), R=0.1 * jnp.eye(m), r=jnp.zeros(m), c=0.0)
    term = QuadraticCost(Q=5.0 * jnp.eye(n), R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=term, N=N)

    clist = ConstraintList(n, m, N)
    clist.add_constraint(ControlBound(m=m, u_min=jnp.array([-4.0]), u_max=jnp.array([4.0]), n=n), range(N - 1))
    clist.add_constraint(
        StateBound(n=n, x_min=jnp.full(n, -6.0), x_max=jnp.full(n, 6.0), m=m),
        range(N),
    )
    clist.add_constraint(
        LinearConstraint(
            n=n,
            m=m,
            A=jnp.array([[1.0, 0.5]]),
            b=jnp.array([1.5]),
            sense=NegativeOrthant(),
            inds=[0, 4],
        ),
        range(2, N - 1),
    )
    clist.add_constraint(GoalConstraint(n=n, m=m, xf=jnp.zeros(n)), N - 1)
    return Problem(model=DiscretizedDynamics(Cartpole(), Euler()), obj=obj, constraints=clist, N=N, dt=DT)


def _fixture() -> tuple[Problem, BoundaryConditions, jnp.ndarray]:
    """A retargeted problem, its boundary conditions, and a non-trivial Operating Point."""
    problem = _problem()
    rng = np.random.default_rng(0)
    nz = N * n + (N - 1) * m
    z_op = jnp.asarray(rng.normal(scale=0.4, size=nz), dtype=jnp.float64)

    bc = BoundaryConditions(
        x0=jnp.array([0.3, 0.1, -0.2, 0.05]),
        t0=jnp.asarray(0.25),
        X_ref=jnp.asarray(rng.normal(scale=0.2, size=(N, n))),
        U_ref=jnp.asarray(rng.normal(scale=0.2, size=(N - 1, m))),
    )
    return retarget_problem(problem, bc), bc, z_op


def _linearize(
    con: Constraint,
    *,
    tk: jnp.ndarray,
    is_term: bool,
    xf_val: jnp.ndarray | None,
    z_op_k: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian block and offset of one constraint about its knot slice of the Operating Point."""
    x_op = z_op_k[:n]
    u_op = None if is_term else z_op_k[n : n + m]
    jx, ju = con.jacobian(x_op, u_op, tk)
    block = np.asarray(jx, dtype=np.float64) if is_term else np.hstack([np.asarray(jx), np.asarray(ju)])
    kwargs = {"xf": xf_val} if isinstance(con, GoalConstraint) and xf_val is not None else {}
    val_op = np.asarray(con.evaluate(x_op, u_op, tk, **kwargs), dtype=np.float64)
    return block, val_op - block @ np.asarray(z_op_k[: block.shape[1]], dtype=np.float64)


def _reference_qp(
    problem: Problem,
    bc: BoundaryConditions,
    z_op: jnp.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build the QP the way the OSQP adapter did before it consumed `quadratic_subproblem`.

    An independent construction: a dense autodiff expansion of the objective, and each dynamics
    step and constraint linearized from its own Jacobian, rather than anything routed through the
    NLP callbacks or the sparsity patterns. Rows come back in canonical order so they line up with
    `QuadraticSubproblem`.
    """
    nz = N * n + (N - 1) * m
    x0_arr = jnp.asarray(bc.x0, dtype=jnp.float64)
    t0_arr = jnp.asarray(bc.t0, dtype=jnp.float64)
    dt_arr = jnp.asarray(problem.dt, dtype=jnp.float64)
    xf_val = bc.xf
    t_stage = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr[:-1])])
    t_term = t0_arr + jnp.sum(dt_arr)
    X_op, U_op = _z_to_trajectory(z_op, N, n, m)

    H_dense = np.asarray(jax.hessian(lambda z: eval_f(problem, z, t0_arr, dt_arr))(z_op), dtype=np.float64)
    g_dense = np.asarray(jax.grad(lambda z: eval_f(problem, z, t0_arr, dt_arr))(z_op), dtype=np.float64)
    P_triu = sp.triu(sp.csc_matrix(H_dense), format="csc")
    q_vec = g_dense - H_dense @ np.asarray(z_op, dtype=np.float64)

    model = problem.model
    A_rows: list[sp.spmatrix] = []
    lo: list[np.ndarray] = []
    hi: list[np.ndarray] = []

    def knot_constraint_rows(k: int) -> None:
        """Append knot k's linearized constraint rows, in the order its evaluator holds them."""
        tk = t_stage[k] if k < N - 1 else t_term
        col_k = k * (n + m)
        is_term = k == N - 1
        z_op_k = z_op[col_k : col_k + (n if is_term else n + m)]
        for con in problem.constraints.knot_evaluators[k].constraints:
            block, val0 = _linearize(con, tk=tk, is_term=is_term, xf_val=xf_val, z_op_k=z_op_k)
            A_con = sp.lil_matrix((int(con.p), nz), dtype=np.float64)
            A_con[:, col_k : col_k + block.shape[1]] = block
            A_rows.append(A_con.tocsr())
            lo.append(-val0 if isinstance(con.cone, ZeroCone) else np.full(int(con.p), -np.inf))
            hi.append(-val0)

    # Initial-condition pin
    A_init = sp.lil_matrix((n, nz), dtype=np.float64)
    A_init[:, :n] = np.eye(n)
    A_rows.append(A_init.tocsr())
    x0_np = np.asarray(x0_arr, dtype=np.float64)
    lo.append(x0_np)
    hi.append(x0_np)

    for k in range(N - 1):
        Ak = np.asarray(model.state_jacobian(X_op[k], U_op[k], t_stage[k], dt_arr[k]), dtype=np.float64)
        Bk = np.asarray(model.control_jacobian(X_op[k], U_op[k], t_stage[k], dt_arr[k]), dtype=np.float64)
        f_op = np.asarray(model.discrete_dynamics(X_op[k], U_op[k], t_stage[k], dt_arr[k]), dtype=np.float64)
        dk = f_op - Ak @ np.asarray(X_op[k], dtype=np.float64) - Bk @ np.asarray(U_op[k], dtype=np.float64)

        A_dyn = sp.lil_matrix((n, nz), dtype=np.float64)
        col_x = k * (n + m)
        A_dyn[:, col_x : col_x + n] = -Ak
        A_dyn[:, col_x + n : col_x + n + m] = -Bk
        A_dyn[:, (k + 1) * (n + m) : (k + 1) * (n + m) + n] = np.eye(n)
        A_rows.append(A_dyn.tocsr())
        lo.append(dk)
        hi.append(dk)
        knot_constraint_rows(k)

    knot_constraint_rows(N - 1)

    return P_triu, q_vec, sp.vstack(A_rows).tocsr(), np.concatenate(lo), np.concatenate(hi)


def test_quadratic_subproblem_matches_direct_extraction() -> None:
    """The QP derived from the NLP callbacks matches the direct per-constraint extraction."""
    problem, bc, z_op = _fixture()

    P_ref, q_ref, A_ref, l_ref, u_ref = _reference_qp(problem, bc, z_op)
    qp = quadratic_subproblem(problem, z_op, bc)

    assert np.allclose(qp.P.toarray(), P_ref.toarray(), rtol=1e-10, atol=1e-10)
    assert np.allclose(qp.q, q_ref, rtol=1e-10, atol=1e-10)
    assert qp.A.shape == A_ref.shape
    assert np.allclose(qp.A.toarray(), A_ref.toarray(), rtol=1e-10, atol=1e-9)
    assert np.allclose(qp.row_lower, l_ref, rtol=1e-10, atol=1e-9, equal_nan=False)
    assert np.allclose(qp.row_upper, u_ref, rtol=1e-10, atol=1e-9)


def test_quadratic_subproblem_blocks_cover_every_row() -> None:
    """Cone blocks tile the canonical rows exactly once, in row order."""
    problem, bc, z_op = _fixture()
    qp = quadratic_subproblem(problem, z_op, bc)

    assert qp.blocks[0].start == 0
    for prev, block in zip(qp.blocks, qp.blocks[1:], strict=False):
        assert block.start == prev.stop
    assert qp.blocks[-1].stop == qp.A.shape[0]


@pytest.mark.parametrize("scale", [0.0, 0.7])
def test_quadratic_subproblem_reproduces_constraint_values(scale: float) -> None:
    """`A z + affine` reproduces the nonlinear constraint vector at the Operating Point itself."""
    problem, bc, z_op = _fixture()
    z_op = z_op * scale
    qp = quadratic_subproblem(problem, z_op, bc)

    c_op = np.asarray(
        eval_g(problem, z_op, jnp.asarray(bc.x0), jnp.asarray(bc.t0), problem.dt, xf=bc.xf), dtype=np.float64
    )
    assert np.allclose(qp.A @ np.asarray(z_op) + qp.affine, c_op, atol=1e-10)
