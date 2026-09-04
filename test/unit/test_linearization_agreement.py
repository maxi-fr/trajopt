import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from trajopt.constraints.constraint_list import ConstraintList
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import QuadraticCost
from trajopt.dynamics.base import AbstractModel, DiscretizedDynamics
from trajopt.dynamics.integrators import RK4
from trajopt.models.affine import AffineModel
from trajopt.models.cartpole import Cartpole
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.subproblem import quadratic_subproblem

DT = 0.05


def _lqr_problem(model: AbstractModel, N: int) -> Problem:
    """LQR problem on `model` with no path constraints, so the transcription is exactly quadratic."""
    n, m = int(model.n), int(model.m)
    cost = QuadraticCost(Q=jnp.eye(n), R=0.1 * jnp.eye(m), r=jnp.zeros(m), c=0.0)
    term = QuadraticCost(Q=5.0 * jnp.eye(n), R=jnp.zeros((m, m)), r=jnp.zeros(m), c=0.0)
    obj = Objective(stage_cost=cost, terminal_cost=term, N=N)
    return Problem(model=model, obj=obj, constraints=ConstraintList(n, m, N), N=N, dt=DT)


def _riccati_rollout(model: AffineModel, x0: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form finite-horizon LQR states (N, n) and controls (N - 1, m) for `_lqr_problem`'s weights."""
    A = np.asarray(model.A, dtype=np.float64)
    B = np.asarray(model.B, dtype=np.float64)
    n, m = A.shape[0], B.shape[1]
    Q, R = np.eye(n), 0.1 * np.eye(m)

    P = 5.0 * np.eye(n)
    gains: list[np.ndarray] = []
    for _ in range(N - 1):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        gains.append(K)
    gains.reverse()

    X = np.zeros((N, n))
    U = np.zeros((N - 1, m))
    X[0] = x0
    for k in range(N - 1):
        U[k] = -gains[k] @ X[k]
        X[k + 1] = A @ X[k] + B @ U[k]
    return X, U


def test_defect_jacobian_matches_model_linearization() -> None:
    """The NLP's defect rows carry exactly the stagewise (A_k, B_k), negated.

    A defect row block is d/dz of ``x_{k+1} - f_d(x_k, u_k)``, so the columns of knot k hold
    ``[-A_k, -B_k]`` and the state columns of knot k + 1 hold the identity. Those A_k, B_k are the
    stacked Jacobians `Problem.linearize` returns at the same Operating Point (Cartpole is
    Euclidean, so error coordinates are the state coordinates and no G-map intervenes).
    """
    N, n, m = 6, 4, 1
    model = DiscretizedDynamics(Cartpole(), RK4())
    problem = _lqr_problem(model, N)

    rng = np.random.default_rng(0)
    nz = N * n + (N - 1) * m
    z_op = jnp.asarray(rng.normal(scale=0.4, size=nz), dtype=jnp.float64)
    t0 = 0.25
    bc = BoundaryConditions(x0=jnp.asarray(z_op[:n]), t0=jnp.asarray(t0))

    qp = quadratic_subproblem(problem, z_op, bc)
    lin = problem.linearize(z_op, t0=t0)

    A_dense = np.asarray(sp.csr_matrix(qp.A).todense())
    for k in range(N - 1):
        rows = A_dense[n + k * n : n + (k + 1) * n, :]
        col_k = k * (n + m)
        col_next = (k + 1) * (n + m)
        np.testing.assert_allclose(rows[:, col_k : col_k + n], -np.asarray(lin.A[k]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(rows[:, col_k + n : col_k + n + m], -np.asarray(lin.B[k]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(rows[:, col_next : col_next + n], np.eye(n), rtol=0, atol=1e-12)


def test_affine_model_is_its_own_linearization() -> None:
    """An AffineModel linearizes to the A, B it was built from at any Operating Point."""
    N = 5
    model = AffineModel(
        A=jnp.array([[1.0, DT], [0.0, 1.0]]),
        B=jnp.array([[0.5 * DT**2], [DT]]),
        d=jnp.array([0.01, -0.02]),
    )
    problem = _lqr_problem(model, N)
    rng = np.random.default_rng(1)
    z_op = jnp.asarray(rng.normal(size=N * 2 + (N - 1) * 1), dtype=jnp.float64)

    lin = problem.linearize(z_op)

    for k in range(N - 1):
        np.testing.assert_allclose(np.asarray(lin.A[k]), np.asarray(model.A), rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(lin.B[k]), np.asarray(model.B), rtol=0, atol=1e-14)


@pytest.mark.parametrize("d", [None, jnp.array([0.02, -0.05])])
def test_nlp_and_qp_transcriptions_agree_on_lqr(d: jnp.ndarray | None) -> None:
    """Ipopt and the QP path reach the same LQR trajectory, and (for d = 0) the Riccati optimum.

    The problem is linear-quadratic with no path constraints, so the Quadratic Subproblem is not an
    approximation of the NLP -- it is the NLP. Any second construction of the QP that drifts from
    what Ipopt is handed shows up here as a trajectory difference.
    """
    N = 12
    model = AffineModel(A=jnp.array([[1.0, DT], [0.0, 1.0]]), B=jnp.array([[0.5 * DT**2], [DT]]), d=d)
    problem = _lqr_problem(model, N)
    x0 = jnp.array([1.5, -0.3])

    res_nlp = MPC(problem, Ipopt(options={"print_level": 0, "tol": 1e-12}), x0=x0).solve()
    res_qp = MPC(problem, OSQP(options={"eps_abs": 1e-12, "eps_rel": 1e-12, "max_iter": 100000}), x0=x0).solve()

    assert res_nlp.success
    assert res_qp.success
    np.testing.assert_allclose(np.asarray(res_qp.trajectory.X), np.asarray(res_nlp.trajectory.X), atol=1e-6)
    np.testing.assert_allclose(np.asarray(res_qp.trajectory.U), np.asarray(res_nlp.trajectory.U), atol=1e-6)

    if d is None:
        X_lqr, U_lqr = _riccati_rollout(model, np.asarray(x0, dtype=np.float64), N)
        np.testing.assert_allclose(np.asarray(res_nlp.trajectory.X), X_lqr, atol=1e-6)
        np.testing.assert_allclose(np.asarray(res_nlp.trajectory.U), U_lqr, atol=1e-6)
        np.testing.assert_allclose(np.asarray(res_qp.trajectory.X), X_lqr, atol=1e-6)
        np.testing.assert_allclose(np.asarray(res_qp.trajectory.U), U_lqr, atol=1e-6)
