# ruff: noqa: RUF001 -- embedded Julia source uses Altro's own field names rho/drho (Greek in Julia)
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.expansions import Expansion
from trajopt.solvers.ilqr import DynamicRegularization, backward_pass
from trajopt.solvers.options import SolverOptions

# Ticket 25: driving Altro.backwardpass! and our backward_pass from the identical randomly
# generated Expansion isolates the recursion itself from cost/dynamics-expansion code, which
# the pre-existing jl_to cross tests already cover.
pytestmark = [pytest.mark.julia, pytest.mark.altro]

_ALTRO_SETUP = """
using Altro, TrajectoryOptimization, LinearAlgebra
const TO = TrajectoryOptimization

function trajopt_ticket25_run_backward_pass(problem_fn, A, B, q, r, Qm, Rm, H, qN, QfN, rho, drho)
    prob, _opts = problem_fn()
    solver = Altro.iLQRSolver(prob)
    N = solver.N
    for k in 1:N-1
        solver.D[k].fx .= A[k, :, :]
        solver.D[k].fu .= B[k, :, :]
        solver.Eerr[k].x .= q[k, :]
        solver.Eerr[k].u .= r[k, :]
        solver.Eerr[k].xx .= Qm[k, :, :]
        solver.Eerr[k].uu .= Rm[k, :, :]
        solver.Eerr[k].ux .= H[k, :, :]
    end
    solver.Eerr[N].x .= qN
    solver.Eerr[N].xx .= QfN
    solver.reg.ρ = rho
    solver.reg.dρ = drho
    Altro.backwardpass!(solver)
    Ks = cat([Matrix(solver.K[k]) for k = 1:N-1]..., dims=3)
    ds = cat([Vector(solver.d[k]) for k = 1:N-1]..., dims=2)
    Sx = cat([Vector(solver.S[k].x) for k = 1:N]..., dims=2)
    Sxx = cat([Matrix(solver.S[k].xx) for k = 1:N]..., dims=3)
    return Ks, ds, Sx, Sxx, Vector(solver.ΔV), solver.reg.ρ, solver.reg.dρ
end
"""


def _pd_matrix(m_: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite matrix built from a random square `m_`."""
    return m_ @ m_.T + np.eye(m_.shape[0])


def _random_expansion(N: int, ne: int, m: int, seed: int) -> Expansion:
    """Random Expansion with PD Q/R blocks (H = 0), sized to match an Altro benchmark problem."""
    rng = np.random.default_rng(seed)
    A = np.stack([rng.normal(size=(ne, ne)) * 0.1 + np.eye(ne) for _ in range(N - 1)])
    B = rng.normal(size=(N - 1, ne, m)) * 0.1
    q = rng.normal(size=(N, ne))
    r = rng.normal(size=(N - 1, m))
    Q = np.stack([_pd_matrix(rng.normal(size=(ne, ne))) for _ in range(N)])
    R = np.stack([_pd_matrix(rng.normal(size=(m, m))) for _ in range(N - 1)])
    H = np.zeros((N - 1, m, ne))
    return Expansion(
        A=jnp.asarray(A),
        B=jnp.asarray(B),
        q=jnp.asarray(q),
        r=jnp.asarray(r),
        Q=jnp.asarray(Q),
        R=jnp.asarray(R),
        H=jnp.asarray(H),
    )


def _assert_backward_pass_matches_altro(
    jl: Any,
    problem_fn_name: str,
    N: int,
    ne: int,
    m: int,
    *,
    rho: float,
    seed: int,
) -> None:
    jl.seval(_ALTRO_SETUP)
    run_bp = jl.seval("trajopt_ticket25_run_backward_pass")
    problem_fn = jl.seval(f"Altro.Problems.{problem_fn_name}")

    exp = _random_expansion(N, ne, m, seed)
    options = SolverOptions()
    reg = DynamicRegularization(rho=jnp.asarray(rho), drho=jnp.asarray(0.0))

    result = backward_pass(exp, reg, options)

    K_jl, d_jl, Sx_jl, Sxx_jl, dV_jl, rho_jl, drho_jl = run_bp(
        problem_fn,
        np.asarray(exp.A),
        np.asarray(exp.B),
        np.asarray(exp.q),
        np.asarray(exp.r),
        np.asarray(exp.Q),
        np.asarray(exp.R),
        np.asarray(exp.H),
        np.asarray(exp.q[-1]),
        np.asarray(exp.Q[-1]),
        rho,
        0.0,
    )

    # Julia stacks with knot as the last axis; move it to the front to match ours.
    K_jl = np.moveaxis(np.asarray(K_jl), -1, 0)
    d_jl = np.moveaxis(np.asarray(d_jl), -1, 0)
    Sx_jl = np.moveaxis(np.asarray(Sx_jl), -1, 0)
    Sxx_jl = np.moveaxis(np.asarray(Sxx_jl), -1, 0)

    assert not bool(result.failed)
    np.testing.assert_allclose(np.asarray(result.K), K_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.d), d_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.S_x), Sx_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.S_xx), Sxx_jl, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result.dV), np.asarray(dV_jl), atol=1e-8)
    np.testing.assert_allclose(float(result.regularization.rho), float(rho_jl), atol=1e-8)
    np.testing.assert_allclose(float(result.regularization.drho), float(drho_jl), atol=1e-8)


def test_cross_backward_pass_pendulum_rho_zero(jl_altro: Any) -> None:
    _assert_backward_pass_matches_altro(jl_altro, "Pendulum", N=51, ne=2, m=1, rho=0.0, seed=1)


def test_cross_backward_pass_pendulum_rho_positive(jl_altro: Any) -> None:
    _assert_backward_pass_matches_altro(jl_altro, "Pendulum", N=51, ne=2, m=1, rho=2.5, seed=2)


def test_cross_backward_pass_cartpole_rho_zero(jl_altro: Any) -> None:
    _assert_backward_pass_matches_altro(jl_altro, "Cartpole", N=101, ne=4, m=1, rho=0.0, seed=3)


def test_cross_backward_pass_cartpole_rho_positive(jl_altro: Any) -> None:
    _assert_backward_pass_matches_altro(jl_altro, "Cartpole", N=101, ne=4, m=1, rho=1.5, seed=4)
