from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from trajopt.cones import AbstractCone, ZeroCone
from trajopt.problem import BoundaryConditions, Problem
from trajopt.transcription.layout import constraint_bounds, primal_bounds
from trajopt.transcription.result import constraint_row_count
from trajopt.transcription.sparsity import hessian_sparsity_pattern, jacobian_sparsity_pattern
from trajopt.transcription.transcription import eval_g, eval_grad_f, eval_h, eval_jac_g


class ConstraintBlock(NamedTuple):
    """One contiguous run of constraint rows sharing a Cone.

    Parameters
    ----------
    cone : AbstractCone
        The Cone the block's rows are required to lie in.
    start : int
        First row of the block in the canonical row order.
    stop : int
        One past the block's last row.
    """

    cone: AbstractCone
    start: int
    stop: int


@dataclass(frozen=True)
class QuadraticSubproblem:
    """The NLP's quadratic subproblem at one Operating Point, in the canonical row order.

    The QP is not a separate transcription: it is the nonlinear program's derived form, the
    objective taken to second order and the constraints linearized about z_op. Every field
    therefore comes from the five NLP callbacks, so a Backend consuming this cannot drift away
    from what Ipopt is handed.

    Rows follow `eval_g`'s canonical order -- the initial-condition pin, then each Knot Point's
    Defect followed by that knot's constraint rows -- so a Backend's row duals are already the
    canonical `lam` and need no permuting.

    `row_lower` and `row_upper` express each Cone as a box on ``A z``, which is exactly what a
    box-form QP solver wants. That is only possible for the box-representable Cones (Zero,
    NegativeOrthant, PositiveOrthant); on a `SecondOrderCone` block those two entries carry the
    fallback equality bounds and mean nothing, and a Backend supporting such a block reads
    `blocks` and `affine` and assembles the conic form itself.

    Parameters
    ----------
    P : sp.csc_matrix
        Upper triangle of the Lagrangian Hessian of shape ``(nz, nz)``.
    q : np.ndarray
        Linear objective term of shape ``(nz,)``, already shifted so the model is written in z
        rather than in ``z - z_op``.
    A : sp.csr_matrix
        Constraint Jacobian at z_op of shape ``(P_rows, nz)``.
    affine : np.ndarray
        Offset of shape ``(P_rows,)`` with ``c(z) ~ A z + affine``.
    row_lower, row_upper : np.ndarray
        Box on ``A z`` of shape ``(P_rows,)`` equivalent to each row's Cone.
    z_lower, z_upper : np.ndarray
        Primal variable limits of shape ``(nz,)``.
    blocks : tuple[ConstraintBlock, ...]
        The Cone of every constraint row, as contiguous blocks covering all ``P_rows`` rows.
    """

    P: sp.csc_matrix
    q: np.ndarray
    A: sp.csr_matrix
    affine: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    z_lower: np.ndarray
    z_upper: np.ndarray
    blocks: tuple[ConstraintBlock, ...]


def constraint_blocks(problem: Problem) -> tuple[ConstraintBlock, ...]:
    """Cone of every canonical constraint row, as contiguous blocks.

    Structural rather than traced: it mirrors the row order `eval_g` emits, which depends only on
    the horizon and the registered constraints.
    """
    N = int(problem.N)
    n = int(problem.model.n)
    knot_evaluators = problem.constraints.knot_evaluators

    blocks: list[ConstraintBlock] = []
    row = 0

    def push(cone: AbstractCone, width: int) -> None:
        """Append a block of `width` rows carrying `cone`, advancing the row cursor."""
        nonlocal row
        if width > 0:
            blocks.append(ConstraintBlock(cone=cone, start=row, stop=row + width))
            row += width

    push(ZeroCone(), n)  # initial-condition pin
    for k in range(N - 1):
        push(ZeroCone(), n)  # dynamics Defect
        if k < len(knot_evaluators):
            for con in knot_evaluators[k].constraints:
                push(con.cone, int(con.p))

    if len(knot_evaluators) > N - 1:
        for con in knot_evaluators[N - 1].constraints:
            push(con.cone, int(con.p))

    return tuple(blocks)


def quadratic_subproblem(problem: Problem, z_op: jax.Array, bc: BoundaryConditions) -> QuadraticSubproblem:
    """Derive `problem`'s quadratic subproblem at the Operating Point z_op.

    `problem`'s objective is expanded as given, so a caller aiming it at a run-time reference
    window retargets the problem first; `bc` is read only for the initial state, the initial
    timestamp and the run-time goal the constraints see.

    Parameters
    ----------
    problem : Problem
        The transcribed problem, already retargeted onto `bc`'s reference window.
    z_op : jax.Array
        Operating Point as a flat Primal Vector of shape ``(N * n + (N - 1) * m,)``.
    bc : BoundaryConditions
        Boundary data of this solve.
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)
    nz = N * n + (N - 1) * m

    x0 = jnp.asarray(bc.x0, dtype=jnp.float64)
    t0 = jnp.asarray(bc.t0, dtype=jnp.float64)
    dt = jnp.asarray(problem.dt, dtype=jnp.float64)
    xf = bc.xf
    z_op_np = np.asarray(z_op, dtype=np.float64)

    # Objective: 0.5 (z - z_op)' H (z - z_op) + g'(z - z_op), rewritten in z with the constant dropped.
    h_rows, h_cols = hessian_sparsity_pattern(N, n, m)
    h_vals = np.asarray(eval_h(problem, z_op, t0=t0, dt=dt), dtype=np.float64)
    H = sp.coo_matrix((h_vals, (h_rows, h_cols)), shape=(nz, nz), dtype=np.float64).tocsc()
    P_triu = sp.triu(H, format="csc")
    grad = np.asarray(eval_grad_f(problem, z_op, t0=t0, dt=dt), dtype=np.float64)
    q_vec = grad - H @ z_op_np

    # Constraints: c(z) ~ c(z_op) + J (z - z_op).
    j_rows, j_cols = jacobian_sparsity_pattern(N, n, m, problem.constraints.p)
    j_vals = np.asarray(eval_jac_g(problem, z_op, x0, t0, dt, xf=xf), dtype=np.float64)
    n_rows = constraint_row_count(problem)
    A_mat = sp.coo_matrix((j_vals, (j_rows, j_cols)), shape=(n_rows, nz), dtype=np.float64).tocsr()
    c_op = np.asarray(eval_g(problem, z_op, x0, t0, dt, xf=xf), dtype=np.float64)
    affine = c_op - A_mat @ z_op_np

    gL, gU = constraint_bounds(problem)
    z_lower, z_upper = primal_bounds(problem)

    return QuadraticSubproblem(
        P=P_triu,
        q=q_vec,
        A=A_mat,
        affine=affine,
        row_lower=gL - affine,
        row_upper=gU - affine,
        z_lower=z_lower,
        z_upper=z_upper,
        blocks=constraint_blocks(problem),
    )
