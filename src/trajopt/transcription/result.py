from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax
import numpy as np

from trajopt.problem import Problem
from trajopt.trajectory import Trajectory

if TYPE_CHECKING:
    from trajopt.problem import MPCState


@runtime_checkable
class SolverResult(Protocol):
    """The result every solver adapter returns, with duals in the canonical row order.

    `lam` and `mu` are normalised by the adapter, not by the caller: each backend knows its
    own row layout and sign convention, so `problem.solve` reads these two fields directly
    rather than sniffing a backend-specific key out of `info`.
    """

    trajectory: Trajectory
    success: bool
    status: Any
    message: str
    cost: float
    Z: jax.Array
    info: dict[str, Any]
    iterations: int
    constraint_violation: float
    lam: np.ndarray
    mu: np.ndarray


def constraint_row_count(problem: Problem) -> int:
    """Return the canonical constraint row count P of the transcribed problem."""
    n = int(problem.model.n)
    N = int(problem.N)
    return n + (N - 1) * n + int(sum(problem.constraints.p))


def blocked_to_canonical(problem: Problem) -> np.ndarray:
    """Return row indices mapping a blocked dual vector into canonical order.

    The canonical order is the one `eval_g` emits: the initial condition, then each knot's
    dynamics defect followed by that knot's constraint rows. OSQP and Clarabel instead stack
    every dynamics row first and every constraint row after, so their duals need permuting.

    Returns
    -------
    np.ndarray
        Index array `idx` of shape ``(P,)`` such that ``dual_blocked[idx]`` is in canonical
        order, where `dual_blocked` holds the ``n + (N - 1) * n`` dynamics rows followed by
        the knot constraint rows in knot order.
    """
    n = int(problem.model.n)
    N = int(problem.N)
    knot_p = list(problem.constraints.p)

    con_base = n + (N - 1) * n
    con_offset = con_base
    idx: list[int] = list(range(n))  # initial condition

    for k in range(N - 1):
        dyn_start = n + k * n
        idx.extend(range(dyn_start, dyn_start + n))
        p_k = knot_p[k] if k < len(knot_p) else 0
        idx.extend(range(con_offset, con_offset + p_k))
        con_offset += p_k

    p_term = knot_p[N - 1] if len(knot_p) > N - 1 else 0
    idx.extend(range(con_offset, con_offset + p_term))

    return np.asarray(idx, dtype=int)


def split_bound_duals(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split signed bound duals into the non-negative (lower, upper) pair solvers expect.

    `mu` follows the ``mult_x_U - mult_x_L`` convention, so a variable pressing against its
    upper limit carries a positive entry. At most one of the two limits is active at a time,
    which is what makes the signed form lossless.
    """
    return np.maximum(-mu, 0.0), np.maximum(mu, 0.0)


def warm_start_duals(problem: Problem, state: "MPCState") -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return `(lam, mu)` from an MPCState when they are usable as a warm start, else `(None, None)`.

    A freshly built MPCState carries zero multipliers, which are not a warm start but the
    absence of one -- handing them over would replace the solver's own initialisation with
    a worse guess. Duals whose length no longer matches the problem are refused for the
    same reason.
    """
    lam = np.asarray(state.lam, dtype=np.float64)
    mu = np.asarray(state.mu, dtype=np.float64)

    nz = int(problem.N) * int(problem.model.n) + (int(problem.N) - 1) * int(problem.model.m)
    if lam.shape != (constraint_row_count(problem),) or mu.shape != (nz,):
        return None, None
    if not np.any(lam) and not np.any(mu):
        return None, None
    return lam, mu
