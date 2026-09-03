from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import jax
import numpy as np

from trajopt.problem import Problem
from trajopt.trajectory import Trajectory

if TYPE_CHECKING:
    from trajopt.problem import BoundaryConditions
    from trajopt.program import Program, WarmStart

SolverStatus = Literal["converged", "infeasible", "iteration_limit", "error"]


@runtime_checkable
class SolverResult(Protocol):
    """The result every solver adapter returns, with duals in the canonical row order.

    `lam` and `mu` are normalised by the adapter, not by the caller: each backend knows its
    own row layout and sign convention, so the `MPC` driver reads these two fields directly
    rather than sniffing a backend-specific key out of `info`.

    Every member is a read-only property rather than a plain attribute so that the backends'
    `NamedTuple` results, whose fields are themselves read-only, satisfy this Protocol
    structurally: a plain attribute annotation implies a setter too, which a `NamedTuple`
    field does not have.
    """

    @property
    def trajectory(self) -> Trajectory:
        """Optimal state and control trajectory."""
        ...

    @property
    def success(self) -> bool:
        """Whether the solver converged to optimality or within tolerance."""
        ...

    @property
    def status(self) -> Any:  # noqa: ANN401 -- native status code per backend: int, str, ...
        """Native backend status code or string."""
        ...

    @property
    def message(self) -> str:
        """Native backend status message."""
        ...

    @property
    def cost(self) -> float:
        """Final objective value."""
        ...

    @property
    def Z(self) -> jax.Array:  # noqa: N802 -- matches the backends' NamedTuple field name
        """Optimal flat primal vector."""
        ...

    @property
    def info(self) -> dict[str, Any]:
        """Raw backend-specific return info dictionary."""
        ...

    @property
    def iterations(self) -> int:
        """Number of solver iterations."""
        ...

    @property
    def constraint_violation(self) -> float:
        """Maximum constraint violation across all constraints."""
        ...

    @property
    def lam(self) -> np.ndarray:
        """Constraint duals in canonical row order, of shape ``(P,)``."""
        ...

    @property
    def mu(self) -> np.ndarray:
        """Signed bound duals of shape ``(N * n + (N - 1) * m,)``."""
        ...


@runtime_checkable
class Solver(Protocol):
    """A solver backend that solves a `Program`'s Problem from boundary conditions and a warm start."""

    def solve(self, program: "Program", bc: "BoundaryConditions", ws: "WarmStart") -> SolverResult:
        """Solve `program`'s problem from `bc` and `ws`, returning the backend's raw result.

        The Program rather than the Problem is the argument because a backend needs both: its
        structural problem, at `program.problem`, and the compiled cores and live handles the
        Program holds on its behalf across receding-horizon steps.
        """
        ...


def normalize_status(*, success: bool, message: str) -> SolverStatus:
    """Map a backend's success flag and native status message onto the normalized status vocabulary.

    Every backend's `message` field already carries a human-readable native status (Ipopt's
    `status_msg`, OSQP's `status`, Clarabel's `status`), so pattern-matching that text is what
    lets one function normalize all three rather than each backend hand-coding its own mapping.
    """
    if success:
        return "converged"
    lowered = message.lower()
    if "infeasib" in lowered:
        return "infeasible"
    if "iter" in lowered:
        return "iteration_limit"
    return "error"


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


def warm_start_duals(problem: Problem, ws: "WarmStart") -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return `(lam, mu)` from a WarmStart when they are usable as one, else `(None, None)`.

    A cold WarmStart carries zero multipliers, which are not a warm start but the absence of
    one -- handing them over would replace the solver's own initialisation with a worse guess.
    Duals whose length no longer matches the problem are refused for the same reason.
    """
    lam = np.asarray(ws.lam, dtype=np.float64)
    mu = np.asarray(ws.mu, dtype=np.float64)

    nz = int(problem.N) * int(problem.model.n) + (int(problem.N) - 1) * int(problem.model.m)
    if lam.shape != (constraint_row_count(problem),) or mu.shape != (nz,):
        return None, None
    if not np.any(lam) and not np.any(mu):
        return None, None
    return lam, mu
