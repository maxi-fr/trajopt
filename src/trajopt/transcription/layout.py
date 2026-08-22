import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import IdentityCone, NegativeOrthant, PositiveOrthant
from trajopt.constraints.constraint_list import BuiltKnotConstraint
from trajopt.problem import Problem


def trajectory_to_z(X: jax.Array, U: jax.Array) -> jax.Array:
    """Interleave states and controls into the flat NLP primal vector Z.

    Parameters
    ----------
    X : jax.Array
        Stacked state trajectory of shape (N, n).
    U : jax.Array
        Stacked control trajectory of shape (N-1, m).

    Returns
    -------
    jax.Array
        Flat primal vector Z of shape (N * n + (N - 1) * m,).
    """
    Z_stages = jnp.concatenate([X[:-1], U], axis=1).reshape(-1)
    return jnp.concatenate([Z_stages, X[-1]])


def z_to_trajectory(
    Z: jax.Array,
    N: int,
    n: int,
    m: int,
) -> tuple[jax.Array, jax.Array]:
    """Recover state and control trajectories from the flat NLP primal vector Z.

    Parameters
    ----------
    Z : jax.Array
        Flat primal vector of shape (N * n + (N - 1) * m,).
    N : int
        Horizon length in knot points.
    n : int
        State dimension.
    m : int
        Control dimension.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        State trajectory X of shape (N, n) and control trajectory U of shape (N-1, m).
    """
    stage_len = (N - 1) * (n + m)
    stage_part = Z[:stage_len].reshape((N - 1, n + m))
    X_stage = stage_part[:, :n]
    U = stage_part[:, n:]
    X_term = Z[stage_len : stage_len + n]
    X = jnp.vstack([X_stage, X_term])
    return X, U


def primal_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Extract solver variable limits zL <= Z <= zU from the problem's box bounds.

    Parameters
    ----------
    problem : Problem
        Problem instance containing constraints and horizon dimensions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower bounds zL and upper bounds zU of shape (N * n + (N - 1) * m,).
    """
    N = int(problem.N)
    n = int(problem.model.n)
    m = int(problem.model.m)

    if hasattr(problem.constraints, "primal_bounds"):
        xL, xU, uL, uU = problem.constraints.primal_bounds()
    else:
        xL = np.full((N, n), -np.inf)
        xU = np.full((N, n), np.inf)
        uL = np.full((N - 1, m), -np.inf)
        uU = np.full((N - 1, m), np.inf)

    zL_stages = np.concatenate([xL[:-1], uL], axis=1).reshape(-1)
    zL = np.concatenate([zL_stages, xL[-1]])

    zU_stages = np.concatenate([xU[:-1], uU], axis=1).reshape(-1)
    zU = np.concatenate([zU_stages, xU[-1]])

    return zL.astype(np.float64), zU.astype(np.float64)


def _cone_bounds(cone: object, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute (gL, gU) bounds corresponding to a cone constraint."""
    if isinstance(cone, NegativeOrthant):
        return np.full(p, -np.inf, dtype=np.float64), np.zeros(p, dtype=np.float64)
    if isinstance(cone, PositiveOrthant):
        return np.zeros(p, dtype=np.float64), np.full(p, np.inf, dtype=np.float64)
    if isinstance(cone, IdentityCone):
        return np.full(p, -np.inf, dtype=np.float64), np.full(p, np.inf, dtype=np.float64)
    # Default equality (ZeroCone or general equality)
    return np.zeros(p, dtype=np.float64), np.zeros(p, dtype=np.float64)


def _evaluator_bounds(evaluator: BuiltKnotConstraint) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute (gL, gU) arrays for all constraints in a knot evaluator."""
    gL_list: list[np.ndarray] = []
    gU_list: list[np.ndarray] = []
    for c in evaluator.constraints:
        lo, hi = _cone_bounds(c.cone, int(c.p))
        gL_list.append(lo)
        gU_list.append(hi)
    return gL_list, gU_list


def constraint_bounds(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper bounds gL <= c(Z) <= gU for the transcribed constraint vector.

    Parameters
    ----------
    problem : Problem
        Problem instance containing model, constraints, and horizon.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Constraint lower and upper bounds of shape (P,).
    """
    N = int(problem.N)
    n = int(problem.model.n)

    gL_list: list[np.ndarray] = []
    gU_list: list[np.ndarray] = []

    # 1. Initial state condition: x0 - x_init = 0
    gL_list.append(np.zeros(n, dtype=np.float64))
    gU_list.append(np.zeros(n, dtype=np.float64))

    knot_evaluators = problem.constraints.knot_evaluators if problem.constraints is not None else ()

    for k in range(N - 1):
        # 2a. Dynamics defect k: x_{k+1} - f_d(x_k, u_k) = 0
        gL_list.append(np.zeros(n, dtype=np.float64))
        gU_list.append(np.zeros(n, dtype=np.float64))

        # 2b. Stage constraints at knot k
        if k < len(knot_evaluators):
            lo_k, hi_k = _evaluator_bounds(knot_evaluators[k])
            gL_list.extend(lo_k)
            gU_list.extend(hi_k)

    # 3. Terminal constraints at knot N - 1
    if len(knot_evaluators) > N - 1:
        lo_term, hi_term = _evaluator_bounds(knot_evaluators[N - 1])
        gL_list.extend(lo_term)
        gU_list.extend(hi_term)

    gL = np.concatenate(gL_list) if gL_list else np.empty(0, dtype=np.float64)
    gU = np.concatenate(gU_list) if gU_list else np.empty(0, dtype=np.float64)
    return gL, gU
