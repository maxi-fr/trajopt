from dataclasses import dataclass
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp

from trajopt.transcription.result import SolverStatus


class TerminationStatus(IntEnum):
    """Why a native solver's loop stopped, mirroring Altro's `@enum(TerminationStatus, ...)`.

    Member order is load-bearing, not cosmetic: Altro compares ordinals directly
    (`status > SOLVE_SUCCEEDED`, `status <= SOLVE_SUCCEEDED`) to decide whether the augmented-
    Lagrangian outer loop breaks and whether ALTRO runs its polish phase (finding C). `LINESEARCH_FAIL`
    is never set by any code path but must keep ordinal 1, since dropping it would shift every
    later member and change what those comparisons mean.
    """

    UNSOLVED = 0
    LINESEARCH_FAIL = 1
    SOLVE_SUCCEEDED = 2
    MAX_ITERATIONS = 3
    MAX_ITERATIONS_OUTER = 4
    MAXIMUM_COST = 5
    STATE_LIMIT = 6
    CONTROL_LIMIT = 7
    NO_PROGRESS = 8
    COST_INCREASE = 9


_STATUS_MAP: dict[TerminationStatus, SolverStatus] = {
    TerminationStatus.SOLVE_SUCCEEDED: "converged",
    TerminationStatus.MAX_ITERATIONS_OUTER: "infeasible",
    TerminationStatus.MAXIMUM_COST: "infeasible",
    TerminationStatus.UNSOLVED: "infeasible",
    TerminationStatus.MAX_ITERATIONS: "iteration_limit",
    TerminationStatus.NO_PROGRESS: "iteration_limit",
    TerminationStatus.LINESEARCH_FAIL: "iteration_limit",
    TerminationStatus.STATE_LIMIT: "error",
    TerminationStatus.CONTROL_LIMIT: "error",
    TerminationStatus.COST_INCREASE: "error",
}


def to_solver_status(status: TerminationStatus) -> SolverStatus:
    """Map an internal `TerminationStatus` onto the public four-value `SolverStatus`.

    Follows reference doc §2's table exactly, preserving the precise Altro exit reason inside
    the solver's own stats while normalizing to the vocabulary the rest of the codebase (Ipopt,
    OSQP, Clarabel adapters) already speaks.
    """
    return _STATUS_MAP[status]


@dataclass(frozen=True)
class SolverOptions:
    """Static configuration for a native iLQR / AL / ALTRO solve.

    A frozen dataclass rather than an `eqx.Module`: these values pick shapes and loop bounds
    during tracing, so making them a pytree would invite someone to trace a tolerance. Field
    names and defaults port `altro_jl/src/solver_opts.jl`'s `SolverOptions` one-for-one, minus
    the eleven options that are never read anywhere in `altro_jl/src/` (overview finding F:
    `iterations_inner`, `bp_reg`, `square_root`, `save_S`, `static_bp`, `reuse_jacobians`,
    `active_set_tolerance_al`, `rho_chol`, `rho_dual`, `solve_type`, plus the logging-only
    `show_summary` / `trim_stats`), `bp_reg_type` (finding H: only `:control` ever executes, so
    the option is dropped and that behaviour is unconditional), `closed_loop_initial_rollout`
    (ticket 27: discarded along with Altro's docstring claim that cached gains are reused), and
    `dynamics_funsig` / `dynamics_diffmethod` (Julia multiple-dispatch knobs for choosing a
    differentiation strategy; JAX's `jax.grad` needs no such choice, so there is nothing to port).

    `bp_reg_max` is also in Altro's dead list -- upstream never reads it, so a repeatedly
    failing Cholesky factorization raises the backward-pass regularization forever. That is
    merely slow in Altro's `while true` loop; under `lax.while_loop` it is an unkillable hang.
    This port keeps `bp_reg_max` and makes it live as ticket 25's regularization loop bound, a
    deliberate divergence from Altro rather than a straight port.
    """

    # Optimality tolerances
    constraint_tolerance: float = 1e-6
    cost_tolerance: float = 1e-4
    cost_tolerance_intermediate: float = 1e-4
    gradient_tolerance: float = 10.0
    gradient_tolerance_intermediate: float = 1.0

    # iLQR
    expected_decrease_tolerance: float = 1e-10
    dJ_counter_limit: int = 10  # noqa: N815 -- ports Altro's `dJ_counter_limit` field name verbatim
    line_search_lower_bound: float = 1e-8
    line_search_upper_bound: float = 10.0
    line_search_decrease_factor: float = 0.5
    iterations_linesearch: int = 20
    max_cost_value: float = 1.0e8
    max_state_value: float = 1.0e8
    max_control_value: float = 1.0e8

    # Backward pass regularization
    bp_reg_initial: float = 0.0
    bp_reg_increase_factor: float = 1.6
    bp_reg_max: float = 1.0e8  # live here (loop bound); dead in Altro (finding F) -- see class docstring
    bp_reg_min: float = 1.0e-8
    bp_reg_fp: float = 10.0

    # Augmented Lagrangian
    use_conic_cost: bool = False
    penalty_initial: float = 1.0
    penalty_scaling: float = 10.0
    penalty_max: float = 1e8
    dual_max: float = 1e8
    iterations_outer: int = 30
    kickout_max_penalty: bool = False
    reset_duals: bool = True
    reset_penalties: bool = True

    # Projected Newton
    force_pn: bool = False
    verbose_pn: bool = False
    n_steps: int = 2
    projected_newton_tolerance: float = 1e-3
    active_set_tolerance_pn: float = 1e-3
    multiplier_projection: bool = True
    rho_primal: float = 1.0e-8
    r_threshold: float = 1.1

    # General
    projected_newton: bool = True
    iterations: int = 1000
    verbose: int = 0


class SolverStats(eqx.Module):
    """Fixed-size pytree of per-iteration solve statistics, traceable end to end.

    A traced `lax.while_loop` cannot append, so each history is preallocated at
    `options.iterations` and written at the current iteration with `.at[i].set(...)`; trimming
    to the counter happens only in the eager wrapper at the Python boundary, never inside the
    trace. Mirrors `altro_jl/src/solver_opts.jl`'s `SolverStats`, minus the fields that are
    Julia-side bookkeeping (`iteration`, `iteration_outer`, `iteration_pn`, `tstart`, `tsolve`,
    `to`, `is_reset`, `parent`) rather than solver state a JAX loop carries.

    Parameters
    ----------
    iterations : jax.Array
        Number of completed iterations so far, as an int32 scalar.
    cost : jax.Array
        Cost history of shape `(options.iterations,)`.
    dJ : jax.Array
        Predicted-minus-actual cost decrease history of shape `(options.iterations,)`.
    c_max : jax.Array
        Max constraint violation history of shape `(options.iterations,)`.
    gradient : jax.Array
        Normalized feedforward gradient history of shape `(options.iterations,)`.
    penalty_max : jax.Array
        Max AL penalty history of shape `(options.iterations,)`.
    dJ_zero_counter : jax.Array
        Consecutive-iteration count of near-zero cost decrease, as an int32 scalar.
    ls_failed : jax.Array
        Whether the most recent line search failed, as a bool scalar.
    """

    iterations: jax.Array
    cost: jax.Array
    dJ: jax.Array  # noqa: N815 -- ports Altro's `dJ` field name verbatim
    c_max: jax.Array
    gradient: jax.Array
    penalty_max: jax.Array
    dJ_zero_counter: jax.Array  # noqa: N815 -- ports Altro's `dJ_zero_counter` field name verbatim
    ls_failed: jax.Array

    @classmethod
    def create(cls, options: SolverOptions) -> "SolverStats":
        """Allocate zeroed history buffers of length `options.iterations`."""
        history = jnp.zeros(options.iterations, dtype=jnp.float64)
        return cls(
            iterations=jnp.asarray(0, dtype=jnp.int32),
            cost=history,
            dJ=history,
            c_max=history,
            gradient=history,
            penalty_max=history,
            dJ_zero_counter=jnp.asarray(0, dtype=jnp.int32),
            ls_failed=jnp.asarray(False),  # noqa: FBT003 -- traced bool scalar, not a boolean-trap argument
        )
