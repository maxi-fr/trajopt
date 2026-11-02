from typing import Any

import numpy as np

from trajopt.transcription.result import split_bound_duals

_SUCCESS_STATUSES = {0, 1}  # 0: Solve_Succeeded, 1: Solved_To_Acceptable_Level
_EMPTY = np.zeros(0, dtype=np.float64)
_MIN_HOMOTOPY_PASSES = 2


def validate_homotopy_parameters(
    beta_initial: float,
    beta_growth: float,
    max_passes: int,
    binary_tolerance: float,
) -> None:
    """Validate binary continuation parameters for positive beta/tolerance, growth > 1, and at least two passes."""
    if (
        beta_initial <= 0
        or beta_growth <= 1
        or max_passes < _MIN_HOMOTOPY_PASSES
        or binary_tolerance <= 0
    ):
        msg = "Binary homotopy needs positive beta/tolerance, growth > 1, and at least two passes."
        raise ValueError(msg)


def solve_continuation_passes(  # noqa: PLR0913 -- continuation loop coordinates solver handles and warm starts
    nlp: Any,  # noqa: ANN401 -- cyipopt's C-extension handle is untyped
    cb: Any,  # noqa: ANN401 -- callback object with beta attribute
    z0: np.ndarray,
    lam0: np.ndarray | None,
    mu0: np.ndarray | None,
    *,
    binary_indices: np.ndarray,
    beta_initial: float = 1.0,
    beta_growth: float = 10.0,
    max_passes: int = 8,
    binary_tolerance: float = 1e-4,
) -> tuple[np.ndarray, dict[str, Any], int, float]:
    """Execute continuation schedule carrying Primal Vector and Multipliers between passes."""
    n_passes = max_passes if len(binary_indices) else 1
    binary_distance = 0.0
    info: dict[str, Any] = {}
    pass_index = 0
    for pass_index in range(n_passes):
        cb.beta = 0.0 if pass_index == 0 else beta_initial * (beta_growth ** (pass_index - 1))
        solve_kwargs: dict[str, Any] = {}
        if mu0 is not None and len(mu0) == len(z0):
            mult_x_l, mult_x_u = split_bound_duals(mu0)
            solve_kwargs["zl"] = mult_x_l
            solve_kwargs["zu"] = mult_x_u
        if lam0 is not None and len(lam0) > 0:
            solve_kwargs["lagrange"] = lam0

        if solve_kwargs:
            nlp.add_option("warm_start_init_point", "yes")
            nlp.add_option("warm_start_bound_push", 1e-9)
            nlp.add_option("warm_start_mult_bound_push", 1e-9)
            z_opt, raw_info = nlp.solve(z0, **solve_kwargs)
        else:
            nlp.add_option("warm_start_init_point", "no")
            z_opt, raw_info = nlp.solve(z0)

        info = dict(raw_info)
        z0 = np.asarray(z_opt, dtype=np.float64)
        lam0 = np.asarray(info.get("mult_g", _EMPTY), dtype=np.float64)
        mult_x_u_arr = np.asarray(info.get("mult_x_U", _EMPTY), dtype=np.float64)
        mult_x_l_arr = np.asarray(info.get("mult_x_L", _EMPTY), dtype=np.float64)
        mu0 = (
            mult_x_u_arr - mult_x_l_arr
            if len(mult_x_u_arr) == len(z0) and len(mult_x_l_arr) == len(z0)
            else None
        )

        a = z0[binary_indices]
        binary_distance = float(np.max(np.minimum(np.abs(a), np.abs(1 - a)))) if len(a) else 0.0
        if int(info.get("status", -1)) not in _SUCCESS_STATUSES:
            break
        if pass_index > 0 and binary_distance <= binary_tolerance:
            break
    return z0, info, pass_index + 1, binary_distance
