from collections.abc import Sequence

import numpy as np


def jacobian_sparsity_pattern(
    N: int,
    n: int,
    m: int,
    p: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute build-time COO sparsity pattern for the transcribed constraint Jacobian.

    Parameters
    ----------
    N : int
        Horizon length in knot points.
    n : int
        State dimension.
    m : int
        Control dimension.
    p : Sequence[int]
        Sequence of length N specifying total constraint dimension at each knot point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row and column index arrays (rows, cols) of shape (nnz_jac,).
    """
    N_int = int(N)
    n_int = int(n)
    m_int = int(m)
    p_seq = tuple(int(pk) for pk in p)

    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []
    curr_row = 0

    # 1. Initial state condition: x0 - x_init (shape n x n at col 0..n-1)
    row_list.append(np.repeat(np.arange(curr_row, curr_row + n_int), n_int))
    col_list.append(np.tile(np.arange(0, n_int), n_int))
    curr_row += n_int

    # 2. Intermediate stages k = 0, ..., N - 2
    for k in range(N_int - 1):
        col_k = k * (n_int + m_int)
        col_next = (k + 1) * (n_int + m_int)

        # 2a. Dynamics defect wrt (x_k, u_k): shape n x (n + m)
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n_int), n_int + m_int))
        col_list.append(np.tile(np.arange(col_k, col_k + n_int + m_int), n_int))

        # 2b. Dynamics defect wrt x_{k+1}: shape n x n
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n_int), n_int))
        col_list.append(np.tile(np.arange(col_next, col_next + n_int), n_int))
        curr_row += n_int

        # 2c. Stage constraints at knot k: shape p[k] x (n + m)
        pk = p_seq[k]
        if pk > 0:
            row_list.append(np.repeat(np.arange(curr_row, curr_row + pk), n_int + m_int))
            col_list.append(np.tile(np.arange(col_k, col_k + n_int + m_int), pk))
            curr_row += pk

    # 3. Terminal constraints at knot N - 1: shape p[N-1] x n
    p_term = p_seq[N_int - 1]
    if p_term > 0:
        col_term = (N_int - 1) * (n_int + m_int)
        row_list.append(np.repeat(np.arange(curr_row, curr_row + p_term), n_int))
        col_list.append(np.tile(np.arange(col_term, col_term + n_int), p_term))
        curr_row += p_term

    rows = np.concatenate(row_list).astype(np.int32) if row_list else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_list).astype(np.int32) if col_list else np.empty(0, dtype=np.int32)
    return rows, cols


def hessian_sparsity_pattern(
    N: int,
    n: int,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute build-time COO sparsity pattern for the lower triangle of the block-diagonal Hessian.

    Parameters
    ----------
    N : int
        Horizon length in knot points.
    n : int
        State dimension.
    m : int
        Control dimension.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row and column index arrays (rows, cols) of shape (nnz_hess,).
    """
    N_int = int(N)
    n_int = int(n)
    m_int = int(m)

    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    # 1. Stage knots k = 0, ..., N - 2: block size d_k = n + m
    d_stage = n_int + m_int
    tril_r_stage, tril_c_stage = np.tril_indices(d_stage)

    for k in range(N_int - 1):
        col_offset = k * d_stage
        row_list.append(col_offset + tril_r_stage)
        col_list.append(col_offset + tril_c_stage)

    # 2. Terminal knot N - 1: block size d_term = n
    col_offset_term = (N_int - 1) * d_stage
    tril_r_term, tril_c_term = np.tril_indices(n_int)
    row_list.append(col_offset_term + tril_r_term)
    col_list.append(col_offset_term + tril_c_term)

    rows = np.concatenate(row_list).astype(np.int32) if row_list else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_list).astype(np.int32) if col_list else np.empty(0, dtype=np.int32)
    return rows, cols
