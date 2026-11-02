from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from trajopt.transcription.layout import canonical_to_nlp_col


def _canonical_jacobian_sparsity(  # noqa: PLR0913, PLR0917 -- canonical sparsity helper takes 6 arguments
    N: int,
    n: int,
    m: int,
    p_seq: tuple[int, ...],
    horizon_columns: Sequence[tuple[int, ...]],
    horizon_rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute build-time COO sparsity pattern for the canonical constraint Jacobian."""
    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []
    curr_row = 0

    # 1. Initial state condition: x0 - x_init (shape n x n at col 0..n-1)
    row_list.append(np.repeat(np.arange(curr_row, curr_row + n), n))
    col_list.append(np.tile(np.arange(0, n), n))
    curr_row += n

    # 2. Intermediate stages k = 0, ..., N - 2
    for k in range(N - 1):
        col_k = k * (n + m)
        col_next = (k + 1) * (n + m)

        # 2a. Dynamics defect wrt (x_k, u_k): shape n x (n + m)
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n), n + m))
        col_list.append(np.tile(np.arange(col_k, col_k + n + m), n))

        # 2b. Dynamics defect wrt x_{k+1}: shape n x n
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n), n))
        col_list.append(np.tile(np.arange(col_next, col_next + n), n))
        curr_row += n

        # 2c. Stage constraints at knot k: shape p[k] x (n + m)
        pk = p_seq[k]
        if pk > 0:
            row_list.append(np.repeat(np.arange(curr_row, curr_row + pk), n + m))
            col_list.append(np.tile(np.arange(col_k, col_k + n + m), pk))
            curr_row += pk

    # 3. Terminal constraints at knot N - 1: shape p[N-1] x n
    p_term = p_seq[N - 1]
    if p_term > 0:
        col_term = (N - 1) * (n + m)
        row_list.append(np.repeat(np.arange(curr_row, curr_row + p_term), n))
        col_list.append(np.tile(np.arange(col_term, col_term + n), p_term))
        curr_row += p_term

    for columns, count in zip(horizon_columns, horizon_rows, strict=True):
        row_list.append(np.repeat(np.arange(curr_row, curr_row + count), len(columns)))
        col_list.append(np.tile(np.asarray(columns), count))
        curr_row += count

    rows = np.concatenate(row_list).astype(np.int32) if row_list else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_list).astype(np.int32) if col_list else np.empty(0, dtype=np.int32)
    return rows, cols


def _blocked_jacobian_sparsity(  # noqa: PLR0913, PLR0917 -- blocked sparsity helper takes 7 arguments
    N: int,
    n: int,
    m: int,
    b: int,
    p_seq: tuple[int, ...],
    horizon_columns: Sequence[tuple[int, ...]],
    horizon_rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute build-time COO sparsity pattern for the blocked constraint Jacobian."""
    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []
    curr_row = 0

    H = N - 1
    B = H // b
    d_blk = b * n + m

    # 1. Initial state condition: x0 - x_init (shape n x n at col 0..n-1)
    row_list.append(np.repeat(np.arange(curr_row, curr_row + n), n))
    col_list.append(np.tile(np.arange(0, n), n))
    curr_row += n

    # 2. Intermediate stages j = 0, ..., H - 1
    for j in range(H):
        k = j // b
        s = j % b
        offset_k = k * d_blk
        col_xj = offset_k + s * n
        col_uk = offset_k + b * n
        col_xnext = offset_k + (s + 1) * n if s < b - 1 else ((k + 1) * d_blk if k < B - 1 else B * d_blk)

        # 2a. Dynamics defect wrt x_j: shape n x n
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n), n))
        col_list.append(np.tile(np.arange(col_xj, col_xj + n), n))

        # 2b. Dynamics defect wrt u_{blk, k}: shape n x m
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n), m))
        col_list.append(np.tile(np.arange(col_uk, col_uk + m), n))

        # 2c. Dynamics defect wrt x_{j+1}: shape n x n
        row_list.append(np.repeat(np.arange(curr_row, curr_row + n), n))
        col_list.append(np.tile(np.arange(col_xnext, col_xnext + n), n))
        curr_row += n

        # 2d. Stage constraints at knot j: shape p[j] x n and p[j] x m
        pk = p_seq[j]
        if pk > 0:
            row_list.append(np.repeat(np.arange(curr_row, curr_row + pk), n))
            col_list.append(np.tile(np.arange(col_xj, col_xj + n), pk))
            row_list.append(np.repeat(np.arange(curr_row, curr_row + pk), m))
            col_list.append(np.tile(np.arange(col_uk, col_uk + m), pk))
            curr_row += pk

    # 3. Terminal constraints at knot N - 1: shape p[N-1] x n
    p_term = p_seq[N - 1]
    if p_term > 0:
        col_term = B * d_blk
        row_list.append(np.repeat(np.arange(curr_row, curr_row + p_term), n))
        col_list.append(np.tile(np.arange(col_term, col_term + n), p_term))
        curr_row += p_term

    for columns, count in zip(horizon_columns, horizon_rows, strict=True):
        mapped_unique = np.unique([canonical_to_nlp_col(int(c), N, n, m, b) for c in columns])
        row_list.append(np.repeat(np.arange(curr_row, curr_row + count), len(mapped_unique)))
        col_list.append(np.tile(mapped_unique, count))
        curr_row += count

    rows = np.concatenate(row_list).astype(np.int32) if row_list else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_list).astype(np.int32) if col_list else np.empty(0, dtype=np.int32)
    return rows, cols


def jacobian_sparsity_pattern(  # noqa: PLR0913, PLR0917 -- knot and Horizon row layouts define the sparse pattern
    N: int,
    n: int,
    m: int,
    p: Sequence[int],
    horizon_columns: Sequence[tuple[int, ...]] = (),
    horizon_rows: Sequence[int] = (),
    b: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute build-time COO sparsity pattern for the transcribed constraint Jacobian.

    Parameters
    ----------
    N : int
        Horizon length in Knot Points.
    n : int
        State dimension.
    m : int
        Control dimension.
    p : Sequence[int]
        Sequence of length N specifying total constraint dimension at each Knot Point.
    horizon_columns : Sequence[tuple[int, ...]]
        Primal Vector columns for each trailing linear Horizon constraint.
    horizon_rows : Sequence[int]
        Row counts paired with ``horizon_columns``.
    b : int
        Block size in native steps. Defaults to 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row and column index arrays (rows, cols) of shape (nnz_jac,).
    """
    N_int = int(N)
    n_int = int(n)
    m_int = int(m)
    b_int = int(b)
    p_seq = tuple(int(pk) for pk in p)

    if b_int == 1:
        return _canonical_jacobian_sparsity(N_int, n_int, m_int, p_seq, horizon_columns, horizon_rows)
    return _blocked_jacobian_sparsity(N_int, n_int, m_int, b_int, p_seq, horizon_columns, horizon_rows)


def hessian_sparsity_pattern(
    N: int,
    n: int,
    m: int,
    b: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute build-time COO sparsity pattern for the lower triangle of the Lagrangian Hessian.

    Parameters
    ----------
    N : int
        Horizon length in Knot Points.
    n : int
        State dimension.
    m : int
        Control dimension.
    b : int
        Block size in native steps. Defaults to 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row and column index arrays (rows, cols) of shape (nnz_hess,).
    """
    N_int = int(N)
    n_int = int(n)
    m_int = int(m)
    b_int = int(b)

    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    if b_int == 1:
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
    else:
        H = N_int - 1
        B = H // b_int
        d_blk = b_int * n_int + m_int
        tril_r_n, tril_c_n = np.tril_indices(n_int)
        tril_r_m, tril_c_m = np.tril_indices(m_int)
        mixed_r = np.repeat(np.arange(m_int), n_int)
        mixed_c = np.tile(np.arange(n_int), m_int)

        for k in range(B):
            offset_k = k * d_blk

            # 1. State-state lower triangles for stages s = 0, ..., b - 1
            for s in range(b_int):
                row_list.append(offset_k + s * n_int + tril_r_n)
                col_list.append(offset_k + s * n_int + tril_c_n)

            # 2. Mixed state-control entries (m x n) for stages s = 0, ..., b - 1
            for s in range(b_int):
                row_list.append(offset_k + b_int * n_int + mixed_r)
                col_list.append(offset_k + s * n_int + mixed_c)

            # 3. Block control lower triangle (m x m)
            row_list.append(offset_k + b_int * n_int + tril_r_m)
            col_list.append(offset_k + b_int * n_int + tril_c_m)

        # 4. Terminal knot N - 1: block size n
        col_offset_term = B * d_blk
        row_list.append(col_offset_term + tril_r_n)
        col_list.append(col_offset_term + tril_c_n)

    rows = np.concatenate(row_list).astype(np.int32) if row_list else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_list).astype(np.int32) if col_list else np.empty(0, dtype=np.int32)
    return rows, cols


def stacked_constraint_pattern(
    jr: np.ndarray,
    jc: np.ndarray,
    n_con: int,
    nz: int,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the canonical CSC template for [J; I] and the Jacobian data mapping.

    Parameters
    ----------
    jr, jc : np.ndarray
        Row and column index arrays of shape ``(nnz_jac,)``.
    n_con : int
        Number of constraint rows in the Jacobian.
    nz : int
        Dimension of the flat Primal Vector.

    Returns
    -------
    tuple[sp.csc_matrix, np.ndarray]
        Canonical CSC matrix of shape ``(n_con + nz, nz)`` with identity entries set to 1.0,
        and an integer index array of shape ``(nnz_jac,)`` mapping Jacobian entries to data offsets.
    """
    all_r = np.concatenate([jr, n_con + np.arange(nz, dtype=np.int32)])
    all_c = np.concatenate([jc, np.arange(nz, dtype=np.int32)])
    perm = np.lexsort((all_r, all_c))
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    jac_map = inv_perm[: len(jr)]
    col_counts = np.bincount(all_c, minlength=nz)
    indptr = np.r_[0, np.cumsum(col_counts)]
    indices = all_r[perm]
    data = np.zeros(len(all_r), dtype=np.float64)
    data[inv_perm[len(jr) :]] = 1.0
    template = sp.csc_matrix((data, indices, indptr), shape=(n_con + nz, nz))
    return template, jac_map
