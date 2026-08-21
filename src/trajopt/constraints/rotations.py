"""Rotational and attitude constraints."""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.cones import ZeroCone
from trajopt.constraints.base import StateConstraint

QUAT_DIM = 4
QUAT_DIFF_DIM = 3


class QuatVecEq(StateConstraint):
    """Quaternion attitude equality constraint: q =~ qf.

    Enforces alignment of unit quaternions in JPL convention (scalar-last [x, y, z, w])
    by penalizing the difference in vector parts, accounting for the double cover (q = -q).

    Parameters
    ----------
    n : int
        State dimension.
    qf : jax.Array | Sequence[float]
        Target unit quaternion [qx, qy, qz, qw] (length 4).
    qind : Sequence[int], optional
        State indices for the quaternion (default (3, 4, 5, 6)).
    m : int, optional
        Control dimension (default 0).
    """

    qf: jax.Array
    qind_arr: jax.Array
    qind: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        qf: jax.Array | Sequence[float],
        qind: Sequence[int] = (3, 4, 5, 6),
        *,
        m: int = 0,
    ) -> None:
        n_int = int(n)
        m_int = int(m)
        qind_tuple = tuple(int(i) for i in qind)

        if len(qind_tuple) != QUAT_DIM:
            msg = f"Quaternion indices qind must have length 4, got {len(qind_tuple)}."
            raise ValueError(msg)

        qf_arr = np.asarray(qf, dtype=float)
        if len(qf_arr) != QUAT_DIM:
            msg = f"Target quaternion qf must have length 4, got {len(qf_arr)}."
            raise ValueError(msg)

        # Normalize qf
        qf_norm = np.linalg.norm(qf_arr)
        if qf_norm > 0:
            qf_arr = qf_arr / qf_norm

        super().__init__(n=n_int, m=m_int, p=QUAT_DIFF_DIM, cone=ZeroCone())
        self.qf = jnp.asarray(qf_arr)
        self.qind = qind_tuple
        self.qind_arr = jnp.asarray(qind_tuple, dtype=int)

    def evaluate(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate quaternion vector difference defect."""
        del u, t
        if x is None:
            msg = "State vector x is required to evaluate QuatVecEq."
            raise ValueError(msg)

        q = x[self.qind_arr]
        norm_q = jnp.linalg.norm(q)
        safe_norm = jnp.maximum(norm_q, 1e-12)
        q_unit = q / safe_norm

        dq = jnp.dot(self.qf, q_unit)
        qf_sign = jnp.where(dq < 0.0, -self.qf, self.qf)

        return -(qf_sign[:3] - q_unit[:3])
