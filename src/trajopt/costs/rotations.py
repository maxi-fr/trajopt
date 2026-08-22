from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.costs.base import CostFunction

_QUAT_DIM = 4


class QuatGeodesicCost(CostFunction):
    """Geodesic quaternion cost with double-cover branch.

    Cost formula:
    l(x, u, t) = 0.5 * x^T Q x + 0.5 * u^T R u + q_lin^T x + r_lin^T u + c
                 + w * min(1 + q_ref^T q, 1 - q_ref^T q)

    For terminal costs:
    l(x, t) = 0.5 * x^T Q x + q_lin^T x + c
              + w * min(1 + q_ref^T q, 1 - q_ref^T q)

    Parameters
    ----------
    Q : jax.Array | Sequence[float]
        State weights of shape (n,) if diagonal or (n, n) if dense.
    R : jax.Array | Sequence[float] | None, optional
        Control weights of shape (m,) if diagonal or (m, m) if dense. Required if not terminal.
    q_ref : jax.Array | Sequence[float]
        Reference unit quaternion [qx, qy, qz, qw] (JPL scalar-last) of shape (4,).
    w : float | jax.Array, optional
        Weight for the geodesic attitude penalty. Defaults to 1.0.
    q_lin : jax.Array | Sequence[float] | None, optional
        Linear state cost of shape (n,). Defaults to zeros.
    r_lin : jax.Array | Sequence[float] | None, optional
        Linear control cost of shape (m,). Defaults to zeros.
    c : float | jax.Array, optional
        Constant cost. Defaults to 0.0.
    qind : Sequence[int], optional
        State indices for the quaternion. Defaults to (3, 4, 5, 6).
    terminal : bool, optional
        Whether this is a terminal cost. Defaults to False.
    m : int | None, optional
        Control dimension. Defaults to 0 if terminal.
    """

    Q: jax.Array
    R: jax.Array
    q_ref: jax.Array
    w: jax.Array
    q_lin: jax.Array
    r_lin: jax.Array
    c: jax.Array
    qind_arr: jax.Array
    qind: tuple[int, ...] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        Q: Sequence[float] | jax.Array,
        R: Sequence[float] | jax.Array | None = None,
        q_ref: Sequence[float] | jax.Array = (0.0, 0.0, 0.0, 1.0),
        w: float | jax.Array = 1.0,
        q_lin: Sequence[float] | jax.Array | None = None,
        r_lin: Sequence[float] | jax.Array | None = None,
        c: float | jax.Array = 0.0,
        *,
        qind: Sequence[int] = (3, 4, 5, 6),
        terminal: bool = False,
        m: int | None = None,
    ) -> None:
        Q_arr = jnp.asarray(Q, dtype=float)
        n = int(Q_arr.shape[-1])
        qind_tuple = tuple(int(i) for i in qind)

        if len(qind_tuple) != _QUAT_DIM:
            msg = f"qind must have length 4, got {len(qind_tuple)}"
            raise ValueError(msg)

        q_ref_arr = np.asarray(q_ref, dtype=float)
        q_ref_norm = np.linalg.norm(q_ref_arr)
        if q_ref_norm > 0:
            q_ref_arr = q_ref_arr / q_ref_norm

        if terminal:
            m_val = 0 if m is None else int(m)
            R_arr = jnp.zeros((m_val,), dtype=Q_arr.dtype)
            r_lin_arr = jnp.zeros((m_val,), dtype=Q_arr.dtype)
        else:
            if R is None:
                msg = "Control cost weights R must be provided for non-terminal cost."
                raise ValueError(msg)
            R_arr = jnp.asarray(R, dtype=float)
            m_val = int(R_arr.shape[-1])
            r_lin_arr = jnp.zeros_like(R_arr) if r_lin is None else jnp.asarray(r_lin, dtype=R_arr.dtype)

        q_lin_arr = jnp.zeros_like(Q_arr) if q_lin is None else jnp.asarray(q_lin, dtype=Q_arr.dtype)
        c_arr = jnp.asarray(c, dtype=Q_arr.dtype)

        super().__init__(n=n, m=m_val, terminal=terminal)
        self.Q = Q_arr
        self.R = R_arr
        self.q_ref = jnp.asarray(q_ref_arr)
        self.w = jnp.asarray(w, dtype=float)
        self.q_lin = q_lin_arr
        self.r_lin = r_lin_arr
        self.c = c_arr
        self.qind = qind_tuple
        self.qind_arr = jnp.asarray(qind_tuple, dtype=int)

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate scalar cost including geodesic quaternion error penalty of shape ().

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Current time. Defaults to 0.0.

        Returns
        -------
        jax.Array
            Scalar cost value of shape ().
        """
        del t
        x_arr = jnp.asarray(x)
        q = x_arr[self.qind_arr]
        dq = jnp.dot(self.q_ref, q)
        geodesic_val = self.w * jnp.minimum(1.0 + dq, 1.0 - dq)

        quad_x = 0.5 * jnp.sum(self.Q * (x_arr**2)) if self.Q.ndim == 1 else 0.5 * jnp.dot(x_arr, self.Q @ x_arr)
        lin_x = jnp.dot(self.q_lin, x_arr)

        if self.terminal or u is None:
            return quad_x + lin_x + self.c + geodesic_val

        u_arr = jnp.asarray(u)
        quad_u = 0.5 * jnp.sum(self.R * (u_arr**2)) if self.R.ndim == 1 else 0.5 * jnp.dot(u_arr, self.R @ u_arr)
        lin_u = jnp.dot(self.r_lin, u_arr)
        return quad_x + quad_u + lin_x + lin_u + self.c + geodesic_val

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate stage costs over stacked trajectory arrays of shape (N-1,).

        Parameters
        ----------
        X : jax.Array
            Stacked state vectors of shape (N-1, n).
        U : jax.Array
            Stacked control vectors of shape (N-1, m).
        t : jax.Array
            Stacked timestamps of shape (N-1,).

        Returns
        -------
        jax.Array
            Stacked stage costs of shape (N-1,).
        """
        return jax.vmap(self.evaluate)(X, U, t)


def LieLQRCost(  # noqa: N802, PLR0913 -- LieLQRCost requires parameters matching optimal control specification
    Q: Sequence[float] | jax.Array,
    R: Sequence[float] | jax.Array,
    xf: Sequence[float] | jax.Array,
    uf: Sequence[float] | jax.Array | None = None,
    w: float | jax.Array | None = None,
    *,
    qind: Sequence[int] = (3, 4, 5, 6),
    terminal: bool = False,
) -> QuatGeodesicCost:
    """Construct an LQR tracking cost with geodesic quaternion penalty for rigid-body states.

    Parameters
    ----------
    Q : Sequence[float] | jax.Array
        State weights of shape (n,) or (ne,) or (9,).
    R : Sequence[float] | jax.Array
        Control weights of shape (m,).
    xf : Sequence[float] | jax.Array
        Reference state of shape (n,).
    uf : Sequence[float] | jax.Array | None, optional
        Reference control of shape (m,). Defaults to zeros.
    w : float | jax.Array | None, optional
        Attitude geodesic weight. Defaults to sum of attitude Q weights or 1.0.
    qind : Sequence[int], optional
        Indices of quaternion in state vector. Defaults to (3, 4, 5, 6).
    terminal : bool, optional
        Whether this is a terminal cost. Defaults to False.

    Returns
    -------
    QuatGeodesicCost
        Configured QuatGeodesicCost instance.
    """
    xf_arr = jnp.asarray(xf, dtype=float)
    n = int(xf_arr.shape[0])
    qind_tuple = tuple(int(i) for i in qind)
    q_ref = xf_arr[jnp.asarray(qind_tuple, dtype=int)]

    Q_in = jnp.asarray(Q, dtype=float)
    Q_full = jnp.zeros(n, dtype=float)

    if Q_in.shape == (n,):
        # 13 states: zero out quaternion indices in Euclidean quadratic term
        w_default = jnp.sum(Q_in[jnp.asarray(qind_tuple, dtype=int)])
        mask = jnp.ones(n, dtype=bool).at[jnp.asarray(qind_tuple, dtype=int)].set(False)
        Q_full = jnp.where(mask, Q_in, 0.0)
    elif Q_in.shape == (12,):
        # 12 error states [r(3), theta(3), v(3), omega(3)]
        w_default = jnp.sum(Q_in[3:6])
        Q_full = Q_full.at[:3].set(Q_in[:3])
        Q_full = Q_full.at[7:10].set(Q_in[6:9])
        Q_full = Q_full.at[10:13].set(Q_in[9:12])
    elif Q_in.shape == (9,):
        # 9 vector states [r(3), v(3), omega(3)]
        w_default = 1.0
        Q_full = Q_full.at[:3].set(Q_in[:3])
        Q_full = Q_full.at[7:10].set(Q_in[3:6])
        Q_full = Q_full.at[10:13].set(Q_in[6:9])
    else:
        w_default = 1.0
        Q_full = Q_in

    w_val = w_default if w is None else jnp.asarray(w, dtype=float)

    if terminal:
        q_lin = -Q_full * xf_arr
        c_val = 0.5 * jnp.sum(Q_full * (xf_arr**2))
        return QuatGeodesicCost(
            Q=Q_full,
            R=None,
            q_ref=q_ref,
            w=w_val,
            q_lin=q_lin,
            c=c_val,
            qind=qind_tuple,
            terminal=True,
            m=0,
        )

    R_arr = jnp.asarray(R, dtype=float)
    m = int(R_arr.shape[-1])
    uf_arr = jnp.zeros(m, dtype=float) if uf is None else jnp.asarray(uf, dtype=float)

    q_lin = -Q_full * xf_arr
    r_lin = -R_arr * uf_arr
    c_val = 0.5 * jnp.sum(Q_full * (xf_arr**2)) + 0.5 * jnp.sum(R_arr * (uf_arr**2))

    return QuatGeodesicCost(
        Q=Q_full,
        R=R_arr,
        q_ref=q_ref,
        w=w_val,
        q_lin=q_lin,
        r_lin=r_lin,
        c=c_val,
        qind=qind_tuple,
        terminal=False,
        m=m,
    )
