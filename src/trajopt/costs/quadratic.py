import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.costs.base import QuadraticCostFunction

_EXPECTED_NDIM_1D = 1
_EXPECTED_NDIM_2D = 2
_EXPECTED_NDIM_3D = 3


def diag_matrix(v: jax.Array) -> jax.Array:
    """Embed diagonal weights of shape (..., n) as full matrices of shape (..., n, n)."""
    return v[..., :, None] * jnp.eye(v.shape[-1], dtype=v.dtype)


def _stack_over(param: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Broadcast a single-knot parameter to the stacked shape, leaving an already-stacked one alone.

    Compares ndim rather than the leading length, which a single-knot parameter can match by
    coincidence whenever a state or control dimension equals the number of stages.
    """
    return param if param.ndim == len(shape) else jnp.broadcast_to(param, shape)


class DiagonalCost(QuadraticCostFunction):
    """Diagonal quadratic cost function storing weights as 1D vectors rather than matrices.

    Represents costs of the form:
    0.5 * sum_i(Q_i * x_i^2) + 0.5 * sum_j(R_j * u_j^2) + q^T x + r^T u + c

    For terminal costs:
    0.5 * sum_i(Q_i * x_i^2) + q^T x + c

    Every parameter is either single-knot or stacked over N - 1 stages along a leading axis;
    Q and R must agree on which.

    Parameters
    ----------
    Q : jax.Array
        State weights of shape (n,) or (N-1, n).
    R : jax.Array | None, optional
        Control weights of shape (m,) or (N-1, m). Required if not terminal.
    q : jax.Array | None, optional
        Linear state cost of shape (n,) or (N-1, n). Defaults to zeros.
    r : jax.Array | None, optional
        Linear control cost of shape (m,) or (N-1, m). Defaults to zeros.
    c : float | jax.Array, optional
        Constant cost of shape () or (N-1,). Defaults to 0.0.
    terminal : bool, optional
        Whether this is a terminal cost. Defaults to False.
    m : int | None, optional
        Control dimension for terminal costs. Defaults to 0 if terminal.
    """

    Q: jax.Array
    R: jax.Array
    q: jax.Array
    r: jax.Array
    c: jax.Array

    def __init__(  # noqa: PLR0913
        self,
        Q: jax.Array,
        R: jax.Array | None = None,
        q: jax.Array | None = None,
        r: jax.Array | None = None,
        c: float | jax.Array = 0.0,
        *,
        terminal: bool = False,
        m: int | None = None,
    ) -> None:
        Q_arr = jnp.asarray(Q)
        n = int(Q_arr.shape[-1])

        if terminal:
            m_val = 0 if m is None else int(m)
            R_arr = jnp.zeros((m_val,), dtype=Q_arr.dtype)
            r_arr = jnp.zeros((m_val,), dtype=Q_arr.dtype)
        else:
            if R is None:
                msg = "Control cost weights R must be provided for non-terminal cost."
                raise ValueError(msg)
            R_arr = jnp.asarray(R)
            stacked_mismatch = Q_arr.ndim == _EXPECTED_NDIM_2D and R_arr.shape[0] != Q_arr.shape[0]
            if R_arr.ndim != Q_arr.ndim or stacked_mismatch:
                msg = (
                    f"DiagonalCost weights must both be diagonal vectors stacked the same way, got Q of shape "
                    f"{Q_arr.shape} and R of shape {R_arr.shape}. Use QuadraticCost for dense weights."
                )
                raise ValueError(msg)
            m_val = int(R_arr.shape[-1])
            r_arr = jnp.zeros_like(R_arr) if r is None else jnp.asarray(r, dtype=R_arr.dtype)

        q_arr = jnp.zeros_like(Q_arr) if q is None else jnp.asarray(q, dtype=Q_arr.dtype)
        c_arr = jnp.asarray(c, dtype=Q_arr.dtype)

        # A stacked cost carries the horizon axis on every parameter, so that is_stacked holds
        # of each one and slicing a stage out never has to guess which leaves are stacked
        if Q_arr.ndim == _EXPECTED_NDIM_2D:
            n_stages = int(Q_arr.shape[0])
            q_arr = _stack_over(q_arr, (n_stages, n))
            r_arr = _stack_over(r_arr, (n_stages, m_val))
            c_arr = _stack_over(c_arr, (n_stages,))

        super().__init__(n=n, m=m_val, terminal=terminal)
        self.Q = Q_arr
        self.R = R_arr
        self.q = q_arr
        self.r = r_arr
        self.c = c_arr

    @property
    def is_diag(self) -> bool:
        """Whether the Hessian is strictly diagonal."""
        return True

    @property
    def is_blockdiag(self) -> bool:
        """Whether the Hessian is block diagonal."""
        return True

    @property
    def is_stacked(self) -> bool:
        """Whether the parameters carry a leading horizon axis."""
        return self.Q.ndim == _EXPECTED_NDIM_2D

    @property
    def H(self) -> None:  # noqa: N802
        """Cross-coupling weights, always None because a diagonal cost couples nothing."""
        return None

    @staticmethod
    def matvec(W: jax.Array, v: jax.Array) -> jax.Array:
        """Multiply diagonal weights W of shape (..., n) by vectors v of shape (..., n)."""
        return W * v

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate scalar cost.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Timestamp. Default is 0.0.

        Returns
        -------
        jax.Array
            Scalar cost value.
        """
        del t
        x_arr = jnp.asarray(x)
        if self.terminal or u is None:
            return 0.5 * jnp.sum(self.Q * (x_arr**2)) + jnp.dot(self.q, x_arr) + self.c
        u_arr = jnp.asarray(u)
        return (
            0.5 * jnp.sum(self.Q * (x_arr**2))
            + 0.5 * jnp.sum(self.R * (u_arr**2))
            + jnp.dot(self.q, x_arr)
            + jnp.dot(self.r, u_arr)
            + self.c
        )

    def gradient(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost gradient.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Timestamp. Default is 0.0.

        Returns
        -------
        jax.Array
            Gradient vector of shape (n + m,) or (n,).
        """
        del t
        x_arr = jnp.asarray(x)
        gx = self.Q * x_arr + self.q
        if self.terminal or u is None:
            return gx
        u_arr = jnp.asarray(u)
        gu = self.R * u_arr + self.r
        return jnp.concatenate([gx, gu])

    def hessian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost Hessian.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Diagonal Hessian matrix of shape (n + m, n + m) or (n, n).
        """
        del x, u, t
        if self.terminal:
            return jnp.diag(self.Q)
        return jnp.diag(jnp.concatenate([self.Q, self.R]))

    def invert(self) -> "DiagonalCost":
        """Analytic inverse of the cost function parameters.

        Returns
        -------
        DiagonalCost
            New DiagonalCost instance with inverted weights (1 / Q, 1 / R).
        """
        if self.terminal:
            return DiagonalCost(
                Q=1.0 / self.Q,
                q=self.q,
                c=self.c,
                terminal=True,
                m=self.m,
            )
        return DiagonalCost(
            Q=1.0 / self.Q,
            R=1.0 / self.R,
            q=self.q,
            r=self.r,
            c=self.c,
            terminal=False,
        )

    def hessian_inverse(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the inverse of the Hessian matrix in O(n + m) time.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Inverted Hessian matrix of shape (n + m, n + m) or (n, n).
        """
        del x, u, t
        if self.terminal:
            return jnp.diag(1.0 / self.Q)
        return jnp.diag(jnp.concatenate([1.0 / self.Q, 1.0 / self.R]))

    def __add__(self, other: QuadraticCostFunction) -> QuadraticCostFunction:
        """Add two cost functions.

        Parameters
        ----------
        other : QuadraticCostFunction
            Other cost function to add.

        Returns
        -------
        QuadraticCostFunction
            Sum of the two cost functions.
        """
        if isinstance(other, DiagonalCost):
            if self.n != other.n or self.m != other.m:
                msg = f"Dimension mismatch in cost addition: ({self.n}, {self.m}) vs ({other.n}, {other.m})"
                raise ValueError(msg)
            is_term = self.terminal and other.terminal
            return DiagonalCost(
                Q=self.Q + other.Q,
                R=self.R + other.R if not is_term else None,
                q=self.q + other.q,
                r=self.r + other.r if not is_term else None,
                c=self.c + other.c,
                terminal=is_term,
                m=self.m,
            )
        if isinstance(other, QuadraticCost):
            return self.to_quadratic() + other
        msg = f"Unsupported operand type for +: {type(other).__name__}"
        raise TypeError(msg)

    def to_quadratic(self) -> "QuadraticCost":
        """Convert DiagonalCost to QuadraticCost.

        Returns
        -------
        QuadraticCost
            Equivalent dense QuadraticCost instance.
        """
        if self.terminal:
            return QuadraticCost(
                Q=diag_matrix(self.Q),
                q=self.q,
                c=self.c,
                terminal=True,
                m=self.m,
            )
        return QuadraticCost(
            Q=diag_matrix(self.Q),
            R=diag_matrix(self.R),
            q=self.q,
            r=self.r,
            c=self.c,
            terminal=False,
        )


class QuadraticCost(QuadraticCostFunction):
    """Dense quadratic cost function with cross-coupling support.

    Represents costs of the form:
    0.5 * x^T Q x + 0.5 * u^T R u + u^T H x + q^T x + r^T u + c

    For terminal costs:
    0.5 * x^T Q x + q^T x + c

    Parameters
    ----------
    Q : jax.Array
        State weights of shape (n, n) or (N-1, n, n), or a diagonal vector of shape (n,).
    R : jax.Array | None, optional
        Control weights of shape (m, m) or (N-1, m, m), or a diagonal vector of shape (m,).
        Required if not terminal.
    H : jax.Array | None, optional
        Cross-coupling weights of shape (m, n) or (N-1, m, n). Defaults to zeros.
    q : jax.Array | None, optional
        Linear state cost of shape (n,) or (N-1, n). Defaults to zeros.
    r : jax.Array | None, optional
        Linear control cost of shape (m,) or (N-1, m). Defaults to zeros.
    c : float | jax.Array, optional
        Constant cost of shape () or (N-1,). Defaults to 0.0.
    terminal : bool, optional
        Whether this is a terminal cost. Defaults to False.
    m : int | None, optional
        Control dimension for terminal costs. Defaults to 0 if terminal.
    """

    Q: jax.Array
    R: jax.Array
    H: jax.Array
    q: jax.Array
    r: jax.Array
    c: jax.Array
    has_cross_coupling: bool = eqx.field(static=True)

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        Q: jax.Array,
        R: jax.Array | None = None,
        H: jax.Array | None = None,
        q: jax.Array | None = None,
        r: jax.Array | None = None,
        c: float | jax.Array = 0.0,
        *,
        terminal: bool = False,
        m: int | None = None,
    ) -> None:
        Q_arr = jnp.asarray(Q)
        if Q_arr.ndim == _EXPECTED_NDIM_1D:
            Q_arr = diag_matrix(Q_arr)

        n = int(Q_arr.shape[-1])

        if terminal:
            m_val = 0 if m is None else int(m)
            R_arr = jnp.zeros((m_val, m_val), dtype=Q_arr.dtype)
            H_arr = jnp.zeros((m_val, n), dtype=Q_arr.dtype)
            r_arr = jnp.zeros((m_val,), dtype=Q_arr.dtype)
            has_cross = False
        else:
            if R is None:
                msg = "Control cost matrix R must be provided for non-terminal cost."
                raise ValueError(msg)
            R_arr = jnp.asarray(R)
            if R_arr.ndim == _EXPECTED_NDIM_1D:
                R_arr = diag_matrix(R_arr)
            m_val = int(R_arr.shape[-1])
            if H is None:
                H_arr = jnp.zeros((*Q_arr.shape[:-2], m_val, n), dtype=Q_arr.dtype)
                has_cross = False
            else:
                H_arr = jnp.asarray(H, dtype=Q_arr.dtype)
                try:
                    has_cross = bool(not np.all(np.asarray(H) == 0))
                except Exception:  # noqa: BLE001
                    has_cross = True
            r_arr = jnp.zeros((m_val,), dtype=Q_arr.dtype) if r is None else jnp.asarray(r, dtype=Q_arr.dtype)

        q_arr = jnp.zeros((n,), dtype=Q_arr.dtype) if q is None else jnp.asarray(q, dtype=Q_arr.dtype)
        c_arr = jnp.asarray(c, dtype=Q_arr.dtype)

        # A stacked cost carries the horizon axis on every parameter, so that is_stacked holds
        # of each one and slicing a stage out never has to guess which leaves are stacked
        if Q_arr.ndim == _EXPECTED_NDIM_3D:
            n_stages = int(Q_arr.shape[0])
            R_arr = _stack_over(R_arr, (n_stages, m_val, m_val))
            H_arr = _stack_over(H_arr, (n_stages, m_val, n))
            q_arr = _stack_over(q_arr, (n_stages, n))
            r_arr = _stack_over(r_arr, (n_stages, m_val))
            c_arr = _stack_over(c_arr, (n_stages,))

        super().__init__(n=n, m=m_val, terminal=terminal)
        self.Q = Q_arr
        self.R = R_arr
        self.H = H_arr
        self.q = q_arr
        self.r = r_arr
        self.c = c_arr
        self.has_cross_coupling = has_cross

    @property
    def is_blockdiag(self) -> bool:
        """Whether the Hessian is block diagonal (H = 0)."""
        if self.terminal:
            return True
        return not self.has_cross_coupling

    @property
    def is_stacked(self) -> bool:
        """Whether the parameters carry a leading horizon axis."""
        return self.Q.ndim == _EXPECTED_NDIM_3D

    @staticmethod
    def matvec(W: jax.Array, v: jax.Array) -> jax.Array:
        """Multiply weight matrices W of shape (..., i, j) by vectors v of shape (..., j)."""
        return jnp.einsum("...ij,...j->...i", W, v)

    def to_quadratic(self) -> "QuadraticCost":
        """Return this cost unchanged; it is already dense."""
        return self

    def stage_costs(self, X: jax.Array, U: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the stacked parameters against X (N-1, n), U (N-1, m), t (N-1,), giving (N-1,)."""
        stage_c = super().stage_costs(X, U, t)
        if self.has_cross_coupling:
            stage_c = stage_c + jnp.sum(U * self.matvec(self.H, X), axis=-1)
        return stage_c

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate scalar cost.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Timestamp. Default is 0.0.

        Returns
        -------
        jax.Array
            Scalar cost value.
        """
        del t
        x_arr = jnp.asarray(x)
        if self.terminal or u is None:
            return 0.5 * jnp.dot(x_arr, self.Q @ x_arr) + jnp.dot(self.q, x_arr) + self.c
        u_arr = jnp.asarray(u)
        val = (
            0.5 * jnp.dot(x_arr, self.Q @ x_arr)
            + 0.5 * jnp.dot(u_arr, self.R @ u_arr)
            + jnp.dot(self.q, x_arr)
            + jnp.dot(self.r, u_arr)
            + self.c
        )
        if self.has_cross_coupling:
            val = val + jnp.dot(u_arr, self.H @ x_arr)
        return val

    def gradient(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost gradient.

        Parameters
        ----------
        x : jax.Array
            State vector of shape (n,).
        u : jax.Array | None, optional
            Control vector of shape (m,). None for terminal costs.
        t : float | jax.Array, optional
            Timestamp. Default is 0.0.

        Returns
        -------
        jax.Array
            Gradient vector of shape (n + m,) or (n,).
        """
        del t
        x_arr = jnp.asarray(x)
        gx = self.Q @ x_arr + self.q
        if self.terminal or u is None:
            return gx
        u_arr = jnp.asarray(u)
        gu = self.R @ u_arr + self.r
        if self.has_cross_coupling:
            gx = gx + self.H.T @ u_arr
            gu = gu + self.H @ x_arr
        return jnp.concatenate([gx, gu])

    def hessian(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate cost Hessian.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Dense Hessian matrix of shape (n + m, n + m) or (n, n).
        """
        del x, u, t
        if self.terminal:
            return self.Q
        top = jnp.hstack([self.Q, self.H.T])
        bot = jnp.hstack([self.H, self.R])
        return jnp.vstack([top, bot])

    def invert(self) -> "QuadraticCost":
        """Analytic inverse of the cost function parameters.

        Returns
        -------
        QuadraticCost
            New QuadraticCost instance with inverted matrix parameters.
        """
        if self.terminal:
            Q_inv = jax.vmap(jnp.linalg.inv)(self.Q) if self.Q.ndim == _EXPECTED_NDIM_3D else jnp.linalg.inv(self.Q)
            return QuadraticCost(
                Q=Q_inv,
                q=self.q,
                c=self.c,
                terminal=True,
                m=self.m,
            )

        if not self.has_cross_coupling:
            if self.Q.ndim == _EXPECTED_NDIM_3D:
                Q_inv = jax.vmap(jnp.linalg.inv)(self.Q)
                R_inv = jax.vmap(jnp.linalg.inv)(self.R)
            else:
                Q_inv = jnp.linalg.inv(self.Q)
                R_inv = jnp.linalg.inv(self.R)
            return QuadraticCost(
                Q=Q_inv,
                R=R_inv,
                H=self.H,
                q=self.q,
                r=self.r,
                c=self.c,
                terminal=False,
            )

        if self.Q.ndim == _EXPECTED_NDIM_3D:

            def inv_knot(Q_k: jax.Array, R_k: jax.Array, H_k: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
                G = jnp.block([[Q_k, H_k.T], [H_k, R_k]])
                G_inv = jnp.linalg.inv(G)
                return G_inv[: self.n, : self.n], G_inv[self.n :, self.n :], G_inv[self.n :, : self.n]

            Q_inv, R_inv, H_inv = jax.vmap(inv_knot)(self.Q, self.R, self.H)
            return QuadraticCost(
                Q=Q_inv,
                R=R_inv,
                H=H_inv,
                q=self.q,
                r=self.r,
                c=self.c,
                terminal=False,
            )

        G = jnp.block([[self.Q, self.H.T], [self.H, self.R]])
        G_inv = jnp.linalg.inv(G)
        return QuadraticCost(
            Q=G_inv[: self.n, : self.n],
            R=G_inv[self.n :, self.n :],
            H=G_inv[self.n :, : self.n],
            q=self.q,
            r=self.r,
            c=self.c,
            terminal=False,
        )

    def hessian_inverse(
        self,
        x: jax.Array | None = None,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate the inverse of the Hessian matrix using shortcuts.

        Parameters
        ----------
        x : jax.Array | None, optional
            State vector.
        u : jax.Array | None, optional
            Control vector.
        t : float | jax.Array, optional
            Timestamp.

        Returns
        -------
        jax.Array
            Inverted Hessian matrix.
        """
        del x, u, t
        if self.terminal:
            return jnp.linalg.inv(self.Q)
        if not self.has_cross_coupling:
            Q_inv = jnp.linalg.inv(self.Q)
            R_inv = jnp.linalg.inv(self.R)
            top = jnp.hstack([Q_inv, jnp.zeros((self.n, self.m), dtype=self.Q.dtype)])
            bot = jnp.hstack([jnp.zeros((self.m, self.n), dtype=self.Q.dtype), R_inv])
            return jnp.vstack([top, bot])
        G = self.hessian()
        return jnp.linalg.inv(G)

    def __add__(self, other: QuadraticCostFunction) -> "QuadraticCost":
        """Add two cost functions.

        Parameters
        ----------
        other : QuadraticCostFunction
            Other cost function to add.

        Returns
        -------
        QuadraticCost
            Sum of the two cost functions.
        """
        if isinstance(other, DiagonalCost):
            other_quad = other.to_quadratic()
            return self + other_quad
        if isinstance(other, QuadraticCost):
            if self.n != other.n or self.m != other.m:
                msg = f"Dimension mismatch in cost addition: ({self.n}, {self.m}) vs ({other.n}, {other.m})"
                raise ValueError(msg)
            is_term = self.terminal and other.terminal
            return QuadraticCost(
                Q=self.Q + other.Q,
                R=self.R + other.R if not is_term else None,
                H=self.H + other.H if not is_term else None,
                q=self.q + other.q,
                r=self.r + other.r if not is_term else None,
                c=self.c + other.c,
                terminal=is_term,
                m=self.m,
            )
        msg = f"Unsupported operand type for +: {type(other).__name__}"
        raise TypeError(msg)


def promote_weights(*weights: jax.Array) -> tuple[type[QuadraticCostFunction], tuple[jax.Array, ...]]:
    """Bring weights to a common representation and choose the cost class that holds them.

    Parameters
    ----------
    *weights : jax.Array
        Weight arrays, each a diagonal vector of shape (n,) or a dense matrix of shape (n, n).

    Returns
    -------
    tuple
        DiagonalCost and the vectors unchanged if every weight is diagonal, otherwise
        QuadraticCost and every weight embedded as a dense matrix.
    """
    arrs = tuple(jnp.asarray(W) for W in weights)
    if all(W.ndim == _EXPECTED_NDIM_1D for W in arrs):
        return DiagonalCost, arrs
    return QuadraticCost, tuple(diag_matrix(W) if W.ndim == _EXPECTED_NDIM_1D else W for W in arrs)


def LQRCost(  # noqa: N802
    Q: jax.Array,
    R: jax.Array,
    xf: jax.Array,
    uf: jax.Array | None = None,
    *,
    terminal: bool = False,
) -> QuadraticCostFunction:
    """Construct an LQR tracking cost function for a single knot point.

    Cost formula:
    0.5 * (x - xf)^T Q (x - xf) + 0.5 * (u - uf)^T R (u - uf)

    Returns DiagonalCost if Q and R are both diagonal vectors, QuadraticCost otherwise.

    Parameters
    ----------
    Q : jax.Array
        State weights of shape (n,) if diagonal or (n, n) if dense.
    R : jax.Array
        Control weights of shape (m,) if diagonal or (m, m) if dense.
    xf : jax.Array
        Reference/goal state of shape (n,).
    uf : jax.Array | None, optional
        Reference/goal control of shape (m,). Defaults to zeros.
    terminal : bool, optional
        Whether this is a terminal cost. Defaults to False.
    """
    xf_arr = jnp.asarray(xf)
    cost_cls, (Q_arr, R_arr) = promote_weights(Q, R)
    m = int(R_arr.shape[-1])
    uf_arr = jnp.zeros(m, dtype=xf_arr.dtype) if uf is None else jnp.asarray(uf, dtype=xf_arr.dtype)

    return cost_cls(
        Q=Q_arr,
        R=R_arr,
        q=-cost_cls.matvec(Q_arr, xf_arr),
        r=-cost_cls.matvec(R_arr, uf_arr),
        c=0.5 * cost_cls.quad_form(Q_arr, xf_arr) + 0.5 * cost_cls.quad_form(R_arr, uf_arr),
        terminal=terminal,
    )
