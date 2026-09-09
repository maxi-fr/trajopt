import jax
import jax.numpy as jnp

from trajopt.dynamics.base import DiscreteDynamics


class AffineModel(DiscreteDynamics):
    """Linear time-invariant discrete model x_{k+1} = A x_k + B u_k + d with optional output y = C x + D u + dy.

    The step map is the one written down, not an integration of it, so the model is its own
    linearization: `linearize` reproduces A and B exactly at every Operating Point and the
    Quadratic Subproblem of an LQR problem on this model is the problem itself.

    Parameters
    ----------
    A : jax.Array
        State transition matrix of shape (n, n).
    B : jax.Array
        Control matrix of shape (n, m).
    d : jax.Array | None, optional
        Affine offset of shape (n,). Defaults to zero.
    C : jax.Array | None, optional
        Output state matrix of shape (p, n). Defaults to None.
    D : jax.Array | None, optional
        Output control matrix of shape (p, m). Defaults to zeros.
    dy : jax.Array | None, optional
        Output affine offset of shape (p,). Defaults to zero.
    """

    A: jax.Array
    B: jax.Array
    d: jax.Array
    C: jax.Array | None = None
    D: jax.Array | None = None
    dy: jax.Array | None = None

    def __init__(  # noqa: PLR0913 -- AffineModel parameterizes linear state and output equations
        self,
        A: jax.Array,
        B: jax.Array,
        d: jax.Array | None = None,
        *,
        C: jax.Array | None = None,
        D: jax.Array | None = None,
        dy: jax.Array | None = None,
    ) -> None:
        A_arr = jnp.asarray(A, dtype=jnp.float64)
        B_arr = jnp.asarray(B, dtype=jnp.float64)
        n = int(A_arr.shape[0])
        m = int(B_arr.shape[1])
        p = int(C.shape[0]) if C is not None else None
        super().__init__(n=n, m=m, ne=n, p=p)
        self.A = A_arr
        self.B = B_arr
        self.d = jnp.zeros(self.n, dtype=jnp.float64) if d is None else jnp.asarray(d, dtype=jnp.float64)
        if C is not None:
            self.C = jnp.asarray(C, dtype=jnp.float64)
            self.D = jnp.zeros((self.p, self.m), dtype=jnp.float64) if D is None else jnp.asarray(D, dtype=jnp.float64)
            self.dy = jnp.zeros(self.p, dtype=jnp.float64) if dy is None else jnp.asarray(dy, dtype=jnp.float64)
        else:
            self.C = None
            self.D = None
            self.dy = None

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Evaluate the next state A x + B u + d of shape (n,); the step map does not depend on t or dt."""
        del t, dt
        return self.A @ x + self.B @ u + self.d

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate output y = C x + D u + dy of shape (p,)."""
        if self.C is None or self.dy is None:
            return super().output(x, u, t)
        del t
        y = self.C @ x + self.dy
        if u is not None and self.D is not None:
            y = y + self.D @ u
        return y

    def output_state_jacobian(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate output state Jacobian C of shape (p, n)."""
        if self.C is None:
            return super().output_state_jacobian(x, u, t)
        del x, u, t
        return self.C

    def output_control_jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate output control Jacobian D of shape (p, m)."""
        if self.D is None:
            return super().output_control_jacobian(x, u, t)
        del x, u, t
        return self.D

    def has_control_feedthrough(self) -> bool:
        """Whether output control matrix D is non-zero."""
        return self.D is not None and not bool(jnp.all(self.D == 0))
