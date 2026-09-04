import jax
import jax.numpy as jnp

from trajopt.dynamics.base import DiscreteDynamics


class AffineModel(DiscreteDynamics):
    """Linear time-invariant discrete model x_{k+1} = A x_k + B u_k + d.

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
    """

    A: jax.Array
    B: jax.Array
    d: jax.Array

    def __init__(self, A: jax.Array, B: jax.Array, d: jax.Array | None = None) -> None:
        A_arr = jnp.asarray(A, dtype=jnp.float64)
        B_arr = jnp.asarray(B, dtype=jnp.float64)
        super().__init__(n=int(A_arr.shape[0]), m=int(B_arr.shape[1]), ne=int(A_arr.shape[0]))
        self.A = A_arr
        self.B = B_arr
        self.d = jnp.zeros(self.n, dtype=jnp.float64) if d is None else jnp.asarray(d, dtype=jnp.float64)

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
