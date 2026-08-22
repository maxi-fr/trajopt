from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractCone(eqx.Module):
    """Abstract base class for convex cones."""

    @abstractmethod
    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x onto the cone.

        Parameters
        ----------
        x
            Vector of shape (n,).

        Returns
        -------
        jax.Array
            Projection Pi(x) of shape (n,).
        """

    @abstractmethod
    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection at x: nabla Pi(x).

        Parameters
        ----------
        x
            Vector of shape (n,).

        Returns
        -------
        jax.Array
            Jacobian matrix of shape (n, n).
        """

    def hessian(self, x: jax.Array, b: jax.Array) -> jax.Array:
        """Evaluate the second-derivative contraction nabla^2 Pi(x)[b] by autodiff of the Jacobian.

        Parameters
        ----------
        x
            Vector of shape (n,).
        b
            Contraction vector of shape (n,).

        Returns
        -------
        jax.Array
            Hessian contraction matrix of shape (n, n).
        """
        x_arr = jnp.asarray(x)
        b_arr = jnp.asarray(b)
        return jax.jacobian(lambda x_: self.jacobian(x_).T @ b_arr)(x_arr)

class ZeroCone(AbstractCone):
    """Zero cone representing equality constraints g(x) = 0."""

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x of shape (n,) onto the zero cone, giving zeros of shape (n,)."""
        return jnp.zeros_like(x)

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the zero cone.

        Parameters
        ----------
        x
            Vector of shape (n,).

        Returns
        -------
        jax.Array
            Zero matrix of shape (n, n).
        """
        x_arr = jnp.asarray(x)
        n = x_arr.shape[0]
        return jnp.zeros((n, n), dtype=x_arr.dtype)

class NegativeOrthant(AbstractCone):
    """Negative orthant representing inequality constraints h(x) <= 0."""

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x of shape (n,) onto the negative orthant, giving min(0, x) of shape (n,)."""
        return jnp.minimum(0.0, jnp.asarray(x))

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the negative orthant.

        Parameters
        ----------
        x
            Vector of shape (n,).

        Returns
        -------
        jax.Array
            Diagonal matrix of shape (n, n) with 1 where x <= 0 and 0 where x > 0.
        """
        x_arr = jnp.asarray(x)
        return jnp.diag((x_arr <= 0.0).astype(x_arr.dtype))

class PositiveOrthant(AbstractCone):
    """Positive orthant representing inequality constraints h(x) >= 0."""

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x of shape (n,) onto the positive orthant, giving max(0, x) of shape (n,)."""
        return jnp.maximum(0.0, jnp.asarray(x))

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the positive orthant.

        Parameters
        ----------
        x
            Vector of shape (n,).

        Returns
        -------
        jax.Array
            Diagonal matrix of shape (n, n) with 1 where x >= 0 and 0 where x < 0.
        """
        x_arr = jnp.asarray(x)
        return jnp.diag((x_arr >= 0.0).astype(x_arr.dtype))

class SecondOrderCone(AbstractCone):
    """Second-order cone (Lorentz cone / ice cream cone): ||v||_2 <= s, where x = [v; s]."""

    eps: float = eqx.field(default=1e-10, static=True)

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x onto the second-order cone.

        Parameters
        ----------
        x
            Vector [v; s] of shape (n,).

        Returns
        -------
        jax.Array
            Projection of shape (n,).
        """
        x_arr = jnp.asarray(x)
        v = x_arr[:-1]
        s = x_arr[-1]
        a = jnp.linalg.norm(v)

        r2 = jnp.sum(v**2)
        safe_a = jnp.sqrt(jnp.maximum(r2, self.eps**2))
        scale = 0.5 * (1.0 + s / safe_a)
        proj_outside = jnp.concatenate([scale * v, jnp.expand_dims(scale * safe_a, 0)])

        return jnp.where(
            a <= -s,
            jnp.zeros_like(x_arr),
            jnp.where(
                a <= s,
                x_arr,
                proj_outside,
            ),
        )

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the second-order cone.

        Parameters
        ----------
        x
            Vector [v; s] of shape (n,).

        Returns
        -------
        jax.Array
            Jacobian matrix of shape (n, n).
        """
        x_arr = jnp.asarray(x)
        n = x_arr.shape[0]
        v = x_arr[:-1]
        s = x_arr[-1]
        a = jnp.linalg.norm(v)

        r2 = jnp.sum(v**2)
        safe_a = jnp.sqrt(jnp.maximum(r2, self.eps**2))

        I_v = jnp.eye(n - 1, dtype=x_arr.dtype)
        vvT = jnp.outer(v, v)

        # Top-left block: 0.5 * ((1 + s/a)*I - (s/a^3)*v*v^T)
        J_vv = 0.5 * ((1.0 + s / safe_a) * I_v - (s / (safe_a**3)) * vvT)
        # Top-right block: 0.5 * (v / a)
        J_vs = 0.5 * (v / safe_a)[:, None]
        # Bottom-left block: 0.5 * (v^T / a)
        J_sv = 0.5 * (v / safe_a)[None, :]
        # Bottom-right scalar: 0.5
        J_ss = jnp.array([[0.5]], dtype=x_arr.dtype)

        J_top = jnp.hstack([J_vv, J_vs])
        J_bot = jnp.hstack([J_sv, J_ss])
        J_outside = jnp.vstack([J_top, J_bot])

        return jnp.where(
            a <= -s,
            jnp.zeros((n, n), dtype=x_arr.dtype),
            jnp.where(
                a <= s,
                jnp.eye(n, dtype=x_arr.dtype),
                J_outside,
            ),
        )
