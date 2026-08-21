"""Conic sets, projections, and derivatives for trajectory optimization constraints."""

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
        x : jax.Array
            Vector to project.

        Returns
        -------
        jax.Array
            Projected vector Pi(x).
        """

    @abstractmethod
    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection at x: nabla Pi(x).

        Parameters
        ----------
        x : jax.Array
            Vector at which to evaluate the Jacobian.

        Returns
        -------
        jax.Array
            Jacobian matrix of shape (n, n).
        """

    def hessian(self, x: jax.Array, b: jax.Array) -> jax.Array:
        """Evaluate the second-derivative contraction nabla^2 Pi(x)[b].

        Derived via automatic differentiation of the projection Jacobian.

        Parameters
        ----------
        x : jax.Array
            Vector at which to evaluate the Hessian contraction.
        b : jax.Array
            Contraction vector.

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
        """Project vector x onto the zero cone.

        Parameters
        ----------
        x : jax.Array
            Input vector.

        Returns
        -------
        jax.Array
            Zero vector of matching shape and dtype.
        """
        return jnp.zeros_like(x)

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the zero cone.

        Parameters
        ----------
        x : jax.Array
            Input vector.

        Returns
        -------
        jax.Array
            Zero matrix of shape (n, n).
        """
        x_arr = jnp.asarray(x)
        n = x_arr.shape[0]
        return jnp.zeros((n, n), dtype=x_arr.dtype)

    def hessian(self, x: jax.Array, b: jax.Array) -> jax.Array:
        """Evaluate the Hessian contraction nabla^2 Pi(x)[b].

        Parameters
        ----------
        x : jax.Array
            Input vector.
        b : jax.Array
            Contraction vector.

        Returns
        -------
        jax.Array
            Zero matrix of shape (n, n).
        """
        del b
        x_arr = jnp.asarray(x)
        n = x_arr.shape[0]
        return jnp.zeros((n, n), dtype=x_arr.dtype)


class NegativeOrthant(AbstractCone):
    """Negative orthant representing inequality constraints h(x) <= 0."""

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x onto the negative orthant.

        Parameters
        ----------
        x : jax.Array
            Input vector.

        Returns
        -------
        jax.Array
            min(0, x).
        """
        return jnp.minimum(0.0, jnp.asarray(x))

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the negative orthant.

        Parameters
        ----------
        x : jax.Array
            Input vector.

        Returns
        -------
        jax.Array
            Diagonal matrix of shape (n, n) with 1 where x <= 0 and 0 where x > 0.
        """
        x_arr = jnp.asarray(x)
        return jnp.diag((x_arr <= 0.0).astype(x_arr.dtype))

    def hessian(self, x: jax.Array, b: jax.Array) -> jax.Array:
        """Evaluate the Hessian contraction nabla^2 Pi(x)[b].

        Parameters
        ----------
        x : jax.Array
            Input vector.
        b : jax.Array
            Contraction vector.

        Returns
        -------
        jax.Array
            Zero matrix of shape (n, n).
        """
        del b
        x_arr = jnp.asarray(x)
        n = x_arr.shape[0]
        return jnp.zeros((n, n), dtype=x_arr.dtype)


class PositiveOrthant(AbstractCone):
    """Positive orthant representing inequality constraints h(x) >= 0."""

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x onto the positive orthant.

        Parameters
        ----------
        x : jax.Array
            Input vector.

        Returns
        -------
        jax.Array
            max(0, x).
        """
        return jnp.maximum(0.0, jnp.asarray(x))

    def jacobian(self, x: jax.Array) -> jax.Array:
        """Evaluate the Jacobian of the projection onto the positive orthant.

        Parameters
        ----------
        x : jax.Array
            Input vector.

        Returns
        -------
        jax.Array
            Diagonal matrix of shape (n, n) with 1 where x >= 0 and 0 where x < 0.
        """
        x_arr = jnp.asarray(x)
        return jnp.diag((x_arr >= 0.0).astype(x_arr.dtype))

    def hessian(self, x: jax.Array, b: jax.Array) -> jax.Array:
        """Evaluate the Hessian contraction nabla^2 Pi(x)[b].

        Parameters
        ----------
        x : jax.Array
            Input vector.
        b : jax.Array
            Contraction vector.

        Returns
        -------
        jax.Array
            Zero matrix of shape (n, n).
        """
        del b
        x_arr = jnp.asarray(x)
        n = x_arr.shape[0]
        return jnp.zeros((n, n), dtype=x_arr.dtype)


class SecondOrderCone(AbstractCone):
    """Second-order cone (Lorentz cone / ice cream cone): ||v||_2 <= s, where x = [v; s]."""

    eps: float = eqx.field(default=1e-10, static=True)

    def status(self, x: jax.Array) -> str:
        """Determine region status of concrete x relative to the cone.

        Parameters
        ----------
        x : jax.Array
            Input vector [v; s].

        Returns
        -------
        str
            "below", "inside", or "outside".
        """
        x_arr = jnp.asarray(x)
        v = x_arr[:-1]
        s = x_arr[-1]
        a = jnp.linalg.norm(v)
        if bool(a <= -s):
            return "below"
        if bool(a <= s):
            return "inside"
        return "outside"

    def project(self, x: jax.Array) -> jax.Array:
        """Project vector x onto the second-order cone.

        Parameters
        ----------
        x : jax.Array
            Input vector [v; s].

        Returns
        -------
        jax.Array
            Projected vector.
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
        x : jax.Array
            Input vector [v; s].

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

    def hessian(self, x: jax.Array, b: jax.Array) -> jax.Array:
        """Evaluate the second-derivative contraction nabla^2 Pi(x)[b].

        Derived via automatic differentiation of the projection Jacobian.

        Parameters
        ----------
        x : jax.Array
            Input vector [v; s].
        b : jax.Array
            Contraction vector.

        Returns
        -------
        jax.Array
            Hessian contraction matrix of shape (n, n).
        """
        x_arr = jnp.asarray(x)
        b_arr = jnp.asarray(b)
        return jax.jacobian(lambda x_: self.jacobian(x_).T @ b_arr)(x_arr)
