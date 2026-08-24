from abc import abstractmethod

import jax

from trajopt.cones import ZeroCone
from trajopt.constraints.base import ConstraintShape
from trajopt.dynamics.base import ContinuousDynamics, DiscreteDynamics, IntegratorCallable


class AbstractDynamicsConstraint(ConstraintShape):
    """Abstract base class for two-point / inter-stage dynamics constraints: c(x_k, u_k, x_{k+1}) = 0."""

    @abstractmethod
    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> jax.Array:
        """Evaluate dynamics defect vector of shape (p,)."""

    def jacobian_x(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> jax.Array:
        """Evaluate Jacobian dc/dx_k of shape (p, n)."""
        return jax.jacobian(lambda x_: self.evaluate(x_, u, x_next, t, dt))(x)

    def jacobian_u(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> jax.Array:
        """Evaluate Jacobian dc/du_k of shape (p, m)."""
        return jax.jacobian(lambda u_: self.evaluate(x, u_, x_next, t, dt))(u)

    def jacobian_x_next(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> jax.Array:
        """Evaluate Jacobian dc/dx_{k+1} of shape (p, n)."""
        return jax.jacobian(lambda xn_: self.evaluate(x, u, xn_, t, dt))(x_next)

    def jacobian(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate Jacobians with respect to (x_k, u_k, x_{k+1}) via AD.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            (dc/dx_k, dc/du_k, dc/dx_{k+1}) of shapes (p, n), (p, m), (p, n).
        """
        jx = self.jacobian_x(x, u, x_next, t, dt)
        ju = self.jacobian_u(x, u, x_next, t, dt)
        jx_next = self.jacobian_x_next(x, u, x_next, t, dt)
        return jx, ju, jx_next


class DynamicsConstraint(AbstractDynamicsConstraint):
    """Explicit discrete dynamics constraint: x_{k+1} - f_d(x_k, u_k, t_k, dt_k) = 0.

    Parameters
    ----------
    model : DiscreteDynamics | ContinuousDynamics
        Dynamics model. If ContinuousDynamics is provided, it is discretized using integrator.
    integrator : IntegratorCallable | None, optional
        Integrator to use if model is continuous (default RK4).
    """

    model: DiscreteDynamics

    def __init__(
        self,
        model: DiscreteDynamics | ContinuousDynamics,
        integrator: IntegratorCallable | None = None,
    ) -> None:
        disc_model = model.discretize(integrator)

        super().__init__(n=disc_model.n, m=disc_model.m, p=disc_model.n, cone=ZeroCone())
        self.model = disc_model

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> jax.Array:
        """Evaluate discrete dynamics defect x_{k+1} - f_d(x_k, u_k, t_k, dt_k)."""
        x_pred = self.model.discrete_dynamics(x, u, t, dt)
        return x_next - x_pred


class ImplicitDynamicsConstraint(AbstractDynamicsConstraint):
    """Implicit collocation dynamics constraint (e.g. implicit midpoint).

    Defect formula: x_k - x_{k+1} + dt * f((x_k + x_{k+1}) / 2, u_k, t_k + dt / 2) = 0.

    Parameters
    ----------
    model : ContinuousDynamics
        Continuous-time dynamics model.
    """

    model: ContinuousDynamics

    def __init__(self, model: ContinuousDynamics) -> None:
        if not isinstance(model, ContinuousDynamics):
            msg = f"ImplicitDynamicsConstraint requires ContinuousDynamics, got {type(model).__name__}."
            raise TypeError(msg)

        super().__init__(n=model.n, m=model.m, p=model.n, cone=ZeroCone())
        self.model = model

    def evaluate(
        self,
        x: jax.Array,
        u: jax.Array,
        x_next: jax.Array,
        t: float | jax.Array = 0.0,
        dt: float | jax.Array = 0.1,
    ) -> jax.Array:
        """Evaluate implicit midpoint collocation defect."""
        x_mid = 0.5 * (x + x_next)
        t_mid = t + 0.5 * dt
        f_mid = self.model.dynamics(x_mid, u, t_mid)
        return x - x_next + dt * f_mid
