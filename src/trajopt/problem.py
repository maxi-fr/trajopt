import equinox as eqx

from trajopt.constraints.constraint_list import BuiltConstraintList, ConstraintList
from trajopt.costs.objective import Objective
from trajopt.dynamics.base import AbstractModel, IntegratorCallable
from trajopt.dynamics.integrators import Integrator


class Problem(eqx.Module):
    """Problem structure holding model, objective, constraints, horizon, and integrator.

    Parameters
    ----------
    model : AbstractModel
        Continuous or discrete dynamical model.
    obj : Objective
        Cost objective with stacked parameters.
    constraints : BuiltConstraintList | ConstraintList | None, optional
        Registered or fused constraint list. Defaults to empty ConstraintList.
    N : int | None, optional
        Horizon length. Defaults to obj.N.
    integrator : Integrator | IntegratorCallable | None, optional
        Integrator instance for continuous models. Defaults to None.
    """

    model: AbstractModel
    obj: Objective
    constraints: BuiltConstraintList
    N: int = eqx.field(static=True)
    integrator: Integrator | IntegratorCallable | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        model: AbstractModel,
        obj: Objective,
        constraints: BuiltConstraintList | ConstraintList | None = None,
        N: int | None = None,
        integrator: Integrator | IntegratorCallable | None = None,
    ) -> None:
        n = int(model.n)
        m = int(model.m)
        N_val = int(N if N is not None else obj.N)

        if constraints is None:
            cl = ConstraintList(n=n, m=m, N=N_val)
            built_con = cl.build()
        elif isinstance(constraints, ConstraintList):
            built_con = constraints.build()
        elif isinstance(constraints, BuiltConstraintList):
            built_con = constraints
        else:
            msg = f"Unsupported constraints type: {type(constraints).__name__}"
            raise TypeError(msg)

        self.model = model
        self.obj = obj
        self.constraints = built_con
        self.N = N_val
        self.integrator = integrator
