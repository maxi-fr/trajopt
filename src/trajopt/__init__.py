"""TrajectoryOptimization (trajopt) in Python.

A high-performance optimal control and trajectory optimization library in Python
using JAX for fast automatic differentiation, vectorized Taylor expansions, and JIT compilation.
"""

from trajopt import _env as _env
from trajopt.cones import (
    AbstractCone,
    NegativeOrthant,
    PositiveOrthant,
    SecondOrderCone,
    ZeroCone,
)
from trajopt.costs import (
    CostFunction,
    DiagonalCost,
    GenericCost,
    LQRCost,
    LQRObjective,
    Objective,
    QuadraticCost,
    QuadraticCostFunction,
    TrackingObjective,
    cost,
    update_reference,
)
from trajopt.dynamics import (
    RK4,
    ContinuousDynamics,
    DiscreteDynamics,
    DiscretizedDynamics,
    EuclideanModel,
    Euler,
    ImplicitMidpoint,
    Integrator,
    euler_step,
    implicit_midpoint_step,
    rk4_step,
    rollout,
    rollout_states,
)
from trajopt.models import (
    Cartpole,
    DubinsCar,
    Pendulum,
)
from trajopt.trajectory import (
    KnotPoint,
    Trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "RK4",
    "AbstractCone",
    "Cartpole",
    "ContinuousDynamics",
    "CostFunction",
    "DiagonalCost",
    "DiscreteDynamics",
    "DiscretizedDynamics",
    "DubinsCar",
    "EuclideanModel",
    "Euler",
    "GenericCost",
    "ImplicitMidpoint",
    "Integrator",
    "KnotPoint",
    "LQRCost",
    "LQRObjective",
    "NegativeOrthant",
    "Objective",
    "Pendulum",
    "PositiveOrthant",
    "QuadraticCost",
    "QuadraticCostFunction",
    "SecondOrderCone",
    "TrackingObjective",
    "Trajectory",
    "ZeroCone",
    "cost",
    "euler_step",
    "implicit_midpoint_step",
    "rk4_step",
    "rollout",
    "rollout_states",
    "update_reference",
]
