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
from trajopt.dynamics import (
    ContinuousDynamics,
    DiscreteDynamics,
    DiscretizedDynamics,
    EuclideanModel,
)
from trajopt.models import (
    Cartpole,
)
from trajopt.trajectory import (
    KnotPoint,
    Trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "AbstractCone",
    "Cartpole",
    "ContinuousDynamics",
    "DiscreteDynamics",
    "DiscretizedDynamics",
    "EuclideanModel",
    "KnotPoint",
    "NegativeOrthant",
    "PositiveOrthant",
    "SecondOrderCone",
    "Trajectory",
    "ZeroCone",
]
