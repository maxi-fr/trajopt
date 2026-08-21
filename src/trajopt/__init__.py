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

__version__ = "0.1.0"

__all__ = [
    "AbstractCone",
    "NegativeOrthant",
    "PositiveOrthant",
    "SecondOrderCone",
    "ZeroCone",
]
