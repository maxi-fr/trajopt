import dataclasses
import importlib
import time
from typing import Any, Self

import jax.numpy as jnp
import numpy as np

try:
    from simulate.controller import Controller
    from simulate.dynamics import Dynamics, StateLog
except ImportError as err:
    msg = "The simulate package is required to use trajopt.simulate. Install with `pip install 'trajopt[simulate]'`."
    raise ImportError(msg) from err

from trajopt.dynamics.base import AbstractModel
from trajopt.dynamics.integrators import Integrator
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.transcription.result import Solver


def _build_model(spec: dict[str, Any] | AbstractModel) -> AbstractModel:
    """Instantiate an AbstractModel from an instance or {class_path, ...} dict."""
    if isinstance(spec, AbstractModel):
        return spec
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**cfg)


def _build_problem(spec: dict[str, Any] | Problem) -> tuple[Problem, BoundaryConditions | None]:
    """Instantiate a Problem and its optional BoundaryConditions from an instance or {class_path, ...} dict."""
    if isinstance(spec, Problem):
        return spec, None
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, func_name = class_path.rsplit(".", 1)
    target = getattr(importlib.import_module(module_name), func_name)
    res = target(**cfg) if cfg else target()
    if isinstance(res, tuple):
        bc = next((item for item in res[1:] if isinstance(item, BoundaryConditions)), None)
        return res[0], bc
    return res, None


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Telemetry log emitted at each receding-horizon MPC step.

    Parameters
    ----------
    u : np.ndarray
        Commanded control action applied at this step, shape (m,).
    cost : float
        Optimal objective value evaluated at the solved trajectory, or NaN on failure.
    solve_time_s : float
        Elapsed wall-clock solver duration in seconds.
    solve_success : bool
        Whether the solver backend converged successfully.
    fallback_used : bool
        Whether a fallback was applied due to solve failure.
    """

    u: np.ndarray
    cost: float
    solve_time_s: float
    solve_success: bool
    fallback_used: bool


class TrajOptMPC(Controller[TrajOptMPCLog]):
    """General-purpose model predictive controller (MPC) component for simulate.

    Parameters
    ----------
    dt : float
        Controller execution time step.
    problem : Problem
        Optimal control problem definition.
    boundary : BoundaryConditions | None, optional
        Boundary conditions the driver starts from, supplying x0, t0 and the reference window.
        Defaults to None, meaning the origin with no run-time target.
    solver : Solver | None, optional
        Solver backend object (e.g. ``Ipopt()``, ``OSQP(operating_point=...)``). Defaults to
        None, meaning ``Ipopt()``.
    """

    def __init__(
        self,
        dt: float,
        problem: Problem,
        boundary: BoundaryConditions | None = None,
        solver: Solver | None = None,
    ) -> None:
        super().__init__(dt)
        x0 = jnp.zeros(problem.model.n) if boundary is None else boundary.x0
        self.mpc = MPC(problem, solver, x0=x0)
        if boundary is not None:
            self.mpc.bc = boundary

    @property
    def problem(self) -> Problem:
        """Optimal control problem the driver runs."""
        return self.mpc.problem

    @property
    def solver(self) -> Solver:
        """Solver backend the driver's Program is compiled for."""
        return self.mpc.solver

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate an MPC controller component from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing ``dt``, ``problem`` (Problem instance or {class_path, ...} dict),
            and optional ``solver``, ``boundary``.

        Returns
        -------
        Self
            Instantiated MPC controller.
        """
        problem, boundary = _build_problem(config["problem"])
        return cls(
            dt=float(config["dt"]),
            problem=problem,
            boundary=boundary if boundary is not None else config.get("boundary"),
            solver=config.get("solver"),
        )

    def update(
        self,
        t: float,
        ref: np.ndarray,
        x_hat: np.ndarray,
    ) -> tuple[np.ndarray, TrajOptMPCLog]:
        """Execute one receding-horizon MPC step.

        Parameters
        ----------
        t : float
            Current simulation time.
        ref : np.ndarray
            Reference signal or goal vector.
        x_hat : np.ndarray
            Current estimated state vector.

        Returns
        -------
        u : np.ndarray
            Commanded control input for this step, shape (m,).
        log : TrajOptMPCLog
            Telemetry log for this MPC step.
        """
        t_start = time.perf_counter()

        self.mpc.measure(jnp.asarray(x_hat), t=t)

        if self.mpc.xf is not None:
            self.mpc.set_goal(jnp.asarray(ref))

        fallback_used = False
        solve_success = True
        try:
            self.mpc.solve()
            optimal_cost = float(self.mpc.cost())
        except Exception:  # noqa: BLE001 -- fallback on any solver failure
            solve_success = False
            fallback_used = True
            optimal_cost = float("nan")

        u_cmd = np.asarray(self.mpc.controls[0], dtype=np.float64)

        self.mpc.shift(self.dt)
        solve_time = time.perf_counter() - t_start

        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=optimal_cost,
            solve_time_s=solve_time,
            solve_success=solve_success,
            fallback_used=fallback_used,
        )


class TrajOptDynamics(Dynamics[StateLog]):
    """Plant dynamics component wrapping a trajopt dynamical model for simulate.

    Parameters
    ----------
    dt : float
        Simulation time step.
    model : AbstractModel
        Continuous or discrete dynamical model from trajopt.
    x0 : np.ndarray | None, optional
        Initial state vector. Defaults to zeros.
    integrator : Integrator | None, optional
        trajopt integrator forwarded to ``model.discretize()``. Ignored if the model is
        already discrete. Defaults to RK4.
    """

    def __init__(
        self,
        dt: float,
        model: AbstractModel,
        x0: np.ndarray | None = None,
        integrator: Integrator | None = None,
    ) -> None:
        super().__init__(dt, integrator=None)
        self.model = model.discretize(integrator)
        self.n_inputs = int(model.m)
        self.x = np.asarray(x0 if x0 is not None else np.zeros(model.n), dtype=np.float64)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the discretized model's next state.

        Parameters
        ----------
        t : float
            Current simulation time.
        x : np.ndarray
            Current state vector, shape (n,).
        u : np.ndarray
            Commanded control input, shape (m,).

        Returns
        -------
        np.ndarray
            Next state, shape (n,).
        """
        x_jax = jnp.asarray(x)
        u_jax = jnp.asarray(u)
        x_next = self.model.evaluate(x_jax, u_jax, t, self.dt)
        return np.asarray(x_next, dtype=np.float64)

    def _make_log(self) -> StateLog:
        """Snapshot pre-step state for simulate telemetry."""
        return StateLog(x=self.x.copy())

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate dynamics component from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing ``dt`` and ``model`` (as an AbstractModel instance or
            {class_path, ...} dictionary). Optional keys: ``x0``, ``integrator`` (an Integrator
            instance or a class path string naming an Integrator subclass).

        Returns
        -------
        Self
            Instantiated TrajOptDynamics plant component.
        """
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            module_name, class_name = integrator.rsplit(".", 1)
            module = importlib.import_module(module_name)
            integrator_cls = getattr(module, class_name)
            integrator = integrator_cls()

        return cls(
            dt=float(config["dt"]),
            model=_build_model(config["model"]),
            x0=config.get("x0"),
            integrator=integrator,
        )
