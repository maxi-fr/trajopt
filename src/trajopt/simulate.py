import dataclasses
import importlib
import time
from typing import Any, Self

import jax.numpy as jnp
import numpy as np

try:
    from simulate.controller import Controller
    from simulate.dynamics import Dynamics, StateLog
    from simulate.integrator import Integrator, rk4
except ImportError as err:
    msg = "The simulate package is required to use trajopt.simulate. Install with `pip install 'trajopt[simulate]'`."
    raise ImportError(msg) from err

from trajopt.dynamics.base import AbstractModel, ContinuousDynamics, DiscreteDynamics
from trajopt.problem import MPCState, Problem
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


def _build_problem(spec: dict[str, Any] | Problem) -> tuple[Problem, MPCState | None]:
    """Instantiate a Problem and optional MPCState from an instance or {class_path, ...} dict."""
    if isinstance(spec, Problem):
        return spec, None
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, func_name = class_path.rsplit(".", 1)
    target = getattr(importlib.import_module(module_name), func_name)
    res = target(**cfg) if cfg else target()
    if isinstance(res, tuple):
        state = res[1] if len(res) > 1 and isinstance(res[1], MPCState) else None
        return res[0], state
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
    initial_state : MPCState | None, optional
        Initial MPC state holding trajectories and multipliers. If None, initialized at the origin.
    solver : Solver | None, optional
        Solver backend object (e.g. ``Ipopt()``, ``OSQP(operating_point=...)``). Defaults to
        None, meaning ``Ipopt()``.
    """

    def __init__(
        self,
        dt: float,
        problem: Problem,
        initial_state: MPCState | None = None,
        solver: Solver | None = None,
    ) -> None:
        super().__init__(dt)
        self.problem = problem
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = MPCState.initial(problem, x0=jnp.zeros(problem.model.n), dt=dt)
        self.solver = solver

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate an MPC controller component from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing ``dt``, ``problem`` (Problem instance or {class_path, ...} dict),
            and optional ``solver``, ``initial_state``.

        Returns
        -------
        Self
            Instantiated MPC controller.
        """
        problem, initial_state = _build_problem(config["problem"])
        return cls(
            dt=float(config["dt"]),
            problem=problem,
            initial_state=initial_state or config.get("initial_state"),
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

        state = self.state.with_measurement(jnp.asarray(x_hat), t=t)

        if state.xf is not None:
            state = state.with_goal(jnp.asarray(ref))

        fallback_used = False
        solve_success = True
        try:
            solved_state = self.problem.solve(state, solver=self.solver)
            optimal_cost = float(self.problem.cost(solved_state))
        except Exception:  # noqa: BLE001 -- fallback on any solver failure
            solve_success = False
            fallback_used = True
            optimal_cost = float("nan")
            solved_state = state

        u_cmd = np.asarray(solved_state.controls[0], dtype=np.float64)

        self.state = solved_state.shift(self.dt)
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
        Integrator callable for continuous models. If None and the model is continuous,
        defaults to simulate.integrator.rk4.
    """

    def __init__(
        self,
        dt: float,
        model: AbstractModel,
        x0: np.ndarray | None = None,
        integrator: Integrator | None = None,
    ) -> None:
        if isinstance(model, ContinuousDynamics) and integrator is None:
            integrator = rk4

        super().__init__(dt, integrator=integrator)
        self.model = model
        self.n_inputs = int(model.m)
        self.x = np.asarray(x0 if x0 is not None else np.zeros(model.n), dtype=np.float64)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the model transition or continuous derivative.

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
            State derivative (continuous) or next state (discrete), shape (n,).
        """
        x_jax = jnp.asarray(x)
        u_jax = jnp.asarray(u)

        if isinstance(self.model, DiscreteDynamics):
            x_next = self.model.evaluate(x_jax, u_jax, t, self.dt)
            return np.asarray(x_next, dtype=np.float64)

        x_dot = self.model.evaluate(x_jax, u_jax, t)
        return np.asarray(x_dot, dtype=np.float64)

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
            {class_path, ...} dictionary). Optional keys: ``x0``, ``integrator``.

        Returns
        -------
        Self
            Instantiated TrajOptDynamics plant component.
        """
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            module_name, func_name = integrator.rsplit(".", 1)
            module = importlib.import_module(module_name)
            integrator = getattr(module, func_name)

        return cls(
            dt=float(config["dt"]),
            model=_build_model(config["model"]),
            x0=config.get("x0"),
            integrator=integrator,
        )
