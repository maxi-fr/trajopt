import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.program import Program
from trajopt.solvers.ilqr import ILQR

N = 8
DT = 0.05
N_STEPS = 20
GOAL = jnp.array([np.pi, 0.0], dtype=jnp.float64)


def _build() -> Problem:
    """Unconstrained pendulum swing-up problem with a quadratic objective the run-time goal retargets."""
    obj = LQRObjective(Q=jnp.eye(2) * DT, R=jnp.eye(1) * DT, Qf=jnp.eye(2) * 10.0, N=N)
    return Problem(model=Pendulum(), obj=obj, constraints=None, N=N, integrator=RK4())


def _count_core_compiles(monkeypatch: pytest.MonkeyPatch, *, moving_goal: bool) -> int:
    """Run an `N_STEPS` receding-horizon ILQR loop and return how many jitted cores got built.

    `Program._build_core` is the only `jax.jit` call site in the solver stack, so counting its
    invocations counts the fresh jitted closures -- and therefore the XLA compilations -- the loop
    forces. With `moving_goal`, `xf` changes every step; otherwise it is held at `GOAL`.
    """
    builds = 0
    original = Program._build_core  # noqa: SLF001 -- counting the one jit call site is the point of the test

    def counting_build_core(self, fn, **static_kwargs):
        """Delegate to `Program._build_core`, counting every core it compiles."""
        nonlocal builds
        builds += 1
        return original(self, fn, **static_kwargs)

    monkeypatch.setattr(Program, "_build_core", counting_build_core)

    problem = _build()
    solver = ILQR()
    x0 = jnp.array([0.1, 0.0], dtype=jnp.float64)
    state = MPCState.initial(problem, x0=x0, dt=DT, xf=GOAL)
    t = 0.0
    for step in range(N_STEPS):
        if moving_goal:
            state = state.with_goal(jnp.array([np.pi + 0.01 * (step + 1), 0.0], dtype=jnp.float64))
        state = state.with_measurement(x0, t)
        state = problem.solve(state, solver=solver)
        state = state.shift(DT)
        t += DT
    return builds


def test_fixed_goal_compiles_core_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control case: a receding-horizon loop with a fixed goal compiles the ILQR core exactly once."""
    assert _count_core_compiles(monkeypatch, moving_goal=False) == 1


def test_moving_goal_compiles_core_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A receding-horizon loop whose goal moves every step must still compile the ILQR core exactly once."""
    assert _count_core_compiles(monkeypatch, moving_goal=True) == 1
