import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.solvers import _jit_cache
from trajopt.solvers.ilqr import ILQR

N = 8
DT = 0.05
N_STEPS = 20
GOAL = jnp.array([np.pi, 0.0], dtype=jnp.float64)


def _build() -> Problem:
    """Unconstrained pendulum swing-up problem with an LQR objective, so `regulates_to_goal` holds."""
    obj = LQRObjective(Q=jnp.eye(2) * DT, R=jnp.eye(1) * DT, Qf=jnp.eye(2) * 10.0, xf=GOAL, N=N)
    return Problem(model=Pendulum(), obj=obj, constraints=None, N=N, integrator=RK4())


def _count_core_compiles(monkeypatch: pytest.MonkeyPatch, *, moving_goal: bool) -> int:
    """Run an `N_STEPS` receding-horizon ILQR loop and return how many jitted cores got built.

    A compile happens exactly when `JitCacheSlot.get_or_build` misses its single-entry cache and
    calls `jax.jit`; the wrapper below replays that hit test before delegating, so the count is
    the number of fresh `jax.jit` closures the loop forces. With `moving_goal`, `xf` changes every
    step; otherwise it is held at `GOAL`.
    """
    misses = 0
    original = _jit_cache.JitCacheSlot.get_or_build

    def counting_get_or_build(self, fn, problem, key, **static_kwargs):
        """Delegate to `JitCacheSlot.get_or_build`, counting the calls that miss the cache."""
        nonlocal misses
        if not (self._problem_ref is problem and self._key == key and self._jitted is not None):
            misses += 1
        return original(self, fn, problem, key, **static_kwargs)

    monkeypatch.setattr(_jit_cache.JitCacheSlot, "get_or_build", counting_get_or_build)

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
    return misses


def test_fixed_goal_compiles_core_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control case: a receding-horizon loop with a fixed goal compiles the ILQR core exactly once."""
    assert _count_core_compiles(monkeypatch, moving_goal=False) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The goal is baked into the jitted core as a compile-time constant: build_warm_start's "
        "_cached_goal_override allocates a new Problem per distinct xf, and JitCacheSlot keys reuse "
        "on problem identity, so a moving goal recompiles every step. The refactor making "
        "BoundaryConditions a traced argument fixes this. strict=True is deliberate: once that "
        "lands this test XPASSes and fails the suite, forcing removal of this marker."
    ),
)
def test_moving_goal_compiles_core_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A receding-horizon loop whose goal moves every step must still compile the ILQR core exactly once."""
    assert _count_core_compiles(monkeypatch, moving_goal=True) == 1
