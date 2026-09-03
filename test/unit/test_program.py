import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.program import Program
from trajopt.solvers.ilqr import ILQR, ilqr_solve

N = 8
DT = 0.05
N_STEPS = 20
GOAL = jnp.array([np.pi, 0.0], dtype=jnp.float64)


def _problem() -> Problem:
    """Unconstrained pendulum swing-up problem whose run-time goal the boundary conditions retarget."""
    obj = LQRObjective(Q=jnp.eye(2) * DT, R=jnp.eye(1) * DT, Qf=jnp.eye(2) * 10.0, N=N)
    return Problem(model=Pendulum(), obj=obj, constraints=None, N=N, dt=DT, integrator=RK4())


def test_core_is_built_once_per_key() -> None:
    """A program compiles one core per `(fn, key)` and hands the same object back on a repeat ask."""
    program = Program(_problem(), ILQR())
    first = program.core(ilqr_solve, key="a", options=ILQR().options, solve_kd_builder=None)
    again = program.core(ilqr_solve, key="a", options=ILQR().options, solve_kd_builder=None)
    other = program.core(ilqr_solve, key="b", options=ILQR().options, solve_kd_builder=None)

    assert again is first
    assert other is not first


def test_driver_holds_one_program_for_its_life() -> None:
    """An MPC builds its solver's program once and keeps it; a second driver gets its own."""
    problem = _problem()
    solver = ILQR()

    mpc = MPC(problem, solver, x0=jnp.zeros(2), xf=GOAL)
    program = mpc.program
    assert mpc.program is program
    assert program.problem is problem
    assert program.solver is solver
    assert MPC(problem, solver, x0=jnp.zeros(2), xf=GOAL).program is not program


def test_mpc_loop_builds_one_program_and_one_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """A receding-horizon loop builds exactly one Program and one compiled core, whatever the boundary conditions do.

    The structural invariant the MPC driver rests on: the Program is the solver's compiled form of
    the Problem, so moving `x0`, `t0` and the reference window between steps is traced data and
    rebuilds nothing. `jax.jit`'s own `_cache_size()` is the independent check that XLA compiled the
    core once too, rather than the program handing back one closure that retraced under it.
    """
    constructions = 0
    original_init = Program.__init__

    def counting_init(self, problem, solver):
        """Delegate to `Program.__init__`, counting every program the loop constructs."""
        nonlocal constructions
        constructions += 1
        original_init(self, problem, solver)

    monkeypatch.setattr(Program, "__init__", counting_init)

    problem = _problem()
    solver = ILQR()
    x0 = jnp.array([0.1, 0.0], dtype=jnp.float64)
    mpc = MPC(problem, solver, x0=x0, xf=GOAL)
    t = 0.0
    for step in range(N_STEPS):
        mpc.set_goal(jnp.array([np.pi + 0.01 * (step + 1), 0.0], dtype=jnp.float64))
        mpc.measure(x0 + 0.01 * step, t)
        mpc.solve()
        mpc.shift(DT)
        t += DT

    assert constructions == 1
    program = mpc.program
    cores = list(program._cores.values())  # noqa: SLF001 -- the compiled cores are what the test pins
    assert len(cores) == 1
    core = cores[0]
    assert isinstance(core, jax.stages.Wrapped)
    assert core._cache_size() == 1  # noqa: SLF001 -- jax's compiled-executable count  # ty: ignore[unresolved-attribute]
