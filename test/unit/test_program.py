import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.pendulum import Pendulum
from trajopt.problem import MPCState, Problem
from trajopt.program import Program, program_for
from trajopt.solvers.ilqr import ILQR, ilqr_solve

N = 8
DT = 0.05
N_STEPS = 20
GOAL = jnp.array([np.pi, 0.0], dtype=jnp.float64)


def _problem() -> Problem:
    """Unconstrained pendulum swing-up problem whose run-time goal the boundary conditions retarget."""
    obj = LQRObjective(Q=jnp.eye(2) * DT, R=jnp.eye(1) * DT, Qf=jnp.eye(2) * 10.0, N=N)
    return Problem(model=Pendulum(), obj=obj, constraints=None, N=N, integrator=RK4())


def test_core_is_built_once_per_key() -> None:
    """A program compiles one core per `(fn, key)` and hands the same object back on a repeat ask."""
    program = Program(_problem(), ILQR())
    first = program.core(ilqr_solve, key="a", options=ILQR().options, solve_kd_builder=None)
    again = program.core(ilqr_solve, key="a", options=ILQR().options, solve_kd_builder=None)
    other = program.core(ilqr_solve, key="b", options=ILQR().options, solve_kd_builder=None)

    assert again is first
    assert other is not first


def test_program_for_reuses_one_program_per_solver_and_problem() -> None:
    """`program_for` builds a solver's program once for a problem and rebuilds it for a different one."""
    problem = _problem()
    solver = ILQR()

    program = program_for(solver, problem)
    assert program_for(solver, problem) is program
    assert program.problem is problem
    assert program.solver is solver
    assert program_for(solver, _problem()) is not program


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
    state = MPCState.initial(problem, x0=x0, dt=DT, xf=GOAL)
    t = 0.0
    for step in range(N_STEPS):
        state = state.with_goal(jnp.array([np.pi + 0.01 * (step + 1), 0.0], dtype=jnp.float64))
        state = state.with_measurement(x0 + 0.01 * step, t)
        state = problem.solve(state, solver=solver)
        state = state.shift(DT)
        t += DT

    assert constructions == 1
    program = program_for(solver, problem)
    assert constructions == 1
    cores = list(program._cores.values())  # noqa: SLF001 -- the compiled cores are what the test pins
    assert len(cores) == 1
    core = cores[0]
    assert isinstance(core, jax.stages.Wrapped)
    assert core._cache_size() == 1  # noqa: SLF001 -- jax's compiled-executable count  # ty: ignore[unresolved-attribute]
