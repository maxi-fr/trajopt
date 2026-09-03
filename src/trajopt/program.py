import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax

if TYPE_CHECKING:
    from trajopt.problem import MPCState, Problem
    from trajopt.transcription.result import Solver, SolverResult


class Program:
    """One solver's compiled, allocated form of a `Problem`: its jitted cores and its live handles.

    A Problem says what the problem *is* -- model, objective shape, constraints, horizon -- and is
    immutable and structural. A Program is what a particular solver had to build in order to run
    that Problem: the `jax.jit` closures specialized to it for the native stagewise backends, and
    (once the QP slice lands) the live C handles an eager backend keeps so it can update a
    factorization instead of setting one up again. It is mutable, eager-side, per-solver, and
    deliberately not a pytree: `vmap` over a Program is given up on purpose.

    A Program is built once and reused across MPC steps. `BoundaryConditions` is a traced argument
    to its cores, never baked into it, so moving `x0`, `t0` or the reference window changes values
    and recompiles nothing. Everything that must instead be a compile-time constant -- the Problem
    and the solver's static configuration -- is the Program's identity, so a different static
    configuration means a different Program rather than a silent retrace of this one.

    Parameters
    ----------
    problem : Problem
        The structural problem this program is specialized to. Held by reference; the cores close
        over it, so it must not be swapped.
    solver : Solver
        The backend this program belongs to, and the source of its static configuration.
    """

    __slots__ = ("_cores", "handles", "problem", "solver")

    def __init__(self, problem: "Problem", solver: "Solver") -> None:
        self.problem = problem
        self.solver = solver
        self._cores: dict[tuple[Any, Any], Callable[..., Any]] = {}
        self.handles: dict[str, Any] = {}

    def core(self, fn: Callable[..., Any], key: object = None, **static_kwargs: object) -> Callable[..., Any]:
        """Return `fn` jitted and specialized to this program's problem, built once per `(fn, key)`.

        `problem` and `static_kwargs` are closed over via `functools.partial` rather than passed as
        jit arguments: `problem` carries structural constraint data (`ALConstraints.build` /
        `PNLayout.build` convert its bounds with eager `np.asarray`, which needs concrete values and
        breaks under trace), and a Python callable in `static_kwargs` cannot be a traced pytree leaf
        at all. `key` distinguishes cores whose `static_kwargs` differ, and must hash and compare
        equal exactly when those values do -- a hashable `SolverOptions`, or a memoized
        `solve_kd_builder` compared by identity.

        Parameters
        ----------
        fn : Callable
            The traced core to specialize (e.g. `al_solve`).
        key : object
            Everything `static_kwargs` binds, compared for equality to decide reuse.
        **static_kwargs : object
            Bound into `fn` alongside `problem` before jitting.

        Returns
        -------
        Callable
            The program's jitted core. Call it with the genuinely dynamic arguments as keywords --
            `problem` occupies `fn`'s first positional slot, so a positional call would collide
            with its keyword binding.
        """
        cache_key = (fn, key)
        core = self._cores.get(cache_key)
        if core is None:
            core = self._build_core(fn, **static_kwargs)
            self._cores[cache_key] = core
        return core

    def _build_core(self, fn: Callable[..., Any], **static_kwargs: object) -> Callable[..., Any]:
        """Specialize `fn` to this program's problem -- the one `jax.jit` call site, so the one place a compile can start."""
        return jax.jit(functools.partial(fn, problem=self.problem, **static_kwargs))

    def solve(self, state: "MPCState") -> "SolverResult":
        """Solve from `state` with this program's solver, returning the backend's raw result."""
        return self.solver.solve(self.problem, state)


def program_for(solver: "Solver", problem: "Problem") -> Program:
    """Return `solver`'s Program for `problem`, building one on first use and reusing it thereafter.

    The Program lives on the solver instance because a Program is per-solver: one solver object
    driven over a receding horizon builds its program once and keeps it, which is the reuse the
    whole refactor exists for. `problem` is compared by identity and held by the Program, so a
    solver pointed at a different Problem object gets a different Program.
    """
    program = getattr(solver, "_program", None)
    if isinstance(program, Program) and program.problem is problem:
        return program
    program = Program(problem, solver)
    object.__setattr__(solver, "_program", program)
    return program
