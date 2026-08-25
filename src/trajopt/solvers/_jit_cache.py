import functools
from collections.abc import Callable
from typing import Any

import jax


class JitCacheSlot:
    """Single-entry cache for one `problem`-specialized `jax.jit` closure.

    Holds a strong reference to the last `problem` it specialized for, so a repeat `.solve()`
    call on the same solver instance and problem object reuses the same compiled closure (and
    therefore XLA's own compilation cache, keyed on that closure's identity and the traced args'
    abstract shapes) instead of recompiling -- the MPC case a jitted core exists for. Holding the
    reference also rules out an `id()` collision with an unrelated, later object: `problem` can't
    be garbage-collected while its cache entry is alive.

    A single slot, not a dict, because the intended reuse case is one solver instance against one
    `problem` object solved repeatedly; a size-1 cache captures that without unbounded growth.
    """

    __slots__ = ("_jitted", "_key", "_problem_ref")

    def __init__(self) -> None:
        self._problem_ref: object = None
        self._key: object = None
        self._jitted: Callable[..., Any] | None = None

    def get_or_build(
        self, fn: Callable[..., Any], problem: object, key: object, **static_kwargs: object
    ) -> Callable[..., Any]:
        """Return `jax.jit(functools.partial(fn, problem=problem, **static_kwargs))`, reused from the last call when `problem` (by identity) and `key` are unchanged.

        `problem` and `static_kwargs` (e.g. `options`, `solve_kd_builder`) are closed over via
        `functools.partial` rather than passed as jit arguments: `problem` carries structural
        constraint data (`ALConstraints.build`/`PNLayout.build` convert its bounds with eager
        `np.asarray`, which requires concrete values and breaks under trace) and any callable in
        `static_kwargs` cannot be a traced pytree leaf at all. `key` should hash/compare equal
        exactly when `static_kwargs`' values do (a hashable `SolverOptions`, and a `solve_kd_builder`
        compared the way the caller wants -- identity for a memoized builder, or `None`).

        Parameters
        ----------
        fn : Callable
            The traced core to specialize (e.g. `al_solve`).
        problem : Problem
            Closed over by identity; a different `problem` object rebuilds the closure.
        key : object
            Everything else `static_kwargs` binds, compared for equality to decide reuse.
        **static_kwargs : object
            Bound into `fn` via `functools.partial` before jitting.

        Returns
        -------
        Callable
            The cached (or freshly built) jitted closure. Call it with the remaining, genuinely
            dynamic arguments as keywords -- `problem` occupies `fn`'s first positional slot, so a
            positional call would collide with its keyword binding.
        """
        if self._problem_ref is problem and self._key == key and self._jitted is not None:
            return self._jitted
        jitted = jax.jit(functools.partial(fn, problem=problem, **static_kwargs))
        self._problem_ref = problem
        self._key = key
        self._jitted = jitted
        return jitted
