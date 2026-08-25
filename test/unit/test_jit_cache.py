import jax.numpy as jnp

from trajopt.solvers._jit_cache import JitCacheSlot


def test_jit_cache_slot_reuses_closure_for_same_problem_identity_and_key() -> None:
    """A second `get_or_build` with the same `problem` object and an equal `key` returns the identical closure."""
    slot = JitCacheSlot()
    problem = object()

    first = slot.get_or_build(lambda x, problem: x, problem, key="k")  # noqa: ARG005 -- fn signature mirrors partial-bound call sites
    second = slot.get_or_build(lambda x, problem: x, problem, key="k")  # noqa: ARG005

    assert first is second


def test_jit_cache_slot_rebuilds_on_different_key() -> None:
    """A `key` that compares unequal to the cached one forces a new closure even for the same `problem`."""
    slot = JitCacheSlot()
    problem = object()

    first = slot.get_or_build(lambda x, problem: x, problem, key="a")  # noqa: ARG005
    second = slot.get_or_build(lambda x, problem: x, problem, key="b")  # noqa: ARG005

    assert first is not second


def test_jit_cache_slot_rebuilds_on_different_problem_identity() -> None:
    """A different `problem` object (even if `==`-equal) forces a new closure, since identity -- not equality -- is compared."""
    slot = JitCacheSlot()

    first = slot.get_or_build(lambda x, problem: x, object(), key="k")  # noqa: ARG005
    second = slot.get_or_build(lambda x, problem: x, object(), key="k")  # noqa: ARG005

    assert first is not second


def test_jit_cache_slot_hit_avoids_retracing() -> None:
    """Calling the cached closure twice with same-shaped args traces the wrapped function only once -- the actual point of caching it."""
    trace_count = 0

    def fn(x: jnp.ndarray, problem: float) -> jnp.ndarray:
        nonlocal trace_count
        trace_count += 1
        return x * 2.0 + problem

    slot = JitCacheSlot()
    problem = 3.0

    for _ in range(3):
        jitted = slot.get_or_build(fn, problem, key="k")
        _ = jitted(x=jnp.arange(4.0))

    assert trace_count == 1
