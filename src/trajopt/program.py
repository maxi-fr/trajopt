import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.transcription.layout import _trajectory_to_z, _z_to_trajectory

if TYPE_CHECKING:
    from trajopt.problem import BoundaryConditions, Problem
    from trajopt.solvers.al import ALConstraints
    from trajopt.trajectory import Trajectory
    from trajopt.transcription.result import Solver, SolverResult


class WarmStart(eqx.Module):
    """The primal and dual iterates one receding-horizon step hands to the next.

    Shape lives in the `Problem`, so every method that needs the horizon takes one rather than
    carrying a second copy of `N`, `n` and `m`.

    Parameters
    ----------
    Z : jax.Array
        Flat primal trajectory of shape (N * n + (N - 1) * m,).
    lam : jax.Array
        Transcription constraint duals of shape (P,).
    mu : jax.Array
        Signed primal-bound duals, one per entry of `Z`.
    al : ALConstraints | None
        Padded augmented-Lagrangian duals and penalties, or None until an AL solve populates them.
    """

    Z: jax.Array
    lam: jax.Array
    mu: jax.Array
    al: "ALConstraints | None" = None

    @classmethod
    def cold(
        cls,
        problem: "Problem",
        x0: jax.Array,
        *,
        initial_trajectory: "Trajectory | None" = None,
        initial_z: jax.Array | None = None,
    ) -> "WarmStart":
        """Zero duals and a primal guess: `initial_z`, else `initial_trajectory`, else x0 held with zero controls."""
        N = int(problem.N)
        n = int(problem.model.n)
        m = int(problem.model.m)

        if initial_z is not None:
            Z = jnp.asarray(initial_z, dtype=jnp.float64)
        elif initial_trajectory is not None:
            Z = _trajectory_to_z(initial_trajectory.X, initial_trajectory.U)
        else:
            Z = _trajectory_to_z(
                jnp.repeat(jnp.asarray(x0, dtype=jnp.float64)[None, :], N, axis=0),
                jnp.zeros((N - 1, m), dtype=jnp.float64),
            )

        P_total = n + (N - 1) * n + sum(problem.constraints.p)
        return cls(Z=Z, lam=jnp.zeros(P_total, dtype=jnp.float64), mu=jnp.zeros(len(Z), dtype=jnp.float64))

    def unpack(self, problem: "Problem") -> tuple[jax.Array, jax.Array]:
        """States of shape (N, n) and controls of shape (N - 1, m) parsed out of `Z`."""
        return _z_to_trajectory(self.Z, int(problem.N), int(problem.model.n), int(problem.model.m))

    def with_x0(self, problem: "Problem", x0: jax.Array) -> "WarmStart":
        """Return this warm start with the first knot of `Z` pinned to the measured state x0 of shape (n,)."""
        X, U = self.unpack(problem)
        return eqx.tree_at(lambda w: w.Z, self, _trajectory_to_z(X.at[0].set(jnp.asarray(x0, dtype=X.dtype)), U))

    def with_primal(self, problem: "Problem", *, X: jax.Array | None = None, U: jax.Array | None = None) -> "WarmStart":
        """Return this warm start with the states and/or controls in `Z` replaced."""
        X_cur, U_cur = self.unpack(problem)
        X_new = X_cur if X is None else jnp.asarray(X, dtype=X_cur.dtype)
        U_new = U_cur if U is None else jnp.asarray(U, dtype=U_cur.dtype)
        return eqx.tree_at(lambda w: w.Z, self, _trajectory_to_z(X_new, U_new))

    def shift(self, problem: "Problem") -> "WarmStart":
        """Advance this warm start one knot: every primal and dual quantity shifts with the horizon.

        Knot k of the new horizon is knot k + 1 of the old one, so each quantity drops its first
        knot and holds its last, mirroring `Z`. Holding is the right vacancy fill for a multiplier
        as well as for a state: the knot that enters the horizon repeats the old final knot's
        state, control and constraint rows, so the old final multiplier is the best available
        estimate, and holding preserves the sign a `NegativeOrthant` row's multiplier must keep.

        Only rows with no counterpart at all are zeroed instead: `lam`'s terminal knot block,
        whose rows come from a different (terminal) evaluator than any stage knot's, and any
        stage knot block that would have to source from that terminal block or from a knot of a
        different row count -- see `_lam_shift_index`. `lam`'s initial-condition block does not move -- it belongs to the x0 pin,
        which stays at the head of the horizon. `al` is masked back to the new horizon's own
        `row_mask` after the shift: its `lam` falls back to zero where the source row is not a
        real row, while its `mu` falls back to the destination knot's own penalty, because a
        penalty is outer-loop schedule state rather than a multiplier and a real row must never be
        left at zero penalty.
        """
        X, U = self.unpack(problem)
        Z = _trajectory_to_z(
            jnp.concatenate([X[1:], X[-1:]], axis=0),
            jnp.concatenate([U[1:], U[-1:]], axis=0),
        )
        return WarmStart(Z=Z, lam=self._shifted_lam(problem), mu=self._shifted_mu(problem), al=_shifted_al(self.al))

    def _shifted_lam(self, problem: "Problem") -> jax.Array:
        """Transcription duals of shape (P,) advanced one knot, gathered through `_lam_shift_index`."""
        index = _lam_shift_index(problem)
        return jnp.where(index >= 0, self.lam[index], 0.0)

    def _shifted_mu(self, problem: "Problem") -> jax.Array:
        """Primal-bound duals of shape (len(Z),) advanced one knot exactly as `Z` is."""
        mu_X, mu_U = _z_to_trajectory(self.mu, int(problem.N), int(problem.model.n), int(problem.model.m))
        return _trajectory_to_z(
            jnp.concatenate([mu_X[1:], mu_X[-1:]], axis=0),
            jnp.concatenate([mu_U[1:], mu_U[-1:]], axis=0),
        )


def _lam_shift_index(problem: "Problem") -> jax.Array:
    """Gather index of shape (P,) mapping the shifted transcription duals onto the old ones.

    Entry i is the old row that new row i takes its multiplier from, or -1 where the new row has
    no counterpart. `constraints_and_jac` lays the rows out as `[x0 pin (n) | defect_k (n), knot
    rows k (p_k) for k < N - 1 | knot rows N - 1]`, a knot contributing no block when p_k is 0.
    The pin maps to itself, defect and stage-knot blocks map one knot forward, and the last
    defect holds its own multiplier (the knot entering the horizon repeats the old final one).
    Rows with no counterpart map to -1: the terminal block, whose rows come from a different
    evaluator than any stage knot's; the last stage block, whose source would be that terminal
    block; and any stage block whose width differs from its source's.
    """
    n = int(problem.model.n)
    N = int(problem.N)
    p = tuple(problem.constraints.p)

    offsets, off = [], n
    for k in range(N):
        defect = None if k == N - 1 else off
        off += 0 if defect is None else n
        block = (off, off + p[k])
        off = block[1]
        offsets.append((defect, block))

    index = np.full(off, -1, dtype=np.int32)
    index[:n] = np.arange(n)
    for k in range(N - 1):
        defect, block = offsets[k]
        src_defect, src_block = offsets[k + 1]
        if defect is not None:
            src = defect if src_defect is None else src_defect
            index[defect : defect + n] = np.arange(src, src + n)
        if k + 1 < N - 1 and block[1] - block[0] == src_block[1] - src_block[0]:
            index[block[0] : block[1]] = np.arange(src_block[0], src_block[1])
    return jnp.asarray(index)


def _shifted_al(al: "ALConstraints | None") -> "ALConstraints | None":
    """Advance padded AL duals and penalties one knot, remasked onto the new horizon's real rows."""
    if al is None:
        return None

    def shift(a: jax.Array) -> jax.Array:
        """Drop the first knot of a (N, p_max) array and hold its last."""
        return jnp.concatenate([a[1:], a[-1:]], axis=0)

    live = al.row_mask & shift(al.row_mask)
    return eqx.tree_at(
        lambda a: (a.lam, a.mu),
        al,
        (jnp.where(live, shift(al.lam), 0.0), jnp.where(live, shift(al.mu), al.mu)),
    )


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

    def solve(self, bc: "BoundaryConditions", ws: WarmStart) -> "SolverResult":
        """Solve this program's problem from boundary conditions `bc` and warm start `ws`."""
        return self.solver.solve(self, bc, ws)
