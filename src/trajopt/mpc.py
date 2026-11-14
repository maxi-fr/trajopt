from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.trajectory import Trajectory
from trajopt.transcription.result import Solver, SolverResult, constraint_row_count


def _targets(
    problem: Problem,
    xf: jax.Array | Sequence[float] | None,
    reference: Trajectory | None,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
    """Build the reference window and terminal goal for the tracking costs and goal constraints.

    Goal and window are independent: a window alone tracks and leaves goal constraints at their
    build-time target, a goal alone is also spread into a constant window so it still regulates
    the cost, and both together track the window while binding the goal.

    Raises
    ------
    ValueError
        If a constant goal is the thing aiming an objective that already tracks a build-time
        reference (which the goal would flatten), or if a target is given that nothing reads.
    """
    if xf is not None and reference is None and problem.obj.carries_reference:
        msg = (
            "This objective already tracks a build-time reference, so a constant goal xf would "
            "silently overwrite it at every knot point. Pass reference=Trajectory(...) with the "
            "window you want tracked, or build the objective with LQRObjective, which carries "
            "shape only."
        )
        raise ValueError(msg)
    if (xf is not None or reference is not None) and not (
        problem.obj.can_retarget or problem.constraints.has_goal_constraint()
    ):
        msg = (
            f"A run-time target was given but nothing in the problem reads it: the "
            f"{type(problem.obj.stage_cost).__name__} objective is not quadratic, so it exposes no linear "
            f"terms to retarget, and no GoalConstraint is registered. Use a quadratic objective, or add a "
            f"GoalConstraint, or leave the target unset."
        )
        raise ValueError(msg)

    xf_arr = None if xf is None else jnp.asarray(xf, dtype=jnp.float64)
    if reference is not None:
        return (
            jnp.asarray(reference.X, dtype=jnp.float64),
            jnp.asarray(reference.U, dtype=jnp.float64),
            xf_arr,
        )
    if xf_arr is not None:
        N, m = int(problem.N), int(problem.model.m)
        return (
            jnp.repeat(xf_arr[None, :], N, axis=0),
            jnp.zeros((N - 1, m), dtype=jnp.float64),
            xf_arr,
        )
    return None, None, None


class MPC:
    """Receding-horizon driver: a `Program`, the current `BoundaryConditions`, and a reference cursor.

    Structure, compiled state and per-step data each sit where they belong. Structure -- model,
    objective shape, constraints, horizon, time grid -- is the immutable `Problem`. The compiled
    and allocated form of that Problem for one solver is the `Program`, built once here and reused
    for the life of the driver. What genuinely changes between steps is split in two: traced
    boundary data in `BoundaryConditions`, and the primal/dual iterates in a private `WarmStart`.

    The reference is a pushed window. `push_reference` stages the point that enters the Horizon at
    the far end; `advance` advances the window by `steps` Knot Points and appends it, holding the last point
    when nothing was pushed. A constant window is shift-invariant, so regulating to a fixed goal
    and tracking a moving one are the same mechanism with nothing special-cased.

    The terminal goal is not part of that window. It rides along in the boundary conditions as its
    own leaf, so a window that genuinely moves does not drag the destination a terminal constraint
    binds onto whatever waypoint happens to sit N knots out.

    Parameters
    ----------
    problem : Problem
        Structural problem to drive. Held by reference and never mutated.
    solver : Solver | None, optional
        Solver backend (e.g. `ALTRO()`, `Ipopt()`, `OSQP(...)`). Defaults to None, meaning
        `Ipopt()`, resolved here rather than at import time to avoid an import cycle.
    x0 : jax.Array | Sequence[float]
        Initial measured state of shape (n,).
    t0 : float | jax.Array, optional
        Initial timestamp. Defaults to 0.0.
    xf : jax.Array | Sequence[float] | None, optional
        Run-time terminal goal of shape (n,) that goal constraints bind. On its own it also
        regulates the cost, spread into a constant reference window. Defaults to None, which
        leaves goal constraints at the target they were built with.
    reference : Trajectory | None, optional
        Reference window of N knot points for the objective to track. May be combined with xf,
        which then names the destination only and no longer aims the cost. Defaults to None.
    initial_trajectory : Trajectory | None, optional
        Warm-start trajectory guess. Defaults to x0 repeated with zero controls.
    initial_z : jax.Array | None, optional
        Flat warm-start guess of shape (N * n + (N - 1) * m,), taking precedence over
        `initial_trajectory`.
    penalty_diagnostics : bool, optional
        Report violations and solved auxiliary values in `result.info["penalties"]`.
    """

    def __init__(  # noqa: PLR0913 -- the driver's boundary data is 6 optional arguments wide
        self,
        problem: Problem,
        solver: Solver | None = None,
        *,
        x0: jax.Array | Sequence[float],
        t0: float | jax.Array = 0.0,
        xf: jax.Array | Sequence[float] | None = None,
        reference: Trajectory | None = None,
        initial_trajectory: Trajectory | None = None,
        initial_z: jax.Array | None = None,
        penalty_diagnostics: bool = False,
    ) -> None:
        """Build the physical MPC state and configure optional penalty diagnostics."""
        if solver is None:
            from trajopt.transcription.ipopt import Ipopt  # noqa: PLC0415 -- avoid an import cycle

            solver = Ipopt()

        X_ref, U_ref, xf_arr = _targets(problem, xf, reference)
        x0_arr = jnp.asarray(x0, dtype=jnp.float64)

        self.program = Program(problem, solver, penalty_diagnostics=penalty_diagnostics)
        self.bc = BoundaryConditions(
            x0=x0_arr,
            t0=jnp.asarray(t0, dtype=jnp.float64),
            X_ref=X_ref,
            U_ref=U_ref,
            xf=xf_arr,
        )
        self.result: SolverResult | None = None
        self._ws = WarmStart.cold(problem, x0_arr, initial_trajectory=initial_trajectory, initial_z=initial_z)
        self._pending_ref: tuple[jax.Array, jax.Array] | None = None
        self._goal_aims_cost = xf_arr is not None and reference is None

    @property
    def problem(self) -> Problem:
        """Structural problem this driver runs, owned by its Program."""
        return self.program.problem

    @property
    def solver(self) -> Solver:
        """Solver backend this driver's Program is compiled for."""
        return self.program.solver

    @property
    def warm_start(self) -> WarmStart:
        """Current primal and dual iterates, as handed to the next solve."""
        return self._ws

    @property
    def x0(self) -> jax.Array:
        """Measured initial state of shape (n,)."""
        return self.bc.x0

    @property
    def t0(self) -> jax.Array:
        """Current timestamp of shape ()."""
        return self.bc.t0

    @property
    def xf(self) -> jax.Array | None:
        """Run-time terminal goal of shape (n,) that goal constraints bind, or None."""
        return self.bc.xf

    @property
    def Z(self) -> jax.Array:  # noqa: N802 -- matches the flat primal vector's name throughout
        """Flat warm-start primal vector of shape (N * n + (N - 1) * m,)."""
        return self._ws.Z

    @property
    def states(self) -> jax.Array:
        """Warm-start state trajectory X of shape (N, n)."""
        return self._ws.unpack(self.problem)[0]

    @property
    def controls(self) -> jax.Array:
        """Warm-start control trajectory U of shape (N - 1, m); `controls[0]` is the command to apply."""
        return self._ws.unpack(self.problem)[1]

    def trajectory(self) -> Trajectory:
        """Return the current warm start as a Trajectory on the absolute time grid starting at t0."""
        problem = self.problem
        X, U = self._ws.unpack(problem)
        dt = problem.dt
        t = self.bc.t0 + jnp.concatenate([jnp.zeros(1, dtype=dt.dtype), jnp.cumsum(dt)])
        return Trajectory(X=X, U=U, t=t, dt=dt)

    def measure(self, x: jax.Array | Sequence[float], t: float | jax.Array | None = None) -> None:
        """Update Boundary Conditions and pin the warm start's first Knot Point to measured state.

        Parameters
        ----------
        x : jax.Array | Sequence[float]
            Measured state of shape (n,).
        t : float | jax.Array | None, optional
            Timestamp of the measurement. Defaults to None, keeping current timestamp t0.
        """
        x_arr = jnp.asarray(x, dtype=self.bc.x0.dtype)
        t0_val = self.bc.t0 if t is None else jnp.asarray(t, dtype=self.bc.t0.dtype)
        self.bc = BoundaryConditions(
            x0=x_arr,
            t0=t0_val,
            X_ref=self.bc.X_ref,
            U_ref=self.bc.U_ref,
            xf=self.bc.xf,
        )
        self._ws = self._ws.with_x0(self.problem, x_arr)

    def set_goal(self, xf: jax.Array | Sequence[float]) -> None:
        """Move the terminal goal to xf of shape (n,), re-aiming the cost when the goal aims it.

        A driver built from a goal alone has no window of its own, so the goal is also spread back
        over the horizon as the constant window the cost tracks. One built with its own reference
        window keeps that window and moves only the destination.

        Raises
        ------
        ValueError
            If this driver was built without a goal, since nothing was checked to read one.
        """
        if self.bc.xf is None:
            msg = "This MPC was built without a goal. Pass xf to MPC(...) to make the goal run-time."
            raise ValueError(msg)
        xf_arr = jnp.asarray(xf, dtype=self.bc.xf.dtype)
        X_ref = jnp.repeat(xf_arr[None, :], int(self.problem.N), axis=0) if self._goal_aims_cost else self.bc.X_ref
        self.bc = BoundaryConditions(x0=self.bc.x0, t0=self.bc.t0, X_ref=X_ref, U_ref=self.bc.U_ref, xf=xf_arr)

    def set_reference(self, window: Trajectory) -> None:
        """Replace the tracked window wholesale with `window`, a Trajectory of N knot points.

        The terminal goal is untouched: the window says what to follow, xf says where to end.
        """
        self.bc = BoundaryConditions(
            x0=self.bc.x0,
            t0=self.bc.t0,
            X_ref=jnp.asarray(window.X, dtype=jnp.float64),
            U_ref=jnp.asarray(window.U, dtype=jnp.float64),
            xf=self.bc.xf,
        )

    def push_reference(
        self,
        x_ref: jax.Array | Sequence[float],
        u_ref: jax.Array | Sequence[float] | None = None,
    ) -> None:
        """Stage the reference point the next `advance` appends at the far end of the window.

        Parameters
        ----------
        x_ref : jax.Array | Sequence[float]
            Reference state of shape (n,) entering the horizon.
        u_ref : jax.Array | Sequence[float] | None, optional
            Reference control of shape (m,) paired with it. Defaults to None, meaning zeros.
        """
        if self.bc.X_ref is None or self.bc.U_ref is None:
            msg = "This MPC was built without a reference window. Pass xf or reference to MPC(...)."
            raise ValueError(msg)
        u = (
            jnp.zeros(self.bc.U_ref.shape[-1], dtype=self.bc.U_ref.dtype)
            if u_ref is None
            else jnp.asarray(u_ref, dtype=self.bc.U_ref.dtype)
        )
        self._pending_ref = (jnp.asarray(x_ref, dtype=self.bc.X_ref.dtype), u)

    def _advanced_reference(self, steps: int) -> tuple[jax.Array | None, jax.Array | None]:
        """Advance the reference window by `steps` Knot Points using endpoint hold semantics."""
        X_ref, U_ref = self.bc.X_ref, self.bc.U_ref
        if X_ref is None or U_ref is None:
            return None, None
        for _ in range(steps):
            x_new, u_new = (X_ref[-1], U_ref[-1]) if self._pending_ref is None else self._pending_ref
            self._pending_ref = None
            X_ref = jnp.concatenate([X_ref[1:], x_new[None, :]], axis=0)
            U_ref = jnp.concatenate([U_ref[1:], u_new[None, :]], axis=0)
        return X_ref, U_ref

    def advance(self, steps: int = 1) -> None:
        """Advance the Horizon by `steps` Knot Points: move warm start, clock, and reference window.

        Parameters
        ----------
        steps : int, optional
            Number of native Knot Points to advance, in [0, H] where H = N - 1. Defaults to 1.

        Raises
        ------
        TypeError
            If `steps` is a boolean or not an integer.
        ValueError
            If `steps` is negative or exceeds H = N - 1.
        """
        if isinstance(steps, (bool, np.bool_)):
            msg = f"steps must be an integer, got {type(steps).__name__}."
            raise TypeError(msg)
        if not isinstance(steps, (int, np.integer)):
            msg = f"steps must be an integer, got {type(steps).__name__}."
            raise TypeError(msg)

        h = int(self.problem.N) - 1
        if steps < 0 or steps > h:
            msg = f"steps must be between 0 and {h}, got {steps}."
            raise ValueError(msg)

        if steps == 0:
            return

        problem = self.problem
        self._ws = self._ws.advance(problem, steps)
        t0_advanced = self.bc.t0 + jnp.sum(problem.dt[:steps])
        X_ref, U_ref = self._advanced_reference(steps)
        self.bc = BoundaryConditions(
            x0=self._ws.unpack(problem)[0][0],
            t0=t0_advanced,
            X_ref=X_ref,
            U_ref=U_ref,
            xf=self.bc.xf,
        )

    def solve(self) -> SolverResult:
        """Solve one Horizon, validate canonical Multipliers, fold into the warm start, and return the Backend's result.

        A Backend that returns no duals (an empty vector) leaves the warm start's own untouched,
        and only the AL solvers populate `res.al`, so a prior AL warm start survives a step taken
        by a solver that knows nothing about it.
        """
        res = self.program.solve(self.bc, self._ws)

        if len(res.lam) > 0:
            p_canon = constraint_row_count(self.problem)
            if len(res.lam) != p_canon:
                msg = f"Multiplier lam has length {len(res.lam)}, expected canonical constraint count {p_canon}."
                raise ValueError(msg)
        if len(res.mu) > 0 and len(res.mu) != len(self._ws.Z):
            msg = f"Multiplier mu has length {len(res.mu)}, expected canonical Primal Vector length {len(self._ws.Z)}."
            raise ValueError(msg)

        lam = jnp.asarray(res.lam, dtype=self._ws.Z.dtype) if len(res.lam) > 0 else self._ws.lam
        mu = jnp.asarray(res.mu, dtype=self._ws.Z.dtype) if len(res.mu) > 0 else self._ws.mu
        al = getattr(res, "al", None)

        self._ws = WarmStart(Z=res.Z, lam=lam, mu=mu, al=self._ws.al if al is None else al)
        self.result = res
        return res

    def cost(self) -> jax.Array:
        """Evaluate the Objective and penalties at the current physical warm start."""
        from trajopt.transcription.transcription import eval_f  # noqa: PLC0415 -- avoid circular import

        problem = self.problem
        return eval_f(problem, self._ws.Z, self.bc.t0, problem.dt, self.bc)
