from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from trajopt.constraints.bounds import ControlBound, StateBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.costs.objective import LQRObjective
from trajopt.dynamics.integrators import RK4
from trajopt.models.cartpole import Cartpole
from trajopt.problem import MPCState, Problem
from trajopt.solvers.altro import ALTRO

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "mpc_cartpole_closed_loop.npz"

N = 40
DT = 0.05
N_STEPS = 40
KICK_STEP = 20
KICK = 1.5
U_MAX = 20.0
TRACK = 0.8


def _build():
    """Cartpole balancing problem of examples/05: bounded force, bounded track, LQR to upright.

    Returns the problem, its discretized model, the initial state, and the upright goal, which is
    now run-time data the MPCState carries rather than a target baked into the objective.
    """
    n, m = 4, 1
    x0 = jnp.array([0.0, np.pi - 0.25, 0.1, -0.2], dtype=jnp.float64)
    xf = jnp.array([0.0, np.pi, 0.0, 0.0], dtype=jnp.float64)

    obj = LQRObjective(
        Q=jnp.diag(jnp.array([5.0, 20.0, 1.0, 2.0])) * DT,
        R=jnp.diag(jnp.array([0.05])) * DT,
        Qf=jnp.diag(jnp.array([50.0, 200.0, 10.0, 20.0])),
        N=N,
    )

    clist = ConstraintList(n=n, m=m, N=N)
    clist.add_constraint(ControlBound(n=n, m=m, u_min=[-U_MAX], u_max=[U_MAX]), range(N - 1))
    clist.add_constraint(
        StateBound(
            n=n,
            m=m,
            x_min=[-TRACK, -np.inf, -np.inf, -np.inf],
            x_max=[TRACK, np.inf, np.inf, np.inf],
        ),
        range(N),
    )

    model = Cartpole(mc=1.0, mp=0.2, l=0.5, g=9.81)
    integrator = RK4()
    prob = Problem(model=model, obj=obj, constraints=clist, N=N, integrator=integrator)
    return prob, model.discretize(integrator), x0, xf


def _record_ipopt_seed(prob, x0, xf):
    """Solve the first horizon with Ipopt, returning the Z that puts ALTRO on the balancing branch.

    Only the one-off golden regeneration calls this; the comparison path replays the Z stored in
    the golden archive, so the test needs no solver extra at run time.
    """
    pytest.importorskip("cyipopt", reason="regenerating the golden needs the Ipopt seed")
    from trajopt.transcription.ipopt import Ipopt

    state = MPCState.initial(prob, x0=x0, dt=DT, xf=xf)
    return np.asarray(prob.solve(state, solver=Ipopt()).Z)


def _run_closed_loop(z_seed):
    """Run the receding-horizon loop from the seeded plan, returning closed-loop states and controls."""
    prob, dmodel, x0, xf = _build()
    state = MPCState.initial(prob, x0=x0, dt=DT, xf=xf, initial_z=jnp.asarray(z_seed))
    solver = ALTRO()

    x_curr, t_curr = x0, 0.0
    X_hist, U_hist = [np.asarray(x_curr)], []

    for step in range(N_STEPS):
        if step == KICK_STEP:
            x_curr = x_curr.at[3].add(KICK)

        state = state.with_measurement(x_curr, t_curr)
        state = prob.solve(state, solver=solver)

        u_cmd = state.controls[0]
        U_hist.append(np.asarray(u_cmd))

        x_curr = dmodel.discrete_dynamics(x_curr, u_cmd, t_curr, DT)
        X_hist.append(np.asarray(x_curr))

        state = state.shift(DT)
        t_curr += DT

    return np.array(X_hist), np.array(U_hist)


@pytest.fixture(scope="module")
def closed_loop():
    """Closed-loop (X, U, golden-or-None); records the golden, seed included, when it is missing."""
    if GOLDEN.exists():
        ref = np.load(GOLDEN)
        X, U = _run_closed_loop(ref["Z_seed"])
        return X, U, ref

    prob, _, x0, xf = _build()
    z_seed = _record_ipopt_seed(prob, x0, xf)
    X, U = _run_closed_loop(z_seed)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, X=X, U=U, Z_seed=z_seed)
    return X, U, None


def test_closed_loop_matches_golden(closed_loop):
    """Closed-loop trajectory reproduces the recorded pre-refactor baseline."""
    X, U, ref = closed_loop
    if ref is None:
        pytest.skip(f"golden recorded at {GOLDEN}; re-run to compare")

    np.testing.assert_allclose(X, ref["X"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(U, ref["U"], rtol=1e-6, atol=1e-8)


def test_closed_loop_respects_bounds(closed_loop):
    """Commanded force and cart position stay inside their box bounds all run."""
    X, U, _ = closed_loop
    assert np.all(np.abs(U) <= U_MAX + 1e-6)
    assert np.all(np.abs(X[:, 0]) <= TRACK + 1e-6)


def test_closed_loop_recovers_from_kick(closed_loop):
    """The pole dips after the impulse, then climbs back and stays near upright, not that it reaches it.

    The claim is recovery, not convergence: within the 2 s window the angle never returns to pi. It
    swings down to a minimum after the kick, rises from there without reversing, and finishes within
    0.2 rad of upright with the angle error at most half of what the dip cost. Anything that lets the
    pole go over the top, stall at the dip, or oscillate on the way back fails this.
    """
    X, _, _ = closed_loop
    theta = X[:, 1]
    post = theta[KICK_STEP:]
    dip = int(np.argmin(post))

    assert dip > 0, "the pole should dip below its post-kick angle before recovering"
    climb = post[dip:]
    assert np.all(np.diff(climb) > 0), "recovery from the dip should be monotone in theta"

    dip_err = abs(float(post[dip]) - np.pi)
    final_err = abs(float(theta[-1]) - np.pi)
    assert final_err < 0.2
    assert final_err < 0.5 * dip_err
    assert np.all(np.abs(theta - np.pi) < 1.0), "the pole never goes over the top"
