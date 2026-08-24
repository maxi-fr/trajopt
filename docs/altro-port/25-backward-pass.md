# 25 — Backward pass and regularization

**What to build:** given an `Expansion` along a trajectory, produce the affine policy that iLQR
rolls out — feedback gains `K`, feedforward `d`, the cost-to-go `S`, and the expected decrease
`ΔV` — with Altro's regularization behaviour when the action-value Hessian is not positive
definite. Demonstrable on its own: for a linear-quadratic problem the gains this returns are the
analytic time-varying LQR gains, and for any problem they match `Altro.backwardpass!` driven from
the same expansion to ~1e-8.

**Blocked by:** 24.

## Architecture

`src/trajopt/solvers/ilqr.py`. Consumes the existing `Expansion` from `expansions.py` unchanged —
reference §1 is right that it is already equivalent to Altro's `CostExpansion` +
`DynamicsExpansion` pair, and no new expansion code belongs in this ticket.

**The traced shape is a `while_loop` over ρ wrapping a `scan` over knots.** Altro's Cholesky
failure handler resets `k = N-1` and restarts the whole recursion with a larger ρ, which is
exactly a retry loop around the sweep:

```code
while_loop(
  cond  = failed and rho <= bp_reg_max,
  body  = one full reversed scan over k = N-2..0, carrying (S_x, S_xx, dV),
          emitting (K, d) per knot and a `failed` flag,
)
```

**Cholesky failure is detected as NaN, not as an `isposdef` flag.** `jnp.linalg.cholesky`
returns NaN for a non-PD input rather than raising, so the scan carries
`failed |= isnan(L).any()` and finishes the sweep regardless. The gains from a failed sweep are
discarded by the retry. This is the one place the traced port cannot mirror Altro's early exit,
and it is a pure performance difference — same answer, one wasted sweep.

**`bp_reg_max` bounds the retry loop.** Finding F: Altro has no bound here and can spin forever.
On exhaustion, return the last gains together with a failure flag; ticket 27 turns that into a
termination status. Do not let the loop run unbounded.

**Regularization is a returned value, not a mutation.** `DynamicRegularization` is two floats
(ρ, dρ). `increase` and `decrease` are pure functions returning a new pair, and the pair rides in
the loop carry. Altro's `decreaseregularization!` runs **once per backward pass, at the end** —
not per knot, and not per retry.

**Get the regularized inverse right.** Finding G: both `K` and `d` are solved against
`Quu_reg = Quu + ρI`, not against `Quu`. Reference §4.2 writes `Quu⁻¹` for both, which is only
correct while ρ = 0 and will silently pass every test that never regularizes. Solve the stacked
`(m, n+1)` system once and negate, matching Altro's sign convention: `K = −Quu_reg⁻¹ Qux`,
`d = −Quu_reg⁻¹ Qu`, so the rollout in ticket 26 reads `u = ū + K δx + α d` with a plus.

Do not port `bp_reg_type` (finding H) — `:state` is broken upstream and never runs.

The cost-to-go Hessian is symmetrized as `(S + Sᵀ)/2` after the update, as in Altro.

## Julia parity

Set up the cross-verification environment here, since this is the first ticket that needs it, and
reference §8.1's plan does not work (finding K). Create a **third** Julia environment — a new
directory with its own `Project.toml` and a committed `Manifest.toml` — that `Pkg.develop`s both
`trajopt_jl/` and `altro_jl/`. Developing Altro into `trajopt_jl/` makes Altro a dependency of
TrajectoryOptimization, which depends on Altro: a cycle. Expect a registry fetch and an Octavian
precompile the first time; commit the resolved Manifest so it happens once. Add a session fixture
separate from `jl_to` so the existing cross tests do not pay the Altro load cost, and skip
cleanly when Julia is unavailable, as `jl_to` already does.

Reference §8.2 rows 3 and 4: gains, `S.x`, `S.xx`, `ΔV` per knot after one backward pass; ρ and
dρ after each increase and decrease event.

## Acceptance criteria

- [ ] `backward_pass` is a pure function from `(Expansion, regularization, options)` to gains, cost-to-go, `ΔV`, new regularization, and a failure flag; it runs under `jax.jit` with no Python-level branching on traced values.
- [ ] On a linear-quadratic problem the gains equal the analytic time-varying LQR gains to 1e-10.
- [ ] Feeding a deliberately indefinite `Quu` triggers the retry: ρ increases by the factor 1.6, the sweep re-runs, and the returned gains come from the successful sweep.
- [ ] The retry loop terminates when ρ exceeds `bp_reg_max`, returning the failure flag rather than hanging; a test asserts this completes.
- [ ] `K` and `d` are both solved against the **regularized** `Quu`; a test with ρ > 0 distinguishes this from the unregularized form.
- [ ] `increase` / `decrease` regularization are pure and match Altro's `ρ = max(ρ·dρ, ρmin)` / `dρ = min(dρ/ϕ, 1/ϕ)` state machine exactly; decrease runs once per backward pass.
- [ ] A Julia environment developing both vendored packages exists with a committed Manifest, plus a session fixture that skips cleanly when Julia is absent.
- [ ] Cross tests match `Altro.backwardpass!` on gains, `S`, and `ΔV` to 1e-8 for pendulum and cartpole, at ρ = 0 and at ρ > 0.
- [ ] pre-commit hooks pass
