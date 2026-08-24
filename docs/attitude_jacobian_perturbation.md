# Attitude Jacobian: left JPL vs. right Hamilton perturbation

This resolves the derivation [ticket 13](tickets/13-rigidbody-quadrotor-error-expansions.md) names as
outstanding: how the rotation block of this port's attitude Jacobian relates to the one Julia builds
from `Rotations.∇differential`, so that the attitude-Jacobian cross-test asserts a derived relation
rather than a fitted one. It is the companion to
[the operand-ordering derivation](quaternion_operand_ordering.md), which settled the same question
for the *error*, and it reuses that document's lemmas.

**Result.** The two are the same map, written in two storage orders:

$$\boxed{\;T\,\Xi(q) \;=\; \nabla_{\text{differential}}(\rho q)\;}$$

where $\Xi$ is `Quaternion.xi`, $\rho$ is the scalar-last-to-scalar-first re-index the quadrotor
cross-test already applies to the state, and $T$ is that same re-index acting on the rows of a
$4 \times 3$ block. No sign, no transpose, no rotation matrix between them — unlike the error, where
the two conventions differ by $R(q_{\text{ref}})^{T}$.

That is not a coincidence and it is not in tension with ticket 13's premise. Julia's Jacobian *is* a
right multiplicative perturbation and this port's *is* a left one. They coincide because $\rho$
reverses the product order, converting one into the other exactly.

**The factor of one half** in `attitude_jacobian` is the derivative of this port's own error map. The
error state is $\delta\theta = 2\operatorname{vec}(\delta q)$, so the chain rule contributes
$\partial\operatorname{vec}(\delta q)/\partial\delta\theta = \tfrac{1}{2}I$. It is not a fitted
constant, and it is the same one half the cross-test applies to the Julia block.

Verified symbolically on free symbols and numerically against a live `Rotations.jl` to `1.1e-16`.

## Notation

Quaternions are written scalar-last, $q = (v, w)$, as this port stores them. The products are the
ones tabulated in [the operand-ordering derivation](quaternion_operand_ordering.md#notation):

| | Product $a \otimes b$ |
| :--- | :--- |
| JPL | $\big(w_b v_a + w_a v_b - v_a \times v_b,\;\; w_a w_b - v_a \cdot v_b\big)$ |
| Hamilton | $\big(w_a v_b + w_b v_a + v_a \times v_b,\;\; w_a w_b - v_a \cdot v_b\big)$ |

Two maps carry quaternions across to Julia, and they are not the same map:

- $B(q) = (-v, w)$, the **bridge** of ticket 02. It preserves the rotation matrix,
  $R_{\text{JPL}}(q) = R_{\text{Ham}}(B(q))$, and by L2 there it is a product *isomorphism*.
- $\rho(q) = (w, v)$, a bare **re-index** into scalar-first storage with no change of sign. This is
  what `test_cross_models.py` applies through `T_quat`, and it satisfies
  $R_{\text{JPL}}(q) = R_{\text{Ham}}(\rho q)^{T}$.

The two are related by conjugation, $\rho(q) = \overline{B(q)}$ up to storage order. Both are correct
bridges to different Julia quantities; the quadrotor cross-test needs $\rho$, because RobotZoo's
`q * F_body` reproduces this port's `R(q)ᵀ @ F_body`.

$T$ denotes the matrix of $\rho$, acting on the four rows of a $4 \times 3$ block:

$$T = \begin{pmatrix} 0&0&0&1 \\ 1&0&0&0 \\ 0&1&0&0 \\ 0&0&1&0 \end{pmatrix}.$$

## L3 — the re-index is an anti-isomorphism

$$\rho(a \otimes_{\text{JPL}} b) \;=\; \rho(b) \otimes_{\text{Ham}} \rho(a).$$

*Proof.* This is L1 of the operand-ordering derivation, $\text{hamilton}(a,b) = \text{jpl}(b,a)$, read
in the other direction: the scalar parts agree termwise, and the vector parts are
$w_b v_a + w_a v_b - v_a \times v_b$ against $w_b v_a + w_a v_b + v_b \times v_a$, equal because the
cross product is antisymmetric. $\square$

Structurally: $B$ is an isomorphism (L2) and conjugation is an anti-automorphism, so their composite
$\rho$ reverses order. It is order reversal that does all the work below. The order-*preserving*
reading is false by a margin that will matter for the test case — symbolically,

$$\rho(a \otimes_{\text{JPL}} b) - \rho(a) \otimes_{\text{Ham}} \rho(b) \;=\; -2\,(v_a \times v_b)$$

in the vector part, zero in the scalar part.

## The two perturbations

**This port, left and JPL.** The error is $q_{\text{err}} = q \otimes_{\text{JPL}} q_{\text{ref}}^{-1}$,
so reconstructing the state from an error means multiplying on the left:

$$q(a) \;=\; \delta q(a) \otimes_{\text{JPL}} q, \qquad \delta q(a) = \big(a,\, \sqrt{1 - \lVert a \rVert^2}\big).$$

Differentiating at $a = 0$, where $\partial\sqrt{1-\lVert a\rVert^{2}}/\partial a = 0$:

$$\frac{\partial q(a)}{\partial a}\bigg|_{a=0} = \begin{pmatrix} w I + [v]_\times \\ -v^{T} \end{pmatrix} = \Xi(q),$$

which is `Quaternion.xi` term for term. The vector block is $wI + [v]_\times$ rather than
$wI - [v]_\times$ precisely because the perturbation is on the left.

**Julia, right and Hamilton.** `Rotations.∇differential(q)` is documented, and implemented, as the
Jacobian of `lmult(q) * QuatMap(ϕ)` at $\phi = 0$ — that is, $q \otimes_{\text{Ham}} (\phi, 1)$ in
scalar-first storage, a perturbation applied on the right. This matches its error convention:
`rotation_error` returns $q_{\text{ref}}^{-1} \otimes_{\text{Ham}} q$, a right multiplicative error.

## The shipping statement

Apply $\rho$ to this port's perturbed quaternion and use L3:

$$\rho\big(\delta q(a) \otimes_{\text{JPL}} q\big) \;=\; \rho(q) \otimes_{\text{Ham}} \rho\big(\delta q(a)\big).$$

To first order $\rho(\delta q(a)) = (1, a)$, which is exactly Julia's `QuatMap(a)`. So the left-hand
side, differentiated at $a = 0$, is $T\,\Xi(q)$ — $\rho$ is linear, so it passes through the
derivative as the matrix $T$ — and the right-hand side is the Jacobian of
`lmult(ρq) QuatMap(a)`, which is $\nabla_{\text{differential}}(\rho q)$. Hence

$$T\,\Xi(q) = \nabla_{\text{differential}}(\rho q). \qquad \square$$

The left-versus-right difference and the JPL-versus-Hamilton difference are the same difference,
applied once each, so they cancel. This is the whole content of the result: ticket 13 was right that
the two Jacobians are built from opposite-handed perturbations, and right that the relation had to be
derived rather than assumed — the derivation happens to end in an identity.

**With the error map folded in.** The shipped Jacobian is
$G(q) = \partial q / \partial \delta\theta$ with $\delta\theta = 2\operatorname{vec}(\delta q)$, so
$a = \delta\theta/2$ and

$$G(q) \;=\; \tfrac{1}{2}\,\Xi(q), \qquad T\,G(q) \;=\; \tfrac{1}{2}\,\nabla_{\text{differential}}(\rho q).$$

The one half appears on both sides. This is why `test_quadrotor_sandwiched_dynamics_expansion_cross`
assembles its Julia block as `0.5 * ∇differential(q)` and compares against `errstate_jacobian`
without any further factor.

## Degenerate cases that must not be used as the test case

The rival hypothesis is that $\rho$ preserves order, that is, that the perturbation is right-handed
on both sides. It produces $wI - [v]_\times$ where the truth has $wI + [v]_\times$, so the two differ
by $2[v]_\times$ — which vanishes **iff $v = 0$**, the identity rotation and its antipode.

At the identity quaternion the shipped $\Xi$ and the rival agree exactly, to `0.0`. Any test case
with $v = 0$ therefore passes against a wrong implementation and is worthless as evidence. The same
goes for a quaternion with a single nonzero vector component tested against a perturbation along that
same axis, where the relevant column of $[v]_\times$ is empty.

The test case is a quaternion with all four components nonzero and distinct,
$q \propto (0.2, -0.5, 0.3, 0.78)$, where the rival is refuted by `1.006` — twelve orders of
magnitude above the `1e-12` the cross-test asserts at. Both the identity case and the informative
case are pinned as tests, the former labelled degenerate, so that nobody later "simplifies" the
informative one into the degenerate one.

## Verification

**Symbolic**, on free real symbols $(x, y, z, w)$ and a free perturbation $(a_1, a_2, a_3)$, with no
unit-norm assumption. All confirmed:

1. $\Xi(q)$ as shipped equals $\partial\big[\delta q(a) \otimes_{\text{JPL}} q\big] / \partial a$ at $a = 0$.
2. L3, and the failure of the order-preserving reading, with the residual $-2(v_a \times v_b)$.
3. $\nabla_{\text{differential}}$ as shipped by `Rotations.jl` equals
   $\partial\big[\rho q \otimes_{\text{Ham}} (\phi, 1)\big] / \partial\phi$ at $\phi = 0$, scalar-first.
4. The shipping statement $T\,\Xi(q) = \nabla_{\text{differential}}(\rho q)$.
5. The one half: $\partial q(\delta\theta) / \partial\delta\theta = \tfrac{1}{2}\Xi(q)$ under
   $\delta\theta = 2\operatorname{vec}(\delta q)$, taking the normalized $\delta q$ rather than the
   first-order one, so the square root is differentiated rather than assumed constant.

**Numerical**, in [test/unit/test_rotations.py](../test/unit/test_rotations.py), which runs without
Julia: the shipped `attitude_jacobian` against a central finite difference of the error-map inverse,
against the hardcoded `∇differential` values below, and against the rival, on the informative pair
and on the degenerate one.

**Against live `Rotations.jl`.** For $q = (0.2, -0.5, 0.3, 0.78)$ normalized,
`∇differential(QuatRotation(ρq))` returns

```text
[-0.20117019  0.50292548 -0.30175529
  0.78456374 -0.30175529 -0.50292548
  0.30175529  0.78456374 -0.20117019
  0.50292548  0.20117019  0.78456374]
```

and $T\,\Xi(q)$ reproduces it to `1.11e-16`, with $T\,G(q)$ against
$\tfrac{1}{2}\nabla_{\text{differential}}$ to `5.55e-17`. The rival differs by `1.006` on the same
quaternion. The finite-difference check of the one half agrees to `4.4e-11` at a step of `1e-6`,
which is the step's own truncation error.
