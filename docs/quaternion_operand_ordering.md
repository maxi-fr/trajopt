# Quaternion operand ordering: JPL error vs. Hamilton error

This resolves the open question in [section 7 of the specification](TRAJECTORY_OPTIMIZATION_SPEC.md#7-rotations-jpl-quaternions-and-the-error-state):
how the reference implementation's attitude error relates to the one `RobotDynamics.state_diff`
builds, so that the rotations cross-tests can assert a stated relation rather than a fitted one.

**Result.** The two are a similarity transform of each other. Their vector parts are the same
relative rotation resolved in two frames related by $R(q_{\text{ref}})$:

$$\operatorname{vec}(\delta q_{\text{Julia}}) \;=\; -\,R(q_{\text{ref}})^{T}\,\operatorname{vec}(q_{\text{err}})$$

with equal scalar parts, where $q_{\text{err}} = q \otimes_{\text{JPL}} q_{\text{ref}}^{-1}$ is
`Quaternion.error_to` and $R(q_{\text{ref}})$ is `Quaternion.to_rot_mat`. **No convention change
is required** — the relation is exact, closed-form, and involves only quantities both sides
already compute.

Verified symbolically for free symbols (no unit-norm or example fitting) and numerically in
[test/unit/test_quaternion_ordering.py](../test/unit/test_quaternion_ordering.py), which agrees
with a live `Rotations.jl` run to `1.1e-16`.

## Notation

Both conventions are written scalar-last, $q = (v, w)$. Julia stores scalar-first; that is a
reindexing, not part of the algebra, and is applied at the boundary.

| | Product $a \otimes b$ |
| :--- | :--- |
| JPL | $\big(w_b v_a + w_a v_b - v_a \times v_b,\;\; w_a w_b - v_a \cdot v_b\big)$ |
| Hamilton | $\big(w_a v_b + w_b v_a + v_a \times v_b,\;\; w_a w_b - v_a \cdot v_b\big)$ |

Conjugate is $\bar q = (-v, w)$ in both, and equals the inverse on the unit sphere. The bridge is

$$B(q) = (-v,\, w),$$

which is `to_hamilton` in the specification and the conjugation inside the reference's `to_scipy`.

## Lemmas

**L1 — order reversal.** $\text{hamilton}(a,b) = \text{jpl}(b,a)$. Immediate from the table: the
scalar parts are identical and $v_b \times v_a = -\,v_a \times v_b$ absorbs the sign difference.

**L2 — the bridge is a product isomorphism $(\text{JPL}) \to (\text{Hamilton})$.**

$$B(a \otimes_{\text{JPL}} b) = B(a) \otimes_{\text{Ham}} B(b)$$

*Proof.* Scalar: $(-v_a)\cdot(-v_b) = v_a \cdot v_b$, so the scalar part is unchanged. Vector:

$$w_a(-v_b) + w_b(-v_a) + (-v_a)\times(-v_b) = -\big(w_b v_a + w_a v_b - v_a \times v_b\big),$$

which is the negated JPL vector part, i.e. $B$ applied to it. $\square$

$B$ is an involution, so it is an isomorphism, not merely a homomorphism. It also commutes with
conjugation, $B(\bar q) = \overline{B(q)}$, and preserves the rotation matrix,
$R_{\text{JPL}}(q) = R_{\text{Ham}}(B(q))$ — the latter is exactly how the reference's
`to_rot_mat` is built. Combining L2 with that gives $R_{\text{JPL}}$ its own homomorphism
property, $R_{\text{JPL}}(a \otimes_{\text{JPL}} b) = R_{\text{JPL}}(a)\,R_{\text{JPL}}(b)$,
which is used below.

## The two error definitions

Write $p = B(q)$ and $p_{\text{ref}} = B(q_{\text{ref}})$ for the Hamilton images.

**Python, bridged.** By L2 and $B(\bar q) = \overline{B(q)}$,

$$A \;:=\; B\big(q \otimes_{\text{JPL}} q_{\text{ref}}^{-1}\big) \;=\; p \otimes_{\text{Ham}} p_{\text{ref}}^{-1}.$$

**Julia.** `Rotations.rotation_error(R1, R2, map)` returns `map⁻¹(R2 \ R1)`
(`Rotations/src/rotation_error.jl:43`, confirmed by running it), and `_state_diff_expr` calls it with
`R1 = q`, `R2 = q0`. So

$$C \;:=\; p_{\text{ref}}^{-1} \otimes_{\text{Ham}} p.$$

### Component form: not a global sign

With $x = \operatorname{vec}(p)$, $y = \operatorname{vec}(p_{\text{ref}})$, and $s = w_{\text{ref}} x - w\, y$, $c = x \times y$:

$$\operatorname{scalar}(A) = \operatorname{scalar}(C) = w\,w_{\text{ref}} + x \cdot y, \qquad \operatorname{vec}(A) = s - c, \qquad \operatorname{vec}(C) = s + c.$$

Hence $A - C = -2\,(x \times y)$ in the vector part and zero in the scalar part. This is the
concrete statement behind the specification's warning: the two differ in one term only, so they
are neither equal nor related by a global sign whenever $s \ne 0$ and $c \ne 0$.

### Closed form: a similarity transform

$$C = p_{\text{ref}}^{-1} \otimes A \otimes p_{\text{ref}}, \qquad\text{equivalently}\qquad A = p \otimes C \otimes p^{-1}.$$

*Proof.* $p_{\text{ref}}^{-1} \otimes \big(p \otimes p_{\text{ref}}^{-1}\big) \otimes p_{\text{ref}} = p_{\text{ref}}^{-1} \otimes p = C$, using associativity only. The second form follows from
$p \otimes C \otimes p^{-1} = p \otimes p_{\text{ref}}^{-1} \otimes p \otimes p^{-1} = A$. $\square$

Multiplied through by $\lVert p_{\text{ref}} \rVert^2$ this is a polynomial identity in eight free
symbols, holding for arbitrary quaternions; the unit-norm case is what ships.

### Matrix form: the shipping statement

Conjugation by a unit quaternion fixes the scalar part and rotates the vector part, so

$$\operatorname{scalar}(C) = \operatorname{scalar}(A), \qquad \operatorname{vec}(C) = R_{\text{Ham}}(p_{\text{ref}})^{T}\,\operatorname{vec}(A).$$

Since $R_{\text{Ham}}(p_{\text{ref}}) = R_{\text{JPL}}(q_{\text{ref}})$ is the reference's
`q_ref.to_rot_mat()`, and $\operatorname{vec}(A) = -\operatorname{vec}(q_{\text{err}})$ by the
definition of $B$, the relation restated entirely in JPL quantities is

$$\boxed{\;\operatorname{vec}(\delta q_{\text{Julia}}) = -\,R(q_{\text{ref}})^{T}\operatorname{vec}(q_{\text{err}}), \qquad \operatorname{scalar}(\delta q_{\text{Julia}}) = \operatorname{scalar}(q_{\text{err}})\;}$$

Geometrically: the same relative rotation, resolved in two different frames. This is the
left-versus-right multiplicative error distinction, which $R(q_{\text{ref}})$ transports between.

### It passes through the error map

Both orderings have equal scalar parts, and equal vector-part norms because a rotation preserves
length. Every error map in play — $\delta\theta = 2v$, the Cayley map $v/w$, and the exponential
map — rescales the vector part by a factor depending only on $w$ and $\lVert v \rVert$. Both
orderings therefore take the *same* factor, and the relation survives the map unchanged:

$$\delta\theta_{\text{Julia}} = -\,R(q_{\text{ref}})^{T}\,\delta\theta_{\text{Python}}$$

for whichever of those maps the cross-test selects. Nothing about the choice of map needs
re-deriving.

## Degenerate cases that must not be used as the test case

The two definitions differ only in the sign of $x \times y$, so any pair with parallel vector
parts makes them **identical**. Two rotations about a common axis, and any pair with
$q_{\text{ref}} = \text{identity}$ (where $R_{\text{ref}} = I$ and the relation collapses to a
plain vector negation), are both in this class.

Such a pair passes the cross-test against a reversed implementation and is therefore worthless as
evidence. Both cases are pinned as explicit tests, labelled as degenerate, so that nobody later
"simplifies" the informative test case into one of them.

The test case is a pure-$x$ rotation against a pure-$y$ rotation, where $x \perp y$ and the
cross-product term is at its largest.

## Verification

**Symbolic**, on free real symbols $(v_1,v_2,v_3,w,r_1,r_2,r_3,w_{\text{ref}})$ with no
unit-norm assumption except where stated, all confirmed: L1, L2, $B(\bar q) = \overline{B(q)}$,
$B(q_{\text{err}}) = A$, equal scalar parts, the $s \mp c$ component form, the norm-free
similarity identity in both directions, the matrix form, the JPL restatement, and
$A - C = -2\,(v \times v_{\text{ref}})$.

**Numerical**, in [test/unit/test_quaternion_ordering.py](../test/unit/test_quaternion_ordering.py),
staged so that a bridge error cannot mask a kernel error:

1. An independently written Hamilton product is checked against the algebra's defining relations
   ($ij = k$, $ji = -k$, $i^2 = -1$) and its matrix formula against a hardcoded $90°$ rotation.
2. The bridge is checked against hardcoded Hamilton values and matrices only. **No conjugated
   comparison exists in the file before this point.**
3. Only then: the isomorphism property, the similarity transform, the matrix form, and the
   error-map invariance — each over the $x/y$ pair plus 200 random pairs.
4. The two degenerate cases, asserted to coincide and labelled uninformative.

**Against live Julia.** Feeding the bridged $x/y$ pair to `Rotations.jl` and evaluating
`q0 \ q` gives, scalar-first,

```text
[0.7168313496334096, -0.34626901463318716, -0.5449383454282484, -0.26323522820783224]
```

and the derived $-R(q_{\text{ref}})^{T}\operatorname{vec}(q_{\text{err}})$ reproduces the vector
part to `1.1e-16`. On this pair only the $z$ component distinguishes the orderings, by
$0.5265$ — a difference far above any cross-test tolerance, which is what makes the pair a real
test.
