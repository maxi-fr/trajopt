from collections.abc import Sequence
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation


class Quaternion(eqx.Module):
    """Unit quaternion representing 3D rotations in JPL convention (scalar-last)."""

    vec: jax.Array
    scalar: jax.Array

    def __init__(self, vec: Sequence[float] | jax.Array | np.ndarray, scalar: float | jax.Array | np.ndarray) -> None:
        """Initialize JPL quaternion with 3D vector part and scalar part."""
        self.vec = jnp.asarray(vec, dtype=float)
        self.scalar = jnp.asarray(scalar, dtype=float).squeeze()

    @classmethod
    def from_array(cls, q: Sequence[float] | jax.Array | np.ndarray, *, scalar_first: bool = False) -> Self:
        """Create a Quaternion from a 4-element array of shape (4,)."""
        q_arr = jnp.asarray(q, dtype=float)
        if scalar_first:
            return cls(q_arr[1:], q_arr[0])
        return cls(q_arr[:3], q_arr[3])

    def to_array(self, *, scalar_first: bool = False) -> jax.Array:
        """Convert quaternion to a 4-element array of shape (4,)."""
        if scalar_first:
            return jnp.array([self.scalar, self.vec[0], self.vec[1], self.vec[2]])
        return jnp.array([self.vec[0], self.vec[1], self.vec[2], self.scalar])

    @classmethod
    def from_scipy(cls, rot: Rotation, *, canonical: bool = True) -> Self:
        """Create a Quaternion from a SciPy Rotation object."""
        q = rot.as_quat(canonical=canonical, scalar_first=False)
        return cls(-q[:3], q[3])

    def to_scipy(self) -> Rotation:
        """Convert quaternion to a SciPy Rotation object."""
        h = to_hamilton(self)
        return Rotation.from_quat(np.asarray(h))

    def to_rot_mat(self) -> jax.Array:
        """Convert quaternion to a direct 3x3 passive rotation matrix of shape (3, 3)."""
        x, y, z = self.vec[0], self.vec[1], self.vec[2]
        w = self.scalar
        return jnp.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
                [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
                [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Multiply two quaternions according to the JPL product convention."""
        qv = self.vec
        w = self.scalar
        qv_other = other.vec
        w_other = other.scalar

        w_ret = w * w_other - jnp.dot(qv, qv_other)
        v_ret = w_other * qv + w * qv_other - jnp.cross(qv, qv_other)
        return Quaternion(v_ret, w_ret)

    def conjugate(self) -> "Quaternion":
        """Calculate conjugate quaternion [-v, w]."""
        return Quaternion(-self.vec, self.scalar)

    def inverse(self) -> "Quaternion":
        """Calculate inverse quaternion for a unit quaternion [-v, w]."""
        return self.conjugate()

    def error_to(self, reference: "Quaternion") -> "Quaternion":
        """Attitude error quaternion relative to reference q_err = q (x) q_ref^-1."""
        return self * reference.conjugate()

    def apply(self, v: Sequence[float] | jax.Array | np.ndarray) -> jax.Array:
        """Rotate vector of shape (3,) using passive Frame rotation."""
        v_arr = jnp.asarray(v, dtype=float)
        qv = self.vec
        w = self.scalar
        t = -2.0 * jnp.cross(qv, v_arr)
        return v_arr + w * t - jnp.cross(qv, t)

    def xi(self, *, scalar_first: bool = False) -> jax.Array:
        """Compute the 4x3 kinematics matrix Xi(q) of shape (4, 3)."""
        qw = self.scalar
        qx, qy, qz = self.vec[0], self.vec[1], self.vec[2]
        res = jnp.array(
            [
                [qw, -qz, qy],
                [qz, qw, -qx],
                [-qy, qx, qw],
                [-qx, -qy, -qz],
            ]
        )
        if scalar_first:
            return jnp.roll(res, 1, axis=0)
        return res

    def kinematics(self, omega: Sequence[float] | jax.Array | np.ndarray, *, scalar_first: bool = False) -> jax.Array:
        """Calculate quaternion time derivative dq/dt = 0.5 * Xi(q) @ omega of shape (4,)."""
        omega_arr = jnp.asarray(omega, dtype=float)
        return 0.5 * (self.xi(scalar_first=scalar_first) @ omega_arr)

    def __neg__(self) -> "Quaternion":
        """Negate quaternion [-v, -w]."""
        return Quaternion(-self.vec, -self.scalar)


def to_hamilton(q: Quaternion) -> jax.Array:
    """Convert JPL Quaternion to Hamilton quaternion array of shape (4,)."""
    return jnp.array([-q.vec[0], -q.vec[1], -q.vec[2], q.scalar])


def from_hamilton(h: Sequence[float] | jax.Array | np.ndarray) -> Quaternion:
    """Convert Hamilton quaternion array of shape (4,) to JPL Quaternion."""
    arr = jnp.asarray(h, dtype=float)
    return Quaternion(-arr[:3], arr[3])


def error_map(q: Quaternion, q_ref: Quaternion) -> jax.Array:
    """Compute multiplicative small-angle attitude error 2 * vec(q (x) q_ref^-1) of shape (3,)."""
    return 2.0 * q.error_to(q_ref).vec


def attitude_jacobian(q: Quaternion) -> jax.Array:
    """Compute attitude Jacobian rotation block 0.5 * Xi(q) of shape (4, 3)."""
    return 0.5 * q.xi()
