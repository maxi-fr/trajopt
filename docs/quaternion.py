# ruff: noqa: FBT001, FBT002
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]
Vec3 = Annotated[FloatArray, Literal[3]]
Vec4 = Annotated[FloatArray, Literal[4]]
Mat3x3 = Annotated[FloatArray, Literal[3]]
Mat4x3 = Annotated[FloatArray, Literal[4, 3]]


@dataclass(frozen=True)
class Quaternion:
    """A unit quaternion representing 3D rotations, following the JPL convention."""

    vec: Vec3
    scalar: float

    def __post_init__(self) -> None:
        """Validate the quaternion vector."""
        object.__setattr__(self, "vec", np.asarray(self.vec))

    @classmethod
    def from_array(cls, q: ArrayLike, scalar_first: bool = False) -> "Quaternion":
        """
        Create a Quaternion from an array-like object.

        Parameters
        ----------
        q : ArrayLike
            Input array representing the quaternion. Expected shape is (4,).
        scalar_first : bool, optional
            If True, the input array is interpreted as [scalar, v1, v2, v3].
            If False, it is interpreted as [v1, v2, v3, scalar].
            Default is False.

        Returns
        -------
        Quaternion
            The created Quaternion object.
        """
        q = np.asarray(q)
        if scalar_first:
            return cls(q[1:], q[0])
        return cls(q[:3], q[3])

    def to_array(self, scalar_first: bool = False) -> Vec4:
        """
        Convert the Quaternion to a numpy array.

        Parameters
        ----------
        scalar_first : bool, optional
            If True, returns [scalar, v1, v2, v3].
            If False, returns [v1, v2, v3, scalar].
            Default is False.

        Returns
        -------
        np.ndarray
            The quaternion as a numpy array.
        """
        if scalar_first:
            return np.array((self.scalar, *self.vec))

        return np.array((*self.vec, self.scalar))

    @classmethod
    def from_scipy(cls, rot: Rotation, canonical: bool = True) -> "Quaternion":
        """
        Create a Quaternion from a scipy.spatial.transform.Rotation object.

        Note: Scipy uses the Hamilton convention for quaternions. This class uses the JPL convention.
        The conversion handles the difference (JPL q = [-v, w] relative to Hamilton q = [v, w] for the same rotation).

        Parameters
        ----------
        rot : scipy.spatial.transform.Rotation
            The rotation object.
        canonical : bool, optional
            Whether to map the quaternion to the canonical hemisphere (w > 0). Default is True.

        Returns
        -------
        Quaternion
            The corresponding JPL quaternion.
        """
        q = rot.as_quat(canonical=canonical, scalar_first=False)
        return cls(-q[:3], q[3])

    def to_scipy(self) -> Rotation:
        """
        Convert the Quaternion to a scipy.spatial.transform.Rotation object.

        Note: Handles the conversion from JPL convention to Hamilton convention used by Scipy.

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation object.
        """
        return Rotation.from_quat(
            self.conjugate().to_array(scalar_first=False)
        )  # conjugate because scipy uses hamilton convention

    def to_rot_mat(self) -> Mat3x3:
        """
        Convert the quaternion to a rotation matrix.

        Returns
        -------
        np.ndarray
            The 3x3 rotation matrix.
        """
        return self.to_scipy().as_matrix()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        r"""
        Multiplies two quaternions according to the JPL convention.

        Note: This is NOT the Hamilton product.

        Formula:
        .. math::
            \begin{bmatrix}
            q_4 \overline{\mathbf{q}}_{1: 3}+\bar{q}_4 \mathbf{q}_{1: 3}-\overline{\mathbf{q}}_{1: 3} \times \mathbf{q}_{1: 3} \\
            \bar{q}_4 q_4-\overline{\mathbf{q}}_{1: 3} \cdot \mathbf{q}_{1: 3}
            \end{bmatrix}

        Parameters
        ----------
        other : Quaternion
            The right-hand side quaternion.

        Returns
        -------
        Quaternion
            The product of the two quaternions.
        """
        qv = self.vec
        w = self.scalar

        qv_other = other.vec
        w_other = other.scalar

        w_ret = w * w_other - np.dot(qv, qv_other)
        v_ret = w_other * qv + w * qv_other - np.cross(qv, qv_other)

        return Quaternion(v_ret, w_ret)

    def mult_jpl(self, rhs: "Quaternion") -> "Quaternion":
        """
        Perform JPL quaternion multiplication (same as * operator).

        Parameters
        ----------
        rhs : Quaternion
            The right-hand side quaternion.

        Returns
        -------
        Quaternion
            The result of self * rhs.
        """
        return self * rhs

    def mult_hamilton(self, rhs: "Quaternion") -> "Quaternion":
        """
        Perform Hamilton quaternion multiplication.

        Parameters
        ----------
        rhs : Quaternion
            The right-hand side quaternion.

        Returns
        -------
        Quaternion
            The Hamilton product of self and rhs.
        """
        return rhs * self

    def conjugate(self) -> "Quaternion":
        """
        Calculate the conjugate of the quaternion.

        For a unit quaternion, this represents the inverse rotation.

        Returns
        -------
        Quaternion
            The conjugate quaternion [-v, w].
        """
        return Quaternion(-self.vec, self.scalar)

    def inverse(self) -> "Quaternion":
        """
        Return the inverse rotation.

        For a unit quaternion the inverse equals the conjugate.

        Returns
        -------
        Quaternion
            The inverse quaternion [-v, w].
        """
        return self.conjugate()

    def error_to(self, reference: "Quaternion") -> "Quaternion":
        r"""
        Attitude error of this quaternion relative to a reference.

        Returns the error quaternion :math:`q_{err} = q \otimes q_{ref}^{-1}`. For
        ``self == reference`` the result is the identity quaternion ``([0, 0, 0], 1)``;
        for a small rotation the vector part is approximately half the rotation vector
        that takes ``reference`` onto ``self``.

        This ordering is the one consistent with the ``q_dot = 0.5 * Xi(q) @ omega``
        kinematics (:meth:`kinematics`): that derivative corresponds to a body-frame
        left perturbation ``q+ = dq (x) q``, so the body-frame error is
        ``dq = q (x) q_ref^-1``.

        Parameters
        ----------
        reference : Quaternion
            The reference (desired) attitude.

        Returns
        -------
        Quaternion
            The error quaternion.
        """
        return self * reference.conjugate()

    def apply(self, v: ArrayLike) -> Vec3:
        """
        Rotates a vector v using this quaternion.

        Parameters
        ----------
        v : ArrayLike
            The 3D vector to rotate.

        Returns
        -------
        np.ndarray
            The rotated vector.
        """
        v = np.asarray(v)

        qv = self.vec
        w = self.scalar

        t = -2.0 * np.cross(qv, v)

        return v + w * t - np.cross(qv, t)

    def kinematics(self, omega: Vec3, scalar_first: bool = False) -> Vec4:
        """
        Calculate the time derivative of the quaternion given an angular velocity.

        dq/dt = 0.5 * Xi(q) * omega

        Parameters
        ----------
        omega : np.ndarray
            Angular velocity vector in the body frame [rad/s].
        scalar_first : bool, optional
            If True, returns the derivative as [dw, dx, dy, dz].
            If False, returns [dx, dy, dz, dw].
            Default is False.

        Returns
        -------
        np.ndarray
            The time derivative of the quaternion.
        """
        return 0.5 * self.xi(scalar_first=scalar_first) @ omega

    def exact_integration(self, omega: Vec3, dt: float) -> "Quaternion":
        """
        Integrates the quaternion given a constant angular velocity over a time step.

        Parameters
        ----------
        omega : np.ndarray
            Constant angular velocity vector in the body frame [rad/s].
        dt : float
             Time step [s].

        Returns
        -------
        Quaternion
            The new quaternion after integration.
        """
        omega = np.asarray(omega) * dt

        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-9:  # Avoid division by zero for very small angular velocities  # noqa: PLR2004
            return self

        delta_theta = omega_norm
        delta_q_vec = (omega / omega_norm) * np.sin(delta_theta / 2)
        delta_q_scalar = np.cos(delta_theta / 2)
        delta_q = Quaternion(delta_q_vec, delta_q_scalar)

        return self * delta_q

    def xi(self, scalar_first: bool = False) -> Mat4x3:
        r"""
        Compute the xi matrix for JPL quaternion kinematics.

        The kinematics equation is:
        .. math::
            \dot{q} = \frac{1}{2} \Xi(\mathbf{q}) \omega

        where:
        .. math::
            \Xi(\mathbf{q}) :=
            \begin{bmatrix}
                q_4 I_3+\left[\mathbf{q}_{1: 3} \times\right] \\
                -\mathbf{q}_{1: 3}^T
            \end{bmatrix} =
            \begin{bmatrix}
                 q_4 & -q_3 &  q_2 \\
                 q_3 &  q_4 & -q_1 \\
                -q_2 &  q_1 &  q_4 \\
                -q_1 & -q_2 & -q_3
            \end{bmatrix}

        Parameters
        ----------
        scalar_first : bool, optional
            If True, the matrix is organized for [scalar, v1, v2, v3] layout.
            If False, the matrix is organized for [v1, v2, v3, scalar] layout.
            Default is False.

        Returns
        -------
        np.ndarray
            The 4x3 xi matrix.
        """
        qw = self.scalar
        qx, qy, qz = self.vec

        res = np.array([[qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw], [-qx, -qy, -qz]])
        if scalar_first:
            return np.roll(res, 1, axis=0)
        return res
