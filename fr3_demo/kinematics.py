"""Dependency-light forward and differential kinematics for the Franka FR3.

The fixed transforms and limits below are from the official ``franka_description``
FR3 model.  The Jacobian is expressed in the robot base frame and ordered as
``[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

JOINT_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
JOINT_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])

# URDF joint-origin transforms: xyz and fixed RPY, followed by rotation about z.
_XYZ = np.array(
    [
        [0.0, 0.0, 0.333],
        [0.0, 0.0, 0.0],
        [0.0, -0.316, 0.0],
        [0.0825, 0.0, 0.0],
        [-0.0825, 0.384, 0.0],
        [0.0, 0.0, 0.0],
        [0.088, 0.0, 0.0],
    ],
    dtype=float,
)
_ROLL = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2])
_LINK8_Z = 0.107


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned EEF bounds in the robot base frame, in metres."""

    minimum: tuple[float, float, float] = (0.10, -0.60, 0.05)
    maximum: tuple[float, float, float] = (0.80, 0.60, 1.00)

    def contains(self, position: FloatArray, tolerance: float = 1e-9) -> bool:
        lower = np.asarray(self.minimum) - tolerance
        upper = np.asarray(self.maximum) + tolerance
        return bool(np.all(position >= lower) and np.all(position <= upper))


def _transform(rotation: FloatArray | None = None, translation: FloatArray | None = None) -> FloatArray:
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def _rotation_x(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _validate_joints(q: FloatArray) -> FloatArray:
    joints = np.asarray(q, dtype=float)
    if joints.shape != (7,):
        raise ValueError(f"Expected seven joint positions, got shape {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("Joint positions must all be finite")
    return joints


def _chain(q: FloatArray) -> tuple[FloatArray, list[FloatArray], list[FloatArray]]:
    joints = _validate_joints(q)
    pose = np.eye(4)
    origins: list[FloatArray] = []
    axes: list[FloatArray] = []

    for index, angle in enumerate(joints):
        pose = pose @ _transform(_rotation_x(float(_ROLL[index])), _XYZ[index])
        origins.append(pose[:3, 3].copy())
        axes.append(pose[:3, 2].copy())
        pose = pose @ _transform(_rotation_z(float(angle)))

    pose = pose @ _transform(translation=np.array([0.0, 0.0, _LINK8_Z]))
    return pose, origins, axes


def forward_kinematics(q: FloatArray) -> FloatArray:
    """Return the base-to-EEF homogeneous transform for seven joint angles."""

    return _chain(q)[0]


def link_positions(q: FloatArray) -> FloatArray:
    """Return base, seven joint origins, and EEF positions for visualization."""

    end_pose, origins, _ = _chain(q)
    return np.vstack([np.zeros(3), *origins, end_pose[:3, 3]])


def geometric_jacobian(q: FloatArray) -> FloatArray:
    """Return the 6x7 geometric Jacobian in the robot base frame."""

    end_pose, origins, axes = _chain(q)
    end_position = end_pose[:3, 3]
    jacobian = np.empty((6, 7), dtype=float)
    for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
        jacobian[:3, index] = np.cross(axis, end_position - origin)
        jacobian[3:, index] = axis
    return jacobian


def damped_pseudoinverse(jacobian: FloatArray, damping: float) -> FloatArray:
    """Compute a numerically stable right pseudoinverse for a 6x7 Jacobian."""

    if jacobian.shape != (6, 7):
        raise ValueError(f"Expected a 6x7 Jacobian, got {jacobian.shape}")
    if damping <= 0:
        raise ValueError("Damping must be positive")
    regularized = jacobian @ jacobian.T + (damping**2) * np.eye(6)
    return jacobian.T @ np.linalg.solve(regularized, np.eye(6))


def limit_joint_velocity(velocity: FloatArray, maximum: float) -> FloatArray:
    """Uniformly scale a joint velocity vector to preserve its direction."""

    peak = float(np.max(np.abs(velocity)))
    if peak <= maximum:
        return velocity
    return velocity * (maximum / peak)


def resolved_rate_step(
    q: FloatArray,
    twist: FloatArray,
    duration: float,
    *,
    damping: float = 0.08,
    max_joint_velocity: float = 0.35,
    joint_margin: float = 0.08,
    nullspace_gain: float = 0.08,
    workspace: WorkspaceBounds | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Convert a Cartesian base-frame twist into one safe joint waypoint.

    Returns ``(q_target, q_velocity)``. Joint speed is uniformly limited, a
    low-gain centring term operates in the Jacobian nullspace, and the step is
    uniformly shortened before a joint margin would be crossed.
    """

    joints = _validate_joints(q)
    requested_twist = np.asarray(twist, dtype=float)
    if requested_twist.shape != (6,) or not np.all(np.isfinite(requested_twist)):
        raise ValueError("Twist must contain six finite values")
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if joint_margin < 0 or np.any(JOINT_LOWER + joint_margin >= JOINT_UPPER - joint_margin):
        raise ValueError("Invalid joint margin")

    jacobian = geometric_jacobian(joints)
    inverse = damped_pseudoinverse(jacobian, damping)
    joint_velocity = inverse @ requested_twist

    # Keep redundant motion away from joint limits without perturbing the task.
    centre = (JOINT_LOWER + JOINT_UPPER) / 2.0
    half_range = (JOINT_UPPER - JOINT_LOWER) / 2.0
    centring = nullspace_gain * (centre - joints) / half_range
    joint_velocity += (np.eye(7) - inverse @ jacobian) @ centring
    joint_velocity = limit_joint_velocity(joint_velocity, max_joint_velocity)

    lower = JOINT_LOWER + joint_margin
    upper = JOINT_UPPER - joint_margin
    step = joint_velocity * duration
    scale = 1.0
    for index, delta in enumerate(step):
        if delta > 0:
            scale = min(scale, float((upper[index] - joints[index]) / delta))
        elif delta < 0:
            scale = min(scale, float((lower[index] - joints[index]) / delta))
    scale = float(np.clip(scale, 0.0, 1.0))
    joint_velocity *= scale
    target = joints + joint_velocity * duration

    if workspace is not None and not workspace.contains(forward_kinematics(target)[:3, 3]):
        raise ValueError("Requested step would leave the configured EEF workspace")
    return target, joint_velocity
