"""Replay a hand-guided demonstration on the arm, so Franka measures the task's force.

While a hand guides the arm, Franka's force estimate mixes the hand with the
contact. Replayed by the robot alone, the same motion gives the force of the
task by itself, and the replay is recorded as an ordinary episode whose
actions are the velocities the robot was commanded.

The replay follows the demonstration's measured joint path: the demo's own
velocity plus a pull back onto the path. Where the blade meets resistance the
arm is pushed off the path, the pull makes the command lead it, and the
controller presses until the blade is where the demonstration had it, up to
the push cap that Bamboo's contact limits enforce.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fr3_demo.kinematics import forward_kinematics, geometric_jacobian

# The replay records the velocities it commanded, as teleop does.
REPLAY_ACTION_SOURCE = "replay_joint_velocity_command"
# Pull back onto the demonstrated path, in 1/s of joint error.
PATH_GAIN = 3.0
# The recorded path is resampled onto this clock, and its velocity smoothed over this window.
PATH_RATE_HZ = 200.0
VELOCITY_WINDOW_S = 0.05
# After the demonstration's end, how close and how long the arm gets to settle on its last pose.
SETTLE_TOLERANCE_RAD = 0.005
SETTLE_TIMEOUT_S = 2.0
# A demo whose gripper opened or closed by more than this (of its full travel) cannot be replayed.
GRIPPER_MOTION = 0.05
# A depth offset eases in over the replay's first seconds and out over its last.
DEPTH_RAMP_S = 1.0
# How often along the path the offset's joint change is worked out; it is interpolated in between.
DEPTH_SAMPLE_S = 0.05


def contact_point_offset_joints(q: np.ndarray, offset_m: np.ndarray, contact_point_m: np.ndarray) -> np.ndarray:
    """The joint change that moves the contact point by ``offset_m`` (base frame) without turning the tool.

    ``contact_point_m`` is in the flange frame, as in [contact]. Damped least
    squares, so it stays bounded near a singularity.
    """

    pose = forward_kinematics(q)
    jacobian = geometric_jacobian(q)
    r = pose[:3, :3] @ np.asarray(contact_point_m, dtype=float)
    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    point = np.vstack((jacobian[:3] - skew @ jacobian[3:], jacobian[3:]))
    target = np.concatenate((np.asarray(offset_m, dtype=float), np.zeros(3)))
    return point.T @ np.linalg.solve(point @ point.T + 1e-4 * np.eye(6), target)


@dataclass(frozen=True)
class ReplayPath:
    """A demonstration's joint path on a uniform clock that starts at 0 s."""

    source: Path
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def start(self) -> np.ndarray:
        return self.positions[0].copy()

    def at(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        """The demonstrated position and velocity ``elapsed`` s in; still at either end."""

        if elapsed <= 0.0:
            return self.positions[0].copy(), np.zeros(7)
        if elapsed >= self.duration:
            return self.positions[-1].copy(), np.zeros(7)
        position = np.array([np.interp(elapsed, self.times, self.positions[:, joint]) for joint in range(7)])
        velocity = np.array([np.interp(elapsed, self.times, self.velocities[:, joint]) for joint in range(7)])
        return position, velocity

    def deepened(
        self, offset_m: np.ndarray, contact_point_m: np.ndarray, ramp_s: float = DEPTH_RAMP_S
    ) -> ReplayPath:
        """The same demonstration with the contact point moved by ``offset_m`` (base frame).

        The offset eases in from the start pose and out to the end pose, both
        of which stay where the demonstration had them, and the tool keeps its
        demonstrated orientation throughout.
        """

        offset = np.asarray(offset_m, dtype=float)
        if not np.any(offset) or len(self.times) < 2:
            return self
        duration = self.duration
        stride = max(1, round(DEPTH_SAMPLE_S * PATH_RATE_HZ))
        coarse = np.unique(np.append(np.arange(0, len(self.times), stride), len(self.times) - 1))
        changes = np.stack(
            [contact_point_offset_joints(self.positions[index], offset, contact_point_m) for index in coarse]
        )
        change = np.column_stack([np.interp(self.times, self.times[coarse], changes[:, joint]) for joint in range(7)])
        edge = np.minimum(self.times, duration - self.times)
        share = np.clip(edge / ramp_s, 0.0, 1.0) if ramp_s > 0 else np.ones_like(edge)
        share = 0.5 - 0.5 * np.cos(np.pi * share)  # eases in and out with no jump in velocity
        return ReplayPath.from_samples(self.source, self.times, self.positions + share[:, None] * change)

    @classmethod
    def from_episode(cls, episode: Path) -> ReplayPath:
        """Read a finished episode's joint path, preferring the full-rate arm samples."""

        with np.load(episode / "trajectory.npz") as trajectory:
            if "raw_force_joint_position" in trajectory.files and len(trajectory["raw_force_robot_timestamp"]) >= 2:
                stamps = np.asarray(trajectory["raw_force_robot_timestamp"], dtype=np.float64)
                positions = np.asarray(trajectory["raw_force_joint_position"], dtype=np.float64)
            else:
                stamps = np.asarray(trajectory["robot_timestamp"], dtype=np.float64)
                positions = np.asarray(trajectory["joint_position"], dtype=np.float64)
            # What the gripper did, not what it was told: closing on a tool it already holds moves nothing.
            gripper = (
                np.asarray(trajectory["gripper_position"], dtype=np.float64).ravel()
                if "gripper_position" in trajectory.files
                else np.zeros(1)
            )
        if gripper.size and float(np.ptp(gripper)) > GRIPPER_MOTION:
            raise ValueError(f"{episode.name} opens or closes the gripper; only arm motion can be replayed")
        if positions.ndim != 2 or positions.shape[1] != 7 or len(positions) != len(stamps):
            raise ValueError(f"{episode.name} has no usable joint path")
        if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(stamps))):
            raise ValueError(f"{episode.name} has non-finite joint samples")
        # Keep strictly increasing controller times (a repeated sample adds nothing).
        keep = np.concatenate(([True], np.diff(stamps) > 0.0))
        stamps, positions = stamps[keep], positions[keep]
        if len(stamps) < 2:
            raise ValueError(f"{episode.name} is too short to replay")
        return cls.from_samples(episode, stamps - stamps[0], positions)

    @classmethod
    def from_samples(cls, source: Path, times: np.ndarray, positions: np.ndarray) -> ReplayPath:
        step = 1.0 / PATH_RATE_HZ
        end = float(times[-1])
        # Uniform, and ending exactly at the end: the last step is between half and one and a half steps long.
        clock = np.append(np.arange(0.0, end - 0.5 * step, step), end)
        resampled = np.column_stack([np.interp(clock, times, positions[:, joint]) for joint in range(7)])
        if len(clock) < 2:
            return cls(source, clock, resampled, np.zeros_like(resampled))
        velocity = np.gradient(resampled, clock, axis=0)
        width = max(1, round(VELOCITY_WINDOW_S * PATH_RATE_HZ))
        if width > 1 and len(clock) > width:
            padded = np.pad(velocity, ((width // 2, width - 1 - width // 2), (0, 0)), mode="edge")
            kernel = np.ones(width) / width
            velocity = np.column_stack([np.convolve(padded[:, joint], kernel, mode="valid") for joint in range(7)])
        return cls(source, clock, resampled, velocity)


def replay_command(
    path: ReplayPath, elapsed: float, q: np.ndarray, max_speed: float, gain: float = PATH_GAIN
) -> np.ndarray:
    """The joint velocity that follows the demonstration from measured ``q``.

    Scaled as a whole, not per joint, when it would pass ``max_speed``, so
    the arm keeps the demonstrated direction while it catches up.
    """

    reference, velocity = path.at(elapsed)
    command = velocity + gain * (reference - np.asarray(q, dtype=float))
    peak = float(np.max(np.abs(command)))
    if peak > max_speed:
        command *= max_speed / peak
    return command


@dataclass(frozen=True)
class ReplayOutcome:
    completed: bool
    reason: str | None
    duration_s: float
    max_path_error_rad: float
    by_operator: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "stopped_because": self.reason,
            "stopped_by_operator": self.by_operator,
            "duration_s": round(self.duration_s, 3),
            "max_path_error_rad": round(self.max_path_error_rad, 4),
        }


def run_replay(
    robot: Any,
    path: ReplayPath,
    *,
    rate_hz: float,
    max_speed: float,
    max_path_error: float,
    gain: float = PATH_GAIN,
    gel_guard: Any | None = None,
    stop_reason: Callable[[], str | None] | None = None,
    on_command: Callable[[np.ndarray], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayOutcome:
    """Stream the demonstration from its start pose, which the arm must already be at.

    It ends at the demonstration's end once the arm has settled. It stops
    early when the operator asks (``stop_reason``), when the gel guard
    objects, when the arm is pushed further than ``max_path_error`` off the
    path (something is in the way), or when Bamboo ends the stream itself (its
    contact guard). A guard stop yields the stream, so the arm goes soft and
    the load falls, rather than freezing it in place.
    """

    period = 1.0 / rate_hz
    started = clock()
    worst = 0.0
    zero = np.zeros(7)

    def record(command: np.ndarray) -> None:
        if on_command is not None:
            on_command(command)

    def stopped(reason: str, *, yielding: bool, by_operator: bool = False) -> ReplayOutcome:
        record(zero)
        try:
            if yielding:
                robot.yield_stream(reason)
            else:
                robot.stream_velocity(zero)
        except RuntimeError:
            pass  # the stream has already ended; the reason we have is the one that matters
        return ReplayOutcome(False, reason, clock() - started, worst, by_operator)

    while True:
        tick = clock()
        elapsed = tick - started
        if stop_reason is not None:
            reason = stop_reason()
            if reason is not None:
                return stopped(reason, yielding=False, by_operator=True)
        if gel_guard is not None:
            reason = gel_guard.check()
            if reason is not None:
                return stopped(reason, yielding=True)
        q = np.asarray(robot.state()["qpos"], dtype=float)
        reference, _ = path.at(elapsed)
        error = float(np.max(np.abs(reference - q)))
        worst = max(worst, error)
        if error > max_path_error:
            return stopped(
                f"the arm was pushed {error:.3f} rad off the demonstration's path (limit {max_path_error:.3f}): "
                "something is in the way",
                yielding=True,
            )
        if elapsed >= path.duration and (error <= SETTLE_TOLERANCE_RAD or elapsed >= path.duration + SETTLE_TIMEOUT_S):
            record(zero)
            robot.stream_velocity(zero)
            return ReplayOutcome(True, None, elapsed, worst)
        command = replay_command(path, elapsed, q, max_speed, gain)
        record(command)
        try:
            robot.stream_velocity(command)
        except RuntimeError as error:
            # Bamboo ended the stream: its contact guard tripped, or the arm faulted.
            record(zero)
            return ReplayOutcome(False, str(error), clock() - started, worst)
        remaining = period - (clock() - tick)
        if remaining > 0:
            sleep(remaining)


def link_replay(demo: Path, replay: Path, outcome: ReplayOutcome, settings: dict[str, Any]) -> None:
    """Note in both episodes' metadata which replay belongs to which demonstration, and how it went."""

    import json

    from fr3_demo.recording import update_episode_metadata

    update_episode_metadata(replay, replay={**outcome.summary(), **settings})
    earlier = json.loads((demo / "metadata.json").read_text(encoding="utf-8")).get("replays", [])
    update_episode_metadata(demo, replays=[*earlier, {"episode": replay.name, "completed": outcome.completed}])
