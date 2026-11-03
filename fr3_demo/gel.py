"""GelSight gel guard: marker shift -> the gel's load, its direction and a level.

The marker tracker (tactile_shear) measures how far the gel surface has moved
since an unloaded reference. A calibration fitted by fr3-gel-calibrate maps
that shift to the force the arm puts through the gel, in newtons and in the
end-effector frame, robot on environment, as Franka reports its own estimate.
That force's direction is the way the arm is pushing, so arm motion along it
loads the gel further. The level is the force over the configured limit (or
the twist over its limit, whichever is larger): guide mode stiffens against
further loading as it rises, and streams yield at level 1.

The gel is pressed whenever the gripper closes, so the reference is retaken
(re-zeroed) after each gripper command, once the image has settled. Everything
fails closed: no reference, lost tracking, stale frames or a processing error
all give an invalid reading, never a small number.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fr3_demo.tactile import TactileFrame
    from fr3_demo.tactile_shear import MarkerShearTracker, ShearEstimate


@dataclass(frozen=True)
class GelCalibration:
    """Maps tracker output to the force through the gel (N, EE frame, robot on environment)."""

    shear_to_force: np.ndarray  # (3, 2): per px of marker shift along image x and y
    spread_to_force: np.ndarray  # (3,): per unit of spread (pressing into the gel)
    limit_n: float  # estimated load at level 1
    twist_limit_rad: float  # twist of the marker field at level 1

    def __post_init__(self) -> None:
        shear = np.asarray(self.shear_to_force, dtype=float)
        spread = np.asarray(self.spread_to_force, dtype=float)
        if (
            shear.shape != (3, 2)
            or spread.shape != (3,)
            or not (np.all(np.isfinite(shear)) and np.all(np.isfinite(spread)))
        ):
            raise ValueError("gel calibration needs a finite 3x2 shear_to_force and a 3-vector spread_to_force")
        if not np.any(shear):
            raise ValueError("gel calibration maps every shift to zero force")
        if not (self.limit_n > 0 and self.twist_limit_rad > 0):
            raise ValueError("gel limit_n and twist_limit_rad must be positive")
        object.__setattr__(self, "shear_to_force", shear)
        object.__setattr__(self, "spread_to_force", spread)

    @classmethod
    def from_config(cls, gel: dict[str, Any]) -> GelCalibration:
        if "shear_to_force" not in gel:
            raise ValueError(
                "[gel] has no calibration: run fr3-gel-calibrate with the marker gel and tool fitted, "
                "then copy what it prints into config.toml"
            )
        return cls(
            shear_to_force=np.asarray(gel["shear_to_force"], dtype=float),
            spread_to_force=np.asarray(gel.get("spread_to_force", [0.0, 0.0, 0.0]), dtype=float),
            limit_n=float(gel.get("limit_n", 10.0)),
            twist_limit_rad=float(gel.get("twist_limit_rad", 0.1)),
        )

    def force(self, estimate: ShearEstimate) -> np.ndarray:
        shear = np.asarray(estimate.shear_px, dtype=float)
        return self.shear_to_force @ shear + self.spread_to_force * float(estimate.spread)

    def level(self, estimate: ShearEstimate) -> tuple[float, np.ndarray]:
        force = self.force(estimate)
        level = max(float(np.linalg.norm(force)) / self.limit_n, abs(float(estimate.twist_rad)) / self.twist_limit_rad)
        return level, force


@dataclass(frozen=True)
class GelReading:
    """The latest gel state, as guide mode and the stream guards consume it."""

    seq: int  # increases with every processed frame
    timestamp: float  # host monotonic time of the frame
    valid: bool
    reason: str  # why it is invalid, or "" when valid
    level: float
    force_ee: tuple[float, float, float]
    direction_ee: tuple[float, float, float] | None  # unit; arm motion along it loads the gel more
    zeroing: bool = False


def _median_motion(previous: np.ndarray | None, current: np.ndarray) -> float:
    """Median distance from each marker to the nearest marker of the previous frame."""

    if previous is None or len(previous) < 4 or len(current) < 4:
        return float("inf")
    distances = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=2).min(axis=1)
    return float(np.median(distances))


class GelMonitor:
    """Runs the marker tracker on every GelSight frame and keeps the latest reading.

    ``on_frame`` is a GelSightCamera observer, so it runs on the capture thread;
    it never raises, because a gel problem must not stop capture or recording.
    ``latest`` may be called from any thread.
    """

    def __init__(
        self,
        calibration: GelCalibration,
        tracker: MarkerShearTracker | None = None,
        settle_frames: int = 8,
        settle_px: float = 0.25,
        min_direction_n: float = 0.5,
    ) -> None:
        if tracker is None:
            from fr3_demo.tactile_shear import MarkerShearTracker

            tracker = MarkerShearTracker()
        self.calibration = calibration
        self._tracker = tracker
        self.settle_frames = int(settle_frames)
        self.settle_px = float(settle_px)
        self.min_direction_n = float(min_direction_n)
        self._lock = threading.Lock()
        self._seq = 0
        self._suspended: str | None = None
        self._zero_requested = True
        self._previous_markers: np.ndarray | None = None
        self._settled = 0
        self._reading = GelReading(0, 0.0, False, "not zeroed yet", 0.0, (0.0, 0.0, 0.0), None, zeroing=True)

    def suspend(self, reason: str) -> None:
        """Report invalid until rezero(), e.g. while the gripper moves and loads the gel."""

        with self._lock:
            self._suspended = reason
            self._reading = replace(self._reading, valid=False, reason=reason, zeroing=True)

    def rezero(self) -> None:
        """Retake the unloaded reference once the gel image has settled (after a grasp)."""

        with self._lock:
            self._suspended = None
            self._zero_requested = True
            self._previous_markers = None
            self._settled = 0
            self._reading = replace(self._reading, valid=False, reason="re-zeroing", zeroing=True)

    @property
    def zeroed(self) -> bool:
        with self._lock:
            return not self._zero_requested

    def wait_zeroed(self, timeout: float) -> GelReading:
        """Block until a reference is taken; raise with the last reason if it is not."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.zeroed:
                return self.latest()
            time.sleep(0.02)
        reason = self.latest().reason
        raise RuntimeError(
            f"The GelSight could not be zeroed within {timeout:.0f} s ({reason}). "
            "Is the marker gel fitted, and is nothing touching it?"
        )

    def on_frame(self, frame: TactileFrame) -> None:
        try:
            self._process(frame)
        except Exception as error:  # noqa: BLE001 - reported as an invalid reading, never raised into capture
            self._publish(frame.captured_monotonic, False, f"gel processing failed: {error}", 0.0, None, None)

    def _process(self, frame: TactileFrame) -> None:
        with self._lock:
            suspended = self._suspended
            zeroing = self._zero_requested
        if suspended is not None:
            self._publish(frame.captured_monotonic, False, suspended, 0.0, None, None)
            return
        if zeroing:
            markers = self._tracker.detect(frame.image)
            moved = _median_motion(self._previous_markers, markers)
            self._previous_markers = markers
            self._settled = self._settled + 1 if moved < self.settle_px else 0
            if self._settled < self.settle_frames:
                self._publish(
                    frame.captured_monotonic, False, "re-zeroing: waiting for the gel to settle", 0.0, None, None
                )
                return
            self._tracker.set_reference(frame.image)
            with self._lock:
                self._zero_requested = False
            self._publish(frame.captured_monotonic, True, "", 0.0, np.zeros(3), None)
            return
        estimate = self._tracker.update(frame.image, frame.captured_monotonic, frame.frame_number)
        if not estimate.valid:
            self._publish(frame.captured_monotonic, False, estimate.reason or "marker tracking failed", 0.0, None, None)
            return
        level, force = self.calibration.level(estimate)
        magnitude = float(np.linalg.norm(force))
        direction = force / magnitude if magnitude >= self.min_direction_n else None
        if not np.isfinite(level):
            self._publish(frame.captured_monotonic, False, "gel level is not finite", 0.0, None, None)
            return
        self._publish(frame.captured_monotonic, True, "", level, force, direction)

    def _publish(
        self,
        timestamp: float,
        valid: bool,
        reason: str,
        level: float,
        force: np.ndarray | None,
        direction: np.ndarray | None,
    ) -> None:
        with self._lock:
            self._seq += 1
            self._reading = GelReading(
                seq=self._seq,
                timestamp=timestamp,
                valid=valid,
                reason=reason,
                level=float(level),
                force_ee=tuple(float(value) for value in (force if force is not None else np.zeros(3))),
                direction_ee=None if direction is None else tuple(float(value) for value in direction),
                zeroing=self._zero_requested or self._suspended is not None,
            )

    def latest(self, max_age: float = 0.2) -> GelReading:
        """The newest reading; invalid once frames are older than ``max_age`` s."""

        with self._lock:
            reading = self._reading
        if reading.valid and time.monotonic() - reading.timestamp > max_age:
            return replace(reading, valid=False, reason="GelSight frames stopped arriving")
        return reading


class GelStreamGuard:
    """When a stream (teleop or a policy) must yield because of the gel.

    It must yield at ``stop_level``, on an invalid reading, or when a re-zero
    after a gripper command takes longer than ``zero_grace_s``; during that
    grace the stream is protected by Bamboo's contact limits alone.
    """

    def __init__(self, monitor: GelMonitor, stop_level: float = 1.0, zero_grace_s: float = 2.5) -> None:
        self.monitor = monitor
        self.stop_level = float(stop_level)
        self.zero_grace_s = float(zero_grace_s)
        self._zero_started: float | None = None

    def check(self) -> str | None:
        """A reason to yield, or None."""

        reading = self.monitor.latest()
        now = time.monotonic()
        if reading.zeroing:
            if self._zero_started is None:
                self._zero_started = now
            if now - self._zero_started > self.zero_grace_s:
                return f"the GelSight did not re-zero within {self.zero_grace_s:.1f} s ({reading.reason})"
            return None
        self._zero_started = None
        if not reading.valid:
            return f"the GelSight signal is not trustworthy ({reading.reason})"
        if reading.level >= self.stop_level:
            load = float(np.linalg.norm(reading.force_ee))
            return f"the gel load reached level {reading.level:.2f} ({load:.1f} N estimated)"
        return None


def limit_twist(twist: np.ndarray, reading: GelReading, rotation: np.ndarray, slow_level: float = 0.5) -> np.ndarray:
    """Remove the part of a commanded flange twist that would load the gel further.

    From ``slow_level`` to level 1 the motion along the gel's load direction
    (and all rotation, which swings the tool tip) fades to zero; backing off
    and sliding sideways stay free. While the gel is re-zeroing the command is
    zero; an invalid reading otherwise slows everything to a quarter, so the
    operator can still back off while Bamboo's contact limits protect.
    """

    command = np.asarray(twist, dtype=float).copy()
    if reading.zeroing:
        return np.zeros_like(command)
    if not reading.valid:
        return 0.25 * command
    if reading.direction_ee is None or reading.level < slow_level:
        return command
    keep = float(np.clip((1.0 - reading.level) / (1.0 - slow_level), 0.0, 1.0))
    direction = np.asarray(rotation, dtype=float) @ np.asarray(reading.direction_ee, dtype=float)
    toward = float(command[:3] @ direction)
    if toward > 0.0:
        command[:3] -= (1.0 - keep) * toward * direction
    command[3:] *= keep
    return command


def fit_calibration(shear_px: np.ndarray, spread: np.ndarray, force_ee: np.ndarray) -> dict[str, Any]:
    """Least-squares map from marker shift (px) and spread to the force through the gel.

    ``force_ee`` is Franka's external force estimate (robot on environment)
    in the end-effector frame, tared, while only the tool is touched, so it
    is exactly the force through the gel. Returns the matrices and how well
    they fit.
    """

    shear = np.asarray(shear_px, dtype=float).reshape(-1, 2)
    press = np.asarray(spread, dtype=float).reshape(-1, 1)
    force = np.asarray(force_ee, dtype=float).reshape(-1, 3)
    if not (len(shear) == len(press) == len(force)) or len(force) < 20:
        raise ValueError("calibration needs at least 20 matched samples")
    features = np.hstack((shear, press))
    solution, *_ = np.linalg.lstsq(features, force, rcond=None)
    predicted = features @ solution
    residual = force - predicted
    total = float(np.sum((force - force.mean(axis=0)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 0.0
    singular = np.linalg.svd(force - force.mean(axis=0), compute_uv=False)
    return {
        "shear_to_force": solution[:2].T,  # (3, 2)
        "spread_to_force": solution[2],  # (3,)
        "r_squared": r_squared,
        "rms_error_n": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "peak_force_n": float(np.max(np.linalg.norm(force, axis=1))),
        # Spread of the pushes: the second singular value says whether more
        # than one direction was pushed, which the in-plane fit needs.
        "push_directions": singular / np.sqrt(len(force)),
    }


def calibrate_main(argv: list[str] | None = None) -> int:
    """Fit the GelSight's force calibration against Franka's own force estimate.

    The arm must stand still in Execution mode with no control loop running,
    and only the tool may be touched: then Franka's external force estimate is
    exactly the force through the gel.
    """

    import argparse
    import queue
    from pathlib import Path

    from fr3_demo.settings import default_config_path, load_config
    from fr3_demo.tactile import GelSightCamera
    from fr3_demo.tactile_shear import MarkerShearTracker

    parser = argparse.ArgumentParser(description=calibrate_main.__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to record pushes")
    parser.add_argument("--server-ip")
    parser.add_argument("--control-port", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    robot_config = config.get("robot", {})
    tactile_config = config.get("tactile", {})
    from bamboo import BambooFrankaClient

    client = BambooFrankaClient(
        server_ip=args.server_ip or robot_config.get("server_ip", "172.16.0.20"),
        control_port=args.control_port or int(robot_config.get("control_port", 5555)),
        enable_gripper=False,
    )
    camera = None
    try:
        state = client.get_joint_states()
        if "ext_wrench" not in state:
            raise RuntimeError("This Bamboo controller does not report Franka's force estimate; update Bamboo")
        if state.get("loop_mode", "idle") != "idle":
            raise RuntimeError("Stop teleop, guide mode or the policy first: the arm must stand still")
        frames: queue.Queue[TactileFrame] = queue.Queue(maxsize=64)
        camera = GelSightCamera(
            device=tactile_config.get("device") or None,
            width=int(tactile_config.get("width", 320)),
            height=int(tactile_config.get("height", 240)),
            rotate_180=bool(tactile_config.get("rotate_180", False)),
        )
        camera.add_frame_observer(lambda frame: frames.put_nowait(frame) if not frames.full() else None)
        camera.start()
        tracker = MarkerShearTracker()

        def force_ee() -> np.ndarray:
            sample = client.get_joint_states()
            rotation = np.asarray(sample["ee_pose"], dtype=float)[:3, :3]
            return rotation.T @ np.asarray(sample["ext_wrench"][:3], dtype=float)

        print("Zeroing: grip the tool as you will use it, and touch nothing for a moment...")
        previous, settled = None, 0
        while settled < 10:
            image = frames.get(timeout=2.0).image
            markers = tracker.detect(image)
            settled = settled + 1 if _median_motion(previous, markers) < 0.25 else 0
            previous = markers
        count = tracker.set_reference(image)
        tare = np.mean([force_ee() for _ in range(50)], axis=0)
        print(f"Zeroed on {count} markers; Franka's force at rest {np.linalg.norm(tare):.2f} N (tared).")
        print(
            f"\nFor the next {args.seconds:.0f} s, press the tool tip with a finger, about 3 to 10 N, slowly, "
            "in several directions: along the tool toward the gripper, and sideways both ways. Hold each push "
            "for a second or two. Touch only the tool, never the gripper or the arm.\n"
        )
        shear, spread, forces = [], [], []
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            frame = frames.get(timeout=2.0)
            estimate = tracker.update(frame.image, frame.captured_monotonic, frame.frame_number)
            force = force_ee() - tare
            if estimate.valid:
                shear.append(estimate.shear_px)
                spread.append(estimate.spread)
                forces.append(force)
            print(
                f"\r  force {np.linalg.norm(force):5.1f} N | marker shift {np.hypot(*estimate.shear_px):5.1f} px | "
                f"{'tracking' if estimate.valid else estimate.reason:30.30s} | {len(forces)} samples",
                end="",
                flush=True,
            )
        print()
        result = fit_calibration(np.asarray(shear), np.asarray(spread), np.asarray(forces))
    finally:
        if camera is not None:
            camera.close()
        client.close()

    directions = result["push_directions"]
    print(
        f"\nFit: R^2 {result['r_squared']:.2f}, error {result['rms_error_n']:.2f} N rms, "
        f"pushes up to {result['peak_force_n']:.1f} N, spread of directions {np.round(directions, 1)} N."
    )
    problems = []
    if result["peak_force_n"] < 3.0:
        problems.append("push harder (at least 3 N)")
    if directions[1] < 1.0:
        problems.append("push in more than one direction")
    if result["r_squared"] < 0.8:
        problems.append("the fit is poor: push slowly, hold each push, and touch only the tool")
    if problems:
        print("Not good enough to use: " + "; ".join(problems) + ". Run it again.")
        return 1
    print("\nCopy into [gel] in config.toml (keep your limit_n):")
    matrix = ", ".join("[" + ", ".join(f"{value:.5f}" for value in row) + "]" for row in result["shear_to_force"])
    print(f"shear_to_force = [{matrix}]")
    print("spread_to_force = [" + ", ".join(f"{value:.3f}" for value in result["spread_to_force"]) + "]")
    return 0
