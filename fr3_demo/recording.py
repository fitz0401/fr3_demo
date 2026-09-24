"""Timestamp-aligned, crash-recoverable raw demonstration recording."""

from __future__ import annotations

import bisect
import json
import logging
import queue
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fr3_demo.cameras import CameraFrame, RealSensePair
from fr3_demo.synchronization import AffineClockMapper

LOG = logging.getLogger("fr3_teleop.recording")
RAW_SCHEMA_VERSION = 3
# DROID stores every view at this size and the LeRobot conversion resizes to it,
# so recording larger only costs disk.
DEFAULT_IMAGE_SIZE = (320, 180)
# Measured on this rig at quality 92: JPEG bytes scale with pixel count.
BYTES_PER_PIXEL = 0.40


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the filesystem that will hold ``path``, created or not."""

    probe = path.expanduser().resolve()
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return shutil.disk_usage(probe).free


def require_free_space(path: Path, minimum_bytes: int) -> int:
    """Refuse to record when the disk cannot hold a useful amount of data."""

    free = free_disk_bytes(path)
    if free < minimum_bytes:
        raise RuntimeError(
            f"Only {free / 1e9:.1f} GB free where {path} would be written; "
            f"at least {minimum_bytes / 1e9:.1f} GB is required. Free space, point --data-dir at "
            f"another disk, or lower --min-free-gb."
        )
    return free


def estimated_bytes_per_frame(image_sizes: list[tuple[int, int]]) -> float:
    """Approximate on-disk cost of one recorded frame across all cameras."""

    return sum(width * height * BYTES_PER_PIXEL for width, height in image_sizes)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


@dataclass(frozen=True)
class RobotSample:
    qpos: np.ndarray
    dq: np.ndarray
    tau_J: np.ndarray
    controller_timestamp: float
    request_send_monotonic: float
    response_receive_monotonic: float


@dataclass(frozen=True)
class GripperSample:
    position: float
    request_send_monotonic: float
    response_receive_monotonic: float
    source: str = "bamboo"

    @property
    def host_timestamp(self) -> float:
        return (self.request_send_monotonic + self.response_receive_monotonic) * 0.5


@dataclass(frozen=True)
class ActionSample:
    value: np.ndarray | float
    host_timestamp: float


class _AsyncImageWriter:
    """Bounded JPEG worker queue; acquisition never performs compression."""

    def __init__(
        self,
        worker_count: int,
        queue_size: int,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Image recording requires: pip install -e '.[recording]'") from error
        self._image_class = Image
        self._image_size = image_size
        self._queue: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue(maxsize=queue_size)
        self._threads = [
            threading.Thread(target=self._worker, name=f"jpeg-writer-{index}", daemon=True)
            for index in range(worker_count)
        ]
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self.high_watermark = 0
        self._closed = False
        for thread in self._threads:
            thread.start()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, image = item
                picture = self._image_class.fromarray(image, mode="RGB")
                if self._image_size is not None and picture.size != self._image_size:
                    # Same filter the LeRobot conversion uses, so the recorded
                    # pixels are exactly what training would have received.
                    picture = picture.resize(self._image_size, self._image_class.Resampling.BICUBIC)
                picture.save(path, quality=92, subsampling=0)
            except BaseException as error:  # noqa: BLE001 - surfaced on the acquisition/control thread
                with self._lock:
                    if self._error is None:
                        self._error = error
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"JPEG writer failed: {error}") from error

    def submit_many(self, items: list[tuple[Path, np.ndarray]]) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("JPEG writer is closed")
        if self._queue.maxsize - self._queue.qsize() < len(items):
            raise RuntimeError(
                f"JPEG writer queue overflow ({self._queue.qsize()}/{self._queue.maxsize}); episode aborted"
            )
        for item in items:
            self._queue.put_nowait(item)
        self.high_watermark = max(self.high_watermark, self._queue.qsize())

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._queue.join()
        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._raise_if_failed()


class RawEpisodeWriter:
    """Write one synchronized episode on an exact host-time sampling grid."""

    def __init__(
        self,
        session_dir: Path,
        episode_index: int,
        fps: float,
        camera_serials: dict[str, str],
        *,
        camera_transforms: dict[str, str] | None = None,
        image_workers: int = 3,
        image_queue_size: int = 120,
        sync_thresholds: dict[str, float] | None = None,
        image_size: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.episode_index = episode_index
        self.fps = fps
        self.started_monotonic = time.monotonic()
        self.path = session_dir / f"episode_{episode_index:06d}.inprogress"
        self.final_path = session_dir / f"episode_{episode_index:06d}"
        required_cameras = {"exterior_image_left", "wrist_image"}
        if not required_cameras.issubset(camera_serials):
            raise ValueError("Recording requires exterior_image_left and wrist_image")
        self._camera_keys = tuple(camera_serials)
        transforms = {
            key: (camera_transforms or {}).get(key, "none")
            for key in self._camera_keys
        }
        if set(transforms.values()) - {"none", "rotate_180"}:
            raise ValueError("Camera transforms must be 'none' or 'rotate_180'")
        if self.path.exists() or self.final_path.exists():
            raise FileExistsError(f"Episode {episode_index} already exists in {session_dir}")
        for key in self._camera_keys:
            (self.path / "frames" / key).mkdir(parents=True)

        self._image_writer = _AsyncImageWriter(image_workers, image_queue_size, image_size)
        self._timestamps: list[float] = []
        self._host_timestamps: list[float] = []
        self._robot_timestamps: list[float] = []
        self._robot_timing: dict[str, list[float]] = {
            "robot_before_timestamp": [],
            "robot_after_timestamp": [],
            "robot_before_host_timestamp": [],
            "robot_after_host_timestamp": [],
            "robot_before_request_send_timestamp": [],
            "robot_before_response_receive_timestamp": [],
            "robot_after_request_send_timestamp": [],
            "robot_after_response_receive_timestamp": [],
            "robot_interpolation_alpha": [],
        }
        self._camera_timing: dict[str, dict[str, list[Any]]] = {
            key: {
                "arrival": [],
                "hardware": [],
                "aligned": [],
                "frame_number": [],
                "timestamp_domain": [],
            }
            for key in self._camera_keys
        }
        self._joint_positions: list[np.ndarray] = []
        self._joint_velocities: list[np.ndarray] = []
        self._joint_torques: list[np.ndarray] = []
        self._action_joint_velocities: list[np.ndarray] = []
        self._action_joint_timestamps: list[float] = []
        self._gripper_positions: list[float] = []
        self._gripper_timing: dict[str, list[Any]] = {
            "gripper_host_timestamp": [],
            "gripper_request_send_timestamp": [],
            "gripper_response_receive_timestamp": [],
            "gripper_sample_age": [],
            "gripper_sample_source": [],
        }
        self._action_gripper_positions: list[float] = []
        self._action_gripper_timestamps: list[float] = []
        self._camera_target_errors_ms: dict[str, list[float]] = {key: [] for key in self._camera_keys}
        self._camera_pair_skews_ms: list[float] = []
        self._robot_interpolation_gaps_ms: list[float] = []
        self._gripper_ages_ms: list[float] = []
        self._action_ages_ms: list[float] = []
        self._sync_thresholds = dict(sync_thresholds or {})
        self._clock_estimates: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {
            "schema_version": RAW_SCHEMA_VERSION,
            "episode_index": episode_index,
            "complete": False,
            "created_at": _utc_now(),
            "fps": fps,
            "timebase": "host_monotonic",
            "timestamp_unit": "seconds",
            "camera_serials": camera_serials,
            "image_size": (
                None
                if image_size is None
                else {"width": image_size[0], "height": image_size[1], "resample": "bicubic"}
            ),
            "force_signal": {
                "field": "joint_torque",
                "source": "franka_tau_J",
                "unit": "Nm",
                "frame": "joint",
            },
            "camera_transforms": transforms,
            "language_instruction": None,
            "frame_count": 0,
        }
        _write_json(self.path / "metadata.json", self._metadata)

    @property
    def frame_count(self) -> int:
        return len(self._timestamps)

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return self._camera_keys

    def add_sample(
        self,
        target_monotonic: float,
        state: dict[str, Any],
        action_joint_velocity: np.ndarray,
        gripper_position: float,
        action_gripper_position: float,
        camera_frames: dict[str, CameraFrame],
        timing: dict[str, Any] | None = None,
    ) -> None:
        timing = timing or {}
        index = self.frame_count
        missing_cameras = set(self._camera_keys) - set(camera_frames)
        if missing_cameras:
            missing = ", ".join(sorted(missing_cameras))
            raise RuntimeError(f"Active recording camera disappeared: {missing}")

        image_items: list[tuple[Path, np.ndarray]] = []
        aligned_camera_times: list[float] = []
        for key in self._camera_keys:
            frame = camera_frames[key]
            image_path = self.path / "frames" / key / f"frame_{index:06d}.jpg"
            image_items.append((image_path, frame.image))
            aligned = float(frame.alignment_timestamp)
            aligned_camera_times.append(aligned)
            values = self._camera_timing[key]
            values["arrival"].append(float(frame.captured_monotonic))
            values["hardware"].append(float(frame.hardware_timestamp))
            values["aligned"].append(aligned)
            values["frame_number"].append(int(frame.frame_number))
            values["timestamp_domain"].append(frame.timestamp_domain)
            self._camera_target_errors_ms[key].append(abs(aligned - target_monotonic) * 1000.0)
        self._image_writer.submit_many(image_items)

        self._timestamps.append(target_monotonic - self.started_monotonic)
        self._host_timestamps.append(target_monotonic)
        self._robot_timestamps.append(float(state["time_sec"]))
        self._joint_positions.append(np.asarray(state["qpos"], dtype=np.float32))
        self._joint_velocities.append(np.asarray(state["dq"], dtype=np.float32))
        self._joint_torques.append(np.asarray(state["tau_J"], dtype=np.float32))
        self._action_joint_velocities.append(np.asarray(action_joint_velocity, dtype=np.float32))
        action_timestamp = float(timing.get("action_joint_host_timestamp", target_monotonic))
        self._action_joint_timestamps.append(action_timestamp)
        self._gripper_positions.append(float(gripper_position))
        self._action_gripper_positions.append(float(action_gripper_position))
        self._action_gripper_timestamps.append(float(timing.get("action_gripper_host_timestamp", target_monotonic)))
        for key in self._robot_timing:
            self._robot_timing[key].append(float(timing.get(key, np.nan)))
        for key in self._gripper_timing:
            default: Any = "unknown" if key == "gripper_sample_source" else np.nan
            self._gripper_timing[key].append(timing.get(key, default))

        self._camera_pair_skews_ms.append((max(aligned_camera_times) - min(aligned_camera_times)) * 1000.0)
        robot_gap = (
            float(timing.get("robot_after_host_timestamp", target_monotonic))
            - float(timing.get("robot_before_host_timestamp", target_monotonic))
        ) * 1000.0
        self._robot_interpolation_gaps_ms.append(max(0.0, robot_gap))
        gripper_age = float(timing.get("gripper_sample_age", 0.0)) * 1000.0
        self._gripper_ages_ms.append(max(0.0, gripper_age))
        self._action_ages_ms.append(max(0.0, target_monotonic - action_timestamp) * 1000.0)

    def _sync_report(self) -> dict[str, Any]:
        warn_threshold = float(self._sync_thresholds.get("camera_pair_warn_ms", 20.0))
        target_threshold = float(self._sync_thresholds.get("camera_target_max_ms", float("inf")))
        pair_threshold = float(self._sync_thresholds.get("camera_pair_reject_ms", float("inf")))
        robot_threshold = float(self._sync_thresholds.get("robot_interpolation_gap_max_ms", float("inf")))
        warning_frames = sum(value > warn_threshold for value in self._camera_pair_skews_ms)
        violations = []
        if any(max(values, default=0.0) > target_threshold for values in self._camera_target_errors_ms.values()):
            violations.append("camera_target_error")
        if max(self._camera_pair_skews_ms, default=0.0) > pair_threshold:
            violations.append("camera_pair_skew")
        if max(self._robot_interpolation_gaps_ms, default=0.0) > robot_threshold:
            violations.append("robot_interpolation_gap")
        return {
            "schema_version": RAW_SCHEMA_VERSION,
            "valid": not violations,
            "violations": violations,
            "timebase": "host_monotonic",
            "thresholds": self._sync_thresholds,
            "camera_target_error_ms": {
                key: _percentiles(values) for key, values in self._camera_target_errors_ms.items()
            },
            "camera_pair_skew_ms": _percentiles(self._camera_pair_skews_ms),
            "camera_pair_warning_frames": warning_frames,
            "robot_interpolation_gap_ms": _percentiles(self._robot_interpolation_gaps_ms),
            "gripper_sample_age_ms": _percentiles(self._gripper_ages_ms),
            "joint_action_age_ms": _percentiles(self._action_ages_ms),
            "jpeg_queue": {
                "capacity": self._image_writer.capacity,
                "high_watermark": self._image_writer.high_watermark,
                "dropped": 0,
            },
            "clock_estimates": self._clock_estimates,
        }

    def finish(self, clock_estimates: dict[str, Any] | None = None) -> Path:
        if self.frame_count < 2:
            raise RuntimeError("An episode needs at least two synchronized frames")
        if clock_estimates is not None:
            self._clock_estimates = clock_estimates
        self._image_writer.close()
        trajectory: dict[str, np.ndarray] = {
            "timestamp": np.asarray(self._timestamps, dtype=np.float64),
            "host_monotonic_timestamp": np.asarray(self._host_timestamps, dtype=np.float64),
            "robot_timestamp": np.asarray(self._robot_timestamps, dtype=np.float64),
            "joint_position": np.stack(self._joint_positions),
            "joint_velocity": np.stack(self._joint_velocities),
            "joint_torque": np.stack(self._joint_torques),
            "action_joint_velocity": np.stack(self._action_joint_velocities),
            "action_joint_host_timestamp": np.asarray(self._action_joint_timestamps, dtype=np.float64),
            "gripper_position": np.asarray(self._gripper_positions, dtype=np.float32)[:, None],
            "action_gripper_position": np.asarray(self._action_gripper_positions, dtype=np.float32)[:, None],
            "action_gripper_host_timestamp": np.asarray(self._action_gripper_timestamps, dtype=np.float64),
        }
        for key, values in self._robot_timing.items():
            trajectory[key] = np.asarray(values, dtype=np.float64)
        for key, values in self._gripper_timing.items():
            dtype = np.str_ if key == "gripper_sample_source" else np.float64
            trajectory[key] = np.asarray(values, dtype=dtype)
        camera_prefixes = {
            "exterior_image_left": "exterior_camera",
            "wrist_image": "wrist_camera",
            "exterior_image_2_left": "exterior2_camera",
        }
        for key, values in self._camera_timing.items():
            prefix = camera_prefixes[key]
            trajectory[f"{prefix}_timestamp"] = np.asarray(values["arrival"], dtype=np.float64)
            trajectory[f"{prefix}_hardware_timestamp"] = np.asarray(values["hardware"], dtype=np.float64)
            trajectory[f"{prefix}_aligned_timestamp"] = np.asarray(values["aligned"], dtype=np.float64)
            trajectory[f"{prefix}_frame_number"] = np.asarray(values["frame_number"], dtype=np.int64)
            trajectory[f"{prefix}_timestamp_domain"] = np.asarray(values["timestamp_domain"], dtype=np.str_)
        np.savez_compressed(self.path / "trajectory.npz", **trajectory)

        report = self._sync_report()
        _write_json(self.path / "sync_report.json", report)
        if not report["valid"]:
            raise RuntimeError(f"Episode failed synchronization checks: {', '.join(report['violations'])}")
        LOG.info(
            "Synchronization passed: camera skew p95 %.1f ms; robot gap p95 %.1f ms; camera warnings %d",
            report["camera_pair_skew_ms"]["p95"],
            report["robot_interpolation_gap_ms"]["p95"],
            report["camera_pair_warning_frames"],
        )
        self._metadata.update(
            {
                "complete": True,
                "finished_at": _utc_now(),
                "frame_count": self.frame_count,
                "duration_seconds": self._timestamps[-1],
                "sync": {
                    "valid": True,
                    "report": "sync_report.json",
                    "camera_pair_skew_p95_ms": report["camera_pair_skew_ms"]["p95"],
                    "robot_interpolation_gap_p95_ms": report["robot_interpolation_gap_ms"]["p95"],
                },
            }
        )
        _write_json(self.path / "metadata.json", self._metadata)
        self.path.rename(self.final_path)
        return self.final_path

    def abort(self, reason: str) -> None:
        try:
            self._image_writer.close()
        except Exception as error:  # noqa: BLE001 - preserve the original acquisition error
            LOG.error("Could not drain JPEG queue while aborting: %s", error)
        self._metadata.update(
            {
                "complete": False,
                "aborted_at": _utc_now(),
                "frame_count": self.frame_count,
                "error": reason,
            }
        )
        _write_json(self.path / "metadata.json", self._metadata)


class DemoCollector:
    """Align independent robot, gripper, action, and camera streams."""

    def __init__(
        self,
        cameras: RealSensePair,
        output_root: Path,
        server_ip: str,
        control_port: int,
        fps: float = 15.0,
        gripper_port: int = 5559,
        gripper_type: str = "robotiq",
        enable_gripper: bool = True,
        *,
        alignment_delay_ms: float = 80.0,
        camera_max_delta_ms: float = 25.0,
        camera_pair_warn_ms: float = 20.0,
        camera_pair_reject_ms: float = 35.0,
        robot_sample_hz: float = 60.0,
        robot_max_gap_ms: float = 30.0,
        gripper_sample_hz: float = 30.0,
        gripper_max_age_ms: float = 2000.0,
        image_workers: int = 3,
        image_queue_size: int = 120,
        image_size: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
        min_free_gb: float = 5.0,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = output_root.expanduser().resolve()
        self.image_size = image_size
        self.min_free_bytes = int(min_free_gb * 1e9)
        free = require_free_space(root, self.min_free_bytes)
        self.session_dir = root / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.cameras = cameras
        self.server_ip = server_ip
        self.control_port = control_port
        self.fps = fps
        self.gripper_port = gripper_port
        self.gripper_type = gripper_type
        self.enable_gripper = enable_gripper
        self.alignment_delay = alignment_delay_ms / 1000.0
        self.camera_max_delta = camera_max_delta_ms / 1000.0
        self.camera_pair_reject = camera_pair_reject_ms / 1000.0
        self.robot_sample_hz = robot_sample_hz
        self.robot_max_gap = robot_max_gap_ms / 1000.0
        self.gripper_sample_hz = gripper_sample_hz
        self.gripper_max_age = gripper_max_age_ms / 1000.0
        self.image_workers = image_workers
        self.image_queue_size = image_queue_size
        bytes_per_frame = estimated_bytes_per_frame(self._recorded_image_sizes(cameras, image_size))
        rate = bytes_per_frame * fps * 3600.0 / 1e9
        hours = free / 1e9 / rate if rate > 0 else float("inf")
        LOG.info(
            "Disk: %.1f GB free at %s; recording costs about %.1f GB/hour, so roughly %.1f hours fit.",
            free / 1e9,
            root,
            rate,
            hours,
        )
        if hours < 1.0:
            LOG.warning("Less than an hour of recording fits on this disk.")
        self._sync_thresholds = {
            "alignment_delay_ms": alignment_delay_ms,
            "camera_target_max_ms": camera_max_delta_ms,
            "camera_pair_warn_ms": camera_pair_warn_ms,
            "camera_pair_reject_ms": camera_pair_reject_ms,
            "robot_interpolation_gap_max_ms": robot_max_gap_ms,
            "gripper_sample_age_max_ms": gripper_max_age_ms,
        }
        _write_json(
            self.session_dir / "session.json",
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "created_at": _utc_now(),
                "fps": fps,
                "timebase": "host_monotonic",
                "timestamp_unit": "seconds",
                "camera_serials": cameras.serials,
                "camera_transforms": cameras.camera_transforms,
                "synchronization": self._sync_thresholds,
                "image_size": (
                    None
                    if image_size is None
                    else {"width": image_size[0], "height": image_size[1], "resample": "bicubic"}
                ),
                "format": "fr3_demo_raw",
            },
        )

        self._lock = threading.Lock()
        self._sensor_lock = threading.Lock()
        self._stop = threading.Event()
        self._arm_ready = threading.Event()
        self._gripper_ready = threading.Event()
        self._threads: list[threading.Thread] = []
        self._writer: RawEpisodeWriter | None = None
        self._error: BaseException | None = None
        self._arm_samples: deque[RobotSample] = deque(maxlen=max(300, int(robot_sample_hz * 10)))
        self._gripper_samples: deque[GripperSample] = deque(maxlen=max(150, int(gripper_sample_hz * 10)))
        self._arm_clock = AffineClockMapper(max_samples=max(300, int(robot_sample_hz * 20)))
        now = time.monotonic()
        self._joint_actions: deque[ActionSample] = deque(
            [ActionSample(np.zeros(7, dtype=np.float32), now)], maxlen=3000
        )
        self._gripper_actions: deque[ActionSample] = deque([ActionSample(0.0, now)], maxlen=1000)
        self._frame_observer: Callable[[dict[str, CameraFrame]], None] | None = None

    @staticmethod
    def _recorded_image_sizes(
        cameras: RealSensePair,
        image_size: tuple[int, int] | None,
    ) -> list[tuple[int, int]]:
        """Size of each written image, for the disk estimate."""

        if image_size is not None:
            return [image_size] * len(cameras.serials)
        # Recording at native resolution: ask each camera for its own mode.
        return [(camera.width, camera.height) for camera in cameras.active_cameras.values()]

    @property
    def active(self) -> bool:
        with self._lock:
            return self._writer is not None

    def _set_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._arm_loop, name="demo-arm-sampler", daemon=True),
            threading.Thread(target=self._alignment_loop, name="demo-aligner", daemon=True),
        ]
        if self.enable_gripper:
            self._threads.append(
                threading.Thread(target=self._gripper_loop, name="demo-gripper-sampler", daemon=True)
            )
        else:
            self._gripper_ready.set()
        for thread in self._threads:
            thread.start()
        if not self._arm_ready.wait(timeout=6.0):
            self.check_health()
            raise RuntimeError("Timed out starting the arm-state sampler")
        if not self._gripper_ready.wait(timeout=6.0):
            self.check_health()
            raise RuntimeError("Timed out starting the gripper-state sampler")
        self.check_health()

    def _next_episode_index(self) -> int:
        indices = []
        for path in self.session_dir.glob("episode_*"):
            try:
                indices.append(int(path.name.split(".")[0].split("_")[-1]))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def start_episode(self) -> Path:
        self.check_health()
        require_free_space(self.session_dir, self.min_free_bytes)
        self.cameras.snapshot()
        with self._sensor_lock:
            if len(self._arm_samples) < 2:
                raise RuntimeError("Arm-state buffer is not ready")
            if self.enable_gripper and not self._gripper_samples:
                raise RuntimeError("Gripper-state buffer is not ready")
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("An episode is already recording")
            self._writer = RawEpisodeWriter(
                self.session_dir,
                self._next_episode_index(),
                self.fps,
                self.cameras.serials,
                camera_transforms=self.cameras.camera_transforms,
                image_workers=self.image_workers,
                image_queue_size=self.image_queue_size,
                sync_thresholds=self._sync_thresholds,
                image_size=self.image_size,
            )
            return self._writer.path

    def _clock_estimates(self) -> dict[str, Any]:
        with self._sensor_lock:
            arm = self._arm_clock.estimate()
        return {
            "cameras": self.cameras.clock_estimates,
            "robot_controller": {
                "scale": arm.scale,
                "offset": arm.offset,
                "sample_count": arm.sample_count,
                "residual_p95_ms": arm.residual_p95 * 1000.0,
            },
        }

    def stop_episode(self) -> Path:
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is None:
            raise RuntimeError("No episode is recording")
        try:
            return writer.finish(self._clock_estimates())
        except Exception as error:
            writer.abort(str(error))
            raise

    def set_frame_observer(self, observer: Callable[[dict[str, CameraFrame]], None] | None) -> None:
        """Hand each recorded frame set to a viewer instead of it reading cameras.

        A live preview must not compete with acquisition for the camera locks,
        so it receives exactly the frames written to disk.  The observer runs on
        the alignment thread and must only store a reference and return.
        """

        self._frame_observer = observer

    def set_action(self, joint_velocity: np.ndarray, host_timestamp: float | None = None) -> None:
        sample = ActionSample(
            np.asarray(joint_velocity, dtype=np.float32).copy(),
            time.monotonic() if host_timestamp is None else float(host_timestamp),
        )
        with self._sensor_lock:
            self._joint_actions.append(sample)

    def set_gripper_action(self, position: float, host_timestamp: float | None = None) -> None:
        sample = ActionSample(
            float(np.clip(position, 0.0, 1.0)),
            time.monotonic() if host_timestamp is None else float(host_timestamp),
        )
        with self._sensor_lock:
            self._gripper_actions.append(sample)

    def set_gripper_observation(
        self,
        position: float,
        host_timestamp: float | None = None,
        *,
        source: str = "manual",
    ) -> None:
        timestamp = time.monotonic() if host_timestamp is None else float(host_timestamp)
        sample = GripperSample(float(np.clip(position, 0.0, 1.0)), timestamp, timestamp, source)
        with self._sensor_lock:
            self._gripper_samples.append(sample)

    def check_health(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Demonstration recorder failed: {error}") from error

    def _arm_loop(self) -> None:
        client = None
        try:
            from bamboo import BambooFrankaClient

            client = BambooFrankaClient(server_ip=self.server_ip, control_port=self.control_port, enable_gripper=False)
            period = 1.0 / self.robot_sample_hz
            while not self._stop.is_set():
                started = time.monotonic()
                state = client.get_joint_states()
                received = time.monotonic()
                if "tau_J" not in state:
                    raise RuntimeError(
                        "Robot state has no tau_J; this bamboo controller cannot report joint torque"
                    )
                sample = RobotSample(
                    qpos=np.asarray(state["qpos"], dtype=np.float32),
                    dq=np.asarray(state["dq"], dtype=np.float32),
                    tau_J=np.asarray(state["tau_J"], dtype=np.float32),
                    controller_timestamp=float(state["time_sec"]),
                    request_send_monotonic=started,
                    response_receive_monotonic=received,
                )
                midpoint = (started + received) * 0.5
                with self._sensor_lock:
                    if not self._arm_samples or sample.controller_timestamp > self._arm_samples[-1].controller_timestamp:
                        self._arm_clock.add(
                            sample.controller_timestamp,
                            midpoint,
                            uncertainty=(received - started) * 0.5,
                        )
                        self._arm_samples.append(sample)
                self._arm_ready.set()
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except BaseException as error:  # noqa: BLE001 - propagated through check_health
            if not self._stop.is_set():
                self._set_error(error)
            self._arm_ready.set()
        finally:
            if client is not None:
                client.close()

    def _gripper_loop(self) -> None:
        client = None
        try:
            from bamboo import BambooFrankaClient

            client = BambooFrankaClient(
                server_ip=self.server_ip,
                control_port=self.control_port,
                gripper_port=self.gripper_port,
                gripper_type=self.gripper_type,
                enable_gripper=True,
            )
            period = 1.0 / self.gripper_sample_hz
            maximum_width = 0.085 if self.gripper_type == "robotiq" else 0.08
            while not self._stop.is_set():
                started = time.monotonic()
                result = client.get_gripper_state()
                received = time.monotonic()
                if not result.get("success", False):
                    raise RuntimeError(str(result.get("error", "Bamboo failed to read gripper state")))
                position = float(np.clip(float(result["state"]["width"]) / maximum_width, 0.0, 1.0))
                with self._sensor_lock:
                    self._gripper_samples.append(GripperSample(position, started, received))
                self._gripper_ready.set()
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except BaseException as error:  # noqa: BLE001 - propagated through check_health
            if not self._stop.is_set():
                self._set_error(error)
            self._gripper_ready.set()
        finally:
            if client is not None:
                client.close()

    def _interpolate_robot(self, target: float) -> tuple[dict[str, Any], dict[str, float]]:
        with self._sensor_lock:
            samples = list(self._arm_samples)
            estimate = self._arm_clock.estimate()
        host_times = [estimate.scale * item.controller_timestamp + estimate.offset for item in samples]
        index = bisect.bisect_left(host_times, target)
        if index == 0 or index >= len(samples):
            raise RuntimeError("Robot-state buffer does not bracket the target timestamp")
        before, after = samples[index - 1], samples[index]
        before_host, after_host = host_times[index - 1], host_times[index]
        gap = after_host - before_host
        if gap <= 0.0 or gap > self.robot_max_gap:
            raise RuntimeError(f"Robot-state interpolation gap is {gap * 1000.0:.1f} ms")
        alpha = float(np.clip((target - before_host) / gap, 0.0, 1.0))
        state = {
            "qpos": before.qpos + alpha * (after.qpos - before.qpos),
            "dq": before.dq + alpha * (after.dq - before.dq),
            "tau_J": before.tau_J + alpha * (after.tau_J - before.tau_J),
            "time_sec": before.controller_timestamp
            + alpha * (after.controller_timestamp - before.controller_timestamp),
        }
        timing = {
            "robot_before_timestamp": before.controller_timestamp,
            "robot_after_timestamp": after.controller_timestamp,
            "robot_before_host_timestamp": before_host,
            "robot_after_host_timestamp": after_host,
            "robot_before_request_send_timestamp": before.request_send_monotonic,
            "robot_before_response_receive_timestamp": before.response_receive_monotonic,
            "robot_after_request_send_timestamp": after.request_send_monotonic,
            "robot_after_response_receive_timestamp": after.response_receive_monotonic,
            "robot_interpolation_alpha": alpha,
        }
        return state, timing

    def _interpolate_gripper(self, target: float) -> tuple[float, dict[str, Any]]:
        if not self.enable_gripper:
            # Nothing is observing the gripper, so there is no observation to
            # age out.  Record a fixed value and say so in the source column.
            return 0.0, {
                "gripper_host_timestamp": target,
                "gripper_request_send_timestamp": target,
                "gripper_response_receive_timestamp": target,
                "gripper_sample_age": 0.0,
                "gripper_sample_source": "disabled",
            }
        with self._sensor_lock:
            samples = list(self._gripper_samples)
        if not samples:
            raise RuntimeError("Gripper-state buffer is empty")
        host_times = [item.host_timestamp for item in samples]
        index = bisect.bisect_left(host_times, target)
        if 0 < index < len(samples):
            before, after = samples[index - 1], samples[index]
            gap = after.host_timestamp - before.host_timestamp
            if 0.0 < gap <= self.gripper_max_age:
                alpha = float(np.clip((target - before.host_timestamp) / gap, 0.0, 1.0))
                position = before.position + alpha * (after.position - before.position)
                nearest = before if alpha < 0.5 else after
                return position, {
                    "gripper_host_timestamp": target,
                    "gripper_request_send_timestamp": nearest.request_send_monotonic,
                    "gripper_response_receive_timestamp": nearest.response_receive_monotonic,
                    "gripper_sample_age": min(target - before.host_timestamp, after.host_timestamp - target),
                    "gripper_sample_source": f"interpolated:{before.source}+{after.source}",
                }
        # Gripper servers can be busy while executing a blocking close/open.
        # Hold the latest prior observation, but preserve its age for auditing.
        prior_index = max(0, index - 1)
        sample = samples[prior_index]
        age = max(0.0, target - sample.host_timestamp)
        if age > self.gripper_max_age:
            raise RuntimeError(f"Gripper observation is stale ({age * 1000.0:.1f} ms)")
        return sample.position, {
            "gripper_host_timestamp": sample.host_timestamp,
            "gripper_request_send_timestamp": sample.request_send_monotonic,
            "gripper_response_receive_timestamp": sample.response_receive_monotonic,
            "gripper_sample_age": age,
            "gripper_sample_source": f"held:{sample.source}",
        }

    def _action_at(self, samples: deque[ActionSample], target: float) -> ActionSample:
        with self._sensor_lock:
            for sample in reversed(samples):
                if sample.host_timestamp <= target:
                    return sample
            return samples[0]

    def _alignment_loop(self) -> None:
        current_writer: RawEpisodeWriter | None = None
        next_target = 0.0
        last_frame_numbers: dict[str, int] = {}
        period = 1.0 / self.fps
        next_disk_check = 0.0
        try:
            while not self._stop.is_set():
                with self._lock:
                    writer = self._writer
                if writer is None:
                    current_writer = None
                    self._stop.wait(0.01)
                    continue
                if writer is not current_writer:
                    current_writer = writer
                    next_target = writer.started_monotonic
                    last_frame_numbers = {}
                    next_disk_check = 0.0

                # Stop cleanly while the episode is still writable rather than
                # letting a JPEG fail part-way through a full disk.
                now = time.monotonic()
                if now >= next_disk_check:
                    require_free_space(self.session_dir, self.min_free_bytes)
                    next_disk_check = now + 2.0

                due = next_target + self.alignment_delay
                remaining = due - time.monotonic()
                if remaining > 0:
                    self._stop.wait(min(remaining, 0.02))
                    continue

                frames = self.cameras.nearest(next_target, self.camera_max_delta, last_frame_numbers)
                missing = set(writer.camera_keys) - set(frames)
                if missing:
                    raise RuntimeError(f"Active recording camera disappeared: {', '.join(sorted(missing))}")
                camera_times = [frame.alignment_timestamp for frame in frames.values()]
                pair_skew = max(camera_times) - min(camera_times)
                if pair_skew > self.camera_pair_reject:
                    raise RuntimeError(f"Inter-camera skew is {pair_skew * 1000.0:.1f} ms")

                state, timing = self._interpolate_robot(next_target)
                gripper, gripper_timing = self._interpolate_gripper(next_target)
                joint_action = self._action_at(self._joint_actions, next_target)
                gripper_action = self._action_at(self._gripper_actions, next_target)
                timing.update(gripper_timing)
                timing["action_joint_host_timestamp"] = joint_action.host_timestamp
                timing["action_gripper_host_timestamp"] = gripper_action.host_timestamp

                written = False
                with self._lock:
                    if self._writer is writer:
                        writer.add_sample(
                            next_target,
                            state,
                            np.asarray(joint_action.value, dtype=np.float32),
                            gripper,
                            float(gripper_action.value),
                            frames,
                            timing,
                        )
                        written = True
                if written and self._frame_observer is not None:
                    try:
                        self._frame_observer(frames)
                    except Exception as error:  # noqa: BLE001 - a viewer must never abort an episode
                        self._frame_observer = None
                        LOG.warning("Detached the frame observer after it failed: %s", error)
                last_frame_numbers = {key: frame.frame_number for key, frame in frames.items()}
                next_target += period
        except BaseException as error:  # noqa: BLE001 - propagated through check_health
            writer_to_abort: RawEpisodeWriter | None = None
            with self._lock:
                if not self._stop.is_set():
                    self._error = error
                writer_to_abort = self._writer
                self._writer = None
            if writer_to_abort is not None:
                writer_to_abort.abort(str(error))

    def close(self) -> Path | None:
        completed: Path | None = None
        if self.active:
            completed = self.stop_episode()
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=6.0)
        return completed
