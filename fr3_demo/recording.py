"""Crash-recoverable raw demonstration recording."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fr3_demo.cameras import RealSensePair


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class RawEpisodeWriter:
    """Write one raw episode as synchronized JPEG frames and numeric arrays."""

    def __init__(
        self,
        session_dir: Path,
        episode_index: int,
        fps: float,
        camera_serials: dict[str, str],
        *,
        wrist_rotate_180: bool = False,
    ) -> None:
        self.episode_index = episode_index
        self.fps = fps
        self.started_monotonic = time.monotonic()
        self.path = session_dir / f"episode_{episode_index:06d}.inprogress"
        self.final_path = session_dir / f"episode_{episode_index:06d}"
        if self.path.exists() or self.final_path.exists():
            raise FileExistsError(f"Episode {episode_index} already exists in {session_dir}")
        (self.path / "frames" / "exterior_image_left").mkdir(parents=True)
        (self.path / "frames" / "wrist_image").mkdir(parents=True)
        self._timestamps: list[float] = []
        self._robot_timestamps: list[float] = []
        self._camera_timestamps: dict[str, list[float]] = {
            "exterior_image_left": [],
            "wrist_image": [],
        }
        self._joint_positions: list[np.ndarray] = []
        self._joint_velocities: list[np.ndarray] = []
        self._action_joint_velocities: list[np.ndarray] = []
        self._gripper_positions: list[float] = []
        self._action_gripper_positions: list[float] = []
        self._metadata: dict[str, Any] = {
            "schema_version": 1,
            "episode_index": episode_index,
            "complete": False,
            "created_at": _utc_now(),
            "fps": fps,
            "camera_serials": camera_serials,
            "camera_transforms": {
                "exterior_image_left": "none",
                "wrist_image": "rotate_180" if wrist_rotate_180 else "none",
            },
            "language_instruction": None,
            "frame_count": 0,
        }
        _write_json(self.path / "metadata.json", self._metadata)

    @property
    def frame_count(self) -> int:
        return len(self._timestamps)

    def add_sample(
        self,
        captured_monotonic: float,
        state: dict[str, Any],
        action_joint_velocity: np.ndarray,
        gripper_position: float,
        action_gripper_position: float,
        camera_frames: dict[str, Any],
    ) -> None:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Image recording requires: pip install -e '.[recording]'") from error

        index = self.frame_count
        for key in ("exterior_image_left", "wrist_image"):
            frame = camera_frames[key]
            image_path = self.path / "frames" / key / f"frame_{index:06d}.jpg"
            Image.fromarray(frame.image, mode="RGB").save(image_path, quality=92, subsampling=0)
            self._camera_timestamps[key].append(float(frame.captured_monotonic))

        self._timestamps.append(captured_monotonic - self.started_monotonic)
        self._robot_timestamps.append(float(state["time_sec"]))
        self._joint_positions.append(np.asarray(state["qpos"], dtype=np.float32))
        self._joint_velocities.append(np.asarray(state["dq"], dtype=np.float32))
        self._action_joint_velocities.append(np.asarray(action_joint_velocity, dtype=np.float32))
        self._gripper_positions.append(float(gripper_position))
        self._action_gripper_positions.append(float(action_gripper_position))

    def finish(self) -> Path:
        if self.frame_count < 2:
            raise RuntimeError("An episode needs at least two synchronized frames")
        np.savez_compressed(
            self.path / "trajectory.npz",
            timestamp=np.asarray(self._timestamps, dtype=np.float64),
            robot_timestamp=np.asarray(self._robot_timestamps, dtype=np.float64),
            exterior_camera_timestamp=np.asarray(self._camera_timestamps["exterior_image_left"], dtype=np.float64),
            wrist_camera_timestamp=np.asarray(self._camera_timestamps["wrist_image"], dtype=np.float64),
            joint_position=np.stack(self._joint_positions),
            joint_velocity=np.stack(self._joint_velocities),
            action_joint_velocity=np.stack(self._action_joint_velocities),
            gripper_position=np.asarray(self._gripper_positions, dtype=np.float32)[:, None],
            action_gripper_position=np.asarray(self._action_gripper_positions, dtype=np.float32)[:, None],
        )
        self._metadata.update(
            {
                "complete": True,
                "finished_at": _utc_now(),
                "frame_count": self.frame_count,
                "duration_seconds": self._timestamps[-1],
            }
        )
        _write_json(self.path / "metadata.json", self._metadata)
        self.path.rename(self.final_path)
        return self.final_path

    def abort(self, reason: str) -> None:
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
    """Record robot state and the latest two camera frames at a fixed rate."""

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
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_dir = output_root.expanduser().resolve() / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            self.session_dir / "session.json",
            {
                "schema_version": 1,
                "created_at": _utc_now(),
                "fps": fps,
                "camera_serials": cameras.serials,
                "camera_transforms": {
                    "exterior_image_left": "none",
                    "wrist_image": "rotate_180" if cameras.wrist_rotate_180 else "none",
                },
                "format": "fr3_demo_raw",
            },
        )
        self.cameras = cameras
        self.server_ip = server_ip
        self.control_port = control_port
        self.fps = fps
        self.gripper_port = gripper_port
        self.gripper_type = gripper_type
        self.enable_gripper = enable_gripper
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: RawEpisodeWriter | None = None
        self._error: BaseException | None = None
        self._action_joint_velocity = np.zeros(7, dtype=np.float32)
        self._gripper_position = 0.0
        self._action_gripper_position = 0.0

    @property
    def active(self) -> bool:
        with self._lock:
            return self._writer is not None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._record_loop, name="demo-recorder", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Timed out starting the demonstration recorder")
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
        self.cameras.snapshot()
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("An episode is already recording")
            self._writer = RawEpisodeWriter(
                self.session_dir,
                self._next_episode_index(),
                self.fps,
                self.cameras.serials,
                wrist_rotate_180=self.cameras.wrist_rotate_180,
            )
            return self._writer.path

    def stop_episode(self) -> Path:
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is None:
            raise RuntimeError("No episode is recording")
        return writer.finish()

    def set_action(self, joint_velocity: np.ndarray) -> None:
        with self._lock:
            self._action_joint_velocity = np.asarray(joint_velocity, dtype=np.float32).copy()

    def set_gripper(self, observed_position: float, action_position: float | None = None) -> None:
        with self._lock:
            self._gripper_position = float(np.clip(observed_position, 0.0, 1.0))
            self._action_gripper_position = float(
                np.clip(observed_position if action_position is None else action_position, 0.0, 1.0)
            )

    def check_health(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Demonstration recorder failed: {error}") from error

    def _record_loop(self) -> None:
        writer_to_abort: RawEpisodeWriter | None = None
        client = None
        try:
            from bamboo import BambooFrankaClient

            client = BambooFrankaClient(
                server_ip=self.server_ip,
                control_port=self.control_port,
                gripper_port=self.gripper_port,
                gripper_type=self.gripper_type,
                enable_gripper=self.enable_gripper,
            )
            self._ready.set()
            period = 1.0 / self.fps
            next_sample = time.monotonic()
            while not self._stop.is_set():
                with self._lock:
                    writer = self._writer
                    action = self._action_joint_velocity.copy()
                    gripper_position = self._gripper_position
                    action_gripper_position = self._action_gripper_position
                if writer is None:
                    next_sample = time.monotonic()
                    self._stop.wait(0.02)
                    continue

                now = time.monotonic()
                if now < next_sample:
                    self._stop.wait(next_sample - now)
                    continue
                next_sample = max(next_sample + period, now)
                state = client.get_joint_states()
                if self.enable_gripper:
                    maximum_width = 0.085 if self.gripper_type == "robotiq" else 0.08
                    gripper_position = float(np.clip(float(state["gripper_state"]) / maximum_width, 0.0, 1.0))
                camera_frames = self.cameras.snapshot()
                captured = time.monotonic()
                with self._lock:
                    if self._writer is writer:
                        writer.add_sample(
                            captured,
                            state,
                            action,
                            gripper_position,
                            action_gripper_position,
                            camera_frames,
                        )
        except Exception as error:  # noqa: BLE001 - propagate any recorder-thread failure to the control thread
            with self._lock:
                self._error = error
                writer_to_abort = self._writer
                self._writer = None
            self._ready.set()
        finally:
            if writer_to_abort is not None:
                writer_to_abort.abort(str(self._error))
            if client is not None:
                client.close()

    def close(self) -> Path | None:
        completed: Path | None = None
        if self.active:
            completed = self.stop_episode()
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        return completed
