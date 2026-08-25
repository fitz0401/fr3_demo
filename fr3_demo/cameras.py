"""Threaded Intel RealSense color-camera capture."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

LOG = logging.getLogger("fr3_teleop.cameras")


def _load_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("RealSense support is not installed; run: pip install -e '.[recording]'") from error
    return rs


def discover_realsense() -> list[dict[str, str]]:
    """Return connected RealSense serial numbers and model names."""

    rs = _load_realsense()
    devices = []
    for device in rs.context().query_devices():
        devices.append(
            {
                "serial": device.get_info(rs.camera_info.serial_number),
                "name": device.get_info(rs.camera_info.name),
            }
        )
    return devices


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    captured_monotonic: float
    hardware_timestamp: float
    frame_number: int


class RealSenseCamera:
    """Continuously capture the latest RGB frame from one RealSense device."""

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._error: BaseException | None = None

    def start(self) -> RealSenseCamera:
        if self._pipeline is not None:
            return self
        rs = _load_realsense()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            pipeline.start(config)
        except RuntimeError as error:
            raise RuntimeError(f"Could not start RealSense {self.serial}: {error}") from error

        self._pipeline = pipeline
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name=f"realsense-{self.serial}", daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = CameraFrame(
                    image=np.asanyarray(color.get_data()).copy(),
                    captured_monotonic=time.monotonic(),
                    hardware_timestamp=float(color.get_timestamp()) / 1000.0,
                    frame_number=int(color.get_frame_number()),
                )
                with self._lock:
                    self._latest = frame
        except RuntimeError as error:
            if not self._stop.is_set():
                with self._lock:
                    self._error = error

    def wait_until_ready(self, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError(f"RealSense {self.serial} failed: {self._error}") from self._error
                if self._latest is not None:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"Timed out waiting for frames from RealSense {self.serial}")

    def snapshot(self, max_age: float = 0.25) -> CameraFrame:
        with self._lock:
            error = self._error
            frame = self._latest
        if error is not None:
            raise RuntimeError(f"RealSense {self.serial} failed: {error}") from error
        if frame is None:
            raise RuntimeError(f"RealSense {self.serial} has not produced a frame")
        age = time.monotonic() - frame.captured_monotonic
        if age > max_age:
            raise RuntimeError(f"RealSense {self.serial} frame is stale ({age:.3f}s)")
        return CameraFrame(frame.image.copy(), frame.captured_monotonic, frame.hardware_timestamp, frame.frame_number)

    def close(self) -> None:
        self._stop.set()
        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError as error:
                LOG.debug("RealSense %s was already stopped: %s", self.serial, error)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._pipeline = None
        self._thread = None

    def __enter__(self) -> RealSenseCamera:  # noqa: PYI034
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()


class RealSensePair:
    """Two required RealSense streams plus an optional second exterior stream."""

    def __init__(
        self,
        exterior_serial: str,
        wrist_serial: str,
        exterior2_serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        exterior2_width: int | None = None,
        exterior2_height: int | None = None,
        exterior2_fps: int | None = None,
        wrist_rotate_180: bool = False,
    ):
        if not exterior_serial or not wrist_serial:
            available = ", ".join(device["serial"] for device in discover_realsense()) or "none"
            raise RuntimeError(
                "Both --external-camera-serial and --wrist-camera-serial are required "
                f"(currently detected: {available})"
            )
        if exterior_serial == wrist_serial:
            raise ValueError("Exterior and wrist camera serial numbers must be different")
        exterior2_serial = exterior2_serial or None
        if exterior2_serial in {exterior_serial, wrist_serial}:
            raise ValueError("Optional exterior camera serial must differ from the required cameras")
        self.exterior = RealSenseCamera(exterior_serial, width, height, fps)
        self.wrist = RealSenseCamera(wrist_serial, width, height, fps)
        self.exterior2 = (
            None
            if exterior2_serial is None
            else RealSenseCamera(
                exterior2_serial,
                exterior2_width or width,
                exterior2_height or height,
                exterior2_fps or fps,
            )
        )
        self.optional_camera_error: str | None = None
        self.wrist_rotate_180 = wrist_rotate_180

    @property
    def serials(self) -> dict[str, str]:
        serials = {"exterior_image_left": self.exterior.serial, "wrist_image": self.wrist.serial}
        exterior2 = getattr(self, "exterior2", None)
        if exterior2 is not None:
            serials["exterior_image_2_left"] = exterior2.serial
        return serials

    @property
    def modes(self) -> dict[str, str]:
        cameras = {"exterior_image_left": self.exterior, "wrist_image": self.wrist}
        exterior2 = getattr(self, "exterior2", None)
        if exterior2 is not None:
            cameras["exterior_image_2_left"] = exterior2
        return {name: f"{camera.width}x{camera.height}@{camera.fps}" for name, camera in cameras.items()}

    def start(self) -> RealSensePair:
        try:
            self.exterior.start()
            self.wrist.start()
            self.exterior.wait_until_ready()
            self.wrist.wait_until_ready()
        except Exception:
            self.close()
            raise
        if self.exterior2 is not None:
            try:
                self.exterior2.start()
                self.exterior2.wait_until_ready()
            except Exception as error:  # noqa: BLE001 - this camera is explicitly optional
                self.optional_camera_error = str(error)
                LOG.warning("Optional exterior camera disabled: %s", error)
                self.exterior2.close()
                self.exterior2 = None
        return self

    def snapshot(self, max_age: float = 0.25) -> dict[str, CameraFrame]:
        wrist = self.wrist.snapshot(max_age)
        if self.wrist_rotate_180:
            wrist = CameraFrame(
                np.rot90(wrist.image, k=2).copy(),
                wrist.captured_monotonic,
                wrist.hardware_timestamp,
                wrist.frame_number,
            )
        frames = {
            "exterior_image_left": self.exterior.snapshot(max_age),
            "wrist_image": wrist,
        }
        exterior2 = getattr(self, "exterior2", None)
        if exterior2 is not None:
            try:
                frames["exterior_image_2_left"] = exterior2.snapshot(max_age)
            except RuntimeError as error:
                self.optional_camera_error = str(error)
                LOG.warning("Optional exterior camera disconnected; disabling it: %s", error)
                exterior2.close()
                self.exterior2 = None
        return frames

    def close(self) -> None:
        exterior2 = getattr(self, "exterior2", None)
        if exterior2 is not None:
            exterior2.close()
        self.wrist.close()
        self.exterior.close()

    def __enter__(self) -> RealSensePair:  # noqa: PYI034
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
