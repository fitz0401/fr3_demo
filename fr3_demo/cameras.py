"""Threaded Intel RealSense color-camera capture."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fr3_demo.synchronization import AffineClockMapper

LOG = logging.getLogger("fr3_teleop.cameras")
GLOBAL_TIMESTAMP_DOMAINS = frozenset({"global_time", "system_time"})


def _shared_clock_mapper() -> AffineClockMapper:
    return AffineClockMapper(
        max_samples=1800,
        lower_envelope=True,
        fit_scale=False,
        reset_on_discontinuity=False,
    )


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
    timestamp_domain: str = "unknown"
    synchronized_monotonic: float | None = None

    @property
    def alignment_timestamp(self) -> float:
        """Best estimate of exposure time on the host monotonic clock."""

        return self.captured_monotonic if self.synchronized_monotonic is None else self.synchronized_monotonic


class RealSenseCamera:
    """Continuously capture the latest RGB frame from one RealSense device."""

    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        global_clock: AffineClockMapper | None = None,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        # Alignment only reaches ~100 ms back, so one second is ample. At
        # 60 fps a 10-second ring would hold ~900 MB for a 960x540 camera and
        # make every nearest() call scan 600 frames.
        self._frames: deque[CameraFrame] = deque(maxlen=max(60, fps))
        self._clock_sample_count = max(300, fps * 20)
        self._clock = AffineClockMapper(max_samples=self._clock_sample_count, lower_envelope=True)
        self._global_clock = global_clock or _shared_clock_mapper()
        self._timestamp_domain: str | None = None
        self._error: BaseException | None = None

    def _mapper_for(self, timestamp_domain: str) -> AffineClockMapper:
        return self._global_clock if timestamp_domain in GLOBAL_TIMESTAMP_DOMAINS else self._clock

    def start(self) -> RealSenseCamera:
        if self._pipeline is not None:
            return self
        rs = _load_realsense()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            profile = pipeline.start(config)
        except RuntimeError as error:
            raise RuntimeError(f"Could not start RealSense {self.serial}: {error}") from error

        # This makes supported devices report timestamps in a common system
        # domain.  The affine host mapping below is still retained because USB
        # delivery latency and SDK clock conversion are not deterministic.
        for sensor in profile.get_device().query_sensors():
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
            except RuntimeError as error:
                LOG.warning("Could not enable global timestamps for RealSense %s: %s", self.serial, error)

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
                captured_monotonic = time.monotonic()
                hardware_timestamp = float(color.get_timestamp()) / 1000.0
                try:
                    timestamp_domain = str(color.get_frame_timestamp_domain()).rsplit(".", 1)[-1]
                except (AttributeError, RuntimeError):
                    timestamp_domain = "unknown"
                frame = CameraFrame(
                    image=np.asanyarray(color.get_data()).copy(),
                    captured_monotonic=captured_monotonic,
                    hardware_timestamp=hardware_timestamp,
                    frame_number=int(color.get_frame_number()),
                    timestamp_domain=timestamp_domain,
                )
                with self._lock:
                    if timestamp_domain != self._timestamp_domain:
                        # RealSense global/system timestamps already run at the
                        # host clock rate; only their epoch/transport offset is
                        # unknown. Hardware clocks retain affine drift fitting.
                        self._clock = AffineClockMapper(
                            max_samples=self._clock_sample_count,
                            lower_envelope=True,
                            fit_scale=timestamp_domain not in GLOBAL_TIMESTAMP_DOMAINS,
                        )
                        self._frames.clear()
                        self._timestamp_domain = timestamp_domain
                    self._mapper_for(timestamp_domain).add(hardware_timestamp, captured_monotonic)
                    self._latest = frame
                    self._frames.append(frame)
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

    def latest_image(self, max_age: float = 0.5) -> np.ndarray | None:
        """Return the newest image, or None, without competing with acquisition.

        Preview consumers only need pixels.  Unlike ``snapshot`` this runs no
        clock-mapper estimate under the capture lock and copies nothing: it
        holds the lock just long enough to read a reference to a frame the
        capture thread has already published and never mutates again.
        """

        with self._lock:
            frame = self._latest
        if frame is None or time.monotonic() - frame.captured_monotonic > max_age:
            return None
        return frame.image

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
        with self._lock:
            mapper = self._mapper_for(frame.timestamp_domain)
        synchronized = mapper.to_host(frame.hardware_timestamp)
        return replace(frame, image=frame.image.copy(), synchronized_monotonic=synchronized)

    def nearest(
        self,
        target_monotonic: float,
        max_delta: float,
        after_frame_number: int | None = None,
    ) -> CameraFrame:
        """Return the nearest unused buffered frame on the host timeline."""

        with self._lock:
            error = self._error
            frames = list(self._frames)
            mapper = self._mapper_for(frames[-1].timestamp_domain) if frames else None
        if error is not None:
            raise RuntimeError(f"RealSense {self.serial} failed: {error}") from error
        mapped: list[tuple[CameraFrame, float]] = []
        if mapper is not None:
            estimate = mapper.estimate()
            mapped = [
                (frame, estimate.scale * frame.hardware_timestamp + estimate.offset)
                for frame in frames
            ]
        if after_frame_number is not None:
            mapped = [(frame, host_time) for frame, host_time in mapped if frame.frame_number > after_frame_number]
        if not mapped:
            raise RuntimeError(f"RealSense {self.serial} has no unused buffered frame")
        frame, synchronized = min(mapped, key=lambda item: abs(item[1] - target_monotonic))
        delta = abs(synchronized - target_monotonic)
        if delta > max_delta:
            raise RuntimeError(
                f"RealSense {self.serial} cannot align target: nearest frame is {delta * 1000.0:.1f} ms away"
            )
        return replace(frame, image=frame.image.copy(), synchronized_monotonic=synchronized)

    @property
    def clock_estimate(self) -> dict[str, float | int]:
        with self._lock:
            mapper = self._mapper_for(self._timestamp_domain or "unknown")
        estimate = mapper.estimate()
        return {
            "scale": estimate.scale,
            "offset": estimate.offset,
            "sample_count": estimate.sample_count,
            "residual_p95_ms": estimate.residual_p95 * 1000.0,
        }

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
        global_clock = _shared_clock_mapper()
        self.exterior = RealSenseCamera(exterior_serial, width, height, fps, global_clock)
        self.wrist = RealSenseCamera(wrist_serial, width, height, fps, global_clock)
        self.exterior2 = (
            None
            if exterior2_serial is None
            else RealSenseCamera(
                exterior2_serial,
                exterior2_width or width,
                exterior2_height or height,
                exterior2_fps or fps,
                global_clock,
            )
        )
        self.optional_camera_error: str | None = None
        self.wrist_rotate_180 = wrist_rotate_180

    @property
    def active_cameras(self) -> dict[str, RealSenseCamera]:
        cameras = {
            "exterior_image_left": self.exterior,
            "wrist_image": self.wrist,
        }
        if self.exterior2 is not None:
            cameras["exterior_image_2_left"] = self.exterior2
        return cameras

    def transform_image(self, key: str, image: np.ndarray) -> np.ndarray:
        if key == "wrist_image" and self.wrist_rotate_180:
            return np.rot90(image, k=2).copy()
        return image

    def _transform_frame(self, key: str, frame: CameraFrame) -> CameraFrame:
        image = self.transform_image(key, frame.image)
        if image is not frame.image:
            return replace(frame, image=image)
        return frame

    @property
    def serials(self) -> dict[str, str]:
        return {key: camera.serial for key, camera in self.active_cameras.items()}

    @property
    def modes(self) -> dict[str, str]:
        return {
            key: f"{camera.width}x{camera.height}@{camera.fps}"
            for key, camera in self.active_cameras.items()
        }

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
        frames = {
            "exterior_image_left": self.exterior.snapshot(max_age),
            "wrist_image": self._transform_frame("wrist_image", self.wrist.snapshot(max_age)),
        }
        exterior2 = self.exterior2
        if exterior2 is not None:
            try:
                frames["exterior_image_2_left"] = exterior2.snapshot(max_age)
            except RuntimeError as error:
                self.optional_camera_error = str(error)
                LOG.warning("Optional exterior camera disconnected; disabling it: %s", error)
                exterior2.close()
                self.exterior2 = None
        return frames

    def nearest(
        self,
        target_monotonic: float,
        max_delta: float,
        after_frame_numbers: dict[str, int] | None = None,
    ) -> dict[str, CameraFrame]:
        """Select one unique frame per active camera nearest a host-time target."""

        after_frame_numbers = after_frame_numbers or {}
        selected = {
            key: self._transform_frame(
                key,
                camera.nearest(target_monotonic, max_delta, after_frame_numbers.get(key)),
            )
            for key, camera in self.active_cameras.items()
        }
        return selected

    @property
    def clock_estimates(self) -> dict[str, dict[str, float | int]]:
        return {key: camera.clock_estimate for key, camera in self.active_cameras.items()}

    def close(self) -> None:
        for camera in reversed(tuple(self.active_cameras.values())):
            camera.close()

    def __enter__(self) -> RealSensePair:  # noqa: PYI034
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
