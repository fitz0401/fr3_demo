"""Publish live camera views to ROS 2/RViz from an already-open camera pair."""

from __future__ import annotations

import array
import logging
import os
import subprocess
import threading
import time
from importlib import resources
from typing import Any

import numpy as np

LOG = logging.getLogger("fr3_teleop.preview")

# The panels in config/cameras.rviz subscribe to exactly these topics.
PREVIEW_TOPICS: dict[str, tuple[str, str]] = {
    "exterior_image_left": ("/fr3_demo/exterior_image_left", "exterior_camera"),
    "wrist_image": ("/fr3_demo/wrist_image", "wrist_camera"),
    "exterior_image_2_left": ("/fr3_demo/exterior_image_2_left", "exterior_camera_2"),
}


def rviz_environment() -> dict[str, str]:
    """Remove editor Snap runtime paths that are binary-incompatible with ROS."""

    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("SNAP", "GTK_", "GIO_")) or key == "LD_PRELOAD":
            environment.pop(key, None)
    return environment


def start_rviz() -> subprocess.Popen[Any]:
    """Launch RViz 2 on the shared three-camera layout."""

    config = resources.files("fr3_demo").joinpath("config/cameras.rviz")
    with resources.as_file(config) as config_path:
        return subprocess.Popen(["rviz2", "-d", str(config_path)], env=rviz_environment())


def stop_rviz(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()


def image_message(image_type: Any, node: Any, image: np.ndarray, frame_id: str) -> Any:
    message = image_type()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = frame_id
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = message.width * 3
    # rclpy's generated setter takes an array.array('B') as-is, but validates
    # any other sequence one byte at a time in Python while holding the GIL:
    # ~90 ms for a 960x540 frame, which stalls the recorder's sampler threads.
    message.data = array.array("B", np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    return message


class CameraPreview:
    """Republish a recorder's live frames so RViz can show them during collection.

    Only one process can open a RealSense, so a separate ``fr3-camera-rviz`` and
    a running recorder cannot coexist.  This publisher instead shows the frames
    the recorder itself produced.

    While an episode records, the collector pushes each aligned frame set here
    through :meth:`submit` and the preview touches no camera at all: polling one
    would hold the capture lock that the acquisition thread needs to append the
    next frame, which is enough to make the recorder miss its alignment target.
    Between episodes there is nothing to record, so it falls back to reading the
    latest buffered image.  Publishing always happens on this class's own thread.
    """

    def __init__(
        self,
        cameras: Any,
        *,
        rate_hz: float = 10.0,
        max_age: float = 0.5,
        launch_viewer: bool = True,
        recording_hold: float = 1.0,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("Preview rate must be positive")
        self._cameras = cameras
        self._period = 1.0 / rate_hz
        self._max_age = max_age
        self._launch_viewer = launch_viewer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rviz: subprocess.Popen[Any] | None = None
        self._node: Any | None = None
        self._rclpy: Any | None = None
        self._image_type: Any | None = None
        self._publishers: dict[str, Any] = {}
        self._owns_rclpy = False
        self._reported: set[str] = set()
        self._recording_hold = recording_hold
        self._slot_lock = threading.Lock()
        self._submitted: dict[str, np.ndarray] | None = None
        self._submitted_at = 0.0

    def start(self) -> CameraPreview:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as error:
            raise RuntimeError("ROS 2 is not sourced; run: source /opt/ros/humble/setup.bash") from error

        self._rclpy = rclpy
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self._node = Node("fr3_demo_collect_preview")
            self._image_type = Image
            self._publishers = {
                key: self._node.create_publisher(Image, topic, qos_profile_sensor_data)
                for key, (topic, _) in PREVIEW_TOPICS.items()
            }
            if self._launch_viewer:
                self._rviz = start_rviz()
            self._thread = threading.Thread(target=self._loop, name="collect-preview", daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def submit(self, frames: dict[str, Any]) -> None:
        """Accept the frames the recorder just wrote; called on its thread.

        This only stores references, so it costs the alignment loop nothing.
        The images already carry the recorder's transforms.
        """

        images = {key: frame.image for key, frame in frames.items()}
        with self._slot_lock:
            self._submitted = images
            self._submitted_at = time.monotonic()

    def _report_once(self, key: str, error: BaseException) -> None:
        if key not in self._reported:
            self._reported.add(key)
            LOG.warning("Camera preview is not showing %s: %s", key, error)

    def _images(self) -> dict[str, np.ndarray]:
        """Read the latest frame per camera without touching shared state."""

        images: dict[str, np.ndarray] = {}
        for key, camera in self._cameras.active_cameras.items():
            try:
                image = camera.latest_image(self._max_age)
            except Exception as error:  # noqa: BLE001 - the recorder owns camera health
                self._report_once(key, error)
                continue
            if image is None:
                self._report_once(key, RuntimeError("no fresh frame"))
                continue
            image = self._cameras.transform_image(key, image)
            self._reported.discard(key)
            images[key] = image
        return images

    def _next_images(self) -> dict[str, np.ndarray]:
        """Show the recorder's frames; poll the cameras only when it is idle.

        The newest recorded set is republished until the recorder has been quiet
        for ``recording_hold``.  Polling only between episodes is the point: a
        poll during one competes for the lock that acquisition needs to buffer
        the next frame, which can cost the recorder its alignment target.
        """

        with self._slot_lock:
            submitted = self._submitted
            age = time.monotonic() - self._submitted_at
        if submitted is not None:
            if age <= self._recording_hold:
                self._reported.clear()
                return submitted
            with self._slot_lock:
                if time.monotonic() - self._submitted_at > self._recording_hold:
                    self._submitted = None
        return self._images()

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            for key, image in self._next_images().items():
                _, frame_id = PREVIEW_TOPICS[key]
                try:
                    self._publishers[key].publish(image_message(self._image_type, self._node, image, frame_id))
                except Exception as error:  # noqa: BLE001 - preview loss must not stop recording
                    self._report_once(f"{key} (publish)", error)
            remaining = self._period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        stop_rviz(self._rviz)
        self._rviz = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
            self._owns_rclpy = False

    def __enter__(self) -> CameraPreview:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
