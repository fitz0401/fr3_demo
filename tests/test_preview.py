import array
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fr3_demo.preview import PREVIEW_TOPICS, CameraPreview, image_message, rviz_environment


class FakeCamera:
    def __init__(self, value: int, error: str | None = None) -> None:
        self.value = value
        self.error = error
        self.closed = False
        self.reads = 0

    def latest_image(self, _max_age: float = 0.5) -> np.ndarray | None:
        self.reads += 1
        if self.error is not None:
            raise RuntimeError(self.error)
        image = np.full((4, 6, 3), self.value, dtype=np.uint8)
        image[0, 0] = 255
        return image

    def close(self) -> None:
        self.closed = True


class FakePair:
    def __init__(self, *, exterior2: FakeCamera | None = None, rotated: set[str] | None = None) -> None:
        self.exterior = FakeCamera(10)
        self.wrist = FakeCamera(20)
        self.exterior2 = exterior2
        self.rotated = rotated or set()

    @property
    def active_cameras(self) -> dict[str, FakeCamera]:
        cameras = {
            "exterior_image_left": self.exterior,
            "wrist_image": self.wrist,
        }
        if self.exterior2 is not None:
            cameras["exterior_image_2_left"] = self.exterior2
        return cameras

    def transform_image(self, key: str, image: np.ndarray) -> np.ndarray:
        if key in self.rotated:
            return np.rot90(image, k=2).copy()
        return image


class FakeNode:
    def get_clock(self) -> SimpleNamespace:
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))


class FakeImage:
    """Stands in for sensor_msgs.msg.Image without requiring ROS."""

    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None, frame_id=None)
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = True
        self.step = 0
        self.data = None


class ImageMessageTest(unittest.TestCase):
    def test_rviz_environment_removes_snap_library_settings(self) -> None:
        source = {
            "PATH": "/opt/ros/humble/bin:/usr/bin",
            "DISPLAY": ":1",
            "SNAP": "/snap/code/current",
            "SNAP_LIBRARY_PATH": "/snap/core20/lib",
            "GTK_PATH": "/snap/code/gtk",
            "GIO_MODULE_DIR": "/snap/code/gio",
            "LD_PRELOAD": "/snap/core20/libpthread.so.0",
        }
        with patch.dict(os.environ, source, clear=True):
            environment = rviz_environment()

        self.assertEqual(environment, {"PATH": source["PATH"], "DISPLAY": ":1"})

    def test_pixels_use_the_rclpy_fast_path(self) -> None:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

        message = image_message(FakeImage, FakeNode(), image, "wrist_camera")

        # rclpy's generated setter accepts an array.array('B') as-is; every
        # other sequence is validated one byte at a time in Python with the GIL
        # held (~90 ms for a 960x540 frame), which starves the sampler threads.
        self.assertIsInstance(message.data, array.array)
        self.assertEqual(message.data.typecode, "B")
        self.assertEqual(message.data.tobytes(), image.tobytes())
        self.assertEqual((message.height, message.width, message.step), (2, 3, 9))
        self.assertEqual(message.encoding, "rgb8")
        self.assertFalse(message.is_bigendian)
        self.assertEqual(message.header.frame_id, "wrist_camera")

    def test_a_rotated_view_is_made_contiguous(self) -> None:
        rotated = np.rot90(np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3), k=2)
        self.assertFalse(rotated.flags["C_CONTIGUOUS"])

        message = image_message(FakeImage, FakeNode(), rotated, "wrist_camera")

        self.assertEqual(message.data.tobytes(), np.ascontiguousarray(rotated).tobytes())


class CameraPreviewTest(unittest.TestCase):
    def test_publishes_every_active_camera_on_the_rviz_topics(self) -> None:
        pair = FakePair(exterior2=FakeCamera(30))
        preview = CameraPreview(pair, launch_viewer=False)

        images = preview._images()

        self.assertEqual(set(images), set(PREVIEW_TOPICS))
        self.assertEqual(images["exterior_image_left"][1, 1, 0], 10)
        self.assertEqual(images["wrist_image"][1, 1, 0], 20)
        self.assertEqual(images["exterior_image_2_left"][1, 1, 0], 30)

    def test_omits_the_optional_camera_when_it_is_not_running(self) -> None:
        preview = CameraPreview(FakePair(), launch_viewer=False)

        self.assertEqual(set(preview._images()), {"exterior_image_left", "wrist_image"})

    def test_applies_each_camera_rotation(self) -> None:
        pair = FakePair(
            exterior2=FakeCamera(30),
            rotated={"wrist_image", "exterior_image_2_left"},
        )
        preview = CameraPreview(pair, launch_viewer=False)

        images = preview._images()

        # The marked corner moves to the opposite corner under a 180° rotation.
        self.assertEqual(images["wrist_image"][-1, -1, 0], 255)
        self.assertEqual(images["wrist_image"][0, 0, 0], 20)
        self.assertEqual(images["exterior_image_2_left"][-1, -1, 0], 255)
        self.assertEqual(images["exterior_image_left"][0, 0, 0], 255)

    def test_a_failing_camera_is_skipped_without_disabling_it(self) -> None:
        pair = FakePair(exterior2=FakeCamera(30, error="frame is stale"))
        preview = CameraPreview(pair, launch_viewer=False)

        images = preview._images()

        self.assertNotIn("exterior_image_2_left", images)
        self.assertIn("exterior_image_left", images)
        # The recorder owns camera health: a preview read must never close or
        # detach a camera the collector is still recording from.
        self.assertFalse(pair.exterior2.closed)
        self.assertIsNotNone(pair.exterior2)

    def test_a_camera_failure_is_reported_once_and_rearmed_after_recovery(self) -> None:
        camera = FakeCamera(30, error="frame is stale")
        pair = FakePair(exterior2=camera)
        preview = CameraPreview(pair, launch_viewer=False)

        preview._images()
        preview._images()
        self.assertEqual(preview._reported, {"exterior_image_2_left"})

        camera.error = None
        preview._images()

        self.assertEqual(preview._reported, set())

    def test_a_camera_with_no_fresh_frame_is_skipped(self) -> None:
        class Idle(FakeCamera):
            def latest_image(self, _max_age: float = 0.5) -> np.ndarray | None:
                return None

        pair = FakePair()
        pair.wrist = Idle(20)
        preview = CameraPreview(pair, launch_viewer=False)

        self.assertEqual(set(preview._images()), {"exterior_image_left"})

    def test_recorded_frames_are_shown_instead_of_polling_the_cameras(self) -> None:
        pair = FakePair()
        preview = CameraPreview(pair, launch_viewer=False)
        recorded = np.full((4, 6, 3), 99, dtype=np.uint8)

        preview.submit({"wrist_image": SimpleNamespace(image=recorded)})
        images = preview._next_images()

        self.assertIs(images["wrist_image"], recorded)
        # An episode is in flight: reading a camera here would contend with the
        # acquisition thread for the very lock it needs to buffer the next frame.
        self.assertEqual(pair.exterior.reads, 0)
        self.assertEqual(pair.wrist.reads, 0)

    def test_keeps_showing_the_last_recorded_frames_between_submissions(self) -> None:
        pair = FakePair()
        preview = CameraPreview(pair, launch_viewer=False)
        recorded = np.full((4, 6, 3), 99, dtype=np.uint8)

        preview.submit({"wrist_image": SimpleNamespace(image=recorded)})
        preview._next_images()
        images = preview._next_images()

        # The preview publishes faster than the recorder produces frames; it
        # must republish rather than fall back to polling mid-episode.
        self.assertIs(images["wrist_image"], recorded)
        self.assertEqual(pair.wrist.reads, 0)

    def test_polls_the_cameras_once_the_recorder_stops_submitting(self) -> None:
        pair = FakePair()
        preview = CameraPreview(pair, launch_viewer=False, recording_hold=0.0)

        preview.submit({"wrist_image": SimpleNamespace(image=np.zeros((4, 6, 3), np.uint8))})
        images = preview._next_images()

        self.assertEqual(set(images), {"exterior_image_left", "wrist_image"})
        self.assertEqual(pair.wrist.reads, 1)

    def test_submitted_frames_keep_the_recorder_transforms(self) -> None:
        pair = FakePair(rotated={"wrist_image"})
        preview = CameraPreview(pair, launch_viewer=False)
        recorded = np.full((4, 6, 3), 99, dtype=np.uint8)
        recorded[0, 0] = 255

        preview.submit({"wrist_image": SimpleNamespace(image=recorded)})

        # The recorder already rotated this image; rotating again would show a
        # different picture from the one on disk.
        self.assertIs(preview._next_images()["wrist_image"], recorded)

    def test_rate_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            CameraPreview(FakePair(), rate_hz=0.0)


if __name__ == "__main__":
    unittest.main()
