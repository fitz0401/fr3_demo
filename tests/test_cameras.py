import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fr3_demo.cameras import CameraFrame, RealSenseCamera, RealSensePair


class _FakeCamera:
    def __init__(self, image: np.ndarray) -> None:
        self.frame = CameraFrame(image, 1.0, 2.0, 3)

    def snapshot(self, _max_age: float) -> CameraFrame:
        return self.frame


class LatestImageTest(unittest.TestCase):
    """The preview path must stay cheap: no clock work, no copy, fresh frames only."""

    def _camera(self, age: float) -> RealSenseCamera:
        camera = RealSenseCamera("serial")
        image = np.full((4, 6, 3), 7, dtype=np.uint8)
        camera._latest = CameraFrame(image, time.monotonic() - age, 2.0, 3)
        return camera

    def test_returns_the_latest_frame_without_copying_it(self) -> None:
        camera = self._camera(age=0.0)

        image = camera.latest_image()

        self.assertIs(image, camera._latest.image)

    def test_returns_none_before_the_first_frame(self) -> None:
        self.assertIsNone(RealSenseCamera("serial").latest_image())

    def test_returns_none_for_a_stale_frame(self) -> None:
        self.assertIsNone(self._camera(age=1.0).latest_image(max_age=0.5))

    def test_nearest_uses_mapped_exposure_time_and_never_reuses_a_frame(self) -> None:
        camera = RealSenseCamera("serial", fps=60)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for frame_number in range(1, 5):
            hardware = 1.0 + frame_number * 0.01
            host = 1001.0 + frame_number * 0.01
            camera._global_clock.add(hardware, host + 0.003)
            camera._frames.append(
                CameraFrame(image, host + 0.003, hardware, frame_number, "global_time")
            )

        first = camera.nearest(1001.021, max_delta=0.01)
        second = camera.nearest(1001.031, max_delta=0.01, after_frame_number=first.frame_number)

        self.assertEqual(first.frame_number, 2)
        self.assertEqual(second.frame_number, 3)
        self.assertAlmostEqual(first.alignment_timestamp, 1001.023)
        self.assertIsNot(first.image, image)


class RealSensePairTest(unittest.TestCase):
    def test_optional_camera_start_failure_is_ignored(self) -> None:
        exterior = MagicMock(serial="external", width=424, height=240, fps=30)
        wrist = MagicMock(serial="wrist", width=424, height=240, fps=30)
        optional = MagicMock(serial="optional", width=640, height=480, fps=30)
        optional.start.side_effect = RuntimeError("USB camera unavailable")

        with patch("fr3_demo.cameras.RealSenseCamera", side_effect=[exterior, wrist, optional]):
            pair = RealSensePair("external", "wrist", "optional")
            result = pair.start()

        self.assertIs(result, pair)
        self.assertIsNone(pair.exterior2)
        self.assertIn("USB camera unavailable", pair.optional_camera_error)
        optional.close.assert_called_once_with()
        exterior.close.assert_not_called()
        wrist.close.assert_not_called()

    def test_only_wrist_image_is_rotated_180_degrees(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
        pair = object.__new__(RealSensePair)
        pair.exterior = _FakeCamera(image)
        pair.exterior2 = _FakeCamera(image + 1)
        pair.wrist = _FakeCamera(image)
        pair.wrist_rotate_180 = True

        frames = pair.snapshot()

        np.testing.assert_array_equal(frames["exterior_image_left"].image, image)
        np.testing.assert_array_equal(frames["exterior_image_2_left"].image, image + 1)
        np.testing.assert_array_equal(frames["wrist_image"].image, image[::-1, ::-1])
        self.assertEqual(frames["wrist_image"].frame_number, 3)

    def test_optional_runtime_disconnect_keeps_required_frames(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        optional = MagicMock()
        optional.snapshot.side_effect = RuntimeError("camera disconnected")
        pair = object.__new__(RealSensePair)
        pair.exterior = _FakeCamera(image)
        pair.wrist = _FakeCamera(image)
        pair.exterior2 = optional
        pair.wrist_rotate_180 = False
        pair.optional_camera_error = None

        frames = pair.snapshot()

        self.assertEqual(set(frames), {"exterior_image_left", "wrist_image"})
        self.assertIsNone(pair.exterior2)
        self.assertIn("camera disconnected", pair.optional_camera_error)
        optional.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
