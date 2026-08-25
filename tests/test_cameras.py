import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fr3_demo.cameras import CameraFrame, RealSensePair


class _FakeCamera:
    def __init__(self, image: np.ndarray) -> None:
        self.frame = CameraFrame(image, 1.0, 2.0, 3)

    def snapshot(self, _max_age: float) -> CameraFrame:
        return self.frame


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
