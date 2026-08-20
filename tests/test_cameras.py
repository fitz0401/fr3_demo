import unittest

import numpy as np

from fr3_demo.cameras import CameraFrame, RealSensePair


class _FakeCamera:
    def __init__(self, image: np.ndarray) -> None:
        self.frame = CameraFrame(image, 1.0, 2.0, 3)

    def snapshot(self, _max_age: float) -> CameraFrame:
        return self.frame


class RealSensePairTest(unittest.TestCase):
    def test_only_wrist_image_is_rotated_180_degrees(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
        pair = object.__new__(RealSensePair)
        pair.exterior = _FakeCamera(image)
        pair.wrist = _FakeCamera(image)
        pair.wrist_rotate_180 = True

        frames = pair.snapshot()

        np.testing.assert_array_equal(frames["exterior_image_left"].image, image)
        np.testing.assert_array_equal(frames["wrist_image"].image, image[::-1, ::-1])
        self.assertEqual(frames["wrist_image"].frame_number, 3)


if __name__ == "__main__":
    unittest.main()
