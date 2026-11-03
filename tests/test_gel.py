import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fr3_demo.gel import (
    GelCalibration,
    GelMonitor,
    GelReading,
    GelStreamGuard,
    fit_calibration,
    limit_twist,
)
from fr3_demo.guide import GuideHeartbeat
from fr3_demo.tactile import GelSightCamera, TactileFrame

GRID = np.array([[x, y] for x in range(20, 300, 30) for y in range(20, 220, 30)], dtype=float)


def _calibration(limit_n: float = 10.0) -> GelCalibration:
    # 0.5 N per px: image x -> EE +x, image y -> EE -z; pressing -> EE +y.
    shear = np.array([[0.5, 0.0], [0.0, 0.0], [0.0, -0.5]])
    return GelCalibration(shear, np.array([0.0, 20.0, 0.0]), limit_n=limit_n, twist_limit_rad=0.1)


def _estimate(shear=(0.0, 0.0), twist=0.0, spread=0.0, valid=True, reason=""):
    return SimpleNamespace(shear_px=shear, twist_rad=twist, spread=spread, valid=valid, reason=reason)


class FakeTracker:
    """Stands in for MarkerShearTracker: fixed markers, scripted estimates."""

    def __init__(self) -> None:
        self.markers = GRID.copy()
        self.estimate = _estimate()
        self.references = 0
        self.fail = False

    def detect(self, image: np.ndarray) -> np.ndarray:
        return self.markers.copy()

    def set_reference(self, image: np.ndarray) -> int:
        self.references += 1
        return len(self.markers)

    def update(self, image: np.ndarray, timestamp: float, frame_number: int):
        if self.fail:
            raise RuntimeError("decoder exploded")
        return self.estimate


def _frame(number: int) -> TactileFrame:
    return TactileFrame(
        image=np.zeros((240, 320, 3), np.uint8), captured_monotonic=time.monotonic(), frame_number=number
    )


def _zeroed_monitor(tracker: FakeTracker, **kwargs) -> GelMonitor:
    monitor = GelMonitor(_calibration(**kwargs), tracker=tracker, settle_frames=3)
    for number in range(5):
        monitor.on_frame(_frame(number))
    assert monitor.zeroed
    return monitor


class GelCalibrationTest(unittest.TestCase):
    def test_config_without_a_fit_explains_how_to_get_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "fr3-gel-calibrate"):
            GelCalibration.from_config({"enabled": True, "limit_n": 10.0})
        with self.assertRaises(ValueError):
            GelCalibration(np.zeros((3, 2)), np.zeros(3), 10.0, 0.1)

    def test_level_is_the_worse_of_force_and_twist(self) -> None:
        calibration = _calibration(limit_n=10.0)
        level, force = calibration.level(_estimate(shear=(8.0, 6.0)))  # 4 N along x, -3 N along z
        np.testing.assert_allclose(force, [4.0, 0.0, -3.0])
        self.assertAlmostEqual(level, 0.5)
        level, _ = calibration.level(_estimate(twist=0.08))
        self.assertAlmostEqual(level, 0.8)


class GelMonitorTest(unittest.TestCase):
    def test_zeroes_only_once_the_gel_has_settled(self) -> None:
        tracker = FakeTracker()
        monitor = GelMonitor(_calibration(), tracker=tracker, settle_frames=3)
        rng = np.random.default_rng(0)
        for number in range(6):  # still moving: markers jump about 2 px per frame
            tracker.markers = GRID + rng.normal(0.0, 2.0, GRID.shape)
            monitor.on_frame(_frame(number))
        self.assertFalse(monitor.zeroed)
        self.assertFalse(monitor.latest().valid)
        tracker.markers = GRID.copy()
        for number in range(6, 11):
            monitor.on_frame(_frame(number))
        self.assertTrue(monitor.zeroed)
        self.assertEqual(tracker.references, 1)
        self.assertTrue(monitor.latest().valid)

    def test_reading_carries_level_and_loading_direction(self) -> None:
        tracker = FakeTracker()
        monitor = _zeroed_monitor(tracker)
        tracker.estimate = _estimate(shear=(16.0, 0.0))  # 8 N along EE +x
        monitor.on_frame(_frame(10))
        reading = monitor.latest()
        self.assertTrue(reading.valid)
        self.assertAlmostEqual(reading.level, 0.8)
        np.testing.assert_allclose(reading.direction_ee, [1.0, 0.0, 0.0])
        tracker.estimate = _estimate(shear=(0.4, 0.0))  # 0.2 N: too small to say which way
        monitor.on_frame(_frame(11))
        self.assertIsNone(monitor.latest().direction_ee)
        self.assertGreater(monitor.latest().seq, reading.seq)

    def test_fails_closed(self) -> None:
        tracker = FakeTracker()
        monitor = _zeroed_monitor(tracker)
        tracker.estimate = _estimate(valid=False, reason="lost track")
        monitor.on_frame(_frame(10))
        self.assertFalse(monitor.latest().valid)
        self.assertIn("lost track", monitor.latest().reason)
        tracker.fail = True
        monitor.on_frame(_frame(11))  # must not raise into the capture thread
        self.assertIn("decoder exploded", monitor.latest().reason)
        tracker.fail = False
        tracker.estimate = _estimate()
        monitor.on_frame(_frame(12))
        self.assertTrue(monitor.latest().valid)
        self.assertFalse(monitor.latest(max_age=-1.0).valid)  # stale frames

    def test_gripper_moves_suspend_it_and_it_rezeroes_after(self) -> None:
        tracker = FakeTracker()
        monitor = _zeroed_monitor(tracker)
        monitor.suspend("the gripper is moving")
        monitor.on_frame(_frame(10))
        reading = monitor.latest()
        self.assertFalse(reading.valid)
        self.assertTrue(reading.zeroing)
        self.assertIn("gripper", reading.reason)
        monitor.rezero()
        for number in range(11, 16):
            monitor.on_frame(_frame(number))
        self.assertTrue(monitor.zeroed)
        self.assertEqual(tracker.references, 2)


class GelStreamGuardTest(unittest.TestCase):
    def test_yields_at_the_limit_or_on_a_bad_signal_but_not_while_rezeroing(self) -> None:
        tracker = FakeTracker()
        monitor = _zeroed_monitor(tracker)
        guard = GelStreamGuard(monitor, zero_grace_s=0.05)
        tracker.estimate = _estimate(shear=(10.0, 0.0))
        monitor.on_frame(_frame(10))
        self.assertIsNone(guard.check())
        tracker.estimate = _estimate(shear=(21.0, 0.0))  # 10.5 N
        monitor.on_frame(_frame(11))
        self.assertIn("level 1.05", guard.check())
        tracker.estimate = _estimate(valid=False, reason="lost track")
        monitor.on_frame(_frame(12))
        self.assertIn("lost track", guard.check())
        monitor.suspend("the gripper is moving")
        self.assertIsNone(guard.check())  # Bamboo's limits cover the grace period
        time.sleep(0.06)
        self.assertIn("did not re-zero", guard.check())


class LimitTwistTest(unittest.TestCase):
    def _reading(self, level: float, valid: bool = True, zeroing: bool = False) -> GelReading:
        return GelReading(1, time.monotonic(), valid, "", level, (0.0, 0.0, -level), (0.0, 0.0, -1.0), zeroing)

    def test_blocks_only_motion_that_loads_the_gel(self) -> None:
        rotation = np.eye(3)
        twist = np.array([0.02, 0.0, -0.05, 0.0, 0.0, 0.3])
        np.testing.assert_allclose(limit_twist(twist, self._reading(0.3), rotation), twist)
        at_limit = limit_twist(twist, self._reading(1.0), rotation)
        np.testing.assert_allclose(at_limit, [0.02, 0.0, 0.0, 0.0, 0.0, 0.0])  # no further down, no swinging
        backing_off = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(limit_twist(backing_off, self._reading(1.0), rotation), backing_off)
        halfway = limit_twist(twist, self._reading(0.75), rotation)
        self.assertAlmostEqual(halfway[2], -0.025)

    def test_direction_is_rotated_into_the_base_frame(self) -> None:
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # tool pointing down
        twist = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])  # base +z is EE -z: the loading direction
        np.testing.assert_allclose(limit_twist(twist, self._reading(1.0), rotation), np.zeros(6))

    def test_rezeroing_stops_and_a_bad_signal_slows(self) -> None:
        twist = np.full(6, 0.1)
        np.testing.assert_allclose(limit_twist(twist, self._reading(0.0, False, True), np.eye(3)), np.zeros(6))
        np.testing.assert_allclose(limit_twist(twist, self._reading(0.0, False), np.eye(3)), 0.25 * twist)


class FitCalibrationTest(unittest.TestCase):
    def test_recovers_a_known_map_from_noisy_pushes(self) -> None:
        rng = np.random.default_rng(3)
        true_shear = np.array([[0.4, 0.05], [0.0, 0.0], [-0.02, -0.6]])
        true_spread = np.array([0.0, 25.0, 0.0])
        shear = rng.uniform(-15, 15, (400, 2))
        spread = rng.uniform(0.0, 0.1, 400)
        force = shear @ true_shear.T + spread[:, None] * true_spread + rng.normal(0.0, 0.2, (400, 3))
        result = fit_calibration(shear, spread, force)
        np.testing.assert_allclose(result["shear_to_force"], true_shear, atol=0.01)
        np.testing.assert_allclose(result["spread_to_force"], true_spread, atol=1.0)
        self.assertGreater(result["r_squared"], 0.95)
        self.assertGreater(result["push_directions"][1], 1.0)
        with self.assertRaises(ValueError):
            fit_calibration(shear[:5], spread[:5], force[:5])


class FakeGel:
    def __init__(self) -> None:
        self.reading = GelReading(7, time.monotonic(), True, "", 0.6, (0.0, 3.0, 0.0), (0.0, 1.0, 0.0))

    def latest(self) -> GelReading:
        return self.reading


class FakeClient:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.closed = False

    def update_guiding(self, **kwargs) -> dict:
        self.updates.append(kwargs)
        return {"success": True}

    def close(self) -> None:
        self.closed = True


class GelHeartbeatTest(unittest.TestCase):
    def test_updates_carry_the_gel_reading(self) -> None:
        client, gel = FakeClient(), FakeGel()
        heartbeat = GuideHeartbeat(lambda: client, rate_hz=200.0, gel=gel)
        heartbeat.start()
        deadline = time.monotonic() + 2.0
        while len(client.updates) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        gel.reading = GelReading(8, time.monotonic(), False, "lost track", 0.0, (0.0, 0.0, 0.0), None)
        while not any(not update["valid"] for update in client.updates) and time.monotonic() < deadline:
            time.sleep(0.005)
        heartbeat.stop()
        first = client.updates[0]
        self.assertEqual(first, {"level": 0.6, "valid": True, "direction": [0.0, 1.0, 0.0], "seq": 7})
        invalid = next(update for update in client.updates if not update["valid"])
        self.assertEqual(invalid, {"level": 0.0, "valid": False, "direction": None, "seq": 8})


class FakeCapture:
    def __init__(self) -> None:
        self.image = np.zeros((240, 320, 3), np.uint8)

    def grab(self) -> bool:
        return True

    def retrieve(self):
        return True, self.image

    def release(self) -> None:
        pass


class ExtraObserverTest(unittest.TestCase):
    def test_a_failing_extra_observer_does_not_stop_capture(self) -> None:
        seen: list[int] = []

        def broken(frame: TactileFrame) -> None:
            raise RuntimeError("gel processing bug")

        with patch.object(GelSightCamera, "_open", staticmethod(lambda index: FakeCapture())):
            camera = GelSightCamera(device=0, fps=200.0, on_frame=lambda frame: seen.append(frame.frame_number))
            camera.add_frame_observer(broken)
            camera.start()
            deadline = time.monotonic() + 2.0
            while len(seen) < 5 and time.monotonic() < deadline:
                time.sleep(0.005)
            camera.close()
        self.assertGreaterEqual(len(seen), 5)
        self.assertGreaterEqual(camera.extra_observer_errors, 5)


if __name__ == "__main__":
    unittest.main()


def _marker_image(shift: tuple[float, float] = (0.0, 0.0), seed: int = 0) -> np.ndarray:
    """A synthetic marker gel: dark dots on an LED-graded background."""

    import cv2

    rng = np.random.default_rng(seed)
    height, width = 240, 320
    y, x = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), np.float32)
    image[..., 0] = 170 + 50 * x / width
    image[..., 1] = 190 - 30 * y / height
    image[..., 2] = 180 + 20 * (x + y) / (width + height)
    jitter = np.random.default_rng(1).normal(0.0, 0.6, (12, 9, 2))
    for row in range(9):
        for column in range(12):
            cx = 22 + 25 * column + jitter[column, row, 0] + shift[0]
            cy = 20 + 25 * row + jitter[column, row, 1] + shift[1]
            cv2.circle(image, (int(np.rint(cx * 16)), int(np.rint(cy * 16))), 4 * 16, (40, 40, 40), -1, cv2.LINE_AA, 4)
    image = cv2.GaussianBlur(image, (3, 3), 0.8) + rng.normal(0.0, 1.5, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class GelMonitorWithTrackerTest(unittest.TestCase):
    def test_a_sheared_gel_reads_its_load_and_direction(self) -> None:
        monitor = GelMonitor(_calibration(limit_n=10.0), settle_frames=3)
        number = 0
        for _ in range(6):
            monitor.on_frame(TactileFrame(_marker_image(seed=number), time.monotonic(), number))
            number += 1
        self.assertTrue(monitor.zeroed, monitor.latest().reason)
        # Shear the dots 16 px along image +x in 1 px steps: 8 N along EE +x.
        for step in range(1, 17):
            monitor.on_frame(TactileFrame(_marker_image((float(step), 0.0), seed=number), time.monotonic(), number))
            number += 1
        reading = monitor.latest()
        self.assertTrue(reading.valid, reading.reason)
        self.assertAlmostEqual(reading.level, 0.8, delta=0.05)
        np.testing.assert_allclose(reading.direction_ee, [1.0, 0.0, 0.0], atol=0.05)
