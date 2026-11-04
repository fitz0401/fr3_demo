import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fr3_demo.gel import (
    GelDirections,
    GelLimits,
    GelMonitor,
    GelReading,
    GelStreamGuard,
    fit_calibration,
    gel_guard_settings,
    limit_twist,
)
from fr3_demo.guide import GuideHeartbeat
from fr3_demo.tactile import GelSightCamera, TactileFrame

GRID = np.array([[x, y] for x in range(20, 300, 30) for y in range(20, 220, 30)], dtype=float)


def _limits() -> GelLimits:
    return GelLimits(shift_px=20.0, twist_rad=0.1, press=0.02)


def _directions() -> GelDirections:
    # image x -> EE +x, image y -> EE -z; pressing -> EE +y.
    return GelDirections(np.array([[0.5, 0.0], [0.0, 0.0], [0.0, -0.5]]), np.array([0.0, 20.0, 0.0]))


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


def _zeroed_monitor(tracker: FakeTracker, directions: GelDirections | None = None) -> GelMonitor:
    monitor = GelMonitor(_limits(), directions or _directions(), tracker=tracker, settle_frames=3)
    for number in range(5):
        monitor.on_frame(_frame(number))
    assert monitor.zeroed
    return monitor


class GelSettingsTest(unittest.TestCase):
    def test_limits_come_first_and_directions_are_optional(self) -> None:
        limits, directions = gel_guard_settings({"enabled": True})
        self.assertEqual(limits, GelLimits())
        self.assertIsNone(directions)
        limits, directions = gel_guard_settings(
            {
                "shift_limit_px": 12.0,
                "twist_limit_deg": 3.0,
                "press_limit_percent": 1.0,
                "shear_to_force": [[1, 0], [0, 0], [0, 1]],
            }
        )
        self.assertEqual(limits.shift_px, 12.0)
        self.assertAlmostEqual(limits.twist_rad, np.radians(3.0))
        self.assertAlmostEqual(limits.press, 0.01)
        np.testing.assert_allclose(directions.twist_to_force, np.zeros(3))
        with self.assertRaises(ValueError):
            GelDirections(np.zeros((3, 2)), np.zeros(3))
        with self.assertRaises(ValueError):
            GelLimits(shift_px=0.0)

    def test_level_is_the_worst_of_shift_twist_and_pressing(self) -> None:
        limits = _limits()
        self.assertAlmostEqual(limits.level(_estimate(shear=(12.0, 16.0))), 1.0)  # a 20 px shift
        self.assertAlmostEqual(limits.level(_estimate(twist=-0.05)), 0.5)
        self.assertAlmostEqual(limits.level(_estimate(spread=0.01)), 0.5)
        self.assertAlmostEqual(limits.level(_estimate(spread=-0.01)), 0.0)  # relieved, not pressed

    def test_direction_is_the_fit_s_load_direction_whatever_its_size(self) -> None:
        directions = _directions()
        np.testing.assert_allclose(directions.direction(_estimate(shear=(8.0, 6.0))), [0.8, 0.0, -0.6])
        np.testing.assert_allclose(directions.direction(_estimate(shear=(80.0, 60.0))), [0.8, 0.0, -0.6])


class GelMonitorTest(unittest.TestCase):
    def test_zeroes_only_once_the_gel_has_settled(self) -> None:
        tracker = FakeTracker()
        monitor = GelMonitor(_limits(), tracker=tracker, settle_frames=3)
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
        tracker.estimate = _estimate(shear=(16.0, 0.0), twist=0.01)  # 16 of 20 px, along EE +x
        monitor.on_frame(_frame(10))
        reading = monitor.latest()
        self.assertTrue(reading.valid)
        self.assertAlmostEqual(reading.level, 0.8)
        np.testing.assert_allclose(reading.direction_ee, [1.0, 0.0, 0.0])
        self.assertEqual(reading.shift_px, (16.0, 0.0))
        self.assertEqual(reading.twist_rad, 0.01)
        tracker.estimate = _estimate(shear=(0.4, 0.0))  # barely loaded: too little to say which way
        monitor.on_frame(_frame(11))
        self.assertIsNone(monitor.latest().direction_ee)
        self.assertGreater(monitor.latest().seq, reading.seq)

    def test_without_directions_the_level_still_guards(self) -> None:
        tracker = FakeTracker()
        monitor = GelMonitor(_limits(), tracker=tracker, settle_frames=3)
        for number in range(5):
            monitor.on_frame(_frame(number))
        tracker.estimate = _estimate(shear=(0.0, 30.0))
        monitor.on_frame(_frame(10))
        self.assertAlmostEqual(monitor.latest().level, 1.5)
        self.assertIsNone(monitor.latest().direction_ee)

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
        tracker.estimate = _estimate(shear=(21.0, 0.0))  # past the 20 px limit
        monitor.on_frame(_frame(11))
        self.assertIn("level 1.05 (shift 21.0 px", guard.check())
        tracker.estimate = _estimate(valid=False, reason="lost track")
        monitor.on_frame(_frame(12))
        self.assertIn("lost track", guard.check())
        monitor.suspend("the gripper is moving")
        self.assertIsNone(guard.check())  # Bamboo's limits cover the grace period
        time.sleep(0.06)
        self.assertIn("did not re-zero", guard.check())


class LimitTwistTest(unittest.TestCase):
    def _reading(self, level: float, valid: bool = True, zeroing: bool = False) -> GelReading:
        return GelReading(1, time.monotonic(), valid, "", level, (0.0, 0.0, -1.0), zeroing)

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
        self.reading = GelReading(7, time.monotonic(), True, "", 0.6, (0.0, 1.0, 0.0))

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
        gel.reading = GelReading(8, time.monotonic(), False, "lost track", 0.0, None)
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
        monitor = GelMonitor(_limits(), _directions(), settle_frames=3)
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


def _synthetic_run(
    seed: int = 0, lag_s: float = 0.06, directions: int = 3, tracked: float = 1.0, z_seen: bool = True
) -> dict:
    """A calibration run: pushes held a second or two, a gel lagging Franka by lag_s."""

    rng = np.random.default_rng(seed)
    force_time = np.arange(0.0, 40.0, 0.005)
    force = np.zeros((len(force_time), 3))
    for start in np.arange(1.0, 38.0, 2.5):
        axis = rng.integers(0, directions)
        amplitude = rng.uniform(3.0, 9.0) * rng.choice([-1.0, 1.0])
        ramp = np.clip((force_time - start) / 0.4, 0.0, 1.0) * np.clip((start + 1.8 - force_time) / 0.4, 0.0, 1.0)
        force[:, axis] += amplitude * ramp
    frame_time = np.arange(0.0, 40.0, 1.0 / 18.0)
    seen = np.stack([np.interp(frame_time - lag_s, force_time, force[:, axis]) for axis in range(3)], axis=1)
    # Image x follows EE x, image y follows EE -z; sideways (EE x) pushes also twist the gel.
    along_z = -1.6 * seen[:, 2] if z_seen else np.zeros(len(frame_time))  # or through a hard finger
    shear = np.stack((2.0 * seen[:, 0], along_z), axis=1) + rng.normal(0.0, 0.05, (len(frame_time), 2))
    twist = 0.01 * seen[:, 0] + rng.normal(0.0, 0.0005, len(frame_time))
    spread = 0.004 * seen[:, 1] + rng.normal(0.0, 0.0002, len(frame_time))
    valid = rng.uniform(size=len(frame_time)) < tracked
    return {
        "frame_time": frame_time,
        "valid": valid,
        "reason": np.where(valid, "", "lost track (markers moved 9.0 px in one frame); unload the gel or re-zero"),
        "shear_px": shear,
        "twist_rad": twist,
        "spread": spread,
        "force_time": force_time,
        "force_ee": force + rng.normal(0.0, 0.2, force.shape),
        "tare_n": np.zeros(3),
    }


class CalibrationRunTest(unittest.TestCase):
    def test_the_fit_uses_twist_and_reports_each_axis(self) -> None:
        run = _synthetic_run()
        from fr3_demo.gel import align_calibration

        lag, fit = align_calibration(run)
        self.assertAlmostEqual(lag, -0.06, delta=0.011)  # the gel lags Franka by 60 ms
        self.assertGreater(fit["r_squared"], 0.95)
        self.assertTrue(np.all(fit["axis_r_squared"] > 0.9))
        # A sideways push shows as both shift and twist; the fit may use either, but must predict it.
        directions = GelDirections(fit["shear_to_force"], fit["spread_to_force"], fit["twist_to_force"])
        estimate = _estimate(shear=(10.0, 0.0), twist=0.05)  # 5 N along EE x
        np.testing.assert_allclose(directions.load(estimate), [5.0, 0.0, 0.0], atol=0.3)

    def test_report_accepts_a_good_run_and_explains_a_bad_one(self) -> None:
        from fr3_demo.gel import report_calibration

        usable, lines = report_calibration(_synthetic_run())
        self.assertTrue(usable)
        self.assertEqual(
            [line.split(" = ")[0] for line in lines], ["shear_to_force", "spread_to_force", "twist_to_force"]
        )
        usable, lines = report_calibration(_synthetic_run(directions=1))
        self.assertFalse(usable)
        self.assertEqual(lines, [])
        usable, _ = report_calibration(_synthetic_run(tracked=0.3))
        self.assertFalse(usable)

    def test_force_that_goes_around_the_gel_is_reported_not_refused(self) -> None:
        from fr3_demo.gel import report_calibration

        usable, lines = report_calibration(_synthetic_run(z_seen=False))
        self.assertTrue(usable)
        self.assertEqual(len(lines), 3)

    def test_the_twist_term_is_optional_in_the_config(self) -> None:
        _, old = gel_guard_settings({"shear_to_force": [[0.5, 0.0], [0.0, 0.0], [0.0, -0.5]]})
        np.testing.assert_allclose(old.twist_to_force, np.zeros(3))
        _, new = gel_guard_settings(
            {"shear_to_force": [[0.5, 0.0], [0.0, 0.0], [0.0, -0.5]], "twist_to_force": [100.0, 0.0, 0.0]}
        )
        np.testing.assert_allclose(new.load(_estimate(twist=0.02)), [2.0, 0.0, 0.0])
