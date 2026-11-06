import unittest

import numpy as np

from fr3_pi05.faults import CAMERA_KEYS, FAULT_PRESETS, FaultSchedule, PourDetector

# A pre-grasp configuration close to the blue-cup demonstrations' start pose.
Q0 = np.array([-0.047, -0.727, -0.028, -2.284, 0.002, 1.575, 0.035])


def run(detector: PourDetector, steps: list[tuple[np.ndarray, float]]) -> list[bool]:
    return [detector.update(q, g) for q, g in steps]


class PourDetectorTest(unittest.TestCase):
    def test_no_pour_without_grasp_even_when_tilted(self) -> None:
        detector = PourDetector()
        tilted = Q0.copy()
        tilted[5] -= 1.4
        self.assertFalse(any(run(detector, [(tilted, 1.0)] * 40)))
        self.assertFalse(detector.pour_started)

    def test_pour_detected_after_grasp_and_tilt_then_cleared_on_release(self) -> None:
        detector = PourDetector(pour_deg=60.0)
        steps = [(Q0, 1.0)] * 10 + [(Q0, 0.784)] * 20  # open, then hold upright for > 1 s
        for k in range(30):  # tilt the wrist (joint 6) by up to 1.4 rad while holding
            q = Q0.copy()
            q[5] -= 1.4 * (k + 1) / 30
            steps.append((q, 0.784))
        out = run(detector, steps)
        self.assertFalse(any(out[:30]), "must not pour while upright")
        self.assertTrue(out[-1], f"tilt {detector.tilt_deg:.1f} deg should count as pouring")
        first = out.index(True)
        self.assertGreaterEqual(detector.history[first][0], 60.0)
        self.assertLess(detector.history[first - 1][0], 60.0)
        self.assertTrue(detector.pour_started)
        # Release: pouring stops, the latch stays.
        detector.update(steps[-1][0], 1.0)
        self.assertFalse(detector.pouring)
        self.assertTrue(detector.pour_started)

    def test_bottle_axis_is_learned_one_second_into_the_grasp(self) -> None:
        detector = PourDetector(control_hz=15.0, grasp_seconds=1.0)
        q = Q0.copy()
        q[5] -= 1.4
        # Grasped while already tilted: the "up" axis is re-chosen from that pose (the tool axis
        # closest to vertical), so the reading stays well below the pour threshold.
        out = run(detector, [(q, 0.784)] * 20)
        self.assertLess(detector.tilt_deg, 45.0)
        self.assertFalse(any(out))
        self.assertEqual(detector.history[13][0], 0.0)  # axis not learned before 15 steps (1 s at 15 Hz)
        self.assertGreater(detector.history[14][0], 0.0)

    def test_rejects_bad_joint_vectors(self) -> None:
        with self.assertRaises(ValueError):
            PourDetector().update(np.zeros(6), 1.0)
        with self.assertRaises(ValueError):
            PourDetector().update(np.full(7, np.nan), 1.0)


class FaultScheduleTest(unittest.TestCase):
    def test_presets_cover_the_offline_conditions(self) -> None:
        for name in ("none", "cams_absent", "cams_black", "force_absent", "force_dead_undeclared"):
            self.assertIn(name, FAULT_PRESETS)
        self.assertEqual(FAULT_PRESETS["cams_absent"]["fault/absent_cameras"], list(CAMERA_KEYS))

    def test_pour_trigger_only_fires_while_pouring(self) -> None:
        schedule = FaultSchedule("cams_absent", "pour")
        detector = PourDetector()
        self.assertEqual(schedule.fields(detector), {})
        detector.pouring = True
        self.assertEqual(schedule.fields(detector), {"fault/absent_cameras": list(CAMERA_KEYS)})

    def test_always_and_never(self) -> None:
        self.assertEqual(FaultSchedule("force_absent", "always").fields(None), {"fault/force": "absent"})
        self.assertEqual(FaultSchedule("force_absent", "never").fields(None), {})
        self.assertEqual(FaultSchedule("none", "always").fields(None), {})

    def test_fields_are_copies(self) -> None:
        fields = FaultSchedule("cams_black", "always").fields(None)
        fields["fault/black_cameras"].append("x")
        self.assertEqual(FAULT_PRESETS["cams_black"]["fault/black_cameras"], list(CAMERA_KEYS))

    def test_unknown_names_raise(self) -> None:
        with self.assertRaises(ValueError):
            FaultSchedule("overhead_gone", "pour")
        with self.assertRaises(ValueError):
            FaultSchedule("cams_absent", "sometimes")


if __name__ == "__main__":
    unittest.main()
