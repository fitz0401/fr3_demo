import threading
import unittest
from collections import deque

import numpy as np

from fr3_demo.recording import ActionSample, DemoCollector, GripperSample, RobotSample
from fr3_demo.synchronization import AffineClockMapper


class ClockMapperTest(unittest.TestCase):
    def test_recovers_offset_and_clock_drift_from_delayed_samples(self) -> None:
        mapper = AffineClockMapper(lower_envelope=True)
        scale = 1.0001
        offset = 500.0
        for index in range(200):
            source = index * 0.05
            transport_delay = 0.003 + (index % 7) * 0.001
            mapper.add(source, scale * source + offset + transport_delay)

        estimate = mapper.estimate()

        self.assertAlmostEqual(estimate.scale, scale, delta=2e-4)
        self.assertAlmostEqual(mapper.to_host(5.0), scale * 5.0 + offset + 0.003, delta=0.002)

    def test_timestamp_domain_jump_resets_the_mapping(self) -> None:
        mapper = AffineClockMapper()
        mapper.add(1.0, 10.0)
        mapper.add(1.1, 10.1)
        mapper.add(1_700_000_000.0, 10.2)

        estimate = mapper.estimate()

        self.assertEqual(estimate.sample_count, 1)
        self.assertAlmostEqual(mapper.to_host(1_700_000_000.0), 10.2)

    def test_global_time_mapping_keeps_unit_scale(self) -> None:
        mapper = AffineClockMapper(lower_envelope=True, fit_scale=False)
        for index in range(100):
            source = 1_700_000_000.0 + index * 0.03
            mapper.add(source, 1000.0 + index * 0.0301 + (index % 5) * 0.002)

        self.assertEqual(mapper.estimate().scale, 1.0)


class StreamAlignmentTest(unittest.TestCase):
    def _collector(self, *, enable_gripper: bool = True) -> DemoCollector:
        collector = object.__new__(DemoCollector)
        collector._sensor_lock = threading.Lock()
        collector._arm_samples = deque()
        collector._gripper_samples = deque()
        collector._joint_actions = deque()
        collector._arm_clock = AffineClockMapper()
        collector.robot_max_gap = 0.03
        collector.gripper_max_age = 2.0
        collector.enable_gripper = enable_gripper
        return collector

    def test_robot_state_is_interpolated_on_host_timeline(self) -> None:
        collector = self._collector()
        before = RobotSample(np.zeros(7), np.zeros(7), np.zeros(7), 1.0, 9.999, 10.001)
        after = RobotSample(np.ones(7), np.full(7, 2.0), np.full(7, 4.0), 1.02, 10.019, 10.021)
        collector._arm_samples.extend((before, after))
        collector._arm_clock.add(1.0, 10.0, 0.001)
        collector._arm_clock.add(1.02, 10.02, 0.001)

        state, timing = collector._interpolate_robot(10.01)

        np.testing.assert_allclose(state["qpos"], 0.5)
        np.testing.assert_allclose(state["dq"], 1.0)
        np.testing.assert_allclose(state["tau_J"], 2.0)
        self.assertAlmostEqual(timing["robot_interpolation_alpha"], 0.5)

    def test_gripper_interpolates_and_action_uses_zero_order_hold(self) -> None:
        collector = self._collector()
        collector._gripper_samples.extend(
            (
                GripperSample(0.0, 10.0, 10.0),
                GripperSample(1.0, 10.1, 10.1),
            )
        )
        collector._joint_actions.extend(
            (
                ActionSample(np.zeros(7), 10.0),
                ActionSample(np.ones(7), 10.06),
            )
        )

        gripper, timing = collector._interpolate_gripper(10.05)
        action = collector._action_at(collector._joint_actions, 10.05)

        self.assertAlmostEqual(gripper, 0.5)
        self.assertTrue(str(timing["gripper_sample_source"]).startswith("interpolated:"))
        np.testing.assert_array_equal(action.value, np.zeros(7))


    def test_a_disabled_gripper_never_goes_stale(self) -> None:
        collector = self._collector(enable_gripper=False)

        # An episode can start long after launch; a fixed value must not age out
        # the way a real observation does.
        position, timing = collector._interpolate_gripper(10_000.0)

        self.assertEqual(position, 0.0)
        self.assertEqual(timing["gripper_sample_source"], "disabled")
        self.assertEqual(timing["gripper_sample_age"], 0.0)


if __name__ == "__main__":
    unittest.main()
