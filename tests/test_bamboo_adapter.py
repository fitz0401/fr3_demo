import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from fr3_demo.teleop import HOME_JOINTS, BambooRobot, stream_home


class BambooAdapterTest(unittest.TestCase):
    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_close_gripper_uses_configured_force(self, client_class: MagicMock) -> None:
        arm = MagicMock()
        arm.get_joint_states.return_value = {
            "qpos": [0.0] * 7,
            "dq": [0.0] * 7,
            "tau_J": [0.0] * 7,
            "ee_pose": np.eye(4).tolist(),
            "time_sec": 1.0,
        }
        gripper = MagicMock()
        gripper.close_gripper.return_value = {"success": True}
        client_class.return_value = arm
        with patch("bamboo.BambooFrankaClient", return_value=gripper):
            robot = BambooRobot("127.0.0.1", 5555, 5559, "robotiq", True, 0.8)

            robot.close_gripper()

        gripper.close_gripper.assert_called_once_with(speed=0.05, force=0.8, blocking=True)

    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_uses_public_bamboo_streaming_api(self, client_class: MagicMock) -> None:
        arm = MagicMock()
        arm.get_joint_states.return_value = {
            "qpos": [0.0] * 7,
            "dq": [0.0] * 7,
            "tau_J": [0.0] * 7,
            "ee_pose": np.eye(4).tolist(),
            "time_sec": 1.0,
        }
        arm.supports_streaming.return_value = True
        arm.start_streaming.return_value = {"success": True}
        arm.stream_joint_velocity.return_value = {"success": True}
        arm.stop_streaming.return_value = {"success": True}
        arm.execute_joint_impedance_path.return_value = {"success": True}
        client_class.return_value = arm

        robot = BambooRobot("127.0.0.1", 5555, 5559, "robotiq", False)
        velocity = np.arange(7, dtype=float) * 0.01
        robot.start_stream(250, 0.35, 1.5)
        robot.stream_velocity(velocity)
        robot.stop_stream()
        home = HOME_JOINTS.copy()
        robot.legacy_command(home, 4.0)
        robot.close()

        client_class.assert_called_once_with(server_ip="127.0.0.1", control_port=5555, enable_gripper=False)
        arm.start_streaming.assert_called_once_with(
            watchdog_ms=250,
            max_joint_velocity=0.35,
            max_joint_acceleration=1.5,
        )
        arm.stream_joint_velocity.assert_has_calls([call(velocity)])
        arm.stop_streaming.assert_called_once_with()
        trajectory_args, trajectory_kwargs = arm.execute_joint_impedance_path.call_args
        np.testing.assert_array_equal(trajectory_args[0], home[None, :])
        self.assertEqual(trajectory_kwargs, {"durations": [4.0], "default_duration": 4.0})
        arm.close.assert_called_once_with()

    @patch("fr3_demo.teleop.time.sleep")
    def test_stream_home_closes_position_error_and_stops(self, _: MagicMock) -> None:
        robot = MagicMock()
        target = HOME_JOINTS.copy()
        start = target.copy()
        start[0] -= 0.1
        robot.state.side_effect = [
            {"qpos": start.tolist(), "dq": [0.0] * 7},
            {"qpos": target.tolist(), "dq": [0.0] * 7},
        ]

        result = stream_home(robot, target, max_velocity=0.2, minimum_timeout=1.0, rate_hz=30.0)

        np.testing.assert_array_equal(result, target)
        first_command = robot.stream_velocity.call_args_list[0].args[0]
        self.assertAlmostEqual(first_command[0], 0.12)
        np.testing.assert_array_equal(robot.stream_velocity.call_args_list[-1].args[0], np.zeros(7))


if __name__ == "__main__":
    unittest.main()
