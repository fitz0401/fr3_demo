import argparse
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fr3_demo.teleop import HOME_JOINTS
from fr3_pi05.runner import _home_robot, _open_gripper


class Pi05RunnerTest(unittest.TestCase):
    def test_startup_gripper_open_is_blocking(self) -> None:
        robot = MagicMock()

        _open_gripper(robot)

        robot.open_gripper.assert_called_once_with()

    @patch("fr3_pi05.runner.stream_home")
    def test_home_uses_temporary_bamboo_stream(self, stream_home: MagicMock) -> None:
        stream_home.return_value = HOME_JOINTS.copy()
        robot = MagicMock()
        args = argparse.Namespace(
            home_speed=0.18,
            max_joint_speed=0.20,
            home_timeout=15.0,
            stream_hz=30.0,
            watchdog_ms=250,
            max_joint_acceleration=1.0,
        )
        stopped = threading.Event()

        result = _home_robot(robot, args, stopped)

        robot.start_stream.assert_called_once_with(250, 0.20, 1.0)
        robot.stop_stream.assert_called_once_with()
        np.testing.assert_array_equal(result, HOME_JOINTS)
        call = stream_home.call_args
        np.testing.assert_array_equal(call.args[1], HOME_JOINTS)
        self.assertEqual(call.args[2:5], (0.18, 15.0, 30.0))
        self.assertFalse(call.kwargs["stop_requested"]())

    @patch("fr3_pi05.runner.stream_home", side_effect=RuntimeError("failed"))
    def test_home_stops_stream_after_failure(self, _stream_home: MagicMock) -> None:
        robot = MagicMock()
        args = argparse.Namespace(
            home_speed=0.18,
            max_joint_speed=0.20,
            home_timeout=15.0,
            stream_hz=30.0,
            watchdog_ms=250,
            max_joint_acceleration=1.0,
        )

        with self.assertRaisesRegex(RuntimeError, "failed"):
            _home_robot(robot, args, threading.Event())

        robot.stop_stream.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
