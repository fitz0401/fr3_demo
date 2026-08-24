import argparse
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fr3_demo.teleop import HOME_JOINTS
from fr3_pi05.runner import GripperWorker, _home_robot, _open_gripper, _parse, _validate_policy_contract


class Pi05RunnerTest(unittest.TestCase):
    def test_gripper_worker_tracks_transition_without_reopening_initial_state(self) -> None:
        robot = MagicMock()
        robot.gripper_position.side_effect = [1.0, 0.79]
        close_started = threading.Event()
        release_close = threading.Event()

        def close_gripper() -> None:
            close_started.set()
            release_close.wait(1.0)

        robot.close_gripper.side_effect = close_gripper
        worker = GripperWorker(robot, enabled=True)
        try:
            worker.wait_ready()
            self.assertFalse(worker.command(1.0))
            self.assertTrue(worker.command(0.0))
            self.assertTrue(close_started.wait(1.0))
            self.assertTrue(worker.moving)
            release_close.set()
            deadline = time.monotonic() + 1.0
            while worker.moving and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(worker.moving)
            self.assertAlmostEqual(worker.position, 0.79)
            robot.open_gripper.assert_not_called()
            robot.close_gripper.assert_called_once_with()
        finally:
            release_close.set()
            worker.close()

    def test_wine_profile_enforces_checkpoint_horizon_and_disables_prefetch(self) -> None:
        args = _parse(
            [
                "--checkpoint",
                "wine_hybrid",
                "--open-loop-horizon",
                "3",
                "--prefetch-actions",
                "2",
            ]
        )

        self.assertEqual(args.open_loop_horizon, 16)
        self.assertEqual(args.prefetch_actions, 0)

    def test_wine_contract_requires_exact_training_metadata(self) -> None:
        metadata = {
            "model": "pi05_wine_hybrid",
            "loader": "wine",
            "action_expert_variant": "gemma_300m_lora",
            "action_horizon": 16,
            "state_history_lags": [45, 75],
            "num_state_frames": 3,
            "joint_observation_dim": 21,
            "gripper_observation_dim": 3,
            "joint_observation_shape": [3, 7],
            "gripper_observation_shape": [3],
            "image_observation_shape": [180, 320, 3],
            "proprio_history_offsets": [0, 45, 75],
            "tasks": ["pour lillet into the jigger", "pour gin into the jigger"],
        }

        _validate_policy_contract("wine_hybrid", metadata)
        metadata["joint_observation_shape"] = [21]
        with self.assertRaisesRegex(RuntimeError, "history contract"):
            _validate_policy_contract("wine_hybrid", metadata)

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
