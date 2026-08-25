import argparse
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fr3_demo.teleop import HOME_JOINTS
from fr3_pi05.runner import (
    GripperWorker,
    _home_robot,
    _open_abort_joystick,
    _open_gripper,
    _parse,
    _print_action_chunk,
    _validate_exterior2_contract,
    _validate_policy_contract,
    _validate_prompt,
)


class Pi05RunnerTest(unittest.TestCase):
    @patch("fr3_pi05.runner.LinuxJoystick")
    def test_missing_joystick_is_optional_for_policy_execution(self, joystick_class: MagicMock) -> None:
        joystick_class.return_value.open.side_effect = RuntimeError("missing")

        joystick, count = _open_abort_joystick("/dev/input/js0", required=False)

        self.assertIsNone(joystick)
        self.assertEqual(count, 0)

    @patch("fr3_pi05.runner.LinuxJoystick")
    def test_required_joystick_preserves_fail_closed_behavior(self, joystick_class: MagicMock) -> None:
        joystick_class.return_value.open.side_effect = RuntimeError("missing")

        with self.assertRaisesRegex(RuntimeError, "missing"):
            _open_abort_joystick("/dev/input/js0", required=True)

    def test_gripper_worker_tracks_transition_without_reopening_initial_state(self) -> None:
        robot = MagicMock()
        robot.gripper_position.side_effect = [1.0, 0.79]
        close_started = threading.Event()
        release_close = threading.Event()

        def close_gripper() -> None:
            close_started.set()
            release_close.wait(1.0)

        robot.close_gripper.side_effect = close_gripper
        worker = GripperWorker(robot, enabled=True, threshold=0.9)
        try:
            worker.wait_ready()
            self.assertFalse(worker.command(1.0))
            self.assertTrue(worker.command(0.8))
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

    def test_debug_chunk_flag_and_gripper_threshold_output(self) -> None:
        args = _parse(["--debug-chunks"])
        self.assertTrue(args.debug_chunks)
        self.assertEqual(args.gripper_threshold, 0.9)
        chunk = np.zeros((2, 8), dtype=float)
        chunk[:, 7] = [0.9, 0.9001]

        with patch("builtins.print") as output:
            _print_action_chunk(chunk, 3, "test", args.gripper_threshold)

        rendered = "\n".join(" ".join(str(value) for value in call.args) for call in output.call_args_list)
        self.assertIn("DEBUG ACTION CHUNK #3 (test), shape=(2, 8)", rendered)
        self.assertIn("0:0.9000->CLOSE", rendered)
        self.assertIn("1:0.9001->OPEN", rendered)

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
            "asset_id": "fitz0401/franka_pour_wine",
            "tasks": ["pour lillet into the jigger", "pour gin into the jigger"],
        }

        _validate_policy_contract("wine_hybrid", metadata)
        metadata["joint_observation_shape"] = [21]
        with self.assertRaisesRegex(RuntimeError, "history contract"):
            _validate_policy_contract("wine_hybrid", metadata)

    def test_prompt_allowlist_comes_from_server_metadata(self) -> None:
        _validate_prompt("new dataset task", {"tasks": []})
        _validate_prompt("new dataset task", {"tasks": ["new dataset task"]})
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            _validate_prompt("wrong task", {"tasks": ["new dataset task"]})

    def test_exterior2_is_required_only_by_models_that_consume_it(self) -> None:
        observation = {"observation/exterior_image_1_left": np.zeros((1, 1, 3))}
        _validate_exterior2_contract({"uses_exterior2": False}, observation)
        with self.assertRaisesRegex(RuntimeError, "L515"):
            _validate_exterior2_contract({"uses_exterior2": True}, observation)
        observation["observation/exterior_image_2_left"] = np.zeros((1, 1, 3))
        _validate_exterior2_contract({"uses_exterior2": True}, observation)

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
