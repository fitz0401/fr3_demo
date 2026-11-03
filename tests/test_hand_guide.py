import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.recording import (
    MEASURED_ACTION_SOURCE,
    RawEpisodeWriter,
    RobotSample,
    forward_difference_velocity,
)
from fr3_demo.teleop import HAND_GUIDE_ACTION_SOURCE, _create_parser, _validate_args, hand_guide_tick


def _finished_episode(session: Path, action_source: str | None, raw_velocity: float | None = None) -> Path:
    writer = RawEpisodeWriter(
        session,
        episode_index=0,
        fps=15.0,
        camera_serials={"exterior_image_left": "external", "wrist_image": "wrist"},
        image_size=None,
        action_source=action_source,
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    for index in range(2):
        target = writer.started_monotonic + index / 15
        writer.add_sample(
            target_monotonic=target,
            state={"qpos": np.zeros(7), "dq": np.zeros(7), "tau_J": np.zeros(7), "time_sec": 1.0 + index / 15},
            action_joint_velocity=np.zeros(7),
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames={
                "exterior_image_left": CameraFrame(image, target, 1.0 + index / 15, index + 1),
                "wrist_image": CameraFrame(image, target, 1.0 + index / 15, index + 1),
            },
        )
    if raw_velocity is not None:
        # The arm moved at a constant rate the whole time, sampled at 1 kHz.
        times = np.arange(0.9, 1.3, 0.001)
        writer.set_raw_force(
            [
                RobotSample(
                    qpos=np.full(7, raw_velocity * (stamp - 0.9), dtype=np.float32),
                    dq=np.full(7, raw_velocity, dtype=np.float32),
                    tau_J=np.zeros(7, dtype=np.float32),
                    controller_timestamp=float(stamp),
                    request_send_monotonic=float(stamp),
                    response_receive_monotonic=float(stamp),
                )
                for stamp in times
            ]
        )
    return writer.finish()


def _finished_metadata(session: Path, action_source: str | None) -> dict:
    return json.loads((_finished_episode(session, action_source) / "metadata.json").read_text())


class HandGuideTest(unittest.TestCase):
    def test_hand_guide_requires_collect(self) -> None:
        parser = _create_parser()
        args = parser.parse_args(["--hand-guide"])
        with self.assertRaises(SystemExit):
            _validate_args(args, parser)

    def test_tick_records_measured_velocity_and_commands_nothing(self) -> None:
        robot = MagicMock()
        robot.state.return_value = {"qpos": [0.1] * 7, "dq": [0.25, 0, 0, 0, 0, 0, -0.5]}
        collector = MagicMock()

        q = hand_guide_tick(robot, collector)

        np.testing.assert_allclose(q, [0.1] * 7)
        recorded = collector.set_action.call_args.args[0]
        np.testing.assert_allclose(recorded, [0.25, 0, 0, 0, 0, 0, -0.5])
        robot.start_stream.assert_not_called()
        robot.stream_velocity.assert_not_called()
        robot.legacy_command.assert_not_called()

    def test_metadata_names_the_action_source_only_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            guided = _finished_metadata(Path(temporary) / "guided", HAND_GUIDE_ACTION_SOURCE)
            teleop = _finished_metadata(Path(temporary) / "teleop", None)
        self.assertEqual(guided["action_source"], "measured_joint_velocity")
        self.assertNotIn("action_source", teleop)

    def test_constant_motion_gives_its_velocity_at_every_row(self) -> None:
        raw_times = np.arange(0.0, 2.0, 0.001)
        positions = np.outer(raw_times, [0.5, -0.25, 0, 0, 0, 0, 1.0])
        rows = np.array([0.0, 0.5, 1.0, 1.99])
        velocity = forward_difference_velocity(rows, raw_times, positions, horizon=1 / 15)
        np.testing.assert_allclose(velocity, np.tile([0.5, -0.25, 0, 0, 0, 0, 1.0], (4, 1)), atol=1e-6)

    def test_measured_episodes_take_their_action_from_raw_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            guided = _finished_episode(Path(temporary) / "guided", MEASURED_ACTION_SOURCE, raw_velocity=0.3)
            teleop = _finished_episode(Path(temporary) / "teleop", None, raw_velocity=0.3)
            guided_action = np.load(guided / "trajectory.npz")["action_joint_velocity"]
            teleop_action = np.load(teleop / "trajectory.npz")["action_joint_velocity"]
            guided_meta = json.loads((guided / "metadata.json").read_text())
        # The loop recorded zero; the raw samples say the arm moved at 0.3 rad/s.
        np.testing.assert_allclose(guided_action, 0.3, atol=1e-4)
        np.testing.assert_allclose(teleop_action, 0.0)
        self.assertEqual(guided_meta["action_derivation"], "forward_difference_of_raw_joint_position_over_one_row")

    def test_teleop_and_recorder_agree_on_the_source_name(self) -> None:
        self.assertEqual(HAND_GUIDE_ACTION_SOURCE, MEASURED_ACTION_SOURCE)


if __name__ == "__main__":
    unittest.main()
