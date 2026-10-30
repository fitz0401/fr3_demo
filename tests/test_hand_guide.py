import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.recording import RawEpisodeWriter
from fr3_demo.teleop import HAND_GUIDE_ACTION_SOURCE, _create_parser, _validate_args, hand_guide_tick


def _finished_metadata(session: Path, action_source: str | None) -> dict:
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
    episode = writer.finish()
    return json.loads((episode / "metadata.json").read_text())


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


if __name__ == "__main__":
    unittest.main()
