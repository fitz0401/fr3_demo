import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.convert_lerobot import convert
from fr3_demo.recording import RawEpisodeWriter


def _write_episode(session: Path, *, include_exterior2: bool = False) -> Path:
    camera_serials = {"exterior_image_left": "external", "wrist_image": "wrist"}
    if include_exterior2:
        camera_serials["exterior_image_2_left"] = "external2"
    writer = RawEpisodeWriter(
        session,
        episode_index=0,
        fps=15.0,
        camera_serials=camera_serials,
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    frames = {
        "exterior_image_left": CameraFrame(image, 10.0, 1.0, 1),
        "wrist_image": CameraFrame(image, 10.0, 1.0, 1),
    }
    if include_exterior2:
        frames["exterior_image_2_left"] = CameraFrame(np.full_like(image, 200), 10.0, 1.0, 1)
    for index in range(2):
        writer.add_sample(
            captured_monotonic=writer.started_monotonic + index / 15,
            state={"qpos": np.arange(7), "dq": np.arange(7) / 10, "time_sec": 20.0 + index / 15},
            action_joint_velocity=np.arange(7) / 100,
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames=frames,
        )
    return writer.finish()


class RawRecordingTest(unittest.TestCase):
    def test_episode_is_atomically_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            episode = _write_episode(session)

            self.assertFalse((session / "episode_000000.inprogress").exists())
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(
                metadata["camera_transforms"],
                {"exterior_image_left": "none", "wrist_image": "none"},
            )
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertEqual(trajectory["joint_position"].shape, (2, 7))
                self.assertEqual(trajectory["action_joint_velocity"].shape, (2, 7))
                self.assertEqual(trajectory["gripper_position"].shape, (2, 1))

    def test_conversion_matches_openpi_droid_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "raw_a"
            second_root = root / "raw_b"
            first_episode = _write_episode(first_root)
            second_episode = _write_episode(second_root, include_exterior2=True)
            for episode, language in (
                (first_episode, "pick up the block"),
                (second_episode, "pour the liquid"),
            ):
                metadata_path = episode / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["language_instruction"] = language
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            class FakeDataset:
                instance = None

                @classmethod
                def create(cls, **kwargs):
                    cls.instance = cls()
                    cls.instance.root = root / "converted"
                    cls.instance.create_args = kwargs
                    cls.instance.frames = []
                    cls.instance.saved = 0
                    return cls.instance

                def add_frame(self, frame):
                    self.frames.append(frame)

                def save_episode(self):
                    self.saved += 1

            with patch("fr3_demo.convert_lerobot._load_lerobot_dataset", return_value=FakeDataset):
                output = convert(
                    [first_root, first_root, second_root],
                    "test/fr3",
                    output_root=root / "datasets",
                )

            dataset = FakeDataset.instance
            self.assertEqual(output, root / "converted")
            self.assertEqual(dataset.create_args["fps"], 15)
            self.assertEqual(dataset.saved, 2)
            self.assertEqual(len(dataset.frames), 4)
            frame = dataset.frames[0]
            self.assertEqual(frame["exterior_image_1_left"].shape, (180, 320, 3))
            self.assertEqual(frame["wrist_image_left"].shape, (180, 320, 3))
            self.assertFalse(frame["exterior_image_2_left"].any())
            self.assertEqual(frame["actions"].shape, (8,))
            self.assertEqual(frame["task"], "pick up the block")
            self.assertEqual(dataset.frames[-1]["task"], "pour the liquid")
            self.assertTrue(dataset.frames[-1]["exterior_image_2_left"].any())


if __name__ == "__main__":
    unittest.main()
