"""The dataset carries what replays and the GelSight record: Franka's force, the gel's deformation, provenance."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_tactile_shear import HEIGHT, WIDTH, _grid, _moved, _render

from fr3_demo.annotate import main as annotate
from fr3_demo.cameras import CameraFrame
from fr3_demo.convert_lerobot import DEFORMATION_KEY, EXTERNAL_FEATURES, TRACKED_KEY, convert, tactile_deformation
from fr3_demo.recording import MEASURED_ACTION_SOURCE, RawEpisodeWriter, RobotSample
from fr3_demo.replay import REPLAY_ACTION_SOURCE
from fr3_demo.tactile import TactileFrame

TACTILE = {"device": "/dev/video0", "width": WIDTH, "height": HEIGHT, "fps": 25.0, "rotate_180": False}
ROWS = 4


def _episode(
    session: Path,
    index: int,
    *,
    external: bool = True,
    gel_shifts: list[tuple[float, float]] | None = None,
    plain_gel: bool = False,
    **writer_options,
) -> Path:
    """An episode with ROWS rows; Franka's wrench is (1, 2, 3, .1, .2, .3) times the row number."""

    writer = RawEpisodeWriter(
        session,
        episode_index=index,
        fps=15.0,
        camera_serials={"exterior_image_left": "external", "wrist_image": "wrist"},
        image_size=None,
        tactile=TACTILE if gel_shifts is not None else None,
        **writer_options,
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    grid = _grid()
    frame = 0
    raw = []
    for row in range(ROWS):
        target = writer.started_monotonic + row / 15
        if gel_shifts is not None:
            for sub in range(2):
                gel = (
                    np.full((HEIGHT, WIDTH, 3), 150, np.uint8)
                    if plain_gel
                    else _render(_moved(grid, shift=gel_shifts[frame]), seed=frame + 1)
                )
                writer.add_tactile_frame(TactileFrame(gel, target - 0.02 + sub / 30, frame))
                frame += 1
        state = {"qpos": np.zeros(7), "dq": np.zeros(7), "tau_J": np.zeros(7), "time_sec": 20.0 + row / 15}
        if external:
            state["ext_wrench"] = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]) * (row + 1)
            state["tau_ext"] = np.full(7, 0.5)
        raw.append(
            RobotSample(
                qpos=np.zeros(7, np.float32),
                dq=np.zeros(7, np.float32),
                tau_J=np.zeros(7, np.float32),
                controller_timestamp=20.0 + row / 15,
                request_send_monotonic=target,
                response_receive_monotonic=target,
                ext_wrench=np.ones(6, np.float32) if external else None,
                tau_ext=np.ones(7, np.float32) if external else None,
            )
        )
        writer.add_sample(
            target_monotonic=target,
            state=state,
            action_joint_velocity=np.zeros(7),
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames={
                "exterior_image_left": CameraFrame(image, target, 1.0 + row / 15, row + 1),
                "wrist_image": CameraFrame(image, target, 1.0 + row / 15, row + 1),
            },
        )
    writer.set_raw_force(raw)
    episode = writer.finish({"cameras": {}, "robot_controller": {"scale": 1.0, "offset": 0.0}})
    metadata = json.loads((episode / "metadata.json").read_text())
    metadata["language_instruction"] = "peel the lemon"
    (episode / "metadata.json").write_text(json.dumps(metadata))
    return episode


def _convert(root: Path, *data_dirs: Path, kind: str = "all"):
    class FakeDataset:
        instance = None

        @classmethod
        def create(cls, **kwargs):
            cls.instance = cls()
            cls.instance.root = root / "converted"
            cls.instance.create_args = kwargs
            cls.instance.frames = []
            return cls.instance

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            pass

    with patch("fr3_demo.convert_lerobot._load_lerobot_dataset", return_value=FakeDataset):
        convert(list(data_dirs), "test/fr3", output_root=root / "datasets", episode_kind=kind)
    return FakeDataset.instance


SHIFTS = [(0.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, -0.5), (2.0, -1.0), (2.5, -1.0), (3.0, -1.0)]


class ForceInTheDatasetTest(unittest.TestCase):
    def test_franka_s_force_estimate_is_carried_row_by_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _episode(root / "raw", 0)
            dataset = _convert(root, root / "raw")
            features = dataset.create_args["features"]
            for key, names in EXTERNAL_FEATURES.items():
                self.assertEqual(features[key]["shape"], (len(names),))
            np.testing.assert_allclose(dataset.frames[2]["external_wrench"], [3.0, 6.0, 9.0, 0.3, 0.6, 0.9], atol=1e-5)
            np.testing.assert_allclose(dataset.frames[0]["external_joint_torque"], 0.5)
            with np.load(Path(dataset.root) / "force" / "episode_000000.npz") as stream:
                self.assertEqual(stream["external_wrench"].shape, (ROWS, 6))
                self.assertEqual(stream["joint_torque"].shape, (ROWS, 7))
                self.assertEqual(len(stream["row_robot_timestamp"]), ROWS)

    def test_it_is_left_out_rather_than_made_up_when_an_episode_lacks_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _episode(root / "raw", 0)
            _episode(root / "raw", 1, external=False)
            dataset = _convert(root, root / "raw")
            self.assertFalse(set(EXTERNAL_FEATURES) & set(dataset.create_args["features"]))
            self.assertNotIn("external_wrench", dataset.frames[0])


class GelDeformationTest(unittest.TestCase):
    def test_the_marker_shift_since_the_first_gel_frame_is_carried_row_by_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _episode(root / "raw", 0, gel_shifts=SHIFTS)
            dataset = _convert(root, root / "raw")
            self.assertEqual(dataset.create_args["features"][DEFORMATION_KEY]["shape"], (4,))
            with np.load(episode / "trajectory.npz") as trajectory:
                rows = trajectory["tactile_frame_index"]
            for frame, index in zip(dataset.frames, rows):
                np.testing.assert_allclose(frame[DEFORMATION_KEY][:2], SHIFTS[index], atol=0.15)
                self.assertEqual(float(frame[TRACKED_KEY][0]), 1.0)

    def test_a_gel_without_markers_leaves_the_feature_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _episode(root / "raw", 0, gel_shifts=SHIFTS, plain_gel=True)
            self.assertIn("no marker grid", tactile_deformation(episode))
            dataset = _convert(root, root / "raw")
            self.assertIn("tactile_image", dataset.create_args["features"])
            self.assertNotIn(DEFORMATION_KEY, dataset.create_args["features"])


class EpisodeRecordTest(unittest.TestCase):
    def test_each_episode_says_where_it_came_from_and_which_demo_a_replay_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "raw" / "session_1"
            _episode(session, 0, action_source=MEASURED_ACTION_SOURCE)
            replay = _episode(
                session, 1, action_source=REPLAY_ACTION_SOURCE, extra_metadata={"replay_of": "episode_000000"}
            )
            metadata = json.loads((replay / "metadata.json").read_text())
            metadata["replay"] = {"completed": True}
            (replay / "metadata.json").write_text(json.dumps(metadata))

            dataset = _convert(root, root / "raw")
            records = json.loads((Path(dataset.root) / "meta" / "fr3_episodes.json").read_text())["episodes"]
            self.assertEqual([record["kind"] for record in records], ["hand_guided", "replay"])
            self.assertEqual(records[1]["replay_of"], {"source": "session_1/episode_000000", "episode_index": 0})
            self.assertEqual(records[1]["action_source"], REPLAY_ACTION_SOURCE)
            self.assertEqual(records[0]["force_stream"], "force/episode_000000.npz")

            # Converting the replays alone: the record still names the demonstration, outside this dataset.
            alone = _convert(root, root / "raw", kind="replays")
            record = json.loads((Path(alone.root) / "meta" / "fr3_episodes.json").read_text())["episodes"][0]
            self.assertEqual(record["replay_of"]["episode_index"], None)


def _labeled(session: Path, index: int, language: str | None, **extra) -> Path:
    episode = session / f"episode_{index:06d}"
    episode.mkdir(parents=True)
    path = episode / "metadata.json"
    path.write_text(json.dumps({"complete": True, "language_instruction": language, **extra}))
    return path


def _language(path: Path) -> str | None:
    return json.loads(path.read_text())["language_instruction"]


class AnnotateReplaysTest(unittest.TestCase):
    def test_a_replay_takes_its_demonstration_s_instruction_without_asking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session_1"
            demo = _labeled(session, 0, None)
            replay = _labeled(session, 1, None, replay_of="episode_000000")
            with patch("builtins.input", return_value="peel the lemon") as prompt:
                annotate(["--data-dir", temporary])
            prompt.assert_called_once()
            self.assertEqual(_language(demo), "peel the lemon")
            self.assertEqual(_language(replay), "peel the lemon")

    def test_a_skipped_demonstration_leaves_its_replay_unlabeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session_1"
            _labeled(session, 0, None)
            replay = _labeled(session, 1, None, replay_of="episode_000000")
            with patch("builtins.input", return_value="") as prompt:
                annotate(["--data-dir", temporary])
            prompt.assert_called_once()
            self.assertIsNone(_language(replay))

    def test_replacing_a_demonstration_s_instruction_carries_over_to_its_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session_1"
            demo = _labeled(session, 0, "old")
            replay = _labeled(session, 1, "old", replay_of="episode_000000")
            other = _labeled(session, 2, "kept apart", replay_of="episode_000000")
            with patch("builtins.input", return_value="new"):
                annotate(["--data-dir", temporary, "--replace"])
            self.assertEqual([_language(path) for path in (demo, replay, other)], ["new", "new", "new"])
            with patch("builtins.input") as prompt:
                annotate(["--data-dir", temporary])
            prompt.assert_not_called()

    def test_a_link_written_while_the_prompt_waits_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session_1"
            demo = _labeled(session, 0, None)

            def answer(_prompt: str) -> str:
                # The replay finishes meanwhile and links itself into the demonstration.
                metadata = json.loads(demo.read_text())
                metadata["replays"] = [{"episode": "episode_000001", "completed": True}]
                demo.write_text(json.dumps(metadata))
                return "peel the lemon"

            with patch("builtins.input", side_effect=answer):
                annotate(["--data-dir", temporary])
            metadata = json.loads(demo.read_text())
        self.assertEqual(metadata["language_instruction"], "peel the lemon")
        self.assertEqual(metadata["replays"], [{"episode": "episode_000001", "completed": True}])


if __name__ == "__main__":
    unittest.main()
