"""Tactile capture, the force/tactile switches, and tactile in the dataset."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.convert_lerobot import convert, write_tactile_sidecar
from fr3_demo.recording import TACTILE_KEY, RawEpisodeWriter
from fr3_demo.tactile import NATIVE_FPS, GelSightCamera, TactileFrame, resolve_device

TACTILE = {"device": "/dev/video20", "width": 32, "height": 24, "fps": 25.0, "rotate_180": False}


def _tactile_frame(index: int, timestamp: float, size: tuple[int, int] = (24, 32)) -> TactileFrame:
    # A per-frame constant makes it checkable which frame a row selected.
    image = np.full((*size, 3), index % 251, dtype=np.uint8)
    return TactileFrame(image=image, captured_monotonic=timestamp, frame_number=index)


def _write_episode(
    session: Path,
    *,
    episode_index: int = 0,
    record_force: bool = True,
    tactile: dict | None = None,
    rows: int = 3,
    tactile_per_row: int = 2,
) -> Path:
    writer = RawEpisodeWriter(
        session,
        episode_index=episode_index,
        fps=15.0,
        camera_serials={"exterior_image_left": "external", "wrist_image": "wrist"},
        record_force=record_force,
        tactile=tactile,
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    tactile_index = 0
    for row in range(rows):
        target = writer.started_monotonic + row / 15
        if tactile is not None:
            # Several tactile frames per grid row: the sensor runs faster than
            # the grid, which is the whole point of the stream.
            for sub in range(tactile_per_row):
                stamp = target - 0.02 + sub * (1.0 / 15) / tactile_per_row
                writer.add_tactile_frame(
                    _tactile_frame(tactile_index, stamp, (tactile["height"], tactile["width"]))
                )
                tactile_index += 1
        state = {
            "qpos": np.arange(7),
            "dq": np.arange(7) / 10,
            "time_sec": 20.0 + row / 15,
        }
        if record_force:
            state["tau_J"] = np.arange(7) * 1.5
        writer.add_sample(
            target_monotonic=target,
            state=state,
            action_joint_velocity=np.arange(7) / 100,
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames={
                "exterior_image_left": CameraFrame(image, target, 1.0 + row / 15, row + 1),
                "wrist_image": CameraFrame(image, target, 1.0 + row / 15, row + 1),
            },
        )
    episode = writer.finish({"cameras": {}, "robot_controller": {"scale": 1.0, "offset": 0.0}})
    metadata = json.loads((episode / "metadata.json").read_text())
    metadata["language_instruction"] = "peel the cucumber"
    (episode / "metadata.json").write_text(json.dumps(metadata))
    return episode


def _convert(root: Path, *data_dirs: Path) -> Path:
    class FakeDataset:
        instance = None

        @classmethod
        def create(cls, **kwargs):
            cls.instance = cls()
            cls.instance.root = root / "converted"
            cls.instance.frames = []
            cls.instance.create_args = kwargs
            return cls.instance

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            pass

    with patch("fr3_demo.convert_lerobot._load_lerobot_dataset", return_value=FakeDataset):
        convert(list(data_dirs), "test/fr3", output_root=root / "datasets")
    return FakeDataset.instance


class DeviceResolutionTest(unittest.TestCase):
    def test_accepts_index_path_or_auto(self) -> None:
        self.assertEqual(resolve_device(20), 20)
        self.assertEqual(resolve_device("20"), 20)
        self.assertEqual(resolve_device("/dev/video20"), 20)

    def test_rejects_an_uninterpretable_device(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_device("/dev/ttyUSB0")

    def test_capture_rate_is_capped_at_the_hardware_mode(self) -> None:
        # The GelSight Mini exposes exactly one mode; asking for more cannot work.
        camera = GelSightCamera(device=20, fps=120.0)
        self.assertEqual(camera.fps, NATIVE_FPS)


class NearestFrameTest(unittest.TestCase):
    def _camera(self) -> GelSightCamera:
        camera = GelSightCamera(device=20, fps=25.0)
        camera._frames = [_tactile_frame(index, 100.0 + index * 0.04) for index in range(5)]
        return camera

    def test_picks_the_closest_frame(self) -> None:
        camera = self._camera()
        self.assertEqual(camera.nearest(100.081, max_delta=0.05).frame_number, 2)
        self.assertEqual(camera.nearest(100.0, max_delta=0.05).frame_number, 0)

    def test_rejects_a_frame_further_than_the_limit(self) -> None:
        camera = self._camera()
        with self.assertRaises(RuntimeError) as caught:
            camera.nearest(101.0, max_delta=0.05)
        self.assertIn("ms away", str(caught.exception))


class ForceSwitchTest(unittest.TestCase):
    def test_force_on_records_torque_and_the_raw_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary) / "raw", record_force=True)
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertIn("joint_torque", trajectory.files)
            metadata = json.loads((episode / "metadata.json").read_text())
            self.assertIsNotNone(metadata["force_signal"])

    def test_force_off_omits_torque_entirely_rather_than_zero_filling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary) / "raw", record_force=False)
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertNotIn("joint_torque", trajectory.files)
                self.assertNotIn("raw_force_joint_torque", trajectory.files)
            metadata = json.loads((episode / "metadata.json").read_text())
            self.assertIsNone(metadata["force_signal"])

    def test_a_controller_without_torque_can_still_record(self) -> None:
        # state has no tau_J at all; with the switch off that must not raise.
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary) / "raw", record_force=False)
            self.assertTrue((episode / "trajectory.npz").is_file())


class TactileRecordingTest(unittest.TestCase):
    def test_every_captured_frame_is_stored_not_one_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(
                Path(temporary) / "raw", tactile=TACTILE, rows=4, tactile_per_row=3
            )
            stored = sorted((episode / "frames" / TACTILE_KEY).glob("frame_*.jpg"))
            self.assertEqual(len(stored), 12)
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertEqual(len(trajectory["raw_tactile_timestamp"]), 12)
                self.assertEqual(len(trajectory["tactile_frame_index"]), 4)

    def test_each_row_indexes_the_nearest_stored_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary) / "raw", tactile=TACTILE, rows=3)
            with np.load(episode / "trajectory.npz") as trajectory:
                stamps = trajectory["raw_tactile_timestamp"]
                rows = trajectory["tactile_frame_index"]
                host = trajectory["host_monotonic_timestamp"]
                aligned = trajectory["tactile_aligned_timestamp"]
            for row, (index, target) in enumerate(zip(rows, host)):
                closest = int(np.argmin(np.abs(stamps - target)))
                self.assertEqual(int(index), closest, f"row {row}")
                self.assertAlmostEqual(aligned[row], stamps[index])

    def test_the_report_states_the_tactile_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(
                Path(temporary) / "raw", tactile=TACTILE, rows=5, tactile_per_row=2
            )
            report = json.loads((episode / "sync_report.json").read_text())
            self.assertIsNotNone(report["tactile"])
            self.assertEqual(report["tactile"]["frames"], 10)
            # Two frames per 15 Hz row is 30 Hz.
            self.assertGreater(report["tactile"]["rate_hz"], 20.0)

    def test_metadata_describes_the_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary) / "raw", tactile=TACTILE)
            signal = json.loads((episode / "metadata.json").read_text())["tactile_signal"]
            self.assertEqual(signal["source"], "gelsight_mini")
            self.assertEqual((signal["width"], signal["height"]), (32, 24))

    def test_no_tactile_leaves_the_episode_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary) / "raw", tactile=None)
            self.assertFalse((episode / "frames" / TACTILE_KEY).exists())
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertNotIn("tactile_frame_index", trajectory.files)
            self.assertIsNone(json.loads((episode / "metadata.json").read_text())["tactile_signal"])
            self.assertIsNone(json.loads((episode / "sync_report.json").read_text())["tactile"])

    def test_a_row_with_no_tactile_frame_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = RawEpisodeWriter(
                Path(temporary) / "raw",
                episode_index=0,
                fps=15.0,
                camera_serials={"exterior_image_left": "e", "wrist_image": "w"},
                tactile=TACTILE,
            )
            image = np.full((8, 12, 3), 127, dtype=np.uint8)
            target = writer.started_monotonic
            with self.assertRaises(RuntimeError) as caught:
                writer.add_sample(
                    target_monotonic=target,
                    state={"qpos": np.arange(7), "dq": np.arange(7), "tau_J": np.arange(7), "time_sec": 1.0},
                    action_joint_velocity=np.arange(7),
                    gripper_position=0.5,
                    action_gripper_position=1.0,
                    camera_frames={
                        "exterior_image_left": CameraFrame(image, target, 1.0, 1),
                        "wrist_image": CameraFrame(image, target, 1.0, 1),
                    },
                )
            self.assertIn("no tactile frame", str(caught.exception))


class TactileConversionTest(unittest.TestCase):
    def test_the_aligned_feature_follows_the_recorded_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _write_episode(root / "raw", tactile=TACTILE, rows=3, tactile_per_row=2)
            dataset = _convert(root, root / "raw")

            self.assertIn(TACTILE_KEY, dataset.create_args["features"])
            self.assertEqual(dataset.create_args["features"][TACTILE_KEY]["shape"], (24, 32, 3))
            with np.load(episode / "trajectory.npz") as trajectory:
                rows = trajectory["tactile_frame_index"]
            declared = dataset.create_args["features"][TACTILE_KEY]["shape"]
            for frame, index in zip(dataset.frames, rows):
                # Every stored frame is a constant plate equal to its index.
                self.assertEqual(int(frame[TACTILE_KEY][0, 0, 0]), int(index) % 251)
                # The tactile image must keep its own 4:3 size: LeRobot rejects
                # a frame whose shape disagrees with the declared feature, and
                # the camera loader resizes to the 16:9 DROID size by default.
                self.assertEqual(frame[TACTILE_KEY].shape, declared)

    def test_the_sidecar_keeps_the_full_rate_stream_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _write_episode(root / "raw", tactile=TACTILE, rows=4, tactile_per_row=3)
            dataset = _convert(root, root / "raw")

            sidecar = Path(dataset.root) / "tactile" / "episode_000000.npz"
            self.assertTrue(sidecar.is_file())
            recorded = sorted((episode / "frames" / TACTILE_KEY).glob("frame_*.jpg"))
            with np.load(sidecar) as written:
                blob = written["jpeg_blob"].tobytes()
                offsets = written["jpeg_offsets"]
                self.assertEqual(len(offsets), len(recorded) + 1)
                self.assertEqual(len(written["timestamp"]), 12)
                self.assertEqual(len(written["row_frame_index"]), 4)
                self.assertEqual(str(written["source_episode"]), episode.name)
                for index, path in enumerate(recorded):
                    stored = blob[offsets[index] : offsets[index + 1]]
                    self.assertEqual(stored, path.read_bytes(), f"frame {index}")

    def test_the_report_records_the_rate_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root / "raw", tactile=TACTILE, rows=5, tactile_per_row=2)
            dataset = _convert(root, root / "raw")

            report = json.loads((Path(dataset.root) / "meta" / "tactile_report.json").read_text())
            self.assertEqual(report["episodes_with_tactile"], 1)
            self.assertEqual(report["aligned_feature"], TACTILE_KEY)
            entry = report["episodes"][0]
            self.assertEqual(entry["tactile_frames"], 10)
            self.assertEqual(entry["grid_frames"], 5)
            self.assertGreater(entry["rate_hz"], 20.0)

    def test_an_episode_without_tactile_gets_no_feature_and_no_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root / "raw", tactile=None)
            dataset = _convert(root, root / "raw")

            self.assertNotIn(TACTILE_KEY, dataset.create_args["features"])
            self.assertFalse((Path(dataset.root) / "tactile").exists())

    def test_tactile_is_omitted_when_only_some_episodes_have_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root / "raw", episode_index=0, tactile=TACTILE)
            _write_episode(root / "raw", episode_index=1, tactile=None)
            dataset = _convert(root, root / "raw")

            self.assertNotIn(TACTILE_KEY, dataset.create_args["features"])

    def test_a_truncated_tactile_directory_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _write_episode(root / "raw", tactile=TACTILE, rows=3, tactile_per_row=2)
            frames = sorted((episode / "frames" / TACTILE_KEY).glob("frame_*.jpg"))
            frames[-1].unlink()

            with self.assertRaises(RuntimeError) as caught:
                write_tactile_sidecar(root / "out", 0, episode)
            self.assertIn("mismatch", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
