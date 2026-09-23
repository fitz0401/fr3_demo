import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.convert_lerobot import convert
from fr3_demo.recording import (
    RawEpisodeWriter,
    estimated_bytes_per_frame,
    free_disk_bytes,
    require_free_space,
)


def _write_episode(
    session: Path,
    *,
    include_exterior2: bool = False,
    image_size: tuple[int, int] | None = None,
) -> Path:
    camera_serials = {"exterior_image_left": "external", "wrist_image": "wrist"}
    if include_exterior2:
        camera_serials["exterior_image_2_left"] = "external2"
    writer = RawEpisodeWriter(
        session,
        episode_index=0,
        fps=15.0,
        camera_serials=camera_serials,
        image_size=image_size,
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    for index in range(2):
        target = writer.started_monotonic + index / 15
        frames = {
            "exterior_image_left": CameraFrame(image, target, 1.0 + index / 15, index + 1),
            "wrist_image": CameraFrame(image, target, 1.0 + index / 15, index + 1),
        }
        if include_exterior2:
            frames["exterior_image_2_left"] = CameraFrame(
                np.full_like(image, 200), target, 1.0 + index / 15, index + 1
            )
        writer.add_sample(
            target_monotonic=target,
            state={
                "qpos": np.arange(7),
                "dq": np.arange(7) / 10,
                "tau_J": np.arange(7) * 1.5,
                "time_sec": 20.0 + index / 15,
            },
            action_joint_velocity=np.arange(7) / 100,
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames=frames,
        )
    return writer.finish()


class DiskSpaceTest(unittest.TestCase):
    def test_free_space_is_measured_for_a_directory_that_does_not_exist_yet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "sessions" / "today"

            self.assertEqual(free_disk_bytes(missing), free_disk_bytes(Path(temporary)))

    def test_recording_is_refused_when_the_disk_is_nearly_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            free = free_disk_bytes(Path(temporary))

            self.assertEqual(require_free_space(Path(temporary), 0), free)
            with self.assertRaises(RuntimeError) as raised:
                require_free_space(Path(temporary), free + 10**12)

            message = str(raised.exception)
            self.assertIn("GB free", message)
            self.assertIn("--min-free-gb", message)

    def test_frame_cost_scales_with_stored_pixels(self) -> None:
        native = estimated_bytes_per_frame([(424, 240), (424, 240), (960, 540)])
        resized = estimated_bytes_per_frame([(320, 180)] * 3)

        self.assertGreater(native / resized, 3.0)


class RawRecordingTest(unittest.TestCase):
    def test_episode_is_atomically_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            episode = _write_episode(session)

            self.assertFalse((session / "episode_000000.inprogress").exists())
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["schema_version"], 3)
            self.assertEqual(metadata["force_signal"]["field"], "joint_torque")
            self.assertEqual(metadata["force_signal"]["unit"], "Nm")
            self.assertEqual(metadata["timebase"], "host_monotonic")
            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(
                metadata["camera_transforms"],
                {"exterior_image_left": "none", "wrist_image": "none"},
            )
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertEqual(trajectory["joint_position"].shape, (2, 7))
                self.assertEqual(trajectory["joint_torque"].shape, (2, 7))
                np.testing.assert_allclose(trajectory["joint_torque"][0], np.arange(7) * 1.5)
                self.assertEqual(trajectory["action_joint_velocity"].shape, (2, 7))
                self.assertEqual(trajectory["gripper_position"].shape, (2, 1))
                self.assertIn("host_monotonic_timestamp", trajectory)
                self.assertIn("exterior_camera_hardware_timestamp", trajectory)
                self.assertIn("exterior_camera_frame_number", trajectory)
            report = json.loads((episode / "sync_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["valid"])
            self.assertEqual(report["jpeg_queue"]["dropped"], 0)

    def test_frames_are_stored_at_the_configured_size(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            episode = _write_episode(session, image_size=(320, 180))

            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["image_size"], {"width": 320, "height": 180, "resample": "bicubic"}
            )
            for key in ("exterior_image_left", "wrist_image"):
                for path in (episode / "frames" / key).glob("frame_*.jpg"):
                    with Image.open(path) as stored:
                        self.assertEqual(stored.size, (320, 180))

    def test_native_resolution_is_kept_when_no_size_is_configured(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode(Path(temporary), image_size=None)

            self.assertIsNone(json.loads((episode / "metadata.json").read_text(encoding="utf-8"))["image_size"])
            path = next((episode / "frames" / "wrist_image").glob("frame_*.jpg"))
            with Image.open(path) as stored:
                self.assertEqual(stored.size, (12, 8))

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

    def test_collector_runs_independent_samplers_and_writes_fixed_grid(self) -> None:
        class FakeCameras:
            wrist_rotate_180 = False

            def __init__(self) -> None:
                self.serials = {"exterior_image_left": "external", "wrist_image": "wrist"}
                self.frame_number = 0

            def snapshot(self, _max_age=0.25):
                now = time.monotonic()
                image = np.zeros((4, 6, 3), dtype=np.uint8)
                return {
                    key: CameraFrame(image, now, now - 1.0, self.frame_number)
                    for key in self.serials
                }

            def nearest(self, target, _max_delta, _after):
                self.frame_number += 1
                image = np.zeros((4, 6, 3), dtype=np.uint8)
                return {
                    key: CameraFrame(
                        image,
                        target + 0.003,
                        target - 1.0,
                        self.frame_number,
                        "global_time",
                        target,
                    )
                    for key in self.serials
                }

            @property
            def clock_estimates(self):
                return {
                    key: {"scale": 1.0, "offset": 1.0, "sample_count": 10, "residual_p95_ms": 1.0}
                    for key in self.serials
                }

        class FakeBamboo:
            def __init__(self, **_kwargs):
                pass

            def get_joint_states(self):
                now = time.monotonic()
                return {"qpos": [now] * 7, "dq": [1.0] * 7, "tau_J": [2.0] * 7, "time_sec": now}

            def get_gripper_state(self):
                return {"success": True, "state": {"width": 0.0425}}

            def close(self):
                pass

        from fr3_demo.recording import DemoCollector

        with tempfile.TemporaryDirectory() as temporary, patch("bamboo.BambooFrankaClient", FakeBamboo):
            collector = DemoCollector(
                FakeCameras(),
                Path(temporary),
                "127.0.0.1",
                5555,
                fps=10.0,
                alignment_delay_ms=20.0,
                camera_max_delta_ms=25.0,
                robot_sample_hz=100.0,
                robot_max_gap_ms=30.0,
                gripper_sample_hz=100.0,
                min_free_gb=0.0,
            )
            collector.start()
            seen_frames: list[dict] = []
            collector.set_frame_observer(seen_frames.append)
            time.sleep(0.04)
            collector.start_episode()
            time.sleep(0.24)
            episode = collector.stop_episode()

            with np.load(episode / "trajectory.npz") as trajectory:
                frame_count = len(trajectory["timestamp"])
                self.assertGreaterEqual(frame_count, 2)
                np.testing.assert_allclose(np.diff(trajectory["timestamp"]), 0.1, atol=1e-6)
                self.assertEqual(trajectory["joint_torque"].shape[1], 7)
                np.testing.assert_allclose(trajectory["joint_torque"], 2.0)
            report = json.loads((episode / "sync_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["valid"])
            # A viewer sees exactly the frames that were recorded.
            self.assertEqual(len(seen_frames), frame_count)
            self.assertEqual(set(seen_frames[0]), {"exterior_image_left", "wrist_image"})

            # A viewer that fails is dropped; the episode still finishes.
            def explode(_frames: dict) -> None:
                raise RuntimeError("viewer crashed")

            collector.set_frame_observer(explode)
            collector.start_episode()
            time.sleep(0.24)
            second = collector.stop_episode()
            collector.close()

            self.assertIsNone(collector._frame_observer)
            self.assertTrue(json.loads((second / "metadata.json").read_text(encoding="utf-8"))["complete"])


if __name__ == "__main__":
    unittest.main()
