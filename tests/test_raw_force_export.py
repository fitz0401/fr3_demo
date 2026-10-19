"""The raw force sidecar, the weight log, and the datasets-version workaround."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.convert_lerobot import (
    convert,
    restore_sequence_feature_tags,
    weights_row,
    write_raw_force,
)
from fr3_demo.recording import (
    RawEpisodeWriter,
    RobotSample,
    override_episode_outcome,
    record_episode_weights,
)

CLOCK_ESTIMATES = {
    "cameras": {},
    "robot_controller": {"scale": 1.000002, "offset": 3286664.74, "sample_count": 1200},
}


# Enough 1 ms samples to bracket both 15 Hz frames (20.0000 and 20.0667).
RAW_SAMPLE_COUNT = 80


def _raw_force_samples(writer: RawEpisodeWriter, count: int = RAW_SAMPLE_COUNT) -> list[RobotSample]:
    """A 1 kHz stream spanning the episode, with torque ramping per joint."""

    samples = []
    for index in range(count):
        controller = 20.0 - 0.005 + index * 0.001
        samples.append(
            RobotSample(
                qpos=np.arange(7, dtype=np.float32) + index * 0.001,
                dq=np.zeros(7, dtype=np.float32),
                tau_J=np.arange(7, dtype=np.float32) * 1.5 + index * 0.01,
                controller_timestamp=controller,
                request_send_monotonic=writer.started_monotonic + index * 0.001,
                response_receive_monotonic=writer.started_monotonic + index * 0.001 + 0.0002,
            )
        )
    return samples


def _write_episode(
    session: Path,
    *,
    episode_index: int = 0,
    raw_force: bool = True,
    weights: dict[str, float] | None = None,
    torque_offset: float = 0.0,
) -> Path:
    writer = RawEpisodeWriter(
        session,
        episode_index=episode_index,
        fps=15.0,
        camera_serials={"exterior_image_left": "external", "wrist_image": "wrist"},
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    samples = _raw_force_samples(writer) if raw_force else []
    for index in range(2):
        target = writer.started_monotonic + index / 15
        controller = 20.0 + index / 15
        if raw_force:
            # Interpolate each joint from the raw stream exactly as the recorder
            # does, so the alignment check has something truthful to verify.
            times = [s.controller_timestamp for s in samples]
            tau_J = np.array(
                [np.interp(controller, times, [float(s.tau_J[joint]) for s in samples]) for joint in range(7)],
                dtype=np.float32,
            )
        else:
            tau_J = np.arange(7, dtype=np.float32) * 1.5
        writer.add_sample(
            target_monotonic=target,
            state={
                "qpos": np.arange(7),
                "dq": np.arange(7) / 10,
                "tau_J": tau_J + torque_offset,
                "time_sec": controller,
            },
            action_joint_velocity=np.arange(7) / 100,
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames={
                "exterior_image_left": CameraFrame(image, target, 1.0 + index / 15, index + 1),
                "wrist_image": CameraFrame(image, target, 1.0 + index / 15, index + 1),
            },
        )
    if raw_force:
        writer.set_raw_force(samples)
    episode = writer.finish(CLOCK_ESTIMATES)
    metadata = json.loads((episode / "metadata.json").read_text())
    metadata["language_instruction"] = "pour lillet into the blue cup"
    (episode / "metadata.json").write_text(json.dumps(metadata))
    if weights is not None:
        record_episode_weights(episode, weights)
    return episode


def _convert(root: Path, *data_dirs: Path) -> Path:
    class FakeDataset:
        instance = None

        @classmethod
        def create(cls, **kwargs):
            cls.instance = cls()
            cls.instance.root = root / "converted"
            cls.instance.frames = []
            return cls.instance

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            pass

    with patch("fr3_demo.convert_lerobot._load_lerobot_dataset", return_value=FakeDataset):
        convert(list(data_dirs), "test/fr3", output_root=root / "datasets")
    return Path(FakeDataset.instance.root)


class RawForceSidecarTest(unittest.TestCase):
    def test_sidecar_carries_the_raw_arrays_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _write_episode(root / "raw", weights={"bottle_before_g": 876.0})
            output = _convert(root, root / "raw")

            sidecar = output / "raw_force" / "episode_000000.npz"
            self.assertTrue(sidecar.is_file())
            with np.load(sidecar) as written, np.load(episode / "trajectory.npz") as source:
                for key in (
                    "raw_force_joint_torque",
                    "raw_force_joint_position",
                    "raw_force_robot_timestamp",
                    "raw_force_request_send_timestamp",
                    "raw_force_response_receive_timestamp",
                    "robot_timestamp",
                    "host_monotonic_timestamp",
                ):
                    self.assertIn(key, written.files)
                    np.testing.assert_array_equal(written[key], source[key])
                    self.assertEqual(written[key].dtype, source[key].dtype)
                self.assertEqual(written["raw_force_joint_torque"].shape, (RAW_SAMPLE_COUNT, 7))
                self.assertEqual(written["robot_timestamp"].shape, (2,))
                self.assertEqual(str(written["source_episode"]), episode.name)
                self.assertAlmostEqual(float(written["robot_clock_scale"]), 1.000002)
                self.assertAlmostEqual(float(written["robot_clock_offset"]), 3286664.74)

    def test_alignment_check_rejects_a_mispaired_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = _write_episode(root / "raw", weights={"bottle_before_g": 876.0})
            # Same raw stream, frames shifted by a whole Nm: the signature of
            # raw arrays paired with another episode's frames.
            bad = _write_episode(root / "other", torque_offset=5.0)
            report = json.loads((good / "sync_report.json").read_text())

            self.assertIsNotNone(write_raw_force(root / "ok", 0, good, report))
            with self.assertRaises(RuntimeError) as caught:
                write_raw_force(root / "bad", 0, bad, report)
            self.assertIn("different episode", str(caught.exception))

    def test_a_frame_before_the_raw_stream_is_skipped_not_failed(self) -> None:
        """The recorder can interpolate frame 0 from a poll taken just before
        the raw buffer starts retaining samples. np.interp would clamp there,
        so that frame is excluded from the comparison instead of failing it."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _write_episode(root / "raw")
            path = episode / "trajectory.npz"
            with np.load(path) as loaded:
                arrays = dict(loaded.items())
            # Drop the leading raw samples so the stream begins after frame 0,
            # exactly as it does when the buffer was not yet retaining them.
            # The remaining samples keep their original timestamps, so every
            # other frame must still line up.
            frame_zero = arrays["robot_timestamp"][0]
            keep = arrays["raw_force_robot_timestamp"] > frame_zero
            for key in list(arrays):
                if key.startswith("raw_force_"):
                    arrays[key] = arrays[key][keep]
            # Make frame 0 disagree wildly with the clamp, so a check that does
            # not skip it cannot pass by accident.
            arrays["joint_torque"][0] = arrays["joint_torque"][0] + 50.0
            np.savez(path, **arrays)
            report = json.loads((episode / "sync_report.json").read_text())

            entry = write_raw_force(root / "out", 0, episode, report)

            self.assertIsNotNone(entry)
            self.assertEqual(entry["frames_outside_raw_span"], 1)
            self.assertLess(entry["alignment_max_abs_diff_nm"], 1e-2)

    def test_episode_without_raw_force_gets_no_sidecar_and_is_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root / "raw", raw_force=False)
            output = _convert(root, root / "raw")

            self.assertFalse((output / "raw_force" / "episode_000000.npz").exists())
            report = json.loads((output / "meta" / "raw_force_report.json").read_text())
            self.assertEqual(report["episodes_with_raw_force"], 0)
            self.assertEqual(len(report["episodes_without_raw_force"]), 1)

    def test_report_records_rate_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root / "raw")
            output = _convert(root, root / "raw")

            report = json.loads((output / "meta" / "raw_force_report.json").read_text())
            entry = report["episodes"][0]
            self.assertEqual(entry["raw_samples"], RAW_SAMPLE_COUNT)
            self.assertAlmostEqual(entry["raw_rate_hz"], 1000.0, places=3)
            self.assertAlmostEqual(entry["max_gap_ms"], 1.0, places=3)
            self.assertLess(entry["alignment_max_abs_diff_nm"], 1e-2)


class EpisodeWeightsLogTest(unittest.TestCase):
    def test_readings_are_written_per_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(
                root / "raw",
                weights={
                    "bottle_before_g": 876.0,
                    "cup_before_g": 80.0,
                    "bottle_after_g": 846.0,
                    "cup_after_g": 111.0,
                },
            )
            output = _convert(root, root / "raw")

            lines = (output / "meta" / "episode_weights.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["episode_index"], 0)
            self.assertEqual(row["bottle_before_g"], 876.0)
            self.assertEqual(row["cup_after_g"], 111.0)
            self.assertEqual(row["poured_g"], 31.0)
            self.assertEqual(row["unit"], "g")
            self.assertEqual(row["source"], "operator_scale")

    def test_missing_weights_become_null_never_zero(self) -> None:
        row = weights_row(3, Path("episode_000003"), {})
        self.assertEqual(row["episode_index"], 3)
        for field in ("bottle_before_g", "cup_before_g", "bottle_after_g", "cup_after_g", "poured_g", "spilled_g"):
            self.assertIsNone(row[field], field)
        self.assertIsNone(row["unit"])

    def test_every_episode_gets_a_line_in_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root / "raw", episode_index=0, weights={"bottle_before_g": 900.0})
            _write_episode(root / "raw", episode_index=1)
            output = _convert(root, root / "raw")

            rows = [json.loads(line) for line in (output / "meta" / "episode_weights.jsonl").read_text().splitlines()]
            self.assertEqual([row["episode_index"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["bottle_before_g"], 900.0)
            self.assertIsNone(rows[1]["bottle_before_g"])


class ReviewedOutcomeTest(unittest.TestCase):
    """A corrected episode must never look like a clean measurement."""

    def _episode(self, root: Path) -> Path:
        return _write_episode(
            root / "raw",
            weights={
                "bottle_before_g": 739.0,
                "cup_before_g": 223.0,
                "bottle_after_g": 738.0,
                "cup_after_g": 219.0,
            },
        )

    def test_override_keeps_the_readings_and_records_what_it_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = self._episode(Path(temporary))
            before = json.loads((episode / "metadata.json").read_text())["weights"]
            self.assertEqual(before["poured_g"], -4.0)
            self.assertEqual(before["spilled_g"], 5.0)

            override_episode_outcome(episode, poured_g=0.0, spilled_g=0.0, reason="no pour")

            after = json.loads((episode / "metadata.json").read_text())["weights"]
            self.assertEqual(after["poured_g"], 0.0)
            self.assertEqual(after["spilled_g"], 0.0)
            # readings untouched
            self.assertEqual(after["bottle_after_g"], 738.0)
            self.assertEqual(after["cup_after_g"], 219.0)
            # superseded values preserved
            self.assertEqual(after["measured_poured_g"], -4.0)
            self.assertEqual(after["measured_spilled_g"], 5.0)
            self.assertEqual(after["outcome_source"], "manual_review")
            self.assertEqual(after["outcome_reason"], "no pour")

    def test_a_second_override_does_not_overwrite_the_measured_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = self._episode(Path(temporary))
            override_episode_outcome(episode, poured_g=0.0, spilled_g=0.0, reason="first")
            override_episode_outcome(episode, poured_g=1.0, spilled_g=0.0, reason="second")

            weights = json.loads((episode / "metadata.json").read_text())["weights"]
            self.assertEqual(weights["measured_poured_g"], -4.0)
            self.assertEqual(weights["outcome_reason"], "second")

    def test_the_correction_reaches_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = self._episode(root)
            override_episode_outcome(episode, poured_g=0.0, spilled_g=0.0, reason="no pour: see review")
            output = _convert(root, root / "raw")

            row = json.loads((output / "meta" / "episode_weights.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["poured_g"], 0.0)
            self.assertEqual(row["spilled_g"], 0.0)
            self.assertEqual(row["measured_poured_g"], -4.0)
            self.assertEqual(row["outcome_source"], "manual_review")
            self.assertEqual(row["outcome_reason"], "no pour: see review")

    def test_an_uncorrected_episode_carries_no_outcome_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._episode(root)
            output = _convert(root, root / "raw")

            row = json.loads((output / "meta" / "episode_weights.jsonl").read_text().splitlines()[0])
            for field in ("outcome_source", "outcome_reason", "measured_poured_g"):
                self.assertNotIn(field, row)

    def test_an_episode_without_weights_cannot_be_corrected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _write_episode(root / "raw")
            with self.assertRaises(RuntimeError):
                override_episode_outcome(episode, poured_g=0.0, spilled_g=0.0, reason="x")


class SequenceFeatureTagTest(unittest.TestCase):
    """datasets 3.x refuses the "List" tag that datasets 4.x+ writes."""

    def _write_parquet(self, path: Path, type_name: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {"joint_torque": pa.array([[0.0] * 7, [1.0] * 7], type=pa.list_(pa.float32(), 7))}
        )
        features = {
            "joint_torque": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": type_name}
        }
        metadata = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(metadata), path)

    def _features_of(self, path: Path) -> dict:
        import pyarrow.parquet as pq

        metadata = pq.read_schema(path).metadata
        return json.loads(metadata[b"huggingface"])["info"]["features"]

    def test_list_tag_is_rewritten_to_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            self._write_parquet(path, "List")

            rewritten = restore_sequence_feature_tags(root)

            self.assertEqual(rewritten, [path])
            self.assertEqual(self._features_of(path)["joint_torque"]["_type"], "Sequence")

    def test_values_survive_the_rewrite(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            self._write_parquet(path, "List")
            before = pq.read_table(path)

            restore_sequence_feature_tags(root)

            after = pq.read_table(path)
            self.assertEqual(before.column("joint_torque").to_pylist(), after.column("joint_torque").to_pylist())
            self.assertTrue(before.schema.remove_metadata().equals(after.schema.remove_metadata()))

    def test_an_uncompressed_parquet_can_be_rewritten(self) -> None:
        """pyarrow reports no compression as "UNCOMPRESSED" but writes "none"."""

        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table({"joint_torque": pa.array([[1.0] * 7], type=pa.list_(pa.float32(), 7))})
            features = {
                "joint_torque": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "List"}
            }
            metadata = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
            pq.write_table(table.replace_schema_metadata(metadata), path, compression="none")

            self.assertEqual(restore_sequence_feature_tags(root), [path])
            self.assertEqual(self._features_of(path)["joint_torque"]["_type"], "Sequence")

    def test_a_failed_rewrite_leaves_the_original_intact(self) -> None:
        """pq.write_table truncates its destination before validating its
        arguments, so rewriting in place destroys the file when a write is
        rejected. The original must survive a failure untouched."""

        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            self._write_parquet(path, "List")
            before = path.read_bytes()

            with (
                patch("pyarrow.parquet.write_table", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                restore_sequence_feature_tags(root)

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(pq.read_table(path).num_rows, 2)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_an_already_portable_dataset_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            self._write_parquet(path, "Sequence")

            self.assertEqual(restore_sequence_feature_tags(root), [])

    def test_no_parquet_files_needs_no_pyarrow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(restore_sequence_feature_tags(Path(temporary)), [])


if __name__ == "__main__":
    unittest.main()
