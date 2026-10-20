import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fr3_demo.annotate import main as annotate_main
from fr3_demo.cameras import CameraFrame
from fr3_demo.recording import (
    WEIGHT_READINGS,
    RawEpisodeWriter,
    RobotSample,
    record_episode_weights,
    update_episode_metadata,
)
from fr3_demo.teleop import WEIGHTS_AFTER, WEIGHTS_BEFORE, prompt_weights


def _write_episode_with_raw_force(session: Path, raw: list[RobotSample]) -> Path:
    """A two-frame episode carrying an independent raw force stream."""

    writer = RawEpisodeWriter(
        session,
        episode_index=0,
        fps=15.0,
        camera_serials={"exterior_image_left": "external", "wrist_image": "wrist"},
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    for index in range(2):
        target = writer.started_monotonic + index / 15
        frames = {
            key: CameraFrame(image, target, 1.0 + index / 15, index + 1)
            for key in ("exterior_image_left", "wrist_image")
        }
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
    writer.set_raw_force(raw)
    return writer.finish()


def _episode(root: Path) -> Path:
    episode = root / "session_test" / "episode_000000"
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text(
        json.dumps({"complete": True, "frame_count": 2, "language_instruction": None}), encoding="utf-8"
    )
    return episode


class RawForceStreamTest(unittest.TestCase):
    """The raw stream must keep its own rate, independent of the 15 Hz grid."""

    def _samples(self, count: int, period: float) -> list[RobotSample]:
        return [
            RobotSample(
                qpos=np.full(7, float(i), dtype=np.float32),
                dq=np.zeros(7, dtype=np.float32),
                tau_J=np.full(7, -i * 0.5, dtype=np.float32),
                controller_timestamp=100.0 + i * period,
                request_send_monotonic=10.0 + i * period,
                response_receive_monotonic=10.0 + i * period + 0.0005,
            )
            for i in range(count)
        ]

    def test_every_arm_sample_is_written_with_its_own_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = self._samples(40, 1 / 60)
            episode = _write_episode_with_raw_force(Path(temporary), samples)

            with np.load(episode / "trajectory.npz") as trajectory:
                torque = trajectory["raw_force_joint_torque"]
                self.assertEqual(torque.shape, (40, 7))
                # Far more rows than the 15 Hz grid, which has 2 frames here.
                self.assertEqual(len(trajectory["joint_torque"]), 2)
                np.testing.assert_allclose(torque[3], np.full(7, -1.5), atol=1e-6)
                self.assertEqual(trajectory["raw_force_joint_position"].shape, (40, 7))
                stamps = trajectory["raw_force_robot_timestamp"]
                self.assertEqual(len(stamps), 40)
                # The achieved rate is recoverable from the data itself.
                self.assertAlmostEqual(1 / float(np.median(np.diff(stamps))), 60.0, places=3)
                self.assertIn("raw_force_request_send_timestamp", trajectory)
                self.assertIn("raw_force_response_receive_timestamp", trajectory)

    def test_metadata_says_the_raw_rate_is_the_polling_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode_with_raw_force(Path(temporary), self._samples(5, 1 / 60))

            force = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))["force_signal"]

            self.assertEqual(force["raw_field"], "raw_force_joint_torque")
            self.assertIn("request/reply", force["raw_note"])

    def test_an_episode_without_raw_samples_omits_the_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_episode_with_raw_force(Path(temporary), [])

            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertNotIn("raw_force_joint_torque", trajectory.files)
                self.assertIn("joint_torque", trajectory.files)


class WeightRecordingTest(unittest.TestCase):
    def test_metadata_update_is_atomic_and_keeps_other_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))

            update_episode_metadata(episode, note="x")

            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["note"], "x")
            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(list(episode.glob("*.tmp")), [])

    def test_poured_and_spilled_are_derived_from_the_four_readings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))

            weights = record_episode_weights(
                episode,
                {
                    "bottle_before_g": 900.0,
                    "cup_before_g": 120.0,
                    "bottle_after_g": 840.0,
                    "cup_after_g": 170.0,
                },
            )["weights"]

            # 50 g reached the cup; the bottle lost 60 g, so 10 g went elsewhere.
            self.assertAlmostEqual(weights["poured_g"], 50.0)
            self.assertAlmostEqual(weights["spilled_g"], 10.0)
            self.assertEqual(weights["unit"], "g")
            self.assertEqual(weights["source"], "operator_scale")

    def test_poured_needs_only_the_cup_and_spilled_needs_the_bottle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))

            weights = record_episode_weights(
                episode, {"cup_before_g": 0.0, "cup_after_g": 48.5}
            )["weights"]

            self.assertAlmostEqual(weights["poured_g"], 48.5)
            # Without bottle readings, spill cannot be known and is not invented.
            self.assertNotIn("spilled_g", weights)

    def test_a_partial_set_of_readings_derives_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))

            weights = record_episode_weights(episode, {"bottle_before_g": 900.0})["weights"]

            self.assertEqual(weights["bottle_before_g"], 900.0)
            self.assertNotIn("poured_g", weights)
            self.assertNotIn("spilled_g", weights)

    def test_prompt_asks_for_each_reading_and_pauses_the_stream_around_input(self) -> None:
        events: list[str] = []
        answers = iter(["900", "120"])

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=lambda _: events.append("input") or next(answers)),
        ):
            readings = prompt_weights(
                WEIGHTS_BEFORE,
                pause_stream=lambda: events.append("pause"),
                resume_stream=lambda: events.append("resume"),
            )

        self.assertEqual(readings, {"bottle_before_g": 900.0, "cup_before_g": 120.0})
        # The watchdog would brake the arm if input() ran with a live stream.
        self.assertEqual(events, ["pause", "input", "input", "resume"])

    def test_before_and_after_prompts_cover_all_four_readings(self) -> None:
        self.assertEqual(
            [key for key, _ in WEIGHTS_BEFORE + WEIGHTS_AFTER],
            [key for key, _ in WEIGHT_READINGS[:1] + WEIGHT_READINGS[1:2] + WEIGHT_READINGS[2:]],
        )

    def test_a_blank_answer_skips_only_that_reading(self) -> None:
        answers = iter(["", "120"])

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=lambda _: next(answers)),
        ):
            self.assertEqual(prompt_weights(WEIGHTS_BEFORE), {"cup_before_g": 120.0})

    def test_the_stream_resumes_even_when_the_operator_interrupts(self) -> None:
        resumed = []

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            self.assertEqual(prompt_weights(WEIGHTS_AFTER, resume_stream=lambda: resumed.append(True)), {})

        self.assertEqual(resumed, [True])

    def test_a_non_interactive_run_never_blocks(self) -> None:
        with patch("sys.stdin.isatty", return_value=False), patch("builtins.input") as asked:
            self.assertEqual(prompt_weights(WEIGHTS_BEFORE), {})

        asked.assert_not_called()

    def test_annotate_fills_weights_in_afterwards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _episode(root)
            answers = iter(["900", "120", "840", "170"])

            with patch("builtins.input", side_effect=lambda _: next(answers)):
                annotate_main(["--data-dir", str(root), "--language", "pour", "--weights"])

            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(metadata["weights"]["poured_g"], 50.0)
            self.assertAlmostEqual(metadata["weights"]["spilled_g"], 10.0)
            self.assertEqual(metadata["language_instruction"], "pour")

    def test_annotate_keeps_existing_weights_unless_replacing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _episode(root)
            record_episode_weights(episode, {"cup_before_g": 0.0, "cup_after_g": 10.0})

            with patch("builtins.input", side_effect=AssertionError("must not ask")):
                annotate_main(["--data-dir", str(root), "--language", "pour", "--weights"])

            self.assertAlmostEqual(
                json.loads((episode / "metadata.json").read_text(encoding="utf-8"))["weights"]["poured_g"],
                10.0,
            )


if __name__ == "__main__":
    unittest.main()


class RumblerFallbackTest(unittest.TestCase):
    """A missing vibration motor must not cost a demonstration session."""

    def test_a_missing_device_degrades_to_silence(self) -> None:
        from fr3_demo.feedback import SilentRumbler, open_rumbler

        with patch(
            "fr3_demo.feedback.Rumbler",
            side_effect=RuntimeError("No force-feedback input device containing 'BETOP' was found"),
        ):
            rumbler = open_rumbler(None)

        self.assertIsInstance(rumbler, SilentRumbler)
        # Every call site in the control loop must stay safe to call.
        rumbler.recording_started()
        rumbler.recording_stopped()
        rumbler.error()
        rumbler.play(2)

    def test_a_working_device_is_returned_unchanged(self) -> None:
        from fr3_demo.feedback import open_rumbler

        sentinel = object()
        with patch("fr3_demo.feedback.Rumbler", return_value=sentinel):
            self.assertIs(open_rumbler("/dev/input/event9"), sentinel)
