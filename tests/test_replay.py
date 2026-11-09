import argparse
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from fr3_demo.cameras import CameraFrame
from fr3_demo.joystick import JoystickSnapshot
from fr3_demo.recording import RawEpisodeWriter, RobotSample
from fr3_demo.replay import (
    REPLAY_ACTION_SOURCE,
    ReplayOutcome,
    ReplayPath,
    link_replay,
    replay_command,
    run_replay,
)
from fr3_demo.teleop import HOME_JOINTS, Mapping, ReplayControls, _create_parser, _validate_args

START = np.array([0.0, -0.7, 0.0, -2.3, 0.0, 1.6, 0.8])


def _demo_episode(folder: Path, *, raw: bool = True, gripper=(1.0, 1.0), speed: float = 0.2) -> Path:
    """An episode in which joint 1 turns at ``speed`` for one second, the rest still."""

    episode = folder / "episode_000000"
    episode.mkdir()
    (episode / "metadata.json").write_text(json.dumps({"episode_index": 0}))
    grid = np.arange(0.0, 1.0 + 1e-9, 1 / 15)
    arrays = {
        "robot_timestamp": 50.0 + grid,
        "joint_position": START + np.outer(speed * grid, np.eye(7)[0]),
        "gripper_position": np.linspace(*gripper, len(grid))[:, None],
        "action_gripper_position": np.zeros((len(grid), 1)),  # pressing A on a held tool moves nothing
    }
    if raw:
        stamps = np.arange(0.0, 1.0 + 1e-9, 0.001)
        arrays["raw_force_robot_timestamp"] = 50.0 + stamps
        arrays["raw_force_joint_position"] = START + np.outer(speed * stamps, np.eye(7)[0])
    np.savez(episode / "trajectory.npz", **arrays)
    return episode


class ReplayPathTest(unittest.TestCase):
    def test_reads_the_full_rate_path_and_its_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = ReplayPath.from_episode(_demo_episode(Path(temporary)))
        self.assertAlmostEqual(path.duration, 1.0, places=6)
        np.testing.assert_allclose(path.start, START)
        position, velocity = path.at(0.5)
        np.testing.assert_allclose(position, START + 0.1 * np.eye(7)[0], atol=1e-6)
        np.testing.assert_allclose(velocity, 0.2 * np.eye(7)[0], atol=1e-6)

    def test_it_is_still_before_and_after(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = ReplayPath.from_episode(_demo_episode(Path(temporary)))
        for elapsed, pose in ((-1.0, START), (5.0, START + 0.2 * np.eye(7)[0])):
            position, velocity = path.at(elapsed)
            np.testing.assert_allclose(position, pose, atol=1e-6)
            np.testing.assert_array_equal(velocity, np.zeros(7))

    def test_falls_back_to_the_grid_without_raw_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = ReplayPath.from_episode(_demo_episode(Path(temporary), raw=False))
        position, velocity = path.at(0.5)
        np.testing.assert_allclose(position, START + 0.1 * np.eye(7)[0], atol=1e-6)
        np.testing.assert_allclose(velocity[0], 0.2, atol=1e-3)

    def test_a_demo_that_moves_the_gripper_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "gripper"):
            ReplayPath.from_episode(_demo_episode(Path(temporary), gripper=(1.0, 0.0)))


class ReplayCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.path = ReplayPath.from_episode(_demo_episode(Path(temporary)))

    def test_on_the_path_it_is_the_demonstrated_velocity(self) -> None:
        reference, velocity = self.path.at(0.5)
        np.testing.assert_allclose(replay_command(self.path, 0.5, reference, 0.35), velocity, atol=1e-9)

    def test_off_the_path_it_pulls_back(self) -> None:
        reference, velocity = self.path.at(0.5)
        behind = reference - 0.01 * np.eye(7)[1]
        command = replay_command(self.path, 0.5, behind, 0.35, gain=3.0)
        np.testing.assert_allclose(command, velocity + 0.03 * np.eye(7)[1], atol=1e-9)

    def test_too_fast_is_scaled_as_a_whole(self) -> None:
        reference, _ = self.path.at(0.5)
        far = reference - np.array([0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
        command = replay_command(self.path, 0.5, far, 0.35, gain=3.0)
        # Unscaled it would be (0.2, 0.6, 0.3, 0, ...): the fastest joint is held to the limit, the rest keep pace.
        self.assertAlmostEqual(float(np.max(np.abs(command))), 0.35)
        np.testing.assert_allclose(command[:3] / command[1], [1 / 3, 1.0, 0.5], atol=1e-9)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TrackingRobot:
    """An arm that moves exactly as commanded, one stream period per command."""

    def __init__(self, clock: FakeClock, start: np.ndarray, follows: bool = True) -> None:
        self.clock = clock
        self.q = np.array(start, dtype=float)
        self.follows = follows
        self.commands: list[np.ndarray] = []
        self.yielded: list[str] = []
        self.failure: Exception | None = None

    def state(self) -> dict:
        return {"qpos": self.q.copy()}

    def stream_velocity(self, velocity: np.ndarray) -> None:
        if self.failure is not None:
            raise self.failure
        self.commands.append(np.array(velocity))
        if self.follows:
            self.q = self.q + np.asarray(velocity) / 30.0

    def yield_stream(self, reason: str) -> None:
        self.yielded.append(reason)


class RunReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.path = ReplayPath.from_episode(_demo_episode(Path(temporary)))
        self.clock = FakeClock()

    def _run(self, robot, **options) -> ReplayOutcome:
        return run_replay(
            robot,
            self.path,
            rate_hz=30.0,
            max_speed=0.35,
            max_path_error=0.05,
            clock=self.clock,
            sleep=self.clock.sleep,
            **options,
        )

    def test_follows_the_demonstration_to_its_end_and_records_each_command(self) -> None:
        robot = TrackingRobot(self.clock, START)
        recorded: list[np.ndarray] = []
        outcome = self._run(robot, on_command=recorded.append)
        self.assertTrue(outcome.completed)
        self.assertLess(outcome.max_path_error_rad, 0.01)
        np.testing.assert_allclose(robot.q, START + 0.2 * np.eye(7)[0], atol=0.006)
        np.testing.assert_array_equal(robot.commands[-1], np.zeros(7))
        self.assertEqual(len(recorded), len(robot.commands))
        self.assertEqual(robot.yielded, [])

    def test_the_operator_stops_it_without_yielding(self) -> None:
        robot = TrackingRobot(self.clock, START)
        calls = iter([None, None, "you pressed X or B"])
        outcome = self._run(robot, stop_reason=lambda: next(calls))
        self.assertFalse(outcome.completed)
        self.assertTrue(outcome.by_operator)
        self.assertEqual(outcome.reason, "you pressed X or B")
        np.testing.assert_array_equal(robot.commands[-1], np.zeros(7))
        self.assertEqual(robot.yielded, [])

    def test_the_gel_guard_makes_it_yield(self) -> None:
        robot = TrackingRobot(self.clock, START)
        guard = MagicMock()
        guard.check.side_effect = [None, None, "the gel reached level 1.02"]
        outcome = self._run(robot, gel_guard=guard)
        self.assertFalse(outcome.completed)
        self.assertFalse(outcome.by_operator)
        self.assertEqual(robot.yielded, ["the gel reached level 1.02"])

    def test_an_arm_held_off_the_path_stops_it(self) -> None:
        robot = TrackingRobot(self.clock, START, follows=False)
        outcome = self._run(robot)
        self.assertFalse(outcome.completed)
        self.assertIn("off the demonstration's path", outcome.reason)
        self.assertEqual(len(robot.yielded), 1)
        # Stuck at the start while the path moves on at 0.2 rad/s: about 0.25 s in.
        self.assertGreater(outcome.max_path_error_rad, 0.05)
        self.assertLess(outcome.duration_s, 0.4)

    def test_a_stream_bamboo_ended_reports_bamboo_s_reason(self) -> None:
        robot = TrackingRobot(self.clock, START)
        robot.failure = RuntimeError("Streaming control stopped: contact guard: 21.3 N for 0.06 N s")
        outcome = self._run(robot)
        self.assertFalse(outcome.completed)
        self.assertIn("contact guard", outcome.reason)


class LinkReplayTest(unittest.TestCase):
    def test_both_episodes_name_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            demo, replay = Path(temporary) / "episode_000000", Path(temporary) / "episode_000001"
            for folder in (demo, replay):
                folder.mkdir()
                (folder / "metadata.json").write_text(json.dumps({"complete": True}))
            outcome = ReplayOutcome(False, "the gel reached level 1.02", 3.2, 0.012)
            link_replay(demo, replay, outcome, {"path_gain": 3.0})
            link_replay(demo, replay, ReplayOutcome(True, None, 5.0, 0.004), {"path_gain": 3.0})
            demo_metadata = json.loads((demo / "metadata.json").read_text())
            replay_metadata = json.loads((replay / "metadata.json").read_text())
        self.assertEqual(
            demo_metadata["replays"],
            [{"episode": "episode_000001", "completed": False}, {"episode": "episode_000001", "completed": True}],
        )
        self.assertTrue(replay_metadata["complete"])
        self.assertEqual(replay_metadata["replay"]["path_gain"], 3.0)
        self.assertTrue(replay_metadata["replay"]["completed"])


def _write_grid_episode(session: Path, state_extra: dict, raw: list[RobotSample] | None = None, **writer_options):
    writer = RawEpisodeWriter(
        session,
        episode_index=0,
        fps=15.0,
        camera_serials={"exterior_image_left": "external", "wrist_image": "wrist"},
        image_size=None,
        **writer_options,
    )
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    for index in range(2):
        target = writer.started_monotonic + index / 15
        writer.add_sample(
            target_monotonic=target,
            state={
                "qpos": np.zeros(7),
                "dq": np.zeros(7),
                "tau_J": np.zeros(7),
                "time_sec": 1.0 + index / 15,
                **state_extra,
            },
            action_joint_velocity=np.zeros(7),
            gripper_position=0.5,
            action_gripper_position=1.0,
            camera_frames={
                "exterior_image_left": CameraFrame(image, target, 1.0 + index / 15, index + 1),
                "wrist_image": CameraFrame(image, target, 1.0 + index / 15, index + 1),
            },
        )
    if raw is not None:
        writer.set_raw_force(raw)
    return writer.finish()


class ExternalForceRecordingTest(unittest.TestCase):
    def test_franka_s_force_estimate_is_recorded_when_reported(self) -> None:
        wrench = np.array([1.0, -2.0, 3.0, 0.1, 0.2, 0.3])
        raw = [
            RobotSample(
                qpos=np.zeros(7, dtype=np.float32),
                dq=np.zeros(7, dtype=np.float32),
                tau_J=np.zeros(7, dtype=np.float32),
                controller_timestamp=1.0 + i * 0.001,
                request_send_monotonic=float(i),
                response_receive_monotonic=float(i),
                ext_wrench=wrench.astype(np.float32),
                tau_ext=np.full(7, 0.5, dtype=np.float32),
            )
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_grid_episode(Path(temporary), {"ext_wrench": wrench, "tau_ext": np.full(7, 0.5)}, raw)
            with np.load(episode / "trajectory.npz") as trajectory:
                np.testing.assert_allclose(trajectory["external_wrench"], np.tile(wrench, (2, 1)), atol=1e-6)
                self.assertEqual(trajectory["external_joint_torque"].shape, (2, 7))
                self.assertEqual(trajectory["raw_force_external_wrench"].shape, (5, 6))
                self.assertEqual(trajectory["raw_force_external_joint_torque"].shape, (5, 7))
            force = json.loads((episode / "metadata.json").read_text())["force_signal"]
        self.assertIn("external_wrench", force["external"]["fields"])

    def test_an_older_controller_records_exactly_what_it_did(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_grid_episode(Path(temporary), {})
            with np.load(episode / "trajectory.npz") as trajectory:
                self.assertNotIn("external_wrench", trajectory.files)
                self.assertIn("joint_torque", trajectory.files)
            force = json.loads((episode / "metadata.json").read_text())["force_signal"]
        self.assertNotIn("external", force)

    def test_an_episode_can_carry_its_own_metadata_but_not_replace_the_recorder_s(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _write_grid_episode(
                Path(temporary),
                {},
                action_source=REPLAY_ACTION_SOURCE,
                extra_metadata={"replay_of": "episode_000003"},
            )
            metadata = json.loads((episode / "metadata.json").read_text())
            with self.assertRaisesRegex(ValueError, "fps"):
                _write_grid_episode(Path(temporary) / "other", {}, extra_metadata={"fps": 30.0})
        self.assertEqual(metadata["replay_of"], "episode_000003")
        self.assertEqual(metadata["action_source"], REPLAY_ACTION_SOURCE)

    def test_the_collector_samples_it_and_takes_per_episode_settings(self) -> None:
        class FakeCameras:
            def __init__(self) -> None:
                self.serials = {"exterior_image_left": "external", "wrist_image": "wrist"}
                self.camera_transforms = {"exterior_image_left": "none", "wrist_image": "none"}
                self.frame_number = 0

            def snapshot(self, _max_age=0.25):
                return {}

            def nearest(self, target, _max_delta, _after):
                self.frame_number += 1
                image = np.zeros((4, 6, 3), dtype=np.uint8)
                return {
                    key: CameraFrame(image, target + 0.003, target - 1.0, self.frame_number, "global_time", target)
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
                return {
                    "qpos": [0.0] * 7,
                    "dq": [0.0] * 7,
                    "tau_J": [2.0] * 7,
                    "time_sec": now,
                    "ext_wrench": [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    "tau_ext": [0.5] * 7,
                }

            def get_gripper_state(self):
                return {"success": True, "state": {"width": 0.0425}}

            def close(self):
                pass

        from fr3_demo.recording import MEASURED_ACTION_SOURCE, DemoCollector

        with tempfile.TemporaryDirectory() as temporary, patch("bamboo.BambooFrankaClient", FakeBamboo):
            collector = DemoCollector(
                FakeCameras(),
                Path(temporary),
                "127.0.0.1",
                5555,
                fps=10.0,
                alignment_delay_ms=20.0,
                robot_sample_hz=100.0,
                gripper_sample_hz=100.0,
                min_free_gb=0.0,
                action_source=MEASURED_ACTION_SOURCE,
            )
            collector.start()
            time.sleep(0.04)
            collector.start_episode(action_source=REPLAY_ACTION_SOURCE, extra_metadata={"replay_of": "episode_000000"})
            collector.set_action(np.full(7, 0.1))
            time.sleep(0.25)
            episode = collector.stop_episode()
            collector.close()
            metadata = json.loads((episode / "metadata.json").read_text())
            with np.load(episode / "trajectory.npz") as trajectory:
                np.testing.assert_allclose(trajectory["external_wrench"][:, 2], 4.0)
                np.testing.assert_allclose(trajectory["external_joint_torque"], 0.5)
                self.assertIn("raw_force_external_wrench", trajectory.files)
                # A replay keeps the velocities it commanded, not ones derived from motion.
                np.testing.assert_allclose(trajectory["action_joint_velocity"][-1], 0.1, atol=1e-6)
        self.assertEqual(metadata["action_source"], REPLAY_ACTION_SOURCE)
        self.assertEqual(metadata["replay_of"], "episode_000000")


def _snapshot(x: int = 0, b: int = 0, back: bool = False, connected: bool = True) -> JoystickSnapshot:
    mapping = Mapping()
    counts = [0] * 12
    counts[mapping.record_button], counts[mapping.open_button] = x, b
    buttons = [False] * 12
    buttons[mapping.quit_button] = back
    return JoystickSnapshot((0.0,) * 6, tuple(buttons), tuple(counts), connected, "pad", 0.0)


class ReplayControlsTest(unittest.TestCase):
    def test_only_presses_after_the_mark_count(self) -> None:
        joystick = MagicMock()
        joystick.snapshot.return_value = _snapshot(x=1)
        controls = ReplayControls(joystick, Mapping(), lambda: False)
        self.assertIsNone(controls.stop_reason())
        joystick.snapshot.return_value = _snapshot(x=1, b=1)
        self.assertEqual(controls.pressed(), (False, True))
        self.assertIsNotNone(controls.stop_reason())
        self.assertFalse(controls.quit)
        controls.mark()
        self.assertIsNone(controls.stop_reason())

    def test_back_a_lost_joystick_or_ctrl_c_end_the_session(self) -> None:
        for snapshot, ending in (
            (_snapshot(back=True), False),
            (_snapshot(connected=False), False),
            (_snapshot(), True),
        ):
            joystick = MagicMock()
            joystick.snapshot.return_value = _snapshot()
            controls = ReplayControls(joystick, Mapping(), lambda ending=ending: ending)
            joystick.snapshot.return_value = snapshot
            with self.subTest(snapshot=snapshot, ending=ending):
                self.assertIsNotNone(controls.stop_reason())
                self.assertTrue(controls.quit)


def _replay_args(**overrides) -> argparse.Namespace:
    values = {
        "contact_limits": {"max_force_n": 15.0, "contact_point": [0.0, 0.0, 0.2]},
        "no_contact_limits": False,
        "home_speed": 0.2,
        "max_joint_speed": 0.35,
        "max_joint_acceleration": 1.5,
        "watchdog_ms": 250,
        "home_timeout": 15.0,
        "stream_rate": 30.0,
        "guide_watchdog_ms": 250,
        "guide_max_joint_velocity": 0.35,
        "guide_idle_hold_s": None,
        "guide_max_start_force": None,
        "replay_depth_mm": 1.5,
        "replay_depth_direction": (0.0, 0.0, -1.0),
        "replay_max_path_error": 0.08,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ReplayFromGuideTest(unittest.TestCase):
    def _run(self, robot, args, *, outcome=None, homing=None, path_error=None, controls=None, wait=None):
        from fr3_demo import teleop

        path = MagicMock(duration=4.0, start=START)
        path.deepened.return_value = path
        collector = MagicMock()
        collector.start_episode.return_value = Path("/tmp/session/episode_000001.inprogress")
        collector.stop_episode.return_value = Path("/tmp/session/episode_000001")
        controls = controls or MagicMock(quit=False, **{"stop_reason.return_value": None})
        old, new = MagicMock(), MagicMock()
        with (
            patch.object(teleop.ReplayPath, "from_episode", side_effect=path_error, return_value=path),
            patch.object(teleop, "stream_home", side_effect=homing) as home,
            patch.object(teleop, "run_replay", return_value=outcome or ReplayOutcome(True, None, 4.0, 0.004)) as run,
            patch.object(teleop, "link_replay") as link,
            patch.object(teleop, "_wait_for_go", return_value=wait) as waited,
            patch.object(teleop.time, "sleep"),
        ):
            result = teleop.replay_from_guide(
                robot, args, Path("/tmp/session/episode_000000"), collector, old, lambda: new, controls, lambda: False
            )
        return result, {
            "old": old,
            "new": new,
            "home": home,
            "run": run,
            "link": link,
            "collector": collector,
            "wait": waited,
            "path": path,
        }

    def test_replays_records_homes_and_floats_again(self) -> None:
        robot = MagicMock()
        result, parts = self._run(robot, _replay_args())
        names = [call[0] for call in robot.method_calls]
        self.assertEqual(
            names,
            [
                "stop_guide",
                "apply_contact_limits",
                "start_stream",
                "stop_stream",
                "clear_contact_limits",
                "start_guide",
            ],
        )
        robot.apply_contact_limits.assert_called_once_with({"max_force_n": 15.0, "contact_point": [0.0, 0.0, 0.2]})
        parts["old"].stop.assert_called_once_with()
        # Home first, wait there for X, then the demonstration's start, and home again after the replay.
        targets = [call.args[1] for call in parts["home"].call_args_list]
        self.assertEqual(len(targets), 3)
        np.testing.assert_array_equal(targets[0], HOME_JOINTS)
        np.testing.assert_array_equal(targets[1], START)
        np.testing.assert_array_equal(targets[2], HOME_JOINTS)
        # It settles closely onto the demonstration's own start before recording.
        from fr3_demo.teleop import START_POSE_TOLERANCE_RAD

        tolerances = [call.kwargs["tolerance"] for call in parts["home"].call_args_list]
        self.assertEqual(tolerances, [0.01, START_POSE_TOLERANCE_RAD, 0.01])
        parts["wait"].assert_called_once()
        parts["collector"].start_episode.assert_called_once_with(
            action_source=REPLAY_ACTION_SOURCE, extra_metadata={"replay_of": "episode_000000"}
        )
        parts["collector"].stop_episode.assert_called_once_with()
        parts["link"].assert_called_once()
        settings = parts["link"].call_args.args[3]
        np.testing.assert_allclose(settings["depth_offset_m"], [0.0, 0.0, -0.0015])
        self.assertIs(result, parts["new"])

    def test_the_replay_follows_the_demonstration_deepened_at_the_contact_point(self) -> None:
        _, parts = self._run(MagicMock(), _replay_args())
        offset, point = parts["path"].deepened.call_args.args
        np.testing.assert_allclose(offset, [0.0, 0.0, -0.0015])
        self.assertEqual(point, [0.0, 0.0, 0.2])
        _, parts = self._run(MagicMock(), _replay_args(replay_depth_mm=0.0))
        parts["path"].deepened.assert_not_called()

    def test_a_stopped_replay_is_kept_and_guide_mode_resumes_where_the_arm_is(self) -> None:
        robot = MagicMock()
        stopped = ReplayOutcome(False, "the gel reached level 1.02", 2.0, 0.01)
        result, parts = self._run(robot, _replay_args(), outcome=stopped)
        self.assertEqual(parts["home"].call_count, 2)  # home and the start; no homing after a stopped replay
        parts["collector"].stop_episode.assert_called_once_with()
        parts["link"].assert_called_once()
        robot.clear_contact_limits.assert_called_once_with()
        self.assertIs(result, parts["new"])

    def test_a_demo_that_cannot_be_replayed_leaves_guide_mode_alone(self) -> None:
        robot = MagicMock()
        result, parts = self._run(robot, _replay_args(), path_error=ValueError("it moves the gripper"))
        self.assertEqual(robot.method_calls, [])
        parts["old"].stop.assert_not_called()
        self.assertIs(result, parts["old"])

    def test_stopping_the_arm_on_its_way_home_records_nothing(self) -> None:
        robot = MagicMock()
        result, parts = self._run(robot, _replay_args(), homing=InterruptedError("Homing interrupted"))
        parts["wait"].assert_not_called()
        parts["collector"].start_episode.assert_not_called()
        robot.stop_stream.assert_called_once_with()
        self.assertIs(result, parts["new"])

    def test_b_at_home_skips_the_replay(self) -> None:
        robot = MagicMock()
        result, parts = self._run(robot, _replay_args(), wait="you skipped it with B")
        parts["wait"].assert_called_once()
        self.assertEqual(parts["home"].call_count, 1)  # it stays home
        parts["collector"].start_episode.assert_not_called()
        self.assertIs(result, parts["new"])

    def test_back_during_a_replay_ends_the_session_without_guiding(self) -> None:
        robot = MagicMock()
        controls = MagicMock(quit=True, **{"stop_reason.return_value": "you pressed Back"})
        stopped = ReplayOutcome(False, "you pressed Back", 1.0, 0.002, by_operator=True)
        result, _ = self._run(robot, _replay_args(), outcome=stopped, controls=controls)
        self.assertIsNone(result)
        robot.start_guide.assert_not_called()
        robot.clear_contact_limits.assert_called_once_with()

    def test_a_failure_still_ends_the_stream_and_clears_the_limits(self) -> None:
        robot = MagicMock()
        robot.start_stream.side_effect = RuntimeError("Franka is not in Execution mode")
        with self.assertRaisesRegex(RuntimeError, "Execution mode"):
            self._run(robot, _replay_args())
        robot.stop_stream.assert_called_once_with()
        robot.clear_contact_limits.assert_called_once_with()
        robot.start_guide.assert_not_called()


class HomeBeforeRecordingTest(unittest.TestCase):
    def _robot(self, *poses):
        robot = MagicMock()
        robot.state.side_effect = [{"qpos": pose} for pose in poses]
        return robot

    def _home(self, robot, homed=None):
        from fr3_demo import teleop

        old = MagicMock()
        with patch.object(teleop, "home_from_guide", return_value=homed or MagicMock()) as home:
            result = teleop.home_before_recording(robot, _replay_args(), old, MagicMock, lambda: False)
        return result, home, old

    def test_an_arm_already_home_records_where_it_is(self) -> None:
        (heartbeat, ready), home, old = self._home(self._robot(HOME_JOINTS + 0.01))
        home.assert_not_called()
        self.assertIs(heartbeat, old)
        self.assertTrue(ready)

    def test_an_arm_elsewhere_is_homed_first(self) -> None:
        homed = MagicMock()
        (heartbeat, ready), home, _ = self._home(self._robot(HOME_JOINTS + 0.3, HOME_JOINTS), homed)
        home.assert_called_once()
        self.assertIs(heartbeat, homed)
        self.assertTrue(ready)

    def test_homing_that_stops_short_does_not_record(self) -> None:
        (_, ready), _, _ = self._home(self._robot(HOME_JOINTS + 0.3, HOME_JOINTS + 0.2))
        self.assertFalse(ready)
        from fr3_demo import teleop

        robot = self._robot(HOME_JOINTS + 0.3)
        with patch.object(teleop, "home_from_guide", return_value=None):  # the session is ending
            self.assertEqual(
                teleop.home_before_recording(robot, _replay_args(), MagicMock(), MagicMock, lambda: True), (None, False)
            )


class WaitForGoTest(unittest.TestCase):
    def _wait(self, presses, gel_reason=None):
        from fr3_demo import teleop

        controls = MagicMock()
        controls.ending.return_value = None
        controls.pressed.side_effect = presses
        guard = None if gel_reason is None else MagicMock(**{"check.return_value": gel_reason})
        robot = MagicMock()
        with patch.object(teleop.time, "sleep"):
            reason = teleop._wait_for_go(robot, _replay_args(), controls, guard)
        return reason, robot, controls

    def test_x_replays_and_b_skips(self) -> None:
        reason, robot, controls = self._wait([(False, False), (False, False), (True, False)])
        self.assertIsNone(reason)
        self.assertEqual(robot.stream_velocity.call_count, 2)  # held still until X
        self.assertEqual(controls.mark.call_count, 2)  # the X that starts it does not also stop it
        reason, _, _ = self._wait([(False, True)])
        self.assertIn("skipped", reason)

    def test_a_gel_loaded_while_waiting_makes_the_arm_go_soft(self) -> None:
        reason, robot, _ = self._wait([(False, False)], gel_reason="the gel reached level 1.10")
        robot.yield_stream.assert_called_once_with("the gel reached level 1.10")
        self.assertIn("keep clear of the tool", reason)


class DeepenedPathTest(unittest.TestCase):
    def test_the_contact_point_moves_by_the_offset_without_turning_the_tool(self) -> None:
        from fr3_demo.kinematics import forward_kinematics

        times = np.linspace(0.0, 6.0, 1201)
        positions = START + np.outer(np.sin(times / 2), [0.2, 0.1, -0.1, 0.15, 0.1, -0.1, 0.2])
        path = ReplayPath.from_samples(Path("demo"), times, positions)
        deep = path.deepened(np.array([0.0, 0.0, -0.0015]), np.array([0.0, 0.0, 0.2]))

        def contact(q):
            pose = forward_kinematics(q)
            return pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, 0.2]), pose[:3, :3]

        for elapsed, share in ((0.0, 0.0), (0.5, 0.5), (3.0, 1.0), (6.0, 0.0)):
            (a, turn_a), (b, turn_b) = contact(path.at(elapsed)[0]), contact(deep.at(elapsed)[0])
            np.testing.assert_allclose(b - a, [0.0, 0.0, -0.0015 * share], atol=2e-5)
            np.testing.assert_allclose(turn_a.T @ turn_b, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(deep.start, path.start)
        self.assertIs(path.deepened(np.zeros(3), np.array([0.0, 0.0, 0.2])), path)


class EpisodeSelectionTest(unittest.TestCase):
    def test_demos_replays_or_both_and_never_a_replay_that_stopped(self) -> None:
        from fr3_demo.convert_lerobot import main, select_episodes

        with tempfile.TemporaryDirectory() as temporary:
            folders = []
            for index, extra in enumerate(
                (
                    {},
                    {"replay_of": "episode_000000", "replay": {"completed": True}},
                    {"replay_of": "episode_000000", "replay": {"completed": False}},
                )
            ):
                folder = Path(temporary) / f"episode_{index:06d}"
                folder.mkdir()
                (folder / "metadata.json").write_text(json.dumps({"complete": True, **extra}))
                folders.append(folder)
            demo, replay, stopped = folders
            self.assertEqual(select_episodes(folders), ([demo, replay], [stopped]))
            self.assertEqual(select_episodes(folders, "demos"), ([demo], [stopped]))
            self.assertEqual(select_episodes(folders, "replays"), ([replay], [stopped]))
            with self.assertRaises(ValueError):
                select_episodes(folders, "some")
            with self.assertRaises(SystemExit):
                main(["--data-dir", temporary, "--repo-id", "x/y", "--episodes", "some"])


class ReplayFlagTest(unittest.TestCase):
    def _args(self, *extra: str, contact=True, gel=None, tactile=False):
        parser = _create_parser()
        args = parser.parse_args(
            ["--collect", "--external-camera-serial", "a", "--wrist-camera-serial", "b", *extra]
            + (["--record-tactile"] if tactile else [])
        )
        args.contact_limits = ({"max_force_n": 15.0} if contact is True else contact) or None
        args.gel = gel
        return parser, args

    def test_replay_needs_hand_guiding_contact_limits_and_the_gel_guard(self) -> None:
        for extra, options in (
            (["--replay"], {}),  # no --hand-guide
            (["--hand-guide", "--replay"], {"contact": False}),
            (["--hand-guide", "--replay"], {"contact": {"max_force_n": 15.0, "shadow": True}}),
            (["--hand-guide", "--replay"], {"contact": {"guard_force_n": 20.0}}),  # no push cap
            (["--hand-guide", "--replay", "--no-contact-limits"], {}),
            (["--hand-guide", "--replay"], {"tactile": True}),  # GelSight in use, guard off
            (["--hand-guide", "--replay", "--no-gel-guard"], {"gel": {"enabled": True}}),
            (["--hand-guide", "--replay-depth-mm", "1"], {}),  # tuning without --replay
            (["--hand-guide", "--replay", "--replay-depth-mm", "4"], {}),
            (["--hand-guide", "--replay", "--replay-depth-direction", "0", "0", "0"], {}),
            (["--hand-guide", "--replay", "--replay-max-path-error", "0.5"], {}),
        ):
            parser, args = self._args(*extra, **options)
            with self.subTest(extra=extra, options=options), self.assertRaises(SystemExit):
                _validate_args(args, parser)

    def test_a_guarded_replay_session_is_accepted(self) -> None:
        parser, args = self._args(
            "--hand-guide", "--replay", "--replay-depth-mm", "2", gel={"enabled": True}, tactile=True
        )
        _validate_args(args, parser)
        parser, args = self._args("--hand-guide", "--replay")  # no GelSight at all
        _validate_args(args, parser)


if __name__ == "__main__":
    unittest.main()
