import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fr3_demo.guide import GuideHeartbeat, choose_guide_mode, guide_failure_advice, hold_message
from fr3_demo.settings import load_config, teleop_defaults
from fr3_demo.teleop import BambooRobot, _create_parser, _validate_args

USER_STOPPED = (
    "Guide mode stopped: Franka guide control error: libfranka: Move command rejected: command not possible "
    'in the current mode ("User stopped")!'
)


class FakeGuideClient:
    def __init__(self, fail_after: int | None = None, error: str = "Guide mode is not running") -> None:
        self.updates = 0
        self.closed = False
        self.fail_after = fail_after
        self.error = error

    def update_guiding(self, level: float = 0.0, valid: bool = True) -> dict:
        self.updates += 1
        if self.fail_after is not None and self.updates > self.fail_after:
            return {"success": False, "error": self.error}
        return {"success": True, "phase": 0, "reason": 0, "level": level, "stale": False}

    def close(self) -> None:
        self.closed = True


class UnreachableGuideClient(FakeGuideClient):
    def update_guiding(self, level: float = 0.0, valid: bool = True) -> dict:
        raise TimeoutError("Bamboo did not answer")


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class GuideModeChoiceTest(unittest.TestCase):
    def test_uses_bamboo_guide_mode_only_when_offered(self) -> None:
        robot = MagicMock()
        robot.supports.return_value = True
        self.assertEqual(choose_guide_mode(robot, force_pilot=False), "impedance")
        self.assertEqual(choose_guide_mode(robot, force_pilot=True), "pilot")
        robot.supports.return_value = False
        self.assertEqual(choose_guide_mode(robot, force_pilot=False), "pilot")

    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_older_bamboo_client_means_pilot_buttons(self, client_class: MagicMock) -> None:
        client = MagicMock(spec=["get_joint_states", "close"])  # no supports(): an older client
        client.get_joint_states.return_value = {"qpos": [0.0] * 7}
        client_class.return_value = client
        robot = BambooRobot("127.0.0.1", 5555, 5559, "robotiq", False)
        self.assertFalse(robot.supports("guide_mode"))
        self.assertEqual(choose_guide_mode(robot, force_pilot=False), "pilot")
        with self.assertRaises(RuntimeError):
            robot.set_load(0.9, [0.0, 0.0, 0.06])


class PayloadTest(unittest.TestCase):
    @patch("fr3_demo.teleop.time.sleep")
    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_set_load_waits_for_franka_to_finish_blending(self, client_class: MagicMock, _sleep: MagicMock) -> None:
        client = MagicMock()
        client.get_joint_states.return_value = {"qpos": [0.0] * 7}
        client.supports.return_value = True
        client.set_load.return_value = {"success": True}
        ramp = [0.2, 0.6, 1.1]
        client.get_capabilities.side_effect = [{"payload": {"m_load": value}} for value in ramp]
        client_class.return_value = client
        robot = BambooRobot("127.0.0.1", 5555, 5559, "robotiq", False)
        robot.set_load(1.1, [0.0, 0.0, 0.07])
        self.assertEqual(client.get_capabilities.call_count, len(ramp))
        client.set_load.assert_called_once_with(1.1, [0.0, 0.0, 0.07])


class GuideHeartbeatTest(unittest.TestCase):
    def test_sends_updates_until_stopped_then_closes_its_connection(self) -> None:
        client = FakeGuideClient()
        heartbeat = GuideHeartbeat(lambda: client, rate_hz=200.0)
        heartbeat.start()
        self.assertTrue(_wait_for(lambda: client.updates >= 5))
        heartbeat.check_health()
        heartbeat.stop()
        self.assertTrue(client.closed)
        sent = client.updates
        time.sleep(0.05)
        self.assertEqual(client.updates, sent)

    def test_an_ended_guide_loop_is_reported_as_ended(self) -> None:
        client = FakeGuideClient(fail_after=2)
        heartbeat = GuideHeartbeat(lambda: client, rate_hz=200.0)
        heartbeat.start()
        self.assertTrue(_wait_for(lambda: client.closed))
        self.assertTrue(heartbeat.ended_on_robot)
        with self.assertRaisesRegex(RuntimeError, "Guide mode ended"):
            heartbeat.check_health()
        heartbeat.stop()

    def test_a_user_stop_comes_with_what_to_do(self) -> None:
        client = FakeGuideClient(fail_after=0, error=USER_STOPPED)
        heartbeat = GuideHeartbeat(lambda: client, rate_hz=200.0)
        heartbeat.start()
        self.assertTrue(_wait_for(lambda: client.closed))
        with self.assertRaisesRegex(RuntimeError, "(?s)User stopped.*Switch Desk to Execution"):
            heartbeat.check_health()
        heartbeat.stop()

    def test_losing_bamboo_means_the_watchdog_holds(self) -> None:
        client = UnreachableGuideClient()
        heartbeat = GuideHeartbeat(lambda: client, rate_hz=200.0)
        heartbeat.start()
        self.assertTrue(_wait_for(lambda: client.closed))
        self.assertFalse(heartbeat.ended_on_robot)
        with self.assertRaisesRegex(RuntimeError, "arm is holding"):
            heartbeat.check_health()
        heartbeat.stop()


class GuideFailureAdviceTest(unittest.TestCase):
    def test_raw_franka_errors_get_advice(self) -> None:
        self.assertIn("Switch Desk to Execution", guide_failure_advice(USER_STOPPED))
        pressed = "Franka guide control error: libfranka: Move command aborted: User Stop pressed!"
        self.assertIn("was pressed", guide_failure_advice(pressed))
        reflex = 'libfranka: Move command aborted: motion aborted by reflex! ["joint_velocity_violation"]'
        self.assertIn("reflex", guide_failure_advice(reflex))

    def test_bamboo_messages_that_already_say_what_to_do_get_none(self) -> None:
        refusal = 'Franka is not in Execution mode (robot mode "User stopped"): white lights mean Programming mode'
        self.assertIsNone(guide_failure_advice(refusal))
        self.assertIsNone(guide_failure_advice("libfranka: something new"))

    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_a_refused_start_says_guide_mode_did_not_start(self, client_class: MagicMock) -> None:
        client = MagicMock()
        client.get_joint_states.return_value = {"qpos": [0.0] * 7}
        client.start_guiding.return_value = {"success": False, "error": USER_STOPPED.split(": ", 1)[1]}
        client_class.return_value = client
        robot = BambooRobot("127.0.0.1", 5555, 5559, "robotiq", False)
        with self.assertRaisesRegex(RuntimeError, "(?s)Could not start guide mode.*Switch Desk to Execution"):
            robot.start_guide(250, 0.35)


class HoldMessageTest(unittest.TestCase):
    def test_only_holds_needing_the_operator_are_announced(self) -> None:
        self.assertIsNone(hold_message({"guide_phase": "free", "guide_hold_reason": "none"}))
        self.assertIsNone(hold_message({"guide_phase": "hold", "guide_hold_reason": "idle"}))
        self.assertIsNone(hold_message({}))  # an older server reports no guide fields
        message = hold_message({"guide_phase": "hold", "guide_hold_reason": "watchdog"})
        self.assertIn("RB", message)
        gel = hold_message({"guide_phase": "hold", "guide_hold_reason": "gel_invalid"}, "lost track")
        self.assertIn("lost track", gel)


class GuideConfigTest(unittest.TestCase):
    def test_guide_and_payload_sections_map_to_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                """
version = 1
[guide]
pilot_buttons = false
watchdog_ms = 200
max_joint_velocity = 0.3
idle_hold_s = 1.5
max_start_force = 6.0
heartbeat_hz = 40.0
[payload]
mass_kg = 0.95
center_of_mass_m = [0.0, 0.0, 0.06]
"""
            )
            defaults = teleop_defaults(load_config(path))
        self.assertEqual(defaults["guide_watchdog_ms"], 200)
        self.assertEqual(defaults["guide_max_joint_velocity"], 0.3)
        self.assertEqual(defaults["guide_idle_hold_s"], 1.5)
        self.assertEqual(defaults["guide_max_start_force"], 6.0)
        self.assertEqual(defaults["guide_heartbeat_hz"], 40.0)
        self.assertFalse(defaults["pilot_guide"])
        self.assertEqual(defaults["payload_mass_kg"], 0.95)
        self.assertEqual(defaults["payload_com_m"], [0.0, 0.0, 0.06])

    def test_without_the_sections_nothing_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text("version = 1\n")
            defaults = teleop_defaults(load_config(path))
        self.assertFalse(any(key.startswith(("guide_", "payload_", "pilot_")) for key in defaults))

    def test_validation(self) -> None:
        parser = _create_parser()
        for argv in (
            ["--payload-mass-kg", "0.9"],  # mass without a center of mass
            ["--payload-mass-kg", "5", "--payload-com-m", "0", "0", "0.06"],
            ["--guide-heartbeat-hz", "5"],  # fewer than three updates per watchdog period
            ["--guide-watchdog-ms", "20"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                _validate_args(parser.parse_args(argv), parser)
        _validate_args(parser.parse_args(["--payload-mass-kg", "0.9", "--payload-com-m", "0", "0", "0.06"]), parser)


if __name__ == "__main__":
    unittest.main()


class HomeFromGuideTest(unittest.TestCase):
    def _args(self, contact_limits=None):
        import argparse

        return argparse.Namespace(
            no_contact_limits=False,
            contact_limits=contact_limits,
            home_speed=0.2,
            max_joint_speed=0.35,
            max_joint_acceleration=1.5,
            watchdog_ms=250,
            home_timeout=15.0,
            stream_rate=30.0,
            guide_watchdog_ms=250,
            guide_max_joint_velocity=0.35,
            guide_idle_hold_s=None,
            guide_max_start_force=None,
        )

    def _run(self, robot, args, stop=lambda: False, gel_guard=None, homing=None):
        from fr3_demo import teleop

        old, new = MagicMock(), MagicMock()
        with (
            patch.object(teleop, "stream_home", side_effect=homing) as home,
            patch.object(teleop.time, "sleep"),
        ):
            result = teleop.home_from_guide(robot, args, old, lambda: new, stop, gel_guard)
        return result, old, new, home

    def test_leaves_guide_mode_homes_and_floats_again(self) -> None:
        robot = MagicMock()
        result, old, new, home = self._run(robot, self._args())
        old.stop.assert_called_once_with()
        names = [call[0] for call in robot.method_calls]
        self.assertEqual(names, ["stop_guide", "start_stream", "stop_stream", "start_guide"])
        home.assert_called_once()
        new.start.assert_called_once_with()
        self.assertIs(result, new)

    def test_the_homing_stream_gets_the_contact_limits(self) -> None:
        robot = MagicMock()
        limits = {"max_force_n": 15.0}
        self._run(robot, self._args(contact_limits=limits))
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
        robot.apply_contact_limits.assert_called_once_with(limits)

    def test_waits_for_a_hand_to_let_go_before_guiding_again(self) -> None:
        robot = MagicMock()
        refusal = RuntimeError("Could not start guide mode: External force estimate is 7.1 N before guiding starts")
        robot.start_guide.side_effect = [refusal, refusal, None]
        result, _, new, _ = self._run(robot, self._args())
        self.assertEqual(robot.start_guide.call_count, 3)
        self.assertIs(result, new)
        robot.start_guide.side_effect = RuntimeError("Franka is not in Execution mode")
        with self.assertRaisesRegex(RuntimeError, "Execution mode"):
            self._run(robot, self._args())

    def test_a_loaded_gel_stops_homing_but_guiding_resumes(self) -> None:
        robot = MagicMock()
        guard = MagicMock()
        guard.check.return_value = "the gel load reached level 1.02 (10.2 N estimated)"
        result, _, new, _ = self._run(
            robot, self._args(), gel_guard=guard, homing=InterruptedError("Homing interrupted")
        )
        robot.stop_stream.assert_called_once_with()
        robot.start_guide.assert_called_once()
        self.assertIs(result, new)

    def test_a_session_ending_during_homing_leaves_guide_mode_off(self) -> None:
        robot = MagicMock()
        result, _, new, _ = self._run(
            robot, self._args(), stop=lambda: True, homing=InterruptedError("Homing interrupted")
        )
        self.assertIsNone(result)
        robot.start_guide.assert_not_called()
        robot.stop_stream.assert_called_once_with()
        new.start.assert_not_called()
