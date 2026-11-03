import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fr3_demo.guide import GuideHeartbeat, choose_guide_mode, hold_message
from fr3_demo.settings import load_config, teleop_defaults
from fr3_demo.teleop import BambooRobot, _create_parser, _validate_args


class FakeGuideClient:
    def __init__(self, fail_after: int | None = None) -> None:
        self.updates = 0
        self.closed = False
        self.fail_after = fail_after

    def update_guiding(self, level: float = 0.0, valid: bool = True) -> dict:
        self.updates += 1
        if self.fail_after is not None and self.updates > self.fail_after:
            return {"success": False, "error": "Guide mode is not running"}
        return {"success": True, "phase": 0, "reason": 0, "level": level, "stale": False}

    def close(self) -> None:
        self.closed = True


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

    def test_a_rejected_update_is_reported(self) -> None:
        client = FakeGuideClient(fail_after=2)
        heartbeat = GuideHeartbeat(lambda: client, rate_hz=200.0)
        heartbeat.start()
        self.assertTrue(_wait_for(lambda: client.closed))
        with self.assertRaisesRegex(RuntimeError, "arm is holding"):
            heartbeat.check_health()
        heartbeat.stop()


class HoldMessageTest(unittest.TestCase):
    def test_only_holds_needing_the_operator_are_announced(self) -> None:
        self.assertIsNone(hold_message({"guide_phase": "free", "guide_hold_reason": "none"}))
        self.assertIsNone(hold_message({"guide_phase": "hold", "guide_hold_reason": "idle"}))
        self.assertIsNone(hold_message({}))  # an older server reports no guide fields
        message = hold_message({"guide_phase": "hold", "guide_hold_reason": "watchdog"})
        self.assertIn("RB", message)


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
