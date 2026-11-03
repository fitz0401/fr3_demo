import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fr3_demo.contact import describe_contact_limits
from fr3_demo.settings import contact_limits, load_config, pi05_defaults, teleop_defaults
from fr3_demo.teleop import BambooRobot


def _config(text: str) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.toml"
        path.write_text("version = 1\n" + text)
        return load_config(path)


class ContactConfigTest(unittest.TestCase):
    def test_off_unless_enabled(self) -> None:
        self.assertIsNone(contact_limits(_config("")))
        self.assertIsNone(contact_limits(_config("[contact]\nenabled = false\nmax_force_n = 10.0\n")))
        self.assertNotIn("contact_limits", teleop_defaults(_config("")))
        self.assertNotIn("contact_limits", pi05_defaults(_config("")))

    def test_enabled_section_maps_to_bamboo_arguments(self) -> None:
        config = _config(
            """
[contact]
enabled = true
shadow = true
max_force_n = 15.0
contact_point_m = [0.0, 0.0, 0.2]
guard_force_n = 20.0
guard_hard_force_n = 30.0
collision_force_n = 35.0
[payload]
mass_kg = 1.1
center_of_mass_m = [0.0, 0.0, 0.07]
"""
        )
        limits = contact_limits(config)
        self.assertEqual(limits["max_force_n"], 15.0)
        self.assertEqual(limits["contact_point"], [0.0, 0.0, 0.2])
        self.assertTrue(limits["shadow"])
        self.assertEqual(limits["guard_impulse_ns"], 0.05)
        self.assertEqual(teleop_defaults(config)["contact_limits"], limits)
        runner = pi05_defaults(config)
        self.assertEqual(runner["contact_limits"], limits)
        self.assertEqual(runner["payload_mass_kg"], 1.1)

    def test_enabled_without_a_limit_or_with_a_bad_point_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            contact_limits(_config("[contact]\nenabled = true\n"))
        with self.assertRaises(ValueError):
            contact_limits(_config("[contact]\nenabled = true\nmax_force_n = 9.0\ncontact_point_m = [0.0, 0.1]\n"))

    def test_description_names_every_layer(self) -> None:
        text = describe_contact_limits(
            {
                "max_force_n": 15.0,
                "shadow": True,
                "contact_point": [0.0, 0.0, 0.2],
                "guard_force_n": 20.0,
                "guard_hard_force_n": 30.0,
                "max_point_speed": 0.0,
                "collision_force_n": 35.0,
            }
        )
        for words in ("push cap 15 N", "shadow", "force guard 20 N, hard 30 N", "tool speed off", "reflex 35 N"):
            self.assertIn(words, text)


class ApplyContactLimitsTest(unittest.TestCase):
    def _robot(self, client_class: MagicMock, supports: bool) -> tuple[BambooRobot, MagicMock]:
        client = MagicMock()
        client.get_joint_states.return_value = {"qpos": [0.0] * 7}
        client.supports.return_value = supports
        client.set_contact_limits.return_value = {"success": True, "limits": {"max_force_n": 15.0}}
        client.clear_contact_limits.return_value = {"success": True}
        client_class.return_value = client
        return BambooRobot("127.0.0.1", 5555, 5559, "robotiq", False), client

    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_without_limits_leftovers_are_cleared(self, client_class: MagicMock) -> None:
        robot, client = self._robot(client_class, supports=True)
        self.assertIsNone(robot.apply_contact_limits(None))
        client.clear_contact_limits.assert_called_once_with()
        client.set_contact_limits.assert_not_called()

    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_an_older_server_is_left_alone_without_limits_and_refused_with_them(self, client_class: MagicMock) -> None:
        robot, client = self._robot(client_class, supports=False)
        self.assertIsNone(robot.apply_contact_limits(None))
        client.clear_contact_limits.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "cannot limit contact forces"):
            robot.apply_contact_limits({"max_force_n": 15.0})

    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_limits_are_sent_and_the_echo_returned(self, client_class: MagicMock) -> None:
        robot, client = self._robot(client_class, supports=True)
        limits = {"max_force_n": 15.0, "contact_point": [0.0, 0.0, 0.2]}
        self.assertEqual(robot.apply_contact_limits(limits), {"max_force_n": 15.0})
        client.set_contact_limits.assert_called_once_with(**limits)


if __name__ == "__main__":
    unittest.main()


class ContactStatusTest(unittest.TestCase):
    def test_status_line_tracks_peaks_and_tolerates_older_servers(self) -> None:
        from fr3_demo.contact import contact_status

        peaks: dict[str, float] = {}
        state = {
            "ext_wrench": [3.0, 4.0, 0.0, 0, 0, 0],
            "contact_push": 9.0,
            "contact_scale": 0.8,
            "guard_trip": "none",
        }
        line = contact_status({**state, "loop_mode": "stream"}, peaks)
        self.assertIn("tool force   5.0 N", line)
        self.assertIn("cap 0.80", line)
        contact_status({**state, "ext_wrench": [0.0, 0.0, 1.0, 0, 0, 0], "contact_scale": 1.0}, peaks)
        self.assertEqual(peaks["force"], 5.0)
        self.assertEqual(peaks["scale"], 0.8)
        older = contact_status({"qpos": [0.0] * 7}, {})
        self.assertIn("push n/a", older)
        guiding = contact_status({"loop_mode": "guide", "guide_lead": 0.01, "guide_ratchet": True}, {})
        self.assertIn("leash 0.010 rad", guiding)
        self.assertIn("ratchet on", guiding)


class ClearAfterYieldTest(unittest.TestCase):
    @patch("fr3_demo.teleop.time.sleep")
    @patch("fr3_demo.teleop.BambooFrankaClient")
    def test_clearing_waits_for_a_yielding_stream_to_end(self, client_class: MagicMock, _sleep: MagicMock) -> None:
        client = MagicMock()
        client.get_joint_states.return_value = {"qpos": [0.0] * 7}
        busy = {"success": False, "error": "Stop the active control loop before clearing contact limits"}
        client.clear_contact_limits.side_effect = [busy, busy, {"success": True}]
        client_class.return_value = client
        BambooRobot("127.0.0.1", 5555, 5559, "robotiq", False).clear_contact_limits()
        self.assertEqual(client.clear_contact_limits.call_count, 3)
