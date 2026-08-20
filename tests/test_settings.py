import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fr3_demo.settings import camera_defaults, load_config, pi05_defaults, teleop_defaults
from fr3_demo.teleop import main


class SettingsTest(unittest.TestCase):
    def test_structured_config_maps_to_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                """
version = 1
[robot]
server_ip = "10.0.0.20"
control_port = 6000
[gripper]
enabled = false
[cameras]
external_serial = "external"
wrist_serial = "wrist"
fps = 15
[workspace]
min = [0.1, -0.2, 0.3]
max = [0.7, 0.2, 0.9]
[pi05]
transport = "zmq"
checkpoint = "custom_droid"
server_host = "gpu"
server_port = 8001
server_ports = { pi05_droid = 8000, custom_droid = 8001 }
open_loop_horizon = 8
""",
                encoding="utf-8",
            )
            config = load_config(path)

            teleop = teleop_defaults(config)
            cameras = camera_defaults(config)
            pi05 = pi05_defaults(config)
            self.assertEqual(teleop["server_ip"], "10.0.0.20")
            self.assertEqual(teleop["control_port"], 6000)
            self.assertTrue(teleop["no_gripper"])
            self.assertEqual(teleop["workspace_min"], [0.1, -0.2, 0.3])
            self.assertEqual(cameras["external_camera_serial"], "external")
            self.assertEqual(cameras["wrist_camera_serial"], "wrist")
            self.assertEqual(cameras["camera_fps"], 15)
            self.assertEqual(pi05["policy_host"], "gpu")
            self.assertEqual(pi05["transport"], "zmq")
            self.assertEqual(pi05["checkpoint"], "custom_droid")
            self.assertEqual(pi05["policy_port"], 8001)
            self.assertEqual(pi05["open_loop_horizon"], 8)

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text("version = 1\n[cameras]\nserial_typo = '123'\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cameras.serial_typo"):
                load_config(path)

    def test_teleop_cli_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                "version = 1\n[robot]\nserver_ip = '10.0.0.20'\n[teleop]\nlinear_speed = 0.05\n",
                encoding="utf-8",
            )
            with patch("fr3_demo.teleop.run", return_value=0) as run:
                result = main(
                    [
                        "--config",
                        str(path),
                        "--server-ip",
                        "10.0.0.30",
                        "--check",
                        "--offline",
                    ]
                )

            self.assertEqual(result, 0)
            args = run.call_args.args[0]
            self.assertEqual(args.server_ip, "10.0.0.30")
            self.assertEqual(args.linear_speed, 0.05)


if __name__ == "__main__":
    unittest.main()
