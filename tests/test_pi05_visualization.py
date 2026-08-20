import os
import unittest
from unittest.mock import patch

from fr3_pi05.visualization import _rviz_environment


class Pi05VisualizationTest(unittest.TestCase):
    def test_rviz_environment_removes_snap_library_settings(self) -> None:
        source = {
            "PATH": "/opt/ros/humble/bin:/usr/bin",
            "DISPLAY": ":1",
            "SNAP": "/snap/code/current",
            "SNAP_LIBRARY_PATH": "/snap/core20/lib",
            "GTK_PATH": "/snap/code/gtk",
            "GIO_MODULE_DIR": "/snap/code/gio",
            "LD_PRELOAD": "/snap/core20/libpthread.so.0",
        }
        with patch.dict(os.environ, source, clear=True):
            environment = _rviz_environment()

        self.assertEqual(environment, {"PATH": source["PATH"], "DISPLAY": ":1"})


if __name__ == "__main__":
    unittest.main()
