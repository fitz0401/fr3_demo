"""Publish both RealSense views to ROS 2 and open RViz."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from fr3_demo.cameras import RealSensePair, discover_realsense
from fr3_demo.preview import CameraPreview
from fr3_demo.settings import camera_defaults, default_config_path, load_config


def list_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List connected Intel RealSense cameras.")
    parser.parse_args(argv)
    devices = discover_realsense()
    if not devices:
        print("No RealSense cameras detected.")
        return 1
    for device in devices:
        print(f"{device['serial']}\t{device['name']}")
    return 0


def rviz_main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Preview exterior and wrist RealSense color images in RViz 2.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="shared TOML config (default: %(default)s; override with FR3_DEMO_CONFIG)",
    )
    parser.add_argument("--external-camera-serial")
    parser.add_argument("--wrist-camera-serial")
    parser.add_argument("--external2-camera-serial", help="optional second exterior RealSense")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--external2-camera-fps", type=int, default=30)
    parser.add_argument("--external2-camera-width", type=int, default=960)
    parser.add_argument("--external2-camera-height", type=int, default=540)
    parser.add_argument(
        "--external-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wrist-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--external2-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no-rviz", action="store_true", help="publish topics without launching RViz")
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=default_config_path())
    bootstrap_args, _ = bootstrap.parse_known_args(raw_argv)
    try:
        defaults = camera_defaults(load_config(bootstrap_args.config))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    environment_defaults = {
        "external_camera_serial": os.environ.get("FR3_EXTERNAL_CAMERA_SERIAL"),
        "wrist_camera_serial": os.environ.get("FR3_WRIST_CAMERA_SERIAL"),
        "external2_camera_serial": os.environ.get("FR3_EXTERNAL2_CAMERA_SERIAL"),
    }
    defaults.update({key: value for key, value in environment_defaults.items() if value})
    parser.set_defaults(config=bootstrap_args.config, **defaults)
    args = parser.parse_args(raw_argv)
    if not args.external_camera_serial or not args.wrist_camera_serial:
        parser.error("camera serials are missing from the config file and command line")

    cameras = RealSensePair(
        args.external_camera_serial,
        args.wrist_camera_serial,
        args.external2_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        exterior2_width=args.external2_camera_width,
        exterior2_height=args.external2_camera_height,
        exterior2_fps=args.external2_camera_fps,
        external_rotate_180=args.external_rotate_180,
        wrist_rotate_180=args.wrist_rotate_180,
        external2_rotate_180=args.external2_rotate_180,
    ).start()
    preview = CameraPreview(cameras, rate_hz=15.0, launch_viewer=not args.no_rviz)
    try:
        preview.start()
        print(
            f"Publishing camera views {cameras.modes}; transforms={cameras.camera_transforms}. "
            "Press Ctrl+C to stop."
        )
        if cameras.optional_camera_error:
            print(f"Optional exterior camera unavailable; continuing without it: {cameras.optional_camera_error}")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        preview.close()
        cameras.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(rviz_main())
