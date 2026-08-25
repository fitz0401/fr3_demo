"""Publish both RealSense views to ROS 2 and open RViz."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

from fr3_demo.cameras import RealSensePair, discover_realsense
from fr3_demo.settings import camera_defaults, default_config_path, load_config


def _rviz_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("SNAP", "GTK_", "GIO_")) or key == "LD_PRELOAD":
            environment.pop(key, None)
    return environment


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


def _image_message(image: object, frame_id: str, node: object) -> object:
    from sensor_msgs.msg import Image

    message = Image()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = frame_id
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = message.width * 3
    message.data = image.tobytes()
    return message


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
        "--wrist-rotate-180",
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

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError("ROS 2 is not sourced; run: source /opt/ros/humble/setup.bash") from error

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
        wrist_rotate_180=args.wrist_rotate_180,
    ).start()
    rclpy.init()
    node = Node("fr3_demo_camera_preview")
    exterior_publisher = node.create_publisher(Image, "/fr3_demo/exterior_image_left", qos_profile_sensor_data)
    exterior2_publisher = node.create_publisher(
        Image, "/fr3_demo/exterior_image_2_left", qos_profile_sensor_data
    )
    wrist_publisher = node.create_publisher(Image, "/fr3_demo/wrist_image", qos_profile_sensor_data)

    def publish() -> None:
        frames = cameras.snapshot()
        exterior_publisher.publish(_image_message(frames["exterior_image_left"].image, "exterior_camera", node))
        if "exterior_image_2_left" in frames:
            exterior2_publisher.publish(
                _image_message(frames["exterior_image_2_left"].image, "exterior_camera_2", node)
            )
        wrist_publisher.publish(_image_message(frames["wrist_image"].image, "wrist_camera", node))

    node.create_timer(1.0 / 15.0, publish)
    rviz_process: subprocess.Popen | None = None
    if not args.no_rviz:
        config = resources.files("fr3_demo").joinpath("config/cameras.rviz")
        with resources.as_file(config) as config_path:
            rviz_process = subprocess.Popen(
                ["rviz2", "-d", str(config_path)],
                env=_rviz_environment(),
            )

    print(
        f"Publishing camera views {cameras.modes}; wrist_rotate_180={args.wrist_rotate_180}. "
        "Press Ctrl+C to stop."
    )
    if cameras.optional_camera_error:
        print(f"Optional exterior camera unavailable; continuing without it: {cameras.optional_camera_error}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rviz_process is not None:
            rviz_process.terminate()
            try:
                rviz_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                rviz_process.kill()
        node.destroy_node()
        rclpy.shutdown()
        cameras.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(rviz_main())
