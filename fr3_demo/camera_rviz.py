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
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--wrist-vertical-flip",
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
        fps=args.camera_fps,
        wrist_vertical_flip=args.wrist_vertical_flip,
    ).start()
    rclpy.init()
    node = Node("fr3_demo_camera_preview")
    exterior_publisher = node.create_publisher(Image, "/fr3_demo/exterior_image_left", qos_profile_sensor_data)
    wrist_publisher = node.create_publisher(Image, "/fr3_demo/wrist_image", qos_profile_sensor_data)

    def publish() -> None:
        frames = cameras.snapshot()
        exterior_publisher.publish(_image_message(frames["exterior_image_left"].image, "exterior_camera", node))
        wrist_publisher.publish(_image_message(frames["wrist_image"].image, "wrist_camera", node))

    node.create_timer(1.0 / 15.0, publish)
    rviz_process: subprocess.Popen | None = None
    if not args.no_rviz:
        config = resources.files("fr3_demo").joinpath("config/cameras.rviz")
        with resources.as_file(config) as config_path:
            rviz_process = subprocess.Popen(["rviz2", "-d", str(config_path)])

    print("Publishing /fr3_demo/exterior_image_left and /fr3_demo/wrist_image. Press Ctrl+C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cameras.close()
        if rviz_process is not None:
            rviz_process.terminate()
            try:
                rviz_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                rviz_process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(rviz_main())
