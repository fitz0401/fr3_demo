"""ROS 2/RViz visualization for FR3 state, cameras, and pi0.5 trajectory."""

from __future__ import annotations

import os
import subprocess
from importlib import resources
from typing import Any

import numpy as np

from fr3_demo.kinematics import forward_kinematics, link_positions


def _rviz_environment() -> dict[str, str]:
    """Remove editor Snap runtime paths that are binary-incompatible with ROS."""

    environment = os.environ.copy()
    incompatible_prefixes = ("SNAP", "GTK_", "GIO_")
    for key in tuple(environment):
        if key.startswith(incompatible_prefixes) or key == "LD_PRELOAD":
            environment.pop(key, None)
    return environment


class RvizBridge:
    """Publish a dependency-light kinematic robot view and predicted EEF path."""

    def __init__(self, launch_rviz: bool = True) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from visualization_msgs.msg import MarkerArray
        except ImportError as error:
            raise RuntimeError("ROS 2 is not sourced; run: source /opt/ros/humble/setup.bash") from error

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("fr3_pi05_visualization")
        self._image_type = Image
        self._marker_array_type = MarkerArray
        self._external = self._node.create_publisher(
            Image, "/fr3_pi05/exterior_image_left", qos_profile_sensor_data
        )
        self._wrist = self._node.create_publisher(Image, "/fr3_pi05/wrist_image", qos_profile_sensor_data)
        self._markers = self._node.create_publisher(MarkerArray, "/fr3_pi05/markers", 10)
        self._rviz: subprocess.Popen[Any] | None = None
        if launch_rviz:
            config = resources.files("fr3_pi05").joinpath("config/pi05.rviz")
            with resources.as_file(config) as config_path:
                self._rviz = subprocess.Popen(
                    ["rviz2", "-d", str(config_path)],
                    env=_rviz_environment(),
                )

    def _image_message(self, image: np.ndarray, frame_id: str) -> Any:
        message = self._image_type()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
        return message

    def _line_marker(
        self,
        marker_id: int,
        namespace: str,
        positions: np.ndarray,
        rgba: tuple[float, float, float, float],
        width: float,
    ) -> Any:
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions]
        return marker

    def publish(
        self,
        exterior_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_position: np.ndarray,
        predicted_joint_path: np.ndarray | None,
    ) -> None:
        self._external.publish(self._image_message(exterior_image, "exterior_camera"))
        self._wrist.publish(self._image_message(wrist_image, "wrist_camera"))

        markers = self._marker_array_type()
        markers.markers.append(
            self._line_marker(0, "measured_robot", link_positions(joint_position), (0.15, 0.55, 1.0, 1.0), 0.035)
        )
        if predicted_joint_path is not None and len(predicted_joint_path):
            eef_path = np.stack([forward_kinematics(q)[:3, 3] for q in predicted_joint_path])
            markers.markers.append(
                self._line_marker(1, "policy_eef_path", eef_path, (0.1, 1.0, 0.25, 1.0), 0.012)
            )
            markers.markers.append(
                self._line_marker(
                    2,
                    "predicted_robot",
                    link_positions(predicted_joint_path[-1]),
                    (1.0, 0.55, 0.05, 0.65),
                    0.018,
                )
            )
        self._markers.publish(markers)
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if self._rviz is not None:
            self._rviz.terminate()
            try:
                self._rviz.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._rviz.kill()
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self) -> RvizBridge:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
