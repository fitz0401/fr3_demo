"""Shared TOML configuration for FR3 teleoperation and collection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib


_SCHEMA: dict[str, set[str]] = {
    "robot": {"server_ip", "control_port"},
    "gripper": {"port", "type", "enabled"},
    "joystick": {"device", "feedback_event"},
    "cameras": {"external_serial", "wrist_serial", "fps"},
    "recording": {"data_dir", "fps"},
    "teleop": {
        "linear_speed",
        "angular_speed",
        "stream_rate",
        "watchdog_ms",
        "max_joint_speed",
        "max_joint_acceleration",
        "deadzone",
        "frame",
        "command_period",
    },
    "home": {"speed", "timeout"},
    "workspace": {"min", "max"},
    "pi05": {
        "server_host",
        "server_port",
        "control_hz",
        "stream_hz",
        "open_loop_horizon",
        "prefetch_actions",
        "max_joint_speed",
        "max_joint_acceleration",
        "joint_margin",
        "watchdog_ms",
        "max_camera_age",
        "max_inference_age",
        "max_rollout_steps",
        "rviz",
    },
}


def default_config_path() -> Path:
    """Return the configured path, repository config, or per-user config path."""

    environment_path = os.environ.get("FR3_DEMO_CONFIG")
    if environment_path:
        return Path(environment_path).expanduser()
    repository_path = Path(__file__).resolve().parents[1] / "config.toml"
    if repository_path.is_file():
        return repository_path
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "fr3_demo" / "config.toml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate an FR3 configuration file."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FR3 config file does not exist: {path}")
    with path.open("rb") as stream:
        config = tomllib.load(stream)

    unknown_sections = set(config) - ({"version"} | set(_SCHEMA))
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(sorted(unknown_sections))}")
    if config.get("version") != 1:
        raise ValueError("FR3 config must contain: version = 1")
    for section, allowed_keys in _SCHEMA.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise TypeError(f"Config [{section}] must be a TOML table")
        unknown_keys = set(values) - allowed_keys
        if unknown_keys:
            names = ", ".join(f"{section}.{key}" for key in sorted(unknown_keys))
            raise ValueError(f"Unknown config key(s): {names}")
    return config


def teleop_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Translate structured TOML values to argparse destination names."""

    robot = config.get("robot", {})
    gripper = config.get("gripper", {})
    joystick = config.get("joystick", {})
    cameras = config.get("cameras", {})
    recording = config.get("recording", {})
    teleop = config.get("teleop", {})
    home = config.get("home", {})
    workspace = config.get("workspace", {})
    mapping = {
        "server_ip": robot.get("server_ip"),
        "control_port": robot.get("control_port"),
        "gripper_port": gripper.get("port"),
        "gripper_type": gripper.get("type"),
        "no_gripper": None if "enabled" not in gripper else not gripper["enabled"],
        "joystick": joystick.get("device"),
        "feedback_event": joystick.get("feedback_event"),
        "external_camera_serial": cameras.get("external_serial"),
        "wrist_camera_serial": cameras.get("wrist_serial"),
        "camera_fps": cameras.get("fps"),
        "data_dir": recording.get("data_dir"),
        "record_fps": recording.get("fps"),
        "linear_speed": teleop.get("linear_speed"),
        "angular_speed": teleop.get("angular_speed"),
        "stream_rate": teleop.get("stream_rate"),
        "watchdog_ms": teleop.get("watchdog_ms"),
        "max_joint_speed": teleop.get("max_joint_speed"),
        "max_joint_acceleration": teleop.get("max_joint_acceleration"),
        "deadzone": teleop.get("deadzone"),
        "frame": teleop.get("frame"),
        "command_period": teleop.get("command_period"),
        "home_speed": home.get("speed"),
        "home_timeout": home.get("timeout"),
        "workspace_min": workspace.get("min"),
        "workspace_max": workspace.get("max"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def camera_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return settings used by the standalone RViz camera preview."""

    cameras = config.get("cameras", {})
    mapping = {
        "external_camera_serial": cameras.get("external_serial"),
        "wrist_camera_serial": cameras.get("wrist_serial"),
        "camera_fps": cameras.get("fps"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def pi05_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Translate shared robot/camera/pi0.5 settings to runner CLI names."""

    robot = config.get("robot", {})
    gripper = config.get("gripper", {})
    joystick = config.get("joystick", {})
    cameras = config.get("cameras", {})
    workspace = config.get("workspace", {})
    pi05 = config.get("pi05", {})
    mapping = {
        "server_ip": robot.get("server_ip"),
        "control_port": robot.get("control_port"),
        "gripper_port": gripper.get("port"),
        "gripper_type": gripper.get("type"),
        "no_gripper": None if "enabled" not in gripper else not gripper["enabled"],
        "joystick": joystick.get("device"),
        "external_camera_serial": cameras.get("external_serial"),
        "wrist_camera_serial": cameras.get("wrist_serial"),
        "camera_fps": cameras.get("fps"),
        "workspace_min": workspace.get("min"),
        "workspace_max": workspace.get("max"),
        "policy_host": pi05.get("server_host"),
        "policy_port": pi05.get("server_port"),
        "control_hz": pi05.get("control_hz"),
        "stream_hz": pi05.get("stream_hz"),
        "open_loop_horizon": pi05.get("open_loop_horizon"),
        "prefetch_actions": pi05.get("prefetch_actions"),
        "max_joint_speed": pi05.get("max_joint_speed"),
        "max_joint_acceleration": pi05.get("max_joint_acceleration"),
        "joint_margin": pi05.get("joint_margin"),
        "watchdog_ms": pi05.get("watchdog_ms"),
        "max_camera_age": pi05.get("max_camera_age"),
        "max_inference_age": pi05.get("max_inference_age"),
        "max_steps": pi05.get("max_rollout_steps"),
        "no_rviz": None if "rviz" not in pi05 else not pi05["rviz"],
    }
    return {key: value for key, value in mapping.items() if value is not None}
