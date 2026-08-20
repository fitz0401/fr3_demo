"""Joystick teleoperation entry point for a Bamboo-controlled Franka FR3."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bamboo import BambooFrankaClient

from fr3_demo.joystick import JoystickSnapshot, LinuxJoystick, axis, button, shaped_axis
from fr3_demo.kinematics import WorkspaceBounds, forward_kinematics, resolved_rate_step
from fr3_demo.settings import default_config_path, load_config, teleop_defaults

LOG = logging.getLogger("fr3_teleop")
HOME_JOINTS = np.array([-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031])


@dataclass(frozen=True)
class Mapping:
    """Native Linux mapping for the 11c0:5506 BETOP controller."""

    close_button: int = 2  # A
    open_button: int = 1  # B
    home_button: int = 9  # Menu/Start
    quit_button: int = 8  # Back/Select
    record_button: int = 3  # X
    x_axis: int = 1  # left stick vertical, forward is negative
    y_axis: int = 0  # left stick horizontal
    z_down_button: int = 6  # LT
    z_up_button: int = 4  # LB
    roll_axis: int = 2  # right stick horizontal
    pitch_axis: int = 3  # right stick vertical
    yaw_down_button: int = 7  # RT
    yaw_up_button: int = 5  # RB


def joystick_twist(
    snapshot: JoystickSnapshot,
    mapping: Mapping,
    linear_speed: float,
    angular_speed: float,
    deadzone: float,
) -> np.ndarray:
    """Map a BETOP-style gamepad snapshot to a base-frame Cartesian twist."""

    def shape(value: float) -> float:
        return shaped_axis(value, deadzone=deadzone)

    z = float(button(snapshot, mapping.z_up_button)) - float(button(snapshot, mapping.z_down_button))
    yaw = float(button(snapshot, mapping.yaw_up_button)) - float(button(snapshot, mapping.yaw_down_button))
    return np.array(
        [
            -shape(axis(snapshot, mapping.x_axis)) * linear_speed,
            shape(axis(snapshot, mapping.y_axis)) * linear_speed,
            z * linear_speed,
            shape(axis(snapshot, mapping.roll_axis)) * angular_speed,
            -shape(axis(snapshot, mapping.pitch_axis)) * angular_speed,
            yaw * angular_speed,
        ]
    )


class BambooRobot:
    """Small teleop adapter around Bamboo's public Python client."""

    def __init__(
        self,
        server_ip: str,
        control_port: int,
        gripper_port: int,
        gripper_type: str,
        enable_gripper: bool,
    ) -> None:
        self._arm = BambooFrankaClient(
            server_ip=server_ip,
            control_port=control_port,
            enable_gripper=False,
        )
        self._gripper: BambooFrankaClient | None = None
        self._server_ip = server_ip
        self._control_port = control_port
        self._gripper_port = gripper_port
        self._gripper_type = gripper_type
        self._enable_gripper = enable_gripper
        try:
            self.state()
        except Exception as error:
            self.close()
            raise RuntimeError(f"Could not connect to Bamboo at {server_ip}:{control_port}: {error}") from error

    @staticmethod
    def _require_success(response: dict, action: str) -> None:
        if not response.get("success", False):
            raise RuntimeError(str(response.get("error", f"Bamboo failed to {action}")))

    def state(self) -> dict:
        return self._arm.get_joint_states()

    def supports_streaming(self) -> bool:
        return self._arm.supports_streaming()

    def start_stream(self, watchdog_ms: int, max_velocity: float, max_acceleration: float) -> None:
        if not self.supports_streaming():
            raise RuntimeError(
                "The running Bamboo controller has no streaming support. Update fitz0401/bamboo on the "
                "real-time machine and launch RunTeleopController."
            )
        response = self._arm.start_streaming(
            watchdog_ms=watchdog_ms,
            max_joint_velocity=max_velocity,
            max_joint_acceleration=max_acceleration,
        )
        self._require_success(response, "start streaming")

    def stream_velocity(self, velocity: np.ndarray) -> None:
        response = self._arm.stream_joint_velocity(velocity)
        self._require_success(response, "update streaming velocity")

    def stop_stream(self) -> None:
        response = self._arm.stop_streaming()
        self._require_success(response, "stop streaming")

    def legacy_command(self, target: np.ndarray, duration: float) -> None:
        response = self._arm.execute_joint_impedance_path(
            np.asarray(target, dtype=float)[None, :],
            durations=[duration],
            default_duration=duration,
        )
        self._require_success(response, "execute legacy waypoint")

    def _gripper_client(self) -> BambooFrankaClient:
        if not self._enable_gripper:
            raise RuntimeError("Gripper control is disabled")
        if self._gripper is None:
            try:
                from bamboo import BambooFrankaClient
            except ImportError as error:
                raise RuntimeError("Bamboo client is required for gripper control") from error
            self._gripper = BambooFrankaClient(
                server_ip=self._server_ip,
                control_port=self._control_port,
                gripper_port=self._gripper_port,
                gripper_type=self._gripper_type,
                enable_gripper=True,
            )
        return self._gripper

    def open_gripper(self) -> None:
        result = self._gripper_client().open_gripper(speed=0.05, force=0.1, blocking=True)
        if not result.get("success", False):
            raise RuntimeError("Bamboo failed to open gripper")

    def close_gripper(self) -> None:
        result = self._gripper_client().close_gripper(speed=0.05, force=0.25, blocking=True)
        if not result.get("success", False):
            raise RuntimeError("Bamboo failed to close gripper")

    def gripper_position(self) -> float:
        """Return normalized gripper openness (0 closed, 1 open)."""

        result = self._gripper_client().get_gripper_state()
        self._require_success(result, "read gripper state")
        width = float(result["state"]["width"])
        maximum_width = 0.085 if self._gripper_type == "robotiq" else 0.08
        return float(np.clip(width / maximum_width, 0.0, 1.0))

    def close(self) -> None:
        if self._gripper is not None:
            self._gripper.close()
        self._arm.close()


def stream_home(
    robot: BambooRobot,
    target: np.ndarray,
    max_velocity: float,
    minimum_timeout: float,
    rate_hz: float,
    stop_requested: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Move to a joint target using Bamboo's bounded streaming controller."""

    target = np.asarray(target, dtype=float)
    period = 1.0 / rate_hz
    position_tolerance = 0.01
    velocity_tolerance = 0.03
    gain = 1.2
    started = time.monotonic()
    deadline: float | None = None

    while True:
        loop_started = time.monotonic()
        if stop_requested is not None and stop_requested():
            robot.stream_velocity(np.zeros(7))
            raise InterruptedError("Homing interrupted")

        state = robot.state()
        q = np.asarray(state["qpos"], dtype=float)
        dq = np.asarray(state["dq"], dtype=float)
        error = target - q
        max_error = float(np.max(np.abs(error)))

        if deadline is None:
            # Large moves need more time even when the configured minimum is short.
            timeout = max(minimum_timeout, max_error / max_velocity + 5.0)
            deadline = started + timeout

        command = np.zeros(7) if max_error <= position_tolerance else np.clip(gain * error, -max_velocity, max_velocity)
        robot.stream_velocity(command)

        if max_error <= position_tolerance and float(np.linalg.norm(dq)) <= velocity_tolerance:
            return q
        if time.monotonic() >= deadline:
            robot.stream_velocity(np.zeros(7))
            raise RuntimeError(f"Homing timed out with {max_error:.3f} rad maximum joint error")

        remaining = period - (time.monotonic() - loop_started)
        if remaining > 0:
            time.sleep(remaining)


def _pose_error(model: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(model[:3, 3] - measured[:3, 3]))
    difference = model[:3, :3].T @ measured[:3, :3]
    angle_error = float(np.arccos(np.clip((np.trace(difference) - 1.0) / 2.0, -1.0, 1.0)))
    return position_error, angle_error


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate a Bamboo-controlled Franka FR3 with a gamepad.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="shared TOML config (default: %(default)s; override with FR3_DEMO_CONFIG)",
    )
    parser.add_argument("--server-ip", default="172.16.0.20", help="machine running Bamboo (default: %(default)s)")
    parser.add_argument("--control-port", type=int, default=5555)
    parser.add_argument("--gripper-port", type=int, default=5559)
    parser.add_argument("--gripper-type", choices=("robotiq", "franka"), default="robotiq")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--joystick", default="/dev/input/js0")
    parser.add_argument("--linear-speed", type=float, default=0.08, help="maximum EEF translation speed in m/s")
    parser.add_argument("--angular-speed", type=float, default=0.35, help="maximum EEF rotation speed in rad/s")
    parser.add_argument("--stream-rate", type=float, default=30.0, help="joystick setpoint rate in Hz")
    parser.add_argument("--watchdog-ms", type=int, default=250, help="controller braking timeout in milliseconds")
    parser.add_argument("--max-joint-speed", type=float, default=0.35, help="teleop joint-speed cap in rad/s")
    parser.add_argument("--max-joint-acceleration", type=float, default=1.5, help="stream acceleration cap in rad/s^2")
    parser.add_argument(
        "--legacy-waypoints",
        action="store_true",
        help="use Bamboo's old blocking waypoint API (compatible but stop-and-go)",
    )
    parser.add_argument("--command-period", type=float, default=0.12, help="legacy waypoint duration in seconds")
    parser.add_argument("--home-speed", type=float, default=0.20, help="maximum homing joint speed in rad/s")
    parser.add_argument("--home-timeout", type=float, default=15.0, help="minimum homing timeout in seconds")
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--frame", choices=("base", "tool"), default="base")
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.10, -0.60, 0.05))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.80, 0.60, 1.00))
    parser.add_argument("--check", action="store_true", help="check controller, model, and joystick without moving")
    parser.add_argument("--dry-run", action="store_true", help="run teleop and print targets without sending commands")
    parser.add_argument("--offline", action="store_true", help="dry-run without connecting to Bamboo")
    parser.add_argument("--collect", action="store_true", help="enable synchronized demonstration collection")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="raw recording root")
    parser.add_argument(
        "--external-camera-serial",
        default=os.environ.get("FR3_EXTERNAL_CAMERA_SERIAL"),
        help="RealSense serial for exterior_image_left (or FR3_EXTERNAL_CAMERA_SERIAL)",
    )
    parser.add_argument(
        "--wrist-camera-serial",
        default=os.environ.get("FR3_WRIST_CAMERA_SERIAL"),
        help="RealSense serial for wrist_image (or FR3_WRIST_CAMERA_SERIAL)",
    )
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--wrist-vertical-flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="flip wrist images top-to-bottom for an inverted camera mount",
    )
    parser.add_argument("--record-fps", type=float, default=15.0, help="synchronized dataset sampling rate")
    parser.add_argument("--feedback-event", help="force-feedback event device; auto-detected by default")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.offline and not (args.dry_run or args.check):
        parser.error("--offline requires --dry-run or --check")
    for name in (
        "linear_speed",
        "angular_speed",
        "stream_rate",
        "command_period",
        "home_speed",
        "home_timeout",
        "max_joint_speed",
        "max_joint_acceleration",
        "camera_fps",
        "record_fps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 50 <= args.watchdog_ms <= 500:
        parser.error("--watchdog-ms must be between 50 and 500")
    if not 0.0 <= args.deadzone < 0.9:
        parser.error("--deadzone must be in [0, 0.9)")
    if np.any(np.asarray(args.workspace_min) >= np.asarray(args.workspace_max)):
        parser.error("each --workspace-min value must be below --workspace-max")
    if args.collect and (args.offline or args.dry_run or args.check or args.legacy_waypoints):
        parser.error(
            "--collect requires live streaming mode (not --offline, --dry-run, --check, or --legacy-waypoints)"
        )
    if args.collect and (not args.external_camera_serial or not args.wrist_camera_serial):
        parser.error(
            "--collect requires --external-camera-serial and --wrist-camera-serial "
            "(or FR3_EXTERNAL_CAMERA_SERIAL and FR3_WRIST_CAMERA_SERIAL)"
        )


def _diagnostics(joystick: LinuxJoystick, robot: BambooRobot | None, offline_q: np.ndarray) -> np.ndarray:
    time.sleep(0.20)  # allow Linux initial-state events to be drained
    snapshot = joystick.snapshot()
    if not snapshot.connected:
        raise RuntimeError("Joystick disconnected during startup")
    print(f"Joystick: {snapshot.name!r}, {len(snapshot.axes)} axes, {len(snapshot.buttons)} buttons")
    if len(snapshot.axes) < 6 or len(snapshot.buttons) < 12:
        raise RuntimeError("Expected the BETOP native layout with at least 6 axes and 12 buttons")

    if robot is None:
        print("Bamboo: offline (nominal FR3 state used)")
        return offline_q

    state = robot.state()
    q = np.asarray(state["qpos"], dtype=float)
    measured_pose = np.asarray(state["ee_pose"], dtype=float)
    model_pose = forward_kinematics(q)
    position_error, angle_error = _pose_error(model_pose, measured_pose)
    print(f"Bamboo: connected; q={np.array2string(q, precision=3)}")
    print(f"EEF: xyz={np.array2string(measured_pose[:3, 3], precision=4)}")
    print(f"Model check: {position_error * 1000:.2f} mm, {np.degrees(angle_error):.3f} deg")
    print(f"Streaming protocol: {'available' if robot.supports_streaming() else 'not installed'}")
    if position_error > 0.01 or angle_error > np.radians(2.0):
        raise RuntimeError("FR3 model does not agree with Bamboo state; motion remains disabled")
    return q


def run(args: argparse.Namespace) -> int:
    mapping = Mapping()
    workspace = WorkspaceBounds(tuple(args.workspace_min), tuple(args.workspace_max))
    nominal_q = HOME_JOINTS.copy()
    robot: BambooRobot | None = None
    cameras: Any | None = None
    collector: Any | None = None
    rumbler: Any | None = None
    streaming_active = False
    stop = False

    def request_stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        if not args.offline:
            robot = BambooRobot(
                args.server_ip,
                args.control_port,
                args.gripper_port,
                args.gripper_type,
                not args.no_gripper,
            )

        if args.collect:
            from fr3_demo.cameras import RealSensePair
            from fr3_demo.feedback import Rumbler
            from fr3_demo.recording import DemoCollector

            cameras = RealSensePair(
                args.external_camera_serial,
                args.wrist_camera_serial,
                fps=args.camera_fps,
                wrist_vertical_flip=args.wrist_vertical_flip,
            ).start()
            collector = DemoCollector(
                cameras,
                args.data_dir,
                args.server_ip,
                args.control_port,
                fps=args.record_fps,
                gripper_port=args.gripper_port,
                gripper_type=args.gripper_type,
                enable_gripper=not args.no_gripper,
            )
            collector.start()
            if robot is not None and not args.no_gripper:
                collector.set_gripper(robot.gripper_position())
            rumbler = Rumbler(args.feedback_event)
            print(f"Raw demonstration session: {collector.session_dir}")

        with LinuxJoystick(args.joystick) as joystick:
            q = _diagnostics(joystick, robot, nominal_q)
            if args.check:
                print("Check complete: no motion command was sent.")
                return 0

            streaming = not args.dry_run and not args.legacy_waypoints
            if streaming and robot is not None:
                robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
                streaming_active = True
                print(f"Streaming control active at {args.stream_rate:.0f} Hz (watchdog: {args.watchdog_ms} ms).")
            elif args.legacy_waypoints and not args.dry_run:
                LOG.warning("Legacy waypoint mode intentionally stops after every command and will feel laggy")

            print("\nTeleoperation is active; moving a control commands the robot immediately.")
            print("Left stick: X/Y | LT/LB: Z down/up | right stick: roll/pitch | RT/RB: yaw")
            controls = "A: close gripper | B: open gripper | Menu: home | Back: quit"
            if collector is not None:
                controls += " | X: start/stop recording"
            print(controls)
            print("Keep a hand on the physical E-stop.\n")

            initial_snapshot = joystick.snapshot()
            last_close_count = initial_snapshot.press_counts[mapping.close_button]
            last_open_count = initial_snapshot.press_counts[mapping.open_button]
            last_home_count = initial_snapshot.press_counts[mapping.home_button]
            last_record_count = initial_snapshot.press_counts[mapping.record_button]
            last_status = 0.0
            idle_sleep = min(0.02, args.command_period / 4.0)
            stream_period = 1.0 / args.stream_rate

            while not stop:
                loop_started = time.monotonic()
                snapshot = joystick.snapshot()
                if not snapshot.connected:
                    raise RuntimeError("Joystick disconnected; teleoperation stopped")
                if button(snapshot, mapping.quit_button):
                    if collector is not None and collector.active:
                        completed = collector.stop_episode()
                        collector.set_action(np.zeros(7))
                        rumbler.recording_stopped()
                        print(f"\nRecording stopped: {completed}")
                    break

                if collector is not None:
                    collector.check_health()

                close_count = snapshot.press_counts[mapping.close_button]
                open_count = snapshot.press_counts[mapping.open_button]
                home_count = snapshot.press_counts[mapping.home_button]
                record_count = snapshot.press_counts[mapping.record_button]
                close_requested = close_count > last_close_count
                open_requested = open_count > last_open_count
                home_requested = home_count > last_home_count
                record_requested = record_count > last_record_count
                last_close_count, last_open_count = close_count, open_count
                last_home_count = home_count
                last_record_count = record_count

                if record_requested and collector is not None:
                    if streaming and robot is not None:
                        robot.stream_velocity(np.zeros(7))
                    collector.set_action(np.zeros(7))
                    if collector.active:
                        completed = collector.stop_episode()
                        rumbler.recording_stopped()
                        print(f"\nRecording stopped: {completed}")
                    else:
                        pending = collector.start_episode()
                        rumbler.recording_started()
                        print(f"\nRecording started: {pending}")
                    continue

                if home_requested:
                    if collector is not None and collector.active:
                        collector.set_action(np.zeros(7))
                        rumbler.error()
                        print("\nStop recording with X before homing.")
                        continue
                    if args.dry_run:
                        q = HOME_JOINTS.copy()
                        print(f"DRY RUN: home q={np.array2string(q, precision=3)}")
                    elif robot is not None:
                        temporary_stream = not streaming
                        if temporary_stream:
                            robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
                        print(f"Homing at up to {min(args.home_speed, args.max_joint_speed):.2f} rad/s...")
                        try:
                            q = stream_home(
                                robot,
                                HOME_JOINTS,
                                min(args.home_speed, args.max_joint_speed),
                                args.home_timeout,
                                args.stream_rate,
                                stop_requested=lambda: stop,
                            )
                        finally:
                            if temporary_stream:
                                robot.stop_stream()
                        print("Homing complete; teleoperation is active.")
                    continue

                if close_requested or open_requested:
                    if streaming and robot is not None:
                        robot.stream_velocity(np.zeros(7))
                    if collector is not None:
                        collector.set_action(np.zeros(7))
                        requested_gripper = 0.0 if close_requested else 1.0
                        collector.set_gripper(requested_gripper, requested_gripper)
                    if args.dry_run:
                        print("DRY RUN: gripper close" if close_requested else "DRY RUN: gripper open")
                    elif robot is not None:
                        robot.close_gripper() if close_requested else robot.open_gripper()
                        if collector is not None:
                            collector.set_gripper(robot.gripper_position())
                    continue

                twist = joystick_twist(snapshot, mapping, args.linear_speed, args.angular_speed, args.deadzone)
                if args.frame == "tool":
                    rotation = forward_kinematics(q)[:3, :3]
                    twist = np.concatenate((rotation @ twist[:3], rotation @ twist[3:]))

                moving = float(np.linalg.norm(twist)) > 1e-8
                if moving:
                    step_duration = stream_period if streaming else args.command_period
                    try:
                        target, joint_velocity = resolved_rate_step(
                            q,
                            twist,
                            step_duration,
                            max_joint_velocity=args.max_joint_speed,
                            workspace=workspace,
                        )
                    except ValueError as error:
                        LOG.warning("Command blocked: %s", error)
                        if collector is not None:
                            collector.set_action(np.zeros(7))
                        if streaming and robot is not None:
                            robot.stream_velocity(np.zeros(7))
                        time.sleep(idle_sleep)
                        continue

                    if args.dry_run:
                        q = target
                        now = time.monotonic()
                        if now - last_status >= 0.2:
                            xyz = forward_kinematics(q)[:3, 3]
                            print(f"\rDRY RUN target xyz={np.array2string(xyz, precision=3)}", end="", flush=True)
                            last_status = now
                        time.sleep(args.command_period)
                    elif robot is not None:
                        if collector is not None:
                            collector.set_action(joint_velocity)
                        if streaming:
                            robot.stream_velocity(joint_velocity)
                        else:
                            robot.legacy_command(target, args.command_period)
                else:
                    if collector is not None:
                        collector.set_action(np.zeros(7))
                    if streaming and robot is not None:
                        # An explicit zero begins braking immediately; the watchdog is the backup.
                        robot.stream_velocity(np.zeros(7))
                    if robot is not None and not streaming and time.monotonic() - last_status >= 0.25:
                        q = np.asarray(robot.state()["qpos"], dtype=float)
                        last_status = time.monotonic()
                    if not streaming:
                        time.sleep(idle_sleep)

                if streaming and robot is not None:
                    q = np.asarray(robot.state()["qpos"], dtype=float)
                    remaining = stream_period - (time.monotonic() - loop_started)
                    if remaining > 0:
                        time.sleep(remaining)

            if args.dry_run:
                print()
            print("Teleoperation stopped; no further waypoints will be sent.")
            if collector is not None:
                print(f"Add language with: fr3-annotate --data-dir {collector.session_dir}")
            return 0
    finally:
        if collector is not None:
            try:
                completed = collector.close()
                if completed is not None:
                    print(f"Recording stopped during shutdown: {completed}")
            except Exception as error:  # noqa: BLE001 - shutdown must continue to camera and robot cleanup
                LOG.error("Could not finalize active recording: %s", error)
        if cameras is not None:
            cameras.close()
        if robot is not None:
            if streaming_active:
                try:
                    robot.stop_stream()
                except Exception as error:  # noqa: BLE001 - closing the client still triggers the watchdog
                    LOG.warning("Could not explicitly stop Bamboo stream: %s", error)
            robot.close()


def collect_main() -> int:
    """Run live teleoperation with demonstration collection enabled."""

    return main(["--collect", *sys.argv[1:]])


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = _create_parser()
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=default_config_path())
    bootstrap_args, _ = bootstrap.parse_known_args(raw_argv)
    try:
        defaults = teleop_defaults(load_config(bootstrap_args.config))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    environment_defaults = {
        "external_camera_serial": os.environ.get("FR3_EXTERNAL_CAMERA_SERIAL"),
        "wrist_camera_serial": os.environ.get("FR3_WRIST_CAMERA_SERIAL"),
    }
    defaults.update({key: value for key, value in environment_defaults.items() if value})
    parser.set_defaults(config=bootstrap_args.config, **defaults)
    args = parser.parse_args(raw_argv)
    _validate_args(args, parser)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run(args)
    except Exception as error:
        LOG.error("%s", error)
        if args.verbose:
            LOG.exception("Teleoperation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
