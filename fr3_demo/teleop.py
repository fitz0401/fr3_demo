"""Joystick teleoperation entry point for a Bamboo-controlled Franka FR3."""

from __future__ import annotations

import argparse
import atexit
import faulthandler
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

from fr3_demo.contact import describe_contact_limits
from fr3_demo.gel import GelStreamGuard, limit_twist
from fr3_demo.guide import GuideHeartbeat, choose_guide_mode, describe_guide_failure, hold_message
from fr3_demo.joystick import JoystickSnapshot, LinuxJoystick, axis, button, shaped_axis
from fr3_demo.kinematics import WorkspaceBounds, forward_kinematics, resolved_rate_step
from fr3_demo.replay import PATH_GAIN, REPLAY_ACTION_SOURCE, ReplayPath, link_replay, run_replay
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
    tool_y_axis: int = 4  # D-pad horizontal: left is positive, right is negative
    tool_z_axis: int = 5  # D-pad vertical: up is negative, down is positive
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
            -shape(axis(snapshot, mapping.y_axis)) * linear_speed,
            z * linear_speed,
            shape(axis(snapshot, mapping.roll_axis)) * angular_speed,
            -shape(axis(snapshot, mapping.pitch_axis)) * angular_speed,
            yaw * angular_speed,
        ]
    )


def joystick_tool_z(
    snapshot: JoystickSnapshot,
    mapping: Mapping,
    linear_speed: float,
    deadzone: float,
) -> float:
    """Return D-pad translation along the current EEF Z axis."""

    return -shaped_axis(axis(snapshot, mapping.tool_z_axis), deadzone=deadzone) * linear_speed


def joystick_tool_y(
    snapshot: JoystickSnapshot,
    mapping: Mapping,
    linear_speed: float,
    deadzone: float,
) -> float:
    """Return D-pad translation along the current EEF Y axis."""

    return -shaped_axis(axis(snapshot, mapping.tool_y_axis), deadzone=deadzone) * linear_speed


def lock_dpad_axis(tool_y: float, tool_z: float, active_axis: str | None) -> tuple[float, float, str | None]:
    """Allow only one D-pad translation axis, retaining the first axis pressed."""

    y_active = abs(tool_y) > 1e-8
    z_active = abs(tool_z) > 1e-8
    if (active_axis == "y" and not y_active) or (active_axis == "z" and not z_active):
        active_axis = None
    if active_axis is None:
        if y_active:
            active_axis = "y"
        elif z_active:
            active_axis = "z"
    return (
        tool_y if active_axis == "y" else 0.0,
        tool_z if active_axis == "z" else 0.0,
        active_axis,
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
        gripper_force: float = 0.8,
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
        self._gripper_force = gripper_force
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

    def yield_stream(self, reason: str) -> None:
        """Make the active stream stop pushing and end (Bamboo goes soft, then holds)."""

        self._require_success(self._arm.yield_streaming(reason), "yield the stream")

    def stream_velocity(self, velocity: np.ndarray) -> None:
        response = self._arm.stream_joint_velocity(velocity)
        self._require_success(response, "update streaming velocity")

    def stop_stream(self) -> None:
        response = self._arm.stop_streaming()
        self._require_success(response, "stop streaming")

    def supports(self, feature: str) -> bool:
        supports = getattr(self._arm, "supports", None)  # absent in older Bamboo clients
        return bool(supports(feature)) if supports is not None else False

    def new_arm_client(self) -> BambooFrankaClient:
        """Open a separate arm connection, for a thread that must not share this one."""

        return BambooFrankaClient(server_ip=self._server_ip, control_port=self._control_port, enable_gripper=False)

    def start_guide(
        self,
        watchdog_ms: int,
        max_joint_velocity: float,
        idle_hold_s: float | None = None,
        max_start_force: float | None = None,
    ) -> None:
        options: dict[str, float] = {}
        if idle_hold_s is not None:
            options["idle_hold_s"] = idle_hold_s
        if max_start_force is not None:
            options["max_start_force"] = max_start_force
        response = self._arm.start_guiding(watchdog_ms=watchdog_ms, max_joint_velocity=max_joint_velocity, **options)
        if not response.get("success", False):
            error = str(response.get("error", "Bamboo refused to start guiding"))
            raise RuntimeError(describe_guide_failure("Could not start guide mode", error))

    def resume_guide(self) -> None:
        self._require_success(self._arm.resume_guiding(), "resume guiding")

    def stop_guide(self) -> None:
        self._require_success(self._arm.stop_guiding(), "stop guiding")

    def set_load(self, mass: float, center_of_mass: list[float] | np.ndarray) -> None:
        if not self.supports("set_load"):
            raise RuntimeError(
                "The running Bamboo controller cannot set a payload. Update Bamboo, or remove the payload "
                "from the config."
            )
        self._require_success(self._arm.set_load(mass, center_of_mass), "set the payload")
        # Franka blends a new load in over about a second. Wait until it reports
        # the full mass, so a guide start right after cannot see a half-applied
        # load as an external force and refuse.
        deadline = time.monotonic() + 3.0
        while True:
            reported = float(self._arm.get_capabilities().get("payload", {}).get("m_load", mass))
            if abs(reported - mass) <= max(0.01, 0.01 * mass):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Franka reports a {reported:.3f} kg payload, not the {mass:.3f} kg requested")
            time.sleep(0.05)

    def apply_contact_limits(self, limits: dict[str, Any] | None) -> dict[str, Any] | None:
        """Set [contact] limits for the following streams; with None, clear any left over.

        Bamboo keeps limits until they are cleared or it restarts, so a session
        without [contact] must not inherit the previous session's. Returns what
        Bamboo applied, or None.
        """

        if limits is None:
            if self.supports("contact_limits"):
                self._require_success(self._arm.clear_contact_limits(), "clear contact limits")
            return None
        if not self.supports("contact_limits"):
            raise RuntimeError(
                "The running Bamboo controller cannot limit contact forces. Update Bamboo, or turn [contact] off."
            )
        response = self._arm.set_contact_limits(**limits)
        self._require_success(response, "set contact limits")
        return dict(response.get("limits", limits))

    def clear_contact_limits(self, timeout: float = 1.5) -> None:
        """Clear the limits; a stream that just yielded may need a moment to end first."""

        deadline = time.monotonic() + timeout
        while True:
            response = self._arm.clear_contact_limits()
            busy = "active control loop" in str(response.get("error", ""))
            if response.get("success", False) or not busy or time.monotonic() >= deadline:
                self._require_success(response, "clear contact limits")
                return
            time.sleep(0.05)

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
        result = self._gripper_client().close_gripper(speed=0.05, force=self._gripper_force, blocking=True)
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


HAND_GUIDE_ACTION_SOURCE = "measured_joint_velocity"
# Teleop already stops driving into a loaded gel at level 1 (gel.limit_twist);
# past this the stream yields, in case the load keeps rising anyway.
GEL_TELEOP_STOP_LEVEL = 1.2
# A shutdown has been seen to hang the terminal. If closing everything down
# takes longer than this, every thread's stack is printed (to show where) and
# the process exits; Bamboo's watchdogs hold or brake the arm without us.
SHUTDOWN_TIMEOUT_S = 45.0
# The same, for the interpreter's own exit once shutdown is done.
EXIT_TIMEOUT_S = 10.0


def _watch_interpreter_exit() -> None:
    faulthandler.dump_traceback_later(EXIT_TIMEOUT_S, exit=True)


def hand_guide_tick(robot: Any, collector: Any | None) -> dict:
    """Record the arm's own motion as the action while an operator guides it.

    Nothing is commanded: the measured joint velocity stands in for the
    joystick velocity teleop records, in the same unit (rad/s). Returns the
    robot state, which also carries guide mode's hold status when available.
    """

    state = robot.state()
    if collector is not None:
        collector.set_action(np.asarray(state["dq"], dtype=float))
    return state


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



# Pause between leaving guide mode and homing, for hands to come off the arm.
GUIDE_HOME_PAUSE_S = 1.0
# How long restarting guide mode after homing waits for the arm to be let go.
GUIDE_RESTART_WAIT_S = 10.0
# The same after a replay, which may leave the blade caught in the fruit.
REPLAY_RESTART_WAIT_S = 30.0


def home_from_guide(
    robot: BambooRobot,
    args: argparse.Namespace,
    heartbeat: GuideHeartbeat | None,
    new_heartbeat: Callable[[], GuideHeartbeat],
    stop_requested: Callable[[], bool],
    gel_guard: Any | None = None,
) -> GuideHeartbeat | None:
    """Leave guide mode, home by streaming, and float the arm again.

    Bamboo runs one control loop at a time: guide mode stops (the arm damps to
    a stop and holds), a stream homes the arm (under [contact] when that is
    set, and stopping early if the gel guard objects), and guide mode starts
    again, waiting for the arm to be let go if a hand is still on it. Returns
    the new heartbeat, or None when the session is ending and guide mode was
    left off.
    """

    if heartbeat is not None:
        heartbeat.stop()
    robot.stop_guide()
    print("\nHoming: hands off the arm. Guide mode resumes after homing.", flush=True)
    time.sleep(GUIDE_HOME_PAUSE_S)
    limits = None if args.no_contact_limits else args.contact_limits
    if limits is not None:
        robot.apply_contact_limits(limits)
    speed = min(args.home_speed, args.max_joint_speed)

    def interrupted() -> bool:
        return stop_requested() or (gel_guard is not None and gel_guard.check() is not None)

    homed = False
    robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
    try:
        print(f"Homing at up to {speed:.2f} rad/s...")
        stream_home(robot, HOME_JOINTS, speed, args.home_timeout, args.stream_rate, stop_requested=interrupted)
        homed = True
    except InterruptedError:
        reason = None if gel_guard is None else gel_guard.check()
        print("Homing stopped" + (f": {reason}" if reason else "") + ".")
    finally:
        robot.stop_stream()
        if limits is not None:
            robot.clear_contact_limits()
    if stop_requested():
        return None
    heartbeat = restart_guide(robot, args, new_heartbeat, stop_requested)
    print(("Homing complete; " if homed else "") + "guide mode is back on: the arm floats again.")
    return heartbeat


def restart_guide(
    robot: BambooRobot,
    args: argparse.Namespace,
    new_heartbeat: Callable[[], GuideHeartbeat],
    stop_requested: Callable[[], bool],
    pushed_message: str = "Let go of the arm: guide mode starts once nothing pushes on it.",
    wait_s: float = GUIDE_RESTART_WAIT_S,
) -> GuideHeartbeat:
    """Start guide mode again, waiting while something still pushes on the arm."""

    deadline = time.monotonic() + wait_s
    asked = False
    while True:
        try:
            robot.start_guide(
                args.guide_watchdog_ms,
                args.guide_max_joint_velocity,
                args.guide_idle_hold_s,
                args.guide_max_start_force,
            )
            break
        except RuntimeError as error:
            # Guide mode refuses to start while something pushes on the arm.
            if "External force estimate" not in str(error) or time.monotonic() >= deadline or stop_requested():
                raise
            if not asked:
                print(pushed_message)
                asked = True
            time.sleep(0.5)
    heartbeat = new_heartbeat()
    heartbeat.start()
    return heartbeat


class ReplayControls:
    """The joystick while a replay runs: X or B stops it, Back stops it and ends the session.

    Only presses made after ``mark`` count, so the X that ended the
    demonstration does not stop its replay. When waiting at the start pose, X
    starts the replay and B skips it.
    """

    def __init__(self, joystick: LinuxJoystick, mapping: Mapping, session_ending: Callable[[], bool]) -> None:
        self._joystick = joystick
        self._mapping = mapping
        self._session_ending = session_ending
        self.quit = False
        self.mark()

    def _counts(self, snapshot: JoystickSnapshot) -> tuple[int, int]:
        return snapshot.press_counts[self._mapping.record_button], snapshot.press_counts[self._mapping.open_button]

    def mark(self) -> None:
        """Forget the presses made so far."""

        self._marks = self._counts(self._joystick.snapshot())

    def pressed(self) -> tuple[bool, bool]:
        """Whether X and B were pressed since ``mark``."""

        x, b = self._counts(self._joystick.snapshot())
        return x > self._marks[0], b > self._marks[1]

    def ending(self) -> str | None:
        """Why the whole session must stop, if it must."""

        if self._session_ending():
            self.quit = True
            return "the session is ending"
        snapshot = self._joystick.snapshot()
        if not snapshot.connected:
            self.quit = True
            return "the joystick disconnected"
        if button(snapshot, self._mapping.quit_button):
            self.quit = True
            return "you pressed Back"
        return None

    def stop_reason(self) -> str | None:
        ending = self.ending()
        if ending is not None:
            return ending
        return "you pressed X or B" if any(self.pressed()) else None


def _end_stream(robot: BambooRobot) -> str | None:
    """Stop the stream; a stream Bamboo already ended reports why instead."""

    try:
        robot.stop_stream()
    except RuntimeError as error:
        return str(error)
    return None


def _guarded_move(
    robot: BambooRobot,
    args: argparse.Namespace,
    target: np.ndarray,
    interrupted: Callable[[], bool],
    why_interrupted: Callable[[], str],
) -> str | None:
    """Stream to a joint pose at homing speed; None once there, else why it stopped on the way."""

    try:
        stream_home(
            robot,
            target,
            min(args.home_speed, args.max_joint_speed),
            args.home_timeout,
            args.stream_rate,
            stop_requested=interrupted,
        )
    except InterruptedError:
        return why_interrupted()
    except RuntimeError as error:
        # It timed out, or Bamboo ended the stream (its contact guard).
        return str(error)
    return None


def _wait_for_go(
    robot: BambooRobot,
    args: argparse.Namespace,
    controls: ReplayControls,
    gel_guard: Any | None,
) -> str | None:
    """Hold the start pose until X; None to replay, else why not."""

    print("At the start pose: press X to replay, B to skip it.", flush=True)
    controls.mark()
    period = 1.0 / args.stream_rate
    while True:
        started = time.monotonic()
        ending = controls.ending()
        if ending is not None:
            return ending
        go, skip = controls.pressed()
        if go:
            controls.mark()
            return None
        if skip:
            return "you skipped it with B"
        reason = None if gel_guard is None else gel_guard.check()
        if reason is not None:
            return reason
        robot.stream_velocity(np.zeros(7))  # keeps the stream's watchdog fed
        remaining = period - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def replay_from_guide(
    robot: BambooRobot,
    args: argparse.Namespace,
    demo: Path,
    collector: Any,
    heartbeat: GuideHeartbeat | None,
    new_heartbeat: Callable[[], GuideHeartbeat],
    controls: ReplayControls,
    stop_requested: Callable[[], bool],
    gel_guard: Any | None = None,
    rumbler: Any | None = None,
) -> GuideHeartbeat | None:
    """Replay the demonstration just recorded, record that replay, home, and float the arm again.

    Guide mode stops and one stream, under [contact], takes the arm to the
    demonstration's start, replays it while the collector records a new
    episode linked to the demonstration, and homes. The replay stops early
    when the operator asks, when the gel guard objects, when the arm is pushed
    off the path, or when Bamboo's contact guard ends the stream; a stopped
    replay is kept but marked, and is not followed by homing: guide mode
    comes back where the arm is, so a caught blade can be freed by hand.
    Returns the new heartbeat, or None when the session is ending.
    """

    try:
        path = ReplayPath.from_episode(demo)
    except (OSError, KeyError, ValueError) as error:
        print(f"\nNot replaying {demo.name}: {error}.")
        return heartbeat
    if heartbeat is not None:
        heartbeat.stop()
    robot.stop_guide()
    print(
        f"\nReplaying {demo.name} ({path.duration:.1f} s): hands off the arm. X or B stops the replay, Back quits.",
        flush=True,
    )
    time.sleep(GUIDE_HOME_PAUSE_S)

    def interrupted() -> bool:
        return controls.stop_reason() is not None or (gel_guard is not None and gel_guard.check() is not None)

    def why_interrupted() -> str:
        reason = controls.stop_reason() or (None if gel_guard is None else gel_guard.check())
        return reason or "it was interrupted"

    homed = False
    stopped = True
    try:
        robot.apply_contact_limits(args.contact_limits)
        robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
        print("Moving to the demonstration's start...")
        reason = _guarded_move(robot, args, path.start, interrupted, why_interrupted)
        if reason is None and args.replay_wait:
            reason = _wait_for_go(robot, args, controls, gel_guard)
        if reason is not None:
            print(f"Replay skipped: {reason}.")
        else:
            stopped = not _record_replay(robot, args, path, demo, collector, controls, gel_guard, rumbler)
            if not stopped:
                print("Homing...")
                reason = _guarded_move(robot, args, HOME_JOINTS, interrupted, why_interrupted)
                homed = reason is None
                if not homed:
                    print(f"Homing stopped: {reason}.")
    finally:
        ended = _end_stream(robot)
        if ended is not None and not stopped:
            LOG.warning("The replay stream ended with: %s", ended)
        robot.clear_contact_limits()
    if stop_requested() or controls.quit:
        return None
    heartbeat = restart_guide(
        robot,
        args,
        new_heartbeat,
        lambda: stop_requested() or controls.quit,
        "Guide mode starts once nothing pushes on the arm: free the blade if it is caught, and keep hands off.",
        REPLAY_RESTART_WAIT_S,
    )
    print(
        ("Homing complete; " if homed else "")
        + "guide mode is back on: the arm floats again. Press X to record the next demonstration."
    )
    return heartbeat


def _record_replay(
    robot: BambooRobot,
    args: argparse.Namespace,
    path: ReplayPath,
    demo: Path,
    collector: Any,
    controls: ReplayControls,
    gel_guard: Any | None,
    rumbler: Any | None,
) -> bool:
    """Replay from the start pose while recording; True when it ran to the end."""

    replay = collector.start_episode(action_source=REPLAY_ACTION_SOURCE, extra_metadata={"replay_of": demo.name})
    if rumbler is not None:
        rumbler.recording_started()
    print(f"⬆️ Recording the replay: {replay}")

    def stop_reason() -> str | None:
        collector.check_health()  # a failed recorder ends the replay with its own error
        return controls.stop_reason()

    outcome = None
    try:
        outcome = run_replay(
            robot,
            path,
            rate_hz=args.stream_rate,
            max_speed=args.max_joint_speed,
            max_path_error=args.replay_max_path_error,
            gel_guard=gel_guard,
            stop_reason=stop_reason,
            on_command=collector.set_action,
        )
    finally:
        collector.set_action(np.zeros(7))
        try:
            replay = collector.stop_episode()
        except Exception as error:  # the replay's own failure, if any, matters more
            if outcome is not None:
                raise
            LOG.error("Could not finish the replay's recording: %s", error)
    link_replay(
        demo,
        replay,
        outcome,
        {
            "path_gain": PATH_GAIN,
            "max_path_error_limit_rad": args.replay_max_path_error,
            "max_joint_speed": args.max_joint_speed,
            "contact_limits": args.contact_limits,
        },
    )
    if outcome.completed:
        if rumbler is not None:
            rumbler.recording_stopped()
        print(
            f"✅ Replay recorded: {replay} (it kept within {outcome.max_path_error_rad:.3f} rad of the "
            "demonstrated path)"
        )
        return True
    if rumbler is not None:
        rumbler.error()
    print(f"⛔ Replay stopped after {outcome.duration_s:.1f} s: {outcome.reason}.")
    print(f"   {replay.name} is kept, marked as stopped (replay.completed = false in its metadata).")
    if not outcome.by_operator:
        print("   Guide mode comes back where the arm is: free the blade by hand, then Menu homes.")
    return False


def _pose_error(model: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(model[:3, 3] - measured[:3, 3]))
    difference = model[:3, :3].T @ measured[:3, :3]
    angle_error = float(np.arccos(np.clip((np.trace(difference) - 1.0) / 2.0, -1.0, 1.0)))
    return position_error, angle_error


# The scale readings the operator takes around each pour. Weighing the bottle
# as well as the cup separates what was poured from what was spilled.
WEIGHTS_BEFORE = (
    ("bottle_before_g", "bottle before the pour"),
    ("cup_before_g", "cup before the pour"),
)
WEIGHTS_AFTER = (
    ("bottle_after_g", "bottle after the pour, back in place"),
    ("cup_after_g", "cup after the pour"),
)


def _store_weights(episode: Path, readings: dict[str, float]) -> None:
    if not readings:
        return
    from fr3_demo.recording import record_episode_weights

    weights = record_episode_weights(episode, readings)["weights"]
    summary = "  ".join(
        f"{key}={weights[key]:.1f}" for key in ("poured_g", "spilled_g") if key in weights
    )
    print(f"  Weights saved to {episode.name}/metadata.json" + (f": {summary}" if summary else ""))


def prompt_weights(
    fields: tuple[tuple[str, str], ...],
    *,
    pause_stream: Callable[[], None] | None = None,
    resume_stream: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Ask for scale readings in grams; blank skips one, Ctrl-C skips the rest.

    The teleoperation stream has a watchdog measured in milliseconds, so the
    caller pauses it before this blocks on the terminal and resumes afterwards;
    otherwise the watchdog brakes the arm while the operator is at the scale.
    """

    if not sys.stdin.isatty():
        return {}
    if pause_stream is not None:
        pause_stream()
    readings: dict[str, float] = {}
    try:
        for key, label in fields:
            for attempt in range(2):
                answer = input(f"  {label} in grams (blank to skip): ").strip()
                if not answer:
                    break
                try:
                    readings[key] = float(answer)
                except ValueError:
                    print("    Enter a number, or leave blank to skip." if attempt == 0 else "    Skipped.")
                    continue
                break
    except (EOFError, KeyboardInterrupt):
        print("\n  Skipped the remaining weights.")
    finally:
        if resume_stream is not None:
            resume_stream()
    return readings


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
    parser.add_argument("--gripper-force", type=float, default=0.8, help="normalized closing force from 0.0 to 1.0")
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
    parser.add_argument(
        "--hand-guide",
        action="store_true",
        help=(
            "record while you move the arm with the robot's own guiding buttons; "
            "no arm command is sent, so there is no gel protection (requires --collect)"
        ),
    )
    parser.add_argument(
        "--pilot-guide",
        action="store_true",
        help="with --hand-guide, use the robot's guiding buttons even when Bamboo can float the arm itself",
    )
    parser.add_argument("--guide-watchdog-ms", type=int, default=250, help="hold the arm if guide updates stop this long")
    parser.add_argument(
        "--guide-max-joint-velocity",
        type=float,
        default=0.35,
        help="joint speed in rad/s above which guiding resists; keep it at the speed a policy can replay",
    )
    parser.add_argument(
        "--guide-idle-hold-s",
        type=float,
        default=None,
        help="hold the arm after this many seconds still, until pushed (0 disables; default: Bamboo's 1 s)",
    )
    parser.add_argument(
        "--guide-max-start-force",
        type=float,
        default=None,
        help="refuse to start guiding above this external force estimate in N (default: Bamboo's 5 N)",
    )
    parser.add_argument("--guide-heartbeat-hz", type=float, default=30.0, help="guide update rate")
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "with --hand-guide: stopping a recording with X replays it on the arm right away (from its start pose, "
            "under [contact] and the gel guard), records the replay as its own episode, then homes"
        ),
    )
    parser.add_argument(
        "--replay-wait",
        action="store_true",
        help="with --replay, wait at the demonstration's start pose for X before replaying (B skips)",
    )
    parser.add_argument(
        "--replay-max-path-error",
        type=float,
        # A replay lags a fast demo by up to ~0.03 rad (the stream's acceleration limit, measured on real demos
        # in simulation), and contact at the push cap adds ~0.01 rad.
        default=0.08,
        help="stop a replay once the arm is pushed this far off the demonstrated path, in rad (default: %(default)s)",
    )
    parser.add_argument(
        "--payload-mass-kg",
        type=float,
        default=None,
        help="payload beyond Desk's end effector (gripper, sensor, tool), set at session start",
    )
    parser.add_argument(
        "--payload-com-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="payload center of mass in the flange frame, in m",
    )
    parser.add_argument(
        "--no-contact-limits",
        action="store_true",
        help="leave the config's [contact] limits off for this session",
    )
    parser.add_argument(
        "--no-gel-guard",
        action="store_true",
        help="leave the config's [gel] GelSight guard off for this session",
    )
    # [contact] and [gel] in the config fill these in (see settings).
    parser.set_defaults(contact_limits=None, gel=None)
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
    parser.add_argument(
        "--external2-camera-serial",
        default=os.environ.get("FR3_EXTERNAL2_CAMERA_SERIAL"),
        help="optional RealSense serial for exterior_image_2_left",
    )
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
        help="rotate exterior_image_left by 180 degrees",
    )
    parser.add_argument(
        "--wrist-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rotate wrist_image by 180 degrees",
    )
    parser.add_argument(
        "--external2-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rotate exterior_image_2_left by 180 degrees",
    )
    parser.add_argument("--record-fps", type=float, default=15.0, help="synchronized dataset sampling rate")
    parser.add_argument(
        "--record-image-width",
        type=int,
        default=320,
        help="width of stored JPEGs; 0 records at the native camera resolution",
    )
    parser.add_argument(
        "--record-image-height",
        type=int,
        default=180,
        help="height of stored JPEGs; 0 records at the native camera resolution",
    )
    parser.add_argument(
        "--weigh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after each episode, ask for the weight read off the scale; task-specific, off by default",
    )
    parser.add_argument(
        "--record-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record joint torque and the full-rate raw_force stream (default: enabled)",
    )
    parser.add_argument(
        "--record-tactile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record the GelSight tactile stream; the sensor must be present (default: disabled)",
    )
    parser.add_argument("--tactile-device", default="", help="tactile index or /dev/videoN; empty auto-detects")
    parser.add_argument("--tactile-width", type=int, default=320, help="stored tactile frame width")
    parser.add_argument("--tactile-height", type=int, default=240, help="stored tactile frame height")
    parser.add_argument(
        "--tactile-fps",
        type=float,
        default=25.0,
        help="tactile capture rate; the GelSight Mini caps at 25 fps",
    )
    parser.add_argument(
        "--tactile-max-age-ms",
        type=float,
        default=60.0,
        help="reject a grid row whose nearest tactile frame is older than this",
    )
    parser.add_argument(
        "--tactile-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rotate the stored tactile image by 180 degrees",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="refuse to record with less free disk than this",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live camera views in RViz while collecting (default: enabled)",
    )
    parser.add_argument(
        "--preview-publish-only",
        action="store_true",
        help="publish the preview topics without launching RViz on this machine",
    )
    parser.add_argument("--preview-rate-hz", type=float, default=10.0, help="RViz preview publish rate")
    parser.add_argument("--alignment-delay-ms", type=float, default=80.0)
    parser.add_argument("--camera-max-delta-ms", type=float, default=25.0)
    parser.add_argument("--camera-pair-warn-ms", type=float, default=20.0)
    parser.add_argument("--camera-pair-reject-ms", type=float, default=35.0)
    parser.add_argument("--robot-sample-hz", type=float, default=60.0)
    parser.add_argument("--robot-max-gap-ms", type=float, default=30.0)
    parser.add_argument("--gripper-sample-hz", type=float, default=30.0)
    parser.add_argument("--gripper-max-age-ms", type=float, default=2000.0)
    parser.add_argument("--image-workers", type=int, default=3)
    parser.add_argument("--image-queue-size", type=int, default=120)
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
        "camera_width",
        "camera_height",
        "external2_camera_fps",
        "external2_camera_width",
        "external2_camera_height",
        "record_fps",
        "alignment_delay_ms",
        "camera_max_delta_ms",
        "camera_pair_warn_ms",
        "camera_pair_reject_ms",
        "robot_sample_hz",
        "robot_max_gap_ms",
        "gripper_sample_hz",
        "gripper_max_age_ms",
        "image_workers",
        "image_queue_size",
        "preview_rate_hz",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 50 <= args.watchdog_ms <= 500:
        parser.error("--watchdog-ms must be between 50 and 500")
    if not 0.0 <= args.deadzone < 0.9:
        parser.error("--deadzone must be in [0, 0.9)")
    if not 0.0 <= args.gripper_force <= 1.0:
        parser.error("--gripper-force must be in [0.0, 1.0]")
    if (args.record_image_width > 0) != (args.record_image_height > 0):
        parser.error("--record-image-width and --record-image-height must both be set, or both be 0 for native")
    if args.record_image_width < 0 or args.record_image_height < 0:
        parser.error("--record-image-width and --record-image-height cannot be negative")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb cannot be negative")
    if args.camera_pair_warn_ms >= args.camera_pair_reject_ms:
        parser.error("--camera-pair-warn-ms must be below --camera-pair-reject-ms")
    if np.any(np.asarray(args.workspace_min) >= np.asarray(args.workspace_max)):
        parser.error("each --workspace-min value must be below --workspace-max")
    if args.collect and (args.offline or args.dry_run or args.check or args.legacy_waypoints):
        parser.error(
            "--collect requires live streaming mode (not --offline, --dry-run, --check, or --legacy-waypoints)"
        )
    if args.hand_guide and not args.collect:
        parser.error("--hand-guide records demonstrations, so it requires --collect (use fr3-collect --hand-guide)")
    replay_tuned = args.replay_wait or args.replay_max_path_error != parser.get_default("replay_max_path_error")
    if replay_tuned and not args.replay:
        parser.error("--replay-wait and --replay-max-path-error only apply with --replay")
    if args.replay:
        if not args.hand_guide:
            parser.error("--replay replays hand-guided demonstrations, so it requires --hand-guide")
        if args.contact_limits is None or args.no_contact_limits:
            parser.error(
                "--replay needs the contact limits: set enabled = true under [contact] in the config "
                "(and measure contact_point_m to where the tool meets the fruit)"
            )
        # A replay presses until the blade is where the demonstration had it; the push cap is what bounds that.
        if args.contact_limits.get("shadow") or not args.contact_limits.get("max_force_n", 0.0) > 0.0:
            parser.error("--replay needs the push cap applied: set max_force_n and shadow = false under [contact]")
        if (args.record_tactile or args.gel is not None) and (args.gel is None or args.no_gel_guard):
            parser.error(
                "--replay needs the gel guard while the GelSight is in use: "
                "set enabled = true under [gel] in the config"
            )
        if not 0.0 < args.replay_max_path_error <= 0.2:
            parser.error("--replay-max-path-error must be in (0, 0.2] rad")
    if not 50 <= args.guide_watchdog_ms <= 500:
        parser.error("--guide-watchdog-ms must be between 50 and 500")
    if args.guide_max_joint_velocity <= 0:
        parser.error("--guide-max-joint-velocity must be positive")
    if args.guide_heartbeat_hz <= 0 or 1000.0 / args.guide_heartbeat_hz > args.guide_watchdog_ms / 3.0:
        parser.error("--guide-heartbeat-hz must send at least three updates per --guide-watchdog-ms period")
    if (args.payload_mass_kg is None) != (args.payload_com_m is None):
        parser.error("--payload-mass-kg and --payload-com-m must be given together")
    if args.payload_mass_kg is not None and not 0.0 <= args.payload_mass_kg <= 3.0:
        parser.error("--payload-mass-kg must be between 0 and 3")
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
    preview: Any | None = None
    tactile: Any | None = None
    rumbler: Any | None = None
    heartbeat: Any | None = None
    streaming_active = False
    guiding_active = False
    contact_limits_set = False
    gel_camera: Any | None = None
    gel: Any | None = None
    gel_resume_pending = False
    gel_warned_level = 0.0
    gel_yielded = False
    stop = False

    def request_stop(*_: object) -> None:
        nonlocal stop
        if stop:
            # Asked twice: get out now, even mid-shutdown, without Python's
            # teardown (half-closed cameras can abort it). Bamboo's watchdog
            # holds or brakes the arm without us.
            print("\nForce quit.", flush=True)
            os._exit(130)
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
                args.gripper_force,
            )
            if args.payload_mass_kg is not None and not (args.dry_run or args.check):
                robot.set_load(args.payload_mass_kg, args.payload_com_m)
                print(f"Payload set: {args.payload_mass_kg:.3f} kg at {list(args.payload_com_m)} m (flange frame).")
            if not (args.dry_run or args.check):
                # Guide mode is not a stream: [contact] does not apply to it.
                wanted = None if (args.no_contact_limits or args.hand_guide) else args.contact_limits
                applied = robot.apply_contact_limits(wanted)
                if applied is not None:
                    contact_limits_set = True
                    print(describe_contact_limits(applied))

        if args.collect:
            from fr3_demo.cameras import RealSensePair
            from fr3_demo.feedback import open_rumbler
            from fr3_demo.recording import DemoCollector

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
            if cameras.optional_camera_error:
                print(f"Optional exterior camera unavailable; continuing without it: {cameras.optional_camera_error}")
            if args.record_tactile:
                from fr3_demo.tactile import GelSightCamera

                tactile = GelSightCamera(
                    device=args.tactile_device or None,
                    width=args.tactile_width,
                    height=args.tactile_height,
                    fps=args.tactile_fps,
                    rotate_180=args.tactile_rotate_180,
                )
                # Deliberately not guarded: record_tactile means the tactile
                # stream is part of the data, so a missing sensor must refuse
                # the session rather than silently collect without it.
                tactile.start()
                print(
                    f"Tactile sensor /dev/video{tactile.device_index} streaming at up to "
                    f"{tactile.fps:.0f} Hz, stored {tactile.width}x{tactile.height}."
                )
            collector = DemoCollector(
                cameras,
                args.data_dir,
                args.server_ip,
                args.control_port,
                fps=args.record_fps,
                gripper_port=args.gripper_port,
                gripper_type=args.gripper_type,
                enable_gripper=not args.no_gripper,
                alignment_delay_ms=args.alignment_delay_ms,
                camera_max_delta_ms=args.camera_max_delta_ms,
                camera_pair_warn_ms=args.camera_pair_warn_ms,
                camera_pair_reject_ms=args.camera_pair_reject_ms,
                robot_sample_hz=args.robot_sample_hz,
                robot_max_gap_ms=args.robot_max_gap_ms,
                gripper_sample_hz=args.gripper_sample_hz,
                gripper_max_age_ms=args.gripper_max_age_ms,
                image_workers=args.image_workers,
                image_queue_size=args.image_queue_size,
                image_size=(
                    (args.record_image_width, args.record_image_height)
                    if args.record_image_width and args.record_image_height
                    else None
                ),
                min_free_gb=args.min_free_gb,
                record_force=args.record_force,
                tactile=tactile,
                tactile_max_age_ms=args.tactile_max_age_ms,
                action_source=HAND_GUIDE_ACTION_SOURCE if args.hand_guide else None,
            )
            collector.start()
            if args.preview:
                try:
                    from fr3_demo.preview import CameraPreview

                    preview = CameraPreview(
                        cameras,
                        rate_hz=args.preview_rate_hz,
                        launch_viewer=not args.preview_publish_only,
                    ).start()
                    # While recording, the preview shows the recorder's own
                    # frames instead of competing with it for the cameras.
                    collector.set_frame_observer(preview.submit)
                    print("Camera preview publishing on /fr3_demo/* for RViz.")
                except Exception as error:  # noqa: BLE001 - a preview is never worth losing a session over
                    LOG.warning("Camera preview disabled: %s", error)
            if robot is not None and not args.no_gripper:
                initial_gripper = robot.gripper_position()
                collector.set_gripper_observation(initial_gripper)
                collector.set_gripper_action(initial_gripper)
            rumbler = open_rumbler(args.feedback_event)
            print(f"Raw demonstration session: {collector.session_dir}")

        if args.gel is not None and not args.no_gel_guard and robot is not None and not (args.dry_run or args.check):
            from fr3_demo.gel import GelMonitor, describe_gel_guard, gel_guard_settings

            gel_limits, gel_directions = gel_guard_settings(args.gel)
            camera = tactile
            if camera is None:
                from fr3_demo.tactile import GelSightCamera

                gel_camera = GelSightCamera(
                    device=args.tactile_device or None,
                    width=args.tactile_width,
                    height=args.tactile_height,
                    fps=args.tactile_fps,
                    rotate_180=args.tactile_rotate_180,
                )
                gel_camera.start()
                camera = gel_camera
            gel = GelMonitor(gel_limits, gel_directions)
            camera.add_frame_observer(gel.on_frame)
            print("Zeroing the GelSight: keep the tool still and out of contact...")
            gel.wait_zeroed(10.0)
            print(describe_gel_guard(gel_limits, gel_directions))

        with LinuxJoystick(args.joystick) as joystick:
            q = _diagnostics(joystick, robot, nominal_q)
            if args.check:
                print("Check complete: no motion command was sent.")
                return 0

            streaming = not args.dry_run and not args.legacy_waypoints and not args.hand_guide
            if args.hand_guide and robot is not None:
                if choose_guide_mode(robot, args.pilot_guide) == "impedance":
                    robot.start_guide(
                        args.guide_watchdog_ms,
                        args.guide_max_joint_velocity,
                        args.guide_idle_hold_s,
                        args.guide_max_start_force,
                    )
                    guiding_active = True
                    heartbeat = GuideHeartbeat(robot.new_arm_client, args.guide_heartbeat_hz, gel=gel)
                    heartbeat.start()
                    gel_note = (
                        "As the gel's load nears its limit the arm stiffens against loading it further; "
                        "backing off stays free."
                        if gel is not None
                        else "Nothing here limits the force on the gel ([gel] is off)."
                    )
                    print(
                        "Hand-guide mode: the arm floats, so grab it and move it; no guiding buttons needed. "
                        f"It will not go faster than {args.guide_max_joint_velocity:.2f} rad/s per joint, holds "
                        "when left still and breaks free when pushed. Pressing the user stop stops the arm and "
                        f"ends guide mode. {gel_note}"
                    )
                else:
                    print(
                        "Hand-guide mode (Pilot buttons): no arm command will be sent. Move the arm with the "
                        "robot's guiding buttons; nothing here limits the force on the gel."
                    )
            elif streaming and robot is not None:
                robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
                streaming_active = True
                print(f"Streaming control active at {args.stream_rate:.0f} Hz (watchdog: {args.watchdog_ms} ms).")
            elif args.legacy_waypoints and not args.dry_run:
                LOG.warning("Legacy waypoint mode intentionally stops after every command and will feel laggy")
            if args.replay:
                if not guiding_active:
                    raise RuntimeError("--replay needs Bamboo's guide mode; with the Pilot buttons it cannot replay")
                print(
                    "Replay: stopping a recording with X replays it on the arm from its start pose, under the "
                    "contact limits" + (" and the gel guard" if gel is not None else "") + ", records the replay "
                    "as its own episode, and homes. X or B stops a replay, Back quits."
                    + (" The arm waits at the start pose for X." if args.replay_wait else "")
                )

            def pause_stream() -> None:
                """Idle the arm before anything blocks for operator input."""

                nonlocal streaming_active
                if streaming_active and robot is not None:
                    robot.stream_velocity(np.zeros(7))
                    robot.stop_stream()
                    streaming_active = False

            def resume_stream() -> None:
                nonlocal streaming_active
                if streaming and robot is not None and not streaming_active:
                    robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
                    streaming_active = True

            if args.hand_guide:
                controls = "A: close gripper | B: open gripper | Back: quit"
                if guiding_active:
                    controls += " | Menu: home | RB: resume after a hold"
            else:
                print("\nTeleoperation is active; moving a control commands the robot immediately.")
                print(
                    "Left stick: X/Y | LT/LB: selected-frame Z down/up | "
                    "D-pad up/down: tool Z +/- | D-pad left/right: tool Y +/- | "
                    "right stick: roll/pitch | RT/RB: yaw"
                )
                controls = "A: close gripper | B: open gripper | Menu: home | Back: quit"
            if collector is not None:
                controls += " | X: start/stop recording" + (" (stopping replays it)" if args.replay else "")
            print(controls)
            print("Keep a hand on the physical E-stop.\n")

            pending_weights: dict[str, float] = {}
            initial_snapshot = joystick.snapshot()
            last_close_count = initial_snapshot.press_counts[mapping.close_button]
            last_open_count = initial_snapshot.press_counts[mapping.open_button]
            last_home_count = initial_snapshot.press_counts[mapping.home_button]
            last_record_count = initial_snapshot.press_counts[mapping.record_button]
            last_resume_count = initial_snapshot.press_counts[mapping.yaw_up_button]
            last_hold_message: str | None = None
            last_status = 0.0
            idle_sleep = min(0.02, args.command_period / 4.0)
            stream_period = 1.0 / args.stream_rate
            dpad_axis_lock: str | None = None

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
                        print(f"\n✅ Recording stopped: {completed}")
                        if args.weigh:
                            print("Weigh the cup, and the bottle back in place:")
                            pending_weights.update(
                                prompt_weights(
                                    WEIGHTS_AFTER,
                                    pause_stream=pause_stream,
                                    resume_stream=resume_stream,
                                )
                            )
                            _store_weights(completed, pending_weights)
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
                        print(f"\n✅ Recording stopped: {completed}")
                        if args.weigh:
                            print("Weigh the cup, and the bottle back in place:")
                            pending_weights.update(
                                prompt_weights(
                                    WEIGHTS_AFTER,
                                    pause_stream=pause_stream,
                                    resume_stream=resume_stream,
                                )
                            )
                            _store_weights(completed, pending_weights)
                        pending_weights.clear()
                        if args.replay and guiding_active and robot is not None:
                            replay_controls = ReplayControls(joystick, mapping, lambda: stop)
                            guiding_active = False
                            heartbeat = replay_from_guide(
                                robot,
                                args,
                                completed,
                                collector,
                                heartbeat,
                                lambda: GuideHeartbeat(robot.new_arm_client, args.guide_heartbeat_hz, gel=gel),
                                replay_controls,
                                lambda: stop,
                                None if gel is None else GelStreamGuard(gel),
                                rumbler,
                            )
                            guiding_active = heartbeat is not None
                            last_hold_message = None
                            if replay_controls.quit and not stop:
                                request_stop()
                            # The presses that steered the replay are spent.
                            spent = joystick.snapshot()
                            last_close_count = spent.press_counts[mapping.close_button]
                            last_open_count = spent.press_counts[mapping.open_button]
                            last_home_count = spent.press_counts[mapping.home_button]
                            last_record_count = spent.press_counts[mapping.record_button]
                            last_resume_count = spent.press_counts[mapping.yaw_up_button]
                    else:
                        if args.weigh:
                            print("\nWeigh the bottle and the cup before this demonstration:")
                            pending_weights.clear()
                            pending_weights.update(
                                prompt_weights(
                                    WEIGHTS_BEFORE,
                                    pause_stream=pause_stream,
                                    resume_stream=resume_stream,
                                )
                            )
                        pending = collector.start_episode()
                        rumbler.recording_started()
                        print(f"\n⬆️ Recording started: {pending}")
                    continue

                if home_requested and args.hand_guide:
                    if not guiding_active or robot is None:
                        rumbler.error()
                        print("\nHoming here needs Bamboo's guide mode; with the Pilot buttons, home in teleop.")
                        continue
                    if collector is not None and collector.active:
                        collector.set_action(np.zeros(7))
                        rumbler.error()
                        print("\nStop recording with X before homing.")
                        continue
                    guiding_active = False
                    heartbeat = home_from_guide(
                        robot,
                        args,
                        heartbeat,
                        lambda: GuideHeartbeat(robot.new_arm_client, args.guide_heartbeat_hz, gel=gel),
                        lambda: stop,
                        None if gel is None else GelStreamGuard(gel),
                    )
                    guiding_active = heartbeat is not None
                    last_hold_message = None
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
                        collector.set_gripper_action(requested_gripper)
                    if args.dry_run:
                        print("DRY RUN: gripper close" if close_requested else "DRY RUN: gripper open")
                    elif robot is not None:
                        if gel is not None:
                            gel.suspend("the gripper is moving")
                        robot.close_gripper() if close_requested else robot.open_gripper()
                        if gel is not None:
                            # The grasp presses the gel: that is the new zero.
                            gel.rezero()
                            gel_resume_pending = guiding_active
                        if collector is not None:
                            collector.set_gripper_observation(robot.gripper_position())
                    continue

                if args.hand_guide and robot is not None:
                    if heartbeat is not None:
                        heartbeat.check_health()
                        resume_count = snapshot.press_counts[mapping.yaw_up_button]
                        if resume_count > last_resume_count:
                            robot.resume_guide()
                        last_resume_count = resume_count
                    state = hand_guide_tick(robot, collector)
                    q = np.asarray(state["qpos"], dtype=float)
                    if gel_resume_pending and gel is not None and gel.zeroed:
                        if state.get("guide_phase") == "hold" and state.get("guide_hold_reason") == "gel_invalid":
                            robot.resume_guide()
                            gel_resume_pending = False
                        elif state.get("guide_phase") == "free":
                            gel_resume_pending = False
                    if heartbeat is not None:
                        message = hold_message(state, None if gel is None else gel.latest().reason)
                        if gel_resume_pending and state.get("guide_hold_reason") == "gel_invalid":
                            message = "Holding while the GelSight re-zeroes after the gripper move."
                        if message != last_hold_message:
                            if message is not None:
                                rumbler.error()
                                print(f"\n{message}")
                            elif last_hold_message is not None:
                                print("\nArm free again.")
                            last_hold_message = message
                    remaining = stream_period - (time.monotonic() - loop_started)
                    if remaining > 0:
                        time.sleep(remaining)
                    continue

                twist = joystick_twist(snapshot, mapping, args.linear_speed, args.angular_speed, args.deadzone)
                tool_y = joystick_tool_y(snapshot, mapping, args.linear_speed, args.deadzone)
                tool_z = joystick_tool_z(snapshot, mapping, args.linear_speed, args.deadzone)
                tool_y, tool_z, dpad_axis_lock = lock_dpad_axis(tool_y, tool_z, dpad_axis_lock)
                rotation = forward_kinematics(q)[:3, :3]
                if args.frame == "tool":
                    twist = np.concatenate((rotation @ twist[:3], rotation @ twist[3:]))
                twist[:3] += rotation[:, 1] * tool_y
                twist[:3] += rotation[:, 2] * tool_z
                if gel is not None:
                    reading = gel.latest()
                    twist = limit_twist(twist, reading, rotation)
                    overloaded = reading.valid and reading.level >= GEL_TELEOP_STOP_LEVEL
                    if overloaded and streaming and robot is not None and not gel_yielded:
                        # Once: the next stream command then fails with this reason.
                        robot.yield_stream(f"GelSight load at level {reading.level:.2f}")
                        gel_yielded = True
                    if reading.valid and reading.level >= 0.5 > gel_warned_level:
                        if rumbler is not None:
                            rumbler.error()
                        print(f"\nGel load at {reading.level:.0%} of its limit: moving further into it is blocked.")
                    gel_warned_level = reading.level if reading.valid else 0.0

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
        print("\nShutting down (Ctrl-C again to force quit)...", flush=True)
        faulthandler.dump_traceback_later(SHUTDOWN_TIMEOUT_S, exit=True)
        atexit.register(_watch_interpreter_exit)
        if collector is not None:
            try:
                completed = collector.close()
                if completed is not None:
                    print(f"✅ Recording stopped during shutdown: {completed}")
            except Exception as error:  # noqa: BLE001 - shutdown must continue to camera and robot cleanup
                LOG.error("Could not finalize active recording: %s", error)
        if preview is not None:
            try:
                preview.close()
            except Exception as error:  # noqa: BLE001 - shutdown must continue to camera and robot cleanup
                LOG.warning("Could not stop the camera preview: %s", error)
        if tactile is not None:
            try:
                tactile.close()
            except Exception as error:  # noqa: BLE001 - shutdown must continue to camera and robot cleanup
                LOG.warning("Could not stop the tactile sensor: %s", error)
        if cameras is not None:
            cameras.close()
        if gel_camera is not None:
            try:
                gel_camera.close()
            except Exception as error:  # noqa: BLE001 - shutdown must continue to robot cleanup
                LOG.warning("Could not stop the GelSight: %s", error)
        if robot is not None:
            if streaming_active:
                try:
                    robot.stop_stream()
                except Exception as error:  # noqa: BLE001 - closing the client still triggers the watchdog
                    LOG.warning("Could not explicitly stop Bamboo stream: %s", error)
            if guiding_active:
                try:
                    robot.stop_guide()
                except Exception as error:  # noqa: BLE001 - without updates the watchdog holds the arm
                    # Guide mode that ended on its own has already said why.
                    if heartbeat is None or not heartbeat.ended_on_robot:
                        LOG.warning("Could not explicitly stop Bamboo guide mode: %s", error)
        if heartbeat is not None:
            heartbeat.stop()
        if robot is not None and contact_limits_set:
            try:
                robot.clear_contact_limits()
            except Exception as error:  # noqa: BLE001 - closing the client must still happen
                LOG.warning("Could not clear Bamboo contact limits: %s", error)
        if robot is not None:
            robot.close()
        faulthandler.cancel_dump_traceback_later()


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
