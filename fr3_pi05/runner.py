"""Run pi0.5 DROID inference and guarded execution on Bamboo/FR3."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from fr3_demo.cameras import RealSensePair
from fr3_demo.joystick import LinuxJoystick
from fr3_demo.kinematics import WorkspaceBounds, forward_kinematics
from fr3_demo.settings import default_config_path, load_config, pi05_defaults
from fr3_demo.teleop import HOME_JOINTS, BambooRobot, Mapping, stream_home
from fr3_pi05.policy import (
    DROID_CONTROL_HZ,
    WINE_HISTORY_OFFSETS,
    InferenceResult,
    InferenceWorker,
    ProprioHistory,
    SafetyViolation,
    build_droid_observation,
    predict_joint_path,
    safe_joint_velocity,
)

LOG = logging.getLogger("fr3_pi05")
WINE_ACTION_HORIZON = 16


class WineHistorySampler:
    """Sample Bamboo proprioception at the demonstration collector's rate."""

    def __init__(self, control_hz: float) -> None:
        self.history = ProprioHistory()
        self.period = 1.0 / control_hz
        self.next_sample = time.monotonic()

    def sample_due(self, robot: BambooRobot, gripper: GripperWorker) -> bool:
        now = time.monotonic()
        if now < self.next_sample:
            return False
        _, q = _state(robot)
        self.history.append(q, gripper.position)
        self.next_sample += self.period
        if self.next_sample < now - self.period:
            self.next_sample = now + self.period
        return True

    def append_control_sample(self, q: np.ndarray, gripper_position: float, now: float) -> None:
        self.history.append(q, gripper_position)
        self.next_sample += self.period
        if self.next_sample < now - self.period:
            self.next_sample = now + self.period

    def fill(self, robot: BambooRobot, gripper: GripperWorker, stopped: threading.Event) -> None:
        duration = max(WINE_HISTORY_OFFSETS) * self.period
        print(
            f"Collecting wine proprio history at {1.0 / self.period:.1f} Hz "
            f"for {duration:.1f} s ({self.history.required_samples} samples)..."
        )
        while not self.history.ready:
            if stopped.is_set():
                raise RuntimeError("Stopped while collecting initial wine proprio history")
            if self.sample_due(robot, gripper):
                continue
            stopped.wait(min(0.02, max(0.0, self.next_sample - time.monotonic())))
        print("Wine proprio history ready: joints=21, gripper=3, offsets=[0, 45, 75].")


class GripperWorker:
    """Own the blocking Bamboo gripper calls without starving arm streaming."""

    def __init__(self, robot: BambooRobot, enabled: bool, threshold: float = 0.9) -> None:
        self._robot = robot
        self._enabled = enabled
        self._threshold = threshold
        self._position = 0.0
        self._desired: float | None = None
        self._moving = False
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._commands: queue.Queue[float | None] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._loop, name="bamboo-gripper", daemon=True)
            self._thread.start()
        else:
            self._ready.set()

    def _loop(self) -> None:
        try:
            with self._lock:
                self._position = self._robot.gripper_position()
                self._desired = 1.0 if self._position > 0.5 else 0.0
        except Exception as error:  # noqa: BLE001 - preserve Bamboo client errors for the control thread
            self._error = error
        finally:
            self._ready.set()
        while self._error is None:
            command = self._commands.get()
            if command is None:
                return
            try:
                if command > 0.5:
                    self._robot.open_gripper()
                    observed = self._robot.gripper_position()
                else:
                    self._robot.close_gripper()
                    observed = self._robot.gripper_position()
                with self._lock:
                    self._position = observed
                    self._moving = False
            except Exception as error:  # noqa: BLE001 - preserve Bamboo client errors for the control thread
                self._error = error
                with self._lock:
                    self._moving = False
                return

    def wait_ready(self, timeout: float = 10.0) -> None:
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out reading the Bamboo gripper state")
        self.check()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Bamboo gripper failed: {self._error}") from self._error

    @property
    def position(self) -> float:
        self.check()
        with self._lock:
            return self._position

    @property
    def moving(self) -> bool:
        self.check()
        with self._lock:
            return self._moving

    def command(self, desired: float) -> bool:
        if not self._enabled:
            return False
        binary = 1.0 if desired > self._threshold else 0.0
        with self._lock:
            if binary == self._desired:
                return False
            self._desired = binary
            self._moving = True
        try:
            self._commands.put_nowait(binary)
        except queue.Full:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            self._commands.put_nowait(binary)
        return True

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._commands.put_nowait(None)
        except queue.Full:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            self._commands.put_nowait(None)
        self._thread.join(timeout=2.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Physical Intelligence pi0.5 DROID inference with two RealSense cameras and Bamboo FR3."
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--prompt", help="language instruction; prompted interactively when omitted")
    parser.add_argument("--transport", choices=("zmq", "websocket"), default="zmq")
    parser.add_argument(
        "--checkpoint",
        choices=("pi05_droid", "custom_droid", "wine_hybrid"),
        default="pi05_droid",
    )
    parser.add_argument("--policy-host", default="10.38.32.253")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument(
        "--zmq-mode",
        choices=("connect", "bind"),
        default="connect",
        help="connect to the GPU, or bind locally and let the GPU connect through a reverse route",
    )
    parser.add_argument("--zmq-bind-host", default="0.0.0.0")
    parser.add_argument("--server-ip", default="172.16.0.20", help="Bamboo controller host")
    parser.add_argument("--control-port", type=int, default=5555)
    parser.add_argument("--gripper-port", type=int, default=5559)
    parser.add_argument("--gripper-type", choices=("robotiq", "franka"), default="robotiq")
    parser.add_argument("--gripper-force", type=float, default=0.8, help="normalized closing force from 0.0 to 1.0")
    parser.add_argument("--no-gripper", action="store_true")
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
        help="rotate wrist images 180 degrees before RViz, recording, and policy inference",
    )
    parser.add_argument("--joystick", default="/dev/input/js0", help="Back button is the software abort")
    parser.add_argument(
        "--require-joystick",
        action="store_true",
        help="refuse execution when the optional Back-button abort joystick is unavailable",
    )
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--stream-hz", type=float, default=30.0)
    parser.add_argument("--open-loop-horizon", type=int, default=15)
    parser.add_argument("--prefetch-actions", type=int, default=4)
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.9,
        help="model gripper value must exceed this threshold to open; otherwise close",
    )
    parser.add_argument("--max-joint-speed", type=float, default=0.20)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.0)
    parser.add_argument("--home-speed", type=float, default=0.20)
    parser.add_argument("--home-timeout", type=float, default=15.0)
    parser.add_argument(
        "--home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="home before a rollout; use --no-home only when the robot is already deliberately positioned",
    )
    parser.add_argument("--joint-margin", type=float, default=0.10)
    parser.add_argument("--watchdog-ms", type=int, default=250)
    parser.add_argument("--max-camera-age", type=float, default=0.25)
    parser.add_argument("--max-inference-age", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="maximum policy steps before stopping; 0 runs until explicitly stopped",
    )
    parser.add_argument("--workspace-min", type=float, nargs=3, default=(0.10, -0.60, 0.05))
    parser.add_argument("--workspace-max", type=float, nargs=3, default=(0.80, 0.60, 1.00))
    parser.add_argument("--no-rviz", action="store_true", help="do not publish ROS topics or launch RViz")
    parser.add_argument("--rviz-publish-only", action="store_true", help="publish ROS topics without launching RViz")
    parser.add_argument("--check", action="store_true", help="perform one inference and safety preview; never move")
    parser.add_argument("--server-only", action="store_true", help="check policy transport/metadata without robot or cameras")
    parser.add_argument("--offline", action="store_true", help="validate configuration and math without hardware/network")
    parser.add_argument(
        "--debug-chunks",
        action="store_true",
        help="print every received action chunk and its thresholded gripper decisions",
    )
    parser.add_argument("--execute", action="store_true", help="allow policy actions to reach Bamboo")
    return parser


def _parse(argv: list[str] | None) -> argparse.Namespace:
    raw = sys.argv[1:] if argv is None else argv
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=default_config_path())
    bootstrap_args, _ = bootstrap.parse_known_args(raw)
    parser = _parser()
    try:
        config = load_config(bootstrap_args.config)
        defaults = pi05_defaults(config)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    environment = {
        "policy_host": os.environ.get("FR3_PI05_HOST"),
        "external_camera_serial": os.environ.get("FR3_EXTERNAL_CAMERA_SERIAL"),
        "wrist_camera_serial": os.environ.get("FR3_WRIST_CAMERA_SERIAL"),
        "external2_camera_serial": os.environ.get("FR3_EXTERNAL2_CAMERA_SERIAL"),
    }
    defaults.update({key: value for key, value in environment.items() if value})
    parser.set_defaults(config=bootstrap_args.config, **defaults)
    args = parser.parse_args(raw)
    if not any(token == "--policy-port" or token.startswith("--policy-port=") for token in raw):
        server_ports = config.get("pi05", {}).get("server_ports", {})
        if isinstance(server_ports, dict) and args.checkpoint in server_ports:
            args.policy_port = int(server_ports[args.checkpoint])
    if not args.external_camera_serial or not args.wrist_camera_serial:
        parser.error("both camera serial numbers are required in config.toml or on the command line")
    if args.checkpoint == "wine_hybrid":
        # These are checkpoint contract values, not rollout tuning parameters.
        args.open_loop_horizon = WINE_ACTION_HORIZON
        args.prefetch_actions = 0
    if args.execute and args.check:
        parser.error("--check never executes; remove either --check or --execute")
    if args.offline and not args.check:
        parser.error("--offline is only supported with --check")
    if args.control_hz <= 0 or args.stream_hz < args.control_hz:
        parser.error("stream-hz must be at least control-hz, and both must be positive")
    if args.checkpoint == "wine_hybrid" and not np.isclose(args.control_hz, DROID_CONTROL_HZ):
        parser.error(
            f"wine_hybrid requires --control-hz {DROID_CONTROL_HZ:g} to match its demonstration history"
        )
    if args.open_loop_horizon < 1 or not 0 <= args.prefetch_actions < args.open_loop_horizon:
        parser.error("prefetch-actions must be in [0, open-loop-horizon)")
    if args.max_steps < 0:
        parser.error("max-steps must be non-negative (0 disables the limit)")
    if args.camera_width < 1 or args.camera_height < 1 or args.camera_fps < 1:
        parser.error("camera width, height, and fps must be positive")
    if args.external2_camera_width < 1 or args.external2_camera_height < 1 or args.external2_camera_fps < 1:
        parser.error("optional camera width, height, and fps must be positive")
    if args.home_speed <= 0 or args.home_timeout <= 0:
        parser.error("home speed and timeout must be positive")
    if not 0.0 <= args.gripper_force <= 1.0:
        parser.error("--gripper-force must be in [0.0, 1.0]")
    if not 0.0 <= args.gripper_threshold <= 1.0:
        parser.error("--gripper-threshold must be in [0.0, 1.0]")
    if args.transport != "zmq" and args.zmq_mode != "connect":
        parser.error("--zmq-mode bind is only valid with --transport zmq")
    return args


def _policy_endpoint(args: argparse.Namespace) -> tuple[str, str]:
    if args.transport == "zmq" and args.zmq_mode == "bind":
        return args.zmq_bind_host, f"local tcp://{args.zmq_bind_host}:{args.policy_port} (GPU connects outward)"
    return args.policy_host, f"{args.policy_host}:{args.policy_port}"


def _state(robot: BambooRobot) -> tuple[dict[str, Any], np.ndarray]:
    state = robot.state()
    q = np.asarray(state.get("qpos"), dtype=float)
    dq = np.asarray(state.get("dq"), dtype=float)
    if q.shape != (7,) or dq.shape != (7,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
        raise RuntimeError("Bamboo returned an invalid FR3 joint state")
    return state, q


def _check_port(host: str, port: int, timeout: float = 5.0) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as error:
        raise RuntimeError(f"Cannot reach pi0.5 server at {host}:{port}: {error}") from error


def _build_camera_observation(
    frames: dict[str, Any],
    joint_position: np.ndarray,
    gripper_position: float | np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    exterior2 = frames.get("exterior_image_2_left")
    return build_droid_observation(
        frames["exterior_image_left"].image,
        frames["wrist_image"].image,
        joint_position,
        gripper_position,
        prompt,
        exterior2_image=None if exterior2 is None else exterior2.image,
    )


def _make_observation(
    cameras: RealSensePair,
    robot: BambooRobot,
    gripper: GripperWorker,
    prompt: str,
    max_camera_age: float,
    history: ProprioHistory | None = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any]]:
    frames = cameras.snapshot(max_camera_age)
    state, q = _state(robot)
    observation_joints: np.ndarray = q
    observation_gripper: float | np.ndarray = gripper.position
    if history is not None:
        observation_joints, observation_gripper = history.observation()
    observation = _build_camera_observation(frames, observation_joints, observation_gripper, prompt)
    return observation, frames, q, state


def _validate_policy_contract(checkpoint: str, metadata: dict[str, Any]) -> None:
    if checkpoint != "wine_hybrid":
        return
    expected = {
        "model": "pi05_wine_hybrid",
        "loader": "wine",
        "action_horizon": WINE_ACTION_HORIZON,
        "state_history_lags": [45, 75],
        "num_state_frames": 3,
        "joint_observation_dim": 21,
        "gripper_observation_dim": 3,
        "joint_observation_shape": [3, 7],
        "gripper_observation_shape": [3],
        "image_observation_shape": [180, 320, 3],
        "proprio_history_offsets": list(WINE_HISTORY_OFFSETS),
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "wine_hybrid server does not advertise the required history contract: "
            + "; ".join(mismatches)
            + ". Pull the latest fr3_demo on the GPU machine and restart the wine server."
        )
    action_expert_variant = metadata.get("action_expert_variant")
    if action_expert_variant not in {"gemma_300m_lora", "gemma_300m"}:
        raise RuntimeError(
            "wine_hybrid server returned an invalid action expert variant: "
            f"{action_expert_variant!r}"
        )
    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) and task.strip() for task in tasks):
        raise RuntimeError("wine_hybrid server returned an invalid task allowlist")
    asset_id = metadata.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise RuntimeError("wine_hybrid server did not advertise its normalization asset ID")


def _validate_prompt(prompt: str, metadata: dict[str, Any]) -> None:
    tasks = metadata.get("tasks") or []
    if tasks and prompt not in tasks:
        choices = " or ".join(repr(task) for task in tasks)
        raise RuntimeError(f"Prompt must exactly match {choices}; got {prompt!r}")


def _validate_exterior2_contract(metadata: dict[str, Any], observation: dict[str, Any]) -> None:
    """Require the optional camera only when the loaded checkpoint consumes it."""

    if metadata.get("uses_exterior2") is True and "observation/exterior_image_2_left" not in observation:
        raise RuntimeError(
            "The policy was launched with USE_EXTERNAL2=1, but exterior_image_2_left is unavailable. "
            "Connect/configure the L515 or restart the GPU server without USE_EXTERNAL2=1."
        )


def _offline_check(args: argparse.Namespace) -> int:
    workspace = WorkspaceBounds(tuple(args.workspace_min), tuple(args.workspace_max))
    q = np.array([-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031])
    zero = safe_joint_velocity(
        q,
        np.zeros(7),
        duration=1.0 / args.control_hz,
        maximum=args.max_joint_speed,
        joint_margin=args.joint_margin,
        workspace=workspace,
    )
    assert not np.any(zero)
    xyz = forward_kinematics(q)[:3, 3]
    print("Offline configuration check passed.")
    print(f"Policy server: {args.policy_host}:{args.policy_port}")
    print(f"Home-pose FK xyz: {np.array2string(xyz, precision=4)}")
    print("No camera, network, Bamboo, or robot command was used.")
    return 0


def _wait_result(
    worker: InferenceWorker,
    timeout: float = 60.0,
    *,
    history_sampler: WineHistorySampler | None = None,
    robot: BambooRobot | None = None,
    gripper: GripperWorker | None = None,
) -> InferenceResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        if history_sampler is not None:
            if robot is None or gripper is None:
                raise ValueError("Wine history sampling requires Bamboo and gripper handles")
            history_sampler.sample_due(robot, gripper)
        sleep_seconds = 0.02
        if history_sampler is not None:
            sleep_seconds = min(sleep_seconds, max(0.0, history_sampler.next_sample - time.monotonic()))
        time.sleep(sleep_seconds)
    raise RuntimeError(f"Timed out after {timeout:.0f}s waiting for pi0.5 inference")


def _announce_execution(args: argparse.Namespace) -> None:
    if not args.execute:
        return
    print("\nPOLICY EXECUTION WILL MOVE THE FR3.")
    print("Clear the workspace and keep a hand on the physical E-stop. Use Back when a joystick is connected.")


def _open_abort_joystick(path: str, required: bool) -> tuple[LinuxJoystick | None, int]:
    """Open the optional Back-button abort device and return its initial press count."""

    joystick: LinuxJoystick | None = None
    try:
        joystick = LinuxJoystick(path).open()
        snapshot = joystick.snapshot()
        if len(snapshot.buttons) <= Mapping().quit_button:
            raise RuntimeError("Joystick does not expose the configured Back abort button")
        return joystick, snapshot.press_counts[Mapping().quit_button]
    except RuntimeError as error:
        if joystick is not None:
            joystick.close()
        if required:
            raise
        LOG.warning("Joystick unavailable; Back abort disabled: %s", error)
        return None, 0


def _print_action_chunk(actions: np.ndarray, sequence: int, source: str, gripper_threshold: float) -> None:
    """Print one validated policy chunk with explicit gripper interpretation."""

    chunk = np.asarray(actions, dtype=float)
    gripper = chunk[:, 7]
    decisions = ["OPEN" if value > gripper_threshold else "CLOSE" for value in gripper]
    print(f"\nDEBUG ACTION CHUNK #{sequence} ({source}), shape={chunk.shape}")
    print("columns: dq0 dq1 dq2 dq3 dq4 dq5 dq6 gripper")
    print(np.array2string(chunk, precision=4, suppress_small=False, max_line_width=180))
    print(
        f"gripper decisions (>{gripper_threshold:g} OPEN, <={gripper_threshold:g} CLOSE): "
        + " ".join(f"{index}:{value:.4f}->{decision}" for index, (value, decision) in enumerate(zip(gripper, decisions)))
    )


def _home_robot(robot: BambooRobot, args: argparse.Namespace, stopped: threading.Event) -> np.ndarray:
    speed = min(args.home_speed, args.max_joint_speed)
    print("\nAUTOMATIC HOMING WILL MOVE THE FR3.")
    print(f"Homing to q={np.array2string(HOME_JOINTS, precision=3)} at up to {speed:.2f} rad/s...")
    robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
    try:
        result = stream_home(
            robot,
            HOME_JOINTS,
            speed,
            args.home_timeout,
            args.stream_hz,
            stop_requested=stopped.is_set,
        )
    finally:
        robot.stop_stream()
    print(f"Homing complete: q={np.array2string(result, precision=3)}")
    return result


def _open_gripper(robot: BambooRobot) -> None:
    print("\nOpening gripper for rollout startup...")
    robot.open_gripper()
    print("Gripper open.")


def run(args: argparse.Namespace) -> int:
    if args.offline:
        return _offline_check(args)
    if args.server_only:
        policy_host, endpoint_label = _policy_endpoint(args)
        if args.transport != "zmq" or args.zmq_mode == "connect":
            _check_port(args.policy_host, args.policy_port)
        if args.transport == "zmq":
            from fr3_pi05.protocol import OpenPiZmqClient

            client = OpenPiZmqClient(
                policy_host,
                args.policy_port,
                connection_mode=args.zmq_mode,
            )
        else:
            from fr3_pi05.protocol import OpenPiWebsocketClient

            client = OpenPiWebsocketClient(args.policy_host, args.policy_port)
        try:
            _validate_policy_contract(args.checkpoint, client.metadata)
            metadata = ", ".join(f"{key}={value}" for key, value in sorted(client.metadata.items())) or "none"
            print(
                f"pi0.5 {args.transport} server ready at {endpoint_label}; "
                f"metadata: {metadata}"
            )
        finally:
            client.close()
        print("No Bamboo or camera connection was opened; the robot did not move.")
        return 0
    prompt = args.prompt or input("Language instruction: ").strip()
    if not prompt:
        raise RuntimeError("Language instruction cannot be empty")
    workspace = WorkspaceBounds(tuple(args.workspace_min), tuple(args.workspace_max))
    policy_host, endpoint_label = _policy_endpoint(args)
    if args.transport != "zmq" or args.zmq_mode == "connect":
        _check_port(args.policy_host, args.policy_port)

    robot: BambooRobot | None = None
    cameras: RealSensePair | None = None
    gripper: GripperWorker | None = None
    policy: InferenceWorker | None = None
    rviz: Any | None = None
    joystick: LinuxJoystick | None = None
    history_sampler: WineHistorySampler | None = None
    stream_started = False
    stopped = threading.Event()

    def request_stop(*_: object) -> None:
        stopped.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        robot = BambooRobot(
            args.server_ip,
            args.control_port,
            args.gripper_port,
            args.gripper_type,
            not args.no_gripper,
            args.gripper_force,
        )
        state, q = _state(robot)
        print(f"Bamboo: q={np.array2string(q, precision=3)}; dq_norm={np.linalg.norm(state['dq']):.4f}")
        if not robot.supports_streaming() and (args.execute or (args.home and not args.check)):
            raise RuntimeError("The Bamboo controller does not support streaming execution")

        if args.home and not args.check:
            _home_robot(robot, args, stopped)
        if not args.check and not args.no_gripper:
            _open_gripper(robot)

        gripper = GripperWorker(robot, not args.no_gripper, args.gripper_threshold)
        gripper.wait_ready()
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
        print(
            f"Cameras ready: {cameras.serials}; modes={cameras.modes}; "
            f"wrist_rotate_180={args.wrist_rotate_180}"
        )
        if cameras.optional_camera_error:
            print(f"Optional exterior camera unavailable; continuing without it: {cameras.optional_camera_error}")
        policy = InferenceWorker(
            policy_host,
            args.policy_port,
            args.open_loop_horizon,
            args.transport,
            args.zmq_mode,
        )
        print(
            f"pi0.5 {args.transport} server connected at {endpoint_label} "
            f"(requested checkpoint: {args.checkpoint})"
        )
        if policy.metadata:
            print(f"Server metadata keys: {', '.join(sorted(policy.metadata))}")
        _validate_policy_contract(args.checkpoint, policy.metadata)
        _validate_prompt(prompt, policy.metadata)
        if not args.no_rviz:
            from fr3_pi05.visualization import RvizBridge

            rviz = RvizBridge(launch_rviz=not args.rviz_publish_only)

        if args.checkpoint == "wine_hybrid":
            history_sampler = WineHistorySampler(args.control_hz)
            history_sampler.fill(robot, gripper, stopped)
        observation, frames, q, _ = _make_observation(
            cameras,
            robot,
            gripper,
            prompt,
            args.max_camera_age,
            history_sampler.history if history_sampler is not None else None,
        )
        _validate_exterior2_contract(policy.metadata, observation)
        policy.submit(observation)
        first = _wait_result(
            policy,
            history_sampler=history_sampler,
            robot=robot,
            gripper=gripper,
        )
        age = first.completed_at - first.requested_at
        if age > args.max_inference_age:
            raise RuntimeError(f"Initial inference was stale ({age:.2f}s > {args.max_inference_age:.2f}s)")
        chunk_sequence = 1
        if args.debug_chunks:
            _print_action_chunk(first.actions, chunk_sequence, "initial", args.gripper_threshold)
        predicted = predict_joint_path(
            q,
            first.actions,
            horizon=args.open_loop_horizon,
            control_hz=args.control_hz,
            maximum=args.max_joint_speed,
            joint_margin=args.joint_margin,
            workspace=workspace,
        )
        if rviz is not None:
            rviz.publish(
                frames["exterior_image_left"].image,
                frames["wrist_image"].image,
                q,
                predicted,
                None
                if "exterior_image_2_left" not in frames
                else frames["exterior_image_2_left"].image,
            )
        print(
            f"Inference check: actions={first.actions.shape}, latency={age:.3f}s, "
            f"preview EEF={np.array2string(forward_kinematics(predicted[-1])[:3, 3], precision=4)}"
        )
        if args.check:
            if rviz is not None and not args.rviz_publish_only:
                preview_deadline = time.monotonic() + 3.0
                while time.monotonic() < preview_deadline:
                    rviz.publish(
                        frames["exterior_image_left"].image,
                        frames["wrist_image"].image,
                        q,
                        predicted,
                        None
                        if "exterior_image_2_left" not in frames
                        else frames["exterior_image_2_left"].image,
                    )
                    time.sleep(0.1)
            print("CHECK PASSED. Bamboo streaming was not started; the robot did not move.")
            return 0

        _announce_execution(args)
        if args.execute:
            joystick, initial_back_count = _open_abort_joystick(args.joystick, args.require_joystick)
            robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
            stream_started = True
            abort_controls = "Back, Ctrl+C, or the physical E-stop" if joystick is not None else "Ctrl+C or the physical E-stop"
            print(f"Bamboo policy streaming ARMED. Use {abort_controls} to stop.")
        else:
            initial_back_count = 0
            print("Inference-only rollout. Add --execute to allow robot motion.")

        action_chunk: np.ndarray | None = first.actions
        next_chunk: np.ndarray | None = None
        action_index = 0
        desired_velocity = np.zeros(7)
        waiting_for_wine_gripper = False
        policy_steps = 0
        stream_period = 1.0 / args.stream_hz
        control_period = 1.0 / args.control_hz
        next_stream = time.monotonic()
        next_control = history_sampler.next_sample if history_sampler is not None else next_stream

        unlimited_steps = args.max_steps == 0
        step_limit_label = "unlimited" if unlimited_steps else str(args.max_steps)
        while (unlimited_steps or policy_steps < args.max_steps) and not stopped.is_set():
            now = time.monotonic()
            if joystick is not None:
                snapshot = joystick.snapshot()
                if not snapshot.connected:
                    if args.require_joystick:
                        raise RuntimeError("Joystick disconnected; stopping policy execution")
                    LOG.warning("Joystick disconnected; Back abort disabled. Use Ctrl+C or the physical E-stop.")
                    joystick.close()
                    joystick = None
                    snapshot = None
                if snapshot is not None and snapshot.press_counts[Mapping().quit_button] != initial_back_count:
                    print("Back pressed; stopping.")
                    break

            result = policy.poll()
            if result is not None:
                inference_age = result.completed_at - result.requested_at
                if inference_age > args.max_inference_age:
                    raise RuntimeError(
                        f"Discarding stale inference ({inference_age:.2f}s > {args.max_inference_age:.2f}s)"
                    )
                chunk_sequence += 1
                if args.debug_chunks:
                    _print_action_chunk(result.actions, chunk_sequence, "replan", args.gripper_threshold)
                if action_chunk is None:
                    action_chunk = result.actions
                    action_index = 0
                elif next_chunk is None:
                    next_chunk = result.actions
                else:
                    raise RuntimeError("Received an unexpected extra policy chunk")

            if now >= next_control:
                _, q = _state(robot)
                gripper_position = gripper.position
                if history_sampler is not None:
                    history_sampler.append_control_sample(q, gripper_position, now)
                    observation_joints, observation_gripper = history_sampler.history.observation()
                else:
                    observation_joints, observation_gripper = q, gripper_position
                frames = cameras.snapshot(args.max_camera_age)
                if waiting_for_wine_gripper and not gripper.moving:
                    waiting_for_wine_gripper = False
                    print("Wine gripper motion complete; replanning from the measured state.")
                if waiting_for_wine_gripper:
                    desired_velocity = np.zeros(7)
                    predicted = None
                elif action_chunk is not None:
                    action = action_chunk[action_index]
                    gripper_value = float(action[7])
                    gripper_decision = "OPEN" if gripper_value > args.gripper_threshold else "CLOSE"
                    gripper_started = False
                    if args.execute:
                        gripper_started = gripper.command(gripper_value)
                    if args.debug_chunks:
                        command_status = (
                            "transition started"
                            if gripper_started
                            else "target unchanged"
                            if args.execute
                            else "not executed"
                        )
                        print(
                            f"DEBUG GRIPPER step={policy_steps} chunk_index={action_index} "
                            f"model={gripper_value:.4f} -> {gripper_decision}; "
                            f"observed={gripper_position:.4f}; {command_status}"
                        )
                    if args.checkpoint == "wine_hybrid" and gripper_started:
                        # Demonstrations pause the arm during blocking gripper motion. Keep the
                        # Bamboo stream alive with zeros, discard this stale chunk, then replan.
                        desired_velocity = np.zeros(7)
                        predicted = None
                        waiting_for_wine_gripper = True
                        action_chunk = None
                        next_chunk = None
                        action_index = 0
                        policy_steps += 1
                        print("Wine gripper transition: pausing arm motion until Bamboo completes it.")
                    else:
                        desired_velocity = safe_joint_velocity(
                            q,
                            action[:7],
                            duration=control_period,
                            maximum=args.max_joint_speed,
                            joint_margin=args.joint_margin,
                            workspace=workspace,
                        )
                        action_index += 1
                        policy_steps += 1
                        remaining = args.open_loop_horizon - action_index
                        predicted = predict_joint_path(
                            q,
                            action_chunk[action_index : args.open_loop_horizon],
                            horizon=max(0, remaining),
                            control_hz=args.control_hz,
                            maximum=args.max_joint_speed,
                            joint_margin=args.joint_margin,
                            workspace=workspace,
                        )
                        if (
                            0 < remaining <= args.prefetch_actions
                            and next_chunk is None
                            and not policy.busy
                        ):
                            observation = _build_camera_observation(
                                frames, observation_joints, observation_gripper, prompt
                            )
                            _validate_exterior2_contract(policy.metadata, observation)
                            policy.submit(observation)
                        if action_index >= args.open_loop_horizon:
                            action_chunk = next_chunk
                            next_chunk = None
                            action_index = 0
                else:
                    desired_velocity = np.zeros(7)
                    predicted = None
                    if not policy.busy:
                        observation = _build_camera_observation(
                            frames, observation_joints, observation_gripper, prompt
                        )
                        _validate_exterior2_contract(policy.metadata, observation)
                        policy.submit(observation)

                if rviz is not None:
                    rviz.publish(
                        frames["exterior_image_left"].image,
                        frames["wrist_image"].image,
                        q,
                        predicted,
                        None
                        if "exterior_image_2_left" not in frames
                        else frames["exterior_image_2_left"].image,
                    )
                if policy_steps and policy_steps % int(max(1, args.control_hz * 2)) == 0:
                    mode = "EXECUTING" if args.execute else "INFERENCE ONLY"
                    print(
                        f"{mode}: step {policy_steps}/{step_limit_label}, "
                        f"|dq_cmd|={np.linalg.norm(desired_velocity):.3f}"
                    )
                next_control += control_period
                if next_control < now - control_period:
                    next_control = now + control_period

            if args.execute and now >= next_stream:
                gripper.check()
                robot.stream_velocity(desired_velocity)
                next_stream += stream_period
                if next_stream < now - stream_period:
                    next_stream = now + stream_period

            sleep_until = min(next_stream if args.execute else next_control, next_control)
            remaining_sleep = sleep_until - time.monotonic()
            if remaining_sleep > 0:
                time.sleep(min(remaining_sleep, 0.01))

        return 0
    except SafetyViolation:
        LOG.exception("Policy safety stop")
        return 2
    finally:
        if stream_started and robot is not None:
            try:
                robot.stream_velocity(np.zeros(7))
                time.sleep(min(0.1, 2.0 / args.stream_hz))
                robot.stop_stream()
            except Exception as error:  # noqa: BLE001 - cleanup must continue even if Bamboo transport fails
                LOG.error("Could not cleanly stop Bamboo streaming: %s", error)
        if joystick is not None:
            joystick.close()
        if policy is not None:
            policy.close()
        if rviz is not None:
            rviz.close()
        if cameras is not None:
            cameras.close()
        if gripper is not None:
            gripper.close()
        if robot is not None:
            robot.close()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(_parse(argv))
    except (OSError, RuntimeError, ValueError) as error:
        LOG.error("%s", error)
        return 1


def check_main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    return main([*raw, "--check"])


if __name__ == "__main__":
    raise SystemExit(main())
