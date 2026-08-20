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
from fr3_demo.teleop import BambooRobot, Mapping
from fr3_pi05.policy import (
    InferenceResult,
    InferenceWorker,
    SafetyViolation,
    build_droid_observation,
    predict_joint_path,
    safe_joint_velocity,
)

LOG = logging.getLogger("fr3_pi05")


class GripperWorker:
    """Own the blocking Bamboo gripper calls without starving arm streaming."""

    def __init__(self, robot: BambooRobot, enabled: bool) -> None:
        self._robot = robot
        self._enabled = enabled
        self._position = 0.0
        self._desired: float | None = None
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
            except Exception as error:  # noqa: BLE001 - preserve Bamboo client errors for the control thread
                self._error = error
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

    def command(self, desired: float) -> None:
        if not self._enabled:
            return
        binary = 1.0 if desired > 0.5 else 0.0
        if binary == self._desired:
            return
        self._desired = binary
        try:
            self._commands.put_nowait(binary)
        except queue.Full:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            self._commands.put_nowait(binary)

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
    parser.add_argument("--checkpoint", choices=("pi05_droid", "custom_droid"), default="pi05_droid")
    parser.add_argument("--policy-host", default="10.38.32.253")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--server-ip", default="172.16.0.20", help="Bamboo controller host")
    parser.add_argument("--control-port", type=int, default=5555)
    parser.add_argument("--gripper-port", type=int, default=5559)
    parser.add_argument("--gripper-type", choices=("robotiq", "franka"), default="robotiq")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--external-camera-serial")
    parser.add_argument("--wrist-camera-serial")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--joystick", default="/dev/input/js0", help="Back button is the software abort")
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--stream-hz", type=float, default=30.0)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--prefetch-actions", type=int, default=4)
    parser.add_argument("--max-joint-speed", type=float, default=0.20)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.0)
    parser.add_argument("--joint-margin", type=float, default=0.10)
    parser.add_argument("--watchdog-ms", type=int, default=250)
    parser.add_argument("--max-camera-age", type=float, default=0.25)
    parser.add_argument("--max-inference-age", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--workspace-min", type=float, nargs=3, default=(0.10, -0.60, 0.05))
    parser.add_argument("--workspace-max", type=float, nargs=3, default=(0.80, 0.60, 1.00))
    parser.add_argument("--no-rviz", action="store_true", help="do not publish ROS topics or launch RViz")
    parser.add_argument("--rviz-publish-only", action="store_true", help="publish ROS topics without launching RViz")
    parser.add_argument("--check", action="store_true", help="perform one inference and safety preview; never move")
    parser.add_argument("--server-only", action="store_true", help="check policy transport/metadata without robot or cameras")
    parser.add_argument("--offline", action="store_true", help="validate configuration and math without hardware/network")
    parser.add_argument("--execute", action="store_true", help="allow policy actions to reach Bamboo")
    parser.add_argument("--yes", action="store_true", help="skip the typed EXECUTE confirmation (for supervised scripts)")
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
    if args.execute and args.check:
        parser.error("--check never executes; remove either --check or --execute")
    if args.offline and not args.check:
        parser.error("--offline is only supported with --check")
    if args.control_hz <= 0 or args.stream_hz < args.control_hz:
        parser.error("stream-hz must be at least control-hz, and both must be positive")
    if args.open_loop_horizon < 1 or not 0 <= args.prefetch_actions < args.open_loop_horizon:
        parser.error("prefetch-actions must be in [0, open-loop-horizon)")
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    return args


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


def _make_observation(
    cameras: RealSensePair,
    robot: BambooRobot,
    gripper: GripperWorker,
    prompt: str,
    max_camera_age: float,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any]]:
    frames = cameras.snapshot(max_camera_age)
    state, q = _state(robot)
    observation = build_droid_observation(
        frames["exterior_image_left"].image,
        frames["wrist_image"].image,
        q,
        gripper.position,
        prompt,
    )
    return observation, frames, q, state


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


def _wait_result(worker: InferenceWorker, timeout: float = 60.0) -> InferenceResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.02)
    raise RuntimeError(f"Timed out after {timeout:.0f}s waiting for pi0.5 inference")


def _confirm_execution(args: argparse.Namespace) -> None:
    if not args.execute or args.yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Refusing non-interactive execution without --yes")
    print("\nPOLICY EXECUTION WILL MOVE THE FR3.")
    print("Clear the workspace, keep a hand on the physical E-stop, and use Back to abort.")
    if input("Type EXECUTE to arm Bamboo streaming: ").strip() != "EXECUTE":
        raise RuntimeError("Execution cancelled")


def run(args: argparse.Namespace) -> int:
    if args.offline:
        return _offline_check(args)
    if args.server_only:
        _check_port(args.policy_host, args.policy_port)
        if args.transport == "zmq":
            from fr3_pi05.protocol import OpenPiZmqClient

            client = OpenPiZmqClient(args.policy_host, args.policy_port)
        else:
            from fr3_pi05.protocol import OpenPiWebsocketClient

            client = OpenPiWebsocketClient(args.policy_host, args.policy_port)
        try:
            metadata = ", ".join(f"{key}={value}" for key, value in sorted(client.metadata.items())) or "none"
            print(
                f"pi0.5 {args.transport} server ready at {args.policy_host}:{args.policy_port}; "
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
    _check_port(args.policy_host, args.policy_port)

    robot: BambooRobot | None = None
    cameras: RealSensePair | None = None
    gripper: GripperWorker | None = None
    policy: InferenceWorker | None = None
    rviz: Any | None = None
    joystick: LinuxJoystick | None = None
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
        )
        state, q = _state(robot)
        print(f"Bamboo: q={np.array2string(q, precision=3)}; dq_norm={np.linalg.norm(state['dq']):.4f}")
        if not robot.supports_streaming() and args.execute:
            raise RuntimeError("The Bamboo controller does not support streaming execution")

        gripper = GripperWorker(robot, not args.no_gripper)
        gripper.wait_ready()
        cameras = RealSensePair(
            args.external_camera_serial,
            args.wrist_camera_serial,
            fps=args.camera_fps,
        ).start()
        print(f"Cameras ready: {cameras.serials}")
        policy = InferenceWorker(args.policy_host, args.policy_port, args.open_loop_horizon, args.transport)
        print(
            f"pi0.5 {args.transport} server connected at {args.policy_host}:{args.policy_port} "
            f"(requested checkpoint: {args.checkpoint})"
        )
        if policy.metadata:
            print(f"Server metadata keys: {', '.join(sorted(policy.metadata))}")
        if not args.no_rviz:
            from fr3_pi05.visualization import RvizBridge

            rviz = RvizBridge(launch_rviz=not args.rviz_publish_only)

        observation, frames, q, _ = _make_observation(cameras, robot, gripper, prompt, args.max_camera_age)
        policy.submit(observation)
        first = _wait_result(policy)
        age = first.completed_at - first.requested_at
        if age > args.max_inference_age:
            raise RuntimeError(f"Initial inference was stale ({age:.2f}s > {args.max_inference_age:.2f}s)")
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
            )
        print(
            f"Inference check: actions={first.actions.shape}, latency={age:.3f}s, "
            f"preview EEF={np.array2string(forward_kinematics(predicted[-1])[:3, 3], precision=4)}"
        )
        if args.check:
            print("CHECK PASSED. Bamboo streaming was not started; the robot did not move.")
            return 0

        _confirm_execution(args)
        if args.execute:
            joystick = LinuxJoystick(args.joystick).open()
            snapshot = joystick.snapshot()
            if len(snapshot.buttons) <= Mapping().quit_button:
                raise RuntimeError("Joystick does not expose the configured Back abort button")
            initial_back_count = snapshot.press_counts[Mapping().quit_button]
            robot.start_stream(args.watchdog_ms, args.max_joint_speed, args.max_joint_acceleration)
            stream_started = True
            print("Bamboo policy streaming ARMED. Press Back, Ctrl+C, or the physical E-stop to stop.")
        else:
            initial_back_count = 0
            print("Inference-only rollout. Add --execute to allow robot motion.")

        action_chunk: np.ndarray | None = first.actions
        next_chunk: np.ndarray | None = None
        action_index = 0
        desired_velocity = np.zeros(7)
        policy_steps = 0
        stream_period = 1.0 / args.stream_hz
        control_period = 1.0 / args.control_hz
        next_stream = time.monotonic()
        next_control = next_stream

        while policy_steps < args.max_steps and not stopped.is_set():
            now = time.monotonic()
            if joystick is not None:
                snapshot = joystick.snapshot()
                if not snapshot.connected:
                    raise RuntimeError("Joystick disconnected; stopping policy execution")
                if snapshot.press_counts[Mapping().quit_button] != initial_back_count:
                    print("Back pressed; stopping.")
                    break

            result = policy.poll()
            if result is not None:
                inference_age = result.completed_at - result.requested_at
                if inference_age > args.max_inference_age:
                    raise RuntimeError(
                        f"Discarding stale inference ({inference_age:.2f}s > {args.max_inference_age:.2f}s)"
                    )
                if action_chunk is None:
                    action_chunk = result.actions
                    action_index = 0
                elif next_chunk is None:
                    next_chunk = result.actions
                else:
                    raise RuntimeError("Received an unexpected extra policy chunk")

            if now >= next_control:
                _, q = _state(robot)
                frames = cameras.snapshot(args.max_camera_age)
                if action_chunk is not None:
                    action = action_chunk[action_index]
                    desired_velocity = safe_joint_velocity(
                        q,
                        action[:7],
                        duration=control_period,
                        maximum=args.max_joint_speed,
                        joint_margin=args.joint_margin,
                        workspace=workspace,
                    )
                    if args.execute:
                        gripper.command(float(action[7]))
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
                    if remaining <= args.prefetch_actions and next_chunk is None and not policy.busy:
                        observation = build_droid_observation(
                            frames["exterior_image_left"].image,
                            frames["wrist_image"].image,
                            q,
                            gripper.position,
                            prompt,
                        )
                        policy.submit(observation)
                    if action_index >= args.open_loop_horizon:
                        action_chunk = next_chunk
                        next_chunk = None
                        action_index = 0
                        if action_chunk is None:
                            desired_velocity = np.zeros(7)
                else:
                    desired_velocity = np.zeros(7)
                    predicted = None
                    if not policy.busy:
                        observation = build_droid_observation(
                            frames["exterior_image_left"].image,
                            frames["wrist_image"].image,
                            q,
                            gripper.position,
                            prompt,
                        )
                        policy.submit(observation)

                if rviz is not None:
                    rviz.publish(
                        frames["exterior_image_left"].image,
                        frames["wrist_image"].image,
                        q,
                        predicted,
                    )
                if policy_steps and policy_steps % int(max(1, args.control_hz * 2)) == 0:
                    mode = "EXECUTING" if args.execute else "INFERENCE ONLY"
                    print(f"{mode}: step {policy_steps}/{args.max_steps}, |dq_cmd|={np.linalg.norm(desired_velocity):.3f}")
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
