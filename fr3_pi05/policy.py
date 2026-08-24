"""OpenPI DROID protocol helpers and action safety checks."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from fr3_demo.kinematics import JOINT_LOWER, JOINT_UPPER, WorkspaceBounds, forward_kinematics, limit_joint_velocity

DROID_CONTROL_HZ = 15.0
DROID_ACTION_DIM = 8
WINE_HISTORY_OFFSETS = (0, 45, 75)


class ProprioHistory:
    """Keep collector-rate joint/gripper samples for the wine policy."""

    def __init__(self, offsets: tuple[int, ...] = WINE_HISTORY_OFFSETS) -> None:
        if not offsets or offsets[0] != 0 or any(offset < 0 for offset in offsets):
            raise ValueError("History offsets must start at zero and be non-negative")
        self.offsets = offsets
        self._samples: deque[tuple[np.ndarray, float]] = deque(maxlen=max(offsets) + 1)

    @property
    def ready(self) -> bool:
        return len(self._samples) == self._samples.maxlen

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def required_samples(self) -> int:
        return int(self._samples.maxlen or 0)

    def append(self, joint_position: np.ndarray, gripper_position: float) -> None:
        joints = np.asarray(joint_position, dtype=np.float32)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("A history sample requires seven finite joint positions")
        if not np.isfinite(gripper_position):
            raise ValueError("A history sample requires a finite gripper position")
        self._samples.append((joints.copy(), float(np.clip(gripper_position, 0.0, 1.0))))

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return values ordered current, then each configured past offset."""

        if not self.ready:
            raise RuntimeError(f"Proprio history needs {self.required_samples} samples; got {self.sample_count}")
        selected = [self._samples[-1 - offset] for offset in self.offsets]
        joints = np.stack([sample[0] for sample in selected]).astype(np.float32, copy=False)
        gripper = np.asarray([sample[1] for sample in selected], dtype=np.float32)
        return joints, gripper


class SafetyViolation(RuntimeError):
    """Raised when a policy action would violate a configured robot bound."""


def _resize(image: np.ndarray) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("pi0.5 image support is missing; run: pip install -e '.[pi05]'") from error
    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    ratio = max(source.width / 224, source.height / 224)
    resized_width = max(1, int(source.width / ratio))
    resized_height = max(1, int(source.height / ratio))
    source = source.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    result = Image.new(source.mode, (224, 224), 0)
    result.paste(source, ((224 - source.width) // 2, (224 - source.height) // 2))
    return np.asarray(result)


def _resize_wine(image: np.ndarray) -> np.ndarray:
    """Match the 320x180 LeRobot frames used by the wine training pipeline."""

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("pi0.5 image support is missing; run: pip install -e '.[pi05]'") from error
    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(source.resize((320, 180), resample=Image.Resampling.BICUBIC))


def build_droid_observation(
    exterior_image: np.ndarray,
    wrist_image: np.ndarray,
    joint_position: np.ndarray,
    gripper_position: float | np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Build the exact observation dictionary expected by ``pi05_droid``."""

    joints = np.asarray(joint_position, dtype=np.float32)
    if joints.shape not in ((7,), (3, 7)) or not np.all(np.isfinite(joints)):
        raise ValueError("DROID observation requires joint positions shaped (7,) or (3, 7)")
    if not prompt.strip():
        raise ValueError("A non-empty language instruction is required")
    gripper = np.asarray(gripper_position, dtype=np.float32)
    if gripper.ndim == 0:
        gripper = gripper.reshape(1)
    if gripper.shape not in ((1,), (3,)) or not np.all(np.isfinite(gripper)):
        raise ValueError("DROID observation requires 1 or 3 finite gripper positions")
    resize_image = _resize_wine if joints.shape == (3, 7) else _resize
    return {
        "observation/exterior_image_1_left": resize_image(np.asarray(exterior_image, dtype=np.uint8)),
        "observation/wrist_image_left": resize_image(np.asarray(wrist_image, dtype=np.uint8)),
        "observation/joint_position": joints,
        "observation/gripper_position": np.clip(gripper, 0.0, 1.0).astype(np.float32, copy=False),
        "prompt": prompt.strip(),
    }


def validate_action_chunk(response: dict[str, Any], minimum_horizon: int) -> np.ndarray:
    """Validate and clip a server response to the physical DROID action schema."""

    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("Policy response does not contain 'actions'")
    actions = np.asarray(response["actions"], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != DROID_ACTION_DIM:
        raise ValueError(f"Expected policy actions shaped [N, 8], got {actions.shape}")
    if actions.shape[0] < minimum_horizon:
        raise ValueError(f"Policy returned {actions.shape[0]} actions, fewer than horizon {minimum_horizon}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned NaN or infinite actions")
    # This matches Physical Intelligence's official DROID rollout example.
    return np.clip(actions, -1.0, 1.0)


def safe_joint_velocity(
    joint_position: np.ndarray,
    requested_velocity: np.ndarray,
    *,
    duration: float,
    maximum: float,
    joint_margin: float,
    workspace: WorkspaceBounds,
) -> np.ndarray:
    """Bound one velocity action by speed, joint margins, and EEF workspace."""

    q = np.asarray(joint_position, dtype=float)
    velocity = np.asarray(requested_velocity, dtype=float)
    if q.shape != (7,) or velocity.shape != (7,):
        raise SafetyViolation("Joint position and velocity must each contain seven values")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(velocity)):
        raise SafetyViolation("Non-finite robot state or policy velocity")
    if duration <= 0 or maximum <= 0 or joint_margin < 0:
        raise ValueError("Invalid action safety parameters")

    velocity = limit_joint_velocity(velocity, maximum)
    lower = JOINT_LOWER + joint_margin
    upper = JOINT_UPPER - joint_margin
    if np.any(lower >= upper):
        raise ValueError("Joint margin leaves no valid range")

    step = velocity * duration
    scale = 1.0
    for index, delta in enumerate(step):
        if delta > 0:
            scale = min(scale, max(0.0, float((upper[index] - q[index]) / delta)))
        elif delta < 0:
            scale = min(scale, max(0.0, float((lower[index] - q[index]) / delta)))
    velocity *= float(np.clip(scale, 0.0, 1.0))
    target = q + velocity * duration
    if not workspace.contains(forward_kinematics(target)[:3, 3]):
        raise SafetyViolation("Policy action would leave the configured EEF workspace")
    return velocity


def predict_joint_path(
    joint_position: np.ndarray,
    action_chunk: np.ndarray,
    *,
    horizon: int,
    control_hz: float,
    maximum: float,
    joint_margin: float,
    workspace: WorkspaceBounds,
) -> np.ndarray:
    """Integrate a guarded action prefix for RViz trajectory preview."""

    q = np.asarray(joint_position, dtype=float).copy()
    path = [q.copy()]
    for action in np.asarray(action_chunk)[:horizon]:
        velocity = safe_joint_velocity(
            q,
            action[:7],
            duration=1.0 / control_hz,
            maximum=maximum,
            joint_margin=joint_margin,
            workspace=workspace,
        )
        q = q + velocity / control_hz
        path.append(q.copy())
    return np.stack(path)


@dataclass(frozen=True)
class InferenceResult:
    actions: np.ndarray
    requested_at: float
    completed_at: float


class InferenceWorker:
    """Serialize blocking websocket inference away from the control loop."""

    def __init__(
        self,
        host: str,
        port: int,
        horizon: int,
        transport: str = "zmq",
        zmq_mode: str = "connect",
    ) -> None:
        from fr3_pi05.protocol import OpenPiWebsocketClient, OpenPiZmqClient

        if transport == "zmq":
            self._policy = OpenPiZmqClient(host, port, connection_mode=zmq_mode)
        elif transport == "websocket":
            self._policy = OpenPiWebsocketClient(host, port)
        else:
            raise ValueError(f"Unsupported policy transport: {transport}")
        self._horizon = horizon
        self._requests: queue.Queue[tuple[dict[str, Any], float] | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult | Exception] = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pi05-inference", daemon=True)
        self._thread.start()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._results.empty()

    def submit(self, observation: dict[str, Any]) -> bool:
        if self.busy or not self._requests.empty():
            return False
        self._busy.set()
        self._requests.put_nowait((observation, time.monotonic()))
        return True

    def poll(self) -> InferenceResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        if isinstance(result, Exception):
            raise result
        return result

    def _loop(self) -> None:
        while True:
            item = self._requests.get()
            if item is None:
                return
            observation, requested_at = item
            try:
                response = self._policy.infer(observation)
                result: InferenceResult | Exception = InferenceResult(
                    validate_action_chunk(response, self._horizon), requested_at, time.monotonic()
                )
            except Exception as error:  # noqa: BLE001 - propagate third-party websocket errors to control thread
                result = error
            self._busy.clear()
            try:
                self._results.put_nowait(result)
            except queue.Full:
                pass

    def close(self) -> None:
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.2)
        if not self._thread.is_alive():
            self._policy.close()
