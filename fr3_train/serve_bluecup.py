"""Stand-alone policy server for the blue-cup force / sensor-dropout checkpoints.

Covers the three checkpoints released for the first real-robot round (see fr3_train/bluecup/DEPLOY.md):

    pi05_bluecup_flag         no sensor dropout (baseline with the same inputs)
    pi05_bluecup_moddrop      random single-channel dropout (ModDrop baseline)
    pi05_bluecup_alloc_fixed  phase-structured three-branch dropout ("v4")

All three share one architecture and one input contract, so one loader serves any of them; the
checkpoint directory decides which weights are loaded. Self-contained like serve_wine.py: the
TrainConfig is rebuilt inline, so the GPU machine needs only upstream OpenPI at the training
revision plus `fr3_train/deploy.patch` (the `gemma_2b_lora_r32` variant and the FFN LoRA fix).

    .venv/bin/python /path/to/fr3_demo/fr3_train/serve_bluecup.py --checkpoint-dir /path/to/ckpt --self-test \
        --golden /path/to/ckpt/golden.npz

Input contract (all keys are required unless marked optional):

    observation/exterior_image_1_left   (H, W, 3) uint8   main exterior camera      -> base_0_rgb
    observation/wrist_image_left        (H, W, 3) uint8   wrist camera              -> left_wrist_0_rgb
    observation/exterior_image_2_left   (H, W, 3) uint8   second exterior (L515)    -> right_wrist_0_rgb
    observation/joint_position          (3, 7) float      current first, then t-45 and t-75 (15 Hz frames)
    observation/gripper_position        (3,) float        same ordering, 1.0 open ... ~0.78 closed
    observation/joint_torque            (7,) float        measured link-side torque tau_J in Nm, current
                                                          frame only, NOT gravity-compensated
    prompt                              str               must be "pour lillet into the blue cup"

Optional fault injection for robustness trials. These reproduce exactly what training-time dropout
and the offline robustness table did, so a real-robot cell can be compared with its offline cell:

    fault/absent_cameras   list of camera keys (e.g. ["exterior_image_1_left", ...]):
                           declared absent -> black picture AND image_mask False
    fault/black_cameras    list of camera keys: picture black, mask stays True (undeclared blackout)
    fault/force            "present" (default) | "absent" (torque zeroed, presence flag 0)
                           | "dead_undeclared" (torque zeroed, presence flag stays 1)

The state the model sees is 32-dim: [q_t, g_t, q_t-45, g_t-45, q_t-75, g_t-75] (24) + tau_J (7) +
force-presence flag (1). The response is `actions` (16, 8) float: joint velocities (rad/s) and the
gripper command, 16 steps at 15 Hz, plus `fault_applied` echoing what was injected.
"""

import dataclasses
import logging
import time

import numpy as np
import tyro
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

LOG = logging.getLogger("pi05_bluecup")

STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 1 + len(STATE_HISTORY_LAGS)
TASKS = ("pour lillet into the blue cup",)
DEFAULT_ASSET_ID = "fitz0401/franka_pour_lillet_blue_cup_force"
NUM_JOINTS = 7
# Client camera key -> pi0.5 image slot, exactly as in training (use_second_exterior=True).
CAMERA_SLOTS = {
    "exterior_image_1_left": "base_0_rgb",
    "wrist_image_left": "left_wrist_0_rgb",
    "exterior_image_2_left": "right_wrist_0_rgb",
}
FORCE_MODES = ("present", "absent", "dead_undeclared")


def _camera_list(value, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        value = [value]
    names = tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(value, dtype=object).ravel())
    unknown = [n for n in names if n not in CAMERA_SLOTS]
    if unknown:
        raise ValueError(f"{key}: unknown camera(s) {unknown}; expected some of {list(CAMERA_SLOTS)}")
    return names


def parse_fault(data: dict) -> dict:
    """Validate the optional fault fields; returns a plain dict that is echoed to the client."""
    absent = _camera_list(data.get("fault/absent_cameras"), "fault/absent_cameras")
    black = _camera_list(data.get("fault/black_cameras"), "fault/black_cameras")
    force = data.get("fault/force", "present")
    force = force.decode("utf-8") if isinstance(force, bytes) else str(force)
    if force not in FORCE_MODES:
        raise ValueError(f"fault/force must be one of {FORCE_MODES}, got {force!r}")
    overlap = set(absent) & set(black)
    if overlap:
        raise ValueError(f"A camera cannot be both absent and black: {sorted(overlap)}")
    return {"absent_cameras": list(absent), "black_cameras": list(black), "force": force}


@dataclasses.dataclass(frozen=True)
class BluecupInputs(_transforms.DataTransformFn):
    """The inference half of training's DroidInputsWithStateHistory(num_state_frames=3,
    use_second_exterior=True, include_force=True, include_force_flag=True), plus the fault fields.
    Defined here, not imported, so this file runs against stock openpi + deploy.patch."""

    model_type: _model.ModelType
    num_state_frames: int = NUM_STATE_FRAMES

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["observation/joint_position"])
        gripper = np.asarray(data["observation/gripper_position"])
        if joints.ndim == 1:
            joints = joints[np.newaxis, :]
        if gripper.ndim == 0:
            gripper = gripper.reshape(1, 1)
        elif gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1) if joints.shape[0] > 1 else gripper.reshape(1, -1)
        if joints.shape[0] != self.num_state_frames or gripper.shape[0] != self.num_state_frames:
            raise ValueError(
                f"Expected {self.num_state_frames} stacked state frames, got joint_position "
                f"{joints.shape} and gripper_position {gripper.shape}. Send the same number of "
                "frames the model was trained with, current first."
            )
        state = np.concatenate([np.concatenate([joints[i], gripper[i]]) for i in range(self.num_state_frames)])

        if "observation/joint_torque" not in data:
            raise ValueError(
                "observation/joint_torque is required (measured tau_J, 7 values, Nm). To run without "
                "force, send the measured torque and set fault/force='absent'."
            )
        torque = np.asarray(data["observation/joint_torque"], dtype=np.float32).reshape(-1)
        if torque.shape[0] != NUM_JOINTS or not np.all(np.isfinite(torque)):
            raise ValueError(f"observation/joint_torque must hold {NUM_JOINTS} finite values, got {torque.shape}")

        fault = parse_fault(data)
        if fault["force"] != "present":
            torque = np.zeros_like(torque)
        present = np.float32(0.0 if fault["force"] == "absent" else 1.0)
        state = np.concatenate([state.astype(np.float32), torque, np.asarray([present], dtype=np.float32)])

        names = tuple(CAMERA_SLOTS.values())
        images = [droid_policy._parse_image(data[f"observation/{key}"]) for key in CAMERA_SLOTS]
        masks = [np.True_, np.True_, np.True_]
        for i, key in enumerate(CAMERA_SLOTS):
            if key in fault["absent_cameras"]:
                images[i], masks[i] = np.zeros_like(images[i]), np.False_
            elif key in fault["black_cameras"]:
                images[i] = np.zeros_like(images[i])

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, masks, strict=True)),
        }
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class _BluecupDataConfig(_config.DataConfigFactory):
    num_state_frames: int = NUM_STATE_FRAMES

    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=_transforms.Group(
                inputs=[BluecupInputs(model_config.model_type, self.num_state_frames)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def _model() -> pi0_config.Pi0Config:
    """The training recipe (_PI05_WINE_MODEL in the training config). Must match exactly."""
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora_r32",
        action_expert_variant="gemma_300m_lora",
    )


class _FaultEcho:
    """Wrap the policy so the response records which fault was actually applied."""

    def __init__(self, policy):
        self._policy = policy

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def infer(self, obs: dict, **kwargs) -> dict:
        fault = parse_fault(obs)
        result = self._policy.infer(obs, **kwargs)
        result["fault_applied"] = fault
        return result


def build_policy(checkpoint_dir: str, asset_id: str = DEFAULT_ASSET_ID):
    config = _config.TrainConfig(
        name="pi05_bluecup",
        exp_name="deploy",
        model=_model(),
        # repo_id only selects the norm stats under the checkpoint's assets/ directory.
        data=_BluecupDataConfig(repo_id=asset_id, base_config=_config.DataConfig(prompt_from_task=True)),
    )
    return _FaultEcho(_policy_config.create_trained_policy(config, checkpoint_dir))


def example_observation(rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    q = np.array([-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031])
    return {
        "observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
        "observation/joint_position": np.tile(q, (NUM_STATE_FRAMES, 1)),
        "observation/gripper_position": np.full((NUM_STATE_FRAMES,), 1.0),
        "observation/joint_torque": np.array([0.4, -40.0, 5.0, 24.0, 1.8, -2.0, -0.2]),
        "prompt": TASKS[0],
    }


def self_test(policy) -> None:
    LOG.info("first call includes JIT compilation and is much slower")
    cases = {
        "intact": {},
        "all cameras absent": {"fault/absent_cameras": list(CAMERA_SLOTS)},
        "all cameras black": {"fault/black_cameras": list(CAMERA_SLOTS)},
        "force absent": {"fault/force": "absent"},
        "force dead undeclared": {"fault/force": "dead_undeclared"},
    }
    for name, extra in cases.items():
        obs = {**example_observation(), **extra}
        t0 = time.monotonic()
        out = policy.infer(obs)
        actions = np.asarray(out["actions"])
        assert actions.shape == (16, 8) and np.all(np.isfinite(actions)), actions.shape
        LOG.info(f"{name:24s} {(time.monotonic() - t0) * 1000:8.1f} ms  {actions.shape}  fault={out['fault_applied']}")
    for name, mutate in {
        "missing joint_torque": lambda o: o.pop("observation/joint_torque"),
        "missing exterior_image_2_left": lambda o: o.pop("observation/exterior_image_2_left"),
        "single-frame state": lambda o: o.update({"observation/joint_position": o["observation/joint_position"][0]}),
        "unknown fault camera": lambda o: o.update({"fault/absent_cameras": ["overhead"]}),
    }.items():
        obs = example_observation()
        mutate(obs)
        try:
            policy.infer(obs)
        except Exception as e:  # noqa: BLE001 - we only care that it refuses
            LOG.info(f"{name} correctly rejected: {type(e).__name__}")
        else:
            raise RuntimeError(f"A request with {name} was ACCEPTED")
    LOG.info("self-test OK")


GOLDEN_FAULTS = {
    "intact": {},
    "cams_absent": {"fault/absent_cameras": list(CAMERA_SLOTS)},
    "cams_black": {"fault/black_cameras": list(CAMERA_SLOTS)},
    "force_absent": {"fault/force": "absent"},
    "force_dead_undeclared": {"fault/force": "dead_undeclared"},
}


def check_golden(policy, path: str, tolerance: float) -> float:
    """Replay golden.npz (written on Leonardo by the training pipeline) and compare actions.

    Same inputs, same diffusion noise: on the same GPU type the difference is ~0; across GPU types
    bf16 rounding leaves ~1e-3. A difference near the size of the actions themselves means the
    deployment does not reproduce training (openpi revision, patch, norm stats, slots, faults)."""
    golden = np.load(path, allow_pickle=False)
    conditions = [str(c) for c in golden["meta/conditions"]]
    phases = [str(p) for p in golden["meta/phases"]]
    worst = 0.0
    for k in range(len(golden["meta/frames"])):
        prefix = f"f{k}/"
        base = {key[len(prefix):]: golden[key] for key in golden.files if key.startswith(prefix + "observation/")}
        base["prompt"] = TASKS[0]
        noise = golden[prefix + "noise"]
        for cond in conditions:
            out = policy.infer({**base, **GOLDEN_FAULTS[cond]}, noise=noise)
            expected = golden[f"{prefix}expected/{cond}"]
            diff = float(np.max(np.abs(np.asarray(out["actions"], np.float32) - expected)))
            scale = float(np.max(np.abs(expected)))
            worst = max(worst, diff)
            LOG.info(f"golden {phases[k]:10s} {cond:22s} max|diff| {diff:.2e}  (max|action| {scale:.3f})")
    verdict = "PASS" if worst <= tolerance else "FAIL"
    LOG.info(f"golden check {verdict}: worst max|diff| {worst:.2e}, tolerance {tolerance:.0e} ({golden['meta/config_name']})")
    if worst > tolerance:
        raise SystemExit(1)
    return worst


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "."
    asset_id: str = DEFAULT_ASSET_ID
    host: str = "0.0.0.0"
    port: int = 8000
    self_test: bool = False
    # Path to golden.npz shipped in the checkpoint bundle; checked after --self-test, or alone.
    golden: str = ""
    golden_tolerance: float = 2e-2


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", force=True)
    policy = build_policy(args.checkpoint_dir, args.asset_id)
    if args.self_test or args.golden:
        if args.self_test:
            self_test(policy)
        if args.golden:
            check_golden(policy, args.golden, args.golden_tolerance)
        return
    websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"model": "pi05_bluecup", "action_horizon": 16, "tasks": list(TASKS)},
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
