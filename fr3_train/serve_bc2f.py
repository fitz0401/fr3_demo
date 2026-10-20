"""Stand-alone policy server for the scope-v5 blue-cup checkpoints (pi05_bc2f_*, 2026-10-20).

Covers the three arms trained on fitz0401/franka_pour_lillet_blue_cup_2_force (see fr3_train/bc2f/DEPLOY_PLAN.md):

    pi05_bc2f_vanilla_s0       no dropout
    pi05_bc2f_moddrop_s0       ModDrop: each camera and the force input dropped during training (p 0.3)
    pi05_bc2f_moddrop_prop_s0  ModDrop + proprioception (state and history) dropped during training (p 0.5)

Dropout was a training-time tool only: every model gets all inputs here, so one loader serves all three and the
checkpoint directory decides which weights are loaded. Self-contained like serve_bluecup.py: the TrainConfig is
rebuilt inline and needs only upstream OpenPI at the training revision (15a9616) plus fr3_train/deploy.patch, and
this directory's force_features.py (pure numpy).

    .venv/bin/python /path/to/fr3_demo/fr3_train/serve_bc2f.py --checkpoint-dir /path/to/bundle --self-test \
        --golden /path/to/bundle/golden.npz

Input contract (all keys required):

    observation/exterior_image_1_left   (H, W, 3) uint8   main exterior camera      -> base_0_rgb
    observation/wrist_image_left        (H, W, 3) uint8   wrist camera              -> left_wrist_0_rgb
    observation/exterior_image_2_left   (H, W, 3) uint8   second exterior (L515)    -> right_wrist_0_rgb
    observation/joint_position          (3, 7) float      current first, then t-45 and t-75 (15 Hz frames)
    observation/gripper_position        (3,) float        same ordering, 1.0 open ... ~0.78 closed
    observation/raw_force               (N, 15) float64   raw arm samples [bamboo time_sec, q (7), tau_J (7)],
                                                          covering at least the 64 ms before robot_timestamp
    observation/robot_timestamp         float             bamboo time_sec of the state sent as the current
                                                          joint position (same clock as raw_force)
    prompt                              str               must be "pour lillet into the blue cup"

The model sees, as labelled prompt segments, State (q_t, gripper_t), History (the two lagged frames) and Force:
the flange wrench (gravity-compensated with the bundle's force_model.json) averaged over 4 bins of the 64 ms before
robot_timestamp. The response is `actions` (16, 8): joint velocities (rad/s) and the gripper command, 16 steps at
15 Hz, plus `force` with the wrench window the model saw and its coverage, for the rollout log.
"""

import dataclasses
import importlib.util
import logging
import pathlib
import sys
import time

import jax.numpy as jnp
import numpy as np
import tyro
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import droid_policy
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def _load_force_features():
    path = pathlib.Path(__file__).resolve().parent / "force_features.py"
    name = "fr3_bc2f_force_features"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


force_features = _load_force_features()

LOG = logging.getLogger("pi05_bc2f")

STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 1 + len(STATE_HISTORY_LAGS)
TASKS = ("pour lillet into the blue cup",)
DEFAULT_ASSET_ID = "fitz0401/franka_pour_lillet_blue_cup_2_force"
MAX_TOKEN_LEN = 240
FORCE_DIMS = force_features.BINS * 6
# Prompt segments in state order, as training's DroidInputsSegmented.segments() with SEGMENT_LABELS.
SEGMENTS = (("State", 8), ("History", 8 * len(STATE_HISTORY_LAGS)), ("Force", FORCE_DIMS))
# Client camera key -> pi0.5 image slot, exactly as in training (use_second_exterior=True).
CAMERA_SLOTS = {
    "exterior_image_1_left": "base_0_rgb",
    "wrist_image_left": "left_wrist_0_rgb",
    "exterior_image_2_left": "right_wrist_0_rgb",
}
RAW_FORCE_KEY = "observation/raw_force"
ROBOT_TIME_KEY = "observation/robot_timestamp"
# The current joint position must agree with the raw stream at robot_timestamp to this tolerance; a larger
# difference means the timestamp and the joints come from different instants (or clocks).
MAX_JOINT_MISMATCH_RAD = 0.02


@dataclasses.dataclass(frozen=True)
class Bc2fInputs(_transforms.DataTransformFn):
    """The inference half of training's DroidInputsSegmented(num_state_frames=3, use_second_exterior=True,
    force_dims=24) with every segment present. `observation/force` (24) is computed by `Bc2fPolicy` from the raw
    stream, the deployment counterpart of training's ForceFromTable. Defined here, not imported, so this file runs
    against stock openpi + deploy.patch."""

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
        if joints.shape != (self.num_state_frames, 7) or gripper.shape[0] != self.num_state_frames:
            raise ValueError(
                f"Expected {self.num_state_frames} stacked state frames, got joint_position {joints.shape} and "
                f"gripper_position {gripper.shape}. Send the current frame first, then t-45 and t-75."
            )
        state = np.concatenate([np.concatenate([joints[i], gripper[i]]) for i in range(self.num_state_frames)])
        force = np.asarray(data["observation/force"], dtype=np.float32).reshape(-1)
        if force.shape[0] != FORCE_DIMS:
            raise ValueError(f"Expected {FORCE_DIMS} force values, got {force.shape}")
        state = np.concatenate([state, force])

        names = tuple(CAMERA_SLOTS.values())
        images = [droid_policy._parse_image(data[f"observation/{key}"]) for key in CAMERA_SLOTS]
        inputs = {
            "state": state,
            "segment_present": np.ones(len(SEGMENTS), dtype=bool),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, [np.True_] * len(names), strict=True)),
        }
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class TokenizeSegments(_transforms.DataTransformFn):
    """Training's transforms.TokenizeSegmentedPrompt + PaligemmaTokenizer.tokenize_segments, inlined:

        "Task: <task>, State: <8 bins>, History: <16 bins>, Force: <24 bins>;\\nAction: "

    with each normalised value discretised into 256 bins on [-1, 1). Overlong prompts raise instead of being
    truncated (truncation would cut the "Action:" cue). Afterwards `state` keeps only the first segment."""

    tokenizer: _tokenizer.PaligemmaTokenizer
    segments: tuple[tuple[str, int], ...] = SEGMENTS

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        state = np.asarray(data["state"])
        present = np.asarray(data.pop("segment_present"), dtype=bool).reshape(-1)
        if present.shape[0] != len(self.segments) or state.shape[-1] != sum(n for _, n in self.segments):
            raise ValueError(f"state {state.shape} / presence {present.shape} do not match the layout {self.segments}")
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        body, start = "", 0
        for (label, dims), on in zip(self.segments, present, strict=True):
            if on:
                body += f", {label}: {' '.join(map(str, np.digitize(state[start : start + dims], bins=bins) - 1))}"
            start += dims
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.tokenizer._tokenizer.encode(f"Task: {cleaned}{body};\nAction: ", add_bos=True)
        max_len = self.tokenizer._max_len
        if len(tokens) > max_len:
            raise ValueError(f"Segmented prompt has {len(tokens)} tokens, more than max_token_len={max_len}")
        mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
        tokens = tokens + [False] * (max_len - len(tokens))
        first = self.segments[0][1]
        kept = state[:first] if present[0] else np.zeros(first, dtype=state.dtype)
        return {**data, "state": kept, "tokenized_prompt": np.asarray(tokens), "tokenized_prompt_mask": np.asarray(mask)}


@dataclasses.dataclass(frozen=True)
class _Bc2fDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=_transforms.Group(
                inputs=[Bc2fInputs(model_config.model_type)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            model_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultPrompt(None),
                    _transforms.ResizeImages(224, 224),
                    TokenizeSegments(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
            ),
        )


def _model_config() -> pi0_config.Pi0Config:
    """The training recipe: _PI05_WINE_MODEL with max_token_len=240. Must match exactly."""
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora_r32",
        action_expert_variant="gemma_300m_lora",
        max_token_len=MAX_TOKEN_LEN,
    )


class Bc2fPolicy:
    """Turns the raw arm stream into the 24 force inputs, runs the policy, and echoes what the model saw."""

    def __init__(self, policy, force_model):
        self._policy = policy
        self._force_model = force_model

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def prepare(self, obs: dict) -> tuple[dict, dict]:
        obs = dict(obs)
        for key in (RAW_FORCE_KEY, ROBOT_TIME_KEY):
            if key not in obs:
                raise ValueError(f"{key} is required: the bc2f checkpoints read the raw arm stream (see the docstring)")
        raw = force_features.clean_raw(obs.pop(RAW_FORCE_KEY))
        t_obs = float(np.asarray(obs.pop(ROBOT_TIME_KEY)).reshape(-1)[0])
        feats = force_features.window_features(self._force_model, raw, t_obs)
        joints = np.asarray(obs["observation/joint_position"], dtype=np.float64).reshape(-1, 7)
        q_raw = np.array([np.interp(t_obs, raw[:, 0], raw[:, 1 + j]) for j in range(7)])
        mismatch = float(np.abs(joints[0] - q_raw).max())
        if mismatch > MAX_JOINT_MISMATCH_RAD:
            raise ValueError(
                f"current joint_position differs from the raw stream at robot_timestamp by {mismatch:.3f} rad: "
                "send the time_sec of the same state as the current joints"
            )
        obs["observation/force"] = feats["force"]
        info = {
            "window": np.asarray(feats["window"], np.float32),
            "valid_fraction": feats["valid_fraction"],
            "samples_in_window": feats["samples_in_window"],
            "staleness_ms": 1000.0 * feats["staleness_s"],
            "joint_mismatch_rad": mismatch,
        }
        return obs, info

    def infer(self, obs: dict, **kwargs) -> dict:
        prepared, info = self.prepare(obs)
        result = self._policy.infer(prepared, **kwargs)
        result["force"] = info
        return result


def build_policy(checkpoint_dir: str, asset_id: str = DEFAULT_ASSET_ID):
    path = pathlib.Path(checkpoint_dir) / "force_model.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the bundle must ship the force model it was trained with")
    config = _config.TrainConfig(
        name="pi05_bc2f",
        exp_name="deploy",
        model=_model_config(),
        # repo_id only selects the norm stats under the checkpoint's assets/ directory.
        data=_Bc2fDataConfig(repo_id=asset_id, base_config=_config.DataConfig(prompt_from_task=True)),
    )
    # create_trained_policy, except that only the actions are unnormalised: the 48 state values are normalised on
    # the way in, but the segmented prompt keeps just the 8-value State segment (padded to 32) in `state`, which
    # the stock Unnormalize (strict, all norm-stat keys) would try to map back with the 48-dim statistics and fail.
    ckpt = pathlib.Path(checkpoint_dir)
    model = config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(ckpt / "assets", data_config.asset_id)
    quantiles = data_config.use_quantile_norm
    policy = _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=quantiles),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize({"actions": norm_stats["actions"]}, use_quantiles=quantiles),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
    )
    return Bc2fPolicy(policy, force_features.ForceModel.from_json(path))


# A still arm at home with an empty gripper (blue_cup_2_force episode 3, frame 20, raw-sample means): the force
# window reads about (0.42, 0.16, -0.23) N, inside the training range at the start-of-episode pause.
HOME_Q = np.array([-0.0472, -0.7308, -0.0279, -2.2839, 0.0022, 1.5756, 0.0328])
HOME_TAU = np.array([0.02, -6.426, -0.308, 23.925, 0.588, 2.438, -0.165])


def example_raw_force(t_obs: float = 100.0, rate_hz: float = 860.0, span_s: float = 0.128, q=HOME_Q, tau=HOME_TAU):
    """A still arm sampled at `rate_hz` over (t_obs - span_s, t_obs], for self-tests and warm-up."""
    t = t_obs - np.arange(int(span_s * rate_hz))[::-1] / rate_hz
    return np.concatenate([t[:, None], np.tile(q, (len(t), 1)), np.tile(tau, (len(t), 1))], axis=1)


def example_observation(rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    return {
        "observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
        "observation/joint_position": np.tile(HOME_Q, (NUM_STATE_FRAMES, 1)),
        "observation/gripper_position": np.full((NUM_STATE_FRAMES,), 1.0),
        RAW_FORCE_KEY: example_raw_force(),
        ROBOT_TIME_KEY: 100.0,
        "prompt": TASKS[0],
    }


def self_test(policy) -> None:
    LOG.info("first call includes JIT compilation and is much slower")
    for name in ("first call", "second call"):
        t0 = time.monotonic()
        out = policy.infer(example_observation())
        actions = np.asarray(out["actions"])
        assert actions.shape == (16, 8) and np.all(np.isfinite(actions)), actions.shape
        LOG.info(f"{name:12s} {(time.monotonic() - t0) * 1000:8.1f} ms  actions {actions.shape}  "
                 f"F_z bins {np.round(out['force']['window'][:, 2], 2)} N  coverage {out['force']['valid_fraction']:.2f}")
    for name, mutate in {
        "missing raw_force": lambda o: o.pop(RAW_FORCE_KEY),
        "missing robot_timestamp": lambda o: o.pop(ROBOT_TIME_KEY),
        "stale raw_force (50 ms)": lambda o: o.update({ROBOT_TIME_KEY: 100.05}),
        "raw_force too short": lambda o: o.update({RAW_FORCE_KEY: example_raw_force(span_s=0.03)}),
        "joints from another instant": lambda o: o.update(
            {"observation/joint_position": np.tile(HOME_Q + 0.1, (NUM_STATE_FRAMES, 1))}),
        "missing exterior_image_2_left": lambda o: o.pop("observation/exterior_image_2_left"),
        "single-frame state": lambda o: o.update({"observation/joint_position": HOME_Q}),
    }.items():
        obs = example_observation()
        mutate(obs)
        try:
            policy.infer(obs)
        except Exception as e:  # noqa: BLE001 - we only care that it refuses
            LOG.info(f"{name} correctly rejected: {type(e).__name__}: {str(e)[:90]}")
        else:
            raise RuntimeError(f"A request with {name} was ACCEPTED")
    LOG.info("self-test OK")


def check_golden(policy, path: str, tolerance: float) -> float:
    """Replay golden.npz (written on Leonardo by the training pipeline) and compare force inputs and actions.

    Same inputs, same diffusion noise: on the same GPU type the action difference is ~0; across GPU types bf16
    rounding leaves ~1e-3. A difference near the size of the actions themselves means the deployment does not
    reproduce training (openpi revision, patch, norm stats, slots, prompt segments, force model)."""
    golden = np.load(path, allow_pickle=False)
    worst, worst_force = 0.0, 0.0
    for k in range(len(golden["meta/frames"])):
        prefix = f"f{k}/"
        obs = {key[len(prefix):]: golden[key] for key in golden.files
               if key.startswith(prefix) and key[len(prefix):].startswith("observation/")}
        obs["prompt"] = TASKS[0]
        prepared, _ = policy.prepare(obs)
        force_diff = float(np.max(np.abs(prepared["observation/force"] - golden[prefix + "expected/force"])))
        out = policy.infer(obs, noise=golden[prefix + "noise"])
        expected = golden[prefix + "expected/actions"]
        diff = float(np.max(np.abs(np.asarray(out["actions"], np.float32) - expected)))
        worst, worst_force = max(worst, diff), max(worst_force, force_diff)
        LOG.info(f"golden {str(golden['meta/labels'][k]):24s} force max|diff| {force_diff:.2e}  "
                 f"actions max|diff| {diff:.2e}  (max|action| {float(np.max(np.abs(expected))):.3f})")
    ok = worst <= tolerance and worst_force <= 1e-3
    LOG.info(f"golden check {'PASS' if ok else 'FAIL'}: actions {worst:.2e} (tolerance {tolerance:.0e}), "
             f"force inputs {worst_force:.2e} (tolerance 1e-03) ({golden['meta/config_name']})")
    if not ok:
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
        metadata={"model": "pi05_bc2f", "action_horizon": 16, "tasks": list(TASKS)},
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
