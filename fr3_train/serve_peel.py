"""Stand-alone policy server for the peeling checkpoints (pi05_peel_*, 2026-11-06; pi05_peel2_*, 2026-11-13).

Round 2 (pi05_peel2_*, fitz0401/fr3_peel2_mix, new peeler / GelSight mount): same inputs and network; the bundle
ships its own norm stats (found automatically), its own force model (the new mount's payload fit) and the gripper
value to pin (BUNDLE.json held_gripper, see pinned_gripper). Arms: splice, mix, moddrop, ronly2 (baselines) and twin,
decl, decl_corr (ours: twin consistency, learned declaration, corrective replay labels); declaration and twins are
training-time only, so here too every model gets every input.

Six arms trained on fitz0401/fr3_peel_mix (hand-guided peeling demonstrations, their robot replays and the spliced
pairs; see fr3_train/peel/DEPLOY_PLAN.md), all with the same inputs at deployment:

    pi05_peel_splice_s0    spliced pairs only (replay cameras + arm stream, hand-guided joints / actions / tactile)
    pi05_peel_mix_s0       spliced + hand-guided + replay episodes, unreliable channels declared absent per frame
    pi05_peel_moddrop_s0   spliced + ModDrop p 0.3 over cameras, force and tactile
    pi05_peel_informed_s0  mix + contact-driven dropout (vision dropped in strokes, touch outside them)
    pi05_peel_konly2_s0    hand-guided episodes only (lower bound)
    pi05_peel_ronly2_s0    replay episodes only (lower bound)

Dropout and declared absence were training-time tools: every model gets every input here. Same structure as
serve_bc2f.py (stock openpi 15a9616 + fr3_train/deploy.patch, force window rebuilt from the raw 1 kHz stream with the
bundle's force_model.json, which folds in the PEELER payload correction), plus a Tactile segment.

    .venv/bin/python /path/to/fr3_demo/fr3_train/serve_peel.py --checkpoint-dir /path/to/bundle --self-test \
        --golden /path/to/bundle/golden.npz

Input contract (all keys required):

    observation/exterior_image_1_left   (H, W, 3) uint8   main exterior camera      -> base_0_rgb
    observation/wrist_image_left        (H, W, 3) uint8   wrist camera              -> left_wrist_0_rgb
    observation/exterior_image_2_left   (H, W, 3) uint8   second exterior (L515)    -> right_wrist_0_rgb
    observation/joint_position          (3, 7) float      current first, then t-45 and t-75 (15 Hz frames)
    observation/gripper_position        (3,) float        IGNORED (echoed): the peeler is clamped and training saw one
                                                          constant, which the server feeds instead (see below)
    observation/tactile                 (4,) float        GelSight marker field since the unloaded reference taken at
                                                          the start of the run: shift x, shift y (px), twist (rad),
                                                          spread -- fr3_demo.gel.GelMonitor's reading, the same
                                                          MarkerShearTracker quantities as the dataset's
                                                          tactile_deformation
    observation/raw_force               (N, 15) float64   raw arm samples [bamboo time_sec, q (7), tau_J (7)],
                                                          covering at least the 64 ms before robot_timestamp
    observation/robot_timestamp         float             bamboo time_sec of the state sent as the current joints
    prompt                              str               must be "peel the lemon"

Why the gripper is pinned: every training frame had gripper_position 0.12334802 (q01 = q99 in the norm stats), so the
normalised value is (x - q01) / 1e-6 and a reading off by 1e-6 moves its prompt token from 0 to 255 or -1, an input the
model never saw. The server feeds the norm-stat constant for all three frames and refuses to start if the bundle's
statistics are not constant there.

The model sees State (q_t, gripper), History (two lagged frames), Force (flange wrench, 4 bins over 64 ms) and Tactile
(the 4 values) as labelled prompt segments. The response is `actions` (16, 8): joint velocities (rad/s) and the
gripper command (always below the open threshold in training; run the arm with --no-gripper), plus `force` (window,
coverage) and `tactile` (what the model saw, the received gripper value) for the rollout log.
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


def _load_sibling(filename: str, name: str):
    path = pathlib.Path(__file__).resolve().parent / filename
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


force_features = _load_sibling("force_features.py", "fr3_peel_force_features")
peel_bundle = _load_sibling("peel_bundle.py", "fr3_peel_bundle")
bundle_asset_id = peel_bundle.bundle_asset_id
pinned_gripper = peel_bundle.pinned_gripper

LOG = logging.getLogger("pi05_peel")

STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 1 + len(STATE_HISTORY_LAGS)
TASKS = ("peel the lemon",)
DEFAULT_ASSET_ID = peel_bundle.DEFAULT_ASSET_ID
MAX_TOKEN_LEN = 256
FORCE_DIMS = force_features.BINS * 6
TACTILE_DIMS = 4
TACTILE_KEY = "observation/tactile"
GRIPPER_DIMS = peel_bundle.GRIPPER_DIMS  # gripper entries of the State and History segments
# Prompt segments in state order, as training's DroidInputsSegmented.segments() with SEGMENT_LABELS.
SEGMENTS = (("State", 8), ("History", 8 * len(STATE_HISTORY_LAGS)), ("Force", FORCE_DIMS), ("Tactile", TACTILE_DIMS))
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
class PeelInputs(_transforms.DataTransformFn):
    """The inference half of training's DroidInputsSegmented(num_state_frames=3, use_second_exterior=True,
    force_dims=24, tactile_dims=4) with every segment present. `observation/force` (24) is computed by `PeelPolicy`
    from the raw stream (training's ForceFromTable); `observation/gripper_position` is replaced by `held_gripper`
    (see the module docstring). Defined here, not imported, so this file runs against stock openpi + deploy.patch."""

    model_type: _model.ModelType
    held_gripper: float
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
        held = np.full((1,), self.held_gripper, dtype=np.float32)
        state = np.concatenate([np.concatenate([joints[i], held]) for i in range(self.num_state_frames)])
        force = np.asarray(data["observation/force"], dtype=np.float32).reshape(-1)
        if force.shape[0] != FORCE_DIMS:
            raise ValueError(f"Expected {FORCE_DIMS} force values, got {force.shape}")
        tactile = np.asarray(data[TACTILE_KEY], dtype=np.float32).reshape(-1)
        if tactile.shape[0] != TACTILE_DIMS or not np.all(np.isfinite(tactile)):
            raise ValueError(f"Expected {TACTILE_DIMS} finite tactile values (shift x, shift y, twist, spread), got {tactile}")
        state = np.concatenate([state, force, tactile])

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

        "Task: <task>, State: <8 bins>, History: <16 bins>, Force: <24 bins>, Tactile: <4 bins>;\\nAction: "

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
class _PeelDataConfig(_config.DataConfigFactory):
    held_gripper: float = 0.0

    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=_transforms.Group(
                inputs=[PeelInputs(model_config.model_type, self.held_gripper)],
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
    """The training recipe: _PI05_WINE_MODEL with max_token_len=256 (52 state values). Must match exactly."""
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora_r32",
        action_expert_variant="gemma_300m_lora",
        max_token_len=MAX_TOKEN_LEN,
    )


class PeelPolicy:
    """Turns the raw arm stream into the 24 force inputs, checks the tactile values, runs the policy, and echoes what
    the model saw."""

    def __init__(self, policy, force_model, held_gripper: float):
        self._policy = policy
        self._force_model = force_model
        self.held_gripper = float(held_gripper)

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def prepare(self, obs: dict) -> tuple[dict, dict]:
        obs = dict(obs)
        for key in (RAW_FORCE_KEY, ROBOT_TIME_KEY):
            if key not in obs:
                raise ValueError(f"{key} is required: the peel checkpoints read the raw arm stream (see the docstring)")
        if TACTILE_KEY not in obs:
            raise ValueError(f"{TACTILE_KEY} is required: the peel checkpoints read the GelSight (see the docstring)")
        tactile = np.asarray(obs[TACTILE_KEY], dtype=np.float64).reshape(-1)
        if tactile.shape[0] != TACTILE_DIMS or not np.all(np.isfinite(tactile)):
            raise ValueError(f"{TACTILE_KEY} must be {TACTILE_DIMS} finite values, got {tactile}")
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
        info["tactile"] = np.asarray(tactile, np.float32)
        info["gripper_received"] = np.asarray(obs["observation/gripper_position"], np.float32).reshape(-1)
        info["gripper_fed"] = self.held_gripper
        return obs, info

    def infer(self, obs: dict, **kwargs) -> dict:
        prepared, info = self.prepare(obs)
        result = self._policy.infer(prepared, **kwargs)
        result["force"] = {k: v for k, v in info.items() if k not in ("tactile", "gripper_received", "gripper_fed")}
        result["tactile"] = {k: info[k] for k in ("tactile", "gripper_received", "gripper_fed")}
        return result


def build_policy(checkpoint_dir: str, asset_id: str | None = None):
    asset_id = asset_id or bundle_asset_id(checkpoint_dir)
    path = pathlib.Path(checkpoint_dir) / "force_model.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the bundle must ship the force model it was trained with")
    ckpt = pathlib.Path(checkpoint_dir)
    stats = _checkpoints.load_norm_stats(ckpt / "assets", asset_id)["state"]
    q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
    expected_dims = 8 * NUM_STATE_FRAMES + FORCE_DIMS + TACTILE_DIMS
    if q01.shape[-1] != expected_dims:
        raise ValueError(f"norm stats have {q01.shape[-1]} state values, expected {expected_dims}")
    held_gripper = pinned_gripper(checkpoint_dir, q01, q99)
    config = _config.TrainConfig(
        name="pi05_peel",
        exp_name="deploy",
        model=_model_config(),
        # repo_id only selects the norm stats under the checkpoint's assets/ directory.
        data=_PeelDataConfig(repo_id=asset_id, base_config=_config.DataConfig(prompt_from_task=True),
                             held_gripper=held_gripper),
    )
    # create_trained_policy, except that only the actions are unnormalised: the 52 state values are normalised on
    # the way in, but the segmented prompt keeps just the 8-value State segment (padded to 32) in `state`, which
    # the stock Unnormalize (strict, all norm-stat keys) would try to map back with the 52-dim statistics and fail.
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
    LOG.info(f"gripper input pinned to {held_gripper!r} ({asset_id})")
    return PeelPolicy(policy, force_features.ForceModel.from_json(path), held_gripper)


# A still arm at the start pose holding the peeler (fr3_peel_mix replay episode 42, frame 20, raw-sample means).
HOME_Q = np.array([-0.0334, -0.7522, -0.0277, -2.2872, -0.0163, 1.5795, 0.04])
HOME_TAU = np.array([-0.298, -5.7, 0.642, 25.275, 1.575, 2.16, 0.173])


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
        "observation/gripper_position": np.full((NUM_STATE_FRAMES,), 0.0),
        TACTILE_KEY: np.zeros(TACTILE_DIMS),
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
                 f"F_z bins {np.round(out['force']['window'][:, 2], 2)} N  coverage {out['force']['valid_fraction']:.2f}  "
                 f"gripper fed {out['tactile']['gripper_fed']:.8f}")
    for name, mutate in {
        "missing raw_force": lambda o: o.pop(RAW_FORCE_KEY),
        "missing robot_timestamp": lambda o: o.pop(ROBOT_TIME_KEY),
        "stale raw_force (50 ms)": lambda o: o.update({ROBOT_TIME_KEY: 100.05}),
        "raw_force too short": lambda o: o.update({RAW_FORCE_KEY: example_raw_force(span_s=0.03)}),
        "joints from another instant": lambda o: o.update(
            {"observation/joint_position": np.tile(HOME_Q + 0.1, (NUM_STATE_FRAMES, 1))}),
        "missing exterior_image_2_left": lambda o: o.pop("observation/exterior_image_2_left"),
        "missing tactile": lambda o: o.pop(TACTILE_KEY),
        "tactile with 2 values": lambda o: o.update({TACTILE_KEY: np.zeros(2)}),
        "tactile not finite": lambda o: o.update({TACTILE_KEY: np.array([np.nan, 0.0, 0.0, 0.0])}),
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
    noise = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    base = policy.infer(example_observation(), noise=noise)["actions"]
    moved = example_observation()
    moved["observation/gripper_position"] = np.full((NUM_STATE_FRAMES,), 0.5)
    other = policy.infer(moved, noise=noise)["actions"]
    if not np.array_equal(np.asarray(base), np.asarray(other)):
        raise RuntimeError("the gripper reading changed the actions: it must be pinned")
    LOG.info("gripper reading ignored (pinned): actions identical for 0.0 and 0.5")
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
    asset_id: str = ""  # default: the one asset the bundle ships (bundle_asset_id)
    host: str = "0.0.0.0"
    port: int = 8000
    self_test: bool = False
    # Path to golden.npz shipped in the checkpoint bundle; checked after --self-test, or alone.
    golden: str = ""
    golden_tolerance: float = 2e-2


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", force=True)
    policy = build_policy(args.checkpoint_dir, args.asset_id or None)
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
        metadata={"model": "pi05_peel", "action_horizon": 16, "tasks": list(TASKS)},
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
