"""Stand-alone policy server for the pi05_wine_hybrid checkpoint.

Self-contained: the TrainConfig is built inline, so the deployment machine does not need the
patched training config from the research workspace. It does need the `gemma_2b_lora_r32` variant,
which is not upstream -- apply `deploy.patch` first.

    uv run examples/wine/serve_wine.py --checkpoint-dir /path/to/ckpt --self-test
    uv run examples/wine/serve_wine.py --checkpoint-dir /path/to/ckpt --port 8000

Two things changed versus the custom_droid server, and both will break a client that assumes the
old contract:

1. **The state is three stacked frames, not one.** The model reads proprioception at t, t-45 and
   t-75 (3 s and 5 s at 15 fps) so that it can tell "carrying the bottle to the glass" from
   "carrying it back" -- pi0.5 has no memory of its own. The client keeps the buffer and sends
   `observation/joint_position` with shape (3, 7) and `observation/gripper_position` with shape
   (3,) or (3, 1), **current first**. Before 75 frames have elapsed, repeat the current frame; that
   is what the training pipeline does at the start of an episode.

2. **The prompt is mandatory.** This model was trained on two tasks that share a scene and differ
   only in the instruction. A default would silently pour the wrong bottle, so a request without a
   prompt is rejected rather than guessed.
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


@dataclasses.dataclass(frozen=True)
class DroidInputsWithStateHistory(_transforms.DataTransformFn):
    """Assemble the stacked state this checkpoint expects. Defined here, not imported, so that this
    file runs against a stock openpi checkout plus `deploy.patch` and nothing else.

    Layout is current-first: [q_t, grip_t, q_{t-lag0}, grip_{t-lag0}, ...]. Index 0 of the stacked
    axis is the present; treating index 0 as a lag would feed the model a stale state that still
    trains and still runs, just wrongly.
    """

    model_type: _model.ModelType
    num_state_frames: int = 1

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
        base_image = droid_policy._parse_image(data["observation/exterior_image_1_left"])
        wrist_image = droid_policy._parse_image(data["observation/wrist_image_left"])
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class _WineDataConfig(_config.DataConfigFactory):
    """The inference half of the training data config, rebuilt from upstream pieces only."""

    num_state_frames: int = 3

    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/exterior_image_1_left": "exterior_image_1_left",
                            "observation/wrist_image_left": "wrist_image_left",
                            "observation/joint_position": "joint_position",
                            "observation/gripper_position": "gripper_position",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            data_transforms=_transforms.Group(
                inputs=[DroidInputsWithStateHistory(model_config.model_type, self.num_state_frames)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory()(model_config),
        )

# Must match training exactly: the LoRA variants decide whether the params load at all, and
# discrete_state_input decides how the prompt (and therefore the state) is built.
def _model(action_expert_variant: str) -> pi0_config.Pi0Config:
    """The training recipe. Everything here must match the checkpoint or the params will not load.

    `action_expert_variant` is the one field that differs between the two released checkpoints:
    `gemma_300m_lora` for the LoRA run and `gemma_300m` for the fully-trained expert. The parameter
    trees differ structurally (71 vs 61 groups -- the LoRA layers simply do not exist in the second),
    so the wrong value raises at load time rather than degrading quietly.
    """
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora_r32",
        action_expert_variant=action_expert_variant,
    )
STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 1 + len(STATE_HISTORY_LAGS)
TASKS = ("pour lillet into the jigger", "pour gin into the jigger")


@dataclasses.dataclass
class Args:
    # Directory holding `params/` and `assets/`.
    checkpoint_dir: str = "."
    # Interface to bind the websocket server to.
    host: str = "0.0.0.0"
    # Port to bind the websocket server to.
    port: int = 8000
    # Run a few dummy inferences and exit, instead of serving.
    self_test: bool = False
    # "gemma_300m_lora" for the LoRA action expert, "gemma_300m" for the fully-trained one.
    # See the DEPLOY.md shipped alongside the checkpoint.
    action_expert_variant: str = "gemma_300m_lora"


def build_policy(checkpoint_dir: str, action_expert_variant: str = "gemma_300m_lora"):
    config = _config.TrainConfig(
        name="pi05_wine_hybrid",
        exp_name="deploy",
        model=_model(action_expert_variant),
        # repo_id only supplies the asset_id under which the norm stats sit inside the checkpoint's
        # `assets/` directory; the dataset itself is not needed at inference.
        data=_WineDataConfig(
            repo_id="fitz0401/franka_pour_wine",
            num_state_frames=NUM_STATE_FRAMES,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
    )
    # No default_prompt on purpose: see the module docstring.
    return _policy_config.create_trained_policy(config, checkpoint_dir)


def self_test(policy) -> None:
    def example(task: str) -> dict:
        rng = np.random.default_rng(0)
        return {
            "observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
            # Current first, then t-45 and t-75. Here they are identical, which is what a client
            # should send during the first 75 frames after startup.
            "observation/joint_position": np.tile(rng.uniform(-1, 1, 7), (NUM_STATE_FRAMES, 1)),
            "observation/gripper_position": np.full((NUM_STATE_FRAMES,), 0.9),
            "prompt": task,
        }

    logging.info(f"state contract: {NUM_STATE_FRAMES} stacked frames, lags {STATE_HISTORY_LAGS} (current first)")
    logging.info("first call includes JIT compilation and is much slower")
    for task in TASKS:
        for i in range(2):
            t0 = time.monotonic()
            actions = policy.infer(example(task))["actions"]
            logging.info(f"{task!r} call {i}: {(time.monotonic() - t0) * 1000:8.1f} ms  {actions.shape} {actions.dtype}")

    # A missing prompt must fail loudly rather than default to one of the two tasks.
    bad = example(TASKS[0])
    del bad["prompt"]
    try:
        policy.infer(bad)
    except Exception as e:  # noqa: BLE001 - we only care that it refuses
        logging.info(f"request without a prompt correctly rejected: {type(e).__name__}: {e}")
    else:
        logging.error("A request without a prompt was ACCEPTED -- the server would guess the task.")

    # A single (unstacked) state must also fail, rather than silently mean something else.
    stale = example(TASKS[0])
    stale["observation/joint_position"] = stale["observation/joint_position"][0]
    stale["observation/gripper_position"] = stale["observation/gripper_position"][:1]
    try:
        policy.infer(stale)
    except ValueError as e:
        logging.info(f"single-frame state correctly rejected: {e}")
    else:
        logging.error("A single-frame state was ACCEPTED -- history would be silently wrong.")

    logging.info("self-test OK")


def main(args: Args) -> None:
    # force=True: importing jax/absl installs a root handler above INFO, which makes a plain
    # basicConfig() a silent no-op and swallows every logging call in this script.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", force=True
    )
    policy = build_policy(args.checkpoint_dir, args.action_expert_variant)
    if args.self_test:
        self_test(policy)
        return
    logging.info(
        f"serving on {args.host}:{args.port}; action expert {args.action_expert_variant}; "
        f"prompt is mandatory, one of {TASKS}"
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "model": "pi05_wine_hybrid",
            "action_horizon": _model(args.action_expert_variant).action_horizon,
            "state_history_lags": list(STATE_HISTORY_LAGS),
            "num_state_frames": NUM_STATE_FRAMES,
            "tasks": list(TASKS),
        },
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
