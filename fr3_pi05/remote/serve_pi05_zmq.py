"""Serve the official OpenPI pi05_droid policy over a direct ZMQ channel."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from fr3_pi05.protocol import packb, unpackb

LOG = logging.getLogger("pi05_zmq_server")


def _build_wine_policy(
    module: Any,
    checkpoint: str,
    action_expert_variant: str,
    asset_id: str,
    use_exterior2: bool,
) -> Any:
    """Compose the deployment data contract without changing training-owned code."""

    required = (
        "_WineDataConfig",
        "_config",
        "_model",
        "_policy_config",
        "_transforms",
        "DroidInputsWithStateHistory",
        "droid_policy",
    )
    if not all(hasattr(module, name) for name in required):
        # Lightweight compatibility path used by protocol tests and older standalone loaders.
        try:
            return module.build_policy(checkpoint, action_expert_variant, asset_id, use_exterior2)
        except TypeError:
            return module.build_policy(checkpoint, action_expert_variant, asset_id)

    data_factory: Any
    if use_exterior2:

        @dataclasses.dataclass(frozen=True)
        class Exterior2Inputs(module.DroidInputsWithStateHistory):
            def __call__(self, data: dict) -> dict:
                inputs = super().__call__(data)
                inputs["image"]["right_wrist_0_rgb"] = module.droid_policy._parse_image(
                    data["observation/exterior_image_2_left"]
                )
                inputs["image_mask"]["right_wrist_0_rgb"] = np.True_
                return inputs

        @dataclasses.dataclass(frozen=True)
        class Exterior2WineDataConfig(module._WineDataConfig):
            def create(self, assets_dirs, model_config):
                config = super().create(assets_dirs, model_config)
                return dataclasses.replace(
                    config,
                    repack_transforms=module._transforms.Group(
                        inputs=[
                            module._transforms.RepackTransform(
                                {
                                    "observation/exterior_image_1_left": "exterior_image_1_left",
                                    "observation/exterior_image_2_left": "exterior_image_2_left",
                                    "observation/wrist_image_left": "wrist_image_left",
                                    "observation/joint_position": "joint_position",
                                    "observation/gripper_position": "gripper_position",
                                    "prompt": "prompt",
                                }
                            )
                        ]
                    ),
                    data_transforms=module._transforms.Group(
                        inputs=[Exterior2Inputs(model_config.model_type, self.num_state_frames)],
                        outputs=[module.droid_policy.DroidOutputs()],
                    ),
                )

        data_factory = Exterior2WineDataConfig
    else:
        data_factory = module._WineDataConfig

    model_config = module._model(action_expert_variant)
    config = module._config.TrainConfig(
        name="pi05_wine_hybrid",
        exp_name="deploy",
        model=model_config,
        data=data_factory(
            repo_id=asset_id,
            num_state_frames=int(module.NUM_STATE_FRAMES),
            base_config=module._config.DataConfig(prompt_from_task=True),
        ),
    )
    return module._policy_config.create_trained_policy(config, checkpoint)


def handle_request(policy: Any, metadata: dict[str, Any], request: Any) -> dict[str, Any]:
    """Handle one decoded request; kept independent for protocol testing."""

    if not isinstance(request, dict):
        return {"success": False, "error": "Request must be a dictionary"}
    operation = request.get("operation")
    if operation == "metadata":
        return {"success": True, "metadata": metadata}
    if operation != "infer":
        return {"success": False, "error": f"Unknown operation: {operation!r}"}
    observation = request.get("observation")
    if not isinstance(observation, dict):
        return {"success": False, "error": "Inference request has no observation dictionary"}
    tasks = metadata.get("tasks")
    if tasks and observation.get("prompt") not in tasks:
        return {
            "success": False,
            "error": f"Prompt must exactly match one of the trained tasks: {tasks}",
        }
    started = time.monotonic()
    result = policy.infer(observation)
    if not isinstance(result, dict):
        return {"success": False, "error": "Policy returned a non-dictionary result"}
    result["server_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
    return {"success": True, "result": result}


def load_policy(
    loader: str,
    config_name: str,
    checkpoint: str,
    default_prompt: str | None,
    wine_loader_path: str | None = None,
    action_expert_variant: str = "gemma_300m_lora",
    asset_id: str | None = None,
    tasks: list[str] | None = None,
    use_exterior2: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load OpenPI lazily so protocol tests do not need the GPU stack."""

    if loader == "custom_droid":
        loader_path = Path(checkpoint) / "serve_custom_droid.py"
        if not loader_path.is_file():
            raise RuntimeError(f"Custom checkpoint loader is missing: {loader_path}")
        module_name = "fr3_pi05_deployed_custom_droid"
        spec = importlib.util.spec_from_file_location(module_name, loader_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import custom checkpoint loader: {loader_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        prompt = default_prompt or module.DEFAULT_PROMPT
        args = module.Args(checkpoint_dir=checkpoint, default_prompt=prompt)
        policy = module.build_policy(args)
        return policy, {
            "model": "pi05_custom_droid_hybrid",
            "action_horizon": int(module.MODEL.action_horizon),
            "default_prompt": prompt,
        }
    if loader == "wine":
        loader_path = (
            Path(wine_loader_path)
            if wine_loader_path is not None
            else Path(__file__).resolve().parents[2] / "fr3_train" / "serve_wine.py"
        )
        if not loader_path.is_file():
            raise RuntimeError(f"Wine checkpoint loader is missing: {loader_path}")
        module_name = "fr3_pi05_deployed_wine"
        spec = importlib.util.spec_from_file_location(module_name, loader_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import wine checkpoint loader: {loader_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        resolved_asset_id = asset_id or getattr(module, "DEFAULT_ASSET_ID", "fitz0401/franka_pour_wine")
        policy = _build_wine_policy(
            module,
            checkpoint,
            action_expert_variant,
            resolved_asset_id,
            use_exterior2,
        )
        history_lags = list(module.STATE_HISTORY_LAGS)
        num_frames = int(module.NUM_STATE_FRAMES)
        resolved_tasks = tasks or []
        return policy, {
            "model": "pi05_wine_hybrid",
            "action_horizon": int(module._model(action_expert_variant).action_horizon),
            "action_expert_variant": action_expert_variant,
            "state_history_lags": history_lags,
            "proprio_history_offsets": [0, *history_lags],
            "num_state_frames": num_frames,
            "joint_observation_shape": [num_frames, 7],
            "gripper_observation_shape": [num_frames],
            "image_observation_shape": [180, 320, 3],
            "joint_observation_dim": num_frames * 7,
            "gripper_observation_dim": num_frames,
            "asset_id": resolved_asset_id,
            "tasks": resolved_tasks,
            "uses_exterior2": use_exterior2,
        }
    if loader != "official":
        raise ValueError(f"Unknown policy loader: {loader}")
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as error:
        raise RuntimeError("Run this script from the official OpenPI uv environment") from error
    policy = policy_config.create_trained_policy(
        training_config.get_config(config_name), checkpoint, default_prompt=default_prompt
    )
    return policy, {"model": config_name}


def warm_up(
    policy: Any,
    prompt: str,
    proprio_history_offsets: tuple[int, ...] = (0,),
    action_horizon: int | None = None,
    use_exterior2: bool = False,
) -> float:
    """Compile one inference before exposing the network endpoint."""

    current_joints = np.array(
        [-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031],
        dtype=np.float32,
    )
    image_shape = (224, 224, 3) if len(proprio_history_offsets) == 1 else (180, 320, 3)
    observation = {
        "observation/exterior_image_1_left": np.zeros(image_shape, dtype=np.uint8),
        "observation/wrist_image_left": np.zeros(image_shape, dtype=np.uint8),
        "observation/joint_position": (
            current_joints
            if len(proprio_history_offsets) == 1
            else np.tile(current_joints, (len(proprio_history_offsets), 1))
        ),
        "observation/gripper_position": np.zeros(len(proprio_history_offsets), dtype=np.float32),
        "prompt": prompt,
    }
    if use_exterior2:
        observation["observation/exterior_image_2_left"] = np.zeros(image_shape, dtype=np.uint8)
    started = time.monotonic()
    result = policy.infer(observation)
    actions = np.asarray(result.get("actions")) if isinstance(result, dict) else np.empty(0)
    if actions.ndim != 2 or actions.shape[1:] != (8,) or not np.all(np.isfinite(actions)):
        raise RuntimeError(f"Policy warm-up returned invalid actions shaped {actions.shape}")
    if action_horizon is not None and actions.shape[0] != action_horizon:
        raise RuntimeError(
            f"Policy warm-up returned horizon {actions.shape[0]}, expected {action_horizon}"
        )
    return (time.monotonic() - started) * 1000.0


def serve(policy: Any, metadata: dict[str, Any], endpoint: str, *, connect: bool = False) -> None:
    try:
        import zmq
    except ImportError as error:
        raise RuntimeError("pyzmq is missing from the OpenPI environment") from error
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    if connect:
        socket.connect(endpoint)
        LOG.info("pi0.5 DROID ZMQ server connected outward to %s", endpoint)
    else:
        socket.bind(endpoint)
        LOG.info("pi0.5 DROID ZMQ server listening on %s", endpoint)
    try:
        while True:
            payload = socket.recv()
            try:
                response = handle_request(policy, metadata, unpackb(payload))
            except Exception:
                response = {"success": False, "error": traceback.format_exc()}
                LOG.exception("Inference request failed")
            socket.send(packb(response))
    finally:
        socket.close(linger=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve official pi05_droid inference over ZMQ.")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config-name", default="pi05_droid")
    parser.add_argument("--loader", choices=("official", "custom_droid", "wine"), default="official")
    parser.add_argument("--checkpoint", required=True, help="existing local checkpoint directory; never downloaded here")
    parser.add_argument("--default-prompt")
    parser.add_argument("--wine-loader-path")
    parser.add_argument("--asset-id", help="checkpoint normalization asset ID, for example owner/dataset")
    parser.add_argument("--tasks-json", help="optional JSON array restricting accepted language instructions")
    parser.add_argument(
        "--use-exterior2",
        action="store_true",
        help="feed exterior_image_2_left to a checkpoint trained with that camera",
    )
    parser.add_argument(
        "--action-expert-variant",
        choices=("gemma_300m_lora", "gemma_300m"),
        default="gemma_300m_lora",
    )
    parser.add_argument(
        "--proprio-history-offsets",
        type=int,
        nargs="+",
        default=(0,),
        metavar="FRAME",
        help="observation history in collector-rate frames, ordered current first",
    )
    parser.add_argument("--no-warmup", action="store_true", help="skip the startup JIT inference")
    parser.add_argument(
        "--connect-endpoint",
        help="connect the REP socket outward (for example tcp://10.34.97.197:8000) instead of binding",
    )
    args = parser.parse_args()
    tasks: list[str] | None = None
    if args.tasks_json is not None:
        try:
            decoded_tasks = json.loads(args.tasks_json)
        except json.JSONDecodeError as error:
            parser.error(f"--tasks-json is invalid JSON: {error}")
        if not isinstance(decoded_tasks, list) or not all(
            isinstance(task, str) and task.strip() for task in decoded_tasks
        ):
            parser.error("--tasks-json must be a JSON array of non-empty strings")
        tasks = [task.strip() for task in decoded_tasks]
    history_offsets = tuple(args.proprio_history_offsets)
    if not history_offsets or history_offsets[0] != 0 or any(offset < 0 for offset in history_offsets):
        parser.error("--proprio-history-offsets must start with 0 and contain non-negative frame offsets")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy, loader_metadata = load_policy(
        args.loader,
        args.config_name,
        args.checkpoint,
        args.default_prompt,
        args.wine_loader_path,
        args.action_expert_variant,
        args.asset_id,
        tasks,
        args.use_exterior2,
    )
    metadata = dict(getattr(policy, "metadata", {}) or {})
    metadata.update(loader_metadata)
    if not args.no_warmup:
        trained_tasks = loader_metadata.get("tasks") or []
        prompt = args.default_prompt or loader_metadata.get("default_prompt") or next(
            iter(trained_tasks), "perform the task"
        )
        LOG.info("warming policy before opening the ZMQ endpoint")
        warmup_ms = warm_up(
            policy,
            str(prompt),
            history_offsets,
            int(loader_metadata["action_horizon"]) if "action_horizon" in loader_metadata else None,
            bool(loader_metadata.get("uses_exterior2", False)),
        )
        metadata["warmup_ms"] = warmup_ms
        LOG.info("policy warm-up complete in %.1f ms", warmup_ms)
    metadata.update(
        {
            "transport": "zmq",
            "loader": args.loader,
            "config_name": args.config_name,
            "checkpoint": args.checkpoint,
            "joint_observation_dim": loader_metadata.get(
                "joint_observation_dim", 7 * len(history_offsets)
            ),
            "gripper_observation_dim": loader_metadata.get(
                "gripper_observation_dim", len(history_offsets)
            ),
            "proprio_history_offsets": loader_metadata.get(
                "proprio_history_offsets", list(history_offsets)
            ),
        }
    )
    endpoint = args.connect_endpoint or f"tcp://{args.bind_host}:{args.port}"
    serve(policy, metadata, endpoint, connect=args.connect_endpoint is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
