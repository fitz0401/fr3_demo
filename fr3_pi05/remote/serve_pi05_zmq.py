"""Serve the official OpenPI pi05_droid policy over a direct ZMQ channel."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from fr3_pi05.protocol import packb, unpackb

LOG = logging.getLogger("pi05_zmq_server")


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


def warm_up(policy: Any, prompt: str) -> float:
    """Compile one inference before exposing the network endpoint."""

    observation = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.array(
            [-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031],
            dtype=np.float32,
        ),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": prompt,
    }
    started = time.monotonic()
    result = policy.infer(observation)
    actions = np.asarray(result.get("actions")) if isinstance(result, dict) else np.empty(0)
    if actions.ndim != 2 or actions.shape[1:] != (8,) or not np.all(np.isfinite(actions)):
        raise RuntimeError(f"Policy warm-up returned invalid actions shaped {actions.shape}")
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
    parser.add_argument("--loader", choices=("official", "custom_droid"), default="official")
    parser.add_argument("--checkpoint", required=True, help="existing local checkpoint directory; never downloaded here")
    parser.add_argument("--default-prompt")
    parser.add_argument("--no-warmup", action="store_true", help="skip the startup JIT inference")
    parser.add_argument(
        "--connect-endpoint",
        help="connect the REP socket outward (for example tcp://10.34.97.197:8000) instead of binding",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy, loader_metadata = load_policy(
        args.loader,
        args.config_name,
        args.checkpoint,
        args.default_prompt,
    )
    metadata = dict(getattr(policy, "metadata", {}) or {})
    metadata.update(loader_metadata)
    if not args.no_warmup:
        prompt = args.default_prompt or loader_metadata.get("default_prompt") or "perform the task"
        LOG.info("warming policy before opening the ZMQ endpoint")
        warmup_ms = warm_up(policy, str(prompt))
        metadata["warmup_ms"] = warmup_ms
        LOG.info("policy warm-up complete in %.1f ms", warmup_ms)
    metadata.update(
        {
            "transport": "zmq",
            "loader": args.loader,
            "config_name": args.config_name,
            "checkpoint": args.checkpoint,
        }
    )
    endpoint = args.connect_endpoint or f"tcp://{args.bind_host}:{args.port}"
    serve(policy, metadata, endpoint, connect=args.connect_endpoint is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
