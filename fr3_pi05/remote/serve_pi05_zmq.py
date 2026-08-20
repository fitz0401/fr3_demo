"""Serve the official OpenPI pi05_droid policy over a direct ZMQ channel."""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from typing import Any

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


def load_policy(config_name: str, checkpoint: str, default_prompt: str | None) -> Any:
    """Load OpenPI lazily so protocol tests do not need the GPU stack."""

    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as error:
        raise RuntimeError("Run this script from the official OpenPI uv environment") from error
    return policy_config.create_trained_policy(
        training_config.get_config(config_name), checkpoint, default_prompt=default_prompt
    )


def serve(policy: Any, metadata: dict[str, Any], endpoint: str) -> None:
    try:
        import zmq
    except ImportError as error:
        raise RuntimeError("pyzmq is missing from the OpenPI environment") from error
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
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
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_droid")
    parser.add_argument("--default-prompt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = load_policy(args.config_name, args.checkpoint, args.default_prompt)
    metadata = getattr(policy, "metadata", {}) or {}
    serve(policy, metadata, f"tcp://{args.bind_host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
