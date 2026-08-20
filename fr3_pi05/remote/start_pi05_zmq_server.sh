#!/usr/bin/env bash
# Start pi05_droid with the FR3 direct-ZMQ wrapper in the foreground.
set -eu

OPENPI_DIR=${1:-${OPENPI_DIR:-}}
FR3_DEMO_DIR=${2:-${FR3_DEMO_DIR:-}}
CHECKPOINT=${3:-${CHECKPOINT:-}}
CONFIG_NAME=${4:-${CONFIG_NAME:-pi05_droid}}
PORT=${PORT:-8000}

if [ -z "$OPENPI_DIR" ] || [ -z "$FR3_DEMO_DIR" ] || [ -z "$CHECKPOINT" ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<A6000-index> PORT=<port> $0 /path/to/openpi /path/to/fr3_demo /local/checkpoint [config-name]" >&2
  exit 2
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "Set CUDA_VISIBLE_DEVICES to the free A6000 GPU index shown by nvidia-smi." >&2
  exit 2
fi
if [ ! -f "$OPENPI_DIR/scripts/serve_policy.py" ]; then
  echo "Not an OpenPI checkout: $OPENPI_DIR" >&2
  exit 2
fi
if [ ! -f "$FR3_DEMO_DIR/fr3_pi05/remote/serve_pi05_zmq.py" ]; then
  echo "Not an fr3_demo checkout: $FR3_DEMO_DIR" >&2
  exit 2
fi
if [ ! -d "$CHECKPOINT" ]; then
  echo "Local checkpoint does not exist: $CHECKPOINT" >&2
  echo "This launcher never downloads checkpoints." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH." >&2
  exit 2
fi

cd "$OPENPI_DIR"
export PYTHONPATH="$FR3_DEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
MODEL_ROOT=$(dirname "$CHECKPOINT")
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$MODEL_ROOT/.openpi_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$MODEL_ROOT/.uv_cache}
exec uv run --with pyzmq python \
  "$FR3_DEMO_DIR/fr3_pi05/remote/serve_pi05_zmq.py" \
  --port "$PORT" \
  --config-name "$CONFIG_NAME" \
  --checkpoint "$CHECKPOINT"
