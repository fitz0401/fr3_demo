#!/usr/bin/env bash
# Start the official OpenPI pi05_droid WebSocket server in the foreground.
set -eu

OPENPI_DIR=${1:-${OPENPI_DIR:-}}
PORT=${PORT:-8000}

if [ -z "$OPENPI_DIR" ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<A6000-index> $0 /absolute/path/to/openpi" >&2
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
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH." >&2
  exit 2
fi

cd "$OPENPI_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec uv run scripts/serve_policy.py --env DROID --port "$PORT"
