#!/usr/bin/env bash
# Start pi05_droid with the FR3 direct-ZMQ wrapper in the foreground.
set -eu

OPENPI_DIR=${1:-${OPENPI_DIR:-}}
FR3_DEMO_DIR=${2:-${FR3_DEMO_DIR:-}}
CHECKPOINT=${3:-${CHECKPOINT:-}}
MODEL_PROFILE=${4:-${MODEL_PROFILE:-pi05_droid}}
PORT=${PORT:-8000}
CONNECT_ENDPOINT=${CONNECT_ENDPOINT:-}

if [ -z "$OPENPI_DIR" ] || [ -z "$FR3_DEMO_DIR" ] || [ -z "$CHECKPOINT" ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<A6000-index> PORT=<port> $0 /path/to/openpi /path/to/fr3_demo /local/checkpoint [pi05_droid|custom_droid|wine_hybrid]" >&2
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

CUSTOM_ASSET_ID=
case "$MODEL_PROFILE" in
  pi05_droid)
    LOADER=official
    CONFIG_NAME=pi05_droid
    ;;
  custom_droid)
    LOADER=custom_droid
    CONFIG_NAME=pi05_droid
    CUSTOM_ASSET_ID=fitz0401/custom_droid
    ;;
  wine_hybrid)
    LOADER=custom_droid
    CONFIG_NAME=pi05_droid
    CUSTOM_ASSET_ID=fitz0401/franka_pour_wine
    ;;
  *)
    echo "Unknown model profile: $MODEL_PROFILE (expected pi05_droid, custom_droid, or wine_hybrid)" >&2
    exit 2
    ;;
esac
if [ "$LOADER" = custom_droid ]; then
  if [ ! -f "$CHECKPOINT/serve_custom_droid.py" ]; then
    echo "Custom loader is missing: $CHECKPOINT/serve_custom_droid.py" >&2
    exit 2
  fi
  if [ ! -f "$CHECKPOINT/assets/$CUSTOM_ASSET_ID/norm_stats.json" ]; then
    echo "Normalization statistics are missing: $CHECKPOINT/assets/$CUSTOM_ASSET_ID/norm_stats.json" >&2
    echo "Refusing to substitute stock DROID or another task's statistics." >&2
    exit 2
  fi
  if ! grep -q 'gemma_2b_lora_r32' "$OPENPI_DIR/src/openpi/models/gemma.py" 2>/dev/null; then
    echo "The custom checkpoint deploy.patch is not applied to this OpenPI checkout." >&2
    echo "Follow the custom checkpoint deployment instructions before launching." >&2
    exit 2
  fi
fi

cd "$OPENPI_DIR"
export PYTHONPATH="$FR3_DEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CACHE_ROOT=${XDG_CACHE_HOME:-$HOME/.cache}/fr3_pi05
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$CACHE_ROOT/openpi}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
mkdir -p "$OPENPI_DATA_HOME" "$UV_CACHE_DIR"
if [ ! -w "$OPENPI_DATA_HOME" ] || [ ! -w "$UV_CACHE_DIR" ]; then
  echo "Runtime cache is not writable; set OPENPI_DATA_HOME and UV_CACHE_DIR to writable paths." >&2
  exit 2
fi
EXTRA_ARGS=()
if [ -n "$CONNECT_ENDPOINT" ]; then
  case "$CONNECT_ENDPOINT" in
    tcp://*) ;;
    *)
      echo "CONNECT_ENDPOINT must start with tcp://" >&2
      exit 2
      ;;
  esac
  EXTRA_ARGS+=(--connect-endpoint "$CONNECT_ENDPOINT")
fi
OPENPI_PYTHON="$OPENPI_DIR/.venv/bin/python"
if [ -x "$OPENPI_PYTHON" ] && "$OPENPI_PYTHON" -c 'import msgpack, zmq' >/dev/null 2>&1; then
  PYTHON_COMMAND=("$OPENPI_PYTHON")
else
  echo "OpenPI environment lacks msgpack/pyzmq; using an isolated uv overlay." >&2
  PYTHON_COMMAND=(uv run --with pyzmq python)
fi
exec "${PYTHON_COMMAND[@]}" \
  "$FR3_DEMO_DIR/fr3_pi05/remote/serve_pi05_zmq.py" \
  --port "$PORT" \
  --loader "$LOADER" \
  --config-name "$CONFIG_NAME" \
  --checkpoint "$CHECKPOINT" \
  "${EXTRA_ARGS[@]}"
