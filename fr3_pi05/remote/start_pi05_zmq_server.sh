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
  echo "Usage: CUDA_VISIBLE_DEVICES=<A6000-index> PORT=<port> $0 /path/to/openpi /path/to/fr3_demo /local/checkpoint [pi05_droid|custom_droid|wine_hybrid|bluecup_force|bc2f_force]" >&2
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

CUSTOM_ASSET_ID=${ASSET_ID:-}
PROPRIO_HISTORY_OFFSETS=(0)
WINE_LOADER_PATH=
BLUECUP_LOADER_PATH=
BC2F_LOADER_PATH=
ACTION_EXPERT_VARIANT=${ACTION_EXPERT_VARIANT:-auto}
TASKS_JSON=${TASKS_JSON:-}
if [ -z "${USE_EXTERNAL2+x}" ]; then
  CONFIG_PYTHON=$OPENPI_DIR/.venv/bin/python
  if [ ! -x "$CONFIG_PYTHON" ]; then
    CONFIG_PYTHON=$(command -v python3)
  fi
  USE_EXTERNAL2=$(
    "$CONFIG_PYTHON" - "$FR3_DEMO_DIR/config.toml" <<'PY'
import pathlib
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

path = pathlib.Path(sys.argv[1])
with path.open("rb") as stream:
    config = tomllib.load(stream)
print("1" if config.get("pi05", {}).get("use_external2", False) else "0")
PY
  )
fi
case "$MODEL_PROFILE" in
  pi05_droid)
    LOADER=official
    CONFIG_NAME=pi05_droid
    ;;
  custom_droid)
    LOADER=custom_droid
    CONFIG_NAME=pi05_droid
    CUSTOM_ASSET_ID=${CUSTOM_ASSET_ID:-fitz0401/custom_droid}
    ;;
  wine_hybrid)
    LOADER=wine
    CONFIG_NAME=pi05_droid
    PROPRIO_HISTORY_OFFSETS=(0 45 75)
    WINE_LOADER_PATH=$FR3_DEMO_DIR/fr3_train/serve_wine.py
    ;;
  bluecup_force)
    # Blue-cup force / sensor-dropout checkpoints (fr3_train/bluecup/DEPLOY_PLAN.md). Three cameras
    # and the measured joint torque are always required; the client never toggles them here.
    LOADER=bluecup
    CONFIG_NAME=pi05_droid
    PROPRIO_HISTORY_OFFSETS=(0 45 75)
    BLUECUP_LOADER_PATH=$FR3_DEMO_DIR/fr3_train/serve_bluecup.py
    ;;
  bc2f_force)
    # Scope-v5 blue-cup checkpoints (fr3_train/bc2f/DEPLOY_PLAN.md): three cameras, proprio history and the raw
    # ~1 kHz arm stream (force window computed on this machine with the bundle's force_model.json).
    LOADER=bc2f
    CONFIG_NAME=pi05_droid
    PROPRIO_HISTORY_OFFSETS=(0 45 75)
    BC2F_LOADER_PATH=$FR3_DEMO_DIR/fr3_train/serve_bc2f.py
    ;;
  *)
    echo "Unknown model profile: $MODEL_PROFILE (expected pi05_droid, custom_droid, wine_hybrid, bluecup_force, or bc2f_force)" >&2
    exit 2
    ;;
esac
if { [ "$LOADER" = wine ] || [ "$LOADER" = bluecup ] || [ "$LOADER" = bc2f ]; } && [ -z "$CUSTOM_ASSET_ID" ]; then
  mapfile -t NORM_FILES < <(find "$CHECKPOINT/assets" -type f -name norm_stats.json 2>/dev/null | sort)
  if [ "${#NORM_FILES[@]}" -ne 1 ]; then
    echo "Expected exactly one normalization file under $CHECKPOINT/assets; found ${#NORM_FILES[@]}." >&2
    echo "Set ASSET_ID=owner/dataset to select one explicitly." >&2
    exit 2
  fi
  CUSTOM_ASSET_ID=${NORM_FILES[0]#"$CHECKPOINT/assets/"}
  CUSTOM_ASSET_ID=${CUSTOM_ASSET_ID%/norm_stats.json}
  echo "Auto-detected normalization asset: $CUSTOM_ASSET_ID"
fi
if [ "$LOADER" = custom_droid ]; then
  if [ ! -f "$CHECKPOINT/serve_custom_droid.py" ]; then
    echo "Custom loader is missing: $CHECKPOINT/serve_custom_droid.py" >&2
    exit 2
  fi
fi
if [ "$LOADER" = wine ] && [ ! -f "$WINE_LOADER_PATH" ]; then
  echo "Wine loader is missing: $WINE_LOADER_PATH" >&2
  echo "Pull the fr3_demo revision containing fr3_train before launching." >&2
  exit 2
fi
if [ "$LOADER" = bluecup ] && [ ! -f "$BLUECUP_LOADER_PATH" ]; then
  echo "Blue-cup loader is missing: $BLUECUP_LOADER_PATH" >&2
  echo "Pull the fr3_demo branch containing fr3_train/serve_bluecup.py before launching." >&2
  exit 2
fi
if [ "$LOADER" = bc2f ] && [ ! -f "$BC2F_LOADER_PATH" ]; then
  echo "bc2f loader is missing: $BC2F_LOADER_PATH" >&2
  echo "Pull the fr3_demo branch containing fr3_train/serve_bc2f.py before launching." >&2
  exit 2
fi
if [ "$LOADER" = bc2f ] && [ ! -f "$CHECKPOINT/force_model.json" ]; then
  echo "$CHECKPOINT/force_model.json is missing: copy the whole bundle directory from Leonardo." >&2
  exit 2
fi
if [ "$LOADER" = wine ] || [ "$LOADER" = bluecup ] || [ "$LOADER" = bc2f ]; then
  EXPECTED_OPENPI_REV=15a9616a00943ada6c20a0f158e3adb39df2ccac
  ACTUAL_OPENPI_REV=$(git -C "$OPENPI_DIR" rev-parse HEAD 2>/dev/null || true)
  if [ "$ACTUAL_OPENPI_REV" != "$EXPECTED_OPENPI_REV" ]; then
    echo "$MODEL_PROFILE requires OpenPI $EXPECTED_OPENPI_REV" >&2
    echo "Current checkout is ${ACTUAL_OPENPI_REV:-not a Git checkout}." >&2
    echo "Use a dedicated checkout at the training revision and apply fr3_train/deploy.patch." >&2
    exit 2
  fi
fi
if [ -n "$CUSTOM_ASSET_ID" ]; then
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
  if ! grep -q 'lora \* self.lora_config.scaling_value' "$OPENPI_DIR/src/openpi/models/lora.py" 2>/dev/null; then
    echo "The mandatory FFN LoRA scaling fix is missing from this OpenPI checkout." >&2
    echo "Apply fr3_train/deploy.patch before launching the checkpoint." >&2
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
if [ -n "$BC2F_LOADER_PATH" ]; then
  EXTRA_ARGS+=(
    --bc2f-loader-path "$BC2F_LOADER_PATH"
    --asset-id "$CUSTOM_ASSET_ID"
  )
fi
if [ -n "$BLUECUP_LOADER_PATH" ]; then
  EXTRA_ARGS+=(
    --bluecup-loader-path "$BLUECUP_LOADER_PATH"
    --asset-id "$CUSTOM_ASSET_ID"
  )
fi
if [ -n "$WINE_LOADER_PATH" ]; then
  EXTRA_ARGS+=(
    --wine-loader-path "$WINE_LOADER_PATH"
    --action-expert-variant "$ACTION_EXPERT_VARIANT"
    --asset-id "$CUSTOM_ASSET_ID"
  )
  if [ -n "$TASKS_JSON" ]; then
    EXTRA_ARGS+=(--tasks-json "$TASKS_JSON")
  fi
  case "$USE_EXTERNAL2" in
    1|true|TRUE|yes|YES) EXTRA_ARGS+=(--use-exterior2) ;;
    0|false|FALSE|no|NO|'') ;;
    *)
      echo "USE_EXTERNAL2 must be 0/1, false/true, or no/yes." >&2
      exit 2
      ;;
  esac
fi
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
  --proprio-history-offsets "${PROPRIO_HISTORY_OFFSETS[@]}" \
  "${EXTRA_ARGS[@]}"
