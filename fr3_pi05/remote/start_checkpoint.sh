#!/usr/bin/env bash
# Start a known pi0.5 contract with either its default or a new checkpoint snapshot.
set -eu

usage() {
  cat >&2 <<'EOF'
Usage: start_checkpoint.sh [PROFILE] [CHECKPOINT]
       start_checkpoint.sh CHECKPOINT

Profiles:
  pi05_droid   official DROID contract (port 8000)
  custom_droid custom DROID contract   (port 8001)
  wine_hybrid  FR3 wine contract       (port 8002)
  bluecup_force FR3 blue-cup force/dropout contract (port 8003)
  bc2f_force   FR3 blue-cup scope-v5 contract: raw 1 kHz force stream (port 8004)

Optional environment overrides:
  CUDA_VISIBLE_DEVICES, FR3_GPU_INDEX, OPENPI_DIR, FR3_DEMO_DIR, PORT,
  ASSET_ID, TASKS_JSON, ACTION_EXPERT_VARIANT
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi
if [ "$#" -gt 2 ] || [ -z "${1:-}" ]; then
  usage
  exit 2
fi

if [ "$#" -eq 1 ]; then
  case "$1" in
    pi05_droid|custom_droid|wine_hybrid|bluecup_force|bc2f_force)
      PROFILE=$1
      CHECKPOINT=
      ;;
    *)
      PROFILE=wine_hybrid
      CHECKPOINT=$1
      ;;
  esac
else
  PROFILE=$1
  CHECKPOINT=$2
fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FR3_DEMO_DIR=${FR3_DEMO_DIR:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}

case "$PROFILE" in
  pi05_droid)
    PORT=${PORT:-8000}
    OPENPI_DIR=${OPENPI_DIR:-/mnt/data/yurui/openpi}
    CHECKPOINT=${CHECKPOINT:-/mnt/data/yurui/.cache/openpi/openpi-assets/checkpoints/pi05_droid}
    ;;
  custom_droid)
    PORT=${PORT:-8001}
    OPENPI_DIR=${OPENPI_DIR:-/mnt/data/yurui/openpi}
    CHECKPOINT=${CHECKPOINT:-/mnt/data/yurui/models/pi05_custom_droid_14999}
    ;;
  wine_hybrid)
    PORT=${PORT:-8002}
    OPENPI_DIR=${OPENPI_DIR:-$HOME/openpi_wine}
    CHECKPOINT=${CHECKPOINT:-/mnt/data/yurui/models/pi05_wine_hybrid_17500}
    ;;
  bluecup_force)
    PORT=${PORT:-8003}
    # Same checkout as wine: OpenPI 15a9616 + fr3_train/deploy.patch.
    OPENPI_DIR=${OPENPI_DIR:-$HOME/openpi_wine}
    # One of pi05_bluecup_{flag_10500,moddrop_14999,alloc_fixed_14500}; restart to switch models.
    CHECKPOINT=${CHECKPOINT:-/mnt/data/yurui/models/pi05_bluecup_alloc_fixed_14500}
    ;;
  bc2f_force)
    PORT=${PORT:-8004}
    # Same checkout as wine: OpenPI 15a9616 + fr3_train/deploy.patch.
    OPENPI_DIR=${OPENPI_DIR:-$HOME/openpi_wine}
    # One of pi05_bc2f_{vanilla,moddrop,moddrop_prop}_s0_14999; restart to switch models.
    CHECKPOINT=${CHECKPOINT:-/mnt/data/yurui/models/pi05_bc2f_moddrop_s0_14999}
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${FR3_GPU_INDEX:-1}}
export PORT OPENPI_DIR FR3_DEMO_DIR CHECKPOINT

echo "Starting profile=$PROFILE checkpoint=$CHECKPOINT port=$PORT GPU=$CUDA_VISIBLE_DEVICES"
exec bash "$SCRIPT_DIR/start_pi05_zmq_server.sh" \
  "$OPENPI_DIR" \
  "$FR3_DEMO_DIR" \
  "$CHECKPOINT" \
  "$PROFILE"
