#!/usr/bin/env bash
# Prepare the user-owned OpenPI checkout required by the wine profile.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FR3_DEMO_DIR=${FR3_DEMO_DIR:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}
SOURCE_OPENPI=${SOURCE_OPENPI:-/mnt/data/yurui/openpi}
WINE_OPENPI=${WINE_OPENPI:-$HOME/openpi_wine}
EXPECTED_REV=15a9616a00943ada6c20a0f158e3adb39df2ccac
DEPLOY_PATCH=$FR3_DEMO_DIR/fr3_train/deploy.patch

if [ ! -f "$DEPLOY_PATCH" ]; then
  echo "Deployment patch is missing: $DEPLOY_PATCH" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH." >&2
  exit 2
fi

if [ ! -e "$WINE_OPENPI" ]; then
  if [ ! -f "$SOURCE_OPENPI/scripts/serve_policy.py" ]; then
    echo "Source is not an OpenPI checkout: $SOURCE_OPENPI" >&2
    exit 2
  fi
  if ! git config --global --get-all safe.directory 2>/dev/null | grep -Fxq "$SOURCE_OPENPI"; then
    git config --global --add safe.directory "$SOURCE_OPENPI"
  fi
  git clone --no-hardlinks "$SOURCE_OPENPI" "$WINE_OPENPI"
  git -C "$WINE_OPENPI" checkout --detach "$EXPECTED_REV"
elif [ ! -f "$WINE_OPENPI/scripts/serve_policy.py" ]; then
  echo "Refusing to replace existing non-OpenPI path: $WINE_OPENPI" >&2
  exit 2
fi

ACTUAL_REV=$(git -C "$WINE_OPENPI" rev-parse HEAD 2>/dev/null || true)
if [ "$ACTUAL_REV" != "$EXPECTED_REV" ]; then
  echo "Existing wine checkout has revision ${ACTUAL_REV:-unknown}; expected $EXPECTED_REV." >&2
  echo "Choose an empty WINE_OPENPI path instead of modifying it implicitly." >&2
  exit 2
fi

if git -C "$WINE_OPENPI" apply --reverse --check "$DEPLOY_PATCH" >/dev/null 2>&1; then
  echo "Wine deployment patch is already applied."
elif git -C "$WINE_OPENPI" apply --check "$DEPLOY_PATCH" >/dev/null 2>&1; then
  git -C "$WINE_OPENPI" apply "$DEPLOY_PATCH"
else
  echo "Deployment patch is neither cleanly applicable nor already applied in $WINE_OPENPI." >&2
  exit 2
fi

(cd "$WINE_OPENPI" && uv sync)
echo "Wine OpenPI environment ready: $WINE_OPENPI"
