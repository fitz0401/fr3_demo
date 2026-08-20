#!/usr/bin/env bash
# Forward a local port to the OpenPI server when lab firewall rules block port 8000.
set -eu

GPU_HOST=${GPU_HOST:-10.38.32.253}
LOCAL_PORT=${LOCAL_PORT:-8000}
REMOTE_PORT=${REMOTE_PORT:-8000}

echo "Forwarding 127.0.0.1:${LOCAL_PORT} to ${GPU_HOST}:127.0.0.1:${REMOTE_PORT}."
echo "Keep this terminal open; press Ctrl+C to close the tunnel."
exec ssh -o ExitOnForwardFailure=yes -NT \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "$GPU_HOST"
