#!/usr/bin/env bash
# Read-only inventory for the pi0.5 inference host.
set -eu

echo "== identity =="
hostname
id
uname -a

echo "== GPUs =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
else
  echo "nvidia-smi: not found"
fi

echo "== storage =="
df -h "$HOME"

echo "== tools =="
for tool in git uv conda python3 tmux; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%s: ' "$tool"
    command -v "$tool"
  else
    echo "$tool: not found"
  fi
done

echo "== candidate OpenPI trees =="
for root in "$HOME" /data /mnt /opt; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 4 -type f -path '*/scripts/serve_policy.py' -print 2>/dev/null || true
  fi
done

echo "== candidate pi05_droid caches/checkpoints =="
for root in "$HOME/.cache" "$HOME" /data /mnt; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 6 \( -iname '*pi05*droid*' -o -path '*/checkpoints/pi05_droid' \) -print 2>/dev/null || true
  fi
done

echo "== port 8000 / server processes =="
ss -ltnp 2>/dev/null | grep ':8000' || true
ps -u "$(id -u)" -o pid,etime,args | grep -E 'serve_policy|openpi' | grep -v grep || true
