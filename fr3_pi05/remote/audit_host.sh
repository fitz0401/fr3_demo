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
df -hT / "$HOME" /mnt/data 2>/dev/null | awk '!seen[$1]++'

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
for root in "$HOME" /home /data /mnt /opt; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 7 -type f -path '*/scripts/serve_policy.py' -print 2>/dev/null || true
  fi
done

echo "== candidate OpenPI revisions and DROID configs =="
for root in "$HOME" /home /data /mnt /opt; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 7 -type f -path '*/scripts/serve_policy.py' -print 2>/dev/null || true
  fi
done | while IFS= read -r serve_script; do
  repository=${serve_script%/scripts/serve_policy.py}
  echo "repository: $repository"
  git -C "$repository" remote -v 2>/dev/null | head -n 4 || true
  git -C "$repository" status --short --branch 2>/dev/null | head -n 20 || true
  grep -nE 'name=.*(droid|DROID|custom)' "$repository/src/openpi/training/config.py" 2>/dev/null | head -n 40 || true
done

echo "== candidate pi05_droid caches/checkpoints =="
for root in "$HOME/.cache" "$HOME" /data /mnt; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 8 \( -iname '*pi05*droid*' -o -path '*/checkpoints/pi05_droid' \) -print 2>/dev/null || true
  fi
done

echo "== candidate checkpoint structure and sizes =="
for root in "$HOME/.cache" "$HOME" /data /mnt; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 8 -type d -iname '*pi05*droid*' -print0 2>/dev/null || true
  fi
done | while IFS= read -r -d '' checkpoint; do
  du -sh "$checkpoint" 2>/dev/null || true
  find "$checkpoint" -maxdepth 3 -type f -printf '  %P\n' 2>/dev/null | head -n 40
done

echo "== ports 8000-8001 / server processes =="
ss -ltnp 2>/dev/null | grep -E ':8000|:8001' || true
ps -u "$(id -u)" -o pid,etime,args | grep -E 'serve_policy|openpi' | grep -v grep || true

echo "== OpenPI process working directories =="
for process in /proc/[0-9]*; do
  pid=${process##*/}
  if [ -r "$process/cmdline" ] && tr '\0' ' ' <"$process/cmdline" 2>/dev/null | grep -qE 'serve_policy|openpi'; then
    printf '%s  ' "$pid"
    readlink -f "$process/cwd" 2>/dev/null || echo "cwd unavailable"
  fi
done
