#!/bin/bash
# Replicate the local blender conda env to each worker (l1 l2 l3 by default).
# Conda envs are path-sensitive: source and dest paths must match exactly,
# which they do here (both /home/vox/miniconda3/envs/blender).
#
# Also rsyncs:
#   - the patched blendertoolbox files (already inside the env)
#   - the project source so worker_blender_render.sh can call the script
#     (the project sync is delta-only — fine to run repeatedly)
#
# Usage:
#   bash sync_blender_env.sh                  # WORKERS=l1 l2 l3
#   WORKERS="l1 l2" bash sync_blender_env.sh
#   FORCE=1 ...                               # rsync even if env exists on worker

set -u

WORKERS="${WORKERS:-l1 l2 l3}"
FORCE="${FORCE:-0}"
LOCAL_ENV="$HOME/miniconda3/envs/blender"
REMOTE_ENV="\$HOME/miniconda3/envs/blender"  # expanded on remote
LOCAL_PROJ="$HOME/rhoi"

[ -d "$LOCAL_ENV" ] || { echo "ERROR: $LOCAL_ENV does not exist"; exit 1; }

echo "[$(date -Iseconds)] syncing blender env to: $WORKERS"
for h in $WORKERS; do
  echo ""
  echo "=== $h ==="
  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" true 2>/dev/null; then
    echo "  $h: SSH unreachable, skipping"
    continue
  fi

  # check if env already present
  exists=$(ssh "$h" "test -x $REMOTE_ENV/bin/python && echo yes || echo no")
  if [ "$exists" = "yes" ] && [ "$FORCE" != "1" ]; then
    echo "  $h: env already present (FORCE=1 to overwrite); skipping rsync"
  else
    echo "  $h: rsync env (~2.7G)..."
    rsync -a --info=progress2 --delete \
          --exclude='__pycache__' --exclude='*.pyc' \
          "$LOCAL_ENV/" "$h:$LOCAL_ENV/"
  fi

  # sync project source (delta only, fast)
  echo "  $h: rsync project source (delta)..."
  rsync -a --delete \
        --include='/robust_hoi_pipeline/***' \
        --include='/third_party/utils_simba/***' \
        --include='/third_party/BlenderToolbox/***' \
        --include='/scripts/***' \
        --include='/confs/***' \
        --include='/run_wonder_hoi.py' \
        --exclude='*' \
        "$LOCAL_PROJ/" "$h:$LOCAL_PROJ/"

  # smoke-test bpy import
  echo "  $h: smoke-testing bpy import..."
  ssh "$h" "$REMOTE_ENV/bin/python -c 'import bpy; print(\"bpy version:\", bpy.app.version_string)'" \
    || { echo "  $h: bpy import FAILED"; continue; }

  echo "  $h: OK"
done

echo ""
echo "[$(date -Iseconds)] done."
