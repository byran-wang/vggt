#!/bin/bash
# Sync local code to remote cluster via rsync.
# Usage: ./sync_to_remote.sh [remote_path]
# Default: ht-dc:/mnt/afs/xinyuan/code/robust_hoi/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${1:-ht-dc:/mnt/afs/xinyuan/code/robust_hoi/}"

rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.note' \
  --exclude='output' \
  --exclude='ho3d_v3' \
  --exclude='*.egg-info' \
  --exclude='*.tar.gz' \
  --exclude='_DATA/' \
  --exclude='.local_preprocess_logs' \
  "$SCRIPT_DIR/" "$REMOTE"
