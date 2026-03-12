#!/usr/bin/env bash
set -euo pipefail

# Copy HOI4D_processed from remote to local.
# Symlinks inside HOI4D_processed point to HOI4D_ori; -L makes rsync follow
# them and copy the actual file/directory content instead of the symlinks.

REMOTE_USER_HOST="shibo@10.30.47.2"
REMOTE_SRC="/mnt/hdd_volume/datasets/HOI4D_ori/HOI4D_processed"
LOCAL_DST="/home/simba-1/Documents/dataset"

CONTROL_PATH="${TMPDIR:-/tmp}/hoi4d-copy-%r@%h:%p"
SSH_BASE_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="${CONTROL_PATH}"
)

echo "Opening SSH control connection to ${REMOTE_USER_HOST} (password once)..."
ssh "${SSH_BASE_OPTS[@]}" -fN "${REMOTE_USER_HOST}"

cleanup() {
  ssh "${SSH_BASE_OPTS[@]}" -O exit "${REMOTE_USER_HOST}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

SSH_RSYNC_CMD="ssh ${SSH_BASE_OPTS[*]}"

mkdir -p "${LOCAL_DST}"

echo "Syncing ${REMOTE_USER_HOST}:${REMOTE_SRC} → ${LOCAL_DST}/ ..."
rsync -aL --no-compress --info=progress2 -e "${SSH_RSYNC_CMD}" \
  "${REMOTE_USER_HOST}:${REMOTE_SRC}" \
  "${LOCAL_DST}/"

echo "Done. HOI4D_processed synced to ${LOCAL_DST}/HOI4D_processed/"
