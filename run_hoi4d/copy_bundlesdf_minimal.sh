#!/usr/bin/env bash
set -euo pipefail

# Minimal BundleSDF result copy script.
# Copies only:
#   - ob_in_cam/
#   - cam_K.txt
#   - textured_mesh.obj
#   - material.mtl
#   - material_0.png
#
# Remote layout:
#   /data1/shibo/Documents/project/BundleSDF/output/<SEQUENCE>/
#
# Local layout (preserved):
#   ./output/<SEQUENCE>/

REMOTE_USER_HOST="shibo@10.30.47.2"
REMOTE_OUTPUT_ROOT="/data1/shibo/Documents/project/BundleSDF/output_hoi4d"
LOCAL_OUTPUT_ROOT="./output_hoi4d"

# Read sequences from a .txt file (one sequence name per line).
# Empty lines and lines starting with # are ignored.
# Full remote path = REMOTE_OUTPUT_ROOT / <sequence name>
#
# Example sequences.txt:
#   ZY20210800004_H4_C1_N50_S240_s03_T2
#   ZY20210800001_H1_C3_N12_S60_s01_T1
SEQUENCES_FILE="/home/simba-1/Documents/project/BundleSDF/run_hoi4d/processed_sequences3.txt"

if [[ ! -f "${SEQUENCES_FILE}" ]]; then
  echo "Sequences file not found: ${SEQUENCES_FILE}"
  exit 1
fi

mapfile -t SEQUENCES < <(grep -v '^\s*#' "${SEQUENCES_FILE}" | grep -v '^\s*$')

if [[ ${#SEQUENCES[@]} -eq 0 ]]; then
  echo "No sequences found in ${SEQUENCES_FILE}"
  exit 1
fi

echo "Found ${#SEQUENCES[@]} sequences to copy."

mkdir -p "${LOCAL_OUTPUT_ROOT}"

# Reuse a single SSH connection so password is requested only once.
CONTROL_PATH="${TMPDIR:-/tmp}/bundlesdf-copy-%r@%h:%p"
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

for seq in "${SEQUENCES[@]}"; do
  echo "=== Copying ${seq} ==="
  remote_seq_dir="${REMOTE_OUTPUT_ROOT}/${seq}"
  local_seq_dir="${LOCAL_OUTPUT_ROOT}/${seq}"
  mkdir -p "${local_seq_dir}"

  # 1) ob_in_cam directory
  rsync -az --info=progress2 -e "${SSH_RSYNC_CMD}" \
    "${REMOTE_USER_HOST}:${remote_seq_dir}/ob_in_cam" \
    "${local_seq_dir}/"

  # 2) cam_K.txt
  rsync -az --info=progress2 -e "${SSH_RSYNC_CMD}" \
    "${REMOTE_USER_HOST}:${remote_seq_dir}/cam_K.txt" \
    "${local_seq_dir}/"

  # 3) textured_mesh.obj
  rsync -az --info=progress2 -e "${SSH_RSYNC_CMD}" \
    "${REMOTE_USER_HOST}:${remote_seq_dir}/textured_mesh.obj" \
    "${local_seq_dir}/"

  # 4) material.mtl
  rsync -az --info=progress2 -e "${SSH_RSYNC_CMD}" \
    "${REMOTE_USER_HOST}:${remote_seq_dir}/material.mtl" \
    "${local_seq_dir}/"

  # 5) material_0.png
  rsync -az --info=progress2 -e "${SSH_RSYNC_CMD}" \
    "${REMOTE_USER_HOST}:${remote_seq_dir}/material_0.png" \
    "${local_seq_dir}/"
done

echo "Done. Copied minimal files to ${LOCAL_OUTPUT_ROOT}/<sequence>/"
