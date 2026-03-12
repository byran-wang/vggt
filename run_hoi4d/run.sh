#!/usr/bin/env bash
# Run BundleSDF on HOI4D sequences via run_ho3d.py + Hoi4dReader
#
# Usage:
#   bash run_hoi4d/run.sh
#   bash run_hoi4d/run.sh /path/to/seq_list.txt
#
# Sequences in the txt file should be in HOI4D relative path format:
#   ZY20210800004/H4/C1/N50/S6/s02/T1
#
# GPU allocation:
#   Automatically queries available GPUs (up to 8) and distributes
#   sequences evenly across them in parallel.

set -uo pipefail

PROCESSED_DIR="/mnt/hdd_volume/datasets/HOI4D_ori/HOI4D_processed"
OUT_DIR="output_hoi4d"
SEQ_TXT="${1:-$(dirname "$0")/release.txt}"
MAX_GPUS=8

if [[ ! -f "$SEQ_TXT" ]]; then
  echo "[ERROR] Sequence list not found: $SEQ_TXT"
  exit 1
fi

# ── 1. Query available GPU IDs ───────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo "[ERROR] nvidia-smi not found. Cannot query GPUs."
  exit 1
fi

ALL_GPUS=()
while IFS= read -r gpu_id; do
  ALL_GPUS+=("$gpu_id")
done < <(nvidia-smi --query-gpu=index --format=csv,noheader | head -n "$MAX_GPUS")

NUM_GPUS=${#ALL_GPUS[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
  echo "[ERROR] No GPUs detected."
  exit 1
fi

echo "[INFO] Detected ${NUM_GPUS} GPU(s): ${ALL_GPUS[*]}"

# ── 2. Collect valid video directories ──────────────────────────────────────
VIDEO_LIST=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="${line//[[:space:]]/}"
  [[ -z "$line" ]] && continue

  seq_name="${line//\//_}"
  video_dir="${PROCESSED_DIR}/train/${seq_name}"

  if [[ ! -d "$video_dir" ]]; then
    echo "[WARN] not found, skip: $video_dir"
    continue
  fi

  VIDEO_LIST+=("$video_dir")
done < "$SEQ_TXT"

NUM_SEQS=${#VIDEO_LIST[@]}
if [[ $NUM_SEQS -eq 0 ]]; then
  echo "[ERROR] No valid video dirs found."
  exit 1
fi

echo "[INFO] Total sequences: ${NUM_SEQS}"
echo "[INFO] Distributing across ${NUM_GPUS} GPU(s)..."

# ── 3. Distribute sequences round-robin across GPUs ─────────────────────────
declare -a GPU_SEQ_LISTS
for (( i=0; i<NUM_GPUS; i++ )); do
  GPU_SEQ_LISTS[i]=""
done

for (( s=0; s<NUM_SEQS; s++ )); do
  gpu_idx=$(( s % NUM_GPUS ))
  if [[ -z "${GPU_SEQ_LISTS[$gpu_idx]}" ]]; then
    GPU_SEQ_LISTS[$gpu_idx]="${VIDEO_LIST[$s]}"
  else
    GPU_SEQ_LISTS[$gpu_idx]="${GPU_SEQ_LISTS[$gpu_idx]}|${VIDEO_LIST[$s]}"
  fi
done

# ── 4. Per-sequence result checker ───────────────────────────────────────────
# Prints a one-line summary for a completed sequence.
# Args: gpu_id seq_name exit_code elapsed_seconds
_report_seq() {
  local gpu_id="$1"
  local seq_name="$2"
  local exit_code="$3"
  local elapsed="$4"

  # Tracking: count pose files
  local n_poses
  n_poses=$(ls "${OUT_DIR}/${seq_name}/ob_in_cam/"*.txt 2>/dev/null | wc -l)

  # Global NeRF: check for textured mesh
  local mesh_file="${OUT_DIR}/${seq_name}/textured_mesh.obj"
  local nerf_status
  if [[ -f "$mesh_file" ]]; then
    local mesh_size
    mesh_size=$(du -sh "$mesh_file" 2>/dev/null | cut -f1)
    nerf_status="OK (${mesh_size})"
  else
    nerf_status="MISSING"
  fi

  local py_status
  if [[ "$exit_code" -eq 0 ]]; then
    py_status="OK"
  else
    py_status="FAILED(exit=${exit_code})"
  fi

  printf "[GPU %s] %-50s | python=%-16s | poses=%-4s | nerf_mesh=%-14s | time=%ds\n" \
    "$gpu_id" "$seq_name" "$py_status" "$n_poses" "$nerf_status" "$elapsed"
}

# ── 5. Launch one worker per GPU in parallel ─────────────────────────────────
mkdir -p "$OUT_DIR"
PIDS=()

for (( i=0; i<NUM_GPUS; i++ )); do
  gpu_id="${ALL_GPUS[$i]}"
  seqs_str="${GPU_SEQ_LISTS[$i]}"
  [[ -z "$seqs_str" ]] && continue

  log_file="${OUT_DIR}/gpu${gpu_id}.log"

  # Count sequences for this GPU
  IFS='|' read -ra seqs_arr <<< "$seqs_str"
  echo "[INFO] GPU ${gpu_id} → ${#seqs_arr[@]} seq(s), log: ${log_file}"

  (
    # Each GPU worker runs its sequences one at a time so we can report after each
    IFS='|' read -ra seqs_arr <<< "$seqs_str"
    for video_dir in "${seqs_arr[@]}"; do
      seq_name=$(basename "$video_dir")
      echo "[GPU ${gpu_id}] START ${seq_name}" | tee -a "$log_file"

      t_start=$(date +%s)
      set +e
      CUDA_VISIBLE_DEVICES="$gpu_id" python run_ho3d.py \
        --video_dirs "$video_dir" \
        --out_dir    "$OUT_DIR" \
        >>"$log_file" 2>&1
      exit_code=$?
      set -e
      t_end=$(date +%s)

      _report_seq "$gpu_id" "$seq_name" "$exit_code" "$(( t_end - t_start ))" | tee -a "$log_file"
    done
  ) &

  PIDS+=($!)
done

echo "[INFO] Launched ${#PIDS[@]} worker(s). Waiting for completion..."
echo ""

# ── 6. Wait and collect failures ─────────────────────────────────────────────
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$(( FAILED + 1 ))
  fi
done

# ── 7. Final summary across all sequences ────────────────────────────────────
echo ""
echo "════════════════════════════════════════ FINAL SUMMARY ════════════════════════════════════════"
printf "%-50s | %-6s | %-4s | %-14s\n" "SEQUENCE" "POSES" "MESH" "MESH_SIZE"
echo "───────────────────────────────────────────────────────────────────────────────────────────────"

total=0; ok_track=0; ok_nerf=0
for video_dir in "${VIDEO_LIST[@]}"; do
  seq_name=$(basename "$video_dir")
  total=$(( total + 1 ))

  n_poses=$(ls "${OUT_DIR}/${seq_name}/ob_in_cam/"*.txt 2>/dev/null | wc -l)
  mesh_file="${OUT_DIR}/${seq_name}/textured_mesh.obj"

  if [[ "$n_poses" -gt 0 ]]; then
    ok_track=$(( ok_track + 1 ))
    track_str="${n_poses}"
  else
    track_str="0 (FAIL)"
  fi

  if [[ -f "$mesh_file" ]]; then
    ok_nerf=$(( ok_nerf + 1 ))
    mesh_size=$(du -sh "$mesh_file" 2>/dev/null | cut -f1)
    nerf_str="OK"
  else
    mesh_size="-"
    nerf_str="MISSING"
  fi

  printf "%-50s | %-6s | %-4s | %-14s\n" "$seq_name" "$track_str" "$nerf_str" "$mesh_size"
done

echo "───────────────────────────────────────────────────────────────────────────────────────────────"
echo "Tracking OK: ${ok_track}/${total}   |   Global NeRF OK: ${ok_nerf}/${total}"
echo "═══════════════════════════════════════════════════════════════════════════════════════════════"

if [[ $FAILED -ne 0 ]]; then
  echo "[ERROR] ${FAILED} GPU worker(s) exited with non-zero status. Check logs in ${OUT_DIR}/gpu*.log"
  exit 1
fi
