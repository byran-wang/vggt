#!/bin/bash
# Per-worker blender rendering for a given seq list. Mirrors the structure
# of /tmp/run_blender_local.sh but runs locally on a worker (l1/l2/l3).
#
# Pre-conditions per seq on this worker (DATA_DIR defaults to round 01):
#   ~/rhoi/output/<seq>/pipeline_joint_opt/
#   ~/rhoi/output/<seq>/pipeline_neus_global/
#   $DATA_DIR/<seq>/SAM3D_aligned_post_process/
#   $DATA_DIR/<seq>/SAM3D_align_filter/best_id.txt
#   $DATA_DIR/<seq>/meta/<cond>.pkl  (camMat for K)
#   $DATA_DIR/<seq>/rgb/             (for image_resolution + overlay)
#
# Worker env: ~/miniconda3/envs/blender (rsync'd via sync_blender_env.sh)
#
# Usage: bash worker_blender_render.sh <seq...>
#
# Optional env knobs (passed through to pipeline_blender_rendering.py):
#   FPS, MESH_TYPE, HAND_MODE, LIGHT_ANGLE, LIGHT_STRENGTH, LIGHT_SIZE,
#   AMBIENT_COLOR (3 floats space-separated), IMG_RES (2 ints "W H"),
#   NUM_SAMPLES, OBJ_RGB ("r g b"), HAND_RGB ("r g b")

set -u

SEQS="$*"
[ -z "$SEQS" ] && { echo "Usage: $0 <seq...>"; exit 2; }

ROUND="${ROUND:-01}"
DATA_DIR="${DATA_DIR:-$HOME/data/rhoi_zed/$ROUND}"
PY="$HOME/miniconda3/envs/blender/bin/python"
SCRIPT="$HOME/rhoi/robust_hoi_pipeline/pipeline_blender_rendering.py"
LOG_DIR="${LOG_DIR:-$HOME/data/rhoi_zed/blender_logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# pull blender env's ffmpeg into PATH (env was rsync'd, so it's local)
export PATH="$HOME/miniconda3/envs/blender/bin:$PATH"

cd "$HOME/rhoi" || { echo "ERROR: ~/rhoi not found"; exit 98; }

# CLI knobs. Light/material params are NOT defaulted here — when an env var is
# unset we omit the flag entirely so pipeline_blender_rendering.py's own
# defaults (the GUI-tuned values baked into the script) take effect.
FPS="${FPS:-6}"
MESH_TYPE="${MESH_TYPE:-neus}"
HAND_MODE="${HAND_MODE:-ho}"
VIS_GT="${VIS_GT:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
LIGHT_ANGLE="${LIGHT_ANGLE:-}"
LIGHT_STRENGTH="${LIGHT_STRENGTH:-}"
LIGHT_SIZE="${LIGHT_SIZE:-}"
AMBIENT_COLOR="${AMBIENT_COLOR:-}"
IMG_RES="${IMG_RES:-}"
OBJ_RGB="${OBJ_RGB:-}"
HAND_RGB="${HAND_RGB:-}"

START=$(date +%s)
N_OK=0; N_SKIP=0; N_FAIL=0
echo "[$(date -Iseconds)] START host=$(hostname) seqs=$SEQS"

for seq in $SEQS; do
  result_dir="$HOME/rhoi/output/$seq/pipeline_joint_opt"
  sam3d_dir="$DATA_DIR/$seq/SAM3D_aligned_post_process"
  best_id_path="$DATA_DIR/$seq/SAM3D_align_filter/best_id.txt"
  neus_dir="$HOME/rhoi/output/$seq/pipeline_neus_global"
  out_dir="$HOME/rhoi/output/$seq/blender_rendering"

  if [ -f "$out_dir/hand_object_mesh.mp4" ]; then
    echo "  $seq: SKIP (mp4 exists)"
    N_SKIP=$((N_SKIP+1)); continue
  fi
  if [ ! -d "$result_dir" ] || [ ! -d "$sam3d_dir" ] || [ ! -f "$best_id_path" ] || [ ! -d "$neus_dir" ]; then
    echo "  $seq: SKIP (missing prerequisites)"
    N_SKIP=$((N_SKIP+1)); continue
  fi
  cond_index=$(cat "$best_id_path")
  log="$LOG_DIR/${seq}.log"
  t0=$(date +%s)
  echo "  $seq: rendering (cond=$cond_index) -> $log"

  EXTRA_ARGS=()
  [ -n "$IMG_RES" ]         && EXTRA_ARGS+=(--image_resolution $IMG_RES)
  [ -n "$OBJ_RGB" ]         && EXTRA_ARGS+=(--obj_mesh_RGB $OBJ_RGB)
  [ -n "$HAND_RGB" ]        && EXTRA_ARGS+=(--hand_mesh_RGB $HAND_RGB)
  [ -n "$LIGHT_ANGLE" ]     && EXTRA_ARGS+=(--light_angle "$LIGHT_ANGLE")
  [ -n "$LIGHT_STRENGTH" ]  && EXTRA_ARGS+=(--light_strength "$LIGHT_STRENGTH")
  [ -n "$LIGHT_SIZE" ]      && EXTRA_ARGS+=(--light_size "$LIGHT_SIZE")
  [ -n "$AMBIENT_COLOR" ]   && EXTRA_ARGS+=(--ambient_color $AMBIENT_COLOR)

  DATASET_DIR="$DATA_DIR" "$PY" "$SCRIPT" \
    --result_folder "$result_dir/" \
    --SAM3D_dir "$sam3d_dir" \
    --cond_index "$cond_index" \
    --out_dir "$out_dir" \
    --mesh_type "$MESH_TYPE" \
    --hand_mode "$HAND_MODE" \
    --vis_gt "$VIS_GT" \
    --fps "$FPS" \
    --number_of_samples "$NUM_SAMPLES" \
    "${EXTRA_ARGS[@]}" \
    > "$log" 2>&1
  rc=$?
  dt=$(($(date +%s)-t0))
  if [ $rc -eq 0 ] && [ -f "$out_dir/hand_object_mesh.mp4" ]; then
    echo "    DONE dt=${dt}s -> $out_dir/hand_object_mesh.mp4"
    N_OK=$((N_OK+1))
  else
    echo "    FAIL rc=$rc dt=${dt}s (see $log)"
    N_FAIL=$((N_FAIL+1))
  fi
done

TOTAL=$(($(date +%s)-START))
echo ""
echo "[$(date -Iseconds)] DONE host=$(hostname) ok=${N_OK} skip=${N_SKIP} fail=${N_FAIL} dt=${TOTAL}s"
echo "Logs: $LOG_DIR"
