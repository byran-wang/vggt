#!/bin/bash
# Local orchestrator: dispatches blender rendering across SSH-reachable workers
# (l1 l2 l3 by default), round-robin sharding the seq list. Each worker runs
# scripts/worker_blender_render.sh on its shard.
#
# Pre-conditions:
#   1. Workers have ~/miniconda3/envs/blender/ (run: bash scripts/sync_blender_env.sh)
#   2. Workers already have pipeline_joint_opt/, pipeline_neus_global/ outputs +
#      ~/data/rhoi_zed/01/<seq>/{rgb,meta,SAM3D_aligned_post_process,SAM3D_align_filter}
#      (these exist from the joint_opt + neus_global cluster passes)
#
# Usage:
#   bash run_blender_cluster.sh                                  # WORKERS=l1 l2 l3, SEQS=all
#   WORKERS="l1 l2" SEQS="001 003 010" bash run_blender_cluster.sh
#   LIGHT_STRENGTH=3.0 LIGHT_ANGLE="(0,-45,-90)" bash run_blender_cluster.sh
#
# Shared env knobs (forwarded to worker_blender_render.sh):
#   FPS, MESH_TYPE, HAND_MODE, LIGHT_ANGLE, LIGHT_STRENGTH, LIGHT_SIZE,
#   AMBIENT_COLOR, NUM_SAMPLES, IMG_RES, OBJ_RGB, HAND_RGB

set -u

WORKERS="${WORKERS:-l1 l2 l3}"
SEQS_INPUT="${SEQS:-}"
DATASET_DIR_LOCAL="${DATASET_DIR_LOCAL:-$HOME/data/rhoi_zed/01}"
LOG_DIR="${LOG_DIR:-$HOME/data/rhoi_zed/blender_cluster_logs_$(date +%Y%m%d_%H%M%S)}"

if [ -z "$SEQS_INPUT" ]; then
  SEQS_INPUT=$(ls -d "$DATASET_DIR_LOCAL"/*/rgb 2>/dev/null \
    | awk -F/ '{print $(NF-1)}' | sort)
fi
SEQ_LIST=($SEQS_INPUT)
N_SEQ=${#SEQ_LIST[@]}
[ $N_SEQ -eq 0 ] && { echo "ERROR: no seqs to dispatch"; exit 1; }

mkdir -p "$LOG_DIR"

# --- readiness check -------------------------------------------------------
declare -a READY=()
echo "[$(date -Iseconds)] checking workers: $WORKERS"
for h in $WORKERS; do
  if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" \
       "test -x ~/miniconda3/envs/blender/bin/python && \
        test -f ~/rhoi/robust_hoi_pipeline/pipeline_blender_rendering.py" \
       >/dev/null 2>&1; then
    READY+=("$h")
    echo "  $h: READY"
  else
    echo "  $h: NOT READY (run scripts/sync_blender_env.sh first)"
  fi
done
N_READY=${#READY[@]}
[ $N_READY -eq 0 ] && { echo "ERROR: no ready workers"; exit 2; }

# --- round-robin shard -----------------------------------------------------
declare -A SHARD
for i in "${!SEQ_LIST[@]}"; do
  h="${READY[$((i % N_READY))]}"
  SHARD[$h]="${SHARD[$h]:-}${SEQ_LIST[$i]} "
done

# --- forward worker_blender_render.sh + pipeline_blender_rendering.py ----
HERE="$(cd "$(dirname "$0")" && pwd)"
for h in "${READY[@]}"; do
  ssh "$h" "mkdir -p ~/rhoi/scripts ~/rhoi/robust_hoi_pipeline" 2>/dev/null
  scp -q "$HERE/scripts/worker_blender_render.sh" "$h:~/rhoi/scripts/worker_blender_render.sh"
  scp -q "$HERE/robust_hoi_pipeline/pipeline_blender_rendering.py" \
         "$h:~/rhoi/robust_hoi_pipeline/pipeline_blender_rendering.py"
  scp -q "$HERE/robust_hoi_pipeline/compose_blender_rgb_overlay.py" \
         "$h:~/rhoi/robust_hoi_pipeline/compose_blender_rgb_overlay.py"
done

# --- forward env knobs ----------------------------------------------------
ENV_PREFIX=""
for v in FPS MESH_TYPE HAND_MODE LIGHT_ANGLE LIGHT_STRENGTH LIGHT_SIZE \
         AMBIENT_COLOR NUM_SAMPLES IMG_RES OBJ_RGB HAND_RGB VIS_GT; do
  if [ -n "${!v:-}" ]; then
    val="${!v}"
    ENV_PREFIX+="$v=$(printf %q "$val") "
  fi
done

# --- launch each worker (background) --------------------------------------
echo ""
echo "[$(date -Iseconds)] dispatching $N_SEQ seqs across $N_READY workers"
echo "[$(date -Iseconds)] LOG_DIR=$LOG_DIR"
[ -n "$ENV_PREFIX" ] && echo "  ENV: $ENV_PREFIX"
for h in "${READY[@]}"; do
  seqs="${SHARD[$h]:-}"
  n=$(echo $seqs | wc -w)
  if [ "$n" -eq 0 ]; then
    echo "  $h: 0 seqs (skipped)"; continue
  fi
  echo "  $h: $n seqs = $seqs"
  LOG="$LOG_DIR/${h}.log"
  ( ssh "$h" "$ENV_PREFIX nohup bash ~/rhoi/scripts/worker_blender_render.sh $seqs > ~/blender_run.log 2>&1 & disown; echo ${h}-pid=\$!" \
      > "$LOG" 2>&1 ) &
done
wait
echo ""
echo "[$(date -Iseconds)] all workers dispatched. Local logs: $LOG_DIR/*.log"
echo "Per-worker remote log: ~/blender_run.log"
echo ""
echo "Tail progress:"
echo "  for h in ${READY[*]}; do echo == \$h ==; ssh \$h 'tail -10 ~/blender_run.log'; done"
