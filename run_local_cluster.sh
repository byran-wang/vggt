#!/bin/bash
# Local multi-worker orchestrator. Dispatches the 5-stage rhoi pipeline across
# SSH-reachable workers (e.g. t1 t2 t3 t4). Each worker runs
# scripts/worker_pipeline.sh on its round-robin shard of the seq list.
#
# Uses the SAME round-robin algorithm as upload_data_local_cluster.sh, so if
# both are called with matching WORKERS + SEQS, each worker already has exactly
# the data it needs. This script verifies the per-shard data is present before
# launching.
#
# Assumptions about each worker (built by the bootstrap tarball + rsync code):
#   - ~/miniconda3/envs/{rhoi,sam3,sam3d-objects,hamer,vggsfm_tmp->rhoi}/
#   - ~/rhoi/ (code + third_party weights + body_models)
#   - ~/data/rhoi_zed/01/<seq>/ (rgb/ir/meta/sam3_prompts)  <-- upload via
#     upload_data_local_cluster.sh with the same WORKERS + SEQS
#   - ~/Documents/project/vggt -> ~/rhoi (legacy symlink)
#
# Usage:
#   bash run_local_cluster.sh                                # WORKERS=l1 l2 l3, ROUND=01, SEQS=all
#   ROUND=02 bash run_local_cluster.sh                       # round 02
#   WORKERS="l1 l2" SEQS="001 003 010" bash run_local_cluster.sh
#   WORKERS="l1 l2 l3" bash run_local_cluster.sh
#   SKIP_DATA_CHECK=1 ...                                    # bypass per-seq data presence check

set -u

ROUND="${ROUND:-01}"
WORKERS="${WORKERS:-l1 l2 l3}"
SEQS_INPUT="${SEQS:-}"
DATASET_DIR_LOCAL="${DATASET_DIR_LOCAL:-$HOME/data/rhoi_zed/$ROUND}"
DATASET_DIR_REMOTE_REL="${DATASET_DIR_REMOTE_REL:-data/rhoi_zed/$ROUND}"  # relative to remote $HOME
LOG_DIR="${LOG_DIR:-$HOME/data/rhoi_zed/logs_${ROUND}_$(date +%Y%m%d_%H%M%S)}"

# If SEQS not set, enumerate every seq that has rgb/ locally.
if [ -z "$SEQS_INPUT" ]; then
  SEQS_INPUT=$(ls -d "$DATASET_DIR_LOCAL"/*/rgb 2>/dev/null \
    | awk -F/ '{print $(NF-1)}' | sort)
fi
SEQ_LIST=($SEQS_INPUT)
N_SEQ=${#SEQ_LIST[@]}
[ $N_SEQ -eq 0 ] && { echo "ERROR: no seqs to dispatch"; exit 1; }

mkdir -p "$LOG_DIR"

# --- readiness check --------------------------------------------------------
declare -a READY=()
echo "[$(date -Iseconds)] checking workers: $WORKERS"
for h in $WORKERS; do
  if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" \
       "test -x ~/miniconda3/envs/rhoi/bin/python && \
        test -d ~/rhoi/third_party/sam3/sam3/model/checkpoints" \
       >/dev/null 2>&1; then
    READY+=("$h")
    echo "  $h: READY"
  else
    echo "  $h: NOT READY (skipped)"
  fi
done
N_READY=${#READY[@]}
[ $N_READY -eq 0 ] && { echo "ERROR: no ready workers"; exit 2; }

# --- round-robin shard seqs to workers (identical to upload_data_local_cluster.sh) -
declare -A SHARD
for i in "${!SEQ_LIST[@]}"; do
  h="${READY[$((i % N_READY))]}"
  SHARD[$h]="${SHARD[$h]:-}${SEQ_LIST[$i]} "
done

# --- verify each worker has its shard data (rgb/ folder per seq) ------------
SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-0}"
if [ "$SKIP_DATA_CHECK" != "1" ]; then
  DATA_ERR=0
  for h in "${READY[@]}"; do
    h_seqs="${SHARD[$h]:-}"
    [ -z "$h_seqs" ] && continue
    missing=$(ssh "$h" "for s in $h_seqs; do \
        [ -d ~/$DATASET_DIR_REMOTE_REL/\$s/rgb ] || echo \$s; done")
    if [ -n "$missing" ]; then
      echo "  $h: MISSING seqs (no rgb/): $missing"
      DATA_ERR=1
    fi
  done
  [ $DATA_ERR -eq 0 ] || { echo "ERROR: some workers missing shard data. Run upload_data_local_cluster.sh first, or set SKIP_DATA_CHECK=1"; exit 3; }
fi

# --- sync worker_pipeline.sh to each ready worker ---------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
for h in "${READY[@]}"; do
  ssh "$h" "mkdir -p ~/rhoi/scripts" 2>/dev/null
  scp -q "$HERE/scripts/worker_pipeline.sh" "$h:~/rhoi/scripts/worker_pipeline.sh"
done

# --- launch each worker (background nohup ssh) ------------------------------
echo ""
echo "[$(date -Iseconds)] dispatching $N_SEQ seqs across $N_READY workers"
echo "[$(date -Iseconds)] LOG_DIR=$LOG_DIR"
for h in "${READY[@]}"; do
  seqs="${SHARD[$h]:-}"
  n=$(echo $seqs | wc -w)
  if [ "$n" -eq 0 ]; then
    echo "  $h: 0 seqs (skipped)"
    continue
  fi
  echo "  $h: $n seqs = $seqs"
  LOG="$LOG_DIR/${h}.log"
  ( ssh "$h" "DATASET_DIR=\$HOME/$DATASET_DIR_REMOTE_REL nohup bash ~/rhoi/scripts/worker_pipeline.sh $seqs > ~/pipeline_run_${ROUND}.log 2>&1 & disown; echo ${h}-pid=\$!" \
      > "$LOG" 2>&1 ) &
done
wait
echo ""
echo "[$(date -Iseconds)] all workers dispatched. Logs: $LOG_DIR/*.log (local)"
echo "  per-worker remote log: ~/pipeline_run.log"
echo ""
echo "Tail progress:"
echo "  for h in ${READY[*]}; do echo == \$h ==; ssh \$h 'tail -5 ~/pipeline_run_${ROUND}.log'; done"
