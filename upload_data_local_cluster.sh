#!/bin/bash
# Shard-aware data distribution for the local worker cluster.
#
# Each worker receives ONLY the seqs assigned to it by round-robin sharding
# (identical algorithm used by run_local_cluster.sh — so the seqs we upload
# here are exactly the ones that worker will process). Total network transfer
# = N_seqs copies (one per seq), not N_seqs x N_workers.
#
# Per-seq payload uploaded:  rgb/  ir/  meta/  sam3_prompts/
# Cluster-generated outputs are excluded (same exclusion set as upload_data_cluster.sh).
#
# Usage:
#   bash upload_data_local_cluster.sh                         # default WORKERS=l1 l2 l3, ROUND=01
#   ROUND=02 bash upload_data_local_cluster.sh                # round 02
#   WORKERS="l1 l2 l3" SEQS="001 002 003" bash upload_data_local_cluster.sh
#   SRC_DIR=/custom/path bash upload_data_local_cluster.sh
#
# After success, run:
#   ROUND=$ROUND WORKERS="<same list>" SEQS="<same list>" bash run_local_cluster.sh

set -eu

ROUND="${ROUND:-01}"
SRC_DIR="${SRC_DIR:-$HOME/data/rhoi_zed/$ROUND}"
WORKERS="${WORKERS:-l1 l2 l3}"
SEQS_INPUT="${SEQS:-}"
DST_REMOTE_DIR="${DST_REMOTE_DIR:-data/rhoi_zed/$ROUND}"  # relative to remote $HOME

[ -d "$SRC_DIR" ] || { echo "ERROR: SRC_DIR '$SRC_DIR' not found"; exit 1; }

# ---------- enumerate seqs (dirs that have rgb/) ----------------------------
if [ -z "$SEQS_INPUT" ]; then
  SEQS_INPUT=$(ls -d "$SRC_DIR"/*/rgb 2>/dev/null | awk -F/ '{print $(NF-1)}' | sort)
fi
SEQ_LIST=($SEQS_INPUT)
N_SEQ=${#SEQ_LIST[@]}
[ $N_SEQ -eq 0 ] && { echo "ERROR: no seqs found (need */rgb under $SRC_DIR)"; exit 1; }

# ---------- reachability (and prep remote dst dir) --------------------------
declare -a READY=()
echo "[$(date -Iseconds)] checking workers: $WORKERS"
for h in $WORKERS; do
  if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" \
       "mkdir -p ~/$DST_REMOTE_DIR" 2>/dev/null; then
    READY+=("$h")
    echo "  $h: reachable"
  else
    echo "  $h: UNREACHABLE (skipped)"
  fi
done
N_READY=${#READY[@]}
[ $N_READY -eq 0 ] && { echo "ERROR: no reachable workers"; exit 2; }

# ---------- round-robin shard (SAME algorithm as run_local_cluster.sh) ------
declare -A SHARD
for i in "${!SEQ_LIST[@]}"; do
  h="${READY[$((i % N_READY))]}"
  SHARD[$h]="${SHARD[$h]:-}${SEQ_LIST[$i]} "
done

# ---------- parallel rsync ---------------------------------------------------
echo ""
echo "[$(date -Iseconds)] distributing $N_SEQ seqs across $N_READY workers"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

declare -a PIDS=() LABELS=()
for h in "${READY[@]}"; do
  list_file="$TMPDIR/${h}.list"
  shard_seqs="${SHARD[$h]:-}"
  if [ -z "$shard_seqs" ]; then
    echo "  $h: 0 seqs (skipped)"
    continue
  fi
  printf '%s\n' $shard_seqs > "$list_file"
  n=$(wc -l < "$list_file")
  echo "  $h: $n seqs = $shard_seqs"
  log="$TMPDIR/${h}.log"
  (
    rsync -ar --info=progress2 \
      --files-from="$list_file" \
      --exclude='*.svo2' \
      --exclude='depth/' --exclude='depth_ZED/' --exclude='depth_fs/' --exclude='ply_fs/' \
      --exclude='mask_hand/' --exclude='mask_object/' \
      --exclude='SAM3D/' --exclude='SAM3D_aligned_*/' --exclude='SAM3D_align_filter/' \
      --exclude='pipeline_preprocess/' --exclude='pipeline_corres/' --exclude='pipeline_joint_opt/' \
      --exclude='foundation_pose_align/' \
      --exclude='hands/' --exclude='mano_fit_ckpt/' \
      --exclude='.zed_parse_done' --exclude='.local_preprocess_logs/' \
      "$SRC_DIR"/ "$h:~/$DST_REMOTE_DIR/" > "$log" 2>&1
  ) &
  PIDS+=($!)
  LABELS+=("$h")
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  ${LABELS[$i]}: DONE"
  else
    echo "  ${LABELS[$i]}: FAILED (see log below)"
    tail -5 "$TMPDIR/${LABELS[$i]}.log" | sed 's/^/    /'
    FAILED=1
  fi
done

# persist logs
LOG_DIR="$HOME/data/rhoi_zed/upload_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
cp "$TMPDIR"/*.log "$LOG_DIR"/ 2>/dev/null || true
echo "[$(date -Iseconds)] logs: $LOG_DIR"

[ $FAILED -eq 0 ] || exit 3
echo ""
echo "Next:"
echo "  WORKERS=\"$WORKERS\" SEQS=\"$SEQS_INPUT\" bash $(dirname "$(readlink -f "$0")")/run_local_cluster.sh"
