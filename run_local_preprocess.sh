#!/bin/bash
# Local preprocessing for ZED self-captured data.
# Per 2026-04-21 architecture: local is light-weight — GUI click only. The
# cluster (ht-dm H100) now owns SAM3 inference + FS depth + SAM3D gen.
#
# Two phases (both foreground, no GPU needed):
#   Phase 0:    ZED_parse_data                  (decode .svo2 -> rgb/ir/meta)
#   Phase 1a:   click OBJECT mask prompts       (ho3d_get_obj_mask_prompt)
#   Phase 1b:   click HAND   mask prompts       (ho3d_get_hand_mask_prompt)
#
# Each phase 1 sub-step: loads SAM3 predictor ONCE (run_HO3D_video.py
# --prompt_only with nargs=+), pops up per-seq live-preview windows; existing
# prompt JSON pre-fills the popup (idempotent re-runs — Enter to accept, or
# edit and re-save).
#
# Usage:
#   bash run_local_preprocess.sh                        # all .svo2 in DATASET_DIR
#   bash run_local_preprocess.sh all                    # same as no args
#   bash run_local_preprocess.sh <seq> [<seq> ...]      # specific seqs
#
# After Phase 1b finishes, upload (sam3_prompts/ + rgb/ir/meta, NO masks —
# cluster generates those from the prompts):
#   bash upload_data_cluster.sh $DATASET_DIR ht-dm:/mnt/afs/xinyuan/data/rhoi_zed

set -eu
# Ctrl+C / SIGTERM at any layer (SAM3 popup, ZED decode, wrapper) aborts the
# entire pipeline — prevents silent fall-through to the next phase.
trap 'echo "[run_local_preprocess] interrupted"; exit 130' INT TERM

# ---- env (override as needed) ----
export DATASET=${DATASET:-zed_xy}
export DATASET_DIR=${DATASET_DIR:-$HOME/data/rhoi_zed/01}
export RHOI_ENV=${RHOI_ENV:-rhoi}          # needs pyzed for ZED_parse_data
export SAM3_ENV=${SAM3_ENV:-sam3}          # run_wonder_hoi -> ho3d_get_*_mask_prompt
# HF cache already populated locally; stay offline to avoid flaky network
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
CONDA_BASE=${CONDA_DIR:-$HOME/miniconda3}
MAIN_PY="$CONDA_BASE/envs/$RHOI_ENV/bin/python"

# ---- resolve seqs: no args or 'all' -> every .svo2 in DATASET_DIR ----
if [[ $# -lt 1 || ( $# -eq 1 && "$1" == "all" ) ]]; then
    SEQS=()
    for f in "$DATASET_DIR"/*.svo2; do
        [ -f "$f" ] || continue
        SEQS+=("$(basename "$f" .svo2)")
    done
    [[ ${#SEQS[@]} -eq 0 ]] && { echo "ERROR: no .svo2 files in $DATASET_DIR"; exit 1; }
    echo "[auto-enumerated ${#SEQS[@]} seqs from $DATASET_DIR]"
else
    SEQS=("$@")
fi

cd "$(dirname "$0")"
CODE_DIR=$(pwd)

# ---- Phase 0: ZED_parse_data (skip if completion stamp exists) ----
echo "================================================================"
echo " Phase 0: ZED_parse_data for ${#SEQS[@]} seq(s)"
echo "================================================================"
for seq in "${SEQS[@]}"; do
    stamp="$DATASET_DIR/$seq/.zed_parse_done"
    if [ -f "$stamp" ]; then
        echo ">>> [P0] $seq: stamp found, skipping"
        continue
    fi
    echo ">>> [P0] $seq: ZED_parse_data"
    "$MAIN_PY" run_wonder_hoi.py \
        --execute_list data_convert --process_list ZED_parse_data \
        --seq_list "$seq" --rebuild || { echo "ZED_parse_data failed on $seq"; exit 1; }
    nframes=$(ls "$DATASET_DIR/$seq/rgb/"*.jpg 2>/dev/null | wc -l)
    echo "frames=$nframes  ts=$(date -Iseconds)" > "$stamp"
    echo ">>> [P0] $seq: stamp written ($nframes frames)"
done

# ---- Phase 1: collect all SAM3 prompts, split into two passes ----
# Pass 1a: OBJECT mask prompt for all seqs (focus on object first, less context-switch)
# Pass 1b: HAND mask prompt for all seqs
# Each pop is a GUI window on frame 0; Enter saves JSON to
# $DATASET_DIR/$seq/sam3_prompts/{hand,object}.json and exits (~1s per popup).
echo ""
echo "================================================================"
echo " Phase 1a: click OBJECT mask prompt for ${#SEQS[@]} seq(s)"
echo "================================================================"
"$MAIN_PY" run_wonder_hoi.py \
    --execute_list data_convert \
    --process_list ho3d_get_obj_mask_prompt \
    --seq_list "${SEQS[@]}" --rebuild \
    || { echo "Phase 1a (obj prompts) failed"; exit 1; }

echo ""
echo "================================================================"
echo " Phase 1b: click HAND mask prompt for ${#SEQS[@]} seq(s)"
echo "================================================================"
"$MAIN_PY" run_wonder_hoi.py \
    --execute_list data_convert \
    --process_list ho3d_get_hand_mask_prompt \
    --seq_list "${SEQS[@]}" --rebuild \
    || { echo "Phase 1b (hand prompts) failed"; exit 1; }

echo ""
echo "================================================================"
echo " Local preprocessing DONE. Upload + cluster pipeline:"
echo "   bash upload_data_cluster.sh $DATASET_DIR ht-dm:/mnt/afs/$USER/data/rhoi_zed"
echo "   ssh ht-dm 'cd /mnt/afs/$USER/rhoi && bash run_wonder_hoi_cluster.sh'"
echo "================================================================"
