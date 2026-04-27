#!/bin/bash
# Upload the locally-preprocessed ZED dataset to a cluster target.
#
# Usage:
#   bash local_upload.sh <local_dataset_dir> <remote>
#
# Example:
#   bash local_upload.sh ~/data/rhoi_zed/01 ht-dm:/mnt/afs/xinyuan/data/rhoi_zed
#   --> after sync, ht-dm will have /mnt/afs/xinyuan/data/rhoi_zed/01/{seq}/...
#
# What gets uploaded (per 2026-04-21 arch — local = only light preprocessing):
#   rgb/ ir/ meta/                  decoded from svo2
#   sam3_prompts/{hand,object}.json first-frame click prompts (INCLUDED — consumed by cluster SAM3 inference)
#
# Excluded (cluster regenerates or doesn't need):
#   - *.svo2                  raw recordings
#   - depth / depth_ZED/      cluster uses FoundationStereo depth instead
#   - mask_hand/ mask_object/ cluster now runs SAM3 inference from sam3_prompts/
#   - SAM3D/                  cluster generates SAM3D meshes from FS depth
#   - .zed_parse_done         local idempotency stamp
#   - .local_preprocess_logs/ local background-job logs

set -eu

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <local_dataset_dir> <remote>"
    echo "example: $0 ~/data/rhoi_zed/01 ht-dm:/mnt/afs/xinyuan/data/rhoi_zed"
    exit 1
fi

LOCAL=${1%/}
REMOTE=$2

[[ -d "$LOCAL" ]] || { echo "ERROR: local dir '$LOCAL' does not exist"; exit 1; }

echo "================================================================"
echo " rsync:"
echo "   src: $LOCAL"
echo "   dst: $REMOTE"
echo "================================================================"

rsync -rltvh --partial --progress \
    --no-owner --no-group \
    --exclude='*.svo2' \
    --exclude='depth' \
    --exclude='depth_ZED/' \
    --exclude='depth_fs/' \
    --exclude='ply_fs/' \
    --exclude='mask_hand/' \
    --exclude='mask_object/' \
    --exclude='SAM3D/' \
    --exclude='SAM3D_aligned_*/' \
    --exclude='SAM3D_align_filter/' \
    --exclude='pipeline_preprocess/' \
    --exclude='pipeline_corres/' \
    --exclude='pipeline_joint_opt/' \
    --exclude='foundation_pose_align/' \
    --exclude='hands/' \
    --exclude='mano_fit_ckpt/' \
    --exclude='.zed_parse_done' \
    --exclude='.local_preprocess_logs/' \
    "$LOCAL" "$REMOTE/"

echo ""
echo "================================================================"
echo " Upload done."
echo " Next on cluster:  bash run_wonder_hoi_cluster.sh <seqs...>"
echo "================================================================"
