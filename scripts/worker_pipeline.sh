#!/bin/bash
# Single-worker pipeline runner (Stages 1..5) for a seq list on one machine.
# Called remotely by run_local_cluster.sh. Expects the host to have:
#   - ~/miniconda3/envs/{rhoi,sam3,sam3d-objects,hamer}/
#   - ~/miniconda3/envs/vggsfm_tmp -> rhoi (symlink)
#   - ~/Documents/project/vggt -> ~/rhoi (symlink, for legacy hardcoded paths)
#   - ~/rhoi/ with code + weights (third_party/*/checkpoints, body_models, ...)
#   - ~/data/rhoi_zed/<ROUND>/<seq>/ with rgb/ir/meta/sam3_prompts/
# Usage:
#   bash worker_pipeline.sh <seq...>                        # default DATASET_DIR=~/data/rhoi_zed/01
#   DATASET_DIR=~/data/rhoi_zed/02 bash worker_pipeline.sh <seq...>
#   JOINT_OPT_ONLY=1 bash worker_pipeline.sh <seq...>       # skip Stages 1-4, run only Stage 5
# No `set -u` — conda activate.d hooks reference unbound vars (ADDR2LINE, ...).

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rhoi

# conda cuda puts libs in lib/, nerfacc JIT looks for lib64/ and -lcudart.so.
# Ensure both symlinks exist (idempotent).
[ ! -e "$CONDA_PREFIX/lib64" ] && ln -s lib "$CONDA_PREFIX/lib64"
if [ ! -e "$CONDA_PREFIX/lib/libcudart.so" ]; then
  cudart=$(find "$CONDA_PREFIX/lib" -name "libcudart.so.1*" 2>/dev/null | head -1)
  [ -n "$cudart" ] && ln -sfn "$(basename "$cudart")" "$CONDA_PREFIX/lib/libcudart.so"
fi

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DATASET="${DATASET:-zed_xy}"
export DATASET_DIR="${DATASET_DIR:-$HOME/data/rhoi_zed/01}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# FoundationStereo / FoundationPose run inside the rhoi env on the cluster
# (the bundled rhoi env has flash_attn / timm / FS / FP deps installed). Local
# default in run_wonder_hoi.py points at envs/foundation_{stereo,pose} for
# backward compatibility, override here.
export RHOI_FS_PYTHON="${RHOI_FS_PYTHON:-$CONDA_PREFIX/bin/python}"
export RHOI_FP_PYTHON="${RHOI_FP_PYTHON:-$CONDA_PREFIX/bin/python}"

cd "$HOME/rhoi" || { echo "ERROR: ~/rhoi not found"; exit 98; }
SEQS="$*"
[ -z "$SEQS" ] && { echo "Usage: $0 <seq...>"; exit 2; }

START=$(date +%s)
echo "[$(date -Iseconds)] START host=$(hostname) seqs=$SEQS"
echo "[$(date -Iseconds)] env: python=$(which python) ninja=$(which ninja) nvcc=$(which nvcc) ffmpeg=$(which ffmpeg)"

FAILED_STAGES=""
run_stage() {
  local name="$1"; shift
  local t0=$(date +%s)
  echo ""
  echo "########## [$(date -Iseconds)] STAGE $name START ##########"
  "$@"
  local rc=$?
  echo "########## STAGE $name exit=$rc dt=$(($(date +%s)-t0))s ##########"
  [ $rc -ne 0 ] && FAILED_STAGES="$FAILED_STAGES \"$name\""
}

JOINT_OPT_ONLY="${JOINT_OPT_ONLY:-0}"

if [ "$JOINT_OPT_ONLY" != "1" ]; then
  # Stage 1: SAM3 masks (hand+object) + FoundationStereo depth + soft-link
  run_stage "1 masks+depth" \
    python run_wonder_hoi.py --execute_list data_convert --process_list \
      ho3d_get_hand_mask ho3d_get_obj_mask get_depth_from_foundation_stereo soft_link_depth \
      --seq_list $SEQS

  # Stage 2: SAM3D mesh gen + align + filter + post_process
  run_stage "2 SAM3D chain" \
    python run_wonder_hoi.py --execute_list obj_process --process_list \
      ho3d_obj_SAM3D_filter_2D ho3d_obj_SAM3D_gen ho3d_obj_SAM3D_filter_3D \
      ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts ho3d_align_SAM3D_fp \
      pipeline_sam3d_align_filter pipeline_sam3d_delete_unused pipeline_sam3d_best_id \
      ho3d_SAM3D_post_process \
      --seq_list $SEQS --rebuild

  # Stage 3: HAMER hand pose + interpolate
  run_stage "3 HAMER" \
    python run_wonder_hoi.py --execute_list data_convert --process_list \
      ho3d_estimate_hand_pose ho3d_interpolate_hamer \
      --seq_list $SEQS

  # Stage 4: MANO fit (intrinsic + trans)
  run_stage "4 MANO fit" \
    python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list \
      fit_hand_intrinsic fit_hand_trans \
      --seq_list $SEQS
else
  echo "[$(date -Iseconds)] JOINT_OPT_ONLY=1 — skipping Stages 1-4, going straight to Stage 5"
fi

# Stage 5: HOI joint opt -> neus global -> align (h/r/o/ho) -> eval + summary.
# hoi_pipeline_blender_rendering is intentionally NOT included: it re-renders
# every frame with Blender (very slow). Run separately if needed.
run_stage "5 joint_opt+eval" \
  python run_wonder_hoi.py --execute_list obj_process --process_list \
    hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
    hoi_pipeline_joint_opt \
    hoi_pipeline_neus_global \
    hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r \
    hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
    hoi_pipeline_eval hoi_pipeline_eval_vis \
    eval_sum eval_sum_vis \
    --seq_list $SEQS --rebuild

TOTAL=$(($(date +%s)-START))
N_MP4=$(ls "$HOME/rhoi/output"/*/pipeline_joint_opt/eval_vis/nvdiffrast_overlay.mp4 2>/dev/null | wc -l)
N_SEQS=$(echo $SEQS | wc -w)
echo ""
if [ -n "$FAILED_STAGES" ]; then
  echo "########## ALL DONE host=$(hostname) dt=${TOTAL}s mp4=${N_MP4}/${N_SEQS} FAILED_STAGES=[${FAILED_STAGES} ] ##########"
  exit 1
fi
echo "########## ALL DONE host=$(hostname) dt=${TOTAL}s mp4=${N_MP4}/${N_SEQS} ##########"
