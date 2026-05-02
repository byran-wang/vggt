#!/bin/bash
# Cluster driver for rhoi (hand-object interaction) — parallels run_wonder_hoi.sh
# (the local/dev driver) but wired for the ht-dm / H100 docker environment and
# run_wonder_hoi_cluster.py (the refactored, lean pipeline entrypoint).
#
# Prerequisites (user's responsibility before running):
#   1. /root/  is already baked by the docker image (envs/{rhoi,hamer,sam3,
#      sam3d-objects}, caches). If running on a FRESH container, run
#         bash scripts/docker_bootstrap.sh
#      first. See .note/28_docker_baking.md for the full recipe.
#   2. $CODE_DIR/third_party/  is populated (git submodule update --init --recursive).
#      Submodule weight files (*.pth / *.pt / *_DATA/) should also be rsync'd in.
#   3. $CODE_DIR/body_models/  contains MANO_*.pkl / SMPLH_*.pkl.
#   4. $DATASET_DIR/<seq>/  has per-seq inputs (rgb/ ir/ meta/ and
#      sam3_prompts/{hand,object}.json). Upload from local via
#         bash upload_data_cluster.sh
#      Masks (mask_hand/ mask_object/) and SAM3D meshes are generated on the
#      cluster from the uploaded prompts — they are NOT uploaded.
#
# Usage:
#   bash run_wonder_hoi_cluster.sh                               # all seq dirs in DATASET_DIR
#   SEQS="002 003" bash run_wonder_hoi_cluster.sh                # only the listed seqs
#   JOINT_OPT_ONLY=1 SEQS="..." bash run_wonder_hoi_cluster.sh   # skip 1/2/3/4, just Stage 5
#   PREPROCESS_ONLY=1 SEQS="..." bash run_wonder_hoi_cluster.sh  # just 1/2/3/4, skip Stage 5
#   DRY_RUN=1 bash run_wonder_hoi_cluster.sh                     # print plan, exit
#
# Recommended deploys for 55 seqs:
#   - 8 pods × 1 GPU × SEQS_PER_GPU=2  (default)  — 16 workers, ~3-4h, safe VRAM
#   - 8 pods × 8 GPU × SEQS_PER_GPU=2  — 55 workers, ~1h, wastes GPUs but fast

set -u

# ---- locate code dir (this script lives at the repo root) ----
CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$CODE_DIR"

# ============================================================================
# USER INPUTS — override via env vars before invoking this script.
# ============================================================================
# All the knobs a user is likely to tweak are gathered here. Everything else
# below this block is internal docker-env plumbing that shouldn't need change.
#
# Quick reference:
#   DATASET          pipeline variant: zed_xy (full ZED) | xper1m (hand-only)
#   DATASET_DIR      where per-seq inputs (rgb/ir/meta/sam3_prompts) live; also
#                    where the pipeline writes derived per-seq outputs
#                    (mask_hand/, mask_object/, depth_fs/, SAM3D*/, hands/,
#                    mano_fit_ckpt/, pipeline_preprocess/, pipeline_corres/).
#                    Default: /mnt/afs/xinyuan/data/rhoi_zed/01  (zed_xy)
#   LOG_DIR          per-seq logs + startup diagnostics. One <seq>.log + one
#                    startup/pod_<id>_<ts>.log per pod.
#                    Default: /mnt/afs/xinyuan/run/rhoi_zed/logs  (zed_xy)
#   OUTPUT_DIR       final Stage 5 results (pipeline_joint_opt/, align_hand_object/,
#                    eval_vis/*.mp4). run_wonder_hoi_cluster.py resolves this
#                    via vggt_code_dir, which is $CODE_DIR/output. To redirect,
#                    set OUTPUT_DIR and this script will symlink
#                    $CODE_DIR/output -> $OUTPUT_DIR. Refuses to overwrite a
#                    pre-existing real output dir.
#                    Default: $CODE_DIR/output
#   SEQS             space-separated seq names. Empty means auto-enumerate
#                    (every subdir of $DATASET_DIR).
#                    Default: <empty> → all seqs in $DATASET_DIR
#   JOINT_OPT_ONLY   if "1", skip 1/2/3/4 and run only Stage 5 (HOI joint_opt +
#                    align + eval_vis). Use when preprocess artifacts are
#                    already on AFS from a prior run.
#                    Default: <unset>
#   PREPROCESS_ONLY  if "1", run 1/2/3/4 but skip Stage 5 (joint_opt/eval_vis).
#                    Use to regenerate preprocess for seqs whose Stage 2 previously
#                    failed (e.g. SAM3D best_id.txt missing).
#                    Default: <unset>  (mutually exclusive with JOINT_OPT_ONLY)
#   NUM_GPUS         override torch.cuda.device_count() detection.
#                    Default: <auto>
#   SEQS_PER_GPU     how many seqs packed onto one GPU (runs as bash-level &
#                    parallel workers). NWORKERS = NUM_GPUS × SEQS_PER_GPU,
#                    capped at the number of seqs this pod owns.
#                    Default: 2. Dashboard suggests 3 would give ~70% GPU util
#                    (vs ~40% at 1), but in practice seq 004 hit CUDA OOM at
#                    SAM3D_gen when 3 seqs landed on the same card
#                    (79 GiB H100, concurrent procs claimed ~60 GiB, leaving
#                    ~533 MiB free). 2 is the safer default; bump back to 3
#                    only if seq footprint is small / GPU is isolated.
#   DRY_RUN          if "1", print the plan and exit without running pipeline.
#                    Default: <unset>
# ============================================================================

# Pipeline shape: zed_xy | xper1m
export DATASET=${DATASET:-zed_xy}

# Per-seq data dir (inputs + derived outputs).
if [[ -z "${DATASET_DIR:-}" ]]; then
    case "$DATASET" in
        zed_xy) export DATASET_DIR=/mnt/afs/xinyuan/data/rhoi_zed/01 ;;
        xper1m) export DATASET_DIR=/mnt/afs/xinyuan/run/rhoi_xper1m ;;
        *) echo "ERROR: no DATASET_DIR default for DATASET=$DATASET — set DATASET_DIR explicitly" >&2; exit 1 ;;
    esac
fi
mkdir -p "$DATASET_DIR"

# Per-seq log dir + startup diagnostics.
if [[ -z "${LOG_DIR:-}" ]]; then
    case "$DATASET" in
        zed_xy) LOG_DIR=/mnt/afs/xinyuan/run/rhoi_zed/logs ;;
        xper1m) LOG_DIR=/mnt/afs/xinyuan/run/rhoi_xper1m/logs ;;
        *) LOG_DIR=$DATASET_DIR/logs ;;
    esac
fi
mkdir -p "$LOG_DIR/startup"

# Final results dir. run_wonder_hoi_cluster.py hardcodes the output location
# as $vggt_code_dir/output (where vggt_code_dir = confs/../ = $CODE_DIR).
# To redirect without touching code, we symlink $CODE_DIR/output -> $OUTPUT_DIR.
# If $CODE_DIR/output is a pre-existing real dir (from prior in-place runs),
# we refuse to overwrite and abort.
OUTPUT_DIR=${OUTPUT_DIR:-$CODE_DIR/output}
if [[ "$OUTPUT_DIR" != "$CODE_DIR/output" ]]; then
    mkdir -p "$OUTPUT_DIR"
    if [[ -L "$CODE_DIR/output" ]]; then
        # Already a symlink — retarget if different.
        existing=$(readlink -f "$CODE_DIR/output" 2>/dev/null || echo "")
        target=$(readlink -f "$OUTPUT_DIR")
        if [[ "$existing" != "$target" ]]; then
            rm "$CODE_DIR/output"
            ln -s "$OUTPUT_DIR" "$CODE_DIR/output"
            echo "[output-dir] symlink retargeted: $CODE_DIR/output -> $OUTPUT_DIR"
        fi
    elif [[ -e "$CODE_DIR/output" ]]; then
        echo "ERROR: OUTPUT_DIR=$OUTPUT_DIR requested but $CODE_DIR/output is a real directory; refusing to overwrite. Move/rename it first or omit OUTPUT_DIR." >&2
        exit 1
    else
        ln -s "$OUTPUT_DIR" "$CODE_DIR/output"
        echo "[output-dir] symlink created: $CODE_DIR/output -> $OUTPUT_DIR"
    fi
fi
export OUTPUT_DIR  # visible to downstream python subprocesses for logging

# ---- Seq selection (WHICH sequences this run processes) ----
# SEQS: space-separated seq names. Empty => auto-enumerate every subdir of
# $DATASET_DIR (or for xper1m, read from confs/sequence_config_xper1m.py).
# Example: SEQS="001 003 005" bash run_wonder_hoi_cluster.sh
: "${SEQS:=}"

# ---- Logic switches (WHAT the pipeline does) ----
# JOINT_OPT_ONLY: if "1", skip Stage 1 (masks+depth) / Stage 2 (SAM3D) / Stage 3 (HAMER) /
#                 Stage 4 (MANO fit) and run only Stage 5 (HOI joint_opt + align + eval_vis).
#                 Requires preprocess artifacts already on AFS for every SEQ in
#                 the list. Typical use: a previous run crashed in joint_opt and
#                 we want to resume without redoing preprocess.
: "${JOINT_OPT_ONLY:=}"

# PREPROCESS_ONLY: if "1", run Stage 1 (masks + depth) / Stage 2 (SAM3D) / Stage 3 (HAMER) /
#                  Stage 4 (MANO fit) but SKIP Stage 5 (joint_opt / eval_vis). Use to
#                  regenerate preprocess for seqs whose Stage 2 previously failed
#                  (e.g. SAM3D best_id.txt missing), without wasting GPU
#                  cycles on joint_opt that we'll run later.
#                  Mutually exclusive with JOINT_OPT_ONLY.
: "${PREPROCESS_ONLY:=}"

# DRY_RUN: if "1", print the 3-level parallelism plan + worker assignment and
#          exit. Pipeline is NOT run. Useful for sanity-checking pod-sharding
#          under different POD_INDEX/POD_COUNT values before committing GPUs.
: "${DRY_RUN:=}"

# ---- Parallelism (HOW MANY processes / how to pack seqs onto GPUs) ----
# NUM_GPUS: number of GPUs visible to this pod. Empty => auto-detect via
#           torch.cuda.device_count(). Override to force a specific count, or
#           to "1" on multi-GPU nodes where you want to restrict to one card.
: "${NUM_GPUS:=}"

# SEQS_PER_GPU: how many seqs are packed onto one GPU concurrently (run in
#               parallel as bash-level & workers). Total workers per pod =
#               min(NUM_GPUS * SEQS_PER_GPU, #seqs_on_this_pod).
# Default 2: observed CUDA OOM at SAM3D_gen with 3 concurrent seqs on one
# H100 (peak ~25-35 GiB/seq). 2 leaves safe headroom. Bump to 3 only if you
# know object footprint is small or the card is isolated.
: "${SEQS_PER_GPU:=2}"

# ============================================================================
# End of user inputs. Everything below is internal plumbing.
# ============================================================================

# ---- env setup (no .env.cluster, all inline) ----
export RHOI_HOME=/root
# CondaEnv in run_wonder_hoi_cluster.py resolves env paths as $ENVS_DIR/<name>.
# Docker bake puts them under /root/envs/ (not ~/miniconda3/envs/).
export ENVS_DIR=${ENVS_DIR:-/root/envs}
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/root/envs/rhoi/bin:/usr/local/cuda-12.4/bin:${PATH:-}

# ---- env role assignment ----
# run_wonder_hoi_cluster.py's CondaEnv registry looks up each role via its env
# var; the actual env used is $ENVS_DIR/$<ROLE>_ENV. Baked docker layout:
#
#   /root/envs/rhoi/           torch 2.1.0+cu118, timm, flash_attn 2.3.6
#                              - main pipeline scripts (robust_hoi_pipeline/...)
#                              - FoundationStereo (core/foundation_stereo) — no extra deps
#                              - FoundationPose (mycpp .so + estimater.py)  — no extra deps
#   /root/envs/hamer/          hamer demo.py + register_mano.py (mmpose, WiLoR-YOLO)
#   /root/envs/sam3/           torch 2.4.1+cu121 + sam3 pkg (mask inference on first frame)
#   /root/envs/sam3d-objects/  Python 3.11 + pytorch3d + sam-3d-objects (3D mesh gen)
#
# FoundationStereo + FoundationPose were historically separate envs upstream
# but their runtime deps are compatible with rhoi (torch 2.1+cu118), so we
# reuse rhoi to avoid baking two more ~6GB envs into the image. To split them
# out again, just pre-set FS_ENV / FP_ENV before invoking this script, e.g.
#     FS_ENV=foundation_stereo FP_ENV=foundation_pose bash run_wonder_hoi_cluster.sh
# (make sure those envs actually exist under $ENVS_DIR).
export RHOI_ENV=${RHOI_ENV:-rhoi}
export FS_ENV=${FS_ENV:-rhoi}           # FoundationStereo reuses rhoi (compatible torch/cuda)
export FP_ENV=${FP_ENV:-rhoi}           # FoundationPose    reuses rhoi (compatible torch/cuda)
export HAMER_ENV=${HAMER_ENV:-hamer}
export SAM3D_ENV=${SAM3D_ENV:-sam3d-objects}
export SAM3_ENV=${SAM3_ENV:-sam3}       # separate env: SAM3 needs torch 2.4 > rhoi's 2.1

# Compilers for runtime nvdiffrast / CUDA kernel JIT builds
export CC=/root/envs/rhoi/bin/x86_64-conda-linux-gnu-gcc
export CXX=/root/envs/rhoi/bin/x86_64-conda-linux-gnu-g++

# HuggingFace / torch.hub: stay offline, use pre-baked cache in /root/.cache/
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Unbuffered python stdout/stderr — critical for split_per_seq below: the
# splitter relies on print_header lines ("========== start: ... for <SEQ> …")
# reaching the pipe in real time to switch routing between per-seq logs.
# With buffering, print_header output lingers in the parent python's buffer
# while subprocess (sam3, FS, ...) output streams through first, causing all
# lines to land in the fallback log until the parent python exits.
export PYTHONUNBUFFERED=1

# (DATASET and DATASET_DIR are set in the USER INPUTS block at the top.)

# xper1m-specific: raw mp4+hdf5 lives under XPER1M_RAW_DIR (read-only);
# xper1m_convert reads from there and materialises under DATASET_DIR/<scene>/.
# Also writes sam3_prompts/hand.json (box + text=right_hand) so SAM3 can run
# non-interactively without a GUI popup.
if [[ "$DATASET" == "xper1m" ]]; then
    export XPER1M_RAW_DIR=${XPER1M_RAW_DIR:-/mnt/afs/zirui/datasets/xperience-10m/data}
    export XPER1M_FRAME_NUMBER=${XPER1M_FRAME_NUMBER:-50}
    # Whitelist: built by generator/scripts/xper1m_build_whitelist.py. When the
    # file exists, sequence_config_xper1m only enumerates eps with enough
    # right-wrist-visible frames — saves compute on useless episodes.
    export XPER1M_WHITELIST=${XPER1M_WHITELIST:-/mnt/afs/xinyuan/run/xper1m_whitelist.txt}
    # Convert mode: pick start from first contiguous visible right-wrist run
    # (body_kp[19] in SLAM world) and write a depth-derived bbox prompt.
    export XPER1M_HAND_VISIBLE_START=${XPER1M_HAND_VISIBLE_START:-1}
    # Force SAM3 non-interactive (xper1m_convert wrote the box prompt already).
    export SAM3_FORCE_NON_INTERACTIVE=${SAM3_FORCE_NON_INTERACTIVE:-1}
fi

# Sanitize LD_LIBRARY_PATH before activate. K8s GPU nodes (and ACP in
# particular) often inject /usr/local/cuda/compat or similar into the
# container's default LD_LIBRARY_PATH. Our cu118 torch (rhoi env) bundles
# its own libcuda/libcudart via DT_RPATH; any externally-prepended CUDA
# compat dir overrides RPATH and links torch against an ABI-incompatible
# libcuda, which silently breaks the pybind class registration. Symptom:
# `AttributeError: module 'torch._C' has no attribute '_OutOfMemoryError'`
# at torch.cuda.__init__ time.
# On ht-dm LD_LIBRARY_PATH is already unset, so this is a no-op there.
unset LD_LIBRARY_PATH

# Activate rhoi venv (conda activate uses unset vars; disable -u around it)
set +u
source /root/envs/rhoi/bin/activate
set -u

# ---- Compute STARTUP_LOG path (LOG_DIR was resolved in USER INPUTS) -----
# Pod identifier for filename. Prefer the submission-provided rank vars, fall
# back to hostname (unique per k8s pod). Timestamp disambiguates re-launches
# of the same pod (ACP restart).
_pod_id="${POD_INDEX:-${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-$(hostname)}}}"
STARTUP_LOG="$LOG_DIR/startup/pod_${_pod_id}_$(date +%Y%m%d_%H%M%S).log"

# ---- startup diagnostic (fast, ~1s) -------------------------------------
# Dumps driver + torch + env state up front so that when a pod fails with
# something like `AttributeError: module 'torch._C' has no attribute
# '_OutOfMemoryError'` we already have the evidence in the log instead of
# having to resubmit with extra probes. Known ACP vs ht-dm divergence points
# that only show up here: NVIDIA driver (libcuda.so), libcudart search path,
# VIRTUAL_ENV leakage, and whether /root/envs/rhoi is the baked copy or an
# overlay from a different mount.
# Tee'd to both stdout (ACP dashboard visibility) and $STARTUP_LOG on AFS
# (post-mortem inspection when the console scrollback is gone or the pod was
# restarted). One file per (pod, launch-timestamp) to cleanly separate re-runs.
{
    echo "================================================================"
    echo " startup diagnostic"
    echo "================================================================"
    # Script version: which copy of this script are we running? Needed when
    # multiple pods / nodes / users share AFS and a stale rsync could leave an
    # outdated script in place. sha256 collides across no two runs; mtime +
    # git-HEAD pin the source commit when CODE_DIR is a git clone.
    _script_path="${BASH_SOURCE[0]}"
    _script_abs="$(readlink -f "$_script_path" 2>/dev/null || echo "$_script_path")"
    _script_sha=$(sha256sum "$_script_abs" 2>/dev/null | awk '{print substr($1,1,12)}')
    _script_mtime=$(stat -c '%y' "$_script_abs" 2>/dev/null | cut -d. -f1)
    _script_size=$(stat -c '%s' "$_script_abs" 2>/dev/null)
    _py_sha=$(sha256sum "$CODE_DIR/run_wonder_hoi_cluster.py" 2>/dev/null | awk '{print substr($1,1,12)}')
    _git_head=$(git -C "$CODE_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo '<not-a-git-repo>')
    _git_dirty=$(git -C "$CODE_DIR" diff --quiet 2>/dev/null && echo clean || echo DIRTY)
    echo "[script]  $_script_abs"
    echo "[script]  sha256=$_script_sha  size=${_script_size}B  mtime=$_script_mtime"
    echo "[script]  py_sha256=$_py_sha   (run_wonder_hoi_cluster.py companion)"
    echo "[script]  git_head=$_git_head  worktree=$_git_dirty"
    echo "[startup] $STARTUP_LOG"
    echo "[host]    $(hostname)   $(uname -r)"
    echo "[driver]  $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi not available')"
    echo "[GPUs]    count=$(nvidia-smi -L 2>/dev/null | wc -l)"
    echo "[venv]    VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
    echo "[PATH]    $(echo $PATH | tr ':' '\n' | head -3 | tr '\n' ':')"
    echo "[LD_LIB]  ${LD_LIBRARY_PATH:-<unset>}"
    echo "[python]  $(which python)  $(python --version 2>&1)"
    # Distributed-launch env vars — we may receive any of these depending on how
    # the pod is submitted:
    #   torchrun            sets RANK / WORLD_SIZE / LOCAL_RANK / NODE_RANK / NNODES / MASTER_*
    #   direct bash pod     usually sets none (single pod = no sharding needed)
    #   ACP framework       may set SENSECORE_PYTORCH_* even without torchrun
    # We print them all up front so we can tell what the launcher is doing.
    echo "[rank]    RANK=${RANK:-<unset>}  WORLD_SIZE=${WORLD_SIZE:-<unset>}  LOCAL_RANK=${LOCAL_RANK:-<unset>}  NODE_RANK=${NODE_RANK:-<unset>}  NNODES=${NNODES:-<unset>}"
    echo "[rank]    MASTER_ADDR=${MASTER_ADDR:-<unset>}  MASTER_PORT=${MASTER_PORT:-<unset>}"
    echo "[rank]    SENSECORE_PYTORCH_NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-<unset>}  SENSECORE_PYTORCH_NNODES=${SENSECORE_PYTORCH_NNODES:-<unset>}"
    echo "[dataset] DATASET=$DATASET  DATASET_DIR=$DATASET_DIR"
    python -c "
import torch
print(f'[torch]   {torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}  devs={torch.cuda.device_count()}')
print(f'[torch._C._OutOfMemoryError] present={hasattr(torch._C, \"_OutOfMemoryError\")}')
import torch.utils.cpp_extension as cpp
print(f'[torch.lib path] {torch.__file__}')
" || echo "[torch]   IMPORT FAILED — pipeline will not run"
    # Verify each downstream env can also import torch (catches per-env issues
    # like hamer having a broken CUDA link while rhoi works).
    for env_name in "${RHOI_ENV}" "${HAMER_ENV}" "${SAM3_ENV}" "${SAM3D_ENV}"; do
        py="$ENVS_DIR/$env_name/bin/python"
        if [ -x "$py" ]; then
            out=$("$py" -c "import torch; print(torch.__version__, hasattr(torch._C, '_OutOfMemoryError'))" 2>&1 | tail -1)
            echo "[env:$env_name] $out"
        else
            echo "[env:$env_name] $py MISSING"
        fi
    done
    echo "================================================================"
} 2>&1 | tee "$STARTUP_LOG"

# ---- sanity checks — fail fast with a clear message if user skipped setup ----
for p in third_party body_models; do
    [ -e "$CODE_DIR/$p" ] || { echo "ERROR: $CODE_DIR/$p missing — populate it per scripts/docker_bootstrap.sh 'NOT in /root/' section." >&2; exit 1; }
done
for env_name in hamer sam3 sam3d-objects; do
    [ -d "$ENVS_DIR/$env_name" ] || { echo "ERROR: $ENVS_DIR/$env_name missing — run scripts/docker_bootstrap.sh first." >&2; exit 1; }
done

# ---- seq list ----
# Enumeration source depends on DATASET:
#   zed_xy  — DATASET_DIR/*/   (data already prepared locally and rsync'd in)
#   xper1m  — sequence_config_xper1m.sequence_name_list, which scans
#             XPER1M_RAW_DIR (raw mp4+hdf5 tree). xper1m_convert then
#             materialises each scene under DATASET_DIR/<scene>/ at runtime.
if [[ -n "${SEQS:-}" ]]; then
    read -r -a SEQS <<<"$SEQS"
    echo "[using SEQS env var: ${#SEQS[@]} seqs]"
elif [[ "$DATASET" == "xper1m" ]]; then
    mapfile -t SEQS < <(python -c "
from confs.sequence_config_xper1m import sequence_name_list
for n in sequence_name_list:
    print(n)
")
    [[ ${#SEQS[@]} -eq 0 ]] && { echo "ERROR: no episodes found under XPER1M_RAW_DIR=$XPER1M_RAW_DIR"; exit 1; }
    echo "[auto-enumerated ${#SEQS[@]} xper1m episodes from $XPER1M_RAW_DIR]"
else
    SEQS=()
    for d in "$DATASET_DIR"/*/; do
        [ -d "$d" ] || continue
        SEQS+=("$(basename "$d")")
    done
    [[ ${#SEQS[@]} -eq 0 ]] && { echo "ERROR: no seq dirs in $DATASET_DIR"; exit 1; }
    echo "[auto-enumerated ${#SEQS[@]} seqs from $DATASET_DIR]"
fi

# ---- sanitize rank env vars for downstream python subprocesses ----------
# ACP's pytorch_ddp framework injects RANK=<pod_idx> (and LOCAL_RANK) somewhere
# in the env. Downstream libraries read blindly: instant-nsr-pl's get_rank()
# returns RANK as-is, then does `torch.cuda.device(get_rank())`. On our 8x1
# single-GPU layout, pod>0 triggers `CUDA error: invalid device ordinal`
# during NeuS training setup in joint_opt, crashing the whole stage.
# Done AFTER startup diagnostic so the diagnostic records what ACP injected.
# Our POD_INDEX/POD_COUNT sharding (below) falls back to SENSECORE_PYTORCH_*
# which we keep untouched.
unset RANK LOCAL_RANK SLURM_PROCID JSM_NAMESPACE_RANK

# ---- main log: capture all post-diagnostic stdout/stderr to AFS ----------
# Startup diagnostic has its own per-pod file (see $STARTUP_LOG).
# This tee captures pod-shard / plan / worker START+DONE / pipeline output /
# SUMMARY — everything after this point. One file per (pod, launch) so ACP
# restarts don't stomp each other.
# Using `exec > >(tee -a ...)` instead of asking the caller to `| tee` lets
# the user submit a plain `bash run_wonder_hoi_cluster.sh` without remembering
# any redirects and still get the full log on AFS.
MAIN_LOG="$LOG_DIR/main/pod_${_pod_id}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$MAIN_LOG")"
exec > >(tee -a "$MAIN_LOG") 2>&1
echo "[main-log] $MAIN_LOG"

# ---- torchrun multi-proc-per-pod guard ----------------------------------
# torchrun with nproc_per_node > 1 spawns N bash processes per pod, all
# running this same script. That'd cause each proc to claim its own seq
# shard (via RANK/WORLD_SIZE) and stomp AFS outputs. We only want ONE
# pipeline manager per pod; NUM_GPUS auto-detect + SEQS_PER_GPU handles
# intra-pod parallelism ourselves. Non-zero LOCAL_RANK means we were
# spawned as a sibling — bow out gracefully.
if [[ -n "${LOCAL_RANK:-}" && "${LOCAL_RANK}" != "0" ]]; then
    echo "[pod-guard] LOCAL_RANK=$LOCAL_RANK != 0 — sibling torchrun proc, exiting (LOCAL_RANK=0 handles this pod)."
    exit 0
fi

# ---- pod sharding (for K8s / ACP multi-replica deployments) -------------
# When N pods each run this script against the same $DATASET_DIR, each pod
# should get a disjoint slice of SEQS (round-robin) to avoid duplicate work
# and AFS output-dir races.
#
# Source priority for (this pod's rank, total pods):
#   1. POD_INDEX / POD_COUNT                             explicit user override
#   2. NODE_RANK / NNODES                                torchrun node-level (CORRECT for pod sharding even with nproc_per_node > 1)
#   3. SENSECORE_PYTORCH_NODE_RANK / _NNODES             ACP framework vars (set even without torchrun)
#   4. RANK / WORLD_SIZE                                 torchrun global (only correct when nproc_per_node=1, i.e. 1 proc/pod)
# (3) takes precedence over (4) because ACP may set both but the SENSECORE
# pair is pod-level while RANK/WORLD_SIZE is torchrun-global.
: "${POD_INDEX:=${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-}}}}"
: "${POD_COUNT:=${NNODES:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-}}}}"
if [[ -n "${POD_INDEX:-}" && -n "${POD_COUNT:-}" && "$POD_COUNT" -gt 1 ]]; then
    SHARDED=()
    for i in "${!SEQS[@]}"; do
        if [[ $((i % POD_COUNT)) -eq $POD_INDEX ]]; then
            SHARDED+=("${SEQS[$i]}")
        fi
    done
    SEQS=("${SHARDED[@]}")
    echo "[pod-shard] POD_INDEX=$POD_INDEX  POD_COUNT=$POD_COUNT  -> ${#SEQS[@]} seqs: ${SEQS[*]}"
    [[ ${#SEQS[@]} -eq 0 ]] && { echo "ERROR: no seqs assigned to POD_INDEX=$POD_INDEX (out of $POD_COUNT)"; exit 1; }
else
    echo "[pod-shard] single pod (no sharding applied; POD_INDEX=${POD_INDEX:-<unset>} POD_COUNT=${POD_COUNT:-<unset>})"
fi

# ---- GPU plan (NUM_GPUS / SEQS_PER_GPU come from USER INPUTS) -----------
if [[ -z "${NUM_GPUS:-}" ]]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
NWORKERS=$((NUM_GPUS * SEQS_PER_GPU))
if [[ ${#SEQS[@]} -lt $NWORKERS ]]; then NWORKERS=${#SEQS[@]}; fi

declare -A WORKER_SEQS
declare -A WORKER_GPU
for i in "${!SEQS[@]}"; do
    wid=$((i % NWORKERS))
    WORKER_SEQS[$wid]+="${SEQS[$i]} "
    WORKER_GPU[$wid]=$((wid % NUM_GPUS))
done

echo "================================================================"
echo " 3-level parallelism plan"
echo "================================================================"
echo "  L1 pod-level     : ${POD_INDEX:-0}/${POD_COUNT:-1} pods, this pod owns ${#SEQS[@]} seqs"
echo "  L2 gpu-level     : ${NUM_GPUS} GPUs on this pod"
echo "  L3 per-gpu seqs  : SEQS_PER_GPU=${SEQS_PER_GPU}  -> NWORKERS=${NWORKERS} on this pod"
echo "  DATASET_DIR      : $DATASET_DIR"
echo "================================================================"
for wid in $(seq 0 $((NWORKERS - 1))); do
    printf "  worker %d (GPU %s):  %s\n" "$wid" "${WORKER_GPU[$wid]}" "${WORKER_SEQS[$wid]}"
done
echo ""

# ---- DRY_RUN: print the plan and exit (lets you verify pod sharding + worker
#      partition without actually running the pipeline) ----
if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN=1 set: plan above; exiting without running the pipeline."
    exit 0
fi

PY="python run_wonder_hoi_cluster.py"

# ---- per-seq log splitter (python) --------------------------------------
# Each worker batches its seqs into a small number of big python invocations
# (stage outer, seq inner). Python prints "========== start: <stage> for <seq>
# ==========" per (stage, seq); we route stdin lines to $LOG_DIR/<seq>.log
# based on that marker. Prelude / cross-seq lines land in worker's fallback.
#
# The splitter script is staged to a temp file *outside* split_per_seq so we
# can safely pipe subprocess stdout into it:
#     { $PY; ... } | split_per_seq fallback.log
# Using an inline `python -c '...'` or `python - <<'EOF'` would collide — the
# heredoc/-c would displace the pipe's stdin and the splitter would see no
# input, silently closing the pipe and killing every $PY with SIGPIPE.
SPLITTER_PY="$(mktemp /tmp/rhoi_splitter.XXXXXX.py)"
cat > "$SPLITTER_PY" <<'PYEOF'
import os, re, sys
# Subprocess output can include non-UTF-8 bytes (e.g. progress-bar chars,
# broken ANSI from libs that assume a TTY). Replace rather than raise.
sys.stdin.reconfigure(errors="replace")
log_dir = os.environ["LOG_DIR"]
fallback = os.environ["FALLBACK"]
# Matches run_wonder_hoi_cluster.py's print_header output:
#   "========== start: <stage description> for <SEQ> =========="
# Seq names can contain hyphens (xper1m UUIDs: xp__<uuid>__ep<N>), so the
# charset must include `-` as well as `_`.
pat = re.compile(r"========== start: .+ for +([A-Za-z0-9_\-]+) +==========\s*$")
current_path = fallback
current_fd = open(fallback, "a", buffering=1, errors="replace")
try:
    for line in sys.stdin:
        m = pat.search(line)
        if m:
            new_path = os.path.join(log_dir, m.group(1) + ".log")
            if new_path != current_path:
                current_fd.close()
                current_fd = open(new_path, "a", buffering=1, errors="replace")
                current_path = new_path
        current_fd.write(line)
finally:
    current_fd.close()
PYEOF
trap 'rm -f "$SPLITTER_PY"' EXIT

split_per_seq() {
    LOG_DIR="$LOG_DIR" FALLBACK="$1" \
    /root/envs/rhoi/bin/python -u "$SPLITTER_PY"
}

# ---- per-worker pipeline: stage outer, seq inner ------------------------
# Each $PY invocation takes the worker's entire seq list via --seq_list, so
# one python process walks all seqs through a bundle of stages. Inside
# run_wonder_hoi_cluster.py's run() loop, the ordering is:
#   for exe in execute_list:
#     for process in process_list:
#       for seq in seq_list:
#         process(seq)
# so a given stage (e.g. HAMER) sees all seqs back-to-back without reloading
# the python process between them.
run_worker() {
    local wid=$1
    local gpu=$2
    shift 2
    local seqs=("$@")          # bash array: ("002" "003" "004")
    local seqs_str="${seqs[*]}" # space-joined: "002 003 004"
    export CUDA_VISIBLE_DEVICES=$gpu
    local fallback="$LOG_DIR/worker_${wid}.log"

    # Header into each seq's log for attribution.
    for s in "${seqs[@]}"; do
        echo ">>> [w${wid} gpu${gpu}] pickup $s  start=$(date -Iseconds)" >> "$LOG_DIR/${s}.log"
    done
    echo ">>> [w${wid} gpu${gpu}] START seqs: ${seqs_str}" | tee -a "$fallback"

    # Resilience policy (intentional):
    #   - NO `set -e` / `pipefail` here. Each $PY invocation runs to completion
    #     and the next one starts regardless of prior return code.
    #   - Inside run_wonder_hoi_cluster.py's run() loop, per-seq failures are
    #     caught; the offending seq is excluded from subsequent stages within
    #     the same Python process, but other seqs keep going.
    #   - Final per-seq verdict is decided post-hoc by `summarize_per_seq`
    #     via Traceback / eval_vis-marker scan over each seq's log.
    # If a stage is truly catastrophic (e.g. import error on first call), it'll
    # keep failing through all seqs and subsequent stages; the summary will show
    # every seq as FAIL. Waste of cycles but no data corruption.

    if [[ "$DATASET" == "xper1m" ]]; then
        # xper1m: hand-only. xper1m_convert is CPU-bound (ffmpeg + h5py) but
        # we run it on the same worker to preserve per-seq locality for the
        # downstream log splitter and output dir naming.
        # --rebuild policy: xper1m_convert MUST rebuild (it's the source-of-
        # truth extraction step that defines which frames exist; any change
        # to start/interval/num_frames would leave stale frames). Downstream
        # per-frame stages (FS depth, hand mask, HAMER, MANO fit) are
        # deterministic overwrites — omit --rebuild to save resume time.
        {
            # Stage 1a: convert raw ep -> ho3d layout (--rebuild: defines frame set)
            $PY --execute_list data_convert --process_list \
                xper1m_convert \
                --seq_list ${seqs_str} --rebuild

            # Stage 1b: FS depth + soft_link + hand mask (per-frame, no --rebuild)
            $PY --execute_list data_convert --process_list \
                get_depth_from_foundation_stereo soft_link_depth \
                ho3d_get_hand_mask \
                --seq_list ${seqs_str}

            # Stage 3: HAMER + interpolate (per-frame, no --rebuild)
            $PY --execute_list data_convert --process_list \
                ho3d_estimate_hand_pose ho3d_interpolate_hamer \
                --seq_list ${seqs_str}

            # Stage 4: MANO fit (per-frame, no --rebuild)
            $PY --execute_list hand_pose_postprocess --process_list \
                fit_hand_intrinsic fit_hand_trans \
                --seq_list ${seqs_str}
        } 2>&1 | split_per_seq "$fallback"
    else
        # JOINT_OPT_ONLY mode: skip 1/2/3/4 (data-layer preprocess already done
        # on AFS), only run the Stage 5 HOI optimization stage. Use this when a
        # previous run crashed during joint_opt (e.g. CUDA rank issue) but left
        # mask_* / depth_fs / SAM3D_align_filter / hands / mano_fit_ckpt intact.
        # Saves ~30-40 min per seq.
        if [[ -n "${JOINT_OPT_ONLY:-}" ]]; then
            echo "[JOINT_OPT_ONLY=1] skipping 1/2/3/4, running only joint_opt + align + eval_vis"
            {
                $PY --execute_list obj_process --process_list \
                    hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
                    hoi_pipeline_joint_opt \
                    hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r \
                    hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
                    hoi_pipeline_eval_vis \
                    --seq_list ${seqs_str} --rebuild
            } 2>&1 | split_per_seq "$fallback"
        elif [[ -n "${PREPROCESS_ONLY:-}" ]]; then
            echo "[PREPROCESS_ONLY=1] running 1/2/3/4 only, skipping Stage 5 joint_opt/align/eval_vis"
            {
                # Stage 1: SAM3 masks + FS depth + soft_link
                $PY --execute_list data_convert --process_list \
                    ho3d_get_hand_mask ho3d_get_obj_mask \
                    get_depth_from_foundation_stereo soft_link_depth \
                    --seq_list ${seqs_str}

                # Stage 2: SAM3D gen + align + filter chain (rebuild: keyframe-dependent)
                $PY --execute_list obj_process --process_list \
                    ho3d_obj_SAM3D_filter_2D \
                    ho3d_obj_SAM3D_gen \
                    ho3d_obj_SAM3D_filter_3D \
                    ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts ho3d_align_SAM3D_fp \
                    pipeline_sam3d_align_filter pipeline_sam3d_delete_unused pipeline_sam3d_best_id \
                    ho3d_SAM3D_post_process \
                    --seq_list ${seqs_str} --rebuild

                # Stage 3: HAMER + interpolate
                $PY --execute_list data_convert --process_list \
                    ho3d_estimate_hand_pose ho3d_interpolate_hamer \
                    --seq_list ${seqs_str}

                # Stage 4: MANO fit
                $PY --execute_list hand_pose_postprocess --process_list \
                    fit_hand_intrinsic fit_hand_trans \
                    --seq_list ${seqs_str}
            } 2>&1 | split_per_seq "$fallback"
        else
            # Full pipeline. --rebuild policy:
            #   1/3/4 — per-frame, keyframe-independent, deterministic overwrite.
            #              Omit --rebuild (no skip logic inside, still re-runs,
            #              but avoids rm -rf race with concurrent pods).
            #   2/5    — keyframe-dependent. If upstream masks/depth/HAMER
            #              change, chosen keyframe may shift and stale align
            #              dirs would produce mixed-state outputs. --rebuild
            #              forces clean regeneration.
            {
                # Stage 1: SAM3 masks + FS depth + soft_link
                $PY --execute_list data_convert --process_list \
                    ho3d_get_hand_mask ho3d_get_obj_mask \
                    get_depth_from_foundation_stereo soft_link_depth \
                    --seq_list ${seqs_str}

                # Stage 2: SAM3D gen + align + filter chain
                $PY --execute_list obj_process --process_list \
                    ho3d_obj_SAM3D_filter_2D \
                    ho3d_obj_SAM3D_gen \
                    ho3d_obj_SAM3D_filter_3D \
                    ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts ho3d_align_SAM3D_fp \
                    pipeline_sam3d_align_filter pipeline_sam3d_delete_unused pipeline_sam3d_best_id \
                    ho3d_SAM3D_post_process \
                    --seq_list ${seqs_str} --rebuild

                # Stage 3: HAMER + interpolate
                $PY --execute_list data_convert --process_list \
                    ho3d_estimate_hand_pose ho3d_interpolate_hamer \
                    --seq_list ${seqs_str}

                # Stage 4: MANO fit
                $PY --execute_list hand_pose_postprocess --process_list \
                    fit_hand_intrinsic fit_hand_trans \
                    --seq_list ${seqs_str}

                # Stage 5: HOI joint opt + align + vis
                $PY --execute_list obj_process --process_list \
                    hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
                    hoi_pipeline_joint_opt \
                    hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r \
                    hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
                    hoi_pipeline_eval_vis \
                    --seq_list ${seqs_str} --rebuild
            } 2>&1 | split_per_seq "$fallback"
        fi
    fi

    echo ">>> [w${wid} gpu${gpu}] DONE seqs: ${seqs_str}  end=$(date -Iseconds)" | tee -a "$fallback"
}

# ---- post-run OK/FAIL verdict per seq -----------------------------------
# Called after `wait`. Inspects each seq's log for a Traceback or a missing
# final-stage marker; prints one line per seq.
# Final-stage marker differs per DATASET:
#   zed_xy  -> "hoi pipeline eval vis for"         (Stage 5 terminus)
#   xper1m  -> "fit hand to trans for"             (Stage 4 terminus)
summarize_per_seq() {
    local ok=0 fail=0 incomplete=0
    local complete_marker incomplete_desc
    case "$DATASET" in
        xper1m) complete_marker="fit hand to trans for"    ; incomplete_desc="never reached fit_hand_trans" ;;
        *)      complete_marker="hoi pipeline eval vis for"; incomplete_desc="never reached eval_vis" ;;
    esac
    for seq in "${SEQS[@]}"; do
        local log="$LOG_DIR/${seq}.log"
        if [ ! -s "$log" ]; then
            echo "  NOLOG  $seq"
            incomplete=$((incomplete+1))
            continue
        fi
        if grep -q "Traceback (most recent call last):" "$log"; then
            local first_err=$(grep -n -m1 "Traceback\|Error:" "$log" 2>/dev/null | head -1)
            echo "  FAIL   $seq  ($log : $first_err)"
            fail=$((fail+1))
        elif ! grep -q "$complete_marker" "$log"; then
            echo "  INCOMP $seq  ($incomplete_desc)"
            incomplete=$((incomplete+1))
        else
            echo "  OK     $seq"
            ok=$((ok+1))
        fi
    done
    echo ""
    echo "total: ${#SEQS[@]}  OK=$ok  FAIL=$fail  INCOMPLETE=$incomplete"
}

# ---- dispatch: each worker runs its seq batch stage-outer / seq-inner ----
for wid in $(seq 0 $((NWORKERS - 1))); do
    [ -z "${WORKER_SEQS[$wid]:-}" ] && continue
    run_worker "$wid" "${WORKER_GPU[$wid]}" ${WORKER_SEQS[$wid]} &
done
wait

echo ""
echo "================================================================"
echo " All workers done. Per-seq verdict:"
echo "================================================================"
summarize_per_seq
echo ""
echo " Logs: $LOG_DIR/<seq>.log  (per-seq) + $LOG_DIR/worker_*.log (prelude / cross-seq)"
echo "================================================================"
