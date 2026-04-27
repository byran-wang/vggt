#!/bin/bash
# One-time /root/ bootstrap for docker image baking.
#
# WHAT THIS SCRIPT DOES (after preflight passes):
#   1. apt install libegl1 / libgles2 / libglvnd0            HAMER's libEGL deps
#   2. rsync $AFS_ROOT/cache/{torch_hub,huggingface,torch_extensions}/
#        -> /root/.cache/                                     pre-baked weights + JIT cache
#   3. tar -xzf hamer_env.tar.gz  -> /root/envs/hamer/        + path fixups
#   4. tar -xzf sam3d_env.tar.gz  -> /root/envs/sam3d-objects/ + path fixups
#   5. tar -xzf sam3_env.tar.gz   -> /root/envs/sam3/          + path fixups
#   6. pip install timm + flash_attn wheel into /root/envs/rhoi
#   7. verify: import torch / timm / flash_attn in all 3 envs
#
# AFTER SUCCESS:
#   docker commit <container> rhoi:vN
#   From that point users just run run_wonder_hoi_cluster.sh on fresh containers.
#
# WHAT USERS NEED TO PREPARE ON AFS (NOT done by this script):
#   Under $AFS_ROOT/ (default /mnt/afs/xinyuan/):
#     envs/{hamer,sam3,sam3d}_env.tar.gz   conda-pack artifacts from local envs
#     cache/{torch_hub,huggingface,torch_extensions}/  pre-populated user caches
#     wheels/flash_attn-*.whl              flash_attn wheel (cu118 + cp310 for rhoi)
#
#   Under $AFS_ROOT/rhoi/third_party/ (cluster code clone — model weights NOT packaged):
#     FoundationStereo/pretrained_models/model_best_bp2.pth   (3.1 G)
#     FoundationPose/weights/                                  (247 M)
#     FoundationPose/mycpp/build/mycpp.cpython-310-*.so        compile once, carry forward
#     hamer/_DATA/                                             (6.1 G)
#     hamer/pretrained_models/detector.pt                       (52 M)
#     sam-3d-objects/checkpoints/hf/                           (12  G, HF model cache)
#     sam3/sam3/model/checkpoints/model.safetensors            (3.3 G)
#
#   Under $AFS_ROOT/body_models/  (MANO + SMPLH pkls, license-restricted)
#
# See .note/28_docker_baking.md for the full recipe + known gotchas.
# ---------------------------------------------------------------------------

# AFS_ROOT can be overridden to point at a non-canonical stash (e.g. per-user).
: "${AFS_ROOT:=/mnt/afs/xinyuan}"

# ---- 0. preflight: verify every file the rest of the script (and the
# pipeline post-bootstrap) needs, with ✓ / ✗ per item. Exits 1 with a concrete
# "what's missing + where to get it" hint if anything is absent. ----
preflight() {
    local ok=1
    echo "================================================================"
    echo " docker_bootstrap.sh preflight   AFS_ROOT=$AFS_ROOT"
    echo "================================================================"
    # check <path> <label> [<min-bytes>]
    # min-bytes: if given, flag file as ✗ even when it exists if size < min
    # (catches partial rsync uploads — `-e` passes on a half-written file).
    check() {
        local path="$1" label="$2" minbytes="${3:-0}"
        if [[ ! -e "$path" ]]; then
            printf "  [\033[31m✗\033[0m] %-52s MISSING\n" "$label"
            ok=0
            return
        fi
        local sz bytes
        sz=$(du -sh "$path" 2>/dev/null | cut -f1)
        if [[ "$minbytes" -gt 0 && -f "$path" ]]; then
            bytes=$(stat -c%s "$path" 2>/dev/null || echo 0)
            if [[ "$bytes" -lt "$minbytes" ]]; then
                local gb=$(( minbytes / 1024 / 1024 / 1024 ))
                printf "  [\033[31m✗\033[0m] %-52s %s  (partial — expected ≥ %sG)\n" "$label" "$sz" "$gb"
                ok=0
                return
            fi
        fi
        printf "  [\033[32m✓\033[0m] %-52s %s\n" "$label" "$sz"
    }

    # min-size thresholds guard against rsync partials — set just below the
    # real size to give headroom for compression-ratio drift across re-packs.
    local MIN_HAMER=$((4 * 1024 * 1024 * 1024))           # ~4.4G
    local MIN_SAM3=$((3 * 1024 * 1024 * 1024))            # ~4.1G
    local MIN_SAM3D=$((7 * 1024 * 1024 * 1024))           # ~8.0G
    local MIN_FS_CKPT=$((3 * 1024 * 1024 * 1024))         # ~3.1G
    local MIN_HAMER_DATA=$((5 * 1024 * 1024 * 1024))      # ~6.1G
    local MIN_SAM3D_HF=$((10 * 1024 * 1024 * 1024))       # ~12G  (du on dir)
    local MIN_SAM3_WEIGHTS=$((3 * 1024 * 1024 * 1024))    # ~3.3G

    echo "[env tarballs — unpacked into /root/envs/ by this script]"
    check "$AFS_ROOT/envs/hamer_env.tar.gz"  "envs/hamer_env.tar.gz"  "$MIN_HAMER"
    check "$AFS_ROOT/envs/sam3_env.tar.gz"   "envs/sam3_env.tar.gz"   "$MIN_SAM3"
    check "$AFS_ROOT/envs/sam3d_env.tar.gz"  "envs/sam3d_env.tar.gz"  "$MIN_SAM3D"

    echo "[caches — rsync'd into /root/.cache/ by this script]"
    check "$AFS_ROOT/cache/torch_hub"         "cache/torch_hub/"
    check "$AFS_ROOT/cache/huggingface"       "cache/huggingface/"
    check "$AFS_ROOT/cache/torch_extensions"  "cache/torch_extensions/"

    echo "[wheels — pip installed into /root/envs/rhoi by this script]"
    check "$AFS_ROOT/wheels/flash_attn-2.3.6+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
          "wheels/flash_attn-*.whl"

    echo "[third_party model weights — NOT baked, read from AFS at runtime]"
    check "$AFS_ROOT/rhoi/third_party/FoundationStereo/pretrained_models/model_best_bp2.pth" \
          "FoundationStereo/pretrained_models/model_best_bp2.pth" "$MIN_FS_CKPT"
    check "$AFS_ROOT/rhoi/third_party/FoundationPose/weights" \
          "FoundationPose/weights/"
    check "$AFS_ROOT/rhoi/third_party/FoundationPose/mycpp/build/mycpp.cpython-310-x86_64-linux-gnu.so" \
          "FoundationPose/mycpp/build/mycpp.cpython-310-*.so"
    check "$AFS_ROOT/rhoi/third_party/hamer/_DATA" \
          "hamer/_DATA/"  "$MIN_HAMER_DATA"
    check "$AFS_ROOT/rhoi/third_party/hamer/pretrained_models/detector.pt" \
          "hamer/pretrained_models/detector.pt"
    check "$AFS_ROOT/rhoi/third_party/sam-3d-objects/checkpoints/hf" \
          "sam-3d-objects/checkpoints/hf/" "$MIN_SAM3D_HF"
    check "$AFS_ROOT/rhoi/third_party/sam3/sam3/model/checkpoints/model.safetensors" \
          "sam3/sam3/model/checkpoints/model.safetensors" "$MIN_SAM3_WEIGHTS"

    echo "[body_models — NOT baked, read from AFS at runtime]"
    check "$AFS_ROOT/body_models/MANO_RIGHT.pkl" "body_models/MANO_RIGHT.pkl"
    check "$AFS_ROOT/body_models/MANO_LEFT.pkl"  "body_models/MANO_LEFT.pkl"
    check "$AFS_ROOT/body_models/SMPLH_male.pkl" "body_models/SMPLH_male.pkl"

    echo ""
    if [[ $ok -eq 0 ]]; then
        cat <<EOF
================================================================
 PREFLIGHT FAILED — see ✗ entries above.

 How to acquire each missing piece:

 1) envs/*.tar.gz  (conda-pack from a machine that has the env installed):

      conda-pack -p ~/miniconda3/envs/<name> -o /tmp/<name>_env.tar.gz \\
                 --ignore-editable-packages --ignore-missing-files
      rsync /tmp/<name>_env.tar.gz ht-dm:$AFS_ROOT/envs/

 2) cache/  (rsync from a machine that has run the pipeline once):

      rsync -a ~/.cache/torch/hub/        ht-dm:$AFS_ROOT/cache/torch_hub/
      rsync -a ~/.cache/huggingface/      ht-dm:$AFS_ROOT/cache/huggingface/
      rsync -a ~/.cache/torch_extensions/ ht-dm:$AFS_ROOT/cache/torch_extensions/

 3) wheels/flash_attn-*.whl:
      https://github.com/Dao-AILab/flash-attention/releases
      (pick cu118 + torch2.1 + cp310 + cxx11abiFALSE)

 4) third_party/ weights  (copy from a machine that has them, or from the
    canonical xinyuan stash at /mnt/afs/xinyuan/rhoi/third_party/):

      rsync -a --include='*/' \\
               --include='*.pth' --include='*.pt' --include='*.safetensors' --include='*.so' \\
               --exclude='*' \\
               /mnt/afs/xinyuan/rhoi/third_party/  ht-dm:\$AFS_ROOT/rhoi/third_party/

 5) body_models/  (MANO / SMPLH, license-restricted — from Shibo's stash):
      rsync -a /mnt/afs/xinyuan/body_models/ ht-dm:\$AFS_ROOT/body_models/

 Full recipe + known gotchas:  .note/28_docker_baking.md
================================================================
EOF
        exit 1
    fi

    echo "================================================================"
    echo " Preflight OK — proceeding with bootstrap."
    echo "================================================================"
    echo ""
}
preflight

set -eux

# ---- 1. system layer ----
# libegl/libgles/libglvnd - HAMER's libEGL deps
# ffmpeg - multiple pipeline steps shell out to /usr/bin/ffmpeg for mp4 assembly
#   (utils_simba/eval_vis.py, robust_hoi_pipeline/eval_sum_vis.py, ...)
apt-get update -qq
apt-get install -y libegl1 libgles2 libglvnd0 ffmpeg

# ---- 2. env-level caches (torch hub + HF hub + torch extensions JIT) ----
mkdir -p /root/.cache/torch/hub /root/.cache/huggingface /root/.cache/torch_extensions
rsync -a "$AFS_ROOT/cache/torch_hub/"         /root/.cache/torch/hub/
rsync -a "$AFS_ROOT/cache/huggingface/"       /root/.cache/huggingface/
rsync -a "$AFS_ROOT/cache/torch_extensions/"  /root/.cache/torch_extensions/

# ---- 3. unpack env tarballs into /root/envs/<name>/ ----
# The three tarballs have INCONSISTENT top-level layout:
#   - hamer_env.tar.gz   has `hamer/` at tar root (pre-existing raw tar)
#   - sam3d_env.tar.gz,
#     sam3_env.tar.gz    flat (conda-pack default: contents at tar root)
# We handle both by extracting each into its own dedicated target dir.
# Note: all `grep -rIl` uses -I so sed can't corrupt binary/ELF files.

# 3a. hamer  (legacy dir-prefixed tarball → -C parent is fine)
rm -rf /root/envs/hamer
tar -xzf "$AFS_ROOT/envs/hamer_env.tar.gz" -C /root/envs/
grep -rIl "/home/vox/miniconda3/envs/hamer" /root/envs/hamer/ | \
    xargs -r sed -i "s|/home/vox/miniconda3/envs/hamer|/root/envs/hamer|g"
# editable finder for hamer / mmpose -> point at AFS shared third_party/hamer
grep -rIl "/home/vox/rhoi/third_party/hamer" /root/envs/hamer/lib/python3.10/site-packages/ | \
    xargs -r sed -i "s|/home/vox/rhoi/third_party/hamer|/mnt/afs/xinyuan/rhoi/third_party/hamer|g"

# 3b. sam3d-objects  (Python 3.11 + CUDA 12.1 + pytorch3d; flat tarball)
rm -rf /root/envs/sam3d-objects
mkdir -p /root/envs/sam3d-objects
tar -xzf "$AFS_ROOT/envs/sam3d_env.tar.gz" -C /root/envs/sam3d-objects/
grep -rIl "/home/vox/miniconda3/envs/sam3d-objects" /root/envs/sam3d-objects/ | \
    xargs -r sed -i "s|/home/vox/miniconda3/envs/sam3d-objects|/root/envs/sam3d-objects|g"
# editable finder for sam3d_objects -> point at AFS shared third_party/sam-3d-objects
grep -rIl "/home/vox/sam-3d-objects" /root/envs/sam3d-objects/lib/python3.11/site-packages/ 2>/dev/null | \
    xargs -r sed -i "s|/home/vox/sam-3d-objects|/mnt/afs/xinyuan/rhoi/third_party/sam-3d-objects|g"

# 3c. sam3  (torch 2.4.1+cu121 for mask inference; flat tarball)
# sam3 was packed with `conda-pack --ignore-editable-packages`, so the sam3
# package itself is installed in editable mode pointing at a local path —
# the last sed rewrites that path to the AFS shared third_party/sam3 clone.
rm -rf /root/envs/sam3
mkdir -p /root/envs/sam3
tar -xzf "$AFS_ROOT/envs/sam3_env.tar.gz" -C /root/envs/sam3/
grep -rIl "/home/vox/miniconda3/envs/sam3" /root/envs/sam3/ | \
    xargs -r sed -i "s|/home/vox/miniconda3/envs/sam3|/root/envs/sam3|g"
# editable finder for sam3 -> point at AFS shared third_party/sam3
# sam3 env is Python 3.11 (not 3.10 like hamer) — use a glob so this stays
# correct if the env is ever re-packed at a different minor version.
grep -rIl "/home/vox/rhoi/third_party/sam3" /root/envs/sam3/lib/python3.*/site-packages/ 2>/dev/null | \
    xargs -r sed -i "s|/home/vox/rhoi/third_party/sam3|/mnt/afs/xinyuan/rhoi/third_party/sam3|g"

# ---- 4. rhoi env extras (base image is missing / has wrong versions) ----
PY=/root/envs/rhoi/bin/python
# pip wrapper's shebang resolves to system python -> always use `python -m pip`
$PY -m pip install --disable-pip-version-check --quiet timm
$PY -m pip install --disable-pip-version-check --quiet \
    "$AFS_ROOT/wheels/flash_attn-2.3.6+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# Base image has trimesh 4.x which renamed VoxelGrid.origin → .translation.
# Pipeline (utils_simba/visibility_test.py) still uses .origin — stay on 3.x.
# Keep this in sync with requirements.txt.
$PY -m pip install --disable-pip-version-check --quiet "trimesh==3.23.5"

# ---- 5. verify ----
set +x
echo ""
echo "=============================================================="
echo " docker bootstrap done. Verifying..."
echo "=============================================================="
$PY -c "import torch, timm, flash_attn; print('rhoi:', torch.__version__, torch.cuda.is_available(), 'timm', timm.__version__, 'flash_attn', flash_attn.__version__)"
/root/envs/hamer/bin/python -c "import torch; print('hamer torch:', torch.__version__, torch.cuda.is_available())"
/root/envs/sam3/bin/python -c "import torch; print('sam3 torch:', torch.__version__, torch.cuda.is_available())"
/root/envs/sam3d-objects/bin/python -c "import torch; print('sam3d torch:', torch.__version__, torch.cuda.is_available())"
echo ""
echo "NOTE: 'import hamer/mmpose/sam3d_objects' will only work once the AFS"
echo "      third_party clone is populated (git submodule update)."
echo ""
echo "Next: in your rhoi clone, run  bash run_wonder_hoi_cluster.sh"
