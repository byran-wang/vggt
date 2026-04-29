#!/bin/bash
# One-shot install for the rhoi-bootstrap bundle. Idempotent — safe to re-run.
#
# Expected to be run from inside the bundle directory, e.g.
#     cd ~/rhoi-bootstrap && bash install.sh
# or equivalently
#     bash ~/rhoi-bootstrap/install.sh
#
# Prereqs on target host:
#   - Linux x86_64
#   - NVIDIA GPU + driver >= 525
#   - ~/ writable with >=60 GB free
#
# After success, this host is READY. Orchestrator on the source machine:
#   rsync -aP ~/data/rhoi_zed/01/  user@thishost:~/data/rhoi_zed/01/
#   WORKERS="thishost ..." bash ~/rhoi/run_local_cluster.sh

set -e
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[$(date -Iseconds)] $*"; }

# ---------- 1. miniconda ----------
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  log "install miniconda"
  bash "$BUNDLE/Miniconda3-latest-Linux-x86_64.sh" -b -p "$HOME/miniconda3"
else
  log "miniconda already at ~/miniconda3 (skip)"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ---------- 2. conda ToS (silent) ----------
log "accept conda ToS"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# ---------- 3. unpack envs ----------
mkdir -p "$HOME/miniconda3/envs"
for env in rhoi sam3 sam3d-objects hamer; do
  dst="$HOME/miniconda3/envs/$env"
  tar="$BUNDLE/envs/${env}_env.tar.gz"
  if [ -x "$dst/bin/python" ]; then
    log "env $env already unpacked (skip)"
    continue
  fi
  [ -f "$tar" ] || { echo "ERROR: $tar missing"; exit 2; }
  log "unpack $env ($(du -h "$tar" | cut -f1))"
  mkdir -p "$dst"
  # conda-pack tarballs sometimes emit "Skipping to next header" on extraction
  # due to broken build-time symlinks; extraction still produces a working env.
  # Ignore tar's non-zero exit; conda-unpack will validate.
  set +e
  tar xzf "$tar" -C "$dst"
  set -e
  # conda-unpack shebang is "/usr/bin/env python"; system PATH may lack python.
  # Call via env's python explicitly.
  "$dst/bin/python" "$dst/bin/conda-unpack"
done

# ---------- 4. symlinks ----------
if [ ! -e "$HOME/miniconda3/envs/vggsfm_tmp" ]; then
  log "symlink vggsfm_tmp -> rhoi"
  ln -s rhoi "$HOME/miniconda3/envs/vggsfm_tmp"
fi
mkdir -p "$HOME/Documents/project"
if [ ! -e "$HOME/Documents/project/vggt" ]; then
  log "symlink ~/Documents/project/vggt -> ~/rhoi"
  ln -s "$HOME/rhoi" "$HOME/Documents/project/vggt"
fi

# ---------- 5. code ----------
if [ ! -d "$HOME/rhoi" ] || [ ! -f "$HOME/rhoi/run_wonder_hoi.py" ]; then
  log "extract code -> ~/rhoi/"
  tar xzf "$BUNDLE/rhoi_code.tar.gz" -C "$HOME"
else
  log "code already at ~/rhoi/ (skip; run rsync to update)"
fi

# ---------- 6. weights ----------
if [ ! -f "$HOME/rhoi/third_party/sam3/sam3/model/checkpoints/sam3.pt" ]; then
  log "extract weights"
  ( cd "$HOME/rhoi" && tar xzf "$BUNDLE/weights.tar.gz" )
else
  log "weights already extracted (skip)"
fi

# ---------- 7. torch_hub cache ----------
if [ -f "$BUNDLE/torch_hub.tar.gz" ] && [ ! -d "$HOME/.cache/torch/hub/facebookresearch_dinov2_main" ]; then
  log "extract torch_hub cache -> ~/.cache/torch/"
  mkdir -p "$HOME/.cache/torch"
  tar xzf "$BUNDLE/torch_hub.tar.gz" -C "$HOME/.cache/torch"
fi

# ---------- 7b. HF cache (timm edgenext for FoundationStereo, etc.) ----------
if [ -f "$BUNDLE/hf_cache.tar.gz" ] && [ ! -d "$HOME/.cache/huggingface/hub/models--timm--edgenext_small.usi_in1k" ]; then
  log "extract HF cache -> ~/.cache/huggingface/"
  mkdir -p "$HOME/.cache/huggingface"
  tar xzf "$BUNDLE/hf_cache.tar.gz" -C "$HOME/.cache/huggingface"
fi

# ---------- 8. rhoi env self-heal (libcudart + lib64 + ffmpeg symlinks) ----------
conda activate rhoi
[ ! -e "$CONDA_PREFIX/lib64" ] && ln -s lib "$CONDA_PREFIX/lib64"
if [ ! -e "$CONDA_PREFIX/lib/libcudart.so" ]; then
  cudart=$(find "$CONDA_PREFIX/lib" -name "libcudart.so.1*" 2>/dev/null | head -1)
  [ -n "$cudart" ] && ln -sfn "$(basename "$cudart")" "$CONDA_PREFIX/lib/libcudart.so"
fi
# ffmpeg in rhoi env is a symlink that conda-pack writes with absolute build-time
# path — dangling on any new host. Rebuild as relative path so it travels.
ffmpeg_link="$CONDA_PREFIX/bin/ffmpeg"
ffmpeg_rel_target="../lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
if [ -e "$CONDA_PREFIX/$ffmpeg_rel_target" ]; then
  if [ ! -e "$ffmpeg_link" ] || [ "$(readlink "$ffmpeg_link")" != "$ffmpeg_rel_target" ]; then
    log "fix ffmpeg symlink (relative path)"
    rm -f "$ffmpeg_link"
    ln -s "$ffmpeg_rel_target" "$ffmpeg_link"
  fi
fi
conda deactivate

# ---------- 8b. hamer env: relocate editable install paths ----------
# conda-pack on the build host writes __editable__*_finder.py with absolute
# paths to the build-host's home (e.g. /home/ubuntu/rhoi/...). On a new host
# (e.g. /home/vox/rhoi/...) those paths break. Sed-replace any /home/<other>/
# prefix to the current $HOME so the finder loads from this host's source.
HAMER_SP="$HOME/miniconda3/envs/hamer/lib/python3.10/site-packages"
if [ -d "$HAMER_SP" ]; then
  for f in "$HAMER_SP"/__editable__*_finder.py; do
    [ -f "$f" ] || continue
    # Match any /home/<user>/rhoi/... and rewrite the /home/<user>/ prefix.
    if grep -qE "/home/[^/]+/rhoi/" "$f" && ! grep -q "$HOME/rhoi/" "$f"; then
      log "relocate editable path in $(basename "$f") -> $HOME"
      sed -i -E "s|/home/[^/]+/rhoi/|$HOME/rhoi/|g" "$f"
      # purge stale .pyc that still cached the old path
      rm -f "$HAMER_SP/__pycache__/$(basename "$f" .py).cpython-"*.pyc
    fi
  done
fi

# ---------- 9. verify ----------
conda activate rhoi
log "=== VERIFY ==="
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
rm -rf ~/.cache/torch_extensions/py310_cu118/nerfacc_cuda
python - <<'PY' 2>&1 | grep -v UserWarning | tail -6
import numpy, torch, timm, flash_attn
print(f"  numpy={numpy.__version__} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
print(f"  timm={timm.__version__} flash_attn={flash_attn.__version__}")
from nerfacc.contraction import ContractionType
print(f"  nerfacc _C OK: {ContractionType.AABB.to_cpp_version()}")
PY
"$HOME/miniconda3/envs/hamer/bin/python" -c "import mmpose; print(f'  mmpose={mmpose.__version__}  ({mmpose.__file__})')" 2>&1 | tail -1
echo "  nvcc:   $(which nvcc)"
echo "  ffmpeg: $(which ffmpeg)"
ffmpeg -version 2>/dev/null | head -1 | sed 's/^/  /'

log "=== READY ==="
echo ""
echo "Next (on orchestrator machine):"
echo "  rsync -aP ~/data/rhoi_zed/01/  \$(whoami)@$(hostname -s):~/data/rhoi_zed/01/"
echo "  WORKERS=\"t1 t2 t3 $(hostname -s)\" bash ~/rhoi/run_local_cluster.sh"
