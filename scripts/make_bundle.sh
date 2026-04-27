#!/bin/bash
# Build a single self-contained tar bundle for USB-copying to a new worker host.
# Output: ~/rhoi-bootstrap.tar.gz (~55 GB).
#
# Bundle contents:
#   rhoi-bootstrap/
#     Miniconda3-latest-Linux-x86_64.sh
#     envs/{rhoi,sam3,sam3d-objects,hamer}_env.tar.gz
#     weights.tar.gz    (third_party/*/checkpoints + hamer/_DATA + body_models + FP mycpp.so)
#     torch_hub.tar.gz  (~/.cache/torch/hub — DINOv2 etc.)
#     rhoi_code.tar.gz  (this repo, minus output/.git/weights/data)
#     install.sh
#     README.md
#
# Run-once on the SOURCE machine. Takes ~15-30 min depending on disk speed.
# Prereq: conda-pack installed (~/.local/bin/conda-pack), 4 envs ready.

set -e

STAGE="${STAGE:-$HOME/rhoi_bundle_staging}"
OUT="${OUT:-$HOME/rhoi-bootstrap.tar.gz}"
PACK="${PACK:-$HOME/.local/bin/conda-pack}"
REPO="${REPO:-$HOME/rhoi}"

log() { echo "[$(date -Iseconds)] $*"; }

log "staging dir: $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/envs"

# ---------- 1. miniconda installer ----------
MI="$HOME/Downloads/Miniconda3-latest-Linux-x86_64.sh"
[ -f "$MI" ] || { echo "ERROR: $MI missing"; exit 1; }
log "copy miniconda installer"
cp -a "$MI" "$STAGE/"

# ---------- 2. conda-pack 4 envs (parallel) ----------
log "conda-pack 4 envs (parallel)"
pack_env() {
  local env="$1"
  local t0=$(date +%s)
  echo "  [pack $env] start"
  "$PACK" -n "$env" -o "$STAGE/envs/${env}_env.tar.gz" \
    --ignore-editable-packages --ignore-missing-files --force \
    > "$STAGE/pack_${env}.log" 2>&1
  echo "  [pack $env] done dt=$(($(date +%s)-t0))s size=$(du -h "$STAGE/envs/${env}_env.tar.gz" | cut -f1)"
}
pack_env rhoi &
pack_env sam3 &
pack_env sam3d-objects &
pack_env hamer &
wait

# ---------- 3. weights tarball ----------
log "pack weights"
( cd "$REPO" && tar czf "$STAGE/weights.tar.gz" \
    third_party/sam3/sam3/model/checkpoints \
    third_party/sam-3d-objects/checkpoints \
    third_party/FoundationStereo/pretrained_models \
    third_party/FoundationPose/weights \
    third_party/FoundationPose/mycpp/build \
    third_party/hamer/_DATA \
    third_party/hamer/pretrained_models \
    body_models )
log "weights.tar.gz size=$(du -h "$STAGE/weights.tar.gz" | cut -f1)"

# ---------- 4. torch_hub cache ----------
if [ -d "$HOME/.cache/torch/hub" ]; then
  log "pack torch_hub cache"
  tar czf "$STAGE/torch_hub.tar.gz" -C "$HOME/.cache/torch" hub
  log "torch_hub.tar.gz size=$(du -h "$STAGE/torch_hub.tar.gz" | cut -f1)"
else
  log "WARN: ~/.cache/torch/hub missing — Stage 2 filter_2D will try github on workers"
fi

# ---------- 5. code ----------
log "pack code"
tar czf "$STAGE/rhoi_code.tar.gz" \
  --exclude=output --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
  --exclude=third_party/sam3/sam3/model/checkpoints \
  --exclude=third_party/sam-3d-objects/checkpoints \
  --exclude=third_party/FoundationStereo/pretrained_models \
  --exclude=third_party/FoundationPose/weights \
  --exclude=third_party/FoundationPose/mycpp/build \
  --exclude=third_party/hamer/_DATA \
  --exclude=third_party/hamer/pretrained_models \
  --exclude=body_models \
  -C "$(dirname "$REPO")" "$(basename "$REPO")"
log "rhoi_code.tar.gz size=$(du -h "$STAGE/rhoi_code.tar.gz" | cut -f1)"

# ---------- 6. install.sh + README ----------
cp -a "$REPO/scripts/install.sh" "$STAGE/install.sh"
cp -a "$REPO/scripts/install.README.md" "$STAGE/README.md" 2>/dev/null || true
chmod +x "$STAGE/install.sh"

# ---------- 7. outer tar ----------
log "final tar: $OUT"
tar czf "$OUT" -C "$(dirname "$STAGE")" "$(basename "$STAGE")" \
    --transform "s,^$(basename "$STAGE"),rhoi-bootstrap,"
log "BUNDLE READY: $OUT ($(du -h "$OUT" | cut -f1))"
echo ""
echo "Next on a target host:"
echo "  cd ~ && tar xzf rhoi-bootstrap.tar.gz && bash rhoi-bootstrap/install.sh"
