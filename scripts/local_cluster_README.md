# Local cluster: bundle + bootstrap + dispatch

Self-contained bundle and orchestration scripts for running the rhoi pipeline
across a small local cluster of GPU workers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Orchestrator (your dev machine, vox @ rhoi)                      │
│                                                                  │
│  upload_data_local_cluster.sh    round-robin shard rsync data    │
│         ↓ rsync                                                  │
│  run_local_cluster.sh            round-robin dispatch + ssh      │
│         ↓ ssh nohup                                              │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐  ┌──────────┐  ┌──────────┐
│ l1 (vox @ worker host 1)            │  │   l2     │  │   l3     │
│                                     │  │          │  │          │
│  worker_pipeline.sh seq1 seq2 ...   │  │  same    │  │  same    │
│    ├─ Stage 1: SAM3 mask + FS depth │  │          │  │          │
│    ├─ Stage 2: SAM3D chain (10 step)│  │          │  │          │
│    ├─ Stage 3: HAMER hand pose      │  │          │  │          │
│    ├─ Stage 4: MANO fit             │  │          │  │          │
│    └─ Stage 5: joint_opt + eval_vis │  │          │  │          │
│                  → mp4 artifacts    │  │          │  │          │
└─────────────────────────────────────┘  └──────────┘  └──────────┘
```

`upload_data_local_cluster.sh` and `run_local_cluster.sh` use the **same
round-robin shard algorithm**, so each worker only receives the data it will
process. `worker_pipeline.sh` is the per-worker entry point — it sets up the
conda env, env vars (`RHOI_FS_PYTHON`, `RHOI_FP_PYTHON`, `HF_HUB_OFFLINE`,
`CUDA_HOME`, …) and runs the five stages serially, collecting per-stage exit
codes.

## Bundle contents

| File | Size | Purpose |
|---|---|---|
| `Miniconda3-latest-Linux-x86_64.sh` | 162 MB | miniconda installer |
| `envs/{rhoi,sam3,sam3d-objects,hamer}_env.tar.gz` | ~25 GB | conda-packed envs (rhoi has flash_attn 2.3.6 + timm + ffmpeg + nerfacc) |
| `weights.tar.gz` | ~34 GB | SAM3 / FS / FP / HAMER weights + MANO / SMPLH (`body_models/`) |
| `torch_hub.tar.gz` | ~2 GB | DINOv2 etc. (Stage 2 filter_2D loads via `torch.hub`) |
| `hf_cache.tar.gz` | ~1.2 GB | HuggingFace cache (FS needs `timm/edgenext_small.usi_in1k`) |
| `rhoi_code.tar.gz` | ~12 GB | rhoi repo + `third_party/sam-3d-objects/checkpoints` |
| `install.sh` | — | idempotent one-shot installer |

## Target host prereqs

- Linux x86_64
- NVIDIA GPU + driver **≥ 525**
- `~/` writable with ≥ 80 GB free

## Bringing up a new worker host

1. Create a clean unprivileged user (recommended `vox` for easy teardown):
   ```bash
   sudo useradd -m -s /bin/bash vox
   echo 'vox: ' | sudo chpasswd            # password = single space
   sudo cp -r ~/.ssh /home/vox/ && sudo chown -R vox:vox /home/vox/.ssh
   ```

2. Drop the bundle into the new user's home and install:
   ```bash
   sudo mv rhoi-bootstrap /home/vox/ && sudo chown -R vox:vox /home/vox/rhoi-bootstrap
   sudo -iu vox bash ~/rhoi-bootstrap/install.sh
   ```

   Takes 15–25 min. `install.sh` is idempotent — re-running is safe.
   On success prints `=== READY ===` plus a verify line for
   `numpy / torch / timm / flash_attn / nerfacc / mmpose / ffmpeg`.

3. Add an alias in `~/.ssh/config` on the orchestrator (e.g. `l1`, `l2`, `l3`)
   pointing at the worker host with `User vox`.

## Running the pipeline (from orchestrator)

Defaults: workers `l1 l2 l3`, source data at `~/data/rhoi_zed/01/`:

```bash
# distribute shards (only the seqs each worker will process)
bash ~/rhoi/upload_data_local_cluster.sh

# launch pipeline
bash ~/rhoi/run_local_cluster.sh

# monitor
for h in l1 l2 l3; do
  echo "== $h =="; ssh $h 'tail -5 ~/pipeline_run.log'
done
```

Override defaults via env vars:
```bash
WORKERS="l1 l2"  SEQS="001 003 010"  bash ~/rhoi/run_local_cluster.sh
SKIP_DATA_CHECK=1 bash ~/rhoi/run_local_cluster.sh   # bypass data-presence guard
```

## Removing a worker

Because everything lives under `vox`, teardown is one command:
```bash
sudo userdel -r vox     # removes /home/vox and ~80 GB of envs+weights+data+output
```

## Caveats and known fragility

These are the parts of `install.sh` that paper over older code rather than fix
it. They work today; future refactors should target removing them.

### `~/Documents/project/vggt` symlink (legacy "vggt" path)

The project was originally named `vggt` and `run_wonder_hoi.py` still hardcodes
`cd ~/Documents/project/vggt/...` in 4 places (lines ~891, 1194, 1251, 1322).
`install.sh` works around this by creating

```
~/Documents/project/vggt  ->  ~/rhoi
```

**Consequences:**
- If `~/Documents/project/vggt` already exists on the host (a real directory,
  or a symlink pointing at something else), `install.sh` silently skips
  creating the symlink and the pipeline will fail at the first `cd
  ~/Documents/project/vggt` with no obvious error.
- If you move or rename `~/rhoi` after install, this symlink dangles.
- Proper fix (TODO): replace those 4 `f"{home_dir}/Documents/project/vggt"`
  with `vggt_code_dir` (already auto-detected in `confs/sequence_config.py`
  via `os.path.dirname(__file__)`) and delete the symlink step from
  `install.sh`.

### `ffmpeg` symlink rebuild

`conda-pack` writes the rhoi env's `bin/ffmpeg` as an absolute symlink to the
build host's `imageio_ffmpeg` binary path. On any other host (different user,
different `$HOME`) it dangles silently — pipelines fail at `eval_vis` step 8
without producing mp4. `install.sh` rebuilds it as a relative symlink:

```
$CONDA_PREFIX/bin/ffmpeg  ->  ../lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
```

**Consequence:** if a future env upgrade changes the imageio_ffmpeg binary
filename (e.g. `-v7.0.3`), this symlink breaks. Update `install.sh` accordingly.

### `mmpose` editable install path rewrite

`mmpose` is a `pip install -e` editable in the hamer env, with `.pth` shim
files (`__editable___mmpose_0_24_0_finder.py`) hardcoding the build host's
ViTPose source path (e.g. `/home/ubuntu/rhoi/.../ViTPose/mmpose`).
`install.sh` `sed`-replaces `/home/<user>/rhoi/` to `$HOME/rhoi/` after extract.

**Consequence:** if the worker user installs rhoi anywhere other than `$HOME/rhoi/`
(e.g. `/opt/rhoi/`), this rewrite misses. Adjust the regex in `install.sh` step 8b.

### `nerfacc` JIT compilation cache (not in bundle)

`nerfacc.contraction.ContractionType.AABB.to_cpp_version()` triggers nvcc to
JIT-compile the CUDA kernel into `~/.cache/torch_extensions/py310_cu118/nerfacc_cuda/`.
`install.sh` step 9 verify deliberately runs this so the kernel is built once
during install (and `ninja` / `nvcc` / `lib64` / `libcudart.so` are all
available). The compiled `.so` then persists for pipeline runs.

**Consequence:** if you delete `~/.cache/torch_extensions/...` post-install,
the *first* pipeline call rebuilds it (~1 min) — fine, but slower first run.

### `RHOI_FS_PYTHON` / `RHOI_FP_PYTHON` env-var override

`run_wonder_hoi.py` calls FoundationStereo / FoundationPose with
`{conda_dir}/envs/foundation_stereo/bin/python` and
`{conda_dir}/envs/foundation_pose/bin/python` by default (matches local dev
setups that have those envs). On the cluster the bundled `rhoi` env contains
all the FS / FP deps already, so `worker_pipeline.sh` exports

```bash
export RHOI_FS_PYTHON="$CONDA_PREFIX/bin/python"
export RHOI_FP_PYTHON="$CONDA_PREFIX/bin/python"
```

**Consequence:** if you copy a snippet from `worker_pipeline.sh` but forget
those exports, run_wonder_hoi.py will try to use `envs/foundation_stereo` (which
doesn't exist on cluster workers) and the FS subprocess will silently fail
with `python: not found` — Stage 2 then crashes on a missing `depth_fs/*.png`.

### Hardcoded `/home/simba/...` in some `third_party/*` files

A handful of files contain `/home/simba/...` hardcoded paths from the original
author's machine: `third_party/FoundationStereo/run_foundation_stereo.py`,
`third_party/instant-nsr-pl/models/geometry.py` (debug snippets), `run_vggt.py`
default args, etc.

**Consequence:** these are *not* on the cluster pipeline path, but if you
invoke them directly (ad-hoc debugging), they will fail. Don't be surprised.

### `RUN_ON_SERVER=true` mode in `confs/sequence_config.py`

Toggling this swaps `home_dir → /data1/shibo/`, `conda_dir → /home/shibo/.conda/`,
and a HO3D dataset_dir. **Don't enable this on cluster workers** — it's the
"original author's server" code path.
