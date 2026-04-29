# rhoi — cluster pipeline manual

How to run the hand-object interaction reconstruction pipeline on a
pre-built H100 docker image backed by an AFS shared filesystem.

---

## 1. Prerequisites

### 1.1 Docker image (pre-baked — nothing to set up inside)

The image ships with:

- **Conda envs** at `/root/envs/{rhoi, hamer, sam3, sam3d-objects}`
- **Torch Hub cache** at `/root/.cache/torch/hub/` (dinov2, vggsfm, superpoint, nvdiffrast/nerfacc JIT)
- **HuggingFace cache** at `/root/.cache/huggingface/` (timm edgenext_small, MoGe ViT-L)
- **flash-attn** installed into the rhoi env

Just launch a container based on this image — no bootstrap step.

### 1.2 Model weights (NOT in the image — you must stage on AFS)

Under `$AFS_ROOT/rhoi/third_party/`:

| Path | Size | Source |
|---|---|---|
| `FoundationStereo/pretrained_models/model_best_bp2.pth` | 3.3 G | [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) |
| `FoundationPose/weights/*/model_best.pth` | 237 M | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) |
| `FoundationPose/mycpp/build/mycpp.cpython-310-*.so` | 155 K | build locally (`cmake + make` in `FoundationPose/mycpp/`), rsync the `.so` |
| `hamer/_DATA/` | 6.1 G | run upstream `fetch_demo_data.sh` in [geopavlakos/hamer](https://github.com/geopavlakos/hamer) |
| `hamer/pretrained_models/detector.pt` | 52 M | HAMER repo |
| `sam-3d-objects/checkpoints/hf/` | 12 G | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) (HF checkpoints) |
| `sam3/sam3/model/checkpoints/sam3.pt` + `model.safetensors` | 3.4 G + 3.3 G | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) (HF checkpoints) |

### 1.3 Hand / body models (license-gated)

Under `$AFS_ROOT/body_models/` — **one directory above** the rhoi clone:

```
MANO_LEFT.pkl, MANO_RIGHT.pkl              https://mano.is.tue.mpg.de
SMPLH_male.pkl, SMPLH_female.pkl           https://mano.is.tue.mpg.de (SMPL+H)
contact_zones.pkl                          (from HOLD / upstream)
sealed_vertices_sem_idx.npy                (from HOLD / upstream)
```

---

## 2. Input / output layout

All paths are overridable via env vars — defaults shown.

| Var | Default | What |
|---|---|---|
| `DATASET_DIR` | `/mnt/afs/$USER/data/rhoi_zed/01` | per-seq input data |
| `LOG_DIR` | `/mnt/afs/$USER/run/rhoi_zed/logs` | per-seq stdout/stderr |
| `ENVS_DIR` | `/root/envs` | where the 4 conda envs live |

**Input** — one directory per seq under `$DATASET_DIR/`:

```
$DATASET_DIR/<seq>/
    rgb/                        RGB images (0000.jpg ...)
    ir/                         IR images (for stereo depth)
    meta/                       per-frame intrinsics + camera pose pickles
    sam3_prompts/
        hand.json               SAM3 first-frame click for hand
        object.json             SAM3 first-frame click for object
```

Masks, depth and SAM3D meshes are **generated on the cluster** — do not upload them.

**Output** — per-seq:

```
$DATASET_DIR/<seq>/pipeline_joint_opt/eval_vis/nvdiffrast_overlay.mp4
```

---

## 3. Conda envs — how to activate

Four envs are pre-built in the image (conda-packed, flat trees under `/root/envs/`).

| Env | Use for |
|---|---|
| `rhoi` | main pipeline, FoundationStereo, FoundationPose, NeuS (torch 2.1) |
| `sam3` | SAM3 mask inference (torch 2.4) |
| `sam3d-objects` | SAM3D mesh generation (Python 3.11 + pytorch3d) |
| `hamer` | HAMER hand pose |

### 3.1 Two activation patterns

**(A) `source .../bin/activate` — full activation**

```bash
source /root/envs/rhoi/bin/activate
```

This is the conda-pack entry point. It runs every `etc/conda/activate.d/*.sh` hook, so env vars set by packages (`CUDA_HOME`, package-level `LD_LIBRARY_PATH` entries, etc.) are applied. Use this when you need an interactive shell inside the env, **and** when entering the top-level rhoi env for a run.

Deactivate with `source /root/envs/rhoi/bin/deactivate` (or exit the shell).

**(B) Direct `/root/envs/<env>/bin/python` — subprocess only**

```bash
/root/envs/sam3/bin/python some_script.py
```

Python's `sys.prefix` / `sys.path` is inferred from the interpreter's location, so **packages installed in that env import fine**. But `activate.d` hooks do **not** run — if a package depends on `CUDA_HOME`, a bespoke `LD_LIBRARY_PATH`, etc., you must have them set already in the calling shell.

### 3.2 What the cluster driver does

1. `run_wonder_hoi_cluster.sh` does a **shell-level activate** of the rhoi env: `source /root/envs/rhoi/bin/activate` — so `CUDA_HOME`, the rhoi env's activate.d hooks, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `NUM_GPUS` are all in the environment.
2. Inside that activated shell, `run_wonder_hoi_cluster.py` spawns **direct `/root/envs/<env>/bin/python`** subprocesses for stages that need sam3 / sam3d / hamer. These inherit the already-exported env vars (A above took care of them), and just need their own `sys.path`, which the direct python binary resolves correctly.

You don't need to activate anything manually to run the pipeline — this is automatic.

---

## 4. Typical workflow

```
   LOCAL                                      CLUSTER (docker on AFS)
   =====                                      =======================
   run_local_preprocess.sh                    (nothing)
     decode .svo2 -> rgb/ir/meta
     click SAM3 prompts (hand + object)

   upload_data_cluster.sh                     -> $DATASET_DIR/<seq>/
     rsync rgb/ir/meta/sam3_prompts
                                              run_wonder_hoi_cluster.sh
                                                Stage 1:  SAM3 masks + FS depth
   sync_code_cluster.sh                       Stage 2:  SAM3D mesh + filter + align
     rsync repo -> AFS (before run)           Stage 3:  HAMER hand pose
                                              Stage 4:  MANO fit
                                              Stage 5:  HOI joint opt + eval mp4

                                              -> $DATASET_DIR/<seq>/pipeline_joint_opt/
                                                 eval_vis/nvdiffrast_overlay.mp4
```

### 4.1 Local preprocess

```bash
# default DATASET_DIR = ~/data/rhoi_zed/01, reads every .svo2 in there
bash run_local_preprocess.sh

# or specific seqs
bash run_local_preprocess.sh 002 003
```

Phases:
- **Phase 0**: decode `.svo2` → `rgb/ ir/ meta/` (stamp `.zed_parse_done` for idempotency)
- **Phase 1a**: GUI popup per-seq — click on the **object** on frame 0
- **Phase 1b**: GUI popup per-seq — click on the **hand**

SAM3 predictor is loaded **once** per phase and reused across all seqs. Existing `sam3_prompts/*.json` pre-fills the popup — hit Enter to keep, or re-click and save.

### 4.2 Upload data to AFS

```bash
bash upload_data_cluster.sh ~/data/rhoi_zed/01 ht-dm:/mnt/afs/$USER/data/rhoi_zed
# -> after sync, ht-dm has /mnt/afs/$USER/data/rhoi_zed/01/<seq>/{rgb,ir,meta,sam3_prompts}
```

Excludes `.svo2`, masks, depth, SAM3D outputs and local stamps — the cluster regenerates them.

### 4.3 Sync code to AFS

```bash
bash sync_code_cluster.sh                                   # default: ht-dc:/mnt/afs/xinyuan/code/robust_hoi/
bash sync_code_cluster.sh ht-dm:/mnt/afs/$USER/rhoi/        # custom remote
```

`--delete` is used (destination mirrors source), so stale files on AFS are cleaned up.

### 4.4 Run on cluster

```bash
ssh ht-dm                                       # container based on the rhoi image
cd /mnt/afs/$USER/rhoi

# every seq found in $DATASET_DIR/
bash run_wonder_hoi_cluster.sh

# only the listed seqs
SEQS="002 003 004" bash run_wonder_hoi_cluster.sh

# skip preprocessing, just joint-opt + eval (Stage 5 only, uses artifacts from a previous run)
JOINT_OPT_ONLY=1 bash run_wonder_hoi_cluster.sh

# only preprocess (Stages 1-4), don't run joint-opt
PREPROCESS_ONLY=1 bash run_wonder_hoi_cluster.sh

# dry-run: print the seq plan + environment diagnostic, then exit
DRY_RUN=1 bash run_wonder_hoi_cluster.sh
```

---

## 5. Runtime env vars (optional overrides)

| Var | Default | Purpose |
|---|---|---|
| `DATASET_DIR` | `/mnt/afs/$USER/data/rhoi_zed/01` | per-seq input root |
| `LOG_DIR` | `/mnt/afs/$USER/run/rhoi_zed/logs` | per-seq log output |
| `ENVS_DIR` | `/root/envs` | where the conda envs live |
| `RHOI_ENV` | `rhoi` | main env (main pipeline + FS + FP reuse) |
| `SAM3_ENV` | `sam3` | SAM3 mask inference |
| `SAM3D_ENV` | `sam3d-objects` | SAM3D mesh gen |
| `HAMER_ENV` | `hamer` | HAMER hand pose |
| `NUM_GPUS` | `torch.cuda.device_count()` | override detected GPU count |
| `SEQS_PER_GPU` | `2` | concurrent seqs per GPU (watch VRAM) |
| `JOINT_OPT_ONLY` | `0` | skip stages 1-4, run only joint-opt + eval |
| `PREPROCESS_ONLY` | `0` | run stages 1-4, skip joint-opt |
| `DRY_RUN` | `0` | print plan, don't run |

---

## 6. File roles

| File | Role |
|---|---|
| `run_wonder_hoi_cluster.sh` | Top-level cluster driver: shards seqs across pods/GPUs, logs per-seq |
| `run_wonder_hoi_cluster.py` | The actual pipeline (5 stages, dispatches to the right env per stage) |
| `run_local_preprocess.sh` | Local: SVO decode + SAM3 prompt clicking |
| `upload_data_cluster.sh` | Local: rsync dataset (rgb/ir/meta + sam3_prompts) to AFS |
| `sync_code_cluster.sh` | Local: rsync the repo itself to AFS (use before every cluster run) |
| `run_wonder_hoi.py` / `run_wonder_hoi.sh` | Upstream local driver — do not modify |

---

## 7. Troubleshooting

- **`FileNotFoundError: sam3.pt`** — AFS missing `third_party/sam3/sam3/model/checkpoints/sam3.pt`. Both `.pt` and `.safetensors` must be present.
- **`ModuleNotFoundError: flash_attn` / `sam3` / etc.** — you're running a python outside `/root/envs/<env>/bin/python`. Use the env's python explicitly (or have the driver do it, which it does).
- **HF Hub timeout during timm / edgenext load** — set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The driver already does this.
- **CUDA "invalid device ordinal" at NeuS setup** — stale `RANK` env from a DDP launcher. The driver unsets it; if you run stages manually, `unset RANK LOCAL_RANK SLURM_PROCID JSM_NAMESPACE_RANK`.
- **torch `_OutOfMemoryError` import error** — `/usr/local/cuda/compat/` shadowed the torch-bundled libcuda. `unset LD_LIBRARY_PATH` (the driver does this).
- **Silent failure in one seq, others pass** — check `$LOG_DIR/<seq>.log` for a Python `Traceback`. The driver greps for it and marks the worker FAIL.
