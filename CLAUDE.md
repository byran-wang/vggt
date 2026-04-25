# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT-based hand-object interaction reconstruction from RGBD sequences. Given an RGBD sequence, it reconstructs object shape, object poses, and hand poses using 3D priors (SAM3D, FoundationPose, NeuS).

## Environment

```bash
conda activate vggsfm_tmp  # Python 3.10, PyTorch 2.1.0 (CUDA 11.8)
export DATASET=ho3d         # or: zed, rs_zijian, zed_zijian
export RUN_ON_SERVER=false   # true on server (changes paths in confs/sequence_config.py)
```

## Key Commands

```bash
seq_list="MC1"

# Hand pose fitting
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans --seq_list $seq_list --rebuild --dataset_type ho3d

# SAM3D generation & alignment
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_gen ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts --seq_list $seq_list --rebuild

# SAM3D post-processing
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_SAM3D_post_process --seq_list $seq_list --rebuild

# Data preprocessing & correspondence
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres --seq_list $seq_list --rebuild

# SAM3D NeuS data preprocessing
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess_sam3d_neus --seq_list $seq_list --rebuild

# Joint optimization
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --rebuild

# Hand-object alignment (hand, rotation, object, hand+object)
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho --seq_list $seq_list --rebuild

# Evaluation & visualization
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_eval hoi_pipeline_eval_vis --seq_list $seq_list --rebuild

# Summary evaluation
python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum eval_sum_vis --seq_list $seq_list --rebuild
```

## Architecture

### Orchestration
- **run_wonder_hoi.py**: Central entry point. Maps `--execute_list` + `--process_list` to pipeline functions. Supports `--rebuild` (clear and regenerate), `--vis`, `--eval` flags.
- **confs/sequence_config.py**: Routes to dataset-specific configs (e.g. `sequence_config_ho3d.py`) based on `DATASET` env var. Each sequence has `cond_idx`, `obj_num`, frame ranges.
- **run.sh**, **run_wonder_hoi.sh**: Shell scripts chaining full pipeline stages.

### Pipeline Stages (`robust_hoi_pipeline/`)
The pipeline processes frames through these stages:

1. **Data preprocessing** (`pipeline_data_preprocess.py`): Load images, depth, masks, intrinsics → cached `.pt` files
2. **SAM3D filtering** (`pipeline_sam3d_filter_2D.py`, `pipeline_sam3d_filter_3D.py`): Filter SAM3D 3D reconstructions by 2D/3D consistency
3. **Alignment** (`pipeline_sam3d_align.py`, `correspondence_alignment.py`): Align SAM3D mesh to image observations
4. **Joint optimization** (`pipeline_joint_opt.py`, 112KB): Core module. Registers frames via PnP + RANSAC, refines poses with reprojection + depth + contact losses, integrates FoundationPose tracking, runs bundle adjustment
5. **Evaluation** (`pipeline_joint_opt_eval.py`): Computes rotation, translation, intrinsic errors vs GT

### Key Pipeline Module: `pipeline_joint_opt.py`
- `prepare_joint_opt_inputs()`: Loads preprocessed data, VGGSfM tracks, SAM3D transform
- `register_first_frame()`: Lifts 2D tracks to 3D at condition frame
- `register_remaining_frames()`: Iteratively registers frames via PnP, FoundationPose fallback, depth alignment
- `joint_optimization()`: Bundle adjustment over keyframes with reprojection + point-to-plane losses
- `main()`: Orchestrates the full pipeline with logging to `output/{seq}/pipeline_joint_opt/log.txt`
- FoundationPose estimator is lazily cached in `_foundation_pose_cache` dict

### Visualization (`viewer/`)
- **viewer_step.py**: Rerun-based interactive viewer for per-frame results
- **viewer_distance.py**: Hand-object distance visualization (ARCTIC InterField style)
- Pipeline vis modules: `pipeline_joint_opt_vis.py`, `pipeline_joint_opt_eval_vis_rerun.py`, `pipeline_joint_opt_eval_vis_gt.py`

### Coordinate Conventions
- **Extrinsics**: `o2c` = object-to-camera (4x4), `c2o = inv(o2c)` = camera-to-object (world)
- **SAM3D poses**: `camera.json["blw2cvc"]` contains scaled o2c; extract scale from rotation columns
- **Depth scale**: `depth_scale` from preprocessed data scales GT space to SAM3D space (divide by it)
- **GT data** (`gt.load_data`): Returns xdict with `o2c`, `is_valid`, `K`, `v3d_c.right` (hand verts in cam), `mesh_name.object`, etc.

### Coding Rules
- **Logging**: Never use `print()`. Always use `from utils_simba.logger import get_logger; logger = get_logger(__name__)` for colored terminal output. `ColoredFormatter` in `third_party/utils_simba/utils_simba/logger.py` provides level-colored output. `pipeline_joint_opt.py` adds a `FileHandler` to also write logs to `log.txt`.
- **Rerun visualization**: Use helper functions from `utils_simba.rerun` (`log_camera_frame`, `load_mesh_as_trimesh`, `get_vertex_colors`, `stamp_frame_text`, `backproject_depth_to_points`). Do not call raw `rr.log` for cameras/meshes when a helper exists.
- **Depth processing**: Use functions from `utils_simba.depth` (`get_depth`, `depth2xyzmap`). Do not write custom depth loading/conversion code.
- **Debug helpers**: All debug/visualization functions for `pipeline_joint_opt.py` live in `robust_hoi_pipeline/pipeline_joint_opt_debug.py`. Do not add debug functions directly to `pipeline_joint_opt.py`; add them to the debug module and import them.

## Data Structure

### HO3D_v3 Dataset (`ho3d_v3/`)
```
ho3d_v3/
├── calibration/{subject}/     # Camera calibration
├── train/{seq_id}/            # rgb/, depth/, meta/, hands/, mask_object/, mask_hand/
├── models/{object_id}/        # YCB 3D models (textured.obj, etc.)
├── processed/{seq_id}.pt      # Preprocessed GT data (used by gt.load_data)
└── evaluation/
```

### Output (`output/{seq_id}/`)
```
output/{seq_id}/
├── pipeline_preprocess/       # Preprocessed frames, frame_list.txt
├── SAM3D_aligned_post_process/# SAM3D results with camera.json per frame
├── pipeline_joint_opt/log.txt # Pipeline log (ANSI colored)
├── results/{frame_id}/        # results.pkl, points.ply, mesh.obj, reproj_error.png
└── metrics_summary/           # Aggregated eval metrics
```

## Third-Party Dependencies
- `third_party/utils_simba/`: Rendering, depth processing, logging utilities
- `third_party/FoundationPose/`: Object pose estimation
- `third_party/instant-nsr-pl/`: NeuS neural SDF (entry: `launch.py`)
- `third_party/SAM3/`: Segment Anything 3D
- `third_party/HAMER/`: Hand pose estimation
- `dependency/LightGlue/`: Feature matching (install: `cd dependency/LightGlue && pip install -e .`)

## HO3D Sequences
19 sequences configured in `confs/sequence_config_ho3d.py`: ABF12, ABF14, GPMF12, GPMF14, MC1, MC4, MDF12, MDF14, ShSu10, ShSu12, ShSu14, SM2, SM4, SMu1, SMu40, BB12, BB13, GSF12, GSF13. Each has a `cond_idx` (conditioning frame) and frame range.

**Note**: ABF12/ABF14 frames beyond 1135 have bad GT annotations (marked invalid in `gt.load_data`).

## SOTA Results (2026-04-24)

| Sequence | ADD AUC | ADD-S AUC | Total Frames | Reg Frames | Keyframes | SAM3D CD | SAM3D F5 | NeuS CD | NeuS F5 | MPJPE RA | CD Right |
|----------|---------|-----------|--------------|------------|-----------|----------|----------|---------|---------|----------|----------|
| ABF12    | 92.96   | 96.73     | 277          | 277        | 171       | 0.70     | 76.01    | 0.54    | 86.60   | 23.45    | 6.72     |
| ABF14    | 90.41   | 95.70     | 277          | 277        | 144       | 0.95     | 53.79    | 0.37    | 99.20   | 29.65    | 3.25     |
| BB12     | 81.85   | 93.12     | 322          | 322        | 280       | 0.63     | 86.06    | 0.26    | 97.91   | 23.11    | 2.91     |
| BB13     | 83.70   | 93.82     | 323          | 323        | 283       | 0.70     | 79.68    | 0.31    | 96.38   | 14.65    | 2.68     |
| GPMF12   | 61.05   | 91.40     | 220          | 220        | 170       | 0.40     | 94.69    | 1.03    | 55.39   | 20.82    | 5.37     |
| GPMF14   | 94.43   | 97.40     | 219          | 219        | 216       | 0.51     | 83.40    | 0.40    | 93.98   | 26.65    | 2.12     |
| GSF12    | 77.32   | 84.54     | 299          | 299        | 166       | 0.41     | 94.15    | 0.37    | 95.97   | 17.26    | 5.90     |
| GSF13    | 92.45   | 96.65     | 319          | 319        | 229       | 0.41     | 91.81    | 0.41    | 93.96   | 13.59    | 1.95     |
| MC1      | 93.94   | 96.64     | 181          | 181        | 160       | 0.54     | 86.38    | 0.54    | 92.07   | 9.55     | 2.22     |
| MC4      | 88.39   | 94.49     | 180          | 180        | 161       | 0.95     | 71.65    | 0.71    | 77.75   | 14.87    | 15.50    |
| MDF12    | 95.18   | 97.28     | 562          | 562        | 403       | 1.87     | 37.09    | 0.54    | 88.69   | 11.89    | 2.13     |
| MDF14    | 95.09   | 97.28     | 562          | 562        | 293       | 1.15     | 55.82    | 0.53    | 88.93   | 19.34    | 1.56     |
| ShSu10   | 91.71   | 95.50     | 371          | 371        | 194       | 0.41     | 97.31    | 0.31    | 99.90   | 8.18     | 2.09     |
| ShSu12   | 80.79   | 91.10     | 371          | 371        | 181       | 1.43     | 53.27    | 0.76    | 73.61   | 9.88     | 25.21    |
| SM2      | 92.64   | 96.91     | 181          | 181        | 167       | 0.46     | 89.80    | 0.50    | 94.37   | 10.31    | 2.98     |
| SM4      | 93.98   | 96.80     | 180          | 180        | 149       | 0.95     | 53.32    | 0.37    | 97.77   | 9.78     | 18.90    |
| SMu1     | 94.88   | 97.69     | 360          | 360        | 354       | 0.58     | 92.86    | 0.56    | 89.74   | 12.83    | 16.69    |
| SMu40    | 93.20   | 97.35     | 400          | 400        | 273       | 0.35     | 96.99    | 1.09    | 55.54   | 11.93    | 1.33     |
| **Avg**  | **88.55** | **95.02** | **311.33** | **311.33** | **221.89** | **0.74** | **77.45** | **0.53** | **87.65** | **15.99** | **6.64** |

Metrics: ADD AUC / ADD-S AUC (object pose, higher=better), SAM3D CD / NeuS CD (chamfer dist cm, lower=better), SAM3D F5 / NeuS F5 (F-score@5mm %, higher=better), MPJPE RA (hand mm, lower=better), CD Right (hand cm, lower=better).
