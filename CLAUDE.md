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

## SOTA Results (2026-04-09)

| Sequence | ADD AUC | ADD-S AUC | Total Frames | Reg Frames | Keyframes | SAM3D CD | SAM3D F5 | MPJPE RA | CD Right |
|----------|---------|-----------|--------------|------------|-----------|----------|----------|----------|----------|
| ABF12    | 76.89   | 93.65     | 277          | 277        | 172       | 0.66     | 79.00    | 23.85    | 3.07     |
| ABF14    | 83.43   | 94.83     | 277          | 277        | 146       | 0.83     | 63.45    | 30.50    | 3.26     |
| BB12     | 44.27   | 67.38     | 322          | 322        | 270       | 0.64     | 83.00    | 22.71    | 11.16    |
| BB13     | 78.29   | 92.16     | 323          | 323        | 284       | 0.33     | 97.09    | 14.21    | 2.12     |
| GPMF12   | 76.39   | 93.40     | 220          | 220        | 153       | 0.37     | 95.71    | 20.80    | 5.10     |
| GPMF14   | 94.06   | 97.19     | 219          | 219        | 216       | 0.50     | 84.76    | 27.25    | 3.15     |
| GSF12    | 87.16   | 93.77     | 279          | 279        | 177       | 0.41     | 94.16    | 17.18    | 2.56     |
| GSF13    | 91.49   | 96.53     | 319          | 319        | 236       | 0.38     | 95.16    | 13.19    | 1.59     |
| MC1      | 91.05   | 95.44     | 181          | 181        | 158       | 0.57     | 86.97    | 9.55     | 2.20     |
| MC4      | 85.57   | 92.82     | 180          | 180        | 160       | 1.48     | 54.43    | 10.99    | 2.80     |
| MDF12    | 80.65   | 90.84     | 562          | 562        | 417       | 1.57     | 57.73    | 13.46    | 4.44     |
| MDF14    | 94.27   | 97.11     | 562          | 562        | 299       | 1.09     | 58.20    | 20.78    | 1.82     |
| ShSu10   | 92.24   | 95.67     | 371          | 371        | 193       | 0.38     | 97.40    | 7.62     | 1.34     |
| ShSu12   | 88.19   | 94.68     | 371          | 371        | 170       | 1.59     | 45.56    | 9.34     | 2.04     |
| SM2      | 93.49   | 96.93     | 181          | 181        | 170       | 0.43     | 91.96    | 10.31    | 2.82     |
| SM4      | 92.51   | 95.79     | 180          | 180        | 153       | 0.98     | 51.01    | 7.25     | 1.48     |
| SMu1     | 94.10   | 97.34     | 360          | 360        | 358       | 0.44     | 94.90    | 10.42    | 1.53     |
| SMu40    | 92.52   | 97.23     | 400          | 400        | 276       | 0.33     | 97.37    | 11.93    | 1.34     |
| **Avg**  | **85.36** | **93.49** | **310.22** | **310.22** | **222.67** | **0.72** | **79.33** | **15.63** | **2.99** |

Metrics: ADD AUC / ADD-S AUC (object pose, higher=better), SAM3D CD (chamfer dist cm, lower=better), SAM3D F5 (F-score@5mm %, higher=better), MPJPE RA (hand mm, lower=better), CD Right (hand cm, lower=better).
