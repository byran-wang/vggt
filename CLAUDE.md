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

## Best SAM3D IDs generated by auto mode (2026-04-24)

| Sequence | BestID | Score    | Coverage | Faces             |
|----------|--------|----------|----------|-------------------|
| ABF12    | 0120   | N/A      | 3/6      | X+,Y-,Z+          |
| ABF14    | 0405   | 0.007471 | 4/6      | X+,X-,Y+,Z+       |
| GPMF12   | 0239   | 0.010542 | 3/6      | X-,Y-,Z+          |
| GPMF14   | 0410   | 0.001299 | 3/6      | X+,Y+,Z+          |
| MC1      | 0700   | 0.005518 | 3/6      | X-,Y+,Z+          |
| MC4      | 0155   | 0.008594 | 2/6      | X+,Z+             |
| MDF12    | 1755   | N/A      | 4/6      | X-,Y+,Z+,Z-       |
| MDF14    | 1790   | 0.004230 | 4/6      | X+,Y-,Z+,Z-       |
| ShSu10   | 0518   | 0.001345 | 3/6      | X-,Y-,Z+          |
| ShSu12   | 0550   | N/A      | 3/6      | X-,Y+,Z+          |
| SM2      | 0018   | 0.002805 | 3/6      | X-,Y+,Z+          |
| SM4      | 0630   | N/A      | 3/6      | X+,Y-,Z-          |
| SMu1     | 0780   | 0.001863 | 3/6      | X-,Y-,Z+          |
| SMu40    | 0400   | 0.009716 | 3/6      | X+,Y+,Z-          |
| BB12     | 0485   | N/A      | 5/6      | X+,Y+,Y-,Z+,Z-    |
| BB13     | 1024   | 0.004680 | 3/6      | X+,Y-,Z+          |
| GSF12    | 0940   | 0.014783 | 4/6      | X-,Y-,Z+,Z-       |
| GSF13    | 0755   | 0.033960 | 2/6      | X-,Y-             |

## SOTA Results (2026-04-27) — sam3d semi-auto

| Sequence | ADD AUC | ADD-S AUC | Total Frames | Reg Frames | Keyframes | SAM3D CD | SAM3D F5 | NeuS CD | NeuS F5 | MPJPE RA | CD Right |
|----------|---------|-----------|--------------|------------|-----------|----------|----------|---------|---------|----------|----------|
| ABF12    | 76.89   | 91.36     | 277          | 277        | 248       | 1.72     | 32.21    | 1.03    | 57.53   | 23.45    | 7.34     |
| ABF14    | 85.90   | 94.94     | 277          | 277        | 272       | 0.65     | 77.56    | 0.32    | 99.87   | 29.65    | 3.69     |
| BB12     | 81.08   | 92.85     | 322          | 322        | 255       | 1.52     | 58.11    | 0.46    | 93.22   | 23.11    | 2.95     |
| BB13     | 79.77   | 93.07     | 322          | 322        | 294       | 0.91     | 67.80    | 0.40    | 93.58   | 14.64    | 2.70     |
| GPMF12   | 59.28   | 91.25     | 219          | 219        | 158       | 0.32     | 100.00   | 0.99    | 59.88   | 20.83    | 5.34     |
| GPMF14   | 87.19   | 94.78     | 219          | 219        | 208       | 1.73     | 40.51    | 0.61    | 79.92   | 26.65    | 2.58     |
| GSF12    | 77.25   | 86.62     | 299          | 299        | 162       | 0.53     | 90.94    | 0.32    | 97.67   | 17.26    | 5.66     |
| GSF13    | 93.52   | 97.19     | 319          | 319        | 189       | 0.51     | 88.71    | 0.41    | 92.48   | 13.59    | 1.96     |
| MC1      | 93.87   | 96.68     | 180          | 180        | 170       | 1.67     | 49.07    | 0.67    | 89.48   | 9.55     | 2.24     |
| MC4      | 88.87   | 94.72     | 180          | 180        | 170       | 0.40     | 96.41    | 0.79    | 78.15   | 14.87    | 15.54    |
| MDF12    | 94.99   | 97.37     | 562          | 562        | 394       | 0.62     | 80.55    | 0.59    | 86.47   | 11.89    | 2.11     |
| MDF14    | 79.06   | 89.11     | 562          | 562        | 377       | 1.46     | 44.31    | 0.77    | 74.88   | 19.34    | 2.86     |
| ShSu10   | 89.89   | 94.43     | 370          | 370        | 346       | 1.19     | 29.43    | 0.51    | 92.12   | 8.17     | 2.39     |
| ShSu12   | 91.64   | 96.08     | 370          | 370        | 200       | 0.94     | 69.05    | 0.35    | 99.49   | 9.89     | 26.18    |
| SM2      | 91.75   | 96.10     | 180          | 180        | 179       | 0.79     | 67.84    | 0.56    | 85.97   | 10.33    | 3.05     |
| SM4      | 94.25   | 97.12     | 180          | 180        | 178       | 0.59     | 84.03    | 0.49    | 88.10   | 9.78     | 18.82    |
| SMu1     | 94.21   | 97.36     | 359          | 359        | 282       | 0.96     | 57.86    | 0.74    | 80.09   | 12.84    | 16.58    |
| SMu40    | 70.40   | 96.31     | 400          | 400        | 300       | 0.45     | 94.94    | 1.14    | 48.55   | 11.93    | 1.61     |
| **Avg**  | **84.99** | **94.30** | **310.94** | **310.94** | **243.44** | **0.94** | **68.30** | **0.62** | **83.19** | **15.99** | **6.87** |

Metrics: ADD AUC / ADD-S AUC (object pose, higher=better), SAM3D CD / NeuS CD (chamfer dist cm, lower=better), SAM3D F5 / NeuS F5 (F-score@5mm %, higher=better), MPJPE RA (hand mm, lower=better), CD Right (hand cm, lower=better).
