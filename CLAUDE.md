# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This projec can reconstruct hand and object interaction from RGBD images offline with 3D prior. Given a RGBD sequncese, it directly infers:
- Object shape 
- Object poses
- Hand poses


## Key Commands

### Installation
```bash
# Create environment
conda create -n vggsfm_tmp python=3.10 -y
conda activate vggsfm_tmp
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# For demos (visualization)
pip install -r requirements_demo.txt
```

### Running the Model
```bash
export seq_list="MC1"
# object reconstruction
cd ~/Documents/project/WonderHOI/code
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_joint_optimization --seq_list $seq_list --rebuild

# hand reconstruction
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot --seq_list $seq_list --rebuild --dataset_type ho3d 

# visulize the object and hand results
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_joint_optimization --seq_list $seq_list --vis

# evaluate the results
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_eval_intrinsic ho3d_eval_trans ho3d_eval_rot --seq_list $seq_list
python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum_intrinsic eval_sum_trans eval_sum_rot --seq_list $seq_list


```
## Data Structure

### HO3D_v3 Dataset Structure
```
ho3d_v3/
├── calibration/           # Camera calibration per subject (ABF1, AP1, BB1, ...)
│   └── {subject}/calibration
├── train/                 # Training sequences
│   └── {seq_id}/          # e.g., ABF10, BB10, MC1, ...
│       ├── rgb/           # RGB images (0000.jpg, 0001.jpg, ...)
│       ├── depth/         # Depth images (0000.png, 0001.png, ...)
│       ├── meta/          # Metadata pickle files (0000.pkl, 0001.pkl, ...)
│       ├── hands/         # Hand fitting data
│       │   ├── 2d_keypoints/
│       │   ├── hpe_vis/
│       │   ├── renders/
│       │   ├── hold_fit.init.npy
│       │   ├── j2d.full.npy
│       │   └── v3d.npy
│       ├── mask_hand/     # Hand segmentation masks
│       └── inpaint_origin/
├── masks_XMem/            # Object segmentation masks per sequence
│   └── {seq_id}/          # 00000.png, 00001.png, ...
├── models/                # YCB object 3D models
│   └── {object_id}/       # e.g., 003_cracker_box, 006_mustard_bottle
│       ├── textured.obj
│       ├── textured.mtl
│       ├── texture_map.png
│       ├── points.xyz
│       └── *.xml
├── evaluation/            # Evaluation sequences
├── processed/             # Preprocessed data cache
└── train.txt, evaluation.txt  # Sequence lists
```

## Output Result Structure
```
output/
├── metrics_summary/                    # Aggregated evaluation metrics
│   ├── eval_intrinsic.txt
│   ├── eval_trans.txt
│   ├── eval_rot.txt
│   └── eval_results.txt
└── {seq_id}/                           # Per-sequence output (e.g., MC1, ABF14, ...)
    ├── data_processed/
    │   └── preprocessed_640_0.pt       # Cached preprocessed tensors
    ├── track_raw/                      # Raw tracking visualizations
    │   └── frame_XXXX.png, tracks_grid.png
    ├── track_filter_vis_thresh/        # Tracks filtered by visibility
    ├── track_filter_max_proj_err/      # Tracks filtered by projection error
    ├── track_filter_frame_track_inlier/# Tracks filtered by inlier count
    ├── 3D_corres/                      # 3D correspondence results
    │   ├── track_raw/
    │   ├── track_vis/
    │   └── eval/
    │       └── corres_XXX.png
    ├── aligned/                        # Alignment results (gen mesh to GT)
    │   ├── transform.json              # Transformation matrix
    │   └── eval/
    │       ├── cond.ply, ref.ply, cond_aligned.ply
    │       └── transform.png
    └── results/                        # Per-frame reconstruction results
        └── {frame_id}/                 # e.g., 0000, 0001, ...
            ├── results.pkl             # Full results dictionary
            ├── points.ply              # Reconstructed 3D points
            ├── white_mesh_remesh_aligned.obj  # Aligned mesh
            ├── reproj_error.png        # Reprojection error visualization
            └── depth_conf/             # Depth confidence maps
                └── depth_unc_XXXX.png
```

## Third-Party Dependencies
- `third_party/utils_simba/`: Custom utilities for rendering, depth processing
- `dependency/LightGlue/`: Feature matching (installed separately)
