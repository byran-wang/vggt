# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Argument parsing and configuration for the COLMAP pipeline.
"""

import argparse
import random
import numpy as np
import torch


def parse_args():
    """Parse command-line arguments for VGGT Demo.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_sfm", action="store_true", default=False, help="Use SfM for reconstruction")

    # BA parameters
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--dataset_type", type=str, default="ZED", help="Dataset type for SfM preprocessing")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument("--min_depth_pixels", type=int, default=500, help="Minimum valid depth pixels to accept a frame")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--use_calibrated_intrinsic", action="store_true", default=False, help="Use calibrated intrinsic for reconstruction")
    parser.add_argument("--min_inlier_per_frame", type=int, default=10, help="Minimum inliers per frame for BA")
    parser.add_argument("--min_inlier_per_track", type=int, default=4, help="Minimum inliers per track for BA")
    parser.add_argument("--min_frame_num", type=int, default=0, help="Minimum number of frames to process")
    parser.add_argument("--max_frame_num", type=int, default=50, help="Maximum number of frames to process")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval for processing")
    parser.add_argument("--min_PnP_inlier_num", type=int, default=200, help="Minimum inliers for registration")
    parser.add_argument("--instance_id", type=int, default=0, help="Instance ID for image preprocessing")
    parser.add_argument("--cond_index", type=int, default=0, help="Conditioning frame index for tracking")
    parser.add_argument("--cond_index_raw", type=int, default=0, help="Conditioning frame index for tracking")

    # Keyframe thresholds
    parser.add_argument("--kf_rot_thresh", type=float, default=5.0, help="Keyframe rotation threshold (degrees)")
    parser.add_argument("--kf_trans_thresh", type=float, default=0.02, help="Keyframe translation threshold (units)")
    parser.add_argument("--kf_depth_thresh", type=float, default=500, help="Keyframe depth change threshold (units)")
    parser.add_argument("--kf_inlier_thresh", type=int, default=10, help="Keyframe inlier count threshold")
    parser.add_argument("--min_track_number", type=int, default=5, help="Minimum track number for 3D point uncertainty; points with fewer tracks get high uncertainty")
    parser.add_argument("--run_ba_on_keyframe", type=int, default=0, help="Run bundle adjustment on keyframes")

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")
