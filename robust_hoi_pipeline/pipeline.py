# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main pipeline orchestration for the COLMAP pipeline.
"""

import os
from pathlib import Path

import torch

from vggt.models.vggt import VGGT

from .args_config import set_seed
from .data_loading import load_images_and_intrinsics
from .track_prediction import predict_initial_tracks_wrapper, sample_points_at_track_locations
from .pose_estimation import estimate_initial_poses, filter_and_verify_tracks
from .optimization import propagate_uncertainty_and_build_image_info
from .correspondence_alignment import get_3D_correspondences, evaluate_3D_corres, align_3D_model_with_images
from .frame_management import register_remaining_frames


def setup_environment(args):
    """Setup device, dtype, output directory, and seed.

    Args:
        args: Arguments with seed and output_dir

    Returns:
        Tuple of (device, dtype) or (None, None) if skipping
    """
    print("Arguments:", vars(args))

    results_dir = Path(args.output_dir) / "results"
    if results_dir.exists():
        results_folders = [entry for entry in results_dir.iterdir() if entry.is_dir()]
        if len(results_folders) > 20:
            print(f"Found {len(results_folders)} result folders in {results_dir}, skipping.")
            return None, None

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    return device, dtype


def demo_fn(args):
    """Master orchestration function for the COLMAP pipeline.

    Coordinates all steps:
    1. Setup environment
    2. Load images and intrinsics
    3. Predict initial tracks
    4. Estimate initial poses
    5. Sample points at track locations
    6. Filter and verify tracks
    7. Propagate uncertainties and build image info
    8. Get 3D correspondences
    9. Align 3D model
    10. Register remaining frames

    Args:
        args: Parsed command-line arguments
    """
    # Step 1: Setup
    device, dtype = setup_environment(args)
    if device is None:
        return

    # Step 2: Load data
    (
        image_dir,
        image_path_list,
        base_image_path_list,
        images,
        original_coords,
        image_masks,
        depth_prior,
        intrinsic,
        depth_conf,
        vggt_fixed_resolution,
        img_load_resolution,
        gen_3d,
    ) = load_images_and_intrinsics(args, device)

    # Step 3: Predict initial tracks
    pred_tracks, pred_vis_scores, points_rgb = predict_initial_tracks_wrapper(
        images, image_masks, args, dtype
    )

    # Step 4: Estimate initial poses
    extrinsic, intrinsic, points_3d, track_mask = estimate_initial_poses(
        images, depth_prior, intrinsic, pred_tracks, pred_vis_scores, args, args.output_dir
    )

    # Step 5: Sample points at track locations
    pred_tracks, pred_vis_scores, pred_confs, sampled_points_3d, points_rgb = sample_points_at_track_locations(
        pred_tracks, pred_vis_scores, points_3d, depth_conf,
        image_masks, points_rgb, args.cond_index, images.shape
    )

    # Step 6: Filter and verify tracks
    track_mask, points_3d, pred_tracks, points_rgb = filter_and_verify_tracks(
        images, sampled_points_3d, extrinsic, intrinsic, pred_tracks, pred_vis_scores,
        points_rgb, args, args.output_dir
    )

    # Step 7: Build image info with uncertainty propagation
    image_info = propagate_uncertainty_and_build_image_info(
        images, image_path_list, base_image_path_list, original_coords,
        image_masks, depth_prior, intrinsic, extrinsic,
        pred_tracks, track_mask, points_3d, points_rgb, args
    )

    # Step 8: Get 3D correspondences
    corres_3d = get_3D_correspondences(
        gen_3d, image_info, reference_idx=args.cond_index,
        out_dir=f"{args.output_dir}/3D_corres"
    )

    # Step 9: Evaluate and align 3D model
    evaluate_3D_corres(corres_3d, gen_3d, image_info, reference_idx=args.cond_index,
                       out_dir=f"{args.output_dir}/3D_corres/eval")

    align_3D_model_with_images(corres_3d, gen_3d, image_info, reference_idx=args.cond_index,
                               out_dir=f"{args.output_dir}/aligned")

    # Step 10: Register remaining frames
    register_remaining_frames(image_info, gen_3d, args)

    print("=" * 50)
    print("Pipeline complete!")
    print("=" * 50)
