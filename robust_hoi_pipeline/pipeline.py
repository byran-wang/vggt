# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main pipeline orchestration for the COLMAP pipeline.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from vggt.models.vggt import VGGT

from .args_config import set_seed
from .data_loading import load_inputs_and_gen3d
from .track_prediction import predict_initial_tracks_wrapper, sample_points_at_track_locations
from .pose_estimation import estimate_initial_poses, filter_and_verify_tracks
from .optimization import propagate_uncertainty_and_build_image_info
from .correspondence_alignment import get_3D_correspondences, evaluate_3D_corres, align_3D_model_with_images
from .frame_management import register_key_frames
from .mask_optimization import optimize_pose_with_mask_loss


class TeeLogger:
    """Tee stdout/stderr to both console and a log file."""

    def __init__(self, log_path):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log_file = open(log_path, "w")

    def write(self, message):
        self.terminal_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def write_stderr(self, message):
        self.terminal_stderr.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal_stdout.flush()
        self.terminal_stderr.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class StderrTee:
    """Wrapper to tee stderr to a TeeLogger."""

    def __init__(self, tee_logger):
        self.tee_logger = tee_logger

    def write(self, message):
        self.tee_logger.write_stderr(message)

    def flush(self):
        self.tee_logger.flush()


def setup_logging(output_dir):
    """Setup logging to both console and file.

    Args:
        output_dir: Directory to save log file

    Returns:
        TeeLogger instance (or None if setup fails)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_dir) / f"pipeline_{timestamp}.log"

    try:
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger
        sys.stderr = StderrTee(tee_logger)
        print(f"[setup_logging] Logging to: {log_path}")
        return tee_logger
    except Exception as e:
        print(f"[setup_logging] Failed to setup logging: {e}")
        return None


def teardown_logging(tee_logger):
    """Restore stdout/stderr and close log file.

    Args:
        tee_logger: TeeLogger instance to close
    """
    if tee_logger is not None:
        sys.stdout = tee_logger.terminal_stdout
        sys.stderr = tee_logger.terminal_stderr
        tee_logger.close()


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


def robust_hoi_pipeline(args):
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
    10. Register key frames

    Args:
        args: Parsed command-line arguments
    """
    # Setup logging to file
    tee_logger = setup_logging(args.output_dir)

    try:
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
        ) = load_inputs_and_gen3d(args, device)

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
            points_rgb, args.output_dir, args.cond_index, args.vis_thresh, args.max_reproj_error, args.min_inlier_per_frame, args.min_inlier_per_track
        )

        # Step 7: Build image info with uncertainty propagation
        image_info = propagate_uncertainty_and_build_image_info(
            images, image_path_list, base_image_path_list, original_coords,
            image_masks, depth_prior, intrinsic, extrinsic,
            pred_tracks, track_mask, points_3d, points_rgb, args
        )

        # # Step 8: Get 3D correspondences
        # corres_3d = get_3D_correspondences(
        #     gen_3d, image_info, reference_idx=args.cond_index,
        #     out_dir=f"{args.output_dir}/3D_corres"
        # )

        # # Step 9: Evaluate and align 3D model
        # evaluate_3D_corres(corres_3d, gen_3d, image_info, reference_idx=args.cond_index,
        #                    out_dir=f"{args.output_dir}/3D_corres/eval")

        # align_3D_model_with_images(corres_3d, gen_3d, image_info, reference_idx=args.cond_index,
        #                            out_dir=f"{args.output_dir}/aligned")

        # Step 10: Register remaining frames
        register_key_frames(image_info, args)

        # # Step 11: Optimize poses and intrinsics using mask loss
        # image_info, gen_3d = optimize_pose_with_mask_loss(image_info, gen_3d, args)

        print("=" * 50)
        print("Pipeline complete!")
        print("=" * 50)

    finally:
        # Restore stdout and close log file
        teardown_logging(tee_logger)
