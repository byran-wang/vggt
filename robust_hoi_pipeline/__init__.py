# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
COLMAP Pipeline Package

This package provides a modular pipeline for 3D reconstruction from RGBD sequences.
"""

from .args_config import parse_args, set_seed
from .data_loading import load_inputs_and_gen3d, get_image_list, save_intrinsics
from .track_prediction import (
    run_VGGT,
    predict_initial_tracks_wrapper,
    remove_duplicate_tracks,
    prep_valid_correspondences,
    sample_points_at_track_locations,
)
from .pose_estimation import (
    estimate_initial_poses,
    estimate_extrinsic,
    verify_tracks_by_geometry,
    filter_and_verify_tracks,
)
from .optimization import (
    propagate_uncertainties,
    bundle_adjust_keyframes,
    build_reconstruction_from_tracks,
    register_new_frame,
    propagate_uncertainty_and_build_image_info,
)
from .correspondence_alignment import (
    get_3D_correspondences,
    evaluate_3D_corres,
    eval_aligned_3D_model,
    align_3D_model_with_images,
    save_aligned_3D_model,
)
# Alias for compatibility
evaluate_3d_corres = evaluate_3D_corres
from .frame_management import (
    find_next_frame,
    check_frame_invalid,
    check_key_frame,
    process_key_frame,
    register_key_frames,
)
from .visualization_io import (
    save_results,
    save_input_data,
    eval_reprojection,
    get_points_uncertainty_colors,
    save_point_cloud_with_conf,
    save_depth_prior_with_uncertainty,
    save_depth_point_clouds,
    save_fused_point_cloud,
)
from .geometry_utils import (
    compute_normals_from_depth,
    axis_angle_to_matrix,
    adjust_intrinsic_for_new_image_size,
    rename_colmap_recons_and_rescale_camera,
)
from .tsdf_fusion import (
    fuse_depth_to_mesh,
    compute_volume_bounds,
    select_keyframes,
    visualize_tsdf_fusion_rerun,
)
from .pipeline import robust_hoi_pipeline, setup_environment

__all__ = [
    # args_config
    "parse_args",
    "set_seed",
    # data_loading
    "load_inputs_and_gen3d",
    "get_image_list",
    "save_intrinsics",
    # track_prediction
    "run_VGGT",
    "predict_initial_tracks_wrapper",
    "remove_duplicate_tracks",
    "prep_valid_correspondences",
    "sample_points_at_track_locations",
    # pose_estimation
    "estimate_initial_poses",
    "estimate_extrinsic",
    "verify_tracks_by_geometry",
    "filter_and_verify_tracks",
    # optimization
    "propagate_uncertainties",
    "bundle_adjust_keyframes",
    "build_reconstruction_from_tracks",
    "register_new_frame",
    "propagate_uncertainty_and_build_image_info",
    # correspondence_alignment
    "get_3D_correspondences",
    "evaluate_3D_corres",
    "eval_aligned_3D_model",
    "align_3D_model_with_images",
    "save_aligned_3D_model",
    # frame_management
    "find_next_frame",
    "check_frame_invalid",
    "check_key_frame",
    "process_key_frame",
    "register_key_frames",
    # visualization_io
    "save_results",
    "save_input_data",
    "eval_reprojection",
    "get_points_uncertainty_colors",
    "save_point_cloud_with_conf",
    "save_depth_prior_with_uncertainty",
    "save_depth_point_clouds",
    "save_fused_point_cloud",
    # geometry_utils
    "compute_normals_from_depth",
    "axis_angle_to_matrix",
    "adjust_intrinsic_for_new_image_size",
    "rename_colmap_recons_and_rescale_camera",
    # tsdf_fusion
    "fuse_depth_to_mesh",
    "compute_volume_bounds",
    "select_keyframes",
    "visualize_tsdf_fusion_rerun",
    # pipeline
    "robust_hoi_pipeline",
    "setup_environment",
]
