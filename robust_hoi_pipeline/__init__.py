# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
COLMAP Pipeline Package

This package provides a modular pipeline for 3D reconstruction from RGBD sequences.

Submodule re-exports below pull in heavy SfM / torch / vggt / lightglue chains.
Each is wrapped in try/except so that consumers needing only a subset (e.g.
pipeline_blender_rendering in a slim Blender env without nvdiffrast / lightglue)
can `from robust_hoi_pipeline.X import Y` without the package init failing.
"""

import warnings as _warnings


def _safe_star(_module: str, _names: list) -> None:
    """Try to re-export `_names` from `.{_module}`; warn (don't raise) if deps missing."""
    try:
        mod = __import__(f"robust_hoi_pipeline.{_module}", fromlist=_names)
    except ImportError as e:
        _warnings.warn(
            f"robust_hoi_pipeline.{_module} unavailable ({type(e).__name__}: {e}); "
            f"affected names: {_names}",
            stacklevel=2,
        )
        return
    for _n in _names:
        if hasattr(mod, _n):
            globals()[_n] = getattr(mod, _n)


_safe_star("args_config", ["parse_args", "set_seed"])
_safe_star("data_loading", ["load_inputs_and_gen3d", "get_image_list", "save_intrinsics"])
_safe_star("track_prediction", [
    "run_VGGT", "predict_initial_tracks_wrapper", "remove_duplicate_tracks",
    "prep_valid_correspondences", "sample_points_at_track_locations",
])
_safe_star("pose_estimation", [
    "estimate_initial_poses", "estimate_extrinsic",
    "verify_tracks_by_geometry", "filter_and_verify_tracks",
])
_safe_star("optimization", [
    "propagate_uncertainties", "bundle_adjust_keyframes",
    "build_reconstruction_from_tracks", "register_new_frame_by_PnP",
    "propagate_uncertainty_and_build_image_info",
])
_safe_star("correspondence_alignment", [
    "get_3D_correspondences", "evaluate_3D_corres", "eval_aligned_3D_model",
    "align_3D_model_with_images", "save_aligned_3D_model",
])
# Alias for compatibility
evaluate_3d_corres = globals().get("evaluate_3D_corres")
_safe_star("frame_management", [
    "find_next_frame", "check_frame_invalid", "check_key_frame",
    "process_key_frame", "register_key_frames",
])
_safe_star("visualization_io", [
    "save_results", "save_input_data", "eval_reprojection",
    "get_points_uncertainty_colors", "save_point_cloud_with_conf",
    "save_depth_prior_with_uncertainty", "save_depth_point_clouds",
    "save_fused_point_cloud",
])
_safe_star("geometry_utils", [
    "compute_normals_from_depth", "axis_angle_to_matrix",
    "adjust_intrinsic_for_new_image_size", "rename_colmap_recons_and_rescale_camera",
])
_safe_star("tsdf_fusion", [
    "fuse_depth_to_mesh", "compute_volume_bounds", "select_keyframes",
    "visualize_tsdf_fusion_rerun",
])
_safe_star("pipeline", ["robust_hoi_pipeline", "setup_environment"])

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
    "register_new_frame_by_PnP",
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
