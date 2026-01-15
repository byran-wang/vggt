# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Track prediction and management functions for the COLMAP pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.dependency.track_predict import predict_tracks
from vggt.utils.visual_track import visualize_tracks_on_images


def run_VGGT(model, images, dtype, resolution=518):
    """Execute VGGT model forward pass for track prediction.

    Args:
        model: VGGT model instance
        images: Input images of shape [B, 3, H, W]
        dtype: Data type for computation
        resolution: Target resolution (default 518 for VGGT)

    Returns:
        Tuple of (extrinsic, intrinsic, depth_map, depth_conf)
    """
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def predict_initial_tracks_wrapper(images, image_masks, args, dtype):
    """Predict initial tracks using VGGSfM tracker.

    Args:
        images: Input image tensors
        image_masks: Image mask tensors
        args: Arguments containing tracking parameters
        dtype: Data type for computation

    Returns:
        Tuple of (pred_tracks, pred_vis_scores, points_rgb)
    """
    with torch.amp.autocast('cuda', dtype=dtype) and torch.no_grad():
        pred_tracks, pred_vis_scores, _, _, points_rgb = predict_tracks(
            images,
            image_masks=image_masks,
            conf=None,
            points_3d=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking,
            complete_non_vis=False,
            query_frame_indexes=[args.cond_index]
        )
    return pred_tracks, pred_vis_scores, points_rgb


def remove_duplicate_tracks(existing_tracks, new_tracks, new_track_mask, new_points_3d, new_points_rgb,
                            ref_frame_idx, dist_thresh=3.0, existing_track_mask=None):
    """Remove new track points that have similar positions to existing tracks.

    Compares track positions across all frames where both tracks are visible.
    A new track is considered a duplicate if its average distance to any existing
    track (across co-visible frames) is below the threshold.

    Args:
        existing_tracks: Existing tracks array of shape [S, N_existing, 2]
        new_tracks: New tracks array of shape [S, N_new, 2]
        new_track_mask: Visibility mask for new tracks [S, N_new]
        new_points_3d: 3D points for new tracks [N_new, 3]
        new_points_rgb: RGB colors for new tracks [N_new, 3] or None
        ref_frame_idx: Reference frame index (used as fallback)
        dist_thresh: Distance threshold in pixels to consider as duplicate
        existing_track_mask: Visibility mask for existing tracks [S, N_existing].
                             If None, assumes all existing tracks are visible in all frames.

    Returns:
        Filtered new_tracks, new_track_mask, new_points_3d, new_points_rgb
    """
    if new_tracks.shape[1] == 0:
        return new_tracks, new_track_mask, new_points_3d, new_points_rgb

    S = new_tracks.shape[0]
    N_new = new_tracks.shape[1]
    N_existing = existing_tracks.shape[1]

    # If no existing track mask provided, assume all are visible
    if existing_track_mask is None:
        existing_track_mask = np.ones((S, N_existing), dtype=bool)

    existing_track_mask = np.asarray(existing_track_mask).astype(bool)
    new_track_mask_bool = np.asarray(new_track_mask).astype(bool)

    keep_mask = np.ones(N_new, dtype=bool)

    for i in range(N_new):
        # Get frames where the new track is visible
        new_vis_frames = np.where(new_track_mask_bool[:, i])[0]
        if len(new_vis_frames) == 0:
            keep_mask[i] = False
            continue

        # For each existing track, compute average distance across co-visible frames
        min_avg_dist = np.inf
        for j in range(N_existing):
            # Find frames where both tracks are visible
            existing_vis_frames = np.where(existing_track_mask[:, j])[0]
            co_visible_frames = np.intersect1d(new_vis_frames, existing_vis_frames)

            if len(co_visible_frames) == 0:
                continue

            # Compute distances across co-visible frames
            new_pts = new_tracks[co_visible_frames, i, :]  # [K, 2]
            existing_pts = existing_tracks[co_visible_frames, j, :]  # [K, 2]
            distances = np.linalg.norm(new_pts - existing_pts, axis=1)  # [K]

            avg_dist = distances.mean()
            min_avg_dist = min(min_avg_dist, avg_dist)

        # If minimum average distance is below threshold, mark as duplicate
        if min_avg_dist < dist_thresh:
            keep_mask[i] = False

    num_removed = (~keep_mask).sum()
    if num_removed > 0:
        print(f"[remove_duplicate_tracks] Removed {num_removed} duplicate tracks (dist_thresh={dist_thresh}px, checked across co-visible frames)")

    new_tracks = new_tracks[:, keep_mask]
    new_track_mask = new_track_mask[:, keep_mask]
    new_points_3d = new_points_3d[keep_mask]
    if new_points_rgb is not None:
        new_points_rgb = new_points_rgb[keep_mask]

    return new_tracks, new_track_mask, new_points_3d, new_points_rgb


def prep_valid_correspondences(points_3d, track_mask, min_inlier_per_frame, min_inlier_per_track):
    """Filter tracks by per-frame/track counts and drop 3D points with no surviving tracks.

    Args:
        points_3d: 3D point coordinates
        track_mask: Track visibility mask
        min_inlier_per_frame: Minimum inliers required per frame
        min_inlier_per_track: Minimum inliers required per track

    Returns:
        Tuple of (filtered_mask, filtered_points_3d, keep_indices)
    """
    mask = np.copy(track_mask)
    if min_inlier_per_frame > 0:
        per_frame = mask.sum(axis=1)
        bad_frames = per_frame < min_inlier_per_frame
        if bad_frames.any():
            mask[bad_frames] = False

    if min_inlier_per_track > 0:
        per_track = mask.sum(axis=0)
        bad_tracks = per_track < min_inlier_per_track
        if bad_tracks.any():
            mask[:, bad_tracks] = False

    keep_pts = mask.sum(axis=0) > 0
    mask = mask[:, keep_pts]
    points_3d = points_3d[keep_pts]
    return mask, points_3d, keep_pts


def sample_points_at_track_locations(pred_tracks, pred_vis_scores, points_3d, depth_conf,
                                      image_masks, points_rgb, query_index, image_shape):
    """Sample points_3d and depth_conf at query point locations.

    This replicates the logic from predict_tracks/_forward_on_query without re-running tracking.

    Args:
        pred_tracks: Predicted track positions
        pred_vis_scores: Predicted visibility scores
        points_3d: 3D point map
        depth_conf: Depth confidence map
        image_masks: Image masks
        points_rgb: Point RGB colors
        query_index: Query frame index
        image_shape: Image shape tuple

    Returns:
        Tuple of (pred_tracks, pred_vis_scores, pred_confs, sampled_points_3d, points_rgb)
    """
    height, width = image_shape[-2:]
    query_points = pred_tracks[query_index]  # Shape: [N, 2]

    if depth_conf is not None and points_3d is not None:
        assert height == width
        assert depth_conf.shape[-2] == depth_conf.shape[-1]
        assert depth_conf.shape[:3] == points_3d.shape[:3]
        scale = depth_conf.shape[-1] / width

        query_points_scaled = np.round(query_points * scale).astype(np.int64)

        pred_confs = depth_conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        sampled_points_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        if image_masks is not None:
            image_masks_np = image_masks.cpu().numpy()
            query_points_raw = np.round(query_points).astype(np.int64)
            pred_confs = pred_confs * image_masks_np[query_index][0][query_points_raw[:, 1], query_points_raw[:, 0]]

        # Heuristic to remove low confidence points
        valid_mask = pred_confs > 1.2
        if valid_mask.sum() > 512:
            pred_tracks = pred_tracks[:, valid_mask]
            pred_vis_scores = pred_vis_scores[:, valid_mask]
            pred_confs = pred_confs[valid_mask]
            sampled_points_3d = sampled_points_3d[valid_mask]
            if points_rgb is not None:
                points_rgb = points_rgb[valid_mask]
    else:
        pred_confs = None
        sampled_points_3d = points_3d

    return pred_tracks, pred_vis_scores, pred_confs, sampled_points_3d, points_rgb
