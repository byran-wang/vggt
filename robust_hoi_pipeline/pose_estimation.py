# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Pose estimation functions for the COLMAP pipeline.
"""

import numpy as np
import torch
import cv2

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.dependency.projection import project_3D_points_np

from .track_prediction import prep_valid_correspondences


def estimate_extrinsic(depth_map, extrinsics, intrinsic, tracks, track_mask, ref_index=0, ransac_reproj_threshold=8.0):
    """Estimate per-frame camera extrinsics (camera-from-world, OpenCV convention).

    Assumptions:
    - Frame 0 defines the world coordinate system (identity extrinsic).
    - `tracks[t, j]` provides the (x, y) pixel of track j in frame t in the same
      pixel coordinate system as `depth_map` and `intrinsic`.
    - `depth_map[t]` gives metric depth along camera Z (OpenCV: z-forward).

    Args:
        depth_map: Depth maps array
        extrinsics: Initial extrinsic matrices
        intrinsic: Camera intrinsic matrix (3x3)
        tracks: Track positions array (T, P, 2)
        track_mask: Track visibility mask (T, P)
        ref_index: Reference frame index
        ransac_reproj_threshold: RANSAC reprojection error threshold

    Returns:
        Updated extrinsic matrices
    """
    depth_map = np.asarray(depth_map)
    tracks = np.asarray(tracks)
    track_mask = np.asarray(track_mask).astype(bool)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)

    if depth_map.ndim == 2:
        depth_map = depth_map[None]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"`tracks` must have shape (T, P, 2), got {tracks.shape}")
    if track_mask.shape[:2] != tracks.shape[:2]:
        raise ValueError(f"`track_mask` must have shape (T, P), got {track_mask.shape}")
    if extrinsics.shape[0] != tracks.shape[0]:
        raise ValueError(f"`extrinsics` must have shape (T, 4, 4), got {extrinsics.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"`intrinsic` must have shape (3, 3), got {intrinsic.shape}")

    num_frames = tracks.shape[0]
    height, width = depth_map.shape[-2:]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    def _sample_depth_nearest(depth_hw: np.ndarray, xy: np.ndarray) -> tuple:
        x = np.rint(xy[:, 0]).astype(np.int32)
        y = np.rint(xy[:, 1]).astype(np.int32)
        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        d = np.zeros((xy.shape[0],), dtype=np.float32)
        valid = in_bounds.copy()
        if valid.any():
            d[valid] = depth_hw[y[valid], x[valid]].astype(np.float32, copy=False)
            valid &= d > 0.0
        return d, valid

    def _unproject(xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
        x = xy[:, 0].astype(np.float64, copy=False)
        y = xy[:, 1].astype(np.float64, copy=False)
        z = depth.astype(np.float64, copy=False)
        X = (x - cx) / fx * z
        Y = (y - cy) / fy * z
        return np.stack([X, Y, z], axis=1)

    def _cam_to_world(points_cam: np.ndarray, extri: np.ndarray) -> np.ndarray:
        R = extri[:3, :3].astype(np.float64, copy=False)
        t = extri[:3, 3].astype(np.float64, copy=False)
        # X_world = R^T (X_cam - t)
        return (points_cam - t[None, :]) @ R

    extrinsic_known = np.zeros(num_frames, dtype=bool)
    extrinsic_known[ref_index] = True

    dist_coeffs = None  # assume no distortion
    frames_to_solve = [i for i in range(num_frames) if i != ref_index]
    for _ in range(num_frames - 1):
        progress = False
        remaining = []
        for frame_idx in frames_to_solve:

            estimated = False
            candidate_refs = [ref_index]
            for ref_idx in candidate_refs:
                vis = track_mask[ref_idx] & track_mask[frame_idx]
                if not np.any(vis):
                    continue

                xy_ref = tracks[ref_idx, vis]
                xy_cur = tracks[frame_idx, vis]
                depth_ref, valid_depth = _sample_depth_nearest(depth_map[ref_idx], xy_ref)

                if not np.any(valid_depth):
                    continue

                xy_ref = xy_ref[valid_depth]
                xy_cur = xy_cur[valid_depth]
                depth_ref = depth_ref[valid_depth]

                if xy_cur.shape[0] < 6:
                    continue

                points_ref_cam = _unproject(xy_ref, depth_ref)
                points_world = _cam_to_world(points_ref_cam, extrinsics[ref_idx])

                object_points = points_world.astype(np.float32, copy=False).reshape(-1, 1, 3)
                image_points = xy_cur.astype(np.float32, copy=False).reshape(-1, 1, 2)

                R_ref = extrinsics[ref_idx, :3, :3]
                t_ref = extrinsics[ref_idx, :3, 3]
                rvec_ref, _ = cv2.Rodrigues(R_ref.astype(np.float64))
                tvec_ref = t_ref.reshape(3, 1).astype(np.float64)

                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    intrinsic,
                    dist_coeffs,
                    rvec=rvec_ref,
                    tvec=tvec_ref,
                    useExtrinsicGuess=True,
                    iterationsCount=1000,
                    reprojectionError=ransac_reproj_threshold,
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_EPNP,
                )

                if not ok or inliers is None or len(inliers) < 6:
                    continue

                inlier_obj = object_points[inliers[:, 0]]
                inlier_img = image_points[inliers[:, 0]]
                ok_refine, rvec_refined, tvec_refined = cv2.solvePnP(
                    inlier_obj,
                    inlier_img,
                    intrinsic,
                    dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok_refine:
                    continue

                R, _ = cv2.Rodrigues(rvec_refined)
                extrinsics[frame_idx, :3, :3] = R.astype(np.float64)
                extrinsics[frame_idx, :3, 3] = tvec_refined.reshape(3).astype(np.float64)
                extrinsic_known[frame_idx] = True
                estimated = True
                progress = True
                break

            if not estimated:
                remaining.append(frame_idx)
        frames_to_solve = remaining
        if not progress:
            break

    # fill any missing with reference pose
    for frame_idx in frames_to_solve:
        extrinsics[frame_idx] = extrinsics[ref_index]

    return extrinsics


def verify_tracks_by_geometry(points3d, extrinsics, intrinsics, tracks, ref_index, masks=None, max_reproj_error=None):
    """Verify tracks by checking reprojection errors against geometric constraints.

    Args:
        points3d: 3D point coordinates
        extrinsics: Camera extrinsic matrices
        intrinsics: Camera intrinsic matrices
        tracks: Track positions
        ref_index: Reference frame index
        masks: Existing track masks
        max_reproj_error: Maximum allowed reprojection error

    Returns:
        Updated track masks
    """
    reproj_mask = None
    if max_reproj_error is not None:
        # project points into the reference view to filter out points behind the camera
        proj_ref, cam_ref = project_3D_points_np(points3d, extrinsics[ref_index][None], intrinsics[ref_index][None])
        valid_ref = cam_ref[:, 2, :] > 0

        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_points_2d[projected_points_cam[:, 2, :] <= 0] = 1e6

        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        reproj_mask = projected_diff < max_reproj_error
        # enforce visibility in reference frame
        reproj_mask = np.logical_and(reproj_mask, valid_ref)

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    return masks


def estimate_initial_poses(images, depth_prior, intrinsic, pred_tracks, pred_vis_scores, args, output_dir):
    """Estimate initial camera extrinsics and unproject depth to 3D point map.

    Args:
        images: Input image tensors
        depth_prior: Depth prior maps
        intrinsic: Camera intrinsic matrix
        pred_tracks: Predicted track positions
        pred_vis_scores: Predicted visibility scores
        args: Arguments with configuration
        output_dir: Output directory for visualizations

    Returns:
        Tuple of (extrinsic, intrinsic, points_3d, track_mask)
    """
    visualize_tracks_on_images(
        images[None],
        torch.from_numpy(pred_tracks[None]),
        torch.from_numpy(pred_vis_scores[None]) >= pred_vis_scores.min(),
        out_dir=f"{output_dir}/track_raw"
    )

    track_mask = pred_vis_scores > args.vis_thresh
    visualize_tracks_on_images(
        images[None],
        torch.from_numpy(pred_tracks[None]),
        torch.from_numpy(track_mask[None]),
        out_dir=f"{output_dir}/track_filter_vis_thresh"
    )

    extrinsic = np.eye(4)[None].repeat(len(images), axis=0).astype(np.float32)
    extrinsic = estimate_extrinsic(
        depth_prior, extrinsic, intrinsic, pred_tracks, track_mask,
        ref_index=args.cond_index,
        ransac_reproj_threshold=args.max_reproj_error
    )

    intrinsic = np.tile(intrinsic[None, :, :], (len(images), 1, 1))
    points_3d = unproject_depth_map_to_point_map(depth_prior[..., None], extrinsic, intrinsic)

    return extrinsic, intrinsic, points_3d, track_mask


def filter_and_verify_tracks(images, points_3d, extrinsic, intrinsic, pred_tracks, pred_vis_scores,
                             points_rgb, output_dir, cond_index, vis_thresh, max_reproj_error, min_inlier_per_frame, min_inlier_per_track):
    """Filter tracks by geometry verification and valid correspondences.

    Args:
        images: Input image tensors
        points_3d: 3D point coordinates
        extrinsic: Camera extrinsic matrices
        intrinsic: Camera intrinsic matrices
        pred_tracks: Predicted track positions
        pred_vis_scores: Predicted visibility scores
        points_rgb: Point RGB colors
        output_dir: Output directory for visualizations

    Returns:
        Tuple of (track_mask, points_3d, pred_tracks, points_rgb)
    """
    track_mask = pred_vis_scores > vis_thresh

    track_mask = verify_tracks_by_geometry(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        ref_index=cond_index,
        masks=track_mask,
        max_reproj_error=max_reproj_error,
    )
    visualize_tracks_on_images(
        images[None],
        torch.from_numpy(pred_tracks[None]),
        torch.from_numpy(track_mask[None]),
        out_dir=f"{output_dir}/track_filter_max_proj_err"
    )

    # Prep valid correspondences and drop 3D points without surviving tracks
    track_mask, points_3d, keep_pts = prep_valid_correspondences(
        points_3d, track_mask, min_inlier_per_frame, min_inlier_per_track
    )
    pred_tracks = pred_tracks[:, keep_pts]
    if points_rgb is not None:
        points_rgb = points_rgb[keep_pts]
    visualize_tracks_on_images(
        images[None],
        torch.from_numpy(pred_tracks[None]),
        torch.from_numpy(track_mask[None]),
        out_dir=f"{output_dir}/track_filter_frame_track_inlier"
    )

    return track_mask, points_3d, pred_tracks, points_rgb
