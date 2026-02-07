# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Frame registration and keyframe management functions for the COLMAP pipeline.
"""

import os

import numpy as np
import torch


def save_keyframe_indices(output_dir, frame_idx):
    """Append a keyframe index to the keyframe indices file.

    Args:
        output_dir: Output directory path
        frame_idx: Keyframe index to append
    """
    results_dir = os.path.join(output_dir)
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, "key_frame_idx.txt")
    with open(filepath, "a") as f:
        f.write(f"{frame_idx}\n")


def save_register_order(output_dir, frame_idx):
    """Append a keyframe index to the keyframe indices file.

    Args:
        output_dir: Output directory path
        frame_idx: Keyframe index to append
    """
    results_dir = os.path.join(output_dir)
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, "register_order.txt")
    with open(filepath, "a") as f:
        f.write(f"{frame_idx}\n")        


def load_register_indices(output_dir):
    """Load keyframe indices from the keyframe indices file.

    Args:
        output_dir: Output directory path containing key_frame_idx.txt

    Returns:
        List of keyframe indices (integers)
    """
    filepath = os.path.join(output_dir, "register_order.txt")
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def find_next_frame(image_info):
    """Select next unregistered frame that shares the most tracks with registered frames.

    Args:
        image_info: Dictionary containing track_mask, registered, and invalid frame flags

    Returns:
        Index of next frame to process, or None if all processed
    """
    track_mask = image_info.get("track_mask")
    registered = image_info.get("registered")
    invalid = image_info.get("invalid")
    if track_mask is None or registered is None or invalid is None:
        return None

    track_mask = np.asarray(track_mask)
    registered = np.asarray(registered)
    invalid = np.asarray(invalid)

    if registered.ndim != 1:
        registered = registered.reshape(-1)
    if invalid.ndim != 1:
        invalid = invalid.reshape(-1)

    num_frames = track_mask.shape[0]
    registered_mask = registered & (~invalid)
    if not np.any(registered_mask):
        return None

    # tracks visible in any registered frame
    vis_in_registered = track_mask[registered_mask].any(axis=0)

    best_idx = None
    best_count = -1
    for idx in range(num_frames):
        if registered[idx] or invalid[idx]:
            continue
        count = np.count_nonzero(track_mask[idx] & vis_in_registered)
        if count > best_count:
            best_count = count
            best_idx = idx
    return best_idx


def check_frame_invalid(image_info, frame_idx, min_inlier_per_frame=10, min_depth_pixels=500):
    """Check if frame has insufficient inliers or depth data.

    Args:
        image_info: Dictionary containing track_mask and depth_priors
        frame_idx: Frame index to check
        min_inlier_per_frame: Minimum required track inliers
        min_depth_pixels: Minimum required valid depth pixels

    Returns:
        True if frame is invalid, False otherwise
    """
    track_mask = image_info.get("track_mask")
    depth_priors = image_info.get("depth_priors")

    if track_mask is not None:
        track_inliers = int(np.count_nonzero(track_mask[frame_idx]))
        if track_inliers < min_inlier_per_frame:
            print(f"[check_frame_invalid] Frame {frame_idx} invalid: insufficient track inliers ({track_inliers} < {min_inlier_per_frame})")
            return True

    if depth_priors is not None:
        depth_map = depth_priors[frame_idx]
        if torch.is_tensor(depth_map):
            depth_map = depth_map.detach().cpu().numpy()
        valid_depth = int(np.count_nonzero(np.asarray(depth_map) > 0))
        if valid_depth < min_depth_pixels:
            print(f"[check_frame_invalid] Frame {frame_idx} invalid: insufficient depth pixels ({valid_depth} < {min_depth_pixels})")
            return True

    return False


def check_key_frame(image_info, frame_idx, rot_thresh, trans_thresh, depth_thresh, frame_inliner_thresh):
    """Heuristically decide if frame should become a keyframe based on validity + pose delta.

    Args:
        image_info: Dictionary containing extrinsics, keyframes, depth_priors, track_mask
        frame_idx: Frame index to check
        rot_thresh: Minimum rotation delta threshold (degrees)
        trans_thresh: Minimum translation delta threshold
        depth_thresh: Minimum depth pixel count
        frame_inliner_thresh: Minimum track inlier count

    Returns:
        True if frame should be a keyframe, False otherwise
    """
    registered = image_info.get("registered")
    invalid = image_info.get("invalid")
    extrinsics = image_info.get("extrinsics")
    keyframes = image_info.get("keyframe")
    depth_priors = image_info.get("depth_priors")
    track_mask = image_info.get("track_mask")

    invalid = np.asarray(invalid).astype(bool)
    if invalid[frame_idx]:
        print(f"[check_key_frame] Frame {frame_idx} is invalid; cannot be keyframe.")
        return False

    # Basic validity checks
    if depth_priors is not None:
        depth_map = depth_priors[frame_idx]
        if torch.is_tensor(depth_map):
            depth_map = depth_map.detach().cpu().numpy()
        valid_depth = int(np.count_nonzero(np.asarray(depth_map) > 0))
        if valid_depth < depth_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient depth pixels ({valid_depth} < {depth_thresh}).")
            return False

    if track_mask is not None:
        tm = np.asarray(track_mask)
        if frame_idx < tm.shape[0]:
            if int(np.count_nonzero(tm[frame_idx])) < frame_inliner_thresh:
                print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient track inliers.")
                return False

    if keyframes is None:
        return True  # no keyframes tracked yet

    keyframes = np.asarray(keyframes).astype(bool)
    past_keys = np.where(keyframes & registered & (np.arange(len(keyframes)) < frame_idx))[0]
    if len(past_keys) == 0:
        print(f"[check_key_frame] Frame {frame_idx} accepted as first keyframe.")
        return True

    T_curr = extrinsics[frame_idx]
    R_curr, t_curr = T_curr[:3, :3], T_curr[:3, 3]

    for kf_idx in past_keys:
        T_prev = extrinsics[kf_idx]
        R_prev, t_prev = T_prev[:3, :3], T_prev[:3, 3]
        R_delta = R_curr @ R_prev.T
        angle = np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)))
        trans = np.linalg.norm(t_curr - t_prev)

        if angle < rot_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient rotation delta ({angle:.2f} < {rot_thresh}).")
            return False

        if trans < trans_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient translation delta ({trans:.3f} < {trans_thresh}).")
            return False

    return True


def check_reprojection_error(image_info, frame_idx, args):
    """Check if frame has high reprojection error using low-uncertainty 3D points.

    Reprojects 3D points (filtered by uncertainty threshold) to the frame and
    computes the mean reprojection error against tracked 2D positions.

    Args:
        image_info: Dictionary containing points_3d, pred_tracks, track_mask,
                    extrinsics, intrinsics, and uncertainties
        frame_idx: Frame index to check
        args: Arguments with unc_thresh and max_reproj_error configuration

    Returns:
        True if frame is invalid (high reprojection error), False otherwise
    """
    points_3d = image_info.get("points_3d")
    pred_tracks = image_info.get("pred_tracks")
    track_mask = image_info.get("track_mask")
    extrinsics = image_info.get("extrinsics")
    intrinsics = image_info.get("intrinsics")
    uncertainties = image_info.get("uncertainties")

    if points_3d is None or pred_tracks is None or track_mask is None:
        print(f"[check_reprjection_error] Missing required data, skipping check")
        return False

    # Get frame-specific data
    frame_track_mask = np.asarray(track_mask[frame_idx]).astype(bool)
    valid_mask = frame_track_mask.copy()

    # Filter by uncertainty threshold
    unc_thresh = getattr(args, 'unc_thresh', 2.0)
    if uncertainties is not None and 'points3d' in uncertainties:
        pts_unc = uncertainties['points3d']
        if pts_unc is not None:
            pts_unc = np.asarray(pts_unc)
            unc_valid = np.isfinite(pts_unc) & (pts_unc <= unc_thresh)
            valid_mask = valid_mask & unc_valid

    num_valid = np.sum(valid_mask)
    if num_valid < 10:
        print(f"[check_reprjection_error] Frame {frame_idx}: insufficient valid points ({num_valid} < 10), skipping check")
        return False

    # Get 3D points and 2D tracks for valid points
    pts_3d = np.asarray(points_3d)[valid_mask]
    tracks_2d = np.asarray(pred_tracks[frame_idx])[valid_mask]

    # Get extrinsic and intrinsic for this frame
    extrinsic = extrinsics[frame_idx]
    intrinsic = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # Project 3D points to camera coordinates
    cam_pts = (R @ pts_3d.T).T + t

    # Filter points behind camera
    valid_z = cam_pts[:, 2] > 0
    if np.sum(valid_z) < 10:
        print(f"[check_reprjection_error] Frame {frame_idx}: insufficient points in front of camera, marking invalid")
        return True

    cam_pts = cam_pts[valid_z]
    tracks_2d = tracks_2d[valid_z]

    # Project to image plane
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    proj_x = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
    proj_y = fy * cam_pts[:, 1] / cam_pts[:, 2] + cy
    proj_2d = np.stack([proj_x, proj_y], axis=1)

    # Compute reprojection errors
    errors = np.linalg.norm(proj_2d - tracks_2d, axis=1)
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    max_reproj_error = getattr(args, 'max_reproj_error', 5.0)

    if mean_error > max_reproj_error:
        print(f"[check_reprjection_error] Frame {frame_idx} invalid: high reprojection error "
              f"(mean={mean_error:.2f}, median={median_error:.2f} > {max_reproj_error})")
        return True

    print(f"[check_reprjection_error] Frame {frame_idx} passed: reprojection error "
          f"(mean={mean_error:.2f}, median={median_error:.2f})")
    return False


def _predict_new_tracks(images, image_masks, frame_idx, args, dtype):
    """Step 1: Predict new tracks using the keyframe as query frame.

    Returns:
        Tuple of (new_pred_tracks, new_pred_vis_scores, new_points_rgb) or None if failed
    """
    from .track_prediction import predict_initial_tracks_wrapper

    print(f"[process_key_frame] Step 1: Predicting new tracks from keyframe {frame_idx}")
    try:
        new_pred_tracks, new_pred_vis_scores, new_points_rgb = predict_initial_tracks_wrapper(
            images, image_masks, args, dtype
        )
        # Convert to numpy
        new_pred_tracks = new_pred_tracks.cpu().numpy() if torch.is_tensor(new_pred_tracks) else new_pred_tracks
        new_pred_vis_scores = new_pred_vis_scores.cpu().numpy() if torch.is_tensor(new_pred_vis_scores) else new_pred_vis_scores
        if new_points_rgb is not None and torch.is_tensor(new_points_rgb):
            new_points_rgb = new_points_rgb.cpu().numpy()
        print(f"[process_key_frame] Predicted {new_pred_tracks.shape[1]} new tracks")
        return new_pred_tracks, new_pred_vis_scores, new_points_rgb
    except Exception as e:
        print(f"[process_key_frame] Track prediction failed: {e}")
        return None


def _sample_3d_points(gen_3d, new_pred_tracks, new_pred_vis_scores, new_points_rgb,
                      image_masks, frame_idx, image_shape):
    """Step 2: Sample 3D points at new track locations.

    Returns:
        Tuple of (new_pred_tracks, new_pred_vis_scores, new_points_3d, new_points_rgb)
    """
    from .track_prediction import sample_points_at_track_locations

    print(f"[process_key_frame] Step 2: Sampling 3D points at track locations")
    if gen_3d is not None and hasattr(gen_3d, 'points_3d'):
        points_3d_map = gen_3d.points_3d
        depth_conf = gen_3d.depth_conf if hasattr(gen_3d, 'depth_conf') else None
        new_pred_tracks, new_pred_vis_scores, _, new_points_3d, new_points_rgb = sample_points_at_track_locations(
            new_pred_tracks, new_pred_vis_scores, points_3d_map, depth_conf,
            image_masks, new_points_rgb, frame_idx, image_shape
        )
    else:
        new_points_3d = None
    return new_pred_tracks, new_pred_vis_scores, new_points_3d, new_points_rgb


def _estimate_poses(depth_priors, extrinsics, intrinsics, new_pred_tracks, new_pred_vis_scores,
                    new_points_3d, frame_idx, args):
    """Step 3: Estimate camera poses from the new tracks.

    Returns:
        Tuple of (new_extrinsics, new_track_mask, new_points_3d)
    """
    from .pose_estimation import estimate_extrinsic

    print(f"[process_key_frame] Step 3: Estimating poses from new tracks")
    new_track_mask = new_pred_vis_scores > args.vis_thresh

    # Use existing extrinsics as initialization
    new_extrinsics = extrinsics.copy()
    intrinsic_single = intrinsics[0] if intrinsics.ndim == 3 else intrinsics

    # Convert depth to numpy
    depth_prior_np = depth_priors.cpu().numpy() if torch.is_tensor(depth_priors) else depth_priors

    # Estimate extrinsics
    new_extrinsics = estimate_extrinsic(
        depth_prior_np, new_extrinsics, intrinsic_single, new_pred_tracks, new_track_mask,
        ref_index=frame_idx,
        ransac_reproj_threshold=args.max_reproj_error
    )

    # Unproject depth to get 3D points if not available
    if new_points_3d is None:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        new_points_3d_map = unproject_depth_map_to_point_map(
            depth_prior_np[..., None], new_extrinsics, intrinsics
        )
        # Sample 3D points at track locations
        query_points = new_pred_tracks[frame_idx]
        H, W = depth_prior_np.shape[-2:]
        scale = new_points_3d_map.shape[-2] / H if new_points_3d_map.shape[-2] != H else 1.0
        query_points_scaled = np.round(query_points * scale).astype(np.int64)
        query_points_scaled[:, 0] = np.clip(query_points_scaled[:, 0], 0, new_points_3d_map.shape[-2] - 1)
        query_points_scaled[:, 1] = np.clip(query_points_scaled[:, 1], 0, new_points_3d_map.shape[-3] - 1)
        new_points_3d = new_points_3d_map[frame_idx][query_points_scaled[:, 1], query_points_scaled[:, 0]]

    return new_extrinsics, new_track_mask, new_points_3d


def _filter_and_verify_tracks(new_points_3d, new_extrinsics, intrinsics, new_pred_tracks,
                               new_track_mask, new_points_rgb, frame_idx, args):
    """Step 4: Filter and verify the new tracks by geometry and inlier counts.

    Returns:
        Tuple of (new_track_mask, new_points_3d, new_pred_tracks, new_points_rgb)
    """
    from .pose_estimation import verify_tracks_by_geometry
    from .track_prediction import prep_valid_correspondences

    print(f"[process_key_frame] Step 4: Filtering and verifying new tracks")

    # Verify by reprojection error
    new_track_mask = verify_tracks_by_geometry(
        new_points_3d, new_extrinsics, intrinsics, new_pred_tracks,
        ref_index=frame_idx, masks=new_track_mask, max_reproj_error=args.max_reproj_error,
    )

    # Filter by per-frame and per-track inlier counts
    new_track_mask, new_points_3d, keep_pts = prep_valid_correspondences(
        new_points_3d, new_track_mask, args.min_inlier_per_frame, args.min_inlier_per_track
    )
    new_pred_tracks = new_pred_tracks[:, keep_pts]
    if new_points_rgb is not None:
        new_points_rgb = new_points_rgb[keep_pts]

    print(f"[process_key_frame] After filtering: {new_pred_tracks.shape[1]} tracks remain")
    return new_track_mask, new_points_3d, new_pred_tracks, new_points_rgb


def _merge_tracks(image_info, new_pred_tracks, new_track_mask, new_points_3d, new_points_rgb, frame_idx, args):
    """Step 5: Merge the new tracks with existing tracks.

    Updates image_info in-place with merged track data.
    """
    from .track_prediction import remove_duplicate_tracks

    print(f"[process_key_frame] Step 5: Merging new tracks with existing tracks")

    existing_tracks = image_info.get("pred_tracks")
    existing_track_mask = image_info.get("track_mask")
    existing_points_3d = image_info.get("points_3d")
    existing_points_rgb = image_info.get("points_rgb")

    if existing_tracks is not None and new_pred_tracks.shape[1] > 0:
        # Remove tracks with similar positions to existing tracks
        dist_thresh = getattr(args, 'duplicate_track_thresh', 3.0)
        new_pred_tracks, new_track_mask, new_points_3d, new_points_rgb = remove_duplicate_tracks(
            existing_tracks, new_pred_tracks, new_track_mask, new_points_3d, new_points_rgb,
            ref_frame_idx=frame_idx, dist_thresh=dist_thresh, existing_track_mask=existing_track_mask,
        )

        if new_pred_tracks.shape[1] > 0:
            # Concatenate existing and new data
            image_info["pred_tracks"] = np.concatenate([existing_tracks, new_pred_tracks], axis=1)
            image_info["track_mask"] = np.concatenate([existing_track_mask, new_track_mask], axis=1)
            image_info["points_3d"] = np.concatenate([existing_points_3d, new_points_3d], axis=0)
            if existing_points_rgb is not None and new_points_rgb is not None:
                image_info["points_rgb"] = np.concatenate([existing_points_rgb, new_points_rgb], axis=0)

            print(f"[process_key_frame] Merged: {existing_tracks.shape[1]} + {new_pred_tracks.shape[1]} = {image_info['pred_tracks'].shape[1]} tracks")

    elif new_pred_tracks.shape[1] > 0:
        # No existing tracks, use new tracks directly
        image_info["pred_tracks"] = new_pred_tracks
        image_info["track_mask"] = new_track_mask
        image_info["points_3d"] = new_points_3d
        image_info["points_rgb"] = new_points_rgb


def _propagate_uncertainties(image_info, args):
    """Step 6: Propagate uncertainties using all keyframes.

    Updates image_info["uncertainties"] in-place.
    """
    from .optimization import propagate_uncertainties

    print(f"[process_key_frame] Step 6: Propagating uncertainties")
    keyframe_indices = np.where(image_info["keyframe"])[0]
    print(f"[process_key_frame] Current keyframes: {keyframe_indices.tolist()}")

    uncertainties = propagate_uncertainties(
        image_info["points_3d"],
        image_info["extrinsics"],
        image_info["intrinsics"],
        image_info["pred_tracks"],
        image_info["depth_priors"],
        image_info["track_mask"],
        rot_thresh=args.kf_rot_thresh,
        trans_thresh=args.kf_trans_thresh,
        depth_thresh=args.kf_depth_thresh,
        track_inlier_thresh=args.kf_inlier_thresh,
        min_track_number=getattr(args, 'min_track_number', 3),
        keyframe_indices=keyframe_indices,
    )
    image_info["uncertainties"] = uncertainties


def _run_bundle_adjustment(image_info, args):
    """Step 7 (optional): Run bundle adjustment on all keyframes."""
    from .optimization import bundle_adjust_keyframes

    keyframe_indices = np.where(image_info["keyframe"])[0]
    if len(keyframe_indices) >= args.min_track_number and getattr(args, 'run_ba_on_keyframe', False):
        print(f"[process_key_frame] Step 7: Running bundle adjustment on {len(keyframe_indices)} keyframes")
        image_info = bundle_adjust_keyframes(
            image_info, ref_frame_idx=args.cond_index, iters=30, lr=1e-3,
            unc_thresh=getattr(args, 'unc_thresh', 2.0),
        )
    return image_info


def _refine_frame_pose_3d(image_info, frame_idx, args):
    """Step 8: Refine keyframe pose using 3D-3D correspondences with RANSAC.

    Computes rigid transformation between optimized 3D points and depth-derived 3D points
    using RANSAC to filter outliers. Only considers points with uncertainty below args.unc_thresh.

    Args:
        image_info: Dictionary containing reconstruction data with ba_valid_points_mask
        frame_idx: Keyframe index to refine
        args: Arguments with configuration (including unc_thresh)

    Returns:
        Updated image_info with refined keyframe extrinsic
    """
    # Get required data
    points_3d = image_info.get("points_3d")
    pred_tracks = image_info.get("pred_tracks")
    track_mask = image_info.get("track_mask")
    extrinsics = image_info.get("extrinsics")
    intrinsics = image_info.get("intrinsics")
    depth_priors = image_info.get("depth_priors")
    uncertainties = image_info.get("uncertainties")

    if points_3d is None or pred_tracks is None or track_mask is None or depth_priors is None:
        return False

    # Get intrinsic and depth for this frame
    intrinsic = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics
    depth_map = depth_priors[frame_idx]
    if torch.is_tensor(depth_map):
        depth_map = depth_map.cpu().numpy()

    # Filter points: must be visible in this frame
    frame_track_mask = np.asarray(track_mask[frame_idx]).astype(bool)
    valid_mask = frame_track_mask.copy()
    if image_info["registered"].sum() >= args.min_track_number:
        # Also filter by uncertainty threshold
        unc_thresh = getattr(args, 'unc_thresh', 2.0)
        if uncertainties is not None and 'points3d' in uncertainties:
            pts_unc = uncertainties['points3d']
            if pts_unc is not None:
                pts_unc = np.asarray(pts_unc)
                # Valid if uncertainty is finite and below threshold
                unc_valid = np.isfinite(pts_unc) & (pts_unc <= unc_thresh)
                num_excluded = np.sum(frame_track_mask & ~unc_valid)
                if num_excluded > 0:
                    print(f"[_refine_frame_pose_3d] Excluding {num_excluded} points with uncertainty > {unc_thresh}")
                valid_mask = valid_mask & unc_valid

    num_valid = np.sum(valid_mask)
    if num_valid < 30:
        print(f"[_refine_frame_pose_3d] Insufficient valid points ({num_valid} < 30), skipping")
        return False

    print(f"[_refine_frame_pose_3d] Refining frame {frame_idx} pose with {num_valid} valid 3D correspondences")

    # Get optimized 3D points (world coordinates)
    world_points = points_3d[valid_mask].astype(np.float64)

    # Get 2D track locations and sample depth
    track_2d = pred_tracks[frame_idx][valid_mask].astype(np.float64)
    H, W = depth_map.shape[:2]

    # Sample depth at track locations
    u = np.round(track_2d[:, 0]).astype(np.int32)
    v = np.round(track_2d[:, 1]).astype(np.int32)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    sampled_depth = depth_map[v, u]

    # Filter out invalid depth
    valid_depth = sampled_depth > 0
    if np.sum(valid_depth) < 10:
        print(f"[_refine_frame_pose_3d] Insufficient valid depth samples ({np.sum(valid_depth)} < 10), skipping")
        return False

    world_points = world_points[valid_depth]
    track_2d = track_2d[valid_depth]
    sampled_depth = sampled_depth[valid_depth]

    # Unproject to camera coordinates using depth
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_cam = (track_2d[:, 0] - cx) * sampled_depth / fx
    y_cam = (track_2d[:, 1] - cy) * sampled_depth / fy
    z_cam = sampled_depth
    cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)

    # RANSAC parameters
    ransac_iters = getattr(args, 'pose_ransac_iters', 1000)
    ransac_thresh = getattr(args, 'pose_ransac_thresh', 0.02)  # 2cm threshold in 3D
    min_inliers = max(6, int(len(world_points) * 0.3))

    best_R, best_t = None, None
    best_inlier_count = 0
    best_inliers = None

    N = len(world_points)
    for _ in range(ransac_iters):
        # Sample 3 points for minimal solution
        indices = np.random.choice(N, 3, replace=False)
        src = world_points[indices]  # world points
        dst = cam_points[indices]    # camera points

        # Compute rigid transformation: cam = R @ world + t
        R, t = _compute_rigid_transform(src, dst)
        if R is None:
            continue

        # Apply transformation to all points
        transformed = (R @ world_points.T).T + t

        # Count inliers
        errors = np.linalg.norm(transformed - cam_points, axis=1)
        inliers = errors < ransac_thresh
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_R, best_t = R, t

    if best_inlier_count < min_inliers:
        print(f"[_refine_frame_pose_3d] RANSAC failed: only {best_inlier_count} inliers (need {min_inliers}), keeping original pose")
        return False

    # Refine with all inliers
    src_inliers = world_points[best_inliers]
    dst_inliers = cam_points[best_inliers]
    R_refined, t_refined = _compute_rigid_transform(src_inliers, dst_inliers)

    if R_refined is None:
        R_refined, t_refined = best_R, best_t

    # Build extrinsic matrix (camera from world)
    # The transformation is: P_cam = R @ P_world + t
    # So extrinsic is [R | t]
    R_init = extrinsics[frame_idx, :3, :3].copy()
    t_init = extrinsics[frame_idx, :3, 3].copy()

    extrinsics[frame_idx, :3, :3] = R_refined.astype(np.float32)
    extrinsics[frame_idx, :3, 3] = t_refined.astype(np.float32)

    # Compute and report improvement
    R_delta = R_refined @ R_init.T
    angle_change = np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)))
    trans_change = np.linalg.norm(t_refined - t_init)

    # Compute final alignment error
    final_transformed = (R_refined @ world_points.T).T + t_refined
    final_errors = np.linalg.norm(final_transformed - cam_points, axis=1)
    mean_error = np.mean(final_errors[best_inliers])

    print(f"[_refine_frame_pose_3d] Refined pose: {best_inlier_count}/{N} inliers, "
          f"mean 3D error: {mean_error:.4f}m, rotation change: {angle_change:.3f}°, "
          f"translation change: {trans_change:.4f}m")
    
    if best_inlier_count < 50:
        return False

    return True


def _compute_rigid_transform(src, dst):
    """Compute rigid transformation (R, t) such that dst = R @ src + t.

    Uses SVD-based Procrustes analysis.

    Args:
        src: Source points [N, 3]
        dst: Destination points [N, 3]

    Returns:
        Tuple of (R, t) or (None, None) if computation fails
    """
    if len(src) < 3:
        return None, None

    # Center the points
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    H = src_centered.T @ dst_centered

    # SVD
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None, None

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = dst_mean - R @ src_mean

    return R, t


def process_key_frame(image_info, frame_idx, args):
    """Process frame designated as keyframe with multi-step workflow.

    Pipeline steps:
        1. Predict new tracks using the keyframe as query frame
        2. Sample 3D points at track locations
        3. Estimate camera poses from the new tracks
        4. Filter and verify the new tracks
        5. Merge the new tracks with existing tracks
        6. Propagate uncertainties using all keyframes
        7. (Optional) Run bundle adjustment on keyframes
        8. Refine keyframe pose using 3D-3D correspondences with valid (low-uncertainty) points

    Args:
        image_info: Dictionary containing reconstruction data
        frame_idx: Frame index to process as keyframe
        args: Arguments with configuration

    Returns:
        Updated image_info with new tracks merged and uncertainties updated
    """
    print(f"[process_key_frame] Processing keyframe at frame {frame_idx}")
    image_info["keyframe"][frame_idx] = True

    # Validate required data
    images = image_info.get("images")
    depth_priors = image_info.get("depth_priors")
    if images is None or depth_priors is None:
        print(f"[process_key_frame] Missing images or depth_priors, skipping")
        return image_info

    # Setup
    dtype = getattr(args, 'dtype', torch.float16)
    original_cond_index = args.cond_index
    args.cond_index = frame_idx

    try:
        # Step 1: Predict new tracks
        result = _predict_new_tracks(images, image_info.get("image_masks"), frame_idx, args, dtype)
        if result is None:
            return image_info
        new_pred_tracks, new_pred_vis_scores, new_points_rgb = result


        # Step 3: Estimate poses from new tracks
        new_extrinsics, new_track_mask, new_points_3d = _estimate_poses(
            depth_priors, image_info["extrinsics"], image_info["intrinsics"],
            new_pred_tracks, new_pred_vis_scores, None, frame_idx, args
        )

        # Step 4: Filter and verify tracks
        new_track_mask, new_points_3d, new_pred_tracks, new_points_rgb = _filter_and_verify_tracks(
            new_points_3d, new_extrinsics, image_info["intrinsics"],
            new_pred_tracks, new_track_mask, new_points_rgb, frame_idx, args
        )

        # Step 5: Merge new tracks with existing tracks
        _merge_tracks(image_info, new_pred_tracks, new_track_mask, new_points_3d, new_points_rgb, frame_idx, args)

        # Step 6: Propagate uncertainties
        _propagate_uncertainties(image_info, args)

        # Step 7: Optional bundle adjustment
        image_info = _run_bundle_adjustment(image_info, args)


    finally:
        # Restore original query index
        args.cond_index = original_cond_index

    print(f"[process_key_frame] Keyframe {frame_idx} processing complete")
    return image_info

def register_condition_frame_as_keyframe(image_info, args):
    num_images = len(image_info["images"])
    image_info["registered"] = np.array([False] * num_images)
    image_info["registered"][args.cond_index] = True

    image_info["invalid"] = np.array([False] * num_images)

    image_info["keyframe"] = np.array([False] * num_images)
    image_info["keyframe"][args.cond_index] = True

    return image_info    

def register_key_frames(image_info, args):
    """Register all remaining frames in the sequence.

    Args:
        image_info: Dictionary containing reconstruction data
        args: Arguments with configuration

    Returns:
        Updated image_info with all frames registered
    """
    # Import here to avoid circular dependency
    from .visualization_io import save_results
    from .optimization import register_new_frame_by_PnP
    
    image_info = register_condition_frame_as_keyframe(image_info, args)
    num_images = len(image_info["images"])
    while image_info["registered"].sum() + image_info["invalid"].sum() < num_images:
        next_frame_idx = find_next_frame(image_info)
        print("+" * 50)
        print(f"Next frame to register: {next_frame_idx}")

        if check_frame_invalid(
            image_info, next_frame_idx,
            min_inlier_per_frame=args.min_inlier_per_frame,
            min_depth_pixels=args.min_depth_pixels
        ):
            image_info["invalid"][next_frame_idx] = True
            continue

        # Register the frame
        register_new_frame_by_PnP(
            image_info, next_frame_idx, args,
        )

        # Refine the frame pose using 3D-3D correspondences
        if image_info["keyframe"].sum() > args.min_track_number:   
            if _refine_frame_pose_3d(image_info, next_frame_idx, args):
                    if (check_reprojection_error(image_info, next_frame_idx, args)):
                        # high reprojection error, mark as invalid
                        image_info["invalid"][next_frame_idx] = True
                    else:
                        image_info["registered"][next_frame_idx] = True
            else:
                # not enough valid 3D points and depth to refine, mark as invalid
                image_info["invalid"][next_frame_idx] = True
        else:
            image_info["registered"][next_frame_idx] = True
        

        # Check if this frame should be a keyframe
        if not image_info["invalid"][next_frame_idx]:
            if check_key_frame(
                image_info, next_frame_idx,
                rot_thresh=args.kf_rot_thresh,
                trans_thresh=args.kf_trans_thresh,
                depth_thresh=args.kf_depth_thresh,
                frame_inliner_thresh=args.kf_inlier_thresh
            ):
                image_info = process_key_frame(image_info, next_frame_idx, args)
                save_keyframe_indices(args.output_dir, next_frame_idx)
        save_results(image_info, gen_3d=None, out_dir=f"{args.output_dir}/results/{next_frame_idx:04d}/", args=args)

        print(f"registered: {image_info['registered'].sum()}, "
              f"keyframes: {image_info['keyframe'].sum()}, invalid: {image_info['invalid'].sum()}")
    return image_info
