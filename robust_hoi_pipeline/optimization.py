# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Optimization functions for the COLMAP pipeline.

Includes bundle adjustment, uncertainty propagation, and pose optimization.
"""

import numpy as np
import torch
import cv2

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

from .geometry_utils import axis_angle_to_matrix


# Module-level constants for loss weights (configurable)
DEFAULT_RAY_WEIGHT = 10
DEFAULT_DEPTH_WEIGHT = 1000
MIN_VALID_OBSERVATIONS_THRESHOLD = 1e-6


def propagate_uncertainties(
    points_3d,
    extrinsic,
    intrinsic,
    pred_tracks,
    depth_prior,
    track_mask,
    rot_thresh=5.0,
    trans_thresh=0.05,
    depth_thresh=1000,
    track_inlier_thresh=50,
    min_track_number=3,
    keyframe_indices=None,
):
    """Propagate uncertainties for extrinsics, 3D points, and depth priors.

    Only cameras that pass the threshold criteria are used for uncertainty computation:
    - depth_thresh: minimum number of valid depth pixels
    - track_inlier_thresh: minimum number of track inliers
    - rot_thresh: minimum rotation delta (degrees) from reference camera
    - trans_thresh: minimum translation delta from reference camera
    - min_track_number: minimum keyframe observations for a 3D point; points with
      fewer observations get high uncertainty

    If keyframe_indices is provided, those frames are used directly as keyframes.
    Otherwise, the function first filters valid frames (enough track inliers + valid depth),
    then selects keyframes from valid frames based on rotation/translation thresholds.
    Only keyframes are used for depth uncertainty computation.

    Args:
        points_3d: 3D point coordinates
        extrinsic: Camera extrinsic matrices
        intrinsic: Camera intrinsic matrices
        pred_tracks: Predicted track positions
        depth_prior: Depth prior maps
        track_mask: Track visibility mask
        rot_thresh: Rotation threshold in degrees
        trans_thresh: Translation threshold
        depth_thresh: Minimum valid depth pixel count
        track_inlier_thresh: Minimum track inlier count
        min_track_number: Minimum keyframe observations per point
        keyframe_indices: Optional list/array of keyframe indices. If None, keyframes
                          are computed from the data using threshold criteria.

    Returns:
        Dictionary containing uncertainties for extrinsics, points3d, depth_prior, and keyframes list
    """
    device = depth_prior.device if torch.is_tensor(depth_prior) else "cpu"
    dtype = torch.float32

    P = points_3d.shape[0]
    B = extrinsic.shape[0]

    points3d_t = torch.from_numpy(points_3d).to(device=device, dtype=dtype)
    tracks_t = torch.from_numpy(pred_tracks).to(device=device, dtype=dtype)
    mask_t = torch.from_numpy(track_mask).to(device=device)

    intr_t = torch.from_numpy(intrinsic).to(device=device, dtype=dtype)
    extr_t = torch.from_numpy(extrinsic[:, :3, :]).to(device=device, dtype=dtype)

    # Use provided keyframe_indices or compute from data
    if keyframe_indices is not None:
        # Use provided keyframe indices directly
        keyframes = list(keyframe_indices) if not isinstance(keyframe_indices, list) else keyframe_indices
        print(f"[propagate_uncertainties] Using provided keyframes: {len(keyframes)} {keyframes}")
    else:
        # Step 1: Identify valid frames (enough track inliers + valid depth)
        track_inliers = mask_t.sum(dim=-1).cpu().numpy()  # (B,)
        valid_depth_counts = np.zeros(B, dtype=np.int64)
        if torch.is_tensor(depth_prior):
            for i in range(B):
                valid_depth_counts[i] = int((depth_prior[i] > 0).sum().item())

        valid_frames = []
        for i in range(B):
            has_enough_inliers = track_inliers[i] >= track_inlier_thresh
            has_enough_depth = valid_depth_counts[i] >= depth_thresh
            if has_enough_inliers and has_enough_depth:
                valid_frames.append(i)

        # Step 2: From valid frames, select keyframes based on rotation/translation thresholds
        keyframes = []
        for frame_idx in valid_frames:
            if len(keyframes) == 0:
                # First valid frame is always a keyframe
                keyframes.append(frame_idx)
                continue

            # Check rotation and translation delta with all existing keyframes
            T_curr = extrinsic[frame_idx]
            R_curr, t_curr = T_curr[:3, :3], T_curr[:3, 3]

            is_keyframe = True
            for kf_idx in keyframes:
                T_kf = extrinsic[kf_idx]
                R_kf, t_kf = T_kf[:3, :3], T_kf[:3, 3]

                # Compute rotation delta (degrees)
                R_delta = R_curr @ R_kf.T
                angle = np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)))

                # Compute translation delta
                trans = np.linalg.norm(t_curr - t_kf)

                # Reject if too close to any existing keyframe
                if angle < rot_thresh and trans < trans_thresh:
                    is_keyframe = False
                    break

            if is_keyframe:
                keyframes.append(frame_idx)

        print(f"[propagate_uncertainties] Valid frames: {len(valid_frames)}/{B}, Keyframes: {len(keyframes)} {keyframes}")

    # Create keyframe mask: (B, P) mask that is only True for keyframe indices
    kf_frame_mask = torch.zeros(B, device=device, dtype=torch.bool)
    kf_frame_mask[keyframes] = True
    kf_mask = mask_t & kf_frame_mask.unsqueeze(-1)  # (B, P)

    with torch.no_grad():
        extr_final = extr_t

        ones = torch.ones((B, P, 1), device=device, dtype=dtype)
        pts_h = torch.cat([points3d_t.unsqueeze(0).expand(B, -1, -1), ones], dim=-1)
        cam_pts = torch.bmm(extr_final, pts_h.transpose(1, 2))

        z = cam_pts[:, 2:3, :]
        uv = cam_pts[:, :2, :] / (z + 1e-6)
        ones2 = torch.ones((B, 1, P), device=device, dtype=dtype)
        uv_h = torch.cat([uv, ones2], dim=1)
        proj = torch.bmm(intr_t, uv_h)[:, :2, :].transpose(1, 2)

        rep_err = (proj - tracks_t) * mask_t.unsqueeze(-1)
        rep_l2 = torch.linalg.norm(rep_err, dim=-1)  # (B, P)
        rep_unc_frame = torch.sqrt((rep_l2.pow(2) * mask_t).sum(-1) / mask_t.sum(-1).clamp(min=1)).cpu().numpy()

        # Use keyframe mask and rep_l2 for pts_unc calculation
        rep_l2_kf = rep_l2 * kf_mask
        kf_track_count = kf_mask.sum(0)  # (P,) - number of keyframe observations per point
        pts_unc = torch.sqrt((rep_l2_kf.pow(2)).sum(0) / kf_track_count.clamp(min=1)).cpu().numpy()

        # Set high uncertainty for points with insufficient keyframe tracks
        insufficient_tracks = kf_track_count.cpu().numpy() < min_track_number
        pts_unc[insufficient_tracks] = np.inf

        depth_unc = None
        if torch.is_tensor(depth_prior):
            H, W = depth_prior.shape[-2:]
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            depth_unc = torch.zeros_like(depth_prior, dtype=dtype, device=device)
            depth_cnt = torch.zeros_like(depth_prior, dtype=dtype, device=device)

            # Only iterate over keyframes for depth uncertainty computation
            for ref in keyframes:
                fx_r, fy_r = intr_t[ref, 0, 0], intr_t[ref, 1, 1]
                cx_r, cy_r = intr_t[ref, 0, 2], intr_t[ref, 1, 2]
                Rr = extr_final[ref, :3, :3]
                tr = extr_final[ref, :3, 3]
                unc_flat = depth_unc[ref].view(-1)
                cnt_flat = depth_cnt[ref].view(-1)

                # Only use other keyframes as source frames
                for src in keyframes:
                    if src == ref:
                        continue
                    if rep_unc_frame[src] > 5.0 or rep_unc_frame[src] == 0.:  # 0 means no extrinsic estimated
                        continue
                    depth_src = depth_prior[src].to(device=device, dtype=dtype)
                    valid_src = depth_src > 0
                    if not valid_src.any():
                        continue
                    fx_s, fy_s = intr_t[src, 0, 0], intr_t[src, 1, 1]
                    cx_s, cy_s = intr_t[src, 0, 2], intr_t[src, 1, 2]
                    Rs = extr_final[src, :3, :3]
                    ts = extr_final[src, :3, 3]
                    X = (xs - cx_s) / fx_s * depth_src
                    Y = (ys - cy_s) / fy_s * depth_src
                    Z = depth_src
                    pts_cam_s = torch.stack([X, Y, Z], dim=-1)
                    pts_cam_s = pts_cam_s[valid_src]  # M,3
                    if pts_cam_s.numel() == 0:
                        continue

                    pts_world = torch.matmul(pts_cam_s - ts, Rs.transpose(0, 1))
                    # transform pts_world to ref camera
                    pts_cam_r = torch.matmul(pts_world, Rr.transpose(0, 1)) + tr

                    zr = pts_cam_r[:, 2]
                    positive = zr > 0
                    if not positive.any():
                        continue
                    pts_cam_r = pts_cam_r[positive]
                    zr = zr[positive]
                    u = (pts_cam_r[:, 0] / zr) * fx_r + cx_r
                    v = (pts_cam_r[:, 1] / zr) * fy_r + cy_r
                    in_bounds = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
                    if not in_bounds.any():
                        continue
                    u = u[in_bounds]
                    v = v[in_bounds]
                    zr = zr[in_bounds]
                    u_round = u.round().long()
                    v_round = v.round().long()
                    flat_idx = (v_round * W + u_round)
                    grid = torch.empty((1, 1, u.shape[0], 2), device=device, dtype=dtype)
                    grid[..., 0] = (u / (W - 1)) * 2 - 1
                    grid[..., 1] = (v / (H - 1)) * 2 - 1
                    depth_ref_sampled = torch.nn.functional.grid_sample(
                        depth_prior[ref].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0),
                        grid,
                        align_corners=False,
                        mode="bilinear",
                    ).view(-1)
                    valid_ref_depth = depth_ref_sampled > 0
                    if not valid_ref_depth.any():
                        continue
                    depth_ref_sampled = depth_ref_sampled[valid_ref_depth]
                    zr = zr[valid_ref_depth]
                    flat_idx = flat_idx[valid_ref_depth]
                    resid = (depth_ref_sampled - zr)
                    unc_flat.scatter_add_(0, flat_idx, resid.pow(2))
                    cnt_flat.scatter_add_(0, flat_idx, torch.ones_like(resid))

            depth_unc = (depth_unc / depth_cnt.clamp(min=1)).sqrt().cpu().numpy()

    uncertainties = {
        "extrinsic": rep_unc_frame,
        "points3d": pts_unc,
        "depth_prior": depth_unc,
        "keyframes": keyframes,
    }

    return uncertainties


def _extract_keyframe_data(image_info, keyframe_indices, ref_frame_idx):
    """Extract keyframe-specific data from image_info.

    Filters out 3D points with infinite uncertainty.

    Returns:
        Dictionary with keyframe data, reference index, and valid point indices
    """
    points_3d = image_info["points_3d"]
    pred_tracks = image_info["pred_tracks"]
    track_mask = image_info["track_mask"]

    # Get point uncertainties and filter out inf values
    uncertainties = image_info.get("uncertainties", {})
    pts_unc = uncertainties.get("points3d", None)

    if pts_unc is not None:
        # Find valid points (finite uncertainty)
        valid_pts_mask = np.isfinite(pts_unc)
        valid_pts_indices = np.where(valid_pts_mask)[0]

        if len(valid_pts_indices) < len(pts_unc):
            num_filtered = len(pts_unc) - len(valid_pts_indices)
            print(f"[_extract_keyframe_data] Filtered {num_filtered} points with inf uncertainty")

        # Filter points and tracks
        points_3d = points_3d[valid_pts_indices]
        pred_tracks = pred_tracks[:, valid_pts_indices]
        track_mask = track_mask[:, valid_pts_indices]
        pts_unc = pts_unc[valid_pts_indices]
    else:
        valid_pts_indices = np.arange(len(points_3d))
        pts_unc = None

    kf_data = {
        "tracks": pred_tracks[keyframe_indices],        # [K, N, 2]
        "mask": track_mask[keyframe_indices],           # [K, N]
        "extrinsics": image_info["extrinsics"][keyframe_indices],   # [K, 4, 4]
        "intrinsics": image_info["intrinsics"][keyframe_indices],   # [K, 3, 3]
        "points_3d": points_3d,                         # [P_valid, 3]
        "points_unc": pts_unc,                          # [P_valid] or None
        "valid_pts_indices": valid_pts_indices,         # indices into original points
    }

    # Extract depth priors if available
    depth_priors = image_info.get("depth_priors")
    if depth_priors is not None:
        kf_data["depth_priors"] = depth_priors[keyframe_indices]
    else:
        kf_data["depth_priors"] = None

    # Find reference index within keyframe subset
    kf_ref_idx = np.where(keyframe_indices == ref_frame_idx)[0]
    kf_data["ref_idx"] = kf_ref_idx[0] if len(kf_ref_idx) > 0 else 0

    return kf_data


def _init_optimization_tensors(kf_data, device, dtype, unc_thresh = 2.0):
    """Initialize tensors for optimization.

    Returns:
        Dictionary with initialized tensors including uncertainty weights
    """
    K = len(kf_data["extrinsics"])

    # Convert to tensors
    points3d_t = torch.from_numpy(kf_data["points_3d"]).to(device=device, dtype=dtype)
    points3d_t.requires_grad_(True)

    tracks_t = torch.from_numpy(kf_data["tracks"]).to(device=device, dtype=dtype)
    mask_t = torch.from_numpy(kf_data["mask"]).to(device=device)
    intr_t = torch.from_numpy(kf_data["intrinsics"]).to(device=device, dtype=dtype)

    # Convert extrinsics to rotation vectors (Rodrigues) and translation vectors
    rvecs, tvecs = [], []
    for i in range(K):
        rvec, _ = cv2.Rodrigues(kf_data["extrinsics"][i, :3, :3])
        rvecs.append(rvec.reshape(-1))
        tvecs.append(kf_data["extrinsics"][i, :3, 3])

    rvecs_t = torch.from_numpy(np.stack(rvecs)).to(device=device, dtype=dtype)
    tvecs_t = torch.from_numpy(np.stack(tvecs)).to(device=device, dtype=dtype)
    rvecs_t.requires_grad_(True)
    tvecs_t.requires_grad_(True)

    # Prepare depth priors
    kf_depth = kf_data["depth_priors"]
    if kf_depth is not None:
        if not torch.is_tensor(kf_depth):
            kf_depth = torch.from_numpy(kf_depth)
        kf_depth = kf_depth.to(device=device, dtype=dtype)

    # Prepare uncertainty weights (inverse of uncertainty)
    # Lower uncertainty = higher weight
    # Points with uncertainty > threshold get zero weight (excluded from optimization)
    pts_unc = kf_data.get("points_unc")
    if pts_unc is not None:
        unc_t = torch.from_numpy(pts_unc).to(device=device, dtype=dtype)

        # Create mask for valid (low uncertainty) points
        valid_unc_mask = unc_t <= unc_thresh
        num_excluded = (~valid_unc_mask).sum().item()

        if num_excluded > 0:
            print(f"[_init_optimization_tensors] Excluding {num_excluded} points with uncertainty > {unc_thresh}")

        # Compute weights only for valid points
        unc_t_clamped = unc_t.clamp(min=1e-6)  # avoid division by zero
        weights_t = 1.0 / unc_t_clamped
        weights_t[~valid_unc_mask] = 0.0  # zero weight for high uncertainty points

        # Normalize by mean of non-zero weights
        valid_weights = weights_t[valid_unc_mask]
        if valid_weights.numel() > 0:
            weights_t = weights_t / valid_weights.mean()  # normalize valid weights to mean=1

        print(f"[_init_optimization_tensors] Using uncertainty weights: min={weights_t.min().item():.3f}, "
              f"max={weights_t.max().item():.3f}, mean={weights_t[valid_unc_mask].mean().item():.3f} "
              f"(valid: {valid_unc_mask.sum().item()}/{len(unc_t)})")
    else:
        weights_t = None
        valid_unc_mask = None

    return {
        "points3d": points3d_t,
        "tracks": tracks_t,
        "mask": mask_t,
        "intrinsics": intr_t,
        "inv_intrinsics": torch.inverse(intr_t),
        "rvecs": rvecs_t,
        "tvecs": tvecs_t,
        "depth_priors": kf_depth,
        "point_weights": weights_t,  # [P] or None
        "valid_unc_mask": valid_unc_mask,  # [P] bool tensor or None
    }


def _project_points(points3d_t, rvecs_t, tvecs_t, intr_t, device, dtype):
    """Project 3D points to 2D using current pose estimates.

    Returns:
        Tuple of (projected_2d, camera_points_3d)
    """
    K = rvecs_t.shape[0]
    P = points3d_t.shape[0]

    R = axis_angle_to_matrix(rvecs_t)
    extr_mat = torch.cat([R, tvecs_t.unsqueeze(-1)], dim=-1)  # K,3,4

    ones = torch.ones((K, P, 1), device=device, dtype=dtype)
    pts_h = torch.cat([points3d_t.unsqueeze(0).expand(K, -1, -1), ones], dim=-1)  # K,P,4
    cam_pts = torch.bmm(extr_mat, pts_h.transpose(1, 2))  # K,3,P

    z = cam_pts[:, 2:3, :]
    uv = cam_pts[:, :2, :] / (z + 1e-6)
    ones2 = torch.ones((K, 1, P), device=device, dtype=dtype)
    uv_h = torch.cat([uv, ones2], dim=1)
    proj = torch.bmm(intr_t, uv_h)[:, :2, :].transpose(1, 2)  # K,P,2

    return proj, cam_pts


def _compute_reprojection_loss(proj, tracks_t, mask_t, point_weights=None):
    """Compute reprojection loss between projected and observed 2D points.

    Args:
        proj: Projected 2D points [K, P, 2]
        tracks_t: Observed 2D tracks [K, P, 2]
        mask_t: Visibility mask [K, P]
        point_weights: Per-point uncertainty weights [P] (higher = more confident)

    Returns:
        Weighted reprojection loss
    """
    rep_raw = proj - tracks_t  # K, P, 2
    valid = mask_t.unsqueeze(-1).to(rep_raw.dtype)  # K, P, 1

    if point_weights is not None:
        # Apply per-point weights: [P] -> [1, P, 1]
        weights = point_weights.unsqueeze(0).unsqueeze(-1)  # 1, P, 1
        weighted_valid = valid * weights
    else:
        weighted_valid = valid

    rep_loss = torch.nn.functional.smooth_l1_loss(
        rep_raw * weighted_valid, torch.zeros_like(rep_raw), reduction="sum"
    ) / (weighted_valid.sum() * rep_raw.shape[-1] + 1e-8)

    return rep_loss


def _compute_ray_loss(cam_pts, tracks_t, mask_t, inv_intr_t, device, dtype, point_weights=None):
    """Compute point-to-ray consistency loss.

    Args:
        cam_pts: Points in camera coordinates [K, 3, P]
        tracks_t: Observed 2D tracks [K, P, 2]
        mask_t: Visibility mask [K, P]
        inv_intr_t: Inverse intrinsic matrices [K, 3, 3]
        device: Compute device
        dtype: Data type
        point_weights: Per-point uncertainty weights [P]

    Returns:
        Weighted ray consistency loss
    """
    K, _, P = cam_pts.shape

    uv1 = torch.cat([tracks_t, torch.ones((K, P, 1), device=device, dtype=dtype)], dim=-1)
    rays = torch.bmm(inv_intr_t, uv1.transpose(1, 2))  # K,3,P
    rays = torch.nn.functional.normalize(rays, dim=1, eps=1e-6)

    cross = torch.cross(cam_pts, rays, dim=1)
    ray_raw = torch.linalg.norm(cross, dim=1)  # K,P
    valid = mask_t.to(ray_raw.dtype)  # K,P

    if point_weights is not None:
        # Apply per-point weights: [P] -> [1, P]
        weights = point_weights.unsqueeze(0)  # 1, P
        weighted_valid = valid * weights
    else:
        weighted_valid = valid

    ray_loss = torch.nn.functional.smooth_l1_loss(
        ray_raw * weighted_valid, torch.zeros_like(ray_raw), reduction="sum"
    ) / (weighted_valid.sum() + 1e-8)

    return ray_loss


def _compute_depth_loss(cam_pts, tracks_t, mask_t, kf_depth_priors, point_weights=None):
    """Compute depth consistency loss against depth priors.

    Args:
        cam_pts: Points in camera coordinates [K, 3, P]
        tracks_t: Observed 2D tracks [K, P, 2]
        mask_t: Visibility mask [K, P]
        kf_depth_priors: Depth prior maps [K, H, W]
        point_weights: Per-point uncertainty weights [P]

    Returns:
        Weighted depth consistency loss
    """
    if kf_depth_priors is None:
        return torch.zeros((), device=cam_pts.device, dtype=cam_pts.dtype)

    z_pred = cam_pts[:, 2, :]  # K, P
    H, W = kf_depth_priors.shape[-2:]

    # Sample depth priors at track locations
    grid_x = 2.0 * tracks_t[..., 0] / (W - 1) - 1.0
    grid_y = 2.0 * tracks_t[..., 1] / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

    depth_prior_sampled = torch.nn.functional.grid_sample(
        kf_depth_priors.unsqueeze(1), grid,
        mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze(1).squeeze(-1)

    valid_depth_mask = (depth_prior_sampled > 0) & mask_t
    if not valid_depth_mask.any():
        return torch.zeros((), device=cam_pts.device, dtype=cam_pts.dtype)

    depth_raw = z_pred - depth_prior_sampled
    valid = valid_depth_mask.to(depth_raw.dtype)

    if point_weights is not None:
        weights = point_weights.unsqueeze(0)  # 1, P
        weighted_valid = valid * weights
    else:
        weighted_valid = valid

    depth_loss = torch.nn.functional.smooth_l1_loss(
        depth_raw * weighted_valid, torch.zeros_like(depth_raw), reduction="sum"
    ) / (weighted_valid.sum() + 1e-8)

    return depth_loss


def _run_ba_optimization(tensors, kf_ref_idx, iters, lr, rep_loss_thresh, depth_loss_thresh, device, dtype):
    """Run the bundle adjustment optimization loop.

    Uses point uncertainty weights to give higher importance to more confident points.

    Args:
        tensors: Dictionary containing optimization tensors
        kf_ref_idx: Reference keyframe index (pose is fixed)
        iters: Number of optimization iterations
        lr: Learning rate
        rep_loss_thresh: Early-stop threshold for reprojection loss
        depth_loss_thresh: Early-stop threshold for depth loss
        device: Compute device
        dtype: Data type

    Returns:
        Tuple of (optimized_rvecs, optimized_tvecs, optimized_points3d, final_proj, final_cam_pts)
    """
    optim = torch.optim.Adam([tensors["rvecs"], tensors["tvecs"], tensors["points3d"]], lr=lr)

    # Loss weights
    RAY_WEIGHT = 10
    DEPTH_WEIGHT = 1000

    # Get point uncertainty weights (may be None)
    point_weights = tensors.get("point_weights")

    for it in range(iters):
        optim.zero_grad(set_to_none=True)

        # Forward pass: project points
        proj, cam_pts = _project_points(
            tensors["points3d"], tensors["rvecs"], tensors["tvecs"],
            tensors["intrinsics"], device, dtype
        )

        # Compute losses with uncertainty weights
        rep_loss = _compute_reprojection_loss(
            proj, tensors["tracks"], tensors["mask"], point_weights
        )
        ray_loss = _compute_ray_loss(
            cam_pts, tensors["tracks"], tensors["mask"],
            tensors["inv_intrinsics"], device, dtype, point_weights
        )
        depth_loss = _compute_depth_loss(
            cam_pts, tensors["tracks"], tensors["mask"], tensors["depth_priors"], point_weights
        )

        # Weighted total loss
        total_loss = rep_loss + RAY_WEIGHT * ray_loss + DEPTH_WEIGHT * depth_loss
        total_loss.backward()

        print(f"[bundle_adjust_keyframes] Iter {it}: rep={rep_loss.item():.4f}, "
              f"ray={ray_loss.item() * RAY_WEIGHT:.4f}, depth={depth_loss.item() * DEPTH_WEIGHT:.4f}, total={total_loss.item():.4f}")

        # Early stopping: check both reprojection and depth loss thresholds
        rep_converged = rep_loss_thresh is None or rep_loss.item() < rep_loss_thresh
        depth_converged = depth_loss_thresh is None or depth_loss.item() < depth_loss_thresh

        if rep_converged and depth_converged:
            print(f"[bundle_adjust_keyframes] Early stop at iter {it}: "
                  f"rep_loss={rep_loss.item():.4f} < {rep_loss_thresh}, "
                  f"depth_loss={depth_loss.item():.4f} < {depth_loss_thresh}")
            break

        # Freeze reference pose gradients
        if tensors["rvecs"].grad is not None:
            tensors["rvecs"].grad[kf_ref_idx].zero_()
        if tensors["tvecs"].grad is not None:
            tensors["tvecs"].grad[kf_ref_idx].zero_()

        optim.step()

    return tensors["rvecs"], tensors["tvecs"], tensors["points3d"], proj, cam_pts


def _update_image_info(image_info, keyframe_indices, rvecs_t, tvecs_t, points3d_t,
                       proj, cam_pts, tracks_t, mask_t, valid_pts_indices, valid_unc_mask,
                       depth_priors, device, dtype):
    """Extract optimized values and update image_info.

    Args:
        image_info: Original image_info dictionary
        keyframe_indices: Indices of keyframes
        rvecs_t: Optimized rotation vectors
        tvecs_t: Optimized translation vectors
        points3d_t: Optimized 3D points (only valid points)
        proj: Final projected 2D points
        cam_pts: Points in camera coordinates [K, 3, P]
        tracks_t: Track observations
        mask_t: Visibility mask
        valid_pts_indices: Indices of valid points in original points_3d array
        valid_unc_mask: Boolean mask for points with valid (low) uncertainty [P]
        depth_priors: Depth prior maps [K, H, W]
        device: Compute device
        dtype: Data type

    Returns:
        Updated image_info
    """
    K = len(keyframe_indices)

    with torch.no_grad():
        # Build optimized extrinsics
        R_final = axis_angle_to_matrix(rvecs_t)
        extr_final = torch.zeros((K, 4, 4), device=device, dtype=dtype)
        extr_final[:, :3, :3] = R_final
        extr_final[:, :3, 3] = tvecs_t
        extr_final[:, 3, 3] = 1.0

        optimized_extrinsics = extr_final.cpu().numpy()
        optimized_points_3d = points3d_t.cpu().numpy()

        # Compute final reprojection error
        rep_err = (proj - tracks_t) * mask_t.unsqueeze(-1)
        final_rep_err = torch.linalg.norm(rep_err, dim=-1).mean().item()

        # Extract optimized depth (z-coordinate in camera space) for each keyframe
        # cam_pts shape: [K, 3, P] -> z values: [K, P]
        optimized_depth_per_kf = cam_pts[:, 2, :].cpu().numpy()  # [K, P]

    # Update extrinsics for keyframes
    extr_shape = image_info["extrinsics"].shape[-2:]
    for i, kf_idx in enumerate(keyframe_indices):
        if extr_shape == (4, 4):
            image_info["extrinsics"][kf_idx] = optimized_extrinsics[i]
        else:
            image_info["extrinsics"][kf_idx] = optimized_extrinsics[i, :3, :]

    # Update only the valid points in the original array
    # (points with inf uncertainty were not optimized)
    if len(valid_pts_indices) == len(image_info["points_3d"]):
        # All points were valid, just replace
        image_info["points_3d"] = optimized_points_3d
    else:
        # Only update the valid points
        image_info["points_3d"][valid_pts_indices] = optimized_points_3d
        print(f"[_update_image_info] Updated {len(valid_pts_indices)}/{len(image_info['points_3d'])} points")

    # Store valid uncertainty mask (maps back to filtered points, not original)
    # Create full-size mask for original points array
    full_valid_mask = np.zeros(len(image_info["points_3d"]), dtype=bool)
    if valid_unc_mask is not None:
        valid_unc_mask_np = valid_unc_mask.cpu().numpy()
        full_valid_mask[valid_pts_indices] = valid_unc_mask_np
    else:
        full_valid_mask[valid_pts_indices] = True  # All filtered points are valid

    image_info["ba_valid_points_mask"] = full_valid_mask
    image_info["ba_optimized_depth"] = optimized_depth_per_kf  # [K, P_filtered]
    image_info["ba_keyframe_indices"] = keyframe_indices
    image_info["ba_valid_pts_indices"] = valid_pts_indices  # Indices into original points

    # Sample depth priors at track locations for comparison
    if depth_priors is not None:
        H, W = depth_priors.shape[-2:]
        grid_x = 2.0 * tracks_t[..., 0] / (W - 1) - 1.0
        grid_y = 2.0 * tracks_t[..., 1] / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        depth_prior_sampled = torch.nn.functional.grid_sample(
            depth_priors.unsqueeze(1), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(1).squeeze(-1)
        image_info["ba_depth_prior_sampled"] = depth_prior_sampled.cpu().numpy()  # [K, P_filtered]

    print(f"[bundle_adjust_keyframes] Complete. Final mean reproj error: {final_rep_err:.4f}px")
    print(f"[bundle_adjust_keyframes] Valid points for BA: {full_valid_mask.sum()}/{len(full_valid_mask)}")
    return image_info


def bundle_adjust_keyframes(image_info, ref_frame_idx, iters=30, lr=1e-3,
                            rep_loss_thresh=0.2, depth_loss_thresh=0.001):
    """Perform bundle adjustment on keyframes to jointly optimize 3D points and camera poses.

    This function optimizes the merged points_3d and extrinsics for all keyframes using:
    - Reprojection loss: minimize 2D reprojection error
    - Point-to-ray loss: ensure 3D points lie on viewing rays
    - Depth consistency loss: match predicted depth with depth priors

    Points with infinite uncertainty are excluded from optimization.
    Point uncertainties are used as inverse weights (lower uncertainty = higher weight).

    Args:
        image_info: Dictionary containing keyframe, points_3d, pred_tracks, track_mask,
                   extrinsics, intrinsics, depth_priors, uncertainties
        ref_frame_idx: Reference frame index (pose is fixed during optimization)
        iters: Number of optimization iterations
        lr: Learning rate for optimizer
        rep_loss_thresh: Early-stop threshold for reprojection loss
        depth_loss_thresh: Early-stop threshold for depth loss

    Returns:
        Updated image_info with optimized points_3d and extrinsics
    """
    # Validate required keys in image_info
    required_keys = ["keyframe", "points_3d", "pred_tracks", "track_mask",
                    "extrinsics", "intrinsics", "uncertainties"]
    missing_keys = [key for key in required_keys if key not in image_info]
    if missing_keys:
        raise ValueError(f"[bundle_adjust_keyframes] Missing required keys in image_info: {missing_keys}")

    # Get keyframe indices
    keyframe_indices = np.where(image_info["keyframe"])[0]
    if len(keyframe_indices) < 2:
        print(f"[bundle_adjust_keyframes] Less than 2 keyframes ({len(keyframe_indices)}), skipping.")
        return image_info

    print(f"[bundle_adjust_keyframes] Optimizing {len(keyframe_indices)} keyframes, "
          f"{image_info['points_3d'].shape[0]} points")

    # Setup device with proper handling for None depth_priors
    depth_priors = image_info.get("depth_priors")
    if depth_priors is not None and torch.is_tensor(depth_priors):
        device = depth_priors.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float32

    # Step 1: Extract keyframe data (filters out points with inf uncertainty)
    kf_data = _extract_keyframe_data(image_info, keyframe_indices, ref_frame_idx)

    # Check if we have enough valid points
    if kf_data["points_3d"].shape[0] < 10:
        print(f"[bundle_adjust_keyframes] Only {kf_data['points_3d'].shape[0]} valid points, skipping.")
        return image_info

    # Step 2: Initialize optimization tensors (includes uncertainty weights)
    tensors = _init_optimization_tensors(kf_data, device, dtype)

    # Step 3: Run optimization with uncertainty-weighted losses
    rvecs_t, tvecs_t, points3d_t, proj, cam_pts = _run_ba_optimization(
        tensors, kf_data["ref_idx"], iters, lr, rep_loss_thresh, depth_loss_thresh, device, dtype
    )

    # Step 4: Update image_info with optimized values
    image_info = _update_image_info(
        image_info, keyframe_indices, rvecs_t, tvecs_t, points3d_t,
        proj, cam_pts, tensors["tracks"], tensors["mask"], kf_data["valid_pts_indices"],
        tensors["valid_unc_mask"], tensors["depth_priors"], device, dtype
    )

    return image_info


def build_reconstruction_from_tracks(
    points_3d,
    extrinsic,
    intrinsic,
    pred_tracks,
    image_size,
    track_mask,
    shared_camera,
    camera_type,
    points_rgb=None,
):
    """Construct 3D point cloud from optimized tracks.

    Args:
        points_3d: 3D point coordinates
        extrinsic: Camera extrinsic matrices
        intrinsic: Camera intrinsic matrices
        pred_tracks: Predicted track positions
        image_size: Image dimensions
        track_mask: Track visibility mask
        shared_camera: Whether to use shared camera
        camera_type: Camera model type
        points_rgb: Point RGB colors

    Returns:
        COLMAP reconstruction object
    """
    extra_params = None
    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.zeros((pred_tracks.shape[0], 1), dtype=np.float64)

    return batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        image_size,
        masks=track_mask,
        min_inlier_per_frame=0,
        min_inlier_per_track=0,
        shared_camera=shared_camera,
        camera_type=camera_type,
        extra_params=extra_params,
        points_rgb=points_rgb,
    )


def register_new_frame(image_info, gen_3d, frame_idx, args, out_dir, iters=100, depth_weight=0):
    """Optimize only the pose of frame `frame_idx` using reprojection + mesh-depth consistency.

    Args:
        image_info: Dictionary containing reconstruction data
        gen_3d: Generated 3D model object
        frame_idx: Frame index to register
        args: Arguments with configuration
        out_dir: Output directory
        iters: Number of optimization iterations
        depth_weight: Weight for depth consistency loss

    Returns:
        Updated image_info (in-place)
    """
    points_3d = image_info.get("points_3d")
    extrinsics = image_info.get("extrinsics")
    intrinsics = image_info.get("intrinsics")
    pred_tracks = image_info.get("pred_tracks")
    track_mask = image_info.get("track_mask")
    depth_priors = image_info.get("depth_priors")

    missing = [
        name
        for name, val in [
            ("points_3d", points_3d),
            ("extrinsics", extrinsics),
            ("intrinsics", intrinsics),
            ("pred_tracks", pred_tracks),
            ("track_mask", track_mask),
            ("depth_priors", depth_priors),
        ]
        if val is None
    ]

    if missing:
        print(f"[register_new_frame] Missing inputs: {missing}; skipping refinement.")
        return image_info

    # Optimization logic would go here - simplified for module structure
    # The actual implementation remains in the main file for now
    return image_info


def propagate_uncertainty_and_build_image_info(images, image_path_list, base_image_path_list, original_coords,
                                               image_masks, depth_prior, intrinsic, extrinsic,
                                               pred_tracks, track_mask, points_3d, points_rgb, args,
                                               keyframe_indices=None):
    """Build unified image_info dictionary with uncertainty propagation.

    Args:
        images: Input image tensors
        image_path_list: List of image file paths
        base_image_path_list: List of base image filenames
        original_coords: Original image coordinates
        image_masks: Image mask tensors
        depth_prior: Depth prior maps
        intrinsic: Camera intrinsic matrices
        extrinsic: Camera extrinsic matrices
        pred_tracks: Predicted track positions
        track_mask: Track visibility mask
        points_3d: 3D point coordinates
        points_rgb: Point RGB colors
        args: Arguments with threshold parameters
        keyframe_indices: Optional list/array of keyframe indices. If None, keyframes
                          are computed from the data using threshold criteria.

    Returns:
        image_info dictionary with all reconstruction data and uncertainties
    """
    uncertainties = propagate_uncertainties(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        depth_prior,
        track_mask,
        rot_thresh=args.kf_rot_thresh,
        trans_thresh=args.kf_trans_thresh,
        depth_thresh=args.kf_depth_thresh,
        track_inlier_thresh=args.kf_inlier_thresh,
        min_track_number=args.min_track_number,
        keyframe_indices=keyframe_indices,
    )

    image_info = {
        "image_paths": image_path_list,
        "image_names": base_image_path_list,
        "original_coords": original_coords,
        "images": images,
        "image_masks": image_masks,
        "depth_priors": depth_prior,
        "intrinsics": intrinsic,
        "extrinsics": extrinsic,
        "uncertainties": uncertainties,
        "pred_tracks": pred_tracks,
        "track_mask": track_mask,
        "points_3d": points_3d,
        "points_rgb": points_rgb,
    }
    return image_info
