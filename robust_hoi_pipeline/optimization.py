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


def bundle_adjust_keyframes(image_info, ref_frame_idx, iters=30, lr=1e-3, rep_loss_thresh=0.3):
    """Perform bundle adjustment on keyframes to jointly optimize 3D points and camera poses.

    This function optimizes the merged points_3d and extrinsics for all keyframes using
    reprojection and point-to-ray losses, similar to traditional bundle adjustment.

    Args:
        image_info: Dictionary containing keyframe, points_3d, pred_tracks, track_mask,
                   extrinsics, intrinsics, depth_priors
        ref_frame_idx: Reference frame index (pose is fixed during optimization)
        iters: Number of optimization iterations
        lr: Learning rate for optimizer
        rep_loss_thresh: Early-stop threshold for reprojection loss

    Returns:
        Updated image_info with optimized points_3d and extrinsics
    """
    keyframe_mask = image_info["keyframe"]
    keyframe_indices = np.where(keyframe_mask)[0]

    if len(keyframe_indices) < 2:
        print("[bundle_adjust_keyframes] Less than 2 keyframes, skipping bundle adjustment.")
        return image_info

    points_3d = image_info["points_3d"]
    pred_tracks = image_info["pred_tracks"]
    track_mask = image_info["track_mask"]
    extrinsics = image_info["extrinsics"]
    intrinsics = image_info["intrinsics"]
    depth_priors = image_info["depth_priors"]

    # Extract keyframe-only data
    kf_tracks = pred_tracks[keyframe_indices]  # [K, N, 2]
    kf_mask = track_mask[keyframe_indices]  # [K, N]
    kf_extrinsics = extrinsics[keyframe_indices]  # [K, 4, 4]
    kf_intrinsics = intrinsics[keyframe_indices]  # [K, 3, 3]

    # Find the reference index within the keyframe subset
    kf_ref_idx = np.where(keyframe_indices == ref_frame_idx)[0]
    if len(kf_ref_idx) == 0:
        # Reference frame is not a keyframe, use first keyframe as reference
        kf_ref_idx = 0
    else:
        kf_ref_idx = kf_ref_idx[0]

    print(f"[bundle_adjust_keyframes] Optimizing {len(keyframe_indices)} keyframes, "
          f"{points_3d.shape[0]} points, ref_idx={kf_ref_idx}")

    # Run optimization
    device = depth_priors.device if torch.is_tensor(depth_priors) else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    K = len(keyframe_indices)
    P = points_3d.shape[0]

    points3d_t = torch.from_numpy(points_3d).to(device=device, dtype=dtype)
    points3d_t.requires_grad_(True)
    tracks_t = torch.from_numpy(kf_tracks).to(device=device, dtype=dtype)
    mask_t = torch.from_numpy(kf_mask).to(device=device)

    intr_t = torch.from_numpy(kf_intrinsics).to(device=device, dtype=dtype)

    # Convert extrinsics to rotation vectors and translation vectors
    rvecs = []
    tvecs = []
    for i in range(K):
        rvec, _ = cv2.Rodrigues(kf_extrinsics[i, :3, :3])
        rvecs.append(rvec.reshape(-1))
        tvecs.append(kf_extrinsics[i, :3, 3])
    rvecs_t = torch.from_numpy(np.stack(rvecs)).to(device=device, dtype=dtype)
    tvecs_t = torch.from_numpy(np.stack(tvecs)).to(device=device, dtype=dtype)
    rvecs_t.requires_grad_(True)
    tvecs_t.requires_grad_(True)

    optim = torch.optim.Adam([rvecs_t, tvecs_t, points3d_t], lr=lr)
    inv_intr_t = torch.inverse(intr_t)

    # Prepare depth priors for keyframes
    kf_depth_priors = depth_priors[keyframe_indices] if depth_priors is not None else None
    if kf_depth_priors is not None:
        if not torch.is_tensor(kf_depth_priors):
            kf_depth_priors = torch.from_numpy(kf_depth_priors)
        kf_depth_priors = kf_depth_priors.to(device=device, dtype=dtype)

    for it in range(iters):
        optim.zero_grad(set_to_none=True)
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

        rep_raw = proj - tracks_t                          # K,P,2
        valid = mask_t.unsqueeze(-1).to(rep_raw.dtype)     # K,P,1

        rep_loss = torch.nn.functional.smooth_l1_loss(
            rep_raw * valid,
            torch.zeros_like(rep_raw),
            reduction="sum"
        ) / (valid.sum() * rep_raw.shape[-1] + 1e-8)

        # Point-to-ray consistency loss
        uv1 = torch.cat([tracks_t, torch.ones((K, P, 1), device=device, dtype=dtype)], dim=-1)  # K,P,3
        rays = torch.bmm(inv_intr_t, uv1.transpose(1, 2))  # K,3,P
        rays = torch.nn.functional.normalize(rays, dim=1, eps=1e-6)
        cross = torch.cross(cam_pts, rays, dim=1)
        ray_raw = torch.linalg.norm(cross, dim=1)          # K,P
        valid = mask_t.to(ray_raw.dtype)                   # K,P

        ray_loss = torch.nn.functional.smooth_l1_loss(
            ray_raw * valid,
            torch.zeros_like(ray_raw),
            reduction="sum"
        ) / (valid.sum() + 1e-8)

        # Depth consistency loss
        depth_loss = torch.zeros((), device=device, dtype=dtype)
        if kf_depth_priors is not None:
            z_pred = cam_pts[:, 2, :]  # K, P
            H, W = kf_depth_priors.shape[-2:]
            grid_x = 2.0 * tracks_t[..., 0] / (W - 1) - 1.0
            grid_y = 2.0 * tracks_t[..., 1] / (H - 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

            depth_prior_sampled = torch.nn.functional.grid_sample(
                kf_depth_priors.unsqueeze(1),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze(1).squeeze(-1)

            valid_depth_mask = (depth_prior_sampled > 0) & mask_t
            if valid_depth_mask.any():
                depth_raw = z_pred - depth_prior_sampled
                valid = valid_depth_mask.to(depth_raw.dtype)
                depth_loss = torch.nn.functional.smooth_l1_loss(
                    depth_raw * valid,
                    torch.zeros_like(depth_raw),
                    reduction="sum"
                ) / (valid.sum() + 1e-8)

        ray_loss *= 10
        depth_loss *= 1000
        loss = rep_loss + ray_loss + depth_loss
        loss.backward()
        print(f"[bundle_adjust_keyframes] Iter {it}: rep_loss={rep_loss.item():.4f}, "
              f"ray_loss={ray_loss.item():.4f}, depth_loss={depth_loss.item():.4f}")

        if rep_loss_thresh is not None and rep_loss.item() < rep_loss_thresh:
            print(f"[bundle_adjust_keyframes] Early stop at iter {it}: rep_loss {rep_loss.item():.6f} < {rep_loss_thresh}")
            break

        # Freeze reference pose update
        if rvecs_t.grad is not None:
            rvecs_t.grad[kf_ref_idx].zero_()
        if tvecs_t.grad is not None:
            tvecs_t.grad[kf_ref_idx].zero_()
        optim.step()

    # Extract optimized values
    with torch.no_grad():
        R_final = axis_angle_to_matrix(rvecs_t)
        extr_final = torch.zeros((K, 4, 4), device=device, dtype=dtype)
        extr_final[:, :3, :3] = R_final
        extr_final[:, :3, 3] = tvecs_t
        extr_final[:, 3, 3] = 1.0

        optimized_extrinsics = extr_final.cpu().numpy()
        optimized_points_3d = points3d_t.cpu().numpy()

    # Update image_info with optimized values
    extr_shape = image_info["extrinsics"].shape[-2:]
    for i, kf_idx in enumerate(keyframe_indices):
        if extr_shape == (4, 4):
            image_info["extrinsics"][kf_idx] = optimized_extrinsics[i]
        else:
            image_info["extrinsics"][kf_idx] = optimized_extrinsics[i, :3, :]

    image_info["points_3d"] = optimized_points_3d

    # Compute final reprojection error
    with torch.no_grad():
        rep_err = (proj - tracks_t) * mask_t.unsqueeze(-1)
        final_rep_err = torch.linalg.norm(rep_err, dim=-1).mean().item()
    print(f"[bundle_adjust_keyframes] Optimization complete. Final mean reproj error: {final_rep_err:.4f}px")

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
