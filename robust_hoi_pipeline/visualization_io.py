# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualization and I/O functions for the COLMAP pipeline.
"""

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import trimesh

import sys
sys.path.append("third_party/utils_simba")
from utils_simba.depth import save_depth


def get_points_uncertainty_colors(points_3d, uncertainties, scale=0.4):
    """Map uncertainty values to RGB colors for visualization.

    Args:
        points_3d: 3D point coordinates
        uncertainties: Uncertainty values for each point (expected in [0, 1] range)

    Returns:
        RGB colors array where green=low uncertainty, red=high uncertainty
    """
    uncertainty = np.asarray(uncertainties, dtype=np.float64)
    uncertainty_colors = np.zeros((len(uncertainty), 3), dtype=np.uint8)
    finite_mask = np.isfinite(uncertainty)
    if finite_mask.any():
        finite_uncertainty = uncertainty[finite_mask]
        finite_uncertainty *= scale # scale to [0, scale] for better color contrast
        # Use absolute uncertainty values clamped to [0, 1] instead of normalizing
        uncertainty_clamped = np.clip(finite_uncertainty, 0.0, 1.0)
        finite_colors = np.stack(
            [
                uncertainty_clamped * 255.0,  # red channel high -> low confidence
                (1.0 - uncertainty_clamped) * 255.0,  # green channel high -> high confidence
                np.zeros_like(uncertainty_clamped),
            ],
            axis=-1,
        ).astype(np.uint8)
        uncertainty_colors[finite_mask] = finite_colors
    # Non-finite uncertainties (e.g., np.inf) remain black.
    if len(uncertainty_colors) != len(points_3d):
        uncertainty_colors = uncertainty_colors[: len(points_3d)]
    return uncertainty_colors


def save_low_uncertainty_points_in_obj_space(image_info, gen_3d, out_dir, args):
    """Save 3D points with low uncertainty transformed to object space.

    Args:
        image_info: Dictionary containing points_3d and uncertainties
        gen_3d: Generated 3D model object with aligned pose
        out_dir: Output directory
        args: Arguments with unc_thresh configuration
    """
    points_3d = image_info.get("points_3d")
    uncertainties = image_info.get("uncertainties")
    aligned_pose = gen_3d.get_aligned_pose() if hasattr(gen_3d, "get_aligned_pose") else None

    if points_3d is None or uncertainties is None or aligned_pose is None:
        return

    pts_unc = uncertainties.get('points3d')
    if pts_unc is None:
        return

    pts_unc = np.asarray(pts_unc)
    # Filter points with uncertainty below threshold
    valid_mask = np.isfinite(pts_unc) & (pts_unc <= args.unc_thresh)
    low_unc_points = np.asarray(points_3d)[valid_mask]

    if len(low_unc_points) == 0:
        return

    # Transform points to object space using aligned pose
    # aligned_pose transforms from object space to world space, so we need its inverse
    aligned_pose_inv = np.linalg.inv(aligned_pose)
    low_unc_points_h = np.concatenate([low_unc_points, np.ones((len(low_unc_points), 1))], axis=1)
    low_unc_points_obj = (aligned_pose_inv @ low_unc_points_h.T).T[:, :3]

    # Save with green color for low uncertainty
    low_unc_colors = np.zeros((len(low_unc_points_obj), 3), dtype=np.uint8)
    low_unc_colors[:, 1] = 255  # Green
    trimesh.PointCloud(low_unc_points_obj, colors=low_unc_colors).export(
        Path(out_dir) / "points_low_unc_obj_space.ply"
    )
    print(f"[save_results] Saved {len(low_unc_points_obj)} low-uncertainty points in object space")


def save_point_cloud_with_conf(points_3d, points_rgb, uncertainties, ply_path, args):
    """Save point cloud with uncertainty-based colors.

    Args:
        points_3d: 3D point coordinates
        points_rgb: Original RGB colors (not used currently)
        uncertainties: Uncertainty values for coloring
        ply_path: Output PLY file path
    """
    conf_colors = get_points_uncertainty_colors(points_3d, uncertainties, scale=1/args.unc_thresh)
    trimesh.PointCloud(points_3d, colors=conf_colors).export(ply_path)


def save_depth_prior_with_uncertainty(depth, depth_unc, out_dir):
    """Save per-pixel depth uncertainty maps as color PNGs.

    Colors: green=high confidence, red=low confidence

    Args:
        depth: Depth maps array
        depth_unc: Depth uncertainty maps
        out_dir: Output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(depth_unc.shape[0]):
        unc = depth_unc[i]
        d = np.asarray(depth[i])
        unc_norm = unc / (unc.max() + 1e-8)
        # invert so high confidence => green high, red low
        red = (unc_norm * 255.0).astype(np.uint8)
        green = ((1.0 - unc_norm) * 255.0).astype(np.uint8)
        blue = np.zeros_like(red, dtype=np.uint8)
        rgb = np.stack([red, green, blue], axis=-1)
        alpha = np.where(d > 0, 255, 0).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(Path(out_dir) / f"depth_unc_{i:04d}.png")


def eval_reprojection(image_info, frame_idx, intr_np, pts_np, tracks_np, mask_np, R_final, t_final, out_dir, uncertainties=None, args=None):
    """Overlay reprojection error vectors on the raw image for a frame.

    Args:
        image_info: Dictionary containing image paths and coordinates
        frame_idx: Frame index to evaluate
        intr_np: Intrinsic matrix (3x3)
        pts_np: 3D points array
        tracks_np: 2D track positions
        mask_np: Track mask
        R_final: Final rotation matrix (3x3)
        t_final: Final translation vector (3,)
        out_dir: Output directory
        uncertainties: Point uncertainties for filtering high-uncertainty points
        args: Arguments with unc_thresh configuration

    Returns:
        Path to saved visualization image, or None on failure
    """
    img_paths = image_info.get("image_paths")
    if not img_paths or frame_idx >= len(img_paths):
        return None

    base_img = Image.open(img_paths[frame_idx]).convert("RGB")
    vis_img = np.array(base_img)

    # Filter by uncertainty threshold if provided
    valid_mask = np.ones(len(pts_np), dtype=bool)
    if uncertainties is not None and args is not None:
        unc_thresh = getattr(args, 'unc_thresh', 2.0)
        pts_unc = np.asarray(uncertainties)
        valid_mask = np.isfinite(pts_unc) & (pts_unc <= unc_thresh)

    # Apply uncertainty filter to all point data
    pts_np_filtered = pts_np[valid_mask]
    tracks_np_filtered = tracks_np[valid_mask]
    mask_np_filtered = np.asarray(mask_np)[valid_mask]
    print(f"[eval_reprojection] Frame {frame_idx}: {len(pts_np_filtered)}/{len(pts_np)} points after uncertainty filtering")

    cam_pts = (R_final @ pts_np_filtered.T).T + t_final
    z = cam_pts[:, 2:3]
    uv = cam_pts[:, :2] / (z + 1e-8)
    proj = (intr_np @ np.concatenate([uv, np.ones_like(z)], axis=1).T).T[:, :2]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_path = out_dir / "reproj_error.png"

    if proj.shape[0] != tracks_np_filtered.shape[0]:
        Image.fromarray(vis_img).save(vis_path)
        return vis_path

    mask_np_filtered = np.asarray(mask_np_filtered).astype(bool)
    end_pts = proj[mask_np_filtered]
    start_pts = tracks_np_filtered[mask_np_filtered]
    if start_pts.shape[0] == 0:
        Image.fromarray(vis_img).save(vis_path)
        return vis_path

    orig_coords = image_info.get("original_coords")
    if orig_coords is not None:
        if torch.is_tensor(orig_coords):
            orig_coords = orig_coords.detach().cpu().numpy()
        else:
            orig_coords = np.asarray(orig_coords)
        if orig_coords.ndim >= 2 and frame_idx < orig_coords.shape[0]:
            x1, y1, x2, y2, width, height = orig_coords[frame_idx]
            if width > 0 and height > 0:
                scale_x = (x2 - x1) / float(width)
                scale_y = (y2 - y1) / float(height)
                offset = np.array([x1, y1], dtype=np.float32)
                scale = np.array([scale_x, scale_y], dtype=np.float32)
                start_pts = (start_pts - offset) / scale
                end_pts = (end_pts - offset) / scale

    errors = np.linalg.norm(end_pts - start_pts, axis=1)

    for s, e, err in zip(start_pts, end_pts, errors):
        start = tuple(np.round(s).astype(int))
        end = tuple(np.round(e).astype(int))
        color = (255, 0, 0) if err >= 2.0 else (0, 0, 255)
        cv2.arrowedLine(
            vis_img,
            start,
            end,
            color=color,
            thickness=1,
            tipLength=0.2,
        )

    Image.fromarray(vis_img).save(vis_path)
    return vis_path


def save_results(image_info, gen_3d, out_dir, args):
    """Persist key reconstruction artifacts for later reuse/inspection.

    Args:
        image_info: Dictionary containing all reconstruction data
        gen_3d: Generated 3D model object
        out_dir: Output directory
    """
    # Import here to avoid circular dependency
    from .correspondence_alignment import save_aligned_3D_model

    os.makedirs(out_dir, exist_ok=True)

    def _to_cpu_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    points_conf_color = get_points_uncertainty_colors(
        points_3d=image_info.get("points_3d"),
        uncertainties=image_info.get("uncertainties")['points3d'],
        scale=1/args.unc_thresh
    )

    # Convert uncertainties dict to numpy
    uncertainties_raw = image_info.get("uncertainties")
    if uncertainties_raw is not None:
        uncertainties_np = {k: _to_cpu_numpy(v) for k, v in uncertainties_raw.items()}
    else:
        uncertainties_np = None

    payload = {
        "intrinsics": _to_cpu_numpy(image_info.get("intrinsics")),
        "extrinsics": _to_cpu_numpy(image_info.get("extrinsics")),
        "original_coords": _to_cpu_numpy(image_info.get("original_coords")),
        "pred_tracks": _to_cpu_numpy(image_info.get("pred_tracks")),
        "track_mask": _to_cpu_numpy(image_info.get("track_mask")),
        "points_3d": _to_cpu_numpy(image_info.get("points_3d")),
        "points_rgb": _to_cpu_numpy(image_info.get("points_rgb")),
        "points_conf_color": _to_cpu_numpy(points_conf_color),
        "registered": _to_cpu_numpy(image_info.get("registered")),
        "invalid": _to_cpu_numpy(image_info.get("invalid")),
        "keyframe": _to_cpu_numpy(image_info.get("keyframe")),
        "aligned_pose": gen_3d.get_aligned_pose() if hasattr(gen_3d, "get_aligned_pose") else None,
        "uncertainties": uncertainties_np,
        # BA optimization data
        "ba_valid_points_mask": _to_cpu_numpy(image_info.get("ba_valid_points_mask")),
        "ba_optimized_depth": _to_cpu_numpy(image_info.get("ba_optimized_depth")),
        "ba_keyframe_indices": _to_cpu_numpy(image_info.get("ba_keyframe_indices")),
        "ba_valid_pts_indices": _to_cpu_numpy(image_info.get("ba_valid_pts_indices")),
        "ba_depth_prior_sampled": _to_cpu_numpy(image_info.get("ba_depth_prior_sampled")),
    }

    out_path = Path(out_dir) / "results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    save_point_cloud_with_conf(
        image_info.get("points_3d"),
        image_info.get("points_rgb"),
        image_info.get("uncertainties")['points3d'],
        Path(out_dir) / "points.ply",
        args=args
    )

    # Save valid BA points as separate PLY (green color for valid points)
    ba_valid_mask = image_info.get("ba_valid_points_mask")
    if ba_valid_mask is not None:
        points_3d = image_info.get("points_3d")
        valid_points = points_3d[ba_valid_mask]
        if len(valid_points) > 0:
            valid_colors = np.zeros((len(valid_points), 3), dtype=np.uint8)
            valid_colors[:, 1] = 255  # Green for valid points
            trimesh.PointCloud(valid_points, colors=valid_colors).export(
                Path(out_dir) / "points_ba_valid.ply"
            )
            print(f"[save_results] Saved {len(valid_points)} BA-valid points to points_ba_valid.ply")

    save_depth_prior_with_uncertainty(
        image_info.get("depth_priors"),
        image_info.get("uncertainties")['depth_prior'],
        Path(out_dir) / "depth_conf"
    )
    if payload["aligned_pose"] is not None:
        save_aligned_3D_model(gen_3d, gen_3d.get_aligned_pose(), out_dir)
    save_low_uncertainty_points_in_obj_space(image_info, gen_3d, out_dir, args)

    print(f"[save_results] Saved reconstruction summary to {out_path}")

    frame_idx = int(Path(out_dir).name)
    eval_reprojection(
        image_info=image_info,
        frame_idx=frame_idx,
        intr_np=payload["intrinsics"][frame_idx],
        pts_np=payload["points_3d"],
        tracks_np=payload["pred_tracks"][frame_idx],
        mask_np=payload["track_mask"][frame_idx],
        R_final=payload["extrinsics"][frame_idx][:3, :3],
        t_final=payload["extrinsics"][frame_idx][:3, 3],
        out_dir=out_dir,
        uncertainties=image_info.get("uncertainties", {}).get("points3d"),
        args=args,
    )


def save_input_data(images, image_masks, depth_prior, gen_3d, image_path_list, out_dir):
    """Save preprocessed inputs to disk for inspection/debugging.

    Args:
        images: Preprocessed image tensors
        image_masks: Image mask tensors
        depth_prior: Depth prior maps
        gen_3d: Generated 3D model object
        image_path_list: List of original image paths
        out_dir: Output directory
    """
    images_dir = Path(out_dir) / "images"
    images_origin_dir = Path(out_dir) / "images_origin"
    masks_dir = Path(out_dir) / "masks"
    depth_dir = Path(out_dir) / "depth_prior"
    for d in (images_dir, images_origin_dir, masks_dir, depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_frames = len(images)
    for idx in range(num_frames):
        depth = depth_prior[idx]
        if torch.is_tensor(depth):
            depth = depth.detach().cpu()

        # Save RGB image
        img = images[idx].detach().cpu()
        img_uint8 = (img.clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).byte().numpy()
        Image.fromarray(img_uint8, mode="RGB").save(images_dir / f"{idx:04d}.png")
        shutil.copy2(image_path_list[idx], images_origin_dir / f"{idx:04d}{Path(image_path_list[idx]).suffix}")

        # Save mask as single-channel PNG
        mask = image_masks[idx].detach().cpu()
        mask_uint8 = (mask.squeeze(0).clamp(0.0, 1.0) * 255.0).byte().numpy()
        Image.fromarray(mask_uint8, mode="L").save(masks_dir / f"{idx:04d}.png")

        # Save depth prior using 24-bit PNG encoding
        save_depth(np.asarray(depth, dtype=np.float32), str(depth_dir / f"{idx:04d}.png"))

    if gen_3d is not None:
        gen3d_dir = Path(out_dir) / "gen_3d"
        gen3d_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = getattr(gen_3d, "mesh_path", None) or getattr(gen_3d, "get_mesh_path", lambda: None)()
        if mesh_path and os.path.exists(mesh_path):
            shutil.copy2(mesh_path, gen3d_dir / Path(mesh_path).name)

        cond_img = getattr(gen_3d, "condition_image_path", None)
        if cond_img and os.path.exists(cond_img):
            shutil.copy2(cond_img, gen3d_dir / Path(cond_img).name)

        depth_map = getattr(gen_3d, "depth_path", None)
        if depth_map and os.path.exists(depth_map):
            shutil.copy2(depth_map, gen3d_dir / Path(depth_map).name)

        camera_path = getattr(gen_3d, "camera_path", None)
        if camera_path and os.path.exists(camera_path):
            shutil.copy2(camera_path, gen3d_dir / Path(camera_path).name)

    # save image_path_list to a text file
    with open(Path(out_dir) / "image_paths.txt", "w") as f:
        for p in image_path_list:
            f.write(f"{p}\n")
