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


def get_points_conf_colors(points_3d, uncertainties):
    """Map uncertainty values to RGB colors for visualization.

    Args:
        points_3d: 3D point coordinates
        uncertainties: Uncertainty values for each point

    Returns:
        RGB colors array where green=high confidence, red=low confidence
    """
    uncertainty = np.asarray(uncertainties, dtype=np.float64)
    conf_colors = np.zeros((len(uncertainty), 3), dtype=np.uint8)
    finite_mask = np.isfinite(uncertainty)
    if finite_mask.any():
        finite_conf = uncertainty[finite_mask]
        conf_norm = finite_conf / (finite_conf.max() + 1e-8)
        finite_colors = np.stack(
            [
                (1.0 - conf_norm) * 255.0,  # red channel high -> low confidence
                conf_norm * 255.0,          # green channel high -> high confidence
                np.zeros_like(conf_norm),
            ],
            axis=-1,
        ).clip(0, 255).astype(np.uint8)
        conf_colors[finite_mask] = finite_colors
    # Non-finite uncertainties (e.g., np.inf) remain black.
    if len(conf_colors) != len(points_3d):
        conf_colors = conf_colors[: len(points_3d)]
    return conf_colors


def save_point_cloud_with_conf(points_3d, points_rgb, uncertainties, ply_path):
    """Save point cloud with uncertainty-based colors.

    Args:
        points_3d: 3D point coordinates
        points_rgb: Original RGB colors (not used currently)
        uncertainties: Uncertainty values for coloring
        ply_path: Output PLY file path
    """
    conf_colors = get_points_conf_colors(points_3d, uncertainties)
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


def eval_reprojection(image_info, frame_idx, intr_np, pts_np, tracks_np, mask_np, R_final, t_final, out_dir):
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

    Returns:
        Path to saved visualization image, or None on failure
    """
    img_paths = image_info.get("image_paths")
    if not img_paths or frame_idx >= len(img_paths):
        return None

    base_img = Image.open(img_paths[frame_idx]).convert("RGB")
    vis_img = np.array(base_img)

    cam_pts = (R_final @ pts_np.T).T + t_final
    z = cam_pts[:, 2:3]
    uv = cam_pts[:, :2] / (z + 1e-8)
    proj = (intr_np @ np.concatenate([uv, np.ones_like(z)], axis=1).T).T[:, :2]

    if proj.shape[0] != tracks_np.shape[0]:
        return None

    mask_np = np.asarray(mask_np).astype(bool)
    end_pts = proj[mask_np]
    start_pts = tracks_np[mask_np]
    if start_pts.shape[0] == 0:
        return None

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_path = out_dir / f"reproj_error.png"
    Image.fromarray(vis_img).save(vis_path)
    return vis_path


def save_results(image_info, gen_3d, out_dir):
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

    points_conf_color = get_points_conf_colors(
        points_3d=image_info.get("points_3d"),
        uncertainties=image_info.get("uncertainties")['points3d']
    )

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
    }

    out_path = Path(out_dir) / "results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    save_point_cloud_with_conf(
        image_info.get("points_3d"),
        image_info.get("points_rgb"),
        image_info.get("uncertainties")['points3d'],
        Path(out_dir) / "points.ply"
    )
    save_depth_prior_with_uncertainty(
        image_info.get("depth_priors"),
        image_info.get("uncertainties")['depth_prior'],
        Path(out_dir) / "depth_conf"
    )
    save_aligned_3D_model(gen_3d, gen_3d.get_aligned_pose(), out_dir)
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
