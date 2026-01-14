# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
3D correspondence and alignment functions for the COLMAP pipeline.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import cv2

from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.projection import project_3D_points_np
from vggt.utils.visual_track import visualize_tracks_on_images


def get_3D_correspondences(gen_3d, reference, reference_idx=0, out_dir=None, min_vis_score=0.2):
    """Compute 3D correspondences between generated model and image observations.

    Uses mesh-based ray-casting for correspondence estimation.

    Args:
        gen_3d: Generated 3D model object
        reference: Reference data dictionary with images, intrinsics, extrinsics
        reference_idx: Reference frame index
        out_dir: Output directory for visualizations
        min_vis_score: Minimum visibility score threshold

    Returns:
        Dictionary containing condition and reference 3D points and uncertainties
    """
    def _pixels_to_world(uv, depth, intrinsic, extrinsic):
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x_cam = (uv[:, 0] - cx) / fx * depth
        y_cam = (uv[:, 1] - cy) / fy * depth
        pts_cam = np.stack([x_cam, y_cam, depth], axis=1)
        R = extrinsic[:3, :3].astype(np.float64, copy=False)
        t = extrinsic[:3, 3].astype(np.float64, copy=False)
        return (pts_cam - t[None, :]) @ R

    ref_images = reference["images"]
    if ref_images is None or reference_idx >= len(ref_images):
        print("[get_3D_correspondences] Invalid reference image index.")
        return None

    # Load condition data
    cond_img_raw = gen_3d.get_cond_image()
    cond_mask = gen_3d.get_cond_mask()
    cond_depth = gen_3d.get_cond_depth()
    if cond_depth.ndim == 3:
        cond_depth = np.squeeze(cond_depth)
    cond_intr = gen_3d.get_cond_intrinsic()
    cond_extr = gen_3d.get_cond_extrinsic()

    if cond_extr.shape[0] == 4:
        cond_extr = cond_extr[:3]

    # Prepare reference data
    ref_img = reference['images'][reference_idx]
    ref_mask = reference['image_masks'][reference_idx]
    ref_depth = reference['depth_priors'][reference_idx]
    ref_depth_unc = reference["uncertainties"]["depth_prior"][reference_idx]
    ref_depth_unc = np.asarray(np.squeeze(ref_depth_unc), dtype=np.float32)

    ref_intr = np.asarray(reference['intrinsics'][reference_idx], dtype=np.float64)
    ref_extr = np.asarray(reference['extrinsics'][reference_idx], dtype=np.float64)
    if ref_extr.shape[0] == 4:
        ref_extr = ref_extr[:3]

    # Resize condition assets to match reference resolution
    h_ref, w_ref = ref_img.shape[1:3]
    h_cond, w_cond = cond_img_raw.shape[1:3]

    cond_img = cond_img_raw
    cond_mask_proc = cond_mask
    if (h_cond, w_cond) != (h_ref, w_ref):
        cond_img = cv2.resize(cond_img_raw.transpose(1, 2, 0), (w_ref, h_ref), interpolation=cv2.INTER_CUBIC)
        cond_img = cond_img.transpose(2, 0, 1)
        cond_mask_proc = cv2.resize(cond_mask.astype(np.uint8), (w_ref, h_ref), interpolation=cv2.INTER_NEAREST) > 0

    # Build 2-frame stack and run predict_tracks
    device = ref_img.device if torch.is_tensor(ref_img) else torch.device("cpu")
    cond_img_t = torch.from_numpy(cond_img).to(device=device, dtype=torch.float32)
    ref_img_t = ref_images[reference_idx]

    imgs_stack = torch.stack([cond_img_t, ref_img_t], dim=0)
    cond_mask_t = torch.from_numpy(cond_mask_proc.astype(np.float32)).unsqueeze(0).to(device=device, dtype=torch.float32)
    ref_mask_t = ref_mask
    masks_stack = torch.cat([cond_mask_t, ref_mask_t], dim=0)

    with torch.no_grad():
        pred_tracks, pred_vis_scores, _, _, _ = predict_tracks(
            imgs_stack,
            image_masks=masks_stack,
            conf=None,
            points_3d=None,
            max_query_pts=1024,
            query_frame_num=2,
            fine_tracking=False,
            complete_non_vis=False,
        )

    if out_dir:
        visualize_tracks_on_images(imgs_stack[None], torch.from_numpy(pred_tracks[None]), out_dir=f"{out_dir}/track_raw")

    vis_mask = pred_vis_scores > min_vis_score
    if out_dir:
        visualize_tracks_on_images(imgs_stack[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(vis_mask[None]), out_dir=f"{out_dir}/track_vis")

    valid = vis_mask[0] & vis_mask[1]
    if not np.any(valid):
        raise ValueError("[get_3D_correspondences] No valid predicted correspondences.")

    cond_pixels = pred_tracks[0, valid]
    ref_pixels = pred_tracks[1, valid]

    cond_pixels_orig = cond_pixels.copy()
    if (h_cond, w_cond) != (h_ref, w_ref):
        cond_pixels_orig[:, 0] *= float(w_cond) / float(w_ref)
        cond_pixels_orig[:, 1] *= float(h_cond) / float(h_ref)

    cond_depth_vals = cond_depth[
        np.clip(np.round(cond_pixels_orig[:, 1]).astype(int), 0, h_cond - 1),
        np.clip(np.round(cond_pixels_orig[:, 0]).astype(int), 0, w_cond - 1),
    ]

    ref_depth_vals = ref_depth[
        np.clip(np.round(ref_pixels[:, 1]).astype(int), 0, ref_depth.shape[0] - 1),
        np.clip(np.round(ref_pixels[:, 0]).astype(int), 0, ref_depth.shape[1] - 1),
    ]
    ref_depth_vals = ref_depth_vals.cpu().numpy() if torch.is_tensor(ref_depth_vals) else ref_depth_vals

    ref_unc_vals = ref_depth_unc[
        np.clip(np.round(ref_pixels[:, 1]).astype(int), 0, ref_depth_unc.shape[0] - 1),
        np.clip(np.round(ref_pixels[:, 0]).astype(int), 0, ref_depth_unc.shape[1] - 1),
    ]

    valid_depth = (cond_depth_vals > 0) & (ref_depth_vals > 0)
    if not np.any(valid_depth):
        raise ValueError("[get_3D_correspondences] No valid depth correspondences.")

    cond_pixels_orig = cond_pixels_orig[valid_depth]
    ref_pixels = ref_pixels[valid_depth]
    cond_depth_vals = cond_depth_vals[valid_depth]
    ref_depth_vals = ref_depth_vals[valid_depth]
    ref_unc_vals = ref_unc_vals[valid_depth]

    cond_world = _pixels_to_world(cond_pixels_orig, cond_depth_vals, cond_intr, cond_extr)
    ref_world = _pixels_to_world(ref_pixels, ref_depth_vals, ref_intr, ref_extr)

    print(f"[get_3D_correspondences] Found {len(cond_world)} 3D correspondences.")

    return {
        "condition_points_world": cond_world,
        "reference_points_world": ref_world,
        "reference_uncertainties": ref_unc_vals,
    }


def evaluate_3D_corres(corres_3d, gen_3d, reference, reference_idx=0, out_dir=None):
    """Evaluate 3D correspondences quality.

    Args:
        corres_3d: 3D correspondence dictionary
        gen_3d: Generated 3D model object
        reference: Reference data dictionary
        reference_idx: Reference frame index
        out_dir: Output directory for visualizations

    Returns:
        Evaluation metrics dictionary or None
    """
    if corres_3d is None:
        print("[evaluate_3D_corres] No correspondences to evaluate.")
        return None

    cond_pts = corres_3d.get("condition_points_world", None)
    ref_pts = corres_3d.get("reference_points_world", None)

    if cond_pts is None or ref_pts is None:
        print("[evaluate_3D_corres] Missing 3D correspondences.")
        return None

    # Compute distance metrics
    dists = np.linalg.norm(cond_pts - ref_pts, axis=1)
    metrics = {
        "mean_dist": float(np.mean(dists)),
        "median_dist": float(np.median(dists)),
        "std_dist": float(np.std(dists)),
        "num_correspondences": len(dists),
    }

    print(f"[evaluate_3D_corres] Mean distance: {metrics['mean_dist']:.4f}")
    return metrics


def eval_aligned_3D_model(cond_pts, ref_pts, aligned_pose, references, reference_idx, out_dir):
    """Evaluate alignment quality of 3D model.

    Args:
        cond_pts: Condition 3D points
        ref_pts: Reference 3D points
        aligned_pose: Alignment transformation
        references: Reference data dictionary
        reference_idx: Reference frame index
        out_dir: Output directory

    Returns:
        Evaluation metrics
    """
    if cond_pts is None or ref_pts is None:
        return None

    # Apply alignment and compute residuals
    if aligned_pose is not None:
        R = aligned_pose[:3, :3]
        t = aligned_pose[:3, 3]
        s = aligned_pose[3, 3] if aligned_pose.shape[0] > 3 else 1.0
        aligned_cond = s * (cond_pts @ R.T) + t
    else:
        aligned_cond = cond_pts

    residuals = np.linalg.norm(aligned_cond - ref_pts, axis=1)
    return {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
    }


def align_3D_model_with_images(corres, gen_3d, references, reference_idx, out_dir, iters=100):
    """Align generated 3D model with reconstructed points using weighted Umeyama alignment.

    Args:
        corres: 3D correspondence dictionary
        gen_3d: Generated 3D model object
        references: Reference data dictionary
        reference_idx: Reference frame index
        out_dir: Output directory
        iters: Number of iterations

    Returns:
        Aligned pose transformation matrix
    """
    if corres is None:
        print("[align_3D_model_with_images] No correspondences provided.")
        return None

    cond_pts = corres.get("condition_points_world")
    ref_pts = corres.get("reference_points_world")
    ref_unc = corres.get("reference_uncertainties")

    if cond_pts is None or ref_pts is None:
        print("[align_3D_model_with_images] Missing points.")
        return None

    # Compute weights from uncertainties (lower uncertainty = higher weight)
    if ref_unc is not None:
        weights = 1.0 / (ref_unc + 1e-6)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(cond_pts)) / len(cond_pts)

    # Weighted Umeyama alignment
    cond_mean = np.average(cond_pts, weights=weights, axis=0)
    ref_mean = np.average(ref_pts, weights=weights, axis=0)

    cond_centered = cond_pts - cond_mean
    ref_centered = ref_pts - ref_mean

    W = np.diag(weights)
    H = cond_centered.T @ W @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_cond = np.sum(weights * np.sum(cond_centered ** 2, axis=1))
    scale = np.sum(S) / var_cond

    # Compute translation
    t = ref_mean - scale * R @ cond_mean

    # Build 4x4 transformation matrix
    aligned_pose = np.eye(4, dtype=np.float64)
    aligned_pose[:3, :3] = scale * R
    aligned_pose[:3, 3] = t

    # Save transformation
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(Path(out_dir) / "transform.json", "w") as f:
            json.dump({"matrix": aligned_pose.tolist()}, f)

    gen_3d.save_aligned_pose(aligned_pose)
    print(f"[align_3D_model_with_images] Alignment complete. Scale: {scale:.4f}")

    return aligned_pose


def save_aligned_3D_model(gen_3d, aligned_pose, output_path):
    """Save aligned 3D model and transformation.

    Args:
        gen_3d: Generated 3D model object
        aligned_pose: Alignment transformation matrix
        output_path: Output directory path
    """
    import shutil

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get mesh path and copy aligned mesh
    mesh_path = getattr(gen_3d, "mesh_path", None)
    if mesh_path and os.path.exists(mesh_path):
        import trimesh
        mesh = trimesh.load(mesh_path)
        if aligned_pose is not None:
            mesh.apply_transform(aligned_pose)
        mesh.export(output_path / "white_mesh_remesh_aligned.obj")

    # Save transformation
    if aligned_pose is not None:
        with open(output_path / "aligned_transform.json", "w") as f:
            json.dump({"matrix": aligned_pose.tolist()}, f)
