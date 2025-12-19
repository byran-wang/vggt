# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import imageio
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
import cv2
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square, load_intrinsics, GEN_3D
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.dependency.sfm_prepair import prepare_features, prepare_matches, prepare_pairs
from vggt.dependency.projection import project_3D_points_np

import sys
sys.path.append("third_party/Hierarchical-Localization/")
from hloc.reconstruction import main as hloc_reconstruction_main
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import save_point_cloud_to_ply

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    parser.add_argument("--use_sfm", action="store_true", default=False, help="Use SfM for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--use_calibrated_intrinsic", action="store_true", default=False, help="Use calibrated intrinsic for reconstruction")
    parser.add_argument("--min_inlier_per_frame", type=int, default=10, help="Minimum inliers per frame for BA")
    parser.add_argument("--min_inlier_per_track", type=int, default=4, help="Minimum inliers per track for BA")
    parser.add_argument("--max_frames", type=int, default=50, help="Maximum number of frames to process")
    parser.add_argument("--instance_id", type=int, default=0, help="Instance ID for image preprocessing")
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

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
def save_intrinsics(intrinsic, filepath):
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0, 2], intrinsic[1, 2]
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.savetxt(filepath, K, fmt="%.8f")


def prep_valid_correspondences(points_3d, track_mask, min_inlier_per_frame, min_inlier_per_track):
    """Filter tracks by per-frame/track counts and drop 3D points with no surviving tracks."""
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


def compute_normals_from_depth(depth_map, intrinsics):
    """Compute per-pixel normals from depth maps."""
    B, H, W = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    z = depth_map
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    pts = torch.stack([x, y, z], dim=-1)  # B,H,W,3

    dx = pts[:, :, 1:] - pts[:, :, :-1]  # B,H,W-1,3
    dy = pts[:, 1:, :] - pts[:, :-1, :]  # B,H-1,W,3

    dx_full = torch.empty_like(pts)
    dy_full = torch.empty_like(pts)
    dx_full[:, :, :-1] = dx
    dx_full[:, :, -1] = dx[:, :, -1]
    dy_full[:, :-1, :] = dy
    dy_full[:, -1, :] = dy[:, -1, :]

    normals = torch.cross(dx_full, dy_full, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    return normals.permute(0, 3, 1, 2)  # B,3,H,W


def axis_angle_to_matrix(rvecs):
    """Convert batched axis-angle vectors to rotation matrices."""
    device, dtype = rvecs.device, rvecs.dtype
    B = rvecs.shape[0]
    theta = torch.linalg.norm(rvecs, dim=1, keepdim=True).clamp(min=1e-9)  # (B,1)
    k = rvecs / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zero, -kz, ky], dim=-1),
            torch.stack([kz, zero, -kx], dim=-1),
            torch.stack([-ky, kx, zero], dim=-1),
        ],
        dim=1,
    )  # (B,3,3)
    I = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3)
    sin_t = torch.sin(theta).view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)
    k_outer = (k[:, :, None] * k[:, None, :])  # (B,3,3)
    R = cos_t * I + (1 - cos_t) * k_outer + sin_t * K
    return R


def optimize_poses_with_losses(
    points_3d,
    extrinsic,
    intrinsic,
    pred_tracks,
    depth_prior,
    track_mask,
    camera_type="SIMPLE_PINHOLE",
    iters=5,
    lr=1e-3,
):
    """Lightweight pose refinement using reprojection and point-to-ray losses; also returns simple uncertainty estimates."""
    device = depth_prior.device if torch.is_tensor(depth_prior) else "cpu"
    dtype = torch.float32

    P = points_3d.shape[0]
    B = extrinsic.shape[0]

    points3d_t = torch.from_numpy(points_3d).to(device=device, dtype=dtype)
    points3d_t.requires_grad_(True)
    tracks_t = torch.from_numpy(pred_tracks).to(device=device, dtype=dtype)
    mask_t = torch.from_numpy(track_mask).to(device=device)

    intr_t = torch.from_numpy(intrinsic).to(device=device, dtype=dtype)
    rvecs = []
    tvecs = []
    for i in range(B):
        rvec, _ = cv2.Rodrigues(extrinsic[i, :3, :3])
        rvecs.append(rvec.reshape(-1))
        tvecs.append(extrinsic[i, :3, 3])
    rvecs_t = torch.from_numpy(np.stack(rvecs)).to(device=device, dtype=dtype)
    tvecs_t = torch.from_numpy(np.stack(tvecs)).to(device=device, dtype=dtype)
    rvecs_t.requires_grad_(True)
    tvecs_t.requires_grad_(True)

    optim = torch.optim.Adam([rvecs_t, tvecs_t, points3d_t], lr=lr)
    inv_intr_t = torch.inverse(intr_t)

    for _ in range(iters):
        optim.zero_grad(set_to_none=True)
        R = axis_angle_to_matrix(rvecs_t)
        extr_mat = torch.cat([R, tvecs_t.unsqueeze(-1)], dim=-1)  # B,3,4
        ones = torch.ones((B, P, 1), device=device, dtype=dtype)
        pts_h = torch.cat([points3d_t.unsqueeze(0).expand(B, -1, -1), ones], dim=-1)  # B,P,4
        cam_pts = torch.bmm(extr_mat, pts_h.transpose(1, 2))  # B,3,P

        z = cam_pts[:, 2:3, :]
        uv = cam_pts[:, :2, :] / (z + 1e-6)
        ones2 = torch.ones((B, 1, P), device=device, dtype=dtype)
        uv_h = torch.cat([uv, ones2], dim=1)
        proj = torch.bmm(intr_t, uv_h)[:, :2, :].transpose(1, 2)  # B,P,2

        rep_err = (proj - tracks_t) * mask_t.unsqueeze(-1)
        rep_loss = torch.nn.functional.smooth_l1_loss(rep_err, torch.zeros_like(rep_err), reduction="sum")

        # sparse 3D correspondence loss: point-to-ray consistency from 2D tracks
        uv1 = torch.cat([tracks_t, torch.ones((B, P, 1), device=device, dtype=dtype)], dim=-1)  # B,P,3
        rays = torch.bmm(inv_intr_t, uv1.transpose(1, 2))  # B,3,P
        rays = torch.nn.functional.normalize(rays, dim=1, eps=1e-6)
        cross = torch.cross(cam_pts, rays, dim=1)
        ray_err = torch.linalg.norm(cross, dim=1) * mask_t
        ray_loss = torch.nn.functional.smooth_l1_loss(ray_err, torch.zeros_like(ray_err), reduction="sum")

        loss = rep_loss + 0.1 * ray_loss
        loss.backward()
        optim.step()
    # propagate uncertainties
    with torch.no_grad():
        R_final = axis_angle_to_matrix(rvecs_t)
        extr_final = torch.cat([R_final, tvecs_t.unsqueeze(-1)], dim=-1)

        ones = torch.ones((B, P, 1), device=device, dtype=dtype)
        pts_h = torch.cat([points3d_t.unsqueeze(0).expand(B, -1, -1), ones], dim=-1)
        cam_pts = torch.bmm(extr_final, pts_h.transpose(1, 2))

        z = cam_pts[:, 2:3, :]
        uv = cam_pts[:, :2, :] / (z + 1e-6)
        ones2 = torch.ones((B, 1, P), device=device, dtype=dtype)
        uv_h = torch.cat([uv, ones2], dim=1)
        proj = torch.bmm(intr_t, uv_h)[:, :2, :].transpose(1, 2)

        rep_err = (proj - tracks_t) * mask_t.unsqueeze(-1)
        rep_l2 = torch.linalg.norm(rep_err, dim=-1)
        rep_unc_frame = torch.sqrt((rep_l2.pow(2) * mask_t).sum(-1) / mask_t.sum(-1).clamp(min=1)).cpu().numpy()

        uv1 = torch.cat([tracks_t, torch.ones((B, P, 1), device=device, dtype=dtype)], dim=-1)
        rays = torch.bmm(inv_intr_t, uv1.transpose(1, 2))
        rays = torch.nn.functional.normalize(rays, dim=1, eps=1e-6)
        cross = torch.cross(cam_pts, rays, dim=1)
        ray_err = torch.linalg.norm(cross, dim=1) * mask_t
        pts_unc = torch.sqrt((ray_err.pow(2)).sum(0) / mask_t.sum(0).clamp(min=1)).cpu().numpy()

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
            for ref in range(B):
                fx_r, fy_r = intr_t[ref, 0, 0], intr_t[ref, 1, 1]
                cx_r, cy_r = intr_t[ref, 0, 2], intr_t[ref, 1, 2]
                Rr = extr_final[ref, :3, :3]
                tr = extr_final[ref, :3, 3]
                unc_flat = depth_unc[ref].view(-1)
                cnt_flat = depth_cnt[ref].view(-1)
                for src in range(B):
                    if src == ref:
                        continue
                    if rep_unc_frame[src] > 5.0 or rep_unc_frame[src] == 0.: # 0 means no extrinsic estimated
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
                    
                    if 0:
                        w2c_s = torch.eye(4)
                        w2c_s[:3, :3] = Rs
                        w2c_s[:3, 3] = ts
                        w2c_s = w2c_s.to(device=device, dtype=dtype)
                        ones = torch.ones((pts_cam_s.shape[0], 1), device=device, dtype=dtype)
                        pts_cam_s = torch.cat([pts_cam_s, ones], dim=-1).transpose(0, 1)  # 4,M
                        pts_world = w2c_s.inverse() @ pts_cam_s  # M,4
                        
                        # transform pts_world to ref camera
                        w2c_r = torch.eye(4)
                        w2c_r[:3, :3] = Rr
                        w2c_r[:3, 3] = tr
                        w2c_r = w2c_r.to(device=device, dtype=dtype)
                        pts_cam_r = w2c_r @ pts_world  # 4,M
                        pts_cam_r = pts_cam_r.transpose(0, 1)  # M,4
                        pts_cam_r = pts_cam_r[:, :3]
                    else:
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
                    
                    # save depth_unc to a png for visualization
                    # depth_unc_debug = (depth_unc / depth_cnt.clamp(min=1)).sqrt().cpu().numpy()
                    # save_depth_prior_with_uncertainty(depth_unc_debug[:1], out_dir="depth_uncertainty_vis")

            depth_unc = (depth_unc / depth_cnt.clamp(min=1)).sqrt().cpu().numpy()

    uncertainties = {
        "extrinsic": rep_unc_frame,
        "points3d": pts_unc,
        "depth_prior": depth_unc,
    }

    return extr_final.detach().cpu().numpy(), points3d_t.detach().cpu().numpy(), uncertainties


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

def estimate_extrinsic(depth_map, intrinsic, tracks, track_mask):
    """
    Estimate per-frame camera extrinsics (camera-from-world, OpenCV convention).

    Assumptions:
    - Frame 0 defines the world coordinate system (identity extrinsic).
    - `tracks[t, j]` provides the (x, y) pixel of track j in frame t in the same
      pixel coordinate system as `depth_map` and `intrinsic`.
    - `depth_map[t]` gives metric depth along camera Z (OpenCV: z-forward).
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
    if intrinsic.shape != (3, 3):
        raise ValueError(f"`intrinsic` must have shape (3, 3), got {intrinsic.shape}")

    num_frames = tracks.shape[0]
    height, width = depth_map.shape[-2:]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    def _sample_depth_nearest(depth_hw: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        R = extri[:, :3].astype(np.float64, copy=False)
        t = extri[:, 3].astype(np.float64, copy=False)
        # X_world = R^T (X_cam - t)
        return (points_cam - t[None, :]) @ R

    extrinsics = np.zeros((num_frames, 3, 4), dtype=np.float64)
    extrinsics[0, :3, :3] = np.eye(3, dtype=np.float64)
    extrinsics[0, :3, 3] = 0.0

    dist_coeffs = None  # assume no distortion
    ransac_reproj_threshold = 8.0

    for frame_idx in range(1, num_frames):
        estimated = False

        for ref_idx in (frame_idx - 1, 0):
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
            estimated = True
            break

        if not estimated:
            extrinsics[frame_idx] = extrinsics[frame_idx - 1]
            print(
                f"[estimate_extrinsic] Warning: PnP failed for frame {frame_idx}, "
                f"carrying pose from frame {frame_idx - 1}."
            )

    return extrinsics


def verify_tracks_by_geometry(points3d, extrinsics, intrinsics, tracks, masks=None, max_reproj_error=None):
    reproj_mask = None
    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    return masks

def evaluate_3d_corres(corres_3d, gen_3d, reference, reference_idx=0, out_dir=None):
    if corres_3d is None:
        print("[evaluate_3d_corres] No correspondences to evaluate.")
        return None
    if out_dir is None:
        print("[evaluate_3d_corres] No output directory specified, skipping visualization.")
        return None
    cond_pts = corres_3d.get("condition_points_world", None)
    ref_pts = corres_3d.get("reference_points_world", None)
    cond_pixels = corres_3d.get("condition_pixels", None)
    ref_pixels = corres_3d.get("reference_pixels", None)
    if cond_pts is None or ref_pts is None:
        print("[evaluate_3d_corres] Missing 3D correspondences.")
        return None

    # Load condition assets
    cond_img = gen_3d.get_cond_image()
    cond_intr = np.asarray(gen_3d.get_cond_intrinsic(), dtype=np.float64)
    cond_extr = np.asarray(gen_3d.get_cond_extrinsic(), dtype=np.float64)
    if cond_extr.shape[0] == 4:
        cond_extr = cond_extr[:3]

    # Load reference assets
    ref_imgs = reference.get("images", None)
    if ref_imgs is None or reference_idx >= len(ref_imgs):
        print("[evaluate_3d_corres] Invalid reference image index.")
        return None
    ref_img_t = ref_imgs[reference_idx]
    ref_img = ref_img_t.detach().cpu().numpy()
    if ref_img.shape[0] == 3:
        ref_img = np.transpose(ref_img, (1, 2, 0))
    ref_img = np.clip(ref_img * 255.0, 0, 255).astype(np.uint8)

    ref_intr = np.asarray(reference["intrinsics"][reference_idx], dtype=np.float64)
    ref_extr = np.asarray(reference["extrinsics"][reference_idx], dtype=np.float64)
    if ref_extr.shape[0] == 4:
        ref_extr = ref_extr[:3]

    # Project 3D points
    cond_proj, _ = project_3D_points_np(cond_pts, cond_extr[None], cond_intr[None])
    ref_proj, _ = project_3D_points_np(ref_pts, ref_extr[None], ref_intr[None])
    cond_proj = cond_proj[0]
    ref_proj = ref_proj[0]

    # Visualize overlays
    cond_viz = cond_img.copy() * 255
    ref_viz = ref_img.copy()
    # Ensure HWC layout
    if cond_viz.ndim == 3 and cond_viz.shape[0] == 3:
        cond_viz = np.transpose(cond_viz, (1, 2, 0))
    if ref_viz.ndim == 3 and ref_viz.shape[0] == 3:
        ref_viz = np.transpose(ref_viz, (1, 2, 0))
    if cond_viz.ndim == 2:
        cond_viz = np.repeat(cond_viz[..., None], 3, axis=-1)
    if ref_viz.ndim == 2:
        ref_viz = np.repeat(ref_viz[..., None], 3, axis=-1)
    cond_viz = cond_viz.astype(np.uint8, copy=False)
    ref_viz = ref_viz.astype(np.uint8, copy=False)
    cond_viz = np.ascontiguousarray(cond_viz)
    ref_viz = np.ascontiguousarray(ref_viz)

    for p in cond_proj.astype(int):
        cv2.circle(cond_viz, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
    for p in ref_proj.astype(int):
        cv2.circle(ref_viz, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

    # Draw correspondence lines in concatenated space if we save visuals

    eval_dir = Path(out_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if cond_viz.shape[0] != ref_viz.shape[0]:
        max_h = max(cond_viz.shape[0], ref_viz.shape[0])
        def _pad_to_h(img, target_h):
            pad_h = target_h - img.shape[0]
            if pad_h <= 0:
                return img
            return np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
        cond_viz_pad = _pad_to_h(cond_viz, max_h)
        ref_viz_pad = _pad_to_h(ref_viz, max_h)
    else:
        cond_viz_pad, ref_viz_pad = cond_viz, ref_viz
    concat_viz = np.concatenate([cond_viz_pad, ref_viz_pad], axis=1)

    # offset ref x coordinates by cond width
    x_offset = cond_viz_pad.shape[1]
    num_corr = min(len(cond_proj), len(ref_proj))
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 256, size=(num_corr, 3), dtype=np.int64)
    for i in range(num_corr):
        p_c = cond_proj[i]
        p_r = ref_proj[i]
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        pt1 = (int(p_c[0]), int(p_c[1]))
        pt2 = (int(p_r[0] + x_offset), int(p_r[1]))
        cv2.line(concat_viz, pt1, pt2, color=tuple(color), thickness=1, lineType=cv2.LINE_AA)

    imageio.imwrite(eval_dir / f"corres_{reference_idx:03d}.png", concat_viz)
    print(f"[evaluate_3d_corres] Saved overlays to {eval_dir}")

    return {
        "cond_proj": cond_proj,
        "ref_proj": ref_proj,
    }


def get_3D_correspondences(gen_3d, reference, reference_idx=0, out_dir=None, min_vis_score=0.2):
    def _to_uint8_img(tensor_img):
        if torch.is_tensor(tensor_img):
            arr = tensor_img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr
        return np.asarray(tensor_img, dtype=np.uint8)

    def _resize_intrinsics(K, src_hw, dst_hw):
        if src_hw == dst_hw:
            return K
        scale_x = dst_hw[1] / float(src_hw[1])
        scale_y = dst_hw[0] / float(src_hw[0])
        K = K.copy()
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y
        return K

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

    # Build a 2-frame stack and run predict_tracks for correspondences
    device = ref_img.device if torch.is_tensor(ref_img) else torch.device("cpu")
    cond_img_t = torch.from_numpy(cond_img).to(device=device, dtype=torch.float32)
    ref_img_t = ref_images[reference_idx]


    imgs_stack = torch.stack([cond_img_t, ref_img_t], dim=0)
    masks_stack = None

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
        torch.cuda.empty_cache()
    
    visualize_tracks_on_images(imgs_stack[None], torch.from_numpy(pred_tracks[None]), out_dir=f"{out_dir}/track_raw")
    vis_mask = pred_vis_scores > min_vis_score
    visualize_tracks_on_images(imgs_stack[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(vis_mask[None]), out_dir=f"{out_dir}/track_vis")
    vis_mask = pred_vis_scores > min_vis_score
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
    ref_depth_vals = ref_depth_vals.cpu().numpy()

    ref_unc_vals = ref_depth_unc[
        np.clip(np.round(ref_pixels[:, 1]).astype(int), 0, ref_depth_unc.shape[0] - 1),
        np.clip(np.round(ref_pixels[:, 0]).astype(int), 0, ref_depth_unc.shape[1] - 1),
    ]
    
    valid_depth = (cond_depth_vals > 0) & (ref_depth_vals > 0)
    if not np.any(valid_depth):
        raise ValueError("[get_3D_correspondences] No valid depth correspondences.")

    cond_pixels_orig = cond_pixels_orig[valid_depth]
    cond_pixels = cond_pixels[valid_depth]
    ref_pixels = ref_pixels[valid_depth]
    cond_depth_vals = cond_depth_vals[valid_depth]
    ref_depth_vals = ref_depth_vals[valid_depth]
    ref_unc_vals = ref_unc_vals[valid_depth]

    cond_world = _pixels_to_world(cond_pixels_orig, cond_depth_vals, cond_intr, cond_extr)
    ref_world = _pixels_to_world(ref_pixels, ref_depth_vals, ref_intr, ref_extr)

    print(f"[get_3D_correspondences] Found {len(cond_world)} 3D correspondences with predict_tracks.")

    corres = {
        "condition_points_world": cond_world,
        "condition_pixels": cond_pixels_orig,
        "reference_points_world": ref_world,
        "reference_pixels": ref_pixels,
        "reference_uncertainty": ref_unc_vals,
    }

    evaluate_3d_corres(corres, gen_3d, reference, reference_idx=0, out_dir=f"{out_dir}/eval")
    return corres

def eval_aligned_3D_model(cond_pts, ref_pts, aligned_pose, references, reference_idx=0, out_dir=None):
    if out_dir is None:
        print("[eval_aligned_3D_model] No output directory specified, skipping visualization.")
        return
    
    R, t, s = (
        aligned_pose["rotation"],
        aligned_pose["translation"],
        aligned_pose["scale"],
    )
    ref_imgs = references.get("images", None)
    ref_img_t = ref_imgs[reference_idx]
    ref_img = ref_img_t.detach().cpu().numpy()
    if ref_img.shape[0] == 3:
        ref_img = np.transpose(ref_img, (1, 2, 0))
    ref_img = np.clip(ref_img * 255.0, 0, 255).astype(np.uint8)
    ref_intr = np.asarray(references["intrinsics"][reference_idx], dtype=np.float64)
    ref_extr = np.asarray(references["extrinsics"][reference_idx], dtype=np.float64)
    if ref_extr.shape[0] == 4:
        ref_extr = ref_extr[:3]

    aligned_pts = (cond_pts @ R.T) * s + t
    proj, _ = project_3D_points_np(aligned_pts, ref_extr[None], ref_intr[None])
    proj = proj[0].astype(int)

    ref_viz = ref_img.copy()
    if ref_viz.ndim == 3 and ref_viz.shape[0] == 3:
        ref_viz = np.transpose(ref_viz, (1, 2, 0))
    if ref_viz.ndim == 2:
        ref_viz = np.repeat(ref_viz[..., None], 3, axis=-1)
    ref_viz = np.ascontiguousarray(ref_viz.astype(np.uint8))
    for p in proj:
        cv2.circle(ref_viz, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)


    align_dir = Path(out_dir)
    align_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(align_dir / "transform.png", ref_viz)
    save_point_cloud_to_ply(aligned_pts, str(align_dir / "cond_aligned.ply"))
    save_point_cloud_to_ply(cond_pts, str(align_dir / "cond.ply"))
    save_point_cloud_to_ply(ref_pts, str(align_dir / "ref.ply"))




def align_3D_model_with_images(corres, gen_3d, references, reference_idx, out_dir=None, iters=200):
    if corres is None:
        print("[align_3D_model_with_images] No correspondences provided.")
        return None

    cond_pts = corres.get("condition_points_world", None)
    ref_pts = corres.get("reference_points_world", None)
    ref_unc = corres.get("reference_uncertainty", None)
    if cond_pts is None or ref_pts is None or len(cond_pts) < 3:
        assert False, "[align_3D_model_with_images] Insufficient 3D correspondences for alignment."
        return None

    def _umeyama_alignment(src, dst, weights=None):
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        if weights is None:
            weights = np.ones(src.shape[0], dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != src.shape[0]:
            weights = np.ones(src.shape[0], dtype=np.float64)
        weights = np.maximum(weights, 1e-8)
        weights = weights / weights.sum()

        mu_src = (weights[:, None] * src).sum(axis=0)
        mu_dst = (weights[:, None] * dst).sum(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst
        cov = (dst_c * weights[:, None]).T @ src_c
        U, S, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = U @ Vt
        var_src = np.sum(weights * np.sum(src_c ** 2, axis=1))
        scale = np.sum(S) / (var_src + 1e-8)
        t = mu_dst - scale * (R @ mu_src)
        return R.astype(np.float32), t.astype(np.float32), float(scale)

    weights = None
    if ref_unc is not None:
        weights = 1.0 / (np.asarray(ref_unc, dtype=np.float64) + 1e-6)
        weights = np.maximum(weights, 1e-6)

    R, t, s = _umeyama_alignment(cond_pts, ref_pts, weights)

    aligned_pose = {"rotation": R, "translation": t, "scale": s}
    import json
    align_dir = Path(out_dir)
    align_dir.mkdir(parents=True, exist_ok=True)
    with open(align_dir / "transform.json", "w") as f:
        json.dump(
            {
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "scale": s,
            },
            f,
            indent=2,
        )
    print(f"[align_3D_model_with_images] Saved transform to {align_dir}")
    save_aligned_3D_model(gen_3d, aligned_pose, f"{out_dir}/eval")
    eval_aligned_3D_model(cond_pts, ref_pts, aligned_pose, references, reference_idx=reference_idx, out_dir=f"{out_dir}/eval")

    return aligned_pose
    



def save_aligned_3D_model(gen_3d, aligned_pose, output_path):
    mesh_path = gen_3d.get_mesh_path()

    os.makedirs(output_path, exist_ok=True)
    out_mesh = Path(output_path) / f"{Path(mesh_path).stem}_aligned.obj"
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = mesh.vertices.astype(np.float32)
        R = aligned_pose["rotation"]
        t = aligned_pose["translation"]
        s = aligned_pose["scale"]
        aligned_vertices = (vertices @ R.T) * s + t
        mesh.vertices = aligned_vertices
        mesh.export(out_mesh)
        print(f"Saved aligned mesh to {out_mesh}")
    except Exception as e:
        print(f"Failed to save aligned mesh: {e}")

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Get image paths and preprocess them
    image_dir = Path(os.path.join(args.scene_dir, "images"))
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    image_path_list = sorted(image_path_list)
    image_path_list = image_path_list[:args.max_frames]
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    # check the frame index range
    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    print(f"Processing images in {image_dir} with the list  {base_image_path_list}")

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518

    img_load_resolution = Image.open(image_path_list[0]).size[0]

    images, original_coords, image_masks, depth_prior = load_and_preprocess_images_square(image_path_list, args.instance_id, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    image_masks = image_masks.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    if 0:
        # Run VGGT to estimate camera and depth
        # Run with 518x518 images
    # Run VGGT for camera and depth estimation
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model.eval()
        model = model.to(device)
        print(f"Model loaded")        
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    else:
        intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", "0000.pkl"))
        depth_conf = np.ones_like(depth_prior)
        with torch.cuda.amp.autocast(dtype=dtype) and torch.no_grad():
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
            )
            torch.cuda.empty_cache()
        visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>= pred_vis_scores.min(), out_dir=f"{args.output_dir}/track_raw")            
        track_mask = pred_vis_scores > args.vis_thresh
        visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(track_mask[None]), out_dir=f"{args.output_dir}/track_filter_vis_thresh")
        extrinsic = estimate_extrinsic(depth_prior, intrinsic, pred_tracks, track_mask)
        
        intrinsic = np.tile(intrinsic[None, :, :], (len(images), 1, 1))
        points_3d = unproject_depth_map_to_point_map(depth_prior[..., None], extrinsic, intrinsic)
        vggt_fixed_resolution = img_load_resolution


        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera


        with torch.cuda.amp.autocast(dtype=dtype) and torch.no_grad():
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                image_masks=image_masks,
                conf=depth_conf,
                points_3d=points_3d,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
                complete_non_vis=False,
            )
            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        # track_mask = pred_vis_scores > args.vis_thresh
        # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(track_mask[None]), out_dir=f"{args.output_dir}/track_filter_vis_thresh")            

        track_mask = verify_tracks_by_geometry(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
        )
        visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(track_mask[None]), out_dir=f"{args.output_dir}/track_filter_max_proj_err")            
        
        # step: Prep valid correspondences and drop 3D points without surviving tracks
        track_mask, points_3d, keep_pts = prep_valid_correspondences(
            points_3d, track_mask, args.min_inlier_per_frame, args.min_inlier_per_track
        )
        pred_tracks = pred_tracks[:, keep_pts]
        if points_rgb is not None:
            points_rgb = points_rgb[keep_pts]
        visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(track_mask[None]), out_dir=f"{args.output_dir}/track_filter_frame_track_inlier")

        # step: optimize 3D points and camera poses using sparse reprojection and point-to-ray losses
        extrinsic, points_3d, uncertainties = optimize_poses_with_losses(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            depth_prior,
            track_mask,
            camera_type=args.camera_type,
            iters=5,
            lr=1e-3,
        )
        gen_3d = GEN_3D(f"{args.scene_dir}/align_mesh_image/0000")
        image_info = {
            "image_paths": image_path_list,
            "images": images,
            "image_masks": image_masks,
            "depth_priors": depth_prior,
            "intrinsics": intrinsic,
            "extrinsics": extrinsic,
            "uncertainties": uncertainties,

        }
        corres = get_3D_correspondences(gen_3d, image_info, reference_idx=0, out_dir=f"{args.output_dir}/3D_corres/", min_vis_score=args.vis_thresh)
        
        aligned_pose = align_3D_model_with_images(
            corres, gen_3d, image_info, reference_idx=0, out_dir=f"{args.output_dir}/aligned/"
        )

        # step: convert to pycolmap reconstruction
        reconstruction, track_masks = build_reconstruction_from_tracks(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            track_mask,
            shared_camera,
            args.camera_type,
            points_rgb=points_rgb,
        )

        reconstruction, track_masks = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            min_inlier_per_frame=args.min_inlier_per_frame,
            min_inlier_per_track=args.min_inlier_per_track,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
            images=images,
            out_dir=args.output_dir,
        )        

        # if reconstruction is None:
        #     raise ValueError("No reconstruction can be built with BA")

        # # Bundle Adjustment
        # ba_options = pycolmap.BundleAdjustmentOptions()
        # ba_options.refine_focal_length = not args.use_calibrated_intrinsic
        # ba_options.refine_principal_point = not args.use_calibrated_intrinsic
        # pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution

            



    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    ba_out_dir = Path(args.output_dir) / "vggt_ba" / "sparse"
    print(f"Saving ba reconstruction to {ba_out_dir}")
    os.makedirs(ba_out_dir, exist_ok=True)
    save_point_cloud_with_conf(points_3d, points_rgb, uncertainties["points3d"], ba_out_dir / "points.ply")
    save_depth_prior_with_uncertainty(depth_prior, uncertainties["depth_prior"], Path(args.output_dir) / "vggt_ba" / "depth_conf")


    reconstruction.write(ba_out_dir)
    if reconstruction is not None:
        print(
            f"Reconstruction statistics:\n{reconstruction.summary()}"
            + f"\n\tnum_input_images = {len(images)}"
        )
    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


def save_point_cloud_with_conf(points_3d, points_rgb, uncertainties, ply_path):
    """Save point cloud; color by uncertainty if provided, else by rgb."""

    conf = uncertainties.astype(np.float64)
    conf_norm = conf / (conf.max() + 1e-8)
    conf_colors = np.stack(
        [
            conf_norm * 255.0,          # green channel high -> high confidence
            (1.0 - conf_norm) * 255.0,  # red channel high -> low confidence
            np.zeros_like(conf_norm),
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    if len(conf_colors) != len(points_3d):
        conf_colors = conf_colors[: len(points_3d)]
    trimesh.PointCloud(points_3d, colors=conf_colors).export(ply_path)


def save_depth_prior_with_uncertainty(depth, depth_unc, out_dir):
    """Save per-pixel depth uncertainty maps as color PNGs (green=high confidence, red=low)."""
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

if __name__ == "__main__":
    args = parse_args()
    demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
