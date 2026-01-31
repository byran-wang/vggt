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
import pickle
import shutil
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import json

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
# import pycolmap
import cv2
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square, load_intrinsics, GEN_3D
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
from vggt.utils.visual_track import visualize_tracks_on_images
# from vggt.dependency.sfm_prepair import prepare_features, prepare_matches, prepare_pairs
from vggt.dependency.projection import project_3D_points_np

import sys
sys.path.append("third_party/Hierarchical-Localization/")
# from hloc.reconstruction import main as hloc_reconstruction_main
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import save_point_cloud_to_ply
from utils_simba.render import diff_renderer, projection_matrix_from_intrinsics
from utils_simba.depth import save_depth

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_sfm", action="store_true", default=False, help="Use SfM for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--dataset_type", type=str, default="ZED", help="Dataset type for SfM preprocessing")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument("--min_depth_pixels", type=int, default=500, help="Minimum valid depth pixels to accept a frame")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--use_calibrated_intrinsic", action="store_true", default=False, help="Use calibrated intrinsic for reconstruction")
    parser.add_argument("--min_inlier_per_frame", type=int, default=10, help="Minimum inliers per frame for BA")
    parser.add_argument("--min_inlier_per_track", type=int, default=4, help="Minimum inliers per track for BA")
    parser.add_argument("--min_frame_num", type=int, default=0, help="Minimum number of frames to process")
    parser.add_argument("--max_frame_num", type=int, default=50, help="Maximum number of frames to process")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval for processing")
    parser.add_argument("--min_PnP_inlier_num", type=int, default=200, help="Minimum inliers for registration")
    parser.add_argument("--instance_id", type=int, default=0, help="Instance ID for image preprocessing")
    parser.add_argument("--cond_index", type=int, default=0, help="Conditioning frame index for tracking")
    parser.add_argument("--cond_index_raw", type=int, default=0, help="Conditioning frame index for tracking")
    parser.add_argument("--kf_rot_thresh", type=float, default=3.0, help="Keyframe rotation threshold (degrees)")
    parser.add_argument("--kf_trans_thresh", type=float, default=0.01, help="Keyframe translation threshold (units)")
    parser.add_argument("--kf_depth_thresh", type=float, default=500, help="Keyframe depth change threshold (units)")
    parser.add_argument("--kf_inlier_thresh", type=int, default=10, help="Keyframe inlier count threshold")
    parser.add_argument("--min_track_number", type=int, default=4, help="Minimum track number for 3D point uncertainty; points with fewer tracks get high uncertainty")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")

def load_images_and_intrinsics(args, device):
    image_dir, image_path_list = get_image_list(args)
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    # check the frame index range
    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    print(f"Processing images in {image_dir} with the list  {base_image_path_list}")

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518

    img_load_resolution = Image.open(image_path_list[0]).size[0]

    images, original_coords, image_masks, depth_prior = load_and_preprocess_images_square(
        image_path_list,
        args,
        target_size=img_load_resolution,
        out_dir=f"{args.output_dir}/data_processed",
    )
    gen_3d = GEN_3D(f"{args.scene_dir}/align_mesh_image/{args.cond_index_raw:04d}")
    save_input_data(images, image_masks, depth_prior, gen_3d, image_path_list, f"{args.output_dir}/results/")

    images = images.to(device)
    original_coords = original_coords.to(device)
    image_masks = image_masks.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", f"{args.cond_index_raw:04d}.pkl"))
    intrinsic = adjust_intrinsic_for_new_image_size(intrinsic, original_coords, frame_idx=args.cond_index)

    depth_conf = np.ones_like(depth_prior)
    return (
        image_dir,
        image_path_list,
        base_image_path_list,
        images,
        original_coords,
        image_masks,
        depth_prior,
        intrinsic,
        depth_conf,
        vggt_fixed_resolution,
        img_load_resolution,
        gen_3d,
    )

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


def remove_duplicate_tracks(existing_tracks, new_tracks, new_track_mask, new_points_3d, new_points_rgb,
                            ref_frame_idx, dist_thresh=3.0):
    """
    Remove new track points that are too close to existing tracks at the reference frame.

    Args:
        existing_tracks: Existing tracks array of shape [S, N_existing, 2]
        new_tracks: New tracks array of shape [S, N_new, 2]
        new_track_mask: Visibility mask for new tracks [S, N_new]
        new_points_3d: 3D points for new tracks [N_new, 3]
        new_points_rgb: RGB colors for new tracks [N_new, 3] or None
        ref_frame_idx: Reference frame index to compare 2D positions
        dist_thresh: Distance threshold in pixels to consider as duplicate

    Returns:
        Filtered new_tracks, new_track_mask, new_points_3d, new_points_rgb
    """
    if new_tracks.shape[1] == 0:
        return new_tracks, new_track_mask, new_points_3d, new_points_rgb

    # Get 2D positions at reference frame
    existing_pts_2d = existing_tracks[ref_frame_idx]  # [N_existing, 2]
    new_pts_2d = new_tracks[ref_frame_idx]  # [N_new, 2]

    # Compute pairwise distances between new and existing points
    # Using broadcasting: [N_new, 1, 2] - [1, N_existing, 2] -> [N_new, N_existing, 2]
    diff = new_pts_2d[:, None, :] - existing_pts_2d[None, :, :]
    distances = np.linalg.norm(diff, axis=2)  # [N_new, N_existing]

    # Find minimum distance to any existing track for each new track
    min_distances = distances.min(axis=1)  # [N_new]

    # Keep only new tracks that are far enough from all existing tracks
    keep_mask = min_distances > dist_thresh

    num_removed = (~keep_mask).sum()
    if num_removed > 0:
        print(f"[remove_duplicate_tracks] Removed {num_removed} duplicate tracks (dist_thresh={dist_thresh}px)")

    new_tracks = new_tracks[:, keep_mask]
    new_track_mask = new_track_mask[:, keep_mask]
    new_points_3d = new_points_3d[keep_mask]
    if new_points_rgb is not None:
        new_points_rgb = new_points_rgb[keep_mask]

    return new_tracks, new_track_mask, new_points_3d, new_points_rgb


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
):
    """Propagate uncertainties for extrinsics, 3D points, and depth priors.

    Only cameras that pass the threshold criteria are used for uncertainty computation:
    - depth_thresh: minimum number of valid depth pixels
    - track_inlier_thresh: minimum number of track inliers
    - rot_thresh: minimum rotation delta (degrees) from reference camera
    - trans_thresh: minimum translation delta from reference camera
    - min_track_number: minimum keyframe observations for a 3D point; points with
      fewer observations get high uncertainty

    The function first filters valid frames (enough track inliers + valid depth),
    then selects keyframes from valid frames based on rotation/translation thresholds.
    Only keyframes are used for depth uncertainty computation.
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
    """
    Perform bundle adjustment on keyframes to jointly optimize 3D points and camera poses.

    This function optimizes the merged points_3d and extrinsics for all keyframes using
    reprojection and point-to-ray losses, similar to traditional bundle adjustment.

    Args:
        image_info: Dictionary containing:
            - "keyframe": Boolean array indicating which frames are keyframes
            - "points_3d": 3D points array [N, 3]
            - "pred_tracks": Track predictions [S, N, 2]
            - "track_mask": Visibility mask [S, N]
            - "extrinsics": Camera extrinsics [S, 4, 4]
            - "intrinsics": Camera intrinsics [S, 3, 3]
            - "depth_priors": Depth maps
        ref_frame_idx: Reference frame index (pose is fixed during optimization)
        iters: Number of optimization iterations
        lr: Learning rate for optimizer
        rep_loss_thresh: Early-stop threshold for reprojection loss (sum of smooth L1)

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
        ) / (valid.sum() * rep_raw.shape[-1] + 1e-8)       # divide by (#valid * 2)

        # Point-to-ray consistency loss
        uv1 = torch.cat([tracks_t, torch.ones((K, P, 1), device=device, dtype=dtype)], dim=-1)  # K,P,3
        rays = torch.bmm(inv_intr_t, uv1.transpose(1, 2))  # K,3,P
        rays = torch.nn.functional.normalize(rays, dim=1, eps=1e-6)
        cross = torch.cross(cam_pts, rays, dim=1)
        ray_raw = torch.linalg.norm(cross, dim=1)          # K,P  (distance to ray)
        valid = mask_t.to(ray_raw.dtype)                   # K,P

        ray_loss = torch.nn.functional.smooth_l1_loss(
            ray_raw * valid,
            torch.zeros_like(ray_raw),
            reduction="sum"
        ) / (valid.sum() + 1e-8)

        # Depth consistency loss: compare z-coordinate in camera space with depth prior
        if kf_depth_priors is not None:
            # Get predicted depth (z-coordinate in camera space)
            z_pred = cam_pts[:, 2, :]  # K, P

            # Sample depth prior at track locations using grid_sample
            H, W = kf_depth_priors.shape[-2:]
            # Normalize track coordinates to [-1, 1] for grid_sample
            grid_x = 2.0 * tracks_t[..., 0] / (W - 1) - 1.0  # K, P
            grid_y = 2.0 * tracks_t[..., 1] / (H - 1) - 1.0  # K, P
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # K, P, 1, 2

            # Sample depth at track locations
            depth_prior_sampled = torch.nn.functional.grid_sample(
                kf_depth_priors.unsqueeze(1),  # K, 1, H, W
                grid,  # K, P, 1, 2
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze(1).squeeze(-1)  # K, P

            # Only compute loss where depth is valid and track is visible
            valid_depth_mask = (depth_prior_sampled > 0) & mask_t
            if valid_depth_mask.any():
                depth_raw = z_pred - depth_prior_sampled       # K,P
                valid = valid_depth_mask.to(depth_raw.dtype)

                depth_loss = torch.nn.functional.smooth_l1_loss(
                    depth_raw * valid,
                    torch.zeros_like(depth_raw),
                    reduction="sum"
                ) / (valid.sum() + 1e-8)
            else:
                depth_loss = torch.zeros((), device=device, dtype=dtype)
        ray_loss *= 10
        depth_loss *= 1000
        loss = rep_loss + ray_loss + depth_loss
        loss.backward()
        print(f"[bundle_adjust_keyframes] Iter {it}: rep_loss={rep_loss.item():.4f}, "
              f"ray_loss={ray_loss.item():.4f}, depth_loss={depth_loss.item():.4f}, total_loss={loss.item():.4f}")

        if rep_loss_thresh is not None and rep_loss.item() < rep_loss_thresh:
            print(
                f"[bundle_adjust_keyframes] Early stop at iter {it}: rep_loss {rep_loss.item():.6f} "
                f"< {rep_loss_thresh}"
            )
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
    # Put optimized keyframe extrinsics back into the full array
    # Handle both 4x4 and 3x4 extrinsic formats
    extr_shape = image_info["extrinsics"].shape[-2:]
    for i, kf_idx in enumerate(keyframe_indices):
        if extr_shape == (4, 4):
            image_info["extrinsics"][kf_idx] = optimized_extrinsics[i]
        else:
            # 3x4 format: only copy the top 3 rows
            image_info["extrinsics"][kf_idx] = optimized_extrinsics[i, :3, :]

    image_info["points_3d"] = optimized_points_3d

    # Compute final reprojection error for logging
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

def estimate_extrinsic(depth_map, extrinsics, intrinsic, tracks, track_mask, ref_index=0, ransac_reproj_threshold = 8.0):
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
        # print(
        #     f"[estimate_extrinsic] Warning: PnP failed for frame {frame_idx}, "
        #     f"carrying pose from reference frame {ref_index}."
        # )

    return extrinsics


def verify_tracks_by_geometry(points3d, extrinsics, intrinsics, tracks, ref_index, masks=None, max_reproj_error=None):
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
    
    visualize_tracks_on_images(imgs_stack[None], torch.from_numpy(pred_tracks[None]), out_dir=f"{out_dir}/track_raw")
    vis_mask = pred_vis_scores > min_vis_score
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

    # Optional refinement with mesh-depth alignment
    try:
        mesh_path = gen_3d.get_mesh_path()
        if mesh_path is not None and os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, force="mesh")
            verts = mesh.vertices.astype(np.float32)
            if mesh.visual.vertex_colors is not None:
                color_np = (mesh.visual.vertex_colors[:, :3] / 255.0).astype(np.float32)
            else:
                color_np = np.ones_like(verts, dtype=np.float32) * 0.5

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            faces = torch.tensor(mesh.faces.astype(np.int32), device=device)

            # Prepare reference data
            ref_intr = torch.from_numpy(np.asarray(references["intrinsics"][reference_idx], dtype=np.float32)).to(device)
            ref_extr = torch.from_numpy(np.asarray(references["extrinsics"][reference_idx], dtype=np.float32)).to(device)
            if ref_extr.shape[0] == 4:
                ref_extr = ref_extr[:3]
            ref_depth = references["depth_priors"][reference_idx]
            ref_depth = torch.from_numpy(ref_depth if isinstance(ref_depth, np.ndarray) else ref_depth.cpu().numpy()).to(device)
            ref_depth = ref_depth.squeeze()
            H_r, W_r = ref_depth.shape[-2], ref_depth.shape[-1]
            ref_unc_map = None
            if references.get("uncertainties") is not None:
                unc_all = references["uncertainties"].get("depth_prior", None)
                if unc_all is not None and len(unc_all) > reference_idx:
                    unc_map = unc_all[reference_idx]
                    unc_map = torch.from_numpy(unc_map if isinstance(unc_map, np.ndarray) else unc_map.cpu().numpy())
                    ref_unc_map = unc_map.to(device).squeeze()

            verts_t = torch.from_numpy(verts).to(device)
            color_t = torch.from_numpy(color_np).to(device)
            R_t = torch.from_numpy(R.astype(np.float32)).to(device)
            t_t = torch.from_numpy(t.astype(np.float32)).to(device)
            s_t = torch.tensor([s], device=device, dtype=torch.float32)

            cond_pts_t = torch.from_numpy(cond_pts.astype(np.float32)).to(device)
            ref_pts_t = torch.from_numpy(ref_pts.astype(np.float32)).to(device)
            w_corr = None
            if weights is not None:
                w_corr = torch.from_numpy(weights.astype(np.float32)).to(device)

            extr4 = torch.eye(4, device=device, dtype=torch.float32)
            extr4[:3, :] = ref_extr
            proj_mat = projection_matrix_from_intrinsics(ref_intr.cpu().numpy(), height=H_r, width=W_r, znear=0.1, zfar=100.0)
            proj_t = torch.from_numpy(proj_mat).to(device)
            glctx = dr.RasterizeGLContext()

            optim = torch.optim.Adam([R_t, t_t, s_t], lr=1e-2)
            verts_sub = verts_t
            if verts_sub.shape[0] > 10000:
                idx = torch.randperm(verts_sub.shape[0], device=device)[:10000]
                verts_sub = verts_sub[idx]
                color_sub = color_t[idx]
            else:
                color_sub = color_t

            for _ in range(50):
                optim.zero_grad(set_to_none=True)
                # transform cond points
                cond_aligned = (cond_pts_t @ R_t.T) * s_t + t_t
                corr_res = cond_aligned - ref_pts_t
                if w_corr is not None:
                    corr_loss = (w_corr[:, None] * corr_res.pow(2)).mean()
                else:
                    corr_loss = corr_res.pow(2).mean()

                verts_aligned = (verts_sub @ R_t.T) * s_t + t_t
                render_color, depth_render = diff_renderer(
                    verts_aligned[None], faces, color_sub[None], proj_t, extr4, (H_r, W_r), glctx
                )
                depth_render = depth_render.squeeze()
                depth_mask = (depth_render > 0) & (ref_depth > 0)
                if depth_mask.any():
                    depth_pred = depth_render[depth_mask]
                    depth_gt = ref_depth[depth_mask]
                    if ref_unc_map is not None:
                        unc_mask = ref_unc_map[depth_mask]
                        w_depth = 1.0 / (unc_mask + 1e-6)
                        depth_loss = (w_depth * (depth_pred - depth_gt).abs()).mean()
                    else:
                        depth_loss = torch.nn.functional.smooth_l1_loss(depth_pred, depth_gt)
                else:
                    depth_loss = torch.tensor(0.0, device=device)

                loss = corr_loss + 0.1 * depth_loss
                loss.backward()
                optim.step()

            R = R_t.cpu().numpy()
            t = t_t.cpu().numpy()
            s = float(s_t.item())
            print("[align_3D_model_with_images] Refined transform using mesh-depth alignment.")
    except Exception as e:
        print(f"[align_3D_model_with_images] Mesh-depth refinement failed: {e}")

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
    gen_3d.save_aligned_pose(aligned_pose)
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

def find_next_frame(image_info):
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

def check_frame_invalid(image_info, frame_idx, min_inlier_per_frame, min_depth_pixels):
    track_mask = image_info.get("track_mask")
    depth_priors = image_info.get("depth_priors")

    track_mask = np.asarray(track_mask)
    if frame_idx >= track_mask.shape[0]:
        print(f"[check_frame_invalid] Frame index {frame_idx} out of bounds.")
        return True
    track_inliers = int(np.count_nonzero(track_mask[frame_idx]))

    depth_map = depth_priors[frame_idx]
    if torch.is_tensor(depth_map):
        depth_map = depth_map.detach().cpu().numpy()
    depth_valid = int(np.count_nonzero(np.asarray(depth_map) > 0))

    if track_inliers < min_inlier_per_frame:
        print(f"[check_frame_invalid] Frame {frame_idx} invalid due to insufficient track inliers: {track_inliers} < {min_inlier_per_frame}")
        return True
    if depth_valid < min_depth_pixels:
        print(f"[check_frame_invalid] Frame {frame_idx} invalid due to insufficient depth pixels: {depth_valid} < {min_depth_pixels}")
        return True
    return False

def save_results(image_info, gen_3d, out_dir):
    """Persist key reconstruction artifacts for later reuse/inspection."""
    os.makedirs(out_dir, exist_ok=True)

    def _to_cpu_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    points_conf_color = get_points_uncertainty_colors(points_3d=image_info.get("points_3d"), uncertainties=image_info.get("uncertainties")['points3d'])

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

    save_point_cloud_with_conf(image_info.get("points_3d"), image_info.get("points_rgb"), image_info.get("uncertainties")['points3d'], Path(out_dir) / "points.ply")
    save_depth_prior_with_uncertainty(image_info.get("depth_priors"), image_info.get("uncertainties")['depth_prior'], Path(out_dir) / "depth_conf")            
    save_aligned_3D_model(gen_3d,  gen_3d.get_aligned_pose(), out_dir)
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
    """Save preprocessed inputs to disk for inspection/debugging."""
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

def eval_reprojection(image_info, frame_idx, intr_np, pts_np, tracks_np, mask_np, R_final, t_final, out_dir):
    """Overlay reprojection error vectors on the raw image for a frame."""
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

def adjust_intrinsic_for_new_image_size(intrinsic, original_coords, frame_idx=0):
    """Rescale/shift intrinsic to the padded+resized image frame using stored original coords."""
    base_intr = np.asarray(intrinsic, dtype=np.float32)
    if base_intr.shape != (3, 3):
        return base_intr

    if original_coords is None:
        return base_intr

    if torch.is_tensor(original_coords):
        orig_np = original_coords.detach().cpu().numpy()
    else:
        orig_np = np.asarray(original_coords)

    if orig_np.ndim < 2 or orig_np.shape[1] < 6:
        return base_intr

    idx = int(frame_idx)
    if idx >= orig_np.shape[0]:
        idx = 0

    x1, y1, x2, y2, width, height = orig_np[idx]
    if width <= 0 or height <= 0:
        return base_intr

    scale_x = (x2 - x1) / float(width)
    scale_y = (y2 - y1) / float(height)

    adjusted = base_intr.copy()
    adjusted[0, 0] *= scale_x
    adjusted[1, 1] *= scale_y
    adjusted[0, 2] = adjusted[0, 2] * scale_x + x1
    adjusted[1, 2] = adjusted[1, 2] * scale_y + y1
    return adjusted

def register_new_frame(image_info, gen_3d, frame_idx, args, out_dir, iters=100, depth_weight=0):
    """Optimize only the pose of frame `frame_idx` using reprojection + mesh-depth consistency."""
    points_3d = image_info.get("points_3d")
    extrinsics = image_info.get("extrinsics")
    intrinsics = image_info.get("intrinsics")
    pred_tracks = image_info.get("pred_tracks")
    track_mask = image_info.get("track_mask")
    depth_priors = image_info.get("depth_priors")
    aligned_pose = gen_3d.get_aligned_pose() if hasattr(gen_3d, "get_aligned_pose") else None

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
    else:
        try:
            intr_np = np.asarray(intrinsics[frame_idx], dtype=np.float32)
            extr = np.asarray(extrinsics[frame_idx], dtype=np.float32)

            pts_np = np.asarray(points_3d, dtype=np.float32)
            tracks_np = np.asarray(pred_tracks[frame_idx], dtype=np.float32)
            mask_np = np.asarray(track_mask[frame_idx]).astype(bool)

            if mask_np.shape[0] != tracks_np.shape[0] or mask_np.shape[0] != pts_np.shape[0]:
                print("[register_new_frame] Track/point mismatch; skipping refinement.")
                return

            pts_sel = pts_np[mask_np]
            tracks_sel = tracks_np[mask_np]
            if len(pts_sel) < 6:
                print(f"[register_new_frame] Not enough correspondences for PnP ({len(pts_sel)} < 6); skipping refinement.")
                return

            dist = np.zeros((4, 1), dtype=np.float32)
            rvec_init, _ = cv2.Rodrigues(extr[:3, :3])
            tvec_init = extr[:3, 3:4]

            success, rvec_ref, tvec_ref, inliers = cv2.solvePnPRansac(
                pts_sel,
                tracks_sel,
                intr_np,
                dist,
                rvec_init.astype(np.float32),
                tvec_init.astype(np.float32),
                useExtrinsicGuess=True,
                iterationsCount=iters,
                reprojectionError=float(args.max_reproj_error) if hasattr(args, "max_reproj_error") else 8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                print(f"[register_new_frame] RANSAC PnP failed for frame {frame_idx}.")
                return

            R_final, _ = cv2.Rodrigues(rvec_ref)
            t_final = tvec_ref.reshape(-1)

            extrinsics[frame_idx, :3, :3] = R_final
            extrinsics[frame_idx, :3, 3] = t_final
            image_info["extrinsics"] = extrinsics

            # Build inlier mask in the original track space for visualization.
            inlier_mask = mask_np.copy()
            if inliers is not None:
                inlier_mask[:] = False
                sel_indices = np.where(mask_np)[0]
                inlier_indices = sel_indices[inliers.flatten()]
                inlier_mask[inlier_indices] = True

            print(f"[register_new_frame] Optimized frame {frame_idx} pose with RANSAC PnP. Inliers: {inlier_mask.sum()}/{len(mask_np)}")

        except Exception as e:
            print(f"[register_new_frame] Pose refinement failed: {e}")

def check_key_frame(image_info, frame_idx, rot_thresh, trans_thresh, depth_thresh, frame_inliner_thresh):
    """Heuristically decide if frame_idx should become a keyframe based on validity + pose delta."""
    registered = image_info.get("registered")
    extrinsics = image_info.get("extrinsics")
    keyframes = image_info.get("keyframe")
    depth_priors = image_info.get("depth_priors")
    track_mask = image_info.get("track_mask")

    registered = np.asarray(registered).astype(bool)
    if not registered[frame_idx]:
        print(f"[check_key_frame] Frame {frame_idx} is not registered; cannot be keyframe.")
        return False

    # Basic validity checks before pose deltas
    if depth_priors is not None:
        depth_map = depth_priors[frame_idx]
        if torch.is_tensor(depth_map):
            depth_map = depth_map.detach().cpu().numpy()
        valid_depth = int(np.count_nonzero(np.asarray(depth_map) > 0))
        if valid_depth < depth_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected as keyframe due to insufficient depth pixels ({valid_depth} < {depth_thresh}).")
            return False

    if track_mask is not None:
        tm = np.asarray(track_mask)
        if frame_idx < tm.shape[0]:
            if int(np.count_nonzero(tm[frame_idx])) < frame_inliner_thresh:
                print(f"[check_key_frame] Frame {frame_idx} rejected as keyframe due to insufficient track inliers ({int(np.count_nonzero(tm[frame_idx]))} < {frame_inliner_thresh}).")
                return False

    if keyframes is None:
        return True  # no keyframes tracked yet

    keyframes = np.asarray(keyframes).astype(bool)
    past_keys = np.where(keyframes & registered & (np.arange(len(keyframes)) < frame_idx))[0]
    if len(past_keys) == 0:
        print(f"[check_key_frame] Frame {frame_idx} accepted as first keyframe.")
        return True  # first keyframe

    T_curr = extrinsics[frame_idx]
    R_curr, t_curr = T_curr[:3, :3], T_curr[:3, 3]

    for kf_idx in past_keys:
        T_prev = extrinsics[kf_idx]
        R_prev, t_prev = T_prev[:3, :3], T_prev[:3, 3]
        R_delta = R_curr @ R_prev.T
        angle = np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)))
        trans = np.linalg.norm(t_curr - t_prev)

        # Require the current frame to exceed both rotation and translation thresholds
        # with respect to every existing keyframe.
        if angle < rot_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected as keyframe due to insufficient rotation delta ({angle:.2f} < {rot_thresh}) with keyframe {kf_idx}.")
            return False

        if trans < trans_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected as keyframe due to insufficient translation delta ({trans:.3f} < {trans_thresh}) with keyframe {kf_idx}.")
            return False

    return True


def process_key_frame(image_info, frame_idx, args):
    """Predict fresh tracks from the new keyframe and update extrinsics/tracks for unregistered frames."""
    images = image_info.get("images")
    image_masks = image_info.get("image_masks")
    depth_priors = image_info.get("depth_priors")
    extrinsics = image_info.get("extrinsics").copy()
    intrinsics = image_info.get("intrinsics").copy()    
    pred_tracks = image_info.get("pred_tracks")
    track_mask = image_info.get("track_mask")
    points_3d = image_info.get("points_3d")
    points_rgb = image_info.get("points_rgb")
    registered = image_info.get("registered")
    invalid = image_info.get("invalid")

    needed = {
        "images": images,
        "image_masks": image_masks,
        "depth_priors": depth_priors,
        "points_3d": points_3d,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        print(f"[process_key_frame] Missing data {missing}; skipping keyframe processing.")
        return

    existing_pred_tracks = pred_tracks
    existing_track_mask = track_mask
    existing_points_3d = points_3d
    existing_points_rgb = points_rgb

    device = images.device if torch.is_tensor(images) else ("cuda" if torch.cuda.is_available() else "cpu")
    depth_conf = torch.ones_like(depth_priors)

    # Build a dense per-frame 3D map from the current depth priors for querying.
    depth_np = depth_priors.detach().cpu().numpy() if torch.is_tensor(depth_priors) else np.asarray(depth_priors)
    extr_np = extrinsics.detach().cpu().numpy() if torch.is_tensor(extrinsics) else np.asarray(extrinsics)
    intr_np = intrinsics.detach().cpu().numpy() if torch.is_tensor(intrinsics) else np.asarray(intrinsics)
    points_3d_dense = unproject_depth_map_to_point_map(depth_np[..., None], extr_np, intr_np)

    with torch.cuda.amp.autocast(dtype=torch.float16 if torch.cuda.is_available() else None), torch.no_grad():
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            image_masks=image_masks,
            conf=depth_conf,
            points_3d=points_3d_dense,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking,
            complete_non_vis=False,
            query_frame_indexes=[frame_idx],
        )

    track_mask = pred_vis_scores > args.vis_thresh

    # Estimate extrinsics for unregistered frames via PnP using the updated tracks
    extrinsics = estimate_extrinsic(depth_priors, extrinsics, intrinsics[0], pred_tracks, track_mask, ref_index=frame_idx, ransac_reproj_threshold=args.max_reproj_error)

    # Geometry-based pruning
    track_mask = verify_tracks_by_geometry(
        points_3d,
        extrinsics,
        intrinsics,
        pred_tracks,
        ref_index=frame_idx,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
    )

    track_mask_new, points_3d_new, keep_pts = prep_valid_correspondences(
        points_3d, track_mask, args.min_inlier_per_frame, args.min_inlier_per_track
    )
    pred_tracks_new = pred_tracks[:, keep_pts]
    points_rgb_new = points_rgb[keep_pts] if points_rgb is not None else None

    # Remove duplicate track points that are too close to existing tracks
    if existing_pred_tracks is not None and len(existing_pred_tracks) > 0:
        pred_tracks_new, track_mask_new, points_3d_new, points_rgb_new = remove_duplicate_tracks(
            existing_pred_tracks, pred_tracks_new, track_mask_new, points_3d_new, points_rgb_new,
            ref_frame_idx=frame_idx, dist_thresh=3.0
        )

    # Append new keyframe tracks/points to existing state instead of overwriting.
    if existing_pred_tracks is not None and existing_track_mask is not None and existing_points_3d is not None:
        image_info["pred_tracks"] = np.concatenate([existing_pred_tracks, pred_tracks_new], axis=1)
        image_info["track_mask"] = np.concatenate([existing_track_mask, track_mask_new], axis=1)
        image_info["points_3d"] = np.concatenate([existing_points_3d, points_3d_new], axis=0)
        if existing_points_rgb is not None and points_rgb_new is not None:
            image_info["points_rgb"] = np.concatenate([existing_points_rgb, points_rgb_new], axis=0)
        else:
            image_info["points_rgb"] = existing_points_rgb if existing_points_rgb is not None else points_rgb_new
    else:
        image_info["pred_tracks"] = pred_tracks_new
        image_info["track_mask"] = track_mask_new
        image_info["points_3d"] = points_3d_new
        image_info["points_rgb"] = points_rgb_new

    # Perform bundle adjustment on all keyframes to jointly optimize 3D points and camera poses
    # Use the first keyframe (cond_index) as reference to keep the coordinate frame consistent
    first_keyframe_idx = np.where(image_info["keyframe"])[0][0] if image_info["keyframe"].any() else args.cond_index
    bundle_adjust_keyframes(image_info, ref_frame_idx=first_keyframe_idx, lr=1e-3)

def get_image_list_ZED(args):
    image_dir = Path(os.path.join(args.scene_dir, "images"))
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    image_path_list = sorted(image_path_list)
    image_path_list = image_path_list[args.min_frame_num:args.max_frame_num:args.frame_interval]

    return image_dir, image_path_list

def get_image_list_HO3D(args):
    image_dir = Path(os.path.join(args.scene_dir, "rgb"))
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    image_path_list = sorted(image_path_list)
    image_path_list = image_path_list[args.min_frame_num:args.max_frame_num:args.frame_interval]

    return image_dir, image_path_list  

def get_image_list(args):
    if args.dataset_type == "ZED":
        return get_image_list_ZED(args)
    elif args.dataset_type == "HO3D":
        return get_image_list_HO3D(args)

    return None, []

# =============================================================================
# Helper functions for demo_fn
# =============================================================================

def setup_environment(args):
    """Setup device, dtype, output directory, and seed."""
    print("Arguments:", vars(args))

    results_dir = Path(args.output_dir) / "results"
    if results_dir.exists():
        results_folders = [entry for entry in results_dir.iterdir() if entry.is_dir()]
        if len(results_folders) > 20:
            print(f"Found {len(results_folders)} result folders in {results_dir}, skipping.")
            return None, None

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    return device, dtype


def predict_initial_tracks_wrapper(images, image_masks, args, dtype):
    """Predict initial tracks using VGGSfM tracker."""
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


def estimate_initial_poses(images, depth_prior, intrinsic, pred_tracks, pred_vis_scores, args, output_dir):
    """Estimate initial camera extrinsics and unproject depth to 3D point map."""
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


def sample_points_at_track_locations(pred_tracks, pred_vis_scores, points_3d, depth_conf,
                                      image_masks, points_rgb, query_index, image_shape):
    """
    Sample points_3d and depth_conf at query point locations.
    This replicates the logic from predict_tracks/_forward_on_query without re-running tracking.
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


def filter_and_verify_tracks(images, points_3d, extrinsic, intrinsic, pred_tracks, pred_vis_scores,
                             points_rgb, args, output_dir):
    """Filter tracks by geometry verification and valid correspondences."""
    track_mask = pred_vis_scores > args.vis_thresh

    track_mask = verify_tracks_by_geometry(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        ref_index=args.cond_index,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
    )
    visualize_tracks_on_images(
        images[None],
        torch.from_numpy(pred_tracks[None]),
        torch.from_numpy(track_mask[None]),
        out_dir=f"{output_dir}/track_filter_max_proj_err"
    )

    # Prep valid correspondences and drop 3D points without surviving tracks
    track_mask, points_3d, keep_pts = prep_valid_correspondences(
        points_3d, track_mask, args.min_inlier_per_frame, args.min_inlier_per_track
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


def propagate_uncertainty_and_build_image_info(images, image_path_list, base_image_path_list, original_coords,
                                   image_masks, depth_prior, intrinsic, extrinsic,
                                   pred_tracks, track_mask, points_3d, points_rgb, args):
    """Optimize poses and build the image info dictionary."""
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


def register_remaining_frames(image_info, gen_3d, args):
    """Register all remaining frames in the sequence."""
    num_images = len(image_info["images"])
    image_info["registered"] = np.array([False] * num_images)
    image_info["registered"][args.cond_index] = True

    image_info["invalid"] = np.array([False] * num_images)

    image_info["keyframe"] = np.array([False] * num_images)
    image_info["keyframe"][args.cond_index] = True

    save_results(image_info, gen_3d, out_dir=f"{args.output_dir}/results/{args.cond_index:04d}/")

    while image_info["registered"].sum() + image_info["invalid"].sum() < num_images:
        next_frame_idx = find_next_frame(image_info)
        print("+" * 50)
        print(f"Next frame to register: {next_frame_idx}, registered: {image_info['registered'].sum()}, keyframes: {image_info['keyframe'].sum()}, invalid: {image_info['invalid'].sum()}")

        if check_frame_invalid(
            image_info, next_frame_idx,
            min_inlier_per_frame=args.min_inlier_per_frame,
            min_depth_pixels=args.min_depth_pixels
        ):
            image_info["invalid"][next_frame_idx] = True
            continue

        register_new_frame(
            image_info, gen_3d, next_frame_idx, args,
            out_dir=Path(args.output_dir) / "results" / f"{next_frame_idx:04d}"
        )
        image_info["registered"][next_frame_idx] = True
        print(f"Frame {next_frame_idx} registered.")

        if check_key_frame(
            image_info, next_frame_idx,
            rot_thresh=args.kf_rot_thresh,
            trans_thresh=args.kf_trans_thresh,
            depth_thresh=args.kf_depth_thresh,
            frame_inliner_thresh=args.kf_inlier_thresh
        ):
            process_key_frame(image_info, next_frame_idx, args)
            image_info["keyframe"][next_frame_idx] = True
            print(f"Frame {next_frame_idx} marked as keyframe.")

        save_results(image_info, gen_3d, out_dir=Path(args.output_dir) / "results" / f"{next_frame_idx:04d}")
        print("-" * 50)


# =============================================================================
# Main demo function
# =============================================================================

def demo_fn(args):
    """
    Main demo function for 3D reconstruction from image sequences.

    Pipeline steps:
    1. Setup environment (device, dtype, output directory)
    2. Load images and intrinsics
    3. Predict initial tracks
    4. Estimate camera poses
    5. Sample 3D points at track locations
    6. Filter and verify tracks
    7. Optimize poses and build image info
    8. Get 3D correspondences and align model
    9. Register remaining frames
    """
    # Step 1: Setup environment
    device, dtype = setup_environment(args)
    if device is None:
        return

    # Step 2: Load images and intrinsics
    (
        _,  # image_dir (unused)
        image_path_list,
        base_image_path_list,
        images,
        original_coords,
        image_masks,
        depth_prior,
        intrinsic,
        depth_conf,
        _,  # vggt_fixed_resolution (unused, overwritten)
        img_load_resolution,
        gen_3d,
    ) = load_images_and_intrinsics(args, device)

    # Step 3: Predict initial tracks
    pred_tracks, pred_vis_scores, points_rgb = predict_initial_tracks_wrapper(
        images, image_masks, args, dtype
    )

    # Step 4: Estimate initial camera poses and unproject depth to 3D
    extrinsic, intrinsic, points_3d_map, _ = estimate_initial_poses(
        images, depth_prior, intrinsic, pred_tracks, pred_vis_scores, args, args.output_dir
    )

    # Step 5: Sample 3D points and confidence at track locations
    pred_tracks, pred_vis_scores, _, points_3d, points_rgb = sample_points_at_track_locations(
        pred_tracks, pred_vis_scores, points_3d_map, depth_conf,
        image_masks, points_rgb, args.cond_index, images.shape
    )

    # Rescale intrinsics to match image resolution
    scale = depth_conf.shape[-1] / images.shape[-1] if depth_conf is not None else 1.0
    intrinsic[:, :2, :] *= scale

    # Step 6: Filter and verify tracks by geometry
    track_mask, points_3d, pred_tracks, points_rgb = filter_and_verify_tracks(
        images, points_3d, extrinsic, intrinsic, pred_tracks, pred_vis_scores,
        points_rgb, args, args.output_dir
    )

    # Step 7: Optimize poses and build image info dictionary
    image_info = propagate_uncertainty_and_build_image_info(
        images, image_path_list, base_image_path_list, original_coords,
        image_masks, depth_prior, intrinsic, extrinsic,
        pred_tracks, track_mask, points_3d, points_rgb, args
    )

    # Step 8: Get 3D correspondences and align generated model with images
    corres = get_3D_correspondences(
        gen_3d, image_info,
        reference_idx=args.cond_index,
        out_dir=f"{args.output_dir}/3D_corres/",
        min_vis_score=args.vis_thresh
    )
    align_3D_model_with_images(
        corres, gen_3d, image_info,
        reference_idx=args.cond_index,
        out_dir=f"{args.output_dir}/aligned/"
    )

    # Step 9: Register remaining frames
    register_remaining_frames(image_info, gen_3d, args)

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


def get_points_uncertainty_colors(points_3d, uncertainties):
    uncertainty = np.asarray(uncertainties, dtype=np.float64)
    uncertainty_colors = np.zeros((len(uncertainty), 3), dtype=np.uint8)
    finite_mask = np.isfinite(uncertainty)
    if finite_mask.any():
        finite_uncertainty = uncertainty[finite_mask]
        uncertainty_norm = finite_uncertainty / (finite_uncertainty.max() + 1e-8)
        finite_colors = np.stack(
            [
                uncertainty_norm * 255.0,  # red channel high -> high uncertainty
                (1.0 - uncertainty_norm) * 255.0,          # green channel high -> low confidence
                np.zeros_like(uncertainty_norm),
            ],
            axis=-1,
        ).clip(0, 255).astype(np.uint8)
        uncertainty_colors[finite_mask] = finite_colors
    # Non-finite uncertainties (e.g., np.inf) remain black.
    if len(uncertainty_colors) != len(points_3d):
        uncertainty_colors = uncertainty_colors[: len(points_3d)]
    return uncertainty_colors

def save_point_cloud_with_conf(points_3d, points_rgb, uncertainties, ply_path):
    """Save point cloud; color by uncertainty if provided, else by rgb."""
    conf_colors = get_points_uncertainty_colors(points_3d, uncertainties)

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
