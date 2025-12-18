# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
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
from vggt.utils.load_fn import load_and_preprocess_images_square, load_intrinsics
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
    """Lightweight pose refinement using reprojection, depth, and point-to-plane losses."""
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

    R_final = axis_angle_to_matrix(rvecs_t)
    extr_final = torch.cat([R_final, tvecs_t.unsqueeze(-1)], dim=-1)
    return extr_final.detach().cpu().numpy(), points3d_t.detach().cpu().numpy()


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
        extrinsic = estimate_extrinsic(depth_prior, intrinsic, pred_tracks, track_mask)
        visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(track_mask[None]), out_dir=f"{args.output_dir}/track_filter_vis_thresh")
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
        extrinsic, points_3d = optimize_poses_with_losses(
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
    reconstruction.write(ba_out_dir)
    
    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(ba_out_dir / "points.ply")
    #TODO print reconstruction summary
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
