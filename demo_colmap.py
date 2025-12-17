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
    parser.add_argument("--min_inlier_per_frame", type=int, default=0, help="Minimum inliers per frame for BA")
    parser.add_argument("--min_inlier_per_track", type=int, default=2, help="Minimum inliers per track for BA")
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

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = Path(os.path.join(args.scene_dir, "images"))
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    image_path_list = sorted(image_path_list)
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
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    else:
        intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", "0000.pkl"))
        depth_conf = np.ones_like(depth_prior)
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
        track_mask = pred_vis_scores > args.vis_thresh
        extrinsic = estimate_extrinsic(depth_prior, intrinsic, pred_tracks, track_mask)
        intrinsic = np.tile(intrinsic[None, :, :], (len(images), 1, 1))
        points_3d = unproject_depth_map_to_point_map(depth_prior[..., None], extrinsic, intrinsic)
        vggt_fixed_resolution = img_load_resolution

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
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
            
            visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>= pred_vis_scores.min(), out_dir=f"{args.output_dir}/track_raw")            

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh
        visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(track_mask[None]), out_dir=f"{args.output_dir}/track_filter_vis_thresh")            
        # TODO: radial distortion, iterative BA, masks
        
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

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = not args.use_calibrated_intrinsic
        ba_options.refine_principal_point = not args.use_calibrated_intrinsic
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution

        if args.use_sfm:
            sfm_dir = Path(f"{args.output_dir}/sfm")
            sfm_pairs_f = Path(sfm_dir / "pairs.txt")
            sfm_feats_f = Path(sfm_dir / "feats.h5")
            sfm_matches_f = Path(sfm_dir / "matches.h5")
            os.makedirs(sfm_dir, exist_ok=True)

            prepare_pairs(image_path_list, 
                          pairs_file=sfm_pairs_f)
            prepare_features(pred_tracks, 
                             track_masks,
                             image_path_list,
                             image_size=images.shape[-2:],        
                             feats_file=sfm_feats_f)
            prepare_matches(pred_tracks, 
                            pairs_file=sfm_pairs_f, 
                            feats_file=sfm_feats_f,
                            matches_file=sfm_matches_f)
            
            print(f"vggt intrinsic:\n{intrinsic[0]}")
            intrinsic_f = None
            if args.use_calibrated_intrinsic:
                print(f"Using calibrated intrinsic for reconstruction")
                intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", "0000.pkl"))
                print(f"calibrated intrinsic:\n{intrinsic[0]}")
                # convert the intrinsic to a text file which can be read by hloc
                intrinsic_f = sfm_dir / "intrinsics.txt"
                save_intrinsics(intrinsic, intrinsic_f)

            model = hloc_reconstruction_main(sfm_dir/"sparse", image_dir, sfm_pairs_f, sfm_feats_f, sfm_matches_f, camera_mode=pycolmap.CameraMode.SINGLE, intrinsic_f=intrinsic_f, image_list=base_image_path_list)
            if model is not None:
                ply_path = sfm_dir / "sparse" / "points.ply"
                ply_path.parent.mkdir(parents=True, exist_ok=True)
                model.export_PLY(ply_path)
                print(f"Exported SfM point cloud to {ply_path}")
            else:
                print("SfM reconstruction failed; skip exporting point cloud.")
            

    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

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
    with torch.no_grad():
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
