# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Align generated 3D model with reconstructed scene using keyframe correspondences.
"""

import argparse
import os
import json
import pickle
from pathlib import Path

import numpy as np

from robust_hoi_pipeline.correspondence_alignment import (
    get_3D_correspondences,
    align_3D_model_with_images,
    save_aligned_3D_model,
)


def _ensure_keyframe_mask(image_info, keyframe_indices):
    keyframe_mask = image_info.get("keyframe")
    if keyframe_mask is not None:
        return image_info

    num_frames = None
    extrinsics = image_info.get("extrinsics")
    image_masks = image_info.get("image_masks")
    images = image_info.get("images")

    if extrinsics is not None:
        num_frames = len(extrinsics)
    elif image_masks is not None:
        num_frames = len(image_masks)
    elif images is not None:
        num_frames = len(images)

    if num_frames is None:
        print("[_ensure_keyframe_mask] Unable to infer number of frames, skipping keyframe mask setup")
        return image_info

    keyframe_mask = np.zeros(num_frames, dtype=bool)
    for idx in keyframe_indices:
        if 0 <= idx < num_frames:
            keyframe_mask[idx] = True
    image_info["keyframe"] = keyframe_mask
    return image_info


def _decompose_similarity(matrix):
    linear = matrix[:3, :3].astype(np.float64)
    col_norms = np.linalg.norm(linear, axis=0)
    scale = float(col_norms.mean()) if np.all(col_norms > 0) else 1.0
    rotation = linear / scale if scale != 0.0 else linear
    translation = matrix[:3, 3].astype(np.float64)
    return rotation, translation, scale


def save_aligned_results(gen_3d, refined_pose, image_info, output_dir, prefix="", unc_thresh=None):
    import trimesh

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mesh_path = getattr(gen_3d, "mesh_path", None)
    if mesh_path and os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path, process=False)
        if refined_pose is not None:
            mesh.apply_transform(refined_pose)
        mesh.export(output_path / f"white_mesh_remesh{prefix}.obj")

    if refined_pose is not None:
        rotation, translation, scale = _decompose_similarity(refined_pose)
        with open(output_path / f"aligned_transform{prefix}.json", "w") as f:
            json.dump(
                {
                    "matrix": refined_pose.tolist(),
                    "rotation": rotation.tolist(),
                    "translation": translation.tolist(),
                    "scale": scale,
                },
                f,
                indent=2,
            )

    # Save the 3D point cloud filtered by uncertainty threshold to a PLY file
    points3d = image_info.get("points_3d")
    uncertainties = image_info.get("uncertainties")

    if points3d is not None:
        points3d_unc = uncertainties.get("points3d") if uncertainties else None

        if unc_thresh is not None and points3d_unc is not None:
            # Filter points where uncertainty does not exceed threshold
            valid_mask = np.isfinite(points3d_unc) & (points3d_unc <= unc_thresh)
            filtered_points = points3d[valid_mask]
            print(f"[save_aligned_results] Filtered {valid_mask.sum()}/{len(points3d)} points with unc <= {unc_thresh}")
        else:
            filtered_points = points3d
            print(f"[save_aligned_results] Saving all {len(points3d)} points (no uncertainty filtering)")

        if len(filtered_points) > 0:
            point_cloud = trimesh.PointCloud(filtered_points)
            point_cloud.export(output_path / f"points3d_{prefix}.ply")
            print(f"[save_aligned_results] Saved point cloud to {output_path / f'points3d_{prefix}.ply'}")


def load_keyframe_indices(results_dir):
    """Load keyframe indices from key_frame_idx.txt.

    Args:
        results_dir: Path to results directory

    Returns:
        List of keyframe indices, or None if file not found
    """
    filepath = Path(results_dir) / "key_frame_idx.txt"
    if not filepath.exists():
        print(f"[load_keyframe_indices] File not found: {filepath}")
        return None

    keyframe_indices = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                keyframe_indices.append(int(line))

    print(f"[load_keyframe_indices] Loaded {len(keyframe_indices)} keyframe indices: {keyframe_indices}")
    return keyframe_indices


def load_image_info_from_results(results_dir, frame_idx, data_processed_dir=None):
    """Load image_info from a specific frame's results.pkl.

    Args:
        results_dir: Path to results directory
        frame_idx: Frame index to load
        data_processed_dir: Optional path to data_processed directory for loading cached images

    Returns:
        image_info dictionary or None if not found
    """
    results_path = Path(results_dir) / f"{frame_idx:04d}" / "results.pkl"
    if not results_path.exists():
        print(f"[load_image_info_from_results] File not found: {results_path}")
        return None

    with open(results_path, "rb") as f:
        image_info = pickle.load(f)

    print(f"[load_image_info_from_results] Loaded image_info from {results_path}")

    # If images are missing, try to load from data_processed cache
    if image_info.get("images") is None and data_processed_dir is not None:
        cache_path = Path(data_processed_dir) / "preprocessed_640_0.pt"
        if cache_path.exists():
            print(f"[load_image_info_from_results] Loading cached images from {cache_path}")
            import torch
            cached = torch.load(cache_path, map_location="cpu")
            image_info["images"] = cached.get("images")
            image_info["image_masks"] = cached.get("masks")
            image_info["depth_priors"] = cached.get("depths")
            if image_info.get("original_coords") is None:
                image_info["original_coords"] = cached.get("original_coords")

    # Create dummy uncertainties if missing (required for get_3D_correspondences)
    if image_info.get("uncertainties") is None:
        depth_priors = image_info.get("depth_priors")
        if depth_priors is not None:
            # Create uniform uncertainty (ones)
            if hasattr(depth_priors, 'shape'):
                depth_unc = np.ones_like(depth_priors)
            else:
                depth_unc = None
            image_info["uncertainties"] = {
                "depth_prior": depth_unc,
                "points3d": None,
                "extrinsic": None,
            }
            print("[load_image_info_from_results] Created dummy uncertainties")

    return image_info


def load_gen_3d(gen_3d_input_dir):
    """Load GEN_3D object from gen_3d directory.

    Args:
        gen_3d_input_dir: Path to gen_3d input directory (e.g., output/{scene_name}/results/)

    Returns:
        GEN_3D object or None if paths don't exist
    """
    # GEN_3D expects path like: {scene_dir}/align_mesh_image/{cond_index:04d}
    # But we're given the output results path, so we need to find the gen_3d path
    gen_3d_path = Path(gen_3d_input_dir) / "gen_3d"
    if not gen_3d_path.exists():
        # Try alternative path structure
        gen_3d_path = Path(gen_3d_input_dir)

    # Check if we have the required files
    mesh_path = gen_3d_path / "white_mesh_remesh.obj"
    if not mesh_path.exists():
        # Try looking in 3D_gen folder
        mesh_path = Path(str(gen_3d_path).replace("align_mesh_image", "3D_gen")) / "white_mesh_remesh.obj"

    print(f"[load_gen_3d] Looking for mesh at: {mesh_path}")

    # Create a simple GEN_3D-like object with the necessary attributes
    class Gen3DWrapper:
        def __init__(self, gen_3d_path, mesh_path):
            self.gen_3D_path = Path(gen_3d_path)
            self.mesh_path = mesh_path
            self.condition_image_path = self.gen_3D_path / "image.png"
            self.camera_path = self.gen_3D_path / "camera.json"
            self.depth_path = self.gen_3D_path / "depth.png"
            self.gen2obj = None

        def get_mesh_path(self):
            return str(self.mesh_path)

        def get_cond_image(self, target_size=None):
            from PIL import Image
            img = Image.open(self.condition_image_path)
            # If there's an alpha channel, blend onto black background
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (0, 0, 0, 255))
                img = Image.alpha_composite(background, img)
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.BICUBIC)
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            return img

        def get_cond_mask(self):
            from PIL import Image
            img = Image.open(self.condition_image_path)
            if img.mode == "RGBA":
                mask = np.array(img.getchannel('A'))
                mask = mask > 0
                return mask
            else:
                width, height = img.size
                return np.ones((height, width), dtype=bool)

        def get_cond_depth(self):
            import sys
            _THIRD_PARTY_UTILS_SIMBA = str(Path(__file__).resolve().parent / "third_party" / "utils_simba")
            if _THIRD_PARTY_UTILS_SIMBA not in sys.path:
                sys.path.insert(0, _THIRD_PARTY_UTILS_SIMBA)
            from utils_simba.depth import get_depth
            return get_depth(str(self.depth_path))

        def get_cond_intrinsic(self):
            import yaml
            with open(self.camera_path, "r") as f:
                camera_data = yaml.safe_load(f)
            return np.array(camera_data["K"])

        def get_cond_extrinsic(self):
            import yaml
            with open(self.camera_path, "r") as f:
                camera_data = yaml.safe_load(f)
            w2c = np.array(camera_data["blw2cvc"])
            return w2c

        def save_aligned_pose(self, gen2obj):
            self.gen2obj = gen2obj

        def get_aligned_pose(self):
            return self.gen2obj

    try:
        gen_3d = Gen3DWrapper(gen_3d_path, mesh_path)
        return gen_3d
    except Exception as e:
        print(f"[load_gen_3d] Failed to create GEN_3D: {e}")
        return None


def main(args):
    input_dir = args.input_dir  # e.g., output/{scene_name}/results/gen_3d
    output_dir = args.output_dir  # e.g., output/{scene_name}/gen_3d_aligned/
    init_pose_image_idx = args.init_pose_image_idx  # e.g., 0

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load image, depth, camera information and white_mesh_remesh from gen_3d_input_dir
    print("=" * 50)
    print("Step 1: Loading gen_3d data...")
    print("=" * 50)

    gen_3d_dir = Path(input_dir) / "results" / "gen_3d"
    gen_3d = load_gen_3d(gen_3d_dir)
    if gen_3d is None:
        print("[main] Failed to load gen_3d, exiting.")
        return

    # Step 2: Get the last keyframes from key_frame_idx.txt
    print("=" * 50)
    print("Step 2: Loading keyframe indices...")
    print("=" * 50)
    
    results_dir = Path(input_dir) / "results"
    if not results_dir.exists():
        results_dir = Path(input_dir)

    keyframe_indices = load_keyframe_indices(results_dir)
    if keyframe_indices is None or len(keyframe_indices) == 0:
        print("[main] No keyframe indices found, using init_pose_image_idx as the only keyframe.")
        raise Exception("No keyframe indices found.")

    last_keyframe_idx = keyframe_indices[-1]
    print(f"[main] Last keyframe index: {last_keyframe_idx}")

    # Step 3: Load the image information from the last step of keyframe processing
    print("=" * 50)
    print("Step 3: Loading image information from last keyframe...")
    print("=" * 50)

    # Try to find data_processed directory for loading cached images
    data_processed_dir = Path(input_dir) / "data_processed"
    
    if not data_processed_dir.exists():
        data_processed_dir = None
    image_info = load_image_info_from_results(results_dir, last_keyframe_idx, data_processed_dir)

    if image_info is None:
        print("[main] Failed to load image_info, exiting.")
        return

    # Step 4: Initial alignment using 3D correspondences
    print("=" * 50)
    print("Step 4: Computing 3D correspondences and alignment...")
    print("=" * 50)

    # Get 3D correspondences between gen_3d and image observations
    corres_3d = get_3D_correspondences(
        gen_3d,
        image_info,
        reference_idx=init_pose_image_idx,
        out_dir=f"{output_dir}/3D_corres",
    )

    if corres_3d is None:
        print("[main] Failed to compute 3D correspondences, exiting.")
        return

    # Align 3D model with images using weighted Umeyama alignment (R, t, scale)
    aligned_pose = align_3D_model_with_images(
        corres_3d,
        gen_3d,
        image_info,
        reference_idx=init_pose_image_idx,
        out_dir=None,
    )

    if aligned_pose is None:
        print("[main] Alignment failed, exiting.")
        return

    # Step 5: Save the aligned results to gen_3d_output_dir
    print("=" * 50)
    print("Step 5: Saving aligned results...")
    print("=" * 50)

    refined_pose = gen_3d.get_aligned_pose()
    save_dir = Path(output_dir) / "init"
    save_aligned_results(gen_3d, refined_pose, image_info, save_dir, unc_thresh=args.unc_thresh)
    print(f"[main] init aligned results saved to {save_dir}")

    print(f"[main] Aligned results saved to {output_dir}")
    print("=" * 50)
    print("Alignment complete!")
    print("=" * 50)

    if args.enable_mask_refine:
        print("=" * 50)
        print("Step 6: Refining alignment with mask optimization...")
        print("=" * 50)

        image_info = _ensure_keyframe_mask(image_info, keyframe_indices)
        try:
            from robust_hoi_pipeline.mask_optimization import optimize_pose_with_mask_loss
        except Exception as exc:
            print(f"[main] Mask optimization unavailable: {exc}")
            return

        image_info, gen_3d = optimize_pose_with_mask_loss(image_info, gen_3d, args)
        refined_pose = gen_3d.get_aligned_pose()
        save_dir = Path(output_dir) / "refined"
        save_aligned_results(gen_3d, refined_pose, image_info, save_dir, unc_thresh=args.unc_thresh)
        print(f"[main] Refined results saved to {save_dir}")



def parse_args():
    parser = argparse.ArgumentParser(description="Align generated 3D model with reconstructed scene")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing gen_3d data (e.g., output/{scene_name}/results/gen_3d)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for aligned results (e.g., output/{scene_name}/gen_3d_aligned/)"
    )
    parser.add_argument(
        "--init_pose_image_idx",
        type=int,
        default=0,
        help="Initial pose image index for alignment reference"
    )
    parser.add_argument(
        "--enable_mask_refine",
        type=bool,
        default=True,
        help="Refine aligned pose using mask IoU loss"
    )
    parser.add_argument(
        "--mask_opt_iters",
        type=int,
        default=200,
        help="Number of iterations for mask-based optimization"
    )
    parser.add_argument(
        "--mask_opt_lr",
        type=float,
        default=1e-2,
        help="Learning rate for mask-based optimization"
    )
    parser.add_argument(
        "--optimize_intrinsic",
        action="store_true",
        help="Also optimize camera intrinsics during mask refinement"
    )
    parser.add_argument(
        "--unc_thresh",
        type=float,
        default=2.0,
        help="Uncertainty threshold for filtering 3D points when saving PLY (points with uncertainty > threshold are excluded)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
