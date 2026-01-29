import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images_square_HO3D, load_intrinsics
from robust_hoi_pipeline.geometry_utils import adjust_intrinsic_for_new_image_size
from vggt.utils.geometry import depth_to_cam_coords_points

import sys
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import save_point_cloud_to_ply


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def get_image_paths(scene_dir, min_frame, max_frame, interval):
    """Load and filter image paths from the scene directory."""
    image_dir = Path(scene_dir) / "rgb"
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    return image_dir, [str(p) for p in image_paths[min_frame:max_frame:interval]]


# -----------------------------------------------------------------------------
# Tensor/Array Conversion
# -----------------------------------------------------------------------------

def to_numpy(tensor):
    """Convert tensor to numpy array if needed."""
    return tensor.numpy() if torch.is_tensor(tensor) else tensor


def squeeze_to_2d(arr):
    """Squeeze a 3D array to 2D by removing singleton dimension."""
    if arr.ndim == 3:
        return arr.squeeze(-1) if arr.shape[-1] == 1 else arr.squeeze(0)
    return arr


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def compute_bbox_volume(points_3d, mask):
    """Compute the bounding box volume of valid 3D points.

    Args:
        points_3d: (H, W, 3) point cloud in camera coordinates
        mask: (H, W) boolean mask for valid points

    Returns:
        float: Volume of the 3D bounding box
    """
    valid_points = points_3d[mask]
    if len(valid_points) == 0:
        return 0.0

    min_coords = valid_points.min(axis=0)
    max_coords = valid_points.max(axis=0)
    bbox_size = max_coords - min_coords

    return float(np.prod(bbox_size))


# -----------------------------------------------------------------------------
# Image Processing
# -----------------------------------------------------------------------------

def normalize_image_to_uint8(image):
    """Convert image to HWC uint8 format."""
    image = to_numpy(image)

    # Convert CHW to HWC if needed
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))

    # Normalize to 0-255
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def create_alpha_channel(mask, target_shape):
    """Create alpha channel from mask, resizing if needed."""
    alpha = (mask.squeeze() * 255).astype(np.uint8)

    if alpha.shape[:2] != target_shape[:2]:
        alpha_img = Image.fromarray(alpha)
        alpha_img = alpha_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        alpha = np.array(alpha_img)

    return alpha


def create_rgba_image(image, mask):
    """Convert image and mask to RGBA format."""
    image = normalize_image_to_uint8(image)
    alpha = create_alpha_channel(mask, image.shape)

    if image.shape[-1] == 3:
        return np.concatenate([image, alpha[..., None]], axis=-1)
    else:
        image[..., 3] = alpha
        return image


# -----------------------------------------------------------------------------
# Frame Processing
# -----------------------------------------------------------------------------

def process_frame(idx, images, depths, masks, intrinsic, original_coords, out_dir, args):
    """Process a single frame: generate point cloud, compute bbox, save outputs.

    Returns:
        tuple: (bbox_volume, num_valid_points)
    """
    frame_out_dir = out_dir / f"{(idx * args.frame_interval):04d}"
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare depth and mask arrays
    depth = squeeze_to_2d(to_numpy(depths[idx]))
    mask = squeeze_to_2d(to_numpy(masks[idx])) > 0.5

    # Adjust intrinsic for preprocessed image size
    adjusted_intrinsic = adjust_intrinsic_for_new_image_size(
        intrinsic, original_coords, frame_idx=idx
    )

    # Generate point cloud from depth
    valid_mask = mask & (depth > 1e-6)
    point_cloud = depth_to_cam_coords_points(depth, adjusted_intrinsic)
    valid_points = point_cloud[valid_mask]

    # Compute bounding box volume
    bbox_volume = compute_bbox_volume(point_cloud, valid_mask)

    # Save point cloud
    if len(valid_points) > 0:
        save_point_cloud_to_ply(valid_points, str(frame_out_dir / "point_map.ply"))

    # Save RGBA image
    rgba = create_rgba_image(images[idx], mask)
    Image.fromarray(rgba).save(frame_out_dir / "image.png")

    return bbox_volume, len(valid_points)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_condition_ids(out_dir, bbox_results):
    """Save frame indices sorted by bbox volume (descending) to condition_id.txt."""
    sorted_results = sorted(bbox_results, key=lambda x: x[1], reverse=True)

    with open(out_dir / "condition_id.txt", "w") as f:
        for idx, volume in sorted_results:
            f.write(f"{idx:04d} {volume:.6f}\n")

    print(f"\nSaved condition IDs to {out_dir / 'condition_id.txt'}")
    print(f"Top 5 frames by bbox volume: {[f'{idx:04d}' for idx, _ in sorted_results[:5]]}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    # Load image paths
    image_dir, image_path_list = get_image_paths(
        args.scene_dir, args.min_frame_num, args.max_frame_num, args.frame_interval
    )
    print(f"Processing {len(image_path_list)} images from {image_dir}")

    # Preprocess images, masks, and depths
    images, original_coords, masks, depths = load_and_preprocess_images_square_HO3D(
        image_path_list,
        args,
        target_size=1024,
        out_dir=f"{args.out_dir}/data_processed",
    )

    # Load camera intrinsics
    intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", "0000.pkl"))

    # Process all frames
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_results = []
    for idx in range(len(images)):
        bbox_volume, num_points = process_frame(
            idx, images, depths, masks, intrinsic, original_coords, out_dir, args
        )
        bbox_results.append((idx, bbox_volume))
        print(f"Frame {idx:04d}: bbox_volume={bbox_volume:.6f}, points={num_points}")

    # Save sorted condition IDs
    save_condition_ids(out_dir, bbox_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get condition IDs by sorting frames by object bbox volume"
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True,
        help="Directory containing scene data (rgb/, meta/)"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for point clouds and images"
    )
    parser.add_argument(
        "--instance_id", type=int, default=0,
        help="Instance ID for image preprocessing"
    )
    parser.add_argument(
        "--min_frame_num", type=int, default=0,
        help="Minimum frame number to process"
    )
    parser.add_argument(
        "--max_frame_num", type=int, default=-1,
        help="Maximum frame number to process (-1 for all)"
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1,
        help="Frame interval for processing"
    )
    args = parser.parse_args()

    main(args)
