import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images_square_ZED, load_intrinsics
from robust_hoi_pipeline.geometry_utils import adjust_intrinsic_for_new_image_size
from vggt.utils.geometry import depth_to_cam_coords_points

import sys
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import save_point_cloud_to_ply


def compute_bbox_area(points_3d, mask):
    """Compute the bounding box area of valid 3D points.

    Args:
        points_3d: (H, W, 3) point cloud in camera coordinates
        mask: (H, W) boolean mask for valid points

    Returns:
        float: Area of the 3D bounding box (width * height * depth)
    """
    valid_points = points_3d[mask]
    if len(valid_points) == 0:
        return 0.0

    min_coords = valid_points.min(axis=0)
    max_coords = valid_points.max(axis=0)
    bbox_size = max_coords - min_coords

    # Return volume as the "area" metric for 3D bbox
    return float(np.prod(bbox_size))


def main(args):
    # Get list of image paths
    image_dir = Path(args.scene_dir) / "rgb"
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Filter by frame range
    if args.max_frame_num > 0:
        image_paths = image_paths[args.min_frame_num:args.max_frame_num:args.frame_interval]

    image_path_list = [str(p) for p in image_paths]
    print(f"Processing {len(image_path_list)} images from {image_dir}")

    # Load and preprocess images, masks, and depths
    images, original_coords, masks, depths = load_and_preprocess_images_square_ZED(
        image_path_list,
        args,
        target_size=1024,
        out_dir=f"{args.out_dir}/data_processed",
    )

    # Load intrinsics from first frame's meta file
    intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", "0000.pkl"))

    # Process each frame
    bbox_areas = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(images)):
        frame_out_dir = out_dir / f"{idx:04d}"
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        # Adjust intrinsic for the preprocessed image size
        adjusted_intrinsic = adjust_intrinsic_for_new_image_size(
            intrinsic, original_coords, frame_idx=idx
        )

        # Get depth and mask for this frame
        depth = depths[idx].numpy() if torch.is_tensor(depths[idx]) else depths[idx]
        mask = masks[idx].numpy() if torch.is_tensor(masks[idx]) else masks[idx]

        # Handle depth shape - squeeze if needed
        if depth.ndim == 3:
            depth = depth.squeeze(-1) if depth.shape[-1] == 1 else depth.squeeze(0)

        # Handle mask shape - convert to boolean
        if mask.ndim == 3:
            mask = mask.squeeze(-1) if mask.shape[-1] == 1 else mask.squeeze(0)
        mask = mask > 0.5  # Convert to boolean

        # Create valid depth mask (positive depth values within mask)
        valid_mask = mask & (depth > 1e-6)

        # Get point cloud from depth and intrinsic
        point_cloud = depth_to_cam_coords_points(depth, adjusted_intrinsic)

        # Compute bbox area for sorting
        bbox_area = compute_bbox_area(point_cloud, valid_mask)
        bbox_areas.append((idx, bbox_area))

        # Save point cloud (only valid points)
        valid_points = point_cloud[valid_mask]
        if len(valid_points) > 0:
            save_point_cloud_to_ply(valid_points, str(frame_out_dir / "point_map.ply"))

        # Save image as RGBA
        image = images[idx]
        if torch.is_tensor(image):
            image = image.numpy()

        # Convert from CHW to HWC if needed
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = np.transpose(image, (1, 2, 0))

        # Normalize to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert mask to alpha channel
        if mask.ndim == 2:
            alpha = (mask * 255).astype(np.uint8)
        else:
            alpha = (mask.squeeze() * 255).astype(np.uint8)

        # Resize alpha to match image if needed
        if alpha.shape[:2] != image.shape[:2]:
            alpha_img = Image.fromarray(alpha)
            alpha_img = alpha_img.resize((image.shape[1], image.shape[0]), Image.NEAREST)
            alpha = np.array(alpha_img)

        # Create RGBA image
        if image.shape[-1] == 3:
            rgba = np.concatenate([image, alpha[..., None]], axis=-1)
        else:
            rgba = image
            rgba[..., 3] = alpha

        # Save RGBA image
        Image.fromarray(rgba).save(frame_out_dir / "image.png")

        print(f"Frame {idx:04d}: bbox_area={bbox_area:.6f}, points={len(valid_points)}")

    # Sort by bbox area (descending - larger objects first)
    sorted_indices = sorted(bbox_areas, key=lambda x: x[1], reverse=True)

    # Save sorted order to condition_id.txt
    with open(out_dir / "condition_id.txt", "w") as f:
        for idx, area in sorted_indices:
            f.write(f"{idx:04d} {area:.6f}\n")

    print(f"\nSaved condition IDs to {out_dir / 'condition_id.txt'}")
    print(f"Top 5 frames by bbox area: {[f'{idx:04d}' for idx, _ in sorted_indices[:5]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get condition IDs by sorting frames by object bbox area")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing scene data (rgb/, meta/)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for point clouds and images")
    parser.add_argument("--instance_id", type=int, default=0, help="Instance ID for image preprocessing")
    parser.add_argument("--min_frame_num", type=int, default=0, help="Minimum frame number to process")
    parser.add_argument("--max_frame_num", type=int, default=0, help="Maximum frame number to process (0 for all)")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval for processing")
    args = parser.parse_args()

    main(args)
