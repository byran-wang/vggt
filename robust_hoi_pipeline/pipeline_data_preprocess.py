import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

# Import depth filtering utilities
from utils_simba.depth import load_filtered_depth, depth2xyzmap, get_depth, save_depth, save_normal, get_normal

# Import normal computation
from robust_hoi_pipeline.geometry_utils import compute_normals_from_depth


def load_intrinsics_from_meta(meta_file: str) -> np.ndarray:
    """Load camera intrinsics from meta pickle file."""
    with open(meta_file, "rb") as f:
        meta_data = pickle.load(f)
    return np.array(meta_data["camMat"], dtype=np.float32)


def prepare_image_list(
    image_dir: Path,
    start: int,
    end: int,
    interval: int,
    cond_index: int
) -> List[int]:
    """Prepare sorted image list with selection by start, interval, end.

    Ensures cond_index is included in the list.

    Args:
        image_dir: Directory containing images
        start: Start frame index
        end: End frame index (-1 for all)
        interval: Frame sampling interval
        cond_index: Condition image index that must be included

    Returns:
        Sorted list of frame indices
    """
    # Get all available image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_frames = len(image_files)

    if end == -1 or end > total_frames:
        end = total_frames

    # Generate frame indices with interval
    frame_indices = list(range(start, end, interval))

    # Ensure cond_index is included
    if cond_index not in frame_indices and cond_index < total_frames:
        frame_indices.append(cond_index)
        frame_indices = sorted(frame_indices)

    return frame_indices


def load_image(image_path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask as grayscale numpy array."""
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L")
    return np.array(mask)


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, output_path: Path):
    """Save point cloud to PLY file.

    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255)
        output_path: Path to save PLY file
    """
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")


def pipeline_data_preprocess(args):
    """Master orchestration function for data preprocessing pipeline.

    Coordinates all steps:
    1. Setup environment
    2. Prepare image list
    3. Load images, masks, intrinsics, and depth
    4. Filter depth
    5. Compute normal maps
    6. Save debug point clouds
    7. Load hand poses
    8. Save all preprocessed data

    Args:
        args: Parsed command-line arguments with:
            - data_dir: Input data directory
            - output_dir: Output directory
            - start: Start frame index
            - end: End frame index (-1 for all)
            - interval: Frame sampling interval
            - cond_index: Condition image index
    """
    data_dir = Path(args.data_dir)
    image_dir = data_dir / "rgb"
    mask_obj_dir = data_dir / "mask_object"
    mask_hand_dir = data_dir / "mask_hand"
    intrinsic_dir = data_dir / "meta"
    depth_dir = data_dir / "depth"
    hand_pose_file = data_dir / "hands" / "hold_fit.init.npy"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare image list
    print("Preparing image list...")
    frame_indices = prepare_image_list(
        image_dir,
        start=getattr(args, 'start', 0),
        end=getattr(args, 'end', -1),
        interval=getattr(args, 'interval', 1),
        cond_index=getattr(args, 'cond_index', 0)
    )
    print(f"Selected {len(frame_indices)} frames: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    # 2. Load hand poses (if available)
    hand_poses = None
    if hand_pose_file.exists():
        print(f"Loading hand poses from {hand_pose_file}")
        hand_poses_raw = np.load(hand_pose_file, allow_pickle=True)
        # Handle different numpy array formats (0-dim arrays contain the actual data)
        if hand_poses_raw.ndim == 0:
            hand_poses = hand_poses_raw.item()  # Extract the actual object
        else:
            hand_poses = hand_poses_raw
        if isinstance(hand_poses, np.ndarray):
            print(f"Loaded hand poses shape: {hand_poses.shape}")
        elif isinstance(hand_poses, dict):
            print(f"Loaded hand poses as dict with {len(hand_poses)} keys")
        else:
            print(f"Loaded hand poses type: {type(hand_poses)}")

    # 3. Process each frame
    preprocessed_data = []

    for idx, frame_idx in enumerate(frame_indices):
        print(f"Processing frame {frame_idx} ({idx+1}/{len(frame_indices)})...")

        # Construct file paths
        image_path = image_dir / f"{frame_idx:04d}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{frame_idx:04d}.png"

        mask_obj_path = mask_obj_dir / f"{frame_idx:04d}.png"
        mask_hand_path = mask_hand_dir / f"{frame_idx:04d}.png"
        depth_path = depth_dir / f"{frame_idx:04d}.png"
        meta_path = intrinsic_dir / f"{frame_idx:04d}.pkl"

        # Check file existence
        if not image_path.exists():
            print(f"  Warning: Image not found: {image_path}")
            continue
        if not depth_path.exists():
            print(f"  Warning: Depth not found: {depth_path}")
            continue
        if not meta_path.exists():
            print(f"  Warning: Meta not found: {meta_path}")
            continue

        # Load image
        image = load_image(image_path)
        H, W = image.shape[:2]

        # Load masks
        mask_obj = load_mask(mask_obj_path)
        mask_hand = load_mask(mask_hand_path)

        # Load intrinsics
        intrinsics = load_intrinsics_from_meta(str(meta_path))

        # Load raw depth first (before filtering)
        raw_depth = get_depth(str(depth_path))
        raw_xyz_map = depth2xyzmap(raw_depth, intrinsics)

        # Load and filter depth
        filtered_depth = load_filtered_depth(
            str(depth_path),
            thresh_min=args.dpeth_min if hasattr(args, 'dpeth_min') else 0.1,
            thresh_max=args.dpeth_max if hasattr(args, 'dpeth_max') else 1.5,
        )

        # Compute point clouds
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        
        filtered_xyz_map = depth2xyzmap(filtered_depth, intrinsics)

        # Compute normal map from filtered depth
        depth_tensor = torch.from_numpy(filtered_depth).float()
        depth_batch = depth_tensor.unsqueeze(0)  # (1, H, W)
        intrinsics_batch = intrinsics_tensor.unsqueeze(0)  # (1, 3, 3)
        normal_map = compute_normals_from_depth(depth_batch, intrinsics_batch)
        normal_map = normal_map.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

        # Get hand pose for this frame
        hand_pose = None
        if hand_poses is not None:
            try:
                if isinstance(hand_poses, dict):
                    # Dict format: keys might be frame indices or strings
                    key = frame_idx if frame_idx in hand_poses else str(frame_idx)
                    hand_pose = hand_poses.get(key, None)
                elif isinstance(hand_poses, np.ndarray) and hand_poses.ndim >= 1:
                    if frame_idx < len(hand_poses):
                        hand_pose = hand_poses[frame_idx]
                elif isinstance(hand_poses, list):
                    if frame_idx < len(hand_poses):
                        hand_pose = hand_poses[frame_idx]
            except Exception as e:
                print(f"  Warning: Could not get hand pose for frame {frame_idx}: {e}")

        # Save debug point clouds for first few frames
        if idx < 5:
            # Save raw point cloud (before filtering)
            valid_mask_raw = raw_depth > 0
            points_raw = raw_xyz_map[valid_mask_raw]
            colors_raw = image[valid_mask_raw]

            if len(points_raw) > 0:
                debug_ply_path_raw = debug_dir / f"pointcloud_raw_{frame_idx:04d}.ply"
                save_point_cloud_ply(points_raw, colors_raw, debug_ply_path_raw)
                print(f"  Saved raw point cloud: {debug_ply_path_raw}")

            # Save filtered point cloud
            valid_mask_filtered = filtered_depth > 0
            points_filtered = filtered_xyz_map[valid_mask_filtered]
            colors_filtered = image[valid_mask_filtered]

            if len(points_filtered) > 0:
                debug_ply_path_filtered = debug_dir / f"pointcloud_filtered_{frame_idx:04d}.ply"
                save_point_cloud_ply(points_filtered, colors_filtered, debug_ply_path_filtered)
                print(f"  Saved filtered point cloud: {debug_ply_path_filtered}")

            # Save raw depth visualization
            if raw_depth.max() > 0:
                depth_vis_raw = (raw_depth / raw_depth.max() * 255).astype(np.uint8)
                cv2.imwrite(str(debug_dir / f"depth_raw_{frame_idx:04d}.png"), depth_vis_raw)

            # Save filtered depth visualization
            if filtered_depth.max() > 0:
                depth_vis = (filtered_depth / filtered_depth.max() * 255).astype(np.uint8)
                cv2.imwrite(str(debug_dir / f"depth_filtered_{frame_idx:04d}.png"), depth_vis)

            # Save normal map visualization
            save_normal(normal_map, str(debug_dir / f"normal_{frame_idx:04d}.png"))

        # Store preprocessed data
        frame_data = {
            'frame_idx': frame_idx,
            'image': image,
            'mask_obj': mask_obj,
            'mask_hand': mask_hand,
            'intrinsics': intrinsics,
            'depth_filtered': filtered_depth,
            'normal_map': normal_map,
            'hand_pose': hand_pose,
        }
        preprocessed_data.append(frame_data)

    # 4. Save preprocessed data to output directory
    print(f"\nSaving preprocessed data to {out_dir}...")
    # Create output subdirectories
    (out_dir / "rgb").mkdir(exist_ok=True)
    (out_dir / "mask_obj").mkdir(exist_ok=True)
    (out_dir / "mask_hand").mkdir(exist_ok=True)
    (out_dir / "depth_filtered").mkdir(exist_ok=True)
    (out_dir / "normal").mkdir(exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)

    # Save each frame's data
    for frame_data in preprocessed_data:
        frame_idx = frame_data['frame_idx']

        # Save image
        Image.fromarray(frame_data['image']).save(out_dir / "rgb" / f"{frame_idx:04d}.png")

        # Save masks
        if frame_data['mask_obj'] is not None:
            Image.fromarray(frame_data['mask_obj']).save(out_dir / "mask_obj" / f"{frame_idx:04d}.png")
        if frame_data['mask_hand'] is not None:
            Image.fromarray(frame_data['mask_hand']).save(out_dir / "mask_hand" / f"{frame_idx:04d}.png")

        # Save filtered depth using utils_simba format (compatible with get_depth)
        save_depth(frame_data['depth_filtered'], str(out_dir / "depth_filtered" / f"{frame_idx:04d}.png"))
        
        # Save normal map using utils_simba format (compatible with get_normal)
        save_normal(frame_data['normal_map'], str(out_dir / "normal" / f"{frame_idx:04d}.png"))

        # Save metadata (intrinsics + hand pose)
        meta = {
            'frame_idx': frame_idx,
            'intrinsics': frame_data['intrinsics'],
            'hand_pose': frame_data['hand_pose'],
        }
        with open(out_dir / "meta" / f"{frame_idx:04d}.pkl", 'wb') as f:
            pickle.dump(meta, f)

    # Save frame list
    frame_list_path = out_dir / "frame_list.txt"
    with open(frame_list_path, 'w') as f:
        for frame_data in preprocessed_data:
            f.write(f"{frame_data['frame_idx']:04d}\n")

    print(f"Saved {len(preprocessed_data)} frames to {out_dir}")
    print(f"Frame list saved to {frame_list_path}")

    return preprocessed_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess HO3D data for robust HOI pipeline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index")
    parser.add_argument("--end", type=int, default=-1,
                        help="End frame index (-1 for all)")
    parser.add_argument("--interval", type=int, default=1,
                        help="Frame sampling interval")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition image index (must be included)")

    args = parser.parse_args()
    pipeline_data_preprocess(args)
