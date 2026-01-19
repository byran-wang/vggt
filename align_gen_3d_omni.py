import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import trimesh


def load_keyframe_indices(results_dir):
    """Load keyframe indices from key_frame_idx.txt."""
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


def load_image_info_from_results(results_dir, frame_idx):
    """Load image_info from a specific frame's results.pkl."""
    results_path = Path(results_dir) / f"{frame_idx:04d}" / "results.pkl"
    if not results_path.exists():
        print(f"[load_image_info_from_results] File not found: {results_path}")
        return None

    with open(results_path, "rb") as f:
        image_info = pickle.load(f)

    print(f"[load_image_info_from_results] Loaded image_info from {results_path}")
    return image_info


def load_aligned_transform(aligned_dir):
    """Load aligned transformation from JSON file."""
    transform_path = Path(aligned_dir) / "transform_refined.json"
    if not transform_path.exists():
        # Try alternative path
        transform_path = Path(aligned_dir) / "aligned_transform.json"
    if not transform_path.exists():
        print(f"[load_aligned_transform] Transform file not found in {aligned_dir}")
        return None

    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    matrix = np.array(transform_data["matrix"])
    print(f"[load_aligned_transform] Loaded transform from {transform_path}")
    return matrix


def filter_points_by_uncertainty(points_3d, uncertainties, unc_thresh):
    """Filter 3D points by uncertainty threshold."""
    if points_3d is None:
        return None, None

    points_unc = uncertainties.get("points3d") if uncertainties else None

    if unc_thresh is not None and points_unc is not None:
        valid_mask = np.isfinite(points_unc) & (points_unc <= unc_thresh)
        filtered_points = points_3d[valid_mask]
        print(f"[filter_points_by_uncertainty] Filtered {valid_mask.sum()}/{len(points_3d)} points with unc <= {unc_thresh}")
        return filtered_points, valid_mask
    else:
        print(f"[filter_points_by_uncertainty] No filtering applied, using all {len(points_3d)} points")
        return points_3d, np.ones(len(points_3d), dtype=bool)


def transform_points_to_object_coords(points, world2obj_matrix):
    """Transform points from world coordinates to object coordinates.

    Args:
        points: (N, 3) array of 3D points in world coordinates
        world2obj_matrix: (4, 4) transformation matrix from world to object coords
                          (inverse of the aligned pose which is obj2world)

    Returns:
        (N, 3) array of 3D points in object coordinates
    """
    if points is None or len(points) == 0:
        return points

    # Convert to homogeneous coordinates
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])  # (N, 4)

    # Apply transformation
    transformed = (world2obj_matrix @ points_homo.T).T  # (N, 4)

    # Convert back to 3D
    return transformed[:, :3]


def save_points_as_ply(points, output_path, colors=None):
    """Save 3D points to PLY file."""
    if points is None or len(points) == 0:
        print(f"[save_points_as_ply] No points to save")
        return None

    point_cloud = trimesh.PointCloud(points, colors=colors)
    point_cloud.export(output_path)
    print(f"[save_points_as_ply] Saved {len(points)} points to {output_path}")
    return output_path


def transform_mesh_to_world_coords(mesh_path, obj2world_matrix, output_path):
    """Transform a mesh from object coordinates to world coordinates.

    Args:
        mesh_path: Path to the input mesh file
        obj2world_matrix: (4, 4) transformation matrix from object to world coords
        output_path: Path to save the transformed mesh

    Returns:
        Path to the saved mesh, or None if failed
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        print(f"[transform_mesh_to_world_coords] Mesh not found: {mesh_path}")
        return None

    mesh = trimesh.load(mesh_path, process=False)
    mesh.apply_transform(obj2world_matrix)
    mesh.export(output_path)
    print(f"[transform_mesh_to_world_coords] Saved transformed mesh to {output_path}")
    return output_path


def main(args):
    keyframe_dir = Path(args.keyframe_dir)  # e.g., output/{scene}/results
    aligned_dir = Path(args.gen3d_aligned_dir)  # e.g., output/{scene}/gen_3d_aligned
    output_dir = Path(args.output_dir)  # e.g., output/{scene}/gen_3d_aligned_omni

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get the last key frame index from {keyframe_dir}/key_frame_idx.txt
    print("=" * 50)
    print("Step 1: Loading keyframe indices...")
    print("=" * 50)

    keyframe_indices = load_keyframe_indices(keyframe_dir)
    if keyframe_indices is None or len(keyframe_indices) == 0:
        raise ValueError("No keyframe indices found")

    last_keyframe_idx = keyframe_indices[-1]
    print(f"[main] Last keyframe index: {last_keyframe_idx}")

    # Step 2: Load gen_3d info from {keyframe_dir}/gen_3d directory
    print("=" * 50)
    print("Step 2: Loading gen_3d info...")
    print("=" * 50)

    gen_3d_dir = keyframe_dir / "gen_3d"
    condition_image_path = gen_3d_dir / "image.png"
    if not condition_image_path.exists():
        raise FileNotFoundError(f"Condition image not found: {condition_image_path}")
    print(f"[main] Condition image: {condition_image_path}")

    # Step 3: Load aligned pose from {aligned_dir}/transform_refined.json
    print("=" * 50)
    print("Step 3: Loading aligned transform...")
    print("=" * 50)

    aligned_pose = load_aligned_transform(aligned_dir)
    if aligned_pose is None:
        raise ValueError("Failed to load aligned transform")

    # Compute inverse transform (world2obj = inv(obj2world))
    world2obj = np.linalg.inv(aligned_pose)
    print(f"[main] Computed world-to-object transform")

    # Step 4: Transform 3D points of the last key frame to object coordinate system
    print("=" * 50)
    print("Step 4: Loading and transforming 3D points...")
    print("=" * 50)

    image_info = load_image_info_from_results(keyframe_dir, last_keyframe_idx)
    if image_info is None:
        raise ValueError(f"Failed to load image_info for frame {last_keyframe_idx}")

    points_3d = image_info.get("points_3d")
    uncertainties = image_info.get("uncertainties")
    points_rgb = image_info.get("points_rgb")

    # Filter points by uncertainty threshold
    filtered_points, valid_mask = filter_points_by_uncertainty(
        points_3d, uncertainties, args.unc_thresh
    )

    if filtered_points is None or len(filtered_points) == 0:
        raise ValueError("No valid points after filtering")

    # Get colors for filtered points
    filtered_colors = None
    if points_rgb is not None:
        filtered_colors = points_rgb[valid_mask]

    # Transform points to object coordinates
    transformed_points = transform_points_to_object_coords(filtered_points, world2obj)
    print(f"[main] Transformed {len(transformed_points)} points to object coordinates")

    # Save transformed points as PLY
    transformed_ply_path = output_dir / "points_object_coords.ply"
    save_points_as_ply(transformed_points, transformed_ply_path, colors=filtered_colors)

    # Step 5: Run Hunyuan3D-Omni inference
    print("=" * 50)
    print("Step 5: Running Hunyuan3D-Omni inference...")
    print("=" * 50)

    hunyuan_dir = Path("/home/simba/Documents/project/Hunyuan3D-Omni")
    python_path = "/home/simba/miniconda3/envs/hunyuan_2.1_omni/bin/python"

    cmd = [
        python_path,
        "inference.py",
        "--control_type", "point",
        "--image_files", str(condition_image_path),
        "--mesh_files", str(transformed_ply_path),
        "--save_dir", str(output_dir),
    ]

    print(f"[main] Running command: {' '.join(cmd)}")
    print(f"[main] Working directory: {hunyuan_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=hunyuan_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[main] Hunyuan3D-Omni completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[main] Hunyuan3D-Omni failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    # Step 6: Transform the output mesh to world coordinates
    print("=" * 50)
    print("Step 6: Transforming output mesh to world coordinates...")
    print("=" * 50)

    # Try common output mesh names from Hunyuan3D-Omni

    input_mesh_path = output_dir / "white_mesh_remesh.obj"
    if input_mesh_path is not None:
        output_mesh_path = str(input_mesh_path).replace(".obj", "_world.obj")
        transform_mesh_to_world_coords(input_mesh_path, aligned_pose, output_mesh_path)
    else:
        print(f"[main] No output mesh found in {output_dir}, skipping world transform")

    print("=" * 50)
    print(f"[main] Results saved to {output_dir}")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D model using Hunyuan3D-Omni with aligned point cloud"
    )
    parser.add_argument(
        "--keyframe_dir",
        type=str,
        required=True,
        help="Directory containing keyframe results (e.g., output/{scene}/results)"
    )
    parser.add_argument(
        "--gen3d_aligned_dir",
        type=str,
        required=True,
        help="Directory containing aligned gen_3d results (e.g., output/{scene}/gen_3d_aligned)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Hunyuan3D-Omni results"
    )
    parser.add_argument(
        "--unc_thresh",
        type=float,
        default=2.0,
        help="Uncertainty threshold for filtering 3D points (points with uncertainty > threshold are excluded)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
