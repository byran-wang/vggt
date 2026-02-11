import json
import numpy as np
import trimesh
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robust_hoi_pipeline.frame_management import load_register_indices


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    result_dir = out_dir / "pipeline_3D_points_align_with_HY"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load the 3D points of the latest keyframe from out_dir/pipeline_joint_opt
    joint_opt_dir = out_dir / "pipeline_joint_opt"
    register_indices = load_register_indices(joint_opt_dir)
    last_register_idx = register_indices[-1]
    image_info_path = joint_opt_dir / f"{last_register_idx:04d}" / "image_info.npy"
    image_info = np.load(image_info_path, allow_pickle=True).item()
    print(f"Loaded image_info from {image_info_path}")

    points_3d = np.array(image_info["points_3d"], dtype=np.float32)  # (M, 3)
    tracks_mask = np.array(image_info["tracks_mask"], dtype=bool)    # (S, M)
    keyframe_flags = np.array(image_info["keyframe"], dtype=bool)    # (S,)

    # Get valid 3D points with track number >= min_track_num
    finite_mask = np.isfinite(points_3d).all(axis=-1)  # (M,)
    track_vis_in_kf = tracks_mask[keyframe_flags].sum(axis=0)  # (M,)
    valid_mask = finite_mask & (track_vis_in_kf >= args.min_track_num)
    valid_points = points_3d[valid_mask]
    print(f"Valid 3D points: {valid_mask.sum()} / {len(points_3d)} "
          f"(min_track_num={args.min_track_num})")

    # Load the transformation from SAM3D to Hunyuan
    transform_path = out_dir / "pipeline_algin_SAM3D_with_HY" / "SAM3D_aligned_with_HY.json"
    with open(transform_path, "r") as f:
        transform_params = json.load(f)
    transform_4x4 = np.array(transform_params["transform_4x4"], dtype=np.float64)
    print(f"Loaded SAM3D-to-Hunyuan transform from {transform_path}")
    print(f"  scale={transform_params['scale']}, rotation_y_deg={transform_params['rotation_y_deg']}, "
          f"translation={transform_params['translation']}")

    # Apply the transformation to align 3D points to Hunyuan coordinate system
    aligned_points = (transform_4x4[:3, :3] @ valid_points.T).T + transform_4x4[:3, 3]
    aligned_points = aligned_points.astype(np.float32)

    # Save aligned 3D points
    aligned_ply_path = result_dir / "points_aligned.ply"
    trimesh.PointCloud(aligned_points).export(str(aligned_ply_path))
    print(f"Saved {len(aligned_points)} aligned points to {aligned_ply_path}")

    # Save original valid 3D points before alignment
    original_ply_path = result_dir / "points_original.ply"
    trimesh.PointCloud(valid_points).export(str(original_ply_path))
    print(f"Saved {len(valid_points)} original points to {original_ply_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Align 3D track points from joint optimization to Hunyuan coordinate system")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/GSF13)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--min_track_num", type=int, default=4,
                        help="Minimum number of keyframe observations for a 3D point to be valid")
    args = parser.parse_args()
    main(args)
