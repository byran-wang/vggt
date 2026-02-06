import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rerun as rr
import trimesh

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import get_depth, get_normal, depth2xyzmap


def load_frame_list(data_preprocess_dir: Path) -> List[int]:
    """Load frame list from preprocessed data directory."""
    frame_list_path = data_preprocess_dir / "frame_list.txt"
    if not frame_list_path.exists():
        raise FileNotFoundError(f"Frame list not found: {frame_list_path}")

    frames = []
    with open(frame_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(int(line))
    return frames


def load_preprocessed_frame(data_preprocess_dir: Path, frame_idx: int) -> Dict:
    """Load preprocessed data for a single frame."""
    from PIL import Image

    data = {}

    # Load RGB image
    rgb_path = data_preprocess_dir / "rgb" / f"{frame_idx:04d}.png"
    if rgb_path.exists():
        data['image'] = np.array(Image.open(rgb_path).convert("RGB"))
    else:
        data['image'] = None

    # Load object mask
    mask_obj_path = data_preprocess_dir / "mask_obj" / f"{frame_idx:04d}.png"
    if mask_obj_path.exists():
        data['mask_obj'] = np.array(Image.open(mask_obj_path).convert("L"))
    else:
        data['mask_obj'] = None

    # Load hand mask
    mask_hand_path = data_preprocess_dir / "mask_hand" / f"{frame_idx:04d}.png"
    if mask_hand_path.exists():
        data['mask_hand'] = np.array(Image.open(mask_hand_path).convert("L"))
    else:
        data['mask_hand'] = None

    # Load filtered depth
    depth_path = data_preprocess_dir / "depth_filtered" / f"{frame_idx:04d}.png"
    if depth_path.exists():
        data['depth'] = get_depth(str(depth_path))
    else:
        data['depth'] = None

    # Load normal map
    normal_path = data_preprocess_dir / "normal" / f"{frame_idx:04d}.png"
    if normal_path.exists():
        data['normal'] = get_normal(str(normal_path))
    else:
        data['normal'] = None

    # Load metadata (intrinsics + hand pose)
    meta_path = data_preprocess_dir / "meta" / f"{frame_idx:04d}.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        data['intrinsics'] = meta.get('intrinsics')
        data['hand_pose'] = meta.get('hand_pose')
    else:
        data['intrinsics'] = None
        data['hand_pose'] = None

    return data


def load_image_info(results_dir: Path, frame_idx: int) -> Optional[Dict]:
    """Load image info from pipeline_joint_opt.py output."""
    info_path = results_dir / f"{frame_idx:04d}" / "image_info.npy"
    if not info_path.exists():
        return None
    return np.load(info_path, allow_pickle=True).item()


def load_summary(results_dir: Path) -> Dict:
    """Load summary from pipeline_joint_opt.py output."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with open(summary_path, "r") as f:
        return json.load(f)


def load_sam3d_mesh(sam3d_dir: Path, cond_idx: int) -> Optional[trimesh.Trimesh]:
    """Load SAM3D mesh."""
    # Try different mesh file names
    mesh_names = ["mesh.obj", "mesh_aligned.obj", "textured.obj"]
    mesh_dir = sam3d_dir / f"{cond_idx:04d}"

    for mesh_name in mesh_names:
        mesh_path = mesh_dir / mesh_name
        if mesh_path.exists():
            print(f"Loading SAM3D mesh from {mesh_path}")
            return trimesh.load(str(mesh_path), force='mesh')

    print(f"Warning: No SAM3D mesh found in {mesh_dir}")
    return None


def load_sam3d_transform(sam3d_dir: Path, cond_idx: int) -> Optional[Dict]:
    """Load SAM3D transformation."""
    transform_path = sam3d_dir / f"{cond_idx:04d}" / "aligned_transform.json"
    if not transform_path.exists():
        return None

    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    rotation = np.array(transform_data["rotation"], dtype=np.float32)
    translation = np.array(transform_data["translation"], dtype=np.float32)
    scale = float(transform_data["scale"])

    obj2cam = np.eye(4, dtype=np.float32)
    obj2cam[:3, :3] = scale * rotation
    obj2cam[:3, 3] = translation

    cam2obj = np.linalg.inv(obj2cam).astype(np.float32)

    return {
        'scale': scale,
        'rotation': rotation,
        'translation': translation,
        'obj2cam': obj2cam,
        'cam2obj': cam2obj,
    }


def visualize_frame(
    frame_idx: int,
    preprocess_data: Dict,
    image_info: Optional[Dict],
    sam3d_transform: Optional[Dict],
    is_keyframe: bool,
):
    """Visualize a single frame in Rerun."""
    rr.set_time_sequence("frame", frame_idx)

    frame_entity = f"frames/{frame_idx:04d}"

    # Log image
    if preprocess_data['image'] is not None:
        rr.log(f"{frame_entity}/image", rr.Image(preprocess_data['image']))

    # Log object mask
    if preprocess_data['mask_obj'] is not None:
        rr.log(f"{frame_entity}/mask_obj", rr.Image(preprocess_data['mask_obj']))

    # Log hand mask
    if preprocess_data['mask_hand'] is not None:
        rr.log(f"{frame_entity}/mask_hand", rr.Image(preprocess_data['mask_hand']))

    # Log depth as image
    if preprocess_data['depth'] is not None:
        depth = preprocess_data['depth']
        # Normalize for visualization
        valid_mask = depth > 0
        if valid_mask.any():
            depth_vis = np.zeros_like(depth)
            depth_vis[valid_mask] = depth[valid_mask]
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            if depth_max > depth_min:
                depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
            rr.log(f"{frame_entity}/depth", rr.Image(depth_vis))

    # Log depth point cloud in world/object space
    if preprocess_data['depth'] is not None and preprocess_data['intrinsics'] is not None:
        depth = preprocess_data['depth']
        K = preprocess_data['intrinsics']
        xyz_map = depth2xyzmap(depth, K)

        valid_mask = depth > 0
        points_cam = xyz_map[valid_mask]

        # Transform to object space if we have the transform
        if sam3d_transform is not None:
            cam2obj = sam3d_transform['cam2obj']
            points_homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
            points_obj = (cam2obj @ points_homo.T).T[:, :3]
        else:
            points_obj = points_cam

        # Get colors from image
        if preprocess_data['image'] is not None:
            colors = preprocess_data['image'][valid_mask] / 255.0
        else:
            colors = np.ones((len(points_obj), 3)) * 0.5

        rr.log(
            f"{frame_entity}/depth_pointcloud",
            rr.Points3D(points_obj, colors=colors, radii=0.001),
        )

    # Log tracks and 3D points from image_info
    if image_info is not None:
        tracks = image_info['tracks']  # (N, 2)
        tracks_mask = image_info['tracks_mask']  # (N,)
        points_3d = image_info['points_3d']  # (N, 3)
        vis_scores = image_info['vis_scores']  # (N,)

        # Valid tracks (in mask)
        valid_2d = tracks_mask
        if valid_2d.any():
            valid_tracks = tracks[valid_2d]
            rr.log(
                f"{frame_entity}/tracks_valid",
                rr.Points2D(
                    valid_tracks,
                    colors=np.array([[0, 255, 0]]),  # Green for valid
                    radii=3.0,
                ),
            )

        # Invalid tracks (not in mask)
        invalid_2d = ~tracks_mask
        if invalid_2d.any():
            invalid_tracks = tracks[invalid_2d]
            rr.log(
                f"{frame_entity}/tracks_invalid",
                rr.Points2D(
                    invalid_tracks,
                    colors=np.array([[255, 0, 0]]),  # Red for invalid
                    radii=2.0,
                ),
            )

        # 3D points (only those with valid depth)
        valid_3d_mask = np.isfinite(points_3d).all(axis=-1) & tracks_mask
        if valid_3d_mask.any():
            valid_points_3d = points_3d[valid_3d_mask]
            # Color by visibility score
            vis = vis_scores[valid_3d_mask]
            colors_3d = np.zeros((len(valid_points_3d), 3))
            colors_3d[:, 1] = vis  # Green channel = visibility
            colors_3d[:, 0] = 1 - vis  # Red channel = 1 - visibility

            rr.log(
                f"{frame_entity}/points_3d",
                rr.Points3D(valid_points_3d, colors=colors_3d, radii=0.003),
            )

    # Log camera pose
    if preprocess_data['intrinsics'] is not None and image_info is not None:
        K = preprocess_data['intrinsics']
        o2c = image_info['o2c']  # (4, 4) object-to-camera

        # Camera pose is the inverse of o2c (camera-to-object)
        c2o = np.linalg.inv(o2c)

        # Extract rotation and translation
        rotation = c2o[:3, :3]
        translation = c2o[:3, 3]

        # Log camera
        H, W = preprocess_data['image'].shape[:2] if preprocess_data['image'] is not None else (480, 640)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        rr.log(
            f"{frame_entity}/camera",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
            ),
        )
        rr.log(
            f"{frame_entity}/camera",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation,
            ),
        )

    # Mark keyframes
    if is_keyframe:
        rr.log(f"{frame_entity}/keyframe", rr.TextLog(f"KEYFRAME {frame_idx}"))


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir / "pipeline_preprocess"
    tracks_dir = out_dir / "pipeline_corres"
    results_dir = out_dir / "results"

    # Initialize Rerun
    rr.init("pipeline_joint_opt_vis", spawn=True)

    # Load frame list
    print("Loading frame list...")
    frame_indices = load_frame_list(data_preprocess_dir)
    print(f"Found {len(frame_indices)} frames")

    # Load summary to get keyframe info
    print("Loading summary...")
    try:
        summary = load_summary(results_dir)
        keyframe_indices = set(summary.get('keyframe_indices', [cond_idx]))
    except FileNotFoundError:
        print("Warning: Summary not found, using cond_idx as only keyframe")
        keyframe_indices = {cond_idx}

    # Load SAM3D transform
    print("Loading SAM3D transform...")
    sam3d_transform = load_sam3d_transform(SAM3D_dir, cond_idx)
    if sam3d_transform is not None:
        print(f"SAM3D scale: {sam3d_transform['scale']}")

    # Load and visualize SAM3D mesh
    print("Loading SAM3D mesh...")
    mesh = load_sam3d_mesh(SAM3D_dir, cond_idx)
    if mesh is not None:
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.uint32)

        # Get vertex colors if available
        if mesh.visual is not None and hasattr(mesh.visual, 'vertex_colors'):
            vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
        else:
            vertex_colors = np.ones((len(vertices), 3)) * 0.7  # Gray

        rr.log(
            "world/sam3d_mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=vertex_colors,
            ),
            static=True,
        )

    # Visualize each frame
    print("Visualizing frames...")
    max_frames = args.max_frames if args.max_frames > 0 else len(frame_indices)

    for i, frame_idx in enumerate(frame_indices[:max_frames]):
        print(f"Processing frame {frame_idx} ({i+1}/{min(max_frames, len(frame_indices))})")

        # Load preprocessed data
        preprocess_data = load_preprocessed_frame(data_preprocess_dir, frame_idx)

        # Load image info (from joint opt)
        image_info = load_image_info(results_dir, frame_idx)

        # Check if keyframe
        is_keyframe = frame_idx in keyframe_indices

        # Visualize
        visualize_frame(
            frame_idx=frame_idx,
            preprocess_data=preprocess_data,
            image_info=image_info,
            sam3d_transform=sam3d_transform,
            is_keyframe=is_keyframe,
        )

    print("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize joint optimization results with Rerun")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory containing pipeline results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Maximum number of frames to visualize (-1 for all)")

    args = parser.parse_args()
    main(args)
