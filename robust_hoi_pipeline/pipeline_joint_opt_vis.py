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

from utils_simba.depth import depth2xyzmap
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_preprocessed_frame
from robust_hoi_pipeline.frame_management import load_keyframe_indices


def load_image_info(results_dir: Path) -> Optional[Dict]:
    """Load image info from pipeline_joint_opt.py output."""
    info_path = results_dir / "image_info.npy"
    if not info_path.exists():
        return None
    return np.load(info_path, allow_pickle=True).item()


def get_frame_image_info(image_info: Dict, frame_idx: int) -> Optional[Dict]:
    """Extract per-frame data from the aggregated image_info dict."""
    if image_info is None:
        return None
    frame_indices = image_info.get("frame_indices")
    if frame_indices is None:
        return None
    try:
        local_idx = frame_indices.index(frame_idx)
    except ValueError:
        return None
    return {
        "tracks": image_info["tracks"][local_idx],
        "vis_scores": image_info["vis_scores"][local_idx],
        "tracks_mask": image_info["tracks_mask"][local_idx],
        "points_3d": image_info["points_3d"],
        "is_keyframe": image_info.get("keyframe", [False] * len(frame_indices))[local_idx],
        "is_register": image_info.get("register", [False] * len(frame_indices))[local_idx],
        "is_invalid": image_info.get("invalid", [False] * len(frame_indices))[local_idx],
        "c2o": image_info["c2o"][local_idx],
    }


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




def visualize_frame(
    frame_idx: int,
    preprocess_data: Dict,
    image_info: Optional[Dict],
    c2o: Optional[np.ndarray],
):
    """Visualize a single frame in Rerun."""
    rr.set_time_sequence("frame", frame_idx)

    frame_entity = f"world/frames/{frame_idx:04d}"

    # Log image
    if preprocess_data['image'] is not None:
        rr.log(f"{frame_entity}/camera", rr.Image(preprocess_data['image']))

    # # Log object mask
    # if preprocess_data['mask_obj'] is not None:
    #     rr.log(f"{frame_entity}/mask_obj", rr.Image(preprocess_data['mask_obj']))

    # # Log hand mask
    # if preprocess_data['mask_hand'] is not None:
    #     rr.log(f"{frame_entity}/mask_hand", rr.Image(preprocess_data['mask_hand']))

    # # Log depth as image
    # if preprocess_data['depth'] is not None:
    #     depth = preprocess_data['depth']
    #     # Normalize for visualization
    #     valid_mask = depth > 0
    #     if valid_mask.any():
    #         depth_vis = np.zeros_like(depth)
    #         depth_vis[valid_mask] = depth[valid_mask]
    #         depth_min = depth[valid_mask].min()
    #         depth_max = depth[valid_mask].max()
    #         if depth_max > depth_min:
    #             depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
    #         rr.log(f"{frame_entity}/depth", rr.Image(depth_vis))



    # Log tracks and 3D points from image_info
    if image_info is not None:
        tracks = image_info['tracks']  # (N, 2)
        tracks_mask = image_info['tracks_mask']  # (N,)
        points_3d = image_info['points_3d']  # (N, 3)
        vis_scores = image_info['vis_scores']  # (N,)

        # Log valid 3D points (finite + track mask)
        valid_3d_mask = np.isfinite(points_3d).all(axis=-1)
        if valid_3d_mask.any():
            valid_points_3d = points_3d[valid_3d_mask]
            vis = vis_scores[valid_3d_mask]
            colors_3d = np.zeros((len(valid_points_3d), 3))
            colors_3d[:, 1] = vis  # Green channel = visibility
            colors_3d[:, 0] = 1 - vis  # Red channel = 1 - visibility

            rr.log(
                f"{frame_entity}/points_3d",
                rr.Points3D(valid_points_3d, colors=colors_3d, radii=0.003),
            )

        # # Valid tracks (in mask)
        # valid_2d = tracks_mask
        # if valid_2d.any():
        #     valid_tracks = tracks[valid_2d]
        #     rr.log(
        #         f"{frame_entity}/tracks_valid",
        #         rr.Points2D(
        #             valid_tracks,
        #             colors=np.array([[0, 255, 0]]),  # Green for valid
        #             radii=3.0,
        #         ),
        #     )

        # # Invalid tracks (not in mask)
        # invalid_2d = ~tracks_mask
        # if invalid_2d.any():
        #     invalid_tracks = tracks[invalid_2d]
        #     rr.log(
        #         f"{frame_entity}/tracks_invalid",
        #         rr.Points2D(
        #             invalid_tracks,
        #             colors=np.array([[255, 0, 0]]),  # Red for invalid
        #             radii=2.0,
        #         ),
        #     )


    # Log camera pose and intrinsics
    if preprocess_data.get('intrinsics') is not None and preprocess_data.get('image') is not None and c2o is not None:
        K = preprocess_data['intrinsics']
        H, W = preprocess_data['image'].shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        rr.log(
            f"{frame_entity}/camera",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                image_plane_distance=0.2,
            ),
        )
        rr.log(
            f"{frame_entity}/camera",
            rr.Transform3D(
                translation=c2o[:3, 3],
                mat3x3=c2o[:3, :3],
            ),
        )

    # # Mark keyframes
    # if image_info is not None and image_info.get("is_keyframe", False):
    #     rr.log(f"{frame_entity}/keyframe", rr.TextLog(f"KEYFRAME {frame_idx}"))


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir / "pipeline_preprocess"
    tracks_dir = out_dir / "pipeline_corres"
    results_dir = out_dir / "pipeline_joint_opt"

    # Initialize Rerun
    rr.init("pipeline_joint_opt_vis", spawn=True)
        # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load frame list
    print("Loading frame list...")
    # frame_indices = load_frame_list(data_preprocess_dir)
    frame_indices = load_keyframe_indices(results_dir)
    breakpoint()
    print(f"Found {len(frame_indices)} frames")

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

    # Visualize only keyframes
    print("Visualizing keyframes...")


    for i, frame_idx in enumerate(frame_indices):
        # Load image info (from joint opt)
        image_info_all = load_image_info(results_dir / f"{frame_idx:04d}")
        if image_info_all is None:
            continue
        image_info = get_frame_image_info(image_info_all, frame_idx)

        if image_info.get("is_keyframe", False) is False:
            continue
        # Load preprocessed data
        preprocess_data = load_preprocessed_frame(data_preprocess_dir, frame_idx)
        # Visualize
        c2o = image_info['c2o']

        visualize_frame(
            frame_idx=frame_idx,
            preprocess_data=preprocess_data,
            image_info=image_info,
            c2o=c2o,
        )



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
