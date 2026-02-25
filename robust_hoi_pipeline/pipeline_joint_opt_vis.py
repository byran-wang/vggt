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
from utils_simba.rerun import log_mesh
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_preprocessed_frame
from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vertex_normals / norms


def load_hand_mesh(data_preprocess_dir: Path, frame_idx: int) -> Optional[trimesh.Trimesh]:
    """Load preprocessed hand mesh (camera space, scaled by 1/obj_scale)."""
    hand_path = data_preprocess_dir / "hand" / f"{frame_idx:04d}_right.obj"
    if not hand_path.exists():
        return None
    try:
        return trimesh.load(str(hand_path), force='mesh', process=False)
    except Exception as e:
        print(f"Warning: Failed to load hand mesh {hand_path}: {e}")
        return None


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
    result = {
        "tracks": image_info["tracks"][local_idx],
        "vis_scores": image_info["vis_scores"][local_idx],
        "tracks_mask": image_info["tracks_mask"][local_idx],
        "points_3d": image_info["points_3d"],
        "is_keyframe": image_info.get("keyframe", [False] * len(frame_indices))[local_idx],
        "is_register": image_info.get("register", [False] * len(frame_indices))[local_idx],
        "is_invalid": image_info.get("invalid", [False] * len(frame_indices))[local_idx],
        "c2o": image_info["c2o"][local_idx],
        "depth_points_obj": None,
    }
    depth_pts = image_info.get("depth_points_obj")
    if depth_pts is not None and local_idx < len(depth_pts):
        result["depth_points_obj"] = depth_pts[local_idx]
    return result


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




def visualize_gt_frame(
    frame_idx: int,
    gt_o2c: np.ndarray,
    preprocess_data: Dict,
):
    """Visualize GT camera pose and image for a single frame."""
    gt_frame_entity = "world/gt_current_frame"
    # Log GT camera pose (o2c inverted to c2o for transform)
    gt_c2o = np.linalg.inv(gt_o2c).astype(np.float32)
    rr.log(
        f"{gt_frame_entity}/camera",
        rr.Transform3D(
            translation=gt_c2o[:3, 3],
            mat3x3=gt_c2o[:3, :3],
        ),
    )
    if preprocess_data.get('intrinsics') is not None and preprocess_data.get('image') is not None:
        K = preprocess_data['intrinsics']
        H, W = preprocess_data['image'].shape[:2]
        rr.log(
            f"{gt_frame_entity}/camera",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[K[0, 0], K[1, 1]],
                principal_point=[K[0, 2], K[1, 2]],
                image_plane_distance=1.0,
            ),
            static=False,
        )
        rr.log(f"{gt_frame_entity}/camera", rr.Image(preprocess_data['image']), static=False)


def visualize_all_cameras(
    image_info_all: Dict,
    current_frame_idx: int,
    scale: float = 1.0,
):
    """Visualize camera frustums for all registered frames in image_info_all."""
    frame_indices = image_info_all.get("frame_indices", [])
    c2o_all = image_info_all["c2o"]  # (N, 4, 4), already scaled and aligned
    keyframe_flags = image_info_all.get("keyframe", [False] * len(frame_indices))
    register_flags = image_info_all.get("register", [False] * len(frame_indices))
    invalid_flags = image_info_all.get("invalid", [False] * len(frame_indices))
    intrinsics = image_info_all.get("intrinsics")

    for local_idx, fid in enumerate(frame_indices):
        is_kf = bool(keyframe_flags[local_idx])
        is_reg = bool(register_flags[local_idx])
        is_inv = bool(invalid_flags[local_idx])
        is_current = (fid == current_frame_idx)

        if not is_reg and not is_kf:
            continue

        c2o = c2o_all[local_idx]
        entity = f"world/all_cameras/{fid:04d}"

        # Color: cyan=current, green=keyframe, blue=registered, red=invalid
        if is_current:
            color = [0, 255, 255]
        elif is_inv:
            color = [255, 0, 0]
        elif is_kf:
            color = [0, 255, 0]
        else:
            color = [100, 100, 255]

        rr.log(
            entity,
            rr.Transform3D(
                translation=c2o[:3, 3],
                mat3x3=c2o[:3, :3],
            ),
        )

        if intrinsics is not None and intrinsics.ndim == 3:
            K = intrinsics[local_idx]
            rr.log(
                entity,
                rr.Pinhole(
                    resolution=[64, 48],  # small frustum
                    focal_length=[K[0, 0] * 64 / 640, K[1, 1] * 48 / 480],
                    principal_point=[32, 24],
                    image_plane_distance=0.05,
                ),
            )

        # Log a colored point at the camera position for visibility
        rr.log(
            f"{entity}/pos",
            rr.Points3D([c2o[:3, 3]], colors=[color], radii=0.1),
        )


def visualize_frame(
    frame_idx: int,
    preprocess_data: Dict,
    image_info: Optional[Dict],
    c2o: Optional[np.ndarray],
    scale: float = 1.0,
    track_vis_count: Optional[np.ndarray] = None,
    min_track_number: int = 4,
    align_pred_to_gt: Optional[np.ndarray] = None,
    hand_mesh: Optional[trimesh.Trimesh] = None,
    vis_hand: bool = True,
):

    frame_entity = "world/current_frame"

    # Log image with colored border based on frame type
    if preprocess_data['image'] is not None:
        img = preprocess_data['image'].copy()
        # Determine border color: green=keyframe, blue=register, gray=invalid
        if image_info is not None:
            border_width = 6
            if image_info.get("is_keyframe", False):
                color = (0, 255, 0)       # Green
            elif image_info.get("is_invalid", False):
                color = (160, 160, 160)    # Gray
            elif image_info.get("is_register", False):
                color = (0, 0, 255)        # Blue
            else:
                color = None
            if color is not None:
                img[:border_width, :] = color
                img[-border_width:, :] = color
                img[:, :border_width] = color
                img[:, -border_width:] = color
        rr.log(f"{frame_entity}/camera", rr.Image(img), static=False)

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
        points_3d = image_info['points_3d'] * scale  # (N, 3)
        vis_scores = image_info['vis_scores']  # (N,)

        # Log valid 3D points (finite + track mask)
        valid_3d_mask = np.isfinite(points_3d).all(axis=-1) # & tracks_mask.astype(bool)
        print(f"Frame {frame_idx}: {valid_3d_mask.sum()} valid 3D points out of {len(points_3d)}")
        if valid_3d_mask.any():
            valid_points_3d = points_3d[valid_3d_mask]
            if align_pred_to_gt is not None:
                valid_points_3d = (align_pred_to_gt[:3, :3] @ valid_points_3d.T).T + align_pred_to_gt[:3, 3]
            # Color by track visibility count: green = well-observed, red = poorly-observed
            colors_3d = np.zeros((len(valid_points_3d), 3))
            if track_vis_count is not None:
                well_observed = track_vis_count[valid_3d_mask] >= min_track_number
                colors_3d[well_observed, 1] = 1.0   # Green
                colors_3d[~well_observed, 0] = 1.0  # Red
            else:
                colors_3d[:, 1] = 1.0  # Default green

            rr.log(
                f"{frame_entity}/points_3d",
                rr.Points3D(valid_points_3d, colors=colors_3d, radii=0.003),
            )

        # Log depth points in object space (from depth-mesh alignment)
        depth_pts_obj = image_info.get('depth_points_obj')
        if depth_pts_obj is not None:
            depth_pts = np.asarray(depth_pts_obj, dtype=np.float32) * scale
            if align_pred_to_gt is not None:
                depth_pts = (align_pred_to_gt[:3, :3] @ depth_pts.T).T + align_pred_to_gt[:3, 3]
            rr.log(
                f"{frame_entity}/depth_points_obj",
                rr.Points3D(
                    depth_pts,
                    colors=np.broadcast_to(np.array([0, 200, 255], dtype=np.uint8), depth_pts.shape),
                    radii=0.0003,
                ),
            )
        else:
            rr.log(
                f"{frame_entity}/depth_points_obj",
                rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0),
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


    # Log hand mesh in object space
    if vis_hand and hand_mesh is not None and c2o is not None:
        hand_verts = np.array(hand_mesh.vertices, dtype=np.float32)
        hand_faces = np.array(hand_mesh.faces, dtype=np.int32)
        # c2o translation is already scaled; scale vertices to match
        hand_verts_obj = (c2o[:3, :3] @ hand_verts.T).T * scale + c2o[:3, 3]
        vertex_normals = compute_vertex_normals(hand_verts_obj, hand_faces)
        rr.log(
            f"{frame_entity}/hand_mesh",
            rr.Mesh3D(
                vertex_positions=hand_verts_obj,
                triangle_indices=hand_faces,
                vertex_normals=vertex_normals,
                mesh_material=rr.Material(albedo_factor=[200, 180, 220]),
            ),
            static=False,
        )
    else:
        rr.log(
            f"{frame_entity}/hand_mesh",
            rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0),
        )

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
                image_plane_distance=1.0,
            ),
            static=False,
        )
        rr.log(
            f"{frame_entity}/camera",
            rr.Transform3D(
                translation=c2o[:3, 3],
                mat3x3=c2o[:3, :3],
            ),
            static=False,
        )

    # # Mark keyframes
    # if image_info is not None and image_info.get("is_keyframe", False):
    #     rr.log(f"{frame_entity}/keyframe", rr.TextLog(f"KEYFRAME {frame_idx}"))


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = data_dir / "pipeline_preprocess"
    tracks_dir = data_dir / "pipeline_corres"
    results_dir = out_dir / "pipeline_joint_opt"

    # Initialize Rerun
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera", origin="world/current_frame/camera"),
                rrb.Spatial2DView(name="Reproj Error", origin="reproj_error"),
            ),
            column_shares=[2, 1],
        ),
    )
    rr.init("pipeline_joint_opt_vis", spawn=True, default_blueprint=blueprint)
    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load frame list
    print("Loading frame list...")
    # frame_indices = load_frame_list(data_preprocess_dir)
    frame_indices = load_register_indices(results_dir)
    print(f"Found {len(frame_indices)} frames")

    # Load and visualize SAM3D mesh
    print("Loading SAM3D mesh...")
    SAM3D_mesh = load_sam3d_mesh(SAM3D_dir, cond_idx)
    sam3d_transform = load_sam3d_transform(SAM3D_dir, args.cond_index)
    sam3d_to_cond_cam = sam3d_transform['sam3d_to_cond_cam']
    scale = sam3d_transform['scale']

    # Load GT data if requested
    data_gt = None
    if args.vis_gt:
        print("Loading GT data...")
        import vggt.utils.gt as gt
        seq_name = out_dir.name
        def get_image_fids():
            return frame_indices
        data_gt = gt.load_data(seq_name, get_image_fids)
        # Convert to numpy for visualization
        gt_o2c = data_gt["o2c"].numpy()
        
        gt_is_valid = data_gt["is_valid"].numpy()
        print(f"Loaded GT data: {len(gt_o2c)} frames")
        gt_mesh_path = data_gt["mesh_name.object"]
        log_mesh("world/gt_mesh", gt_mesh_path, static=True)


    # Compute alignment transform from pred object space to GT object space
    # using the condition frame as the reference

    align_pred_to_gt = None

    if data_gt is not None and len(gt_o2c) > 0:
        image_info = load_image_info(results_dir / f"{frame_indices[-1]:04d}")
        pred_frame_indices = image_info["frame_indices"]
        cond_local = pred_frame_indices.index(cond_idx)
        gt_cond_idx = frame_indices.index(cond_idx)        
        first_c2o = np.array(image_info["c2o"])
        first_c2o[:, :3, 3] *= scale
        align_pred_to_gt = np.linalg.inv(gt_o2c[gt_cond_idx]) @ np.linalg.inv(first_c2o[cond_local])

    if SAM3D_mesh is not None:
        vertices = np.array(SAM3D_mesh.vertices, dtype=np.float32) * scale
        faces = np.array(SAM3D_mesh.faces, dtype=np.uint32)

        # Align SAM3D mesh to GT space if alignment is available
        if align_pred_to_gt is not None:
            verts_homo = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float32)])
            vertices = (align_pred_to_gt @ verts_homo.T).T[:, :3].astype(np.float32)

        # Get vertex colors if available
        if SAM3D_mesh.visual is not None and hasattr(SAM3D_mesh.visual, 'vertex_colors'):
            vertex_colors = np.array(SAM3D_mesh.visual.vertex_colors)[:, :3] / 255.0
        else:
            vertex_colors = np.ones((len(vertices), 3)) * 0.7  # Gray

        # rr.log(
        #     "world/sam3d/mesh",
        #     rr.Mesh3D(
        #         vertex_positions=vertices,
        #         triangle_indices=faces,
        #         vertex_colors=vertex_colors,
        #     ),
        #     static=True,
        # )
        rr.log(
            "world/sam3d/points",
            rr.Points3D(vertices, colors=np.broadcast_to(np.array([255, 255, 0], dtype=np.uint8), vertices.shape), radii=0.0003),
            static=True,
        )

    # Load HY omni mesh (in SAM3D space) from pipeline_HY_to_SAM3D
    hy_omni_path = out_dir / "pipeline_HY_to_SAM3D" / "HY_omni_in_sam3d.obj"
    if hy_omni_path.exists():
        hy_omni_mesh = trimesh.load(str(hy_omni_path), process=False)
        hy_verts = np.array(hy_omni_mesh.vertices, dtype=np.float32) * scale
        hy_faces = np.array(hy_omni_mesh.faces, dtype=np.uint32)

        if align_pred_to_gt is not None:
            verts_homo = np.hstack([hy_verts, np.ones((len(hy_verts), 1), dtype=np.float32)])
            hy_verts = (align_pred_to_gt @ verts_homo.T).T[:, :3].astype(np.float32)

        if hy_omni_mesh.visual is not None and hasattr(hy_omni_mesh.visual, 'vertex_colors'):
            hy_colors = np.array(hy_omni_mesh.visual.vertex_colors)[:, :3] / 255.0
        else:
            hy_colors = np.ones((len(hy_verts), 3)) * 0.5  # Dark gray

        # rr.log(
        #     "world/hy_omni/mesh",
        #     rr.Mesh3D(
        #         vertex_positions=hy_verts,
        #         triangle_indices=hy_faces,
        #         vertex_colors=hy_colors,
        #     ),
        #     static=True,
        # )

        rr.log(
            "world/hy_omni/points",
            rr.Points3D(hy_verts, colors=np.broadcast_to(np.array([0, 0, 255], dtype=np.uint8), hy_verts.shape), radii=0.0003),
            static=True,
        )
        print(f"Visualized HY omni mesh: {len(hy_verts)} vertices")
    else:
        print(f"HY omni mesh not found at {hy_omni_path}, skipping")

    # Visualize only keyframes
    print("Visualizing keyframes...")

    for i, frame_idx in enumerate(frame_indices):
        # Load image info (from joint opt)
        image_info_all = load_image_info(results_dir / f"{frame_idx:04d}")
        if image_info_all is None:
            print(f"Failed to load image info for frame {frame_idx}")
            continue
        image_info_all["c2o"][:, :3, 3] *= scale
        if align_pred_to_gt is not None:
            image_info_all["c2o"] = (align_pred_to_gt @ image_info_all["c2o"]).astype(np.float32)
        if image_info_all is None:
            continue
        image_info = get_frame_image_info(image_info_all, frame_idx)

        # Compute per-track visibility count across keyframes
        kf_flags = np.array(image_info_all.get("keyframe", [False] * len(image_info_all["frame_indices"])))
        kf_track_mask = np.array(image_info_all["tracks_mask"])[kf_flags].astype(bool)
        track_vis_count = kf_track_mask.sum(axis=0) if len(kf_track_mask) > 0 else np.zeros(len(image_info_all["points_3d"]))
        if args.vis_type == "registered_valid" and image_info.get("is_invalid", False):
            print(f"Skipping invalid frame {frame_idx} in registered_valid mode")
            continue
        if args.vis_type == "keyframes" and image_info.get("is_invalid", False) and not image_info.get("is_keyframe", False):
            print(f"Skipping non-keyframe {frame_idx} in keyframes mode")
            continue

        preprocess_data = load_preprocessed_frame(data_preprocess_dir, frame_idx)
        hand_mesh = load_hand_mesh(data_preprocess_dir, frame_idx)
        # Visualize
        c2o = image_info['c2o']

        rr.set_time_sequence("frame", i)
        visualize_frame(
            frame_idx=frame_idx,
            preprocess_data=preprocess_data,
            image_info=image_info,
            c2o=c2o,
            scale=scale,
            track_vis_count=track_vis_count,
            min_track_number=args.min_track_number,
            align_pred_to_gt=align_pred_to_gt,
            hand_mesh=hand_mesh,
            vis_hand=args.vis_hand,
        )

        # Visualize all camera poses
        if args.vis_all_cameras:
            visualize_all_cameras(image_info_all, frame_idx, scale=scale)

        # Visualize GT camera pose
        if data_gt is not None and i < len(gt_o2c) and bool(gt_is_valid[i]):
            visualize_gt_frame(frame_idx, gt_o2c[i], preprocess_data)

        # Log reprojection error image if available
        reproj_img_path = results_dir / f"{frame_idx:04d}" / "reproj_error.png"
        if reproj_img_path.exists():
            from PIL import Image
            reproj_img = np.array(Image.open(reproj_img_path).convert("RGB"))
            rr.log("reproj_error", rr.Image(reproj_img), static=False)



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
    parser.add_argument("--vis_type", type=str, default="all", choices=["all", "registered_valid", "keyframes"],
                        help="Type of frames to visualize")
    parser.add_argument("--vis_gt", action="store_true", default=True,
                        help="Visualize ground truth mesh and camera poses")
    parser.add_argument("--min_track_number", type=int, default=4,
                        help="Minimum track visibility count for green coloring")
    parser.add_argument("--vis_all_cameras", action="store_true", default=True,
                        help="Visualize all camera poses from image_info, not just the current frame")
    parser.add_argument("--vis_hand", action="store_true", default=False,
                        help="Visualize hand mesh in object space")

    args = parser.parse_args()
    main(args)
