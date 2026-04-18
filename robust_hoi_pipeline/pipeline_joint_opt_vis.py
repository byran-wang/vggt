import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rerun as rr
import trimesh

import re
import cv2

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import depth2xyzmap
from utils_simba.rerun import log_mesh, log_camera_frame
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_preprocessed_frame
from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform


def parse_frame_logs(log_path: Path) -> Dict[int, str]:
    """Parse log.txt into per-frame log blocks.

    Each block starts with 'Next frame to register: {fid} (local idx {idx})'
    and ends with '+++...+++'. Returns {frame_id: log_text}.
    """
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    pattern = re.compile(
        r"Next frame to register: (\d+) \(local idx \d+\)"
        r"(.*?)"
        r"(?=\+{10,}|\Z)",
        re.DOTALL,
    )
    result = {}
    for m in pattern.finditer(text):
        fid = int(m.group(1))
        block = m.group(0).strip()
        result[fid] = block
    return result


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
    """Load image info from pipeline_joint_opt.py output.

    Supports compressed format (.pkl.gz), split format (shared + per-frame),
    and legacy single-file .npy format for backwards compatibility.
    """
    import gzip
    import pickle

    # Try compressed format first
    gz_path = results_dir / "image_info.pkl.gz"
    npy_path = results_dir / "image_info.npy"

    if gz_path.exists():
        with gzip.open(gz_path, "rb") as f:
            per_frame = pickle.load(f)
    elif npy_path.exists():
        per_frame = np.load(npy_path, allow_pickle=True).item()
    else:
        return None

    # Try loading shared static data
    shared_gz = results_dir.parent / "shared_info.pkl.gz"
    shared_npy = results_dir.parent / "shared_info.npy"
    if shared_gz.exists():
        with gzip.open(shared_gz, "rb") as f:
            shared = pickle.load(f)
        shared.update(per_frame)
        return shared
    elif shared_npy.exists():
        shared = np.load(shared_npy, allow_pickle=True).item()
        shared.update(per_frame)
        return shared

    # Legacy: single file contains everything
    return per_frame


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
        "points_3d": image_info["points_3d"],
        "track_vis_count": image_info.get("track_vis_count"),
        "is_keyframe": image_info.get("keyframe", [False] * len(frame_indices))[local_idx],
        "is_register": image_info.get("register", [False] * len(frame_indices))[local_idx],
        "is_invalid": image_info.get("invalid", [False] * len(frame_indices))[local_idx],
        "c2o": image_info["c2o"][local_idx],
        "depth_after_PnP": image_info.get("depth_after_PnP", [None] * len(frame_indices))[local_idx],
        "depth_after_align_mesh": image_info.get("depth_after_align_mesh", [None] * len(frame_indices))[local_idx],
        "depth_after_keyframes_opt": image_info.get("depth_after_keyframes_opt", [None] * len(frame_indices))[local_idx],
        "depth_after_reset_when_pnp_fail": image_info.get("depth_after_reset_when_pnp_fail", [None] * len(frame_indices))[local_idx],
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


def log_neus_mesh_from_dir(
    save_dir: Path,
    entity_path: str,
    scale: float,
    align_transform: Optional[np.ndarray] = None,
) -> None:
    """Find the latest .obj under save_dir and log it to rerun as a non-static mesh."""
    if not save_dir.exists():
        return
    mesh_files = list(save_dir.rglob("*.obj"))
    if not mesh_files:
        return
    latest_mesh_file = max(mesh_files, key=lambda p: p.stat().st_mtime)
    neus_mesh = trimesh.load(str(latest_mesh_file), force="mesh")
    mesh_colors = None
    if hasattr(neus_mesh.visual, "vertex_colors") and neus_mesh.visual.vertex_colors is not None:
        mesh_colors = np.asarray(neus_mesh.visual.vertex_colors)[:, :3]
    vertices = np.array(neus_mesh.vertices, dtype=np.float32) * scale
    if align_transform is not None:
        verts_homo = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float32)])
        vertices = (align_transform @ verts_homo.T).T[:, :3].astype(np.float32)
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=neus_mesh.faces,
            vertex_normals=neus_mesh.vertex_normals if neus_mesh.vertex_normals is not None else None,
            # vertex_colors=mesh_colors,
        ),
        static=False,
    )



def log_static_trimesh_mesh(
    entity_path: str,
    mesh: trimesh.Trimesh,
    scale: float,
    align_transform: Optional[np.ndarray] = None,
    default_color: float = 0.7,
) -> None:
    """Log a trimesh mesh to rerun with optional scaling and alignment."""
    vertices = np.array(mesh.vertices, dtype=np.float32) * scale
    faces = np.array(mesh.faces, dtype=np.uint32)

    if align_transform is not None:
        verts_homo = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float32)])
        vertices = (align_transform @ verts_homo.T).T[:, :3].astype(np.float32)

    if mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
        vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        vertex_colors = np.ones((len(vertices), 3), dtype=np.float32) * default_color

    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
            vertex_colors=vertex_colors,
        ),
        static=True,
    )

    # rr.log(
    #     "world/sam3d/points",
    #     rr.Points3D(vertices, colors=np.broadcast_to(np.array([255, 255, 0], dtype=np.uint8), vertices.shape), radii=0.0003),
    #     static=True,
    # )    


def find_latest_mesh_file(mesh_dir: Path) -> Optional[Path]:
    """Return the newest OBJ mesh under mesh_dir."""
    if not mesh_dir.exists():
        return None

    mesh_files = sorted(mesh_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mesh_files[0] if mesh_files else None


def visualize_hy_omni_points(
    hy_omni_path: Path,
    scale: float,
    align_transform: Optional[np.ndarray] = None,
) -> None:
    """Load and visualize the HY omni mesh as points in rerun."""
    if not hy_omni_path.exists():
        print(f"HY omni mesh not found at {hy_omni_path}, skipping")
        return

    hy_omni_mesh = trimesh.load(str(hy_omni_path), process=False)
    hy_verts = np.array(hy_omni_mesh.vertices, dtype=np.float32) * scale

    if align_transform is not None:
        verts_homo = np.hstack([hy_verts, np.ones((len(hy_verts), 1), dtype=np.float32)])
        hy_verts = (align_transform @ verts_homo.T).T[:, :3].astype(np.float32)

    rr.log(
        "world/hy_omni/points",
        rr.Points3D(
            hy_verts,
            colors=np.broadcast_to(np.array([0, 0, 255], dtype=np.uint8), hy_verts.shape),
            radii=0.0003,
        ),
        static=True,
    )
    print(f"Visualized HY omni mesh: {len(hy_verts)} vertices")


def log_novel_view_camera(
    entity_path: str = "world/novel_view/camera",
    elevation_deg: float = 45.0,
    azimuth_deg: float = 135.0,
    distance: float = 0.4,
    image_width: int = 512,
    image_height: int = 512,
    fov_deg: float = 45.0,
    scale: float = 1.0,
    align_transform = None,
) -> None:
    """Log a static novel-view camera frustum in Rerun.

    The camera looks toward the world origin from a position defined by
    spherical coordinates (elevation, azimuth, distance) in a Y-up world.
    Intrinsics are derived from the given image size and vertical FOV.
    """
    distance = distance / scale
    cx, cy = image_width / 2.0, image_height / 2.0
    f = cy / np.tan(np.deg2rad(fov_deg) / 2.0)

    elev = np.deg2rad(elevation_deg)
    azim = np.deg2rad(azimuth_deg)
    cam_pos = np.array([
        distance * np.cos(elev) * np.sin(azim),
        distance * np.sin(elev),
        distance * np.cos(elev) * np.cos(azim),
    ], dtype=np.float32)

    # Look-at rotation: camera Z points toward origin
    z_axis = -cam_pos / np.linalg.norm(cam_pos)
    x_axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_c2w = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    R_c2w[:, 0] *= -1
    R_c2w[:, 1] *= -1
    cam_pos *= scale
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = cam_pos
    if align_transform is not None:
        c2w = align_transform @ c2w
        cam_pos = c2w[:3, 3]
        R_c2w = c2w[:3, :3]

    rr.log(entity_path, rr.Transform3D(translation=cam_pos, mat3x3=R_c2w), static=True)
    rr.log(
        entity_path,
        rr.Pinhole(
            resolution=[image_width, image_height],
            focal_length=[f, f],
            principal_point=[cx, cy],
            image_plane_distance=0.2,
        ),
        static=True,
    )


def compute_pred_to_gt_alignment(
    data_gt,
    gt_o2c: Optional[np.ndarray],
    gt_is_valid: Optional[np.ndarray],
    results_dir: Path,
    frame_indices: List[int],
    cond_idx: int,
    scale: float,
) -> Optional[np.ndarray]:
    """Compute pred-to-GT alignment using the condition frame or the next valid GT frame."""
    if data_gt is None or gt_o2c is None or gt_is_valid is None or len(gt_o2c) == 0:
        return None

    image_info = load_image_info(results_dir / f"{frame_indices[-1]:04d}")
    if image_info is None:
        print("Warning: failed to load image_info for alignment; skip pred-to-GT alignment.")
        return None

    pred_frame_indices = image_info.get("frame_indices", [])
    if len(pred_frame_indices) == 0:
        print("Warning: image_info has empty frame_indices; skip pred-to-GT alignment.")
        return None

    # Try condition frame first; if its GT is invalid, walk register order to find a valid GT frame.
    try:
        start_idx = frame_indices.index(cond_idx)
    except ValueError:
        start_idx = 0
        print(f"Warning: cond_idx {cond_idx} not found in register_order, fallback search from start.")

    search_order = list(range(start_idx, len(frame_indices))) + list(range(0, start_idx))
    gt_cond_idx = None
    ref_frame_idx = None
    for idx in search_order:
        if idx >= len(gt_o2c) or idx >= len(gt_is_valid):
            continue
        if not bool(gt_is_valid[idx]):
            continue
        fid = frame_indices[idx]
        if fid not in pred_frame_indices:
            continue
        gt_cond_idx = idx
        ref_frame_idx = fid
        break

    if gt_cond_idx is None:
        print("Warning: no valid GT frame found in register_order for alignment; skip pred-to-GT alignment.")
        return None

    if ref_frame_idx != cond_idx:
        print(f"Info: cond_idx {cond_idx} GT invalid/unavailable, fallback to frame {ref_frame_idx} for alignment.")

    cond_local = pred_frame_indices.index(ref_frame_idx)
    first_c2o = np.array(image_info["c2o"])
    first_c2o[:, :3, 3] *= scale
    return np.linalg.inv(gt_o2c[gt_cond_idx]) @ np.linalg.inv(first_c2o[cond_local])




def visualize_gt_frame(
    frame_idx: int,
    gt_o2c: np.ndarray,
    preprocess_data: Dict,
    jpeg_quality: int = 85,
):
    """Visualize GT camera pose and image for a single frame."""
    gt_c2o = np.linalg.inv(gt_o2c).astype(np.float32)
    K = preprocess_data.get('intrinsics')
    img = preprocess_data.get('image')
    log_camera_frame(
        "world/gt_current_frame/camera", K, gt_c2o, img,
        image_plane_distance=1.0, jpeg_quality=jpeg_quality, static=False,
    )


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
    jpeg_quality: int = 85,
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

        # Draw object and hand mask boundaries on the image
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_obj = preprocess_data.get('mask_obj')
        if mask_obj is not None:
            mask_u8 = (mask_obj > 0).astype(np.uint8) * 255
            boundary = cv2.dilate(mask_u8, kernel) - cv2.erode(mask_u8, kernel)
            img[boundary > 127] = [0, 255, 0]  # Green for object

        mask_hand = preprocess_data.get('mask_hand')
        if mask_hand is not None:
            mask_u8 = (mask_hand > 0).astype(np.uint8) * 255
            boundary = cv2.dilate(mask_u8, kernel) - cv2.erode(mask_u8, kernel)
            img[boundary > 127] = [255, 0, 255]  # Magenta for hand

        rr.log(f"{frame_entity}/camera", rr.Image(img).compress(jpeg_quality=80), static=False)

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
        points_3d = image_info['points_3d'] * scale  # (N, 3)

        # Log valid 3D points (finite)
        valid_3d_mask = np.isfinite(points_3d).all(axis=-1)
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

        # Log depth points after PnP registration (before any refinement)
        depth_after_pnp = image_info.get('depth_after_PnP')
        if depth_after_pnp is not None:
            pts = np.asarray(depth_after_pnp, dtype=np.float32) * scale
            if align_pred_to_gt is not None:
                pts = (align_pred_to_gt[:3, :3] @ pts.T).T + align_pred_to_gt[:3, 3]
            rr.log(
                f"{frame_entity}/depth_after_PnP",
                rr.Points3D(
                    pts,
                    colors=np.broadcast_to(np.array([100, 165, 100], dtype=np.uint8), pts.shape),
                    radii=0.0005,
                ),
            )
        else:
            rr.log(
                f"{frame_entity}/depth_after_PnP",
                rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0),
            )

        # Log depth points after align_frame_with_mesh_depth (magenta)
        depth_after_align = image_info.get('depth_after_align_mesh')
        if depth_after_align is not None:
            pts = np.asarray(depth_after_align, dtype=np.float32) * scale
            if align_pred_to_gt is not None:
                pts = (align_pred_to_gt[:3, :3] @ pts.T).T + align_pred_to_gt[:3, 3]
            rr.log(
                f"{frame_entity}/depth_after_align_mesh",
                rr.Points3D(
                    pts,
                    colors=np.broadcast_to(np.array([255, 0, 255], dtype=np.uint8), pts.shape),
                    radii=0.0005,
                ),
            )
        else:
            rr.log(
                f"{frame_entity}/depth_after_align_mesh",
                rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0),
            )

        # Log depth points after joint keyframe optimization (yellow)
        depth_after_kf_opt = image_info.get('depth_after_keyframes_opt')
        if depth_after_kf_opt is not None:
            pts = np.asarray(depth_after_kf_opt, dtype=np.float32) * scale
            if align_pred_to_gt is not None:
                pts = (align_pred_to_gt[:3, :3] @ pts.T).T + align_pred_to_gt[:3, 3]
            rr.log(
                f"{frame_entity}/depth_after_keyframes_opt",
                rr.Points3D(
                    pts,
                    colors=np.broadcast_to(np.array([255, 255, 0], dtype=np.uint8), pts.shape),
                    radii=0.0005,
                ),
            )
        else:
            rr.log(
                f"{frame_entity}/depth_after_keyframes_opt",
                rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0),
            )

        # Log depth points after reset when PnP fails (red)
        depth_after_reset = image_info.get('depth_after_reset_when_pnp_fail')
        if depth_after_reset is not None:
            pts = np.asarray(depth_after_reset, dtype=np.float32) * scale
            if align_pred_to_gt is not None:
                pts = (align_pred_to_gt[:3, :3] @ pts.T).T + align_pred_to_gt[:3, 3]
            rr.log(
                f"{frame_entity}/depth_after_reset_when_pnp_fail",
                rr.Points3D(
                    pts,
                    colors=np.broadcast_to(np.array([255, 80, 80], dtype=np.uint8), pts.shape),
                    radii=0.0005,
                ),
            )
        else:
            rr.log(
                f"{frame_entity}/depth_after_reset_when_pnp_fail",
                rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0),
            )

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

    # Log masked depth as 3D point clouds in object space

    depth = preprocess_data.get('depth')
    K = preprocess_data.get('intrinsics')
    if depth is not None and K is not None and c2o is not None:
        mask_obj = preprocess_data.get('mask_obj')
        mask_hand = preprocess_data.get('mask_hand')

        def _backproject_masked_depth(depth_map, mask, K, c2o, scale, max_pts=5000):
            """Back-project masked depth pixels to 3D object-space points."""
            valid = (depth_map > 0) & (mask > 0)
            ys, xs = np.where(valid)
            if len(ys) == 0:
                return None
            if len(ys) > max_pts:
                sel = np.random.choice(len(ys), max_pts, replace=False)
                ys, xs = ys[sel], xs[sel]
            zs = depth_map[ys, xs].astype(np.float32)
            xc = (xs.astype(np.float32) - K[0, 2]) * zs / K[0, 0]
            yc = (ys.astype(np.float32) - K[1, 2]) * zs / K[1, 1]
            pts_cam = np.stack([xc, yc, zs], axis=-1)
            pts_obj = (c2o[:3, :3] @ pts_cam.T).T * scale + c2o[:3, 3]
            return pts_obj

        # Object depth points (orange)
        if mask_obj is not None:
            pts_obj_depth = _backproject_masked_depth(depth, mask_obj, K, c2o, scale)
            if pts_obj_depth is not None:
                rr.log(
                    f"{frame_entity}/depth_obj",
                    rr.Points3D(
                        pts_obj_depth,
                        colors=np.broadcast_to(np.array([255, 165, 0], dtype=np.uint8), pts_obj_depth.shape),
                        radii=0.0005,
                    ),
                    static=False,
                )
            else:
                rr.log(f"{frame_entity}/depth_obj",
                       rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0))

        # Hand depth points (cyan)
        if mask_hand is not None:
            pts_hand_depth = _backproject_masked_depth(depth, mask_hand, K, c2o, scale)
            if pts_hand_depth is not None:
                rr.log(
                    f"{frame_entity}/depth_hand",
                    rr.Points3D(
                        pts_hand_depth,
                        colors=np.broadcast_to(np.array([0, 200, 255], dtype=np.uint8), pts_hand_depth.shape),
                        radii=0.0005,
                    ),
                    static=False,
                )
            else:
                rr.log(f"{frame_entity}/depth_hand",
                       rr.Points3D([[0, 0, 0]], colors=[[0, 0, 0, 0]], radii=0.0))

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


def load_gt_visualization_data(out_dir: Path, frame_indices: list):
    """Load GT object and hand data for visualization.

    Returns (data_gt, gt_o2c, gt_is_valid, gt_hand_verts_cam, gt_hand_faces).
    All arrays are numpy. Returns Nones if loading fails.
    """
    import vggt.utils.gt as gt

    seq_name = out_dir.name

    def get_image_fids():
        return frame_indices

    print("Loading GT data...")
    data_gt = gt.load_data(seq_name, get_image_fids)

    gt_o2c = data_gt["o2c"].numpy()
    gt_is_valid = data_gt["is_valid"].numpy()
    print(f"Loaded GT data: {len(gt_o2c)} frames")

    gt_mesh_path = data_gt["mesh_name.object"]
    log_mesh("world/gt_mesh", gt_mesh_path, static=True)

    gt_hand_verts_cam = data_gt["v3d_c.right"] if "v3d_c.right" in data_gt else None
    gt_hand_faces = data_gt["faces.right"] if "faces.right" in data_gt else None
    if gt_hand_verts_cam is not None and not isinstance(gt_hand_verts_cam, np.ndarray):
        gt_hand_verts_cam = gt_hand_verts_cam.numpy()
    if gt_hand_faces is not None and not isinstance(gt_hand_faces, np.ndarray):
        gt_hand_faces = gt_hand_faces.numpy()

    if gt_hand_verts_cam is not None and gt_hand_faces is not None:
        from common.body_models import seal_mano_mesh_np
        gt_hand_verts_cam, gt_hand_faces = seal_mano_mesh_np(
            gt_hand_verts_cam, gt_hand_faces.astype(np.int64), is_rhand=True
        )
        gt_hand_faces = gt_hand_faces.astype(np.int32)
        print(f"Loaded GT hand mesh (sealed): {gt_hand_verts_cam.shape[1]} vertices, {len(gt_hand_faces)} faces")

    return data_gt, gt_o2c, gt_is_valid, gt_hand_verts_cam, gt_hand_faces


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = data_dir / "pipeline_preprocess"
    tracks_dir = data_dir / "pipeline_corres"
    results_dir = out_dir / "pipeline_joint_opt"
    neus_init_dir = out_dir / "pipeline_neus_init"
    neus_joint_opt_dir = out_dir / "pipeline_joint_opt"

    # Parse per-frame log blocks from log.txt
    frame_logs = parse_frame_logs(results_dir / "txt.log")

    # Initialize Rerun
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera", origin="world/current_frame/camera"),
                rrb.Spatial2DView(name="Reproj Error", origin="reproj_error"),
                rrb.TextLogView(name="Frame Log", origin="frame_log"),
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
    gt_o2c = None
    gt_is_valid = None
    gt_hand_verts_cam = None
    gt_hand_faces = None
    if args.vis_gt:
        data_gt, gt_o2c, gt_is_valid, gt_hand_verts_cam, gt_hand_faces = load_gt_visualization_data(
            out_dir,
            frame_indices,
        )


    # Compute alignment transform from pred object space to GT object space
    # using the condition frame as the reference

    align_pred_to_gt = compute_pred_to_gt_alignment(
        data_gt=data_gt,
        gt_o2c=gt_o2c,
        gt_is_valid=gt_is_valid,
        results_dir=results_dir,
        frame_indices=frame_indices,
        cond_idx=cond_idx,
        scale=scale,
    )

    if SAM3D_mesh is not None:
        log_static_trimesh_mesh(
            entity_path="world/sam3d/mesh",
            mesh=SAM3D_mesh,
            scale=scale,
            align_transform=align_pred_to_gt,
            default_color=0.7,
        )

    neus_mesh_dir = neus_init_dir / "neus_training" /  "joint_opt" / "save"
    neus_mesh_path = find_latest_mesh_file(neus_mesh_dir)
    if neus_mesh_path is None:
        print(f"NeuS mesh not found under {neus_mesh_dir}, skipping")
    else:
        print(f"Loading NeuS mesh from {neus_mesh_path}")
        import trimesh
        neus_mesh = trimesh.load(str(neus_mesh_path), force="mesh", process=False)
        if isinstance(neus_mesh, trimesh.Trimesh):
            log_static_trimesh_mesh(
                entity_path="world/neus_init/mesh",
                mesh=neus_mesh,
                scale=scale,
                align_transform=align_pred_to_gt,
                default_color=0.3,
            )
        else:
            print(f"Warning: failed to load NeuS mesh geometry from {neus_mesh_path}, skipping")
    

    # Load HY omni mesh (in SAM3D space) from pipeline_HY_to_SAM3D
    hy_omni_path = out_dir / "pipeline_HY_to_SAM3D" / "HY_omni_in_sam3d.obj"
    visualize_hy_omni_points(
        hy_omni_path=hy_omni_path,
        scale=scale,
        align_transform=align_pred_to_gt,
    )
    log_novel_view_camera(scale=scale, align_transform=align_pred_to_gt)

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

        # Use pre-computed track visibility count
        track_vis_count = image_info.get("track_vis_count")
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
            jpeg_quality=args.jpeg_quality,
        )

        # Visualize all camera poses
        if args.vis_all_cameras:
            visualize_all_cameras(image_info_all, frame_idx, scale=scale)

        # Visualize GT camera pose and hand mesh
        if data_gt is not None and i < len(gt_o2c) and bool(gt_is_valid[i]):
            visualize_gt_frame(frame_idx, gt_o2c[i], preprocess_data, jpeg_quality=args.jpeg_quality)

            # Log GT hand mesh in object space
            if gt_hand_verts_cam is not None and gt_hand_faces is not None and i < len(gt_hand_verts_cam):
                gt_c2o_i = np.linalg.inv(gt_o2c[i]).astype(np.float32)
                hv_cam = gt_hand_verts_cam[i].astype(np.float32)
                hv_obj = (gt_c2o_i[:3, :3] @ hv_cam.T).T + gt_c2o_i[:3, 3]
                vertex_normals = compute_vertex_normals(hv_obj, gt_hand_faces)
                rr.log(
                    "world/gt_hand",
                    rr.Mesh3D(
                        vertex_positions=hv_obj,
                        triangle_indices=gt_hand_faces,
                        vertex_normals=vertex_normals,
                        mesh_material=rr.Material(albedo_factor=[200, 180, 160]),
                    ),
                    static=False,
                )

        # Log reprojection error image if available
        reproj_img_path = results_dir / f"{frame_idx:04d}" / "reproj_error.png"
        if reproj_img_path.exists():
            from PIL import Image
            reproj_img = np.array(Image.open(reproj_img_path).convert("RGB"))
            rr.log("reproj_error", rr.Image(reproj_img).compress(jpeg_quality=args.jpeg_quality), static=False)

        # Log per-frame registration log
        if frame_idx in frame_logs:
            rr.log("frame_log", rr.TextLog(frame_logs[frame_idx]), static=False)

        neus_joint_opt_mesh_path = neus_joint_opt_dir / "neus_training" / f"{frame_idx:04d}" / "joint_opt" / "save"
        log_neus_mesh_from_dir(
            save_dir=neus_joint_opt_mesh_path,
            entity_path="world/neus_joint_opt/mesh",
            scale=scale,
            align_transform=align_pred_to_gt,
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
    parser.add_argument("--vis_type", type=str, default="all", choices=["all", "registered_valid", "keyframes"],
                        help="Type of frames to visualize")
    parser.add_argument("--vis_gt", type=int, default=1,
                        help="Visualize ground truth mesh and camera poses")
    parser.add_argument("--min_track_number", type=int, default=4,
                        help="Minimum track visibility count for green coloring")
    parser.add_argument("--vis_all_cameras", action="store_true", default=True,
                        help="Visualize all camera poses from image_info, not just the current frame")
    parser.add_argument("--vis_hand", action="store_true", default=True,
                        help="Visualize hand mesh in object space")
    parser.add_argument("--jpeg_quality", type=int, default=30,
                        help="JPEG compression quality for camera images (0-100)")

    args = parser.parse_args()
    main(args)
