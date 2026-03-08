import numpy as np

import json
import os
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

import vggt.utils.eval_modules as eval_m
import vggt.utils.gt as gt
import torch
import trimesh
import smplx
from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform, load_preprocessed_frame
from viewer.viewer_step import HandDataProvider
from utils_simba.geometry import transform_points
device = "cuda:0"


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


def load_mesh_as_trimesh(mesh_path: Path):
    """Load a mesh file and return a single Trimesh (supports scene-based GLB)."""
    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(transform)
            meshes.append(geom)
        if len(meshes) == 0:
            return None
        return trimesh.util.concatenate(meshes)

    return None


def build_mesh_object_predictions(
    mesh_path: Path,
    frame_indices: np.ndarray,
    valid_extrinsics: np.ndarray,
    c2o: np.ndarray,
    scale: float,
    sam3d_to_cond_cam: np.ndarray,
    cond_index: int,
):
    """Load a mesh in SAM3D space and convert it to per-frame camera-space vertices."""
    if not mesh_path.exists():
        return None

    mesh = load_mesh_as_trimesh(mesh_path)
    if mesh is None:
        print(f"Failed to load mesh geometry from {mesh_path}")
        return None

    if cond_index not in frame_indices.tolist():
        print(f"Condition index {cond_index} not found in frame_indices, skip {mesh_path}")
        return None

    verts_sam3d = np.array(mesh.vertices, dtype=np.float64)
    cond_local = frame_indices.tolist().index(cond_index)
    c2o_cond_scaled = c2o[cond_local].copy()
    c2o_cond_scaled[:3, 3] *= scale
    sam3d_to_obj = c2o_cond_scaled @ sam3d_to_cond_cam  # (4, 4)

    verts_homo = np.hstack([verts_sam3d, np.ones((len(verts_sam3d), 1), dtype=np.float64)])
    verts_obj = (sam3d_to_obj @ verts_homo.T).T[:, :3]

    v3d_right_list = []
    for i in range(len(valid_extrinsics)):
        o2c_i = valid_extrinsics[i]
        v_cam = (o2c_i[:3, :3] @ verts_obj.T).T + o2c_i[:3, 3]
        bbox_center = (v_cam.min(axis=0) + v_cam.max(axis=0)) / 2.0
        v_cam_ra = (v_cam - bbox_center).astype(np.float32)
        v3d_right_list.append(torch.tensor(v_cam_ra))

    faces = np.array(mesh.faces, dtype=np.int64)
    return {
        "v3d_ra.object": v3d_right_list,
        "v3d_right.object": v3d_right_list,
        "faces": {"object": torch.tensor(faces)},
    }


def load_object_mesh_from_sam3d(
    SAM3D_dir: Path,
    cond_index: int,
    frame_indices_full: np.ndarray,
    c2o_full: np.ndarray,
    scale: float,
    sam3d_to_cond_cam: np.ndarray,
):
    """Load mesh from SAM3D directory and convert canonical vertices to object space."""
    # Match pipeline_joint_opt_eval.py preference.
    mesh_candidates = [
        SAM3D_dir.parent / "SAM3D" / f"{cond_index:04d}" / "scene.glb",
        SAM3D_dir / f"{cond_index:04d}" / "mesh.obj",
        SAM3D_dir / f"{cond_index:04d}" / "scene.glb",
    ]
    mesh_path = next((p for p in mesh_candidates if p.exists()), None)
    if mesh_path is None:
        return None, None, None

    mesh = load_mesh_as_trimesh(mesh_path)
    if mesh is None:
        return None, None, None

    frame_list = frame_indices_full.tolist() if isinstance(frame_indices_full, np.ndarray) else list(frame_indices_full)
    if cond_index not in frame_list:
        return None, None, None

    verts_obj = mesh.vertices * scale
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    vertex_colors = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(verts_obj) and vc.shape[1] >= 3:
            vertex_colors = vc[:, :3].astype(np.uint8)
    elif hasattr(mesh, "visual") and hasattr(mesh.visual, "to_color") and callable(mesh.visual.to_color):
        vc = np.asarray(mesh.visual.to_color().vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(verts_obj) and vc.shape[1] >= 3:
            vertex_colors = vc[:, :3].astype(np.uint8)

    return verts_obj, faces, vertex_colors


def find_joint_opt_mesh_from_ckpt(joint_opt_ckpt: Path):
    """Find an exported NeuS mesh near the fixed checkpoint path."""
    if not joint_opt_ckpt.exists():
        return None

    save_dir = joint_opt_ckpt.parent.parent / "save"
    if not save_dir.exists():
        return None

    mesh_candidates = sorted(save_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mesh_candidates[0] if mesh_candidates else None





def visualize_in_rerun(extrinsics, frame_indices, valid_flags, SAM3D_dir, cond_index, scale, jpeg_quality=85):
    """Visualize object-to-camera extrinsics and SAM3D mesh in rerun.

    Args:
        extrinsics: (N, 4, 4) object-to-camera matrices
        frame_indices: (N,) frame index array
        valid_flags: (N,) boolean mask for valid frames
        SAM3D_dir: Path to SAM3D_aligned_post_process directory
        cond_index: Condition frame index
        scale: SAM3D-to-metric scale factor
    """
    import rerun as rr
    rr.init("pipeline_joint_opt_eval", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load and log SAM3D mesh
    mesh_path = SAM3D_dir / f"{cond_index:04d}" / "mesh.obj"
    if mesh_path.exists():
        sam3d_mesh = trimesh.load(str(mesh_path), force='mesh')
        verts = np.array(sam3d_mesh.vertices, dtype=np.float32) * scale
        faces = np.array(sam3d_mesh.faces, dtype=np.uint32)
        vertex_colors = None
        if sam3d_mesh.visual is not None and hasattr(sam3d_mesh.visual, 'vertex_colors'):
            vertex_colors = np.array(sam3d_mesh.visual.vertex_colors)[:, :3]
        rr.log(
            "world/sam3d_mesh",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces,
                vertex_colors=vertex_colors,
            ),
            static=True,
        )

    # Log camera poses from extrinsics (object-to-camera), intrinsic and image
    data_preprocess_dir = SAM3D_dir.parent / "pipeline_preprocess"
    for i, fid in enumerate(frame_indices):
        if not valid_flags[i]:
            continue
        c2o_i = np.linalg.inv(extrinsics[i]).astype(np.float32)
        entity = f"world/cameras/{fid:04d}"
        rr.log(entity, rr.Transform3D(translation=c2o_i[:3, 3], mat3x3=c2o_i[:3, :3]))

        preprocess_data = load_preprocessed_frame(data_preprocess_dir, fid)
        K = preprocess_data.get("intrinsics")
        img = preprocess_data.get("image")
        if K is not None and img is not None:
            H, W = img.shape[:2]
            rr.log(
                f"{entity}/camera",
                rr.Pinhole(
                    resolution=[W, H],
                    focal_length=[float(K[0, 0]), float(K[1, 1])],
                    principal_point=[float(K[0, 2]), float(K[1, 2])],
                    image_plane_distance=0.02,
                ),
            )
            rr.log(f"{entity}/camera", rr.Image(img).compress(jpeg_quality=jpeg_quality))



def align_pred_to_gt(pred_o2c, gt_o2c, valid_frame_indices,
                     cond_index, register_indices):
    """Align predicted extrinsics to GT object space using a shared anchor frame.

    Uses the condition frame as anchor. If it is not among valid frames,
    falls back to the first frame in register_indices that is valid.

    Args:
        valid_extrinsics: (M, 4, 4) predicted object-to-camera for valid frames
        gt_o2c: (M, 4, 4) GT object-to-camera for the same valid frames
        valid_frame_indices: (M,) frame indices corresponding to the matrices
        cond_index: preferred anchor frame index
        register_indices: ordered list of registered frame indices to search

    Returns:
        (M, 4, 4) aligned predicted extrinsics
    """
    valid_list = valid_frame_indices.tolist()
    if cond_index in valid_list:
        anchor_idx = valid_list.index(cond_index)
    else:
        anchor_idx = None
        for ri in register_indices:
            if ri in valid_list:
                anchor_idx = valid_list.index(ri)
                print(f"[align] cond_index {cond_index} not in valid frames, "
                      f"using frame {ri} as anchor")
                break
        if anchor_idx is None:
            raise ValueError(
                "No registered frame found in valid_frame_indices for alignment")
    # anchor_idx = 0 # hardcode to use the first valid frame as anchor
    align_tf = np.linalg.inv(pred_o2c[anchor_idx]) @ gt_o2c[anchor_idx]
    return pred_o2c @ align_tf


def compute_frustum_lines(K, H, W, c2o, depth=0.02):
    """Compute camera frustum line segments in world (object) space.

    Returns a list of (2,3) line segments: 4 edges from origin to corners
    and 4 edges forming the image-plane rectangle.
    """
    K_inv = np.linalg.inv(K[:3, :3])
    corners_px = np.array([
        [0, 0, 1],
        [W, 0, 1],
        [W, H, 1],
        [0, H, 1],
    ], dtype=np.float64)
    # Unproject to camera space at given depth
    corners_cam = (K_inv @ corners_px.T).T * depth  # (4, 3)
    origin_cam = np.zeros(3, dtype=np.float64)

    # Transform to world space
    R = c2o[:3, :3].astype(np.float64)
    t = c2o[:3, 3].astype(np.float64)
    origin_w = t.copy()
    corners_w = (R @ corners_cam.T).T + t  # (4, 3)

    segments = []
    # 4 edges from origin to each corner
    for c in corners_w:
        segments.append(np.stack([origin_w, c]))
    # 4 edges of the rectangle
    for j in range(4):
        segments.append(np.stack([corners_w[j], corners_w[(j + 1) % 4]]))
    return segments


def visualize_gt_and_pred_in_rerun(
    data_gt,
    data_pred,
    pred_extrinsics,
    frame_indices,
    SAM3D_dir,
    cond_index,
    frame_indices_full,
    c2o_full,
    scale,
    sam3d_to_cond_cam,
                                   jpeg_quality=85, world_mode="camera", history_window=50):
    """Visualize GT and predicted poses with meshes in rerun.

    Args:
        data_gt: Ground truth data dict (from gt.load_data) with keys:
            mesh_name.object, o2c, K, is_valid
        pred_extrinsics: (M, 4, 4) predicted object-to-camera matrices for valid frames
        frame_indices: (M,) frame indices corresponding to pred_extrinsics
        SAM3D_dir: Path to SAM3D_aligned_post_process directory
        jpeg_quality: JPEG quality for compressed images
        world_mode: "camera" (camera fixed, object moves) or
                    "object" (object fixed at identity, cameras move)
    """
    import rerun as rr
    rr.init("pipeline_joint_opt_eval", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load GT mesh canonical vertices/faces
    gt_mesh_path = data_gt["mesh_name.object"]
    gt_mesh = load_mesh_as_trimesh(Path(gt_mesh_path)) if os.path.exists(gt_mesh_path) else None
    gt_verts_can = np.array(gt_mesh.vertices, dtype=np.float32) if gt_mesh is not None else None
    gt_faces = np.array(gt_mesh.faces, dtype=np.uint32) if gt_mesh is not None else None
    gt_vertex_colors = None
    if gt_mesh is not None and gt_verts_can is not None:
        colors = None
        mesh_colors = None
        if colors is None and hasattr(gt_mesh, "visual") and hasattr(gt_mesh.visual, "vertex_colors"):
            mesh_colors = np.asarray(gt_mesh.visual.vertex_colors)
            if mesh_colors.ndim != 2 or mesh_colors.shape[0] != gt_verts_can.shape[0]:
                mesh_colors = None
        elif colors is None and hasattr(gt_mesh, "visual") and hasattr(gt_mesh.visual, "to_color") and callable(gt_mesh.visual.to_color):
            mesh_colors = gt_mesh.visual.to_color().vertex_colors
            if mesh_colors.ndim != 2 or mesh_colors.shape[0] != gt_verts_can.shape[0]:
                mesh_colors = None

        if mesh_colors is not None and mesh_colors.shape[1] >= 3:
            gt_vertex_colors = np.asarray(mesh_colors[:, :3]).astype(np.uint8)

    gt_o2c = data_gt["o2c"].numpy() if torch.is_tensor(data_gt["o2c"]) else np.array(data_gt["o2c"])
    gt_is_valid = data_gt["is_valid"].numpy() if torch.is_tensor(data_gt["is_valid"]) else np.array(data_gt["is_valid"])
    gt_K = data_gt["K"].numpy() if torch.is_tensor(data_gt["K"]) else np.array(data_gt["K"])
    data_preprocess_dir = SAM3D_dir.parent / "pipeline_preprocess"

    pred_verts, pred_faces, pred_colors = load_object_mesh_from_sam3d(SAM3D_dir, cond_index, frame_indices_full, c2o_full, scale, sam3d_to_cond_cam)
    # TODO: need algin the object mesh from SAM3D to GT canonical space before visualization if they are not aligned.

    # Hand mesh payloads in camera space.
    pred_hand_verts_cam = data_pred.get("v3d_c.right")
    pred_hand_faces = data_pred.get("faces.right")
    gt_hand_verts_cam = data_gt.get("v3d_c.right")
    gt_hand_faces = data_gt.get("faces.right")
    if torch.is_tensor(pred_hand_verts_cam):
        pred_hand_verts_cam = pred_hand_verts_cam.detach().cpu().numpy()
    if torch.is_tensor(pred_hand_faces):
        pred_hand_faces = pred_hand_faces.detach().cpu().numpy()
    if torch.is_tensor(gt_hand_verts_cam):
        gt_hand_verts_cam = gt_hand_verts_cam.detach().cpu().numpy()
    if torch.is_tensor(gt_hand_faces):
        gt_hand_faces = gt_hand_faces.detach().cpu().numpy()

    if pred_hand_verts_cam is not None and pred_hand_faces is not None:
        from common.body_models import seal_mano_mesh_np
        pred_hand_verts_cam, pred_hand_faces = seal_mano_mesh_np(
            pred_hand_verts_cam, np.asarray(pred_hand_faces).astype(np.int64), is_rhand=True
        )
        pred_hand_faces = pred_hand_faces.astype(np.int32)
    if gt_hand_verts_cam is not None and gt_hand_faces is not None:
        from common.body_models import seal_mano_mesh_np
        gt_hand_verts_cam, gt_hand_faces = seal_mano_mesh_np(
            gt_hand_verts_cam, np.asarray(gt_hand_faces).astype(np.int64), is_rhand=True
        )
        gt_hand_faces = gt_hand_faces.astype(np.int32)

    def _log_hand_meshes(frame_i: int):
        if pred_hand_verts_cam is not None and pred_hand_faces is not None and frame_i < len(pred_hand_verts_cam):
            hv_pred = np.asarray(pred_hand_verts_cam[frame_i], dtype=np.float32)
            vn_pred = compute_vertex_normals(hv_pred, pred_hand_faces)
            rr.log(
                "world/pred_hand",
                rr.Mesh3D(
                    vertex_positions=hv_pred,
                    triangle_indices=pred_hand_faces,
                    vertex_normals=vn_pred,
                    mesh_material=rr.Material(albedo_factor=[200, 180, 220]),
                ),
                static=False,
            )
        if gt_hand_verts_cam is not None and gt_hand_faces is not None and frame_i < len(gt_hand_verts_cam):
            hv_gt = np.asarray(gt_hand_verts_cam[frame_i], dtype=np.float32)
            vn_gt = compute_vertex_normals(hv_gt, gt_hand_faces)
            rr.log(
                "world/gt_hand",
                rr.Mesh3D(
                    vertex_positions=hv_gt,
                    triangle_indices=gt_hand_faces,
                    vertex_normals=vn_gt,
                    mesh_material=rr.Material(albedo_factor=[200, 180, 160]),
                ),
                static=False,
            )

    mesh_kwargs = {}
    if gt_verts_can is not None and gt_faces is not None:
        mesh_kwargs = {
            "vertex_positions": gt_verts_can.astype(np.float32),
            "triangle_indices": gt_faces,
        }
        if gt_vertex_colors is not None:
            mesh_kwargs["vertex_colors"] = gt_vertex_colors
    if world_mode == "object":
        # Object-centric: mesh at identity, cameras shown frame by frame
        # with sliding history window.
        if mesh_kwargs:
            rr.log("world/object/mesh", rr.Mesh3D(**mesh_kwargs), static=True)

        segs_per_frustum = 8
        max_history_segs = history_window * segs_per_frustum
        pred_history_segs = []
        gt_history_segs = []

        for i, fid in enumerate(frame_indices):
            rr.set_time_sequence("frame", i)

            preprocess_data = load_preprocessed_frame(data_preprocess_dir, fid)
            img = preprocess_data.get("image")
            K_pred = preprocess_data.get("intrinsics")

            pred_c2o = np.linalg.inv(pred_extrinsics[i]).astype(np.float32)
            if img is not None and K_pred is not None:
                H, W = img.shape[:2]
                segs = compute_frustum_lines(K_pred, H, W, pred_c2o, depth=0.1)
                # Current frame frustum (bright)
                rr.log("world/pred_camera/frustum",
                       rr.LineStrips3D(segs, colors=[[0, 120, 255]]))
                # Sliding history window (dimmed)
                pred_history_segs.extend(segs)
                if len(pred_history_segs) > max_history_segs:
                    pred_history_segs = pred_history_segs[-max_history_segs:]
                rr.log("world/pred_camera/history",
                       rr.LineStrips3D(pred_history_segs, colors=[[0, 60, 128]]))

            if i < len(gt_o2c) and bool(gt_is_valid[i]):
                gt_c2o = np.linalg.inv(gt_o2c[i]).astype(np.float32)
                if img is not None:
                    H, W = img.shape[:2]
                    gt_K_i = gt_K if gt_K.ndim == 2 else gt_K[i]
                    segs = compute_frustum_lines(gt_K_i, H, W, gt_c2o, depth=0.1)
                    rr.log("world/gt_camera/frustum",
                           rr.LineStrips3D(segs, colors=[[0, 200, 0]]))
                    gt_history_segs.extend(segs)
                    if len(gt_history_segs) > max_history_segs:
                        gt_history_segs = gt_history_segs[-max_history_segs:]
                    rr.log("world/gt_camera/history",
                           rr.LineStrips3D(gt_history_segs, colors=[[0, 100, 0]]))
            _log_hand_meshes(i)
    else:
        # Camera-centric (default): transform mesh vertices per frame and log transformed meshes.
        for i, fid in enumerate(frame_indices):
            rr.set_time_sequence("frame", i)

            preprocess_data = load_preprocessed_frame(data_preprocess_dir, fid)
            img = preprocess_data.get("image")
            K_pred = preprocess_data.get("intrinsics")

            pred_o2c = pred_extrinsics[i].astype(np.float32)
            pred_entity = "world/pred_camera"
            rr.log(pred_entity, rr.Transform3D(
                translation=np.zeros_like(pred_o2c[:3, 3]), mat3x3=np.eye(3)))
            if gt_verts_can is not None and gt_faces is not None:
                pred_verts_cam = (pred_o2c[:3, :3] @ gt_verts_can.T).T + pred_o2c[:3, 3]
                pred_mesh_kwargs = {
                    "vertex_positions": pred_verts_cam.astype(np.float32),
                    "triangle_indices": gt_faces,
                }
                if gt_vertex_colors is not None:
                    pred_mesh_kwargs["vertex_colors"] = gt_vertex_colors
                rr.log("world/pred_mesh_cam", rr.Mesh3D(**pred_mesh_kwargs))
            if img is not None and K_pred is not None:
                H, W = img.shape[:2]
                rr.log(
                    f"{pred_entity}/camera",
                    rr.Pinhole(
                        resolution=[W, H],
                        focal_length=[float(K_pred[0, 0]), float(K_pred[1, 1])],
                        principal_point=[float(K_pred[0, 2]), float(K_pred[1, 2])],
                        image_plane_distance=1.0,
                    ),
                )
                rr.log(f"{pred_entity}/camera", rr.Image(img).compress(jpeg_quality=jpeg_quality))

            if i < len(gt_o2c) and bool(gt_is_valid[i]):
                gt_o2c_i = gt_o2c[i].astype(np.float32)
                gt_entity = "world/gt_camera"
                rr.log(gt_entity, rr.Transform3D(
                    translation=np.zeros_like(gt_o2c_i[:3, 3]), mat3x3=np.eye(3)))
                if gt_verts_can is not None and gt_faces is not None:
                    gt_verts_cam = (gt_o2c_i[:3, :3] @ gt_verts_can.T).T + gt_o2c_i[:3, 3]
                    gt_mesh_kwargs = {
                        "vertex_positions": gt_verts_cam.astype(np.float32),
                        "triangle_indices": gt_faces,
                    }
                    if gt_vertex_colors is not None:
                        gt_mesh_kwargs["vertex_colors"] = gt_vertex_colors
                    rr.log("world/gt_mesh_cam", rr.Mesh3D(**gt_mesh_kwargs))
                if img is not None:
                    H, W = img.shape[:2]
                    rr.log(
                        f"{gt_entity}/camera",
                        rr.Pinhole(
                            resolution=[W, H],
                            focal_length=[float(gt_K[0, 0]), float(gt_K[1, 1])],
                            principal_point=[float(gt_K[0, 2]), float(gt_K[1, 2])],
                            image_plane_distance=1.0,
                        ),
                    )
                    rr.log(f"{gt_entity}/camera", rr.Image(img).compress(jpeg_quality=jpeg_quality))
            _log_hand_meshes(i)


def filter_invalid_gt_frames(data_gt, data_pred):
    """Remove GT-invalid frames from both data_gt and data_pred.

    Filters all per-frame entries (those whose first dimension equals the
    number of frames) in data_gt using its ``is_valid`` flag, and applies
    the same mask to ``data_pred``.

    Args:
        data_gt: Ground truth xdict from gt.load_data
        data_pred: Prediction dict with extrinsics, valid_frame_indices, is_valid

    Returns:
        Tuple of (data_gt, data_pred) with invalid frames removed.
    """
    gt_is_valid = data_gt["is_valid"]
    if torch.is_tensor(gt_is_valid):
        gt_valid_mask = gt_is_valid.bool().numpy()
    else:
        gt_valid_mask = np.asarray(gt_is_valid).astype(bool)

    if gt_valid_mask.all():
        return data_gt, data_pred

    num_filtered = int((~gt_valid_mask).sum())
    print(f"[filter] Removing {num_filtered} GT-invalid frames "
          f"({gt_valid_mask.sum()}/{len(gt_valid_mask)} remain)")

    # Filter per-frame entries in data_gt whose first dim matches the mask length
    n = len(gt_valid_mask)
    mask_tensor = torch.from_numpy(gt_valid_mask)
    for k in list(data_gt.keys()):
        v = data_gt[k]
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
            dict.__setitem__(data_gt, k, v[mask_tensor])
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            dict.__setitem__(data_gt, k, v[gt_valid_mask])

    # Filter per-frame entries in data_pred whose first dim matches
    n_pred = len(data_pred["is_valid"])
    mask_tensor_pred = torch.from_numpy(gt_valid_mask)
    for k in list(data_pred.keys()):
        v = data_pred[k]
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_pred:
            data_pred[k] = v[gt_valid_mask]
        elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n_pred:
            data_pred[k] = v[mask_tensor_pred]

    return data_gt, data_pred


def load_pred_data(results_dir, SAM3D_dir, cond_index):
    """Load image info and build prediction data dict with valid extrinsics.

    Args:
        results_dir: Path to pipeline_joint_opt results directory
        SAM3D_dir: Path to SAM3D_aligned_post_process directory
        cond_index: Condition frame index

    Returns:
        Tuple of (data_pred, valid_extrinsics, valid_frame_indices,
                  frame_indices, register_indices, c2o, scale, sam3d_to_cond_cam)
    """
    register_indices = load_register_indices(results_dir)
    last_register_idx = register_indices[-1]

    # Load per-frame data (compressed or legacy format)
    import gzip
    gz_path = results_dir / f"{last_register_idx:04d}" / "image_info.pkl.gz"
    npy_path = results_dir / f"{last_register_idx:04d}" / "image_info.npy"
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as f:
            image_info = pickle.load(f)
    else:
        image_info = np.load(npy_path, allow_pickle=True).item()

    # Merge shared static data if using split format
    shared_gz = results_dir / "shared_info.pkl.gz"
    shared_npy = results_dir / "shared_info.npy"
    if shared_gz.exists():
        with gzip.open(shared_gz, "rb") as f:
            shared = pickle.load(f)
        shared.update(image_info)
        image_info = shared
    elif shared_npy.exists():
        shared = np.load(shared_npy, allow_pickle=True).item()
        shared.update(image_info)
        image_info = shared

    sam3d_transform = load_sam3d_transform(SAM3D_dir, cond_index)
    sam3d_to_cond_cam = sam3d_transform['sam3d_to_cond_cam']
    scale = sam3d_transform['scale']

    frame_indices = np.array(image_info["frame_indices"])
    register_flags = np.array(image_info["register"], dtype=bool)
    invalid_flags = np.array(image_info["invalid"], dtype=bool)
    valid_flags = register_flags & ~invalid_flags
    c2o = np.array(image_info["c2o"])  # (N, 4, 4) camera-to-object (SAM3D scaled)
    extrinsics = c2o.copy()
    extrinsics[:, :3, 3] *= scale
    extrinsics = np.linalg.inv(extrinsics)  # object-to-camera

    valid_extrinsics = extrinsics[valid_flags]
    valid_frame_indices = frame_indices[valid_flags]

    seq_name = results_dir.parent.name

    data_pred = {
        "extrinsics": valid_extrinsics,
        "valid_frame_indices": valid_frame_indices,
        "is_valid": np.ones(len(valid_frame_indices), dtype=np.float32),
        "full_seq_name": seq_name,
        "total_frames": len(frame_indices),
        "registered_frames": int(register_flags.sum()),
        "keyframe_count": int(np.array(image_info.get("keyframe", []), dtype=bool).sum()),
        "invalid_frames": int(invalid_flags.sum()),
    }

    return (data_pred, frame_indices, register_indices, c2o, scale, sam3d_to_cond_cam, valid_flags)


def load_hand_predictions(results_dir, hand_mode, frame_indices, valid_flags, device="cuda:0"):
    """Load hand MANO predictions and compute j3d_ra.right and root.right.

    Hand fit data covers all original dataset frames. We first select the
    pipeline frames using ``frame_indices`` (like ``gt.load_data`` uses
    ``selected_fids``), then filter to valid frames using ``valid_flags``.

    Args:
        results_dir: Path to pipeline_joint_opt results directory
        hand_mode: Hand fit mode (e.g. 'trans', 'rot', 'intrinsic')
        frame_indices: (N,) array of frame IDs used by the pipeline
        valid_flags: (N,) boolean mask for valid frames within pipeline frames
        device: Torch device string

    Returns:
        Dict with 'j3d_ra.right' (torch, M,21,3) and 'root.right' (numpy, M,3)
        for valid frames, or None if hand data is unavailable.
    """
    hand_provider = HandDataProvider(results_dir)
    if not hand_provider.has_hand:
        print("[hand] No hand data available, skip hand metrics")
        return None

    hand_poses = hand_provider.get_hand_poses(hand_mode)
    beta = hand_provider.get_hand_beta(hand_mode)
    h2c_transls = hand_provider.get_hand_transls(hand_mode)
    h2c_rots = hand_provider.get_hand_rots(hand_mode)

    if hand_poses is None or beta is None or h2c_transls is None or h2c_rots is None:
        print(f"[hand] Missing hand parameters for mode '{hand_mode}', skip hand metrics")
        return None

    # Select only the pipeline frames from the full hand data (like gt.load_data)
    max_fid = max(int(np.max(frame_indices)), 0)
    if len(hand_poses) <= max_fid:
        print(f"[hand] Hand data length {len(hand_poses)} too short for max frame index {max_fid}, skip")
        return None

    hand_poses = hand_poses[frame_indices]
    h2c_transls = np.asarray(h2c_transls)[frame_indices]
    h2c_rots = np.asarray(h2c_rots)[frame_indices]

    hand_poses_t = torch.as_tensor(hand_poses, device=device, dtype=torch.float32)
    beta_t = torch.as_tensor(beta, device=device, dtype=torch.float32)
    betas_t = beta_t.unsqueeze(0).repeat(hand_poses_t.shape[0], 1)
    h2c_transls_np = np.asarray(h2c_transls)
    h2c_rots_t = torch.as_tensor(h2c_rots, device=device, dtype=torch.float32)

    mano_layer = smplx.create(
        model_path='./body_models/MANO_RIGHT.pkl',
        model_type="mano", use_pca=False, is_rhand=True,
    ).to(torch.device(device))

    with torch.no_grad():
        hand_out = mano_layer(
            betas=betas_t,
            hand_pose=hand_poses_t,
            transl=torch.zeros_like(
                torch.as_tensor(h2c_transls_np, device=device, dtype=torch.float32)),
            global_orient=h2c_rots_t,
        )

    hand_jnts_can = hand_out.joints.cpu().numpy()  # (N, 21, 3)
    hand_verts_can = hand_out.vertices.cpu().numpy()  # (N, V, 3)
    hand_faces = np.asarray(mano_layer.faces, dtype=np.int32)  # (F, 3)

    # Root-aligned canonical joints
    j3d_ra_right = hand_jnts_can - hand_jnts_can[:, 0:1, :]  # (N, 21, 3)

    # Hand joints in camera space
    if h2c_transls_np.ndim == 2 and h2c_transls_np.shape[1] == 3:
        h2c_transforms = np.repeat(np.eye(4)[None], h2c_transls_np.shape[0], axis=0)
        h2c_transforms[:, :3, 3] = h2c_transls_np
    else:
        h2c_transforms = h2c_transls_np
    hand_jnts_c = transform_points(hand_jnts_can, h2c_transforms)  # (N, 21, 3)
    hand_verts_c = transform_points(hand_verts_can, h2c_transforms)  # (N, V, 3)
    root_right = hand_jnts_c[:, 0, :]  # (N, 3)

    # Select valid frames
    j3d_ra_right = j3d_ra_right[valid_flags]
    root_right = root_right[valid_flags]
    hand_verts_c = hand_verts_c[valid_flags]

    return {
        "j3d_ra.right": torch.from_numpy(j3d_ra_right).float(),
        "root.right": root_right.astype(np.float32),
        "v3d_c.right": hand_verts_c.astype(np.float32),
        "faces.right": hand_faces,
    }


def main(args):
    from tqdm import tqdm

    results_dir = Path(args.result_folder)
    SAM3D_dir = Path(args.SAM3D_dir)
    (data_pred, frame_indices, register_indices, c2o_pred, scale,
     sam3d_to_cond_cam, valid_flags) = load_pred_data(results_dir, SAM3D_dir, args.cond_index)

    seq_name = results_dir.parent.name

    # Load hand predictions (before GT filtering so they get filtered together)
    hand_data = load_hand_predictions(SAM3D_dir.parent, args.hand_mode, frame_indices, valid_flags)
    if hand_data is not None:
        data_pred["j3d_ra.right"] = hand_data["j3d_ra.right"]
        data_pred["root.right"] = hand_data["root.right"]
        data_pred["v3d_c.right"] = hand_data["v3d_c.right"]
        data_pred["faces.right"] = hand_data["faces.right"]

    def get_image_fids():
        return data_pred["valid_frame_indices"].tolist()

    data_gt = gt.load_data(seq_name, get_image_fids)

    # Filter out frames that are invalid in GT from both data_gt and data_pred
    data_gt, data_pred = filter_invalid_gt_frames(data_gt, data_pred)

    gt_o2c_all = data_gt["o2c"].numpy() if torch.is_tensor(data_gt["o2c"]) else np.array(data_gt["o2c"])
    aligned_pred_o2c = align_pred_to_gt(
        data_pred["extrinsics"], gt_o2c_all, data_pred["valid_frame_indices"],
        args.cond_index, register_indices,
    )

    data_pred["extrinsics"] = aligned_pred_o2c

    # Compute v3d_right.object (object verts relative to hand root) if hand data available
    if "root.right" in data_pred:
        v3d_can = data_gt.get("v3d_can.object")
        if v3d_can is not None:
            v3d_can_np = v3d_can.numpy() if torch.is_tensor(v3d_can) else np.array(v3d_can)
            root_right = data_pred["root.right"]
            if torch.is_tensor(root_right):
                root_right = root_right.numpy()
            v3d_right_list = []
            for i in range(len(aligned_pred_o2c)):
                o2c_i = aligned_pred_o2c[i]
                v_can_i = v3d_can_np[0] if v3d_can_np.ndim == 3 else v3d_can_np
                v_cam = (o2c_i[:3, :3] @ v_can_i.T).T + o2c_i[:3, 3]
                v_right = v_cam - root_right[i]
                v3d_right_list.append(torch.from_numpy(v_right.astype(np.float32)))
            data_pred["v3d_right.object"] = v3d_right_list

    visualize_gt_and_pred_in_rerun(
        data_gt, data_pred, aligned_pred_o2c, data_pred["valid_frame_indices"], SAM3D_dir,
        cond_index=args.cond_index,
        frame_indices_full=frame_indices,
        c2o_full=c2o_pred,
        scale=scale,
        sam3d_to_cond_cam=sam3d_to_cond_cam,
        jpeg_quality=args.jpeg_quality, world_mode=args.world_mode,
        history_window=args.history_window,
    )
    

    out_p = args.out_dir
    os.makedirs(out_p, exist_ok=True)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="")
    parser.add_argument("--SAM3D_dir", type=str, required=True, help="Path to SAM3D_aligned_post_process directory")
    parser.add_argument("--cond_index", type=int, required=True,help="Condition frame index")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--jpeg_quality", type=int, default=85,
                         help="JPEG quality for compressed image logging to rerun (1-100)")
    parser.add_argument("--hand_mode", type=str, default="trans",
                         help="Hand fit mode for HandDataProvider (e.g. 'rot', 'trans', 'intrinsic')")
    parser.add_argument("--world_mode", type=str, default="camera",
                         choices=["camera", "object"],
                         help="'camera': camera fixed, object moves. "
                              "'object': object fixed at identity, cameras move.")
    parser.add_argument("--history_window", type=int, default=50,
                         help="Number of past camera frustums to show in object mode")
    
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
