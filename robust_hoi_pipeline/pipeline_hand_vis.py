import numpy as np

import json
import os
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
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform
from viewer.viewer_step import HandDataProvider
from utils_simba.geometry import transform_points
device = "cuda:0"



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


def find_joint_opt_mesh_from_ckpt(joint_opt_ckpt: Path):
    """Find an exported NeuS mesh near the fixed checkpoint path."""
    if not joint_opt_ckpt.exists():
        return None

    save_dir = joint_opt_ckpt.parent.parent / "save"
    if not save_dir.exists():
        return None

    mesh_candidates = sorted(save_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mesh_candidates[0] if mesh_candidates else None




def visualize_in_rerun(extrinsics, frame_indices, valid_flags, SAM3D_dir, cond_index, scale):
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
            rr.log(f"{entity}/camera", rr.Image(img))



def align_pred_to_gt(valid_extrinsics, gt_o2c, valid_frame_indices,
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
    align_tf = np.linalg.inv(valid_extrinsics[anchor_idx]) @ gt_o2c[anchor_idx]
    return valid_extrinsics @ align_tf


def visualize_gt_and_pred_in_rerun(data_gt, pred_extrinsics, frame_indices, SAM3D_dir):
    """Visualize GT and predicted poses with rotated 3D points in rerun.

    For each frame, transforms the GT canonical mesh vertices by the o2c pose
    (object-to-camera) and logs them as colored point clouds alongside camera
    intrinsics and images for both GT and predicted poses.

    Args:
        data_gt: Ground truth data dict (from gt.load_data) with keys:
            mesh_name.object, o2c, K, is_valid
        pred_extrinsics: (M, 4, 4) predicted object-to-camera matrices for valid frames
        frame_indices: (M,) frame indices corresponding to pred_extrinsics
        SAM3D_dir: Path to SAM3D_aligned_post_process directory
    """
    import rerun as rr
    rr.init("pipeline_joint_opt_eval", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load GT mesh canonical vertices
    gt_mesh_path = data_gt["mesh_name.object"]
    gt_mesh = trimesh.load(str(gt_mesh_path), force='mesh') if os.path.exists(gt_mesh_path) else None
    gt_verts_can = np.array(gt_mesh.vertices, dtype=np.float32) if gt_mesh is not None else None

    gt_o2c = data_gt["o2c"].numpy() if torch.is_tensor(data_gt["o2c"]) else np.array(data_gt["o2c"])
    gt_is_valid = data_gt["is_valid"].numpy() if torch.is_tensor(data_gt["is_valid"]) else np.array(data_gt["is_valid"])
    gt_K = data_gt["K"].numpy() if torch.is_tensor(data_gt["K"]) else np.array(data_gt["K"])
    data_preprocess_dir = SAM3D_dir.parent / "pipeline_preprocess"

    for i, fid in enumerate(frame_indices):
        rr.set_time_sequence("frame", i)

        preprocess_data = load_preprocessed_frame(data_preprocess_dir, fid)
        img = preprocess_data.get("image")
        K_pred = preprocess_data.get("intrinsics")

        # Predicted: rotate vertices by predicted o2c, log as blue points
        pred_o2c = pred_extrinsics[i].astype(np.float32)
        if gt_verts_can is not None:
            pred_verts_cam = (pred_o2c[:3, :3] @ gt_verts_can.T).T + pred_o2c[:3, 3]
            rr.log(
                "world/pred_points",
                rr.Points3D(
                    pred_verts_cam.astype(np.float32),
                    colors=np.broadcast_to(
                        np.array([[0, 0, 255]], dtype=np.uint8), (len(pred_verts_cam), 3)),
                    radii=0.0003,
                ),
            )
        pred_entity = "world/pred_camera"
        rr.log(pred_entity, rr.Transform3D(
            translation=np.zeros_like(pred_o2c[:3, 3]), mat3x3=np.eye(3)))
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
            rr.log(f"{pred_entity}/camera", rr.Image(img))

        # GT: rotate vertices by GT o2c, log as green points
        if i < len(gt_o2c) and bool(gt_is_valid[i]):
            gt_o2c_i = gt_o2c[i].astype(np.float32)
            if gt_verts_can is not None:
                gt_verts_cam = (gt_o2c_i[:3, :3] @ gt_verts_can.T).T + gt_o2c_i[:3, 3]
                rr.log(
                    "world/gt_points",
                    rr.Points3D(
                        gt_verts_cam.astype(np.float32),
                        colors=np.broadcast_to(
                            np.array([[0, 255, 0]], dtype=np.uint8), (len(gt_verts_cam), 3)),
                        radii=0.0003,
                    ),
                )
            gt_entity = "world/gt_camera"
            rr.log(gt_entity, rr.Transform3D(
                translation=np.zeros_like(gt_o2c_i[:3, 3]), mat3x3=np.eye(3)))
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
                rr.log(f"{gt_entity}/camera", rr.Image(img))


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
    image_info = np.load(
        results_dir / f"{last_register_idx:04d}" / "image_info.npy",
        allow_pickle=True,
    ).item()

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


def load_hand_predictions(results_dir, hand_mode, frame_indices, device="cuda:0"):
    """Load hand MANO predictions and compute j3d_ra.right and root.right.

    Hand fit data covers all original dataset frames. We first select the
    pipeline frames using ``frame_indices`` (like ``gt.load_data`` uses
    ``selected_fids``), then filter to valid frames using ``valid_flags``.

    Args:
        results_dir: Path to pipeline_joint_opt results directory
        hand_mode: Hand fit mode (e.g. 'trans', 'rot', 'intrinsic')
        frame_indices: (N,) array of frame IDs used by the pipeline
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
    hand_verts_can = hand_out.vertices.cpu().numpy()  # (N, 778, 3)
    hand_faces = np.asarray(mano_layer.faces, dtype=np.int64).copy()  # (F, 3)

    # Root-aligned canonical joints
    j3d_ra_right = hand_jnts_can - hand_jnts_can[:, 0:1, :]  # (N, 21, 3)

    # Hand joints in camera space
    if h2c_transls_np.ndim == 2 and h2c_transls_np.shape[1] == 3:
        h2c_transforms = np.repeat(np.eye(4)[None], h2c_transls_np.shape[0], axis=0)
        h2c_transforms[:, :3, 3] = h2c_transls_np
    else:
        h2c_transforms = h2c_transls_np
    hand_jnts_c = transform_points(hand_jnts_can, h2c_transforms)  # (N, 21, 3)
    hand_verts_c = transform_points(hand_verts_can, h2c_transforms)  # (N, 778, 3)
    root_right = hand_jnts_c[:, 0, :]  # (N, 3)

    # Build full o2c (object-to-camera) 4x4 matrices from h2c_rots + h2c_transls
    from scipy.spatial.transform import Rotation as Rot
    h2c_rots_np = np.asarray(h2c_rots)
    rot_mats = Rot.from_rotvec(h2c_rots_np).as_matrix()  # (N, 3, 3)
    o2c = np.repeat(np.eye(4, dtype=np.float32)[None], len(h2c_rots_np), axis=0)
    o2c[:, :3, :3] = rot_mats
    o2c[:, :3, 3] = h2c_transls_np

    return {
        "j3d_ra.right": torch.from_numpy(j3d_ra_right).float(),
        "root.right": root_right.astype(np.float32),
        "v3d_c.right": hand_verts_c.astype(np.float32),
        "faces.right": hand_faces,
        "o2c": o2c.astype(np.float32),
    }


def visualize_hand_in_rerun(data_gt, hand_pred_data, valid_frame_indices, data_dir,
                            vis_space="object", pred_align="GT", cond_index=0, sam3d_data=None):
    """Visualize GT and predicted hand meshes per frame in Rerun.

    For each valid frame, logs:
    - GT hand mesh (green) and GT object mesh (gray) from ground truth data
    - Predicted hand mesh (blue) from MANO forward pass with predicted parameters
    - SAM3D mesh (orange) if sam3d_data is provided
    - Camera image with pinhole projection

    Args:
        data_gt: Ground truth data dict from gt.load_data (filtered to valid frames)
        hand_pred_data: Dict with 'v3d_c.right', 'faces.right', 'o2c',
            or None if unavailable
        valid_frame_indices: (M,) frame indices for valid frames
        data_dir: Path to HO3D sequence directory containing rgb/
        vis_space: 'object' or 'camera'
        pred_align: 'GT' or 'SAM3D' — which reference to align pred poses to
        cond_index: condition frame index (used for SAM3D alignment anchor)
        sam3d_data: dict with 'sam3d_to_cond_cam', 'scale', 'mesh_path', or None
    """
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            column_shares=[2, 1],
        ),
    )
    rr.init("pipeline_hand_vis", spawn=True, default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Extract GT arrays (may be torch tensors after xdict.to_torch)
    def _to_np(v):
        if v is None:
            return None
        return v.numpy() if torch.is_tensor(v) else np.asarray(v)

    gt_v3d_h = _to_np(data_gt.get("v3d_c.right"))
    gt_faces_h = _to_np(data_gt.get("faces.right"))
    gt_is_valid = _to_np(data_gt.get("is_valid"))
    gt_v3d_o = _to_np(data_gt.get("v3d_c.object"))
    gt_faces_o = _to_np(data_gt.get("faces.object"))

    pred_v3d_h = hand_pred_data.get("v3d_c.right") if hand_pred_data else None
    pred_faces_h = hand_pred_data.get("faces.right") if hand_pred_data else None
    pred_o2c = hand_pred_data.get("o2c").copy() if hand_pred_data and hand_pred_data.get("o2c") is not None else None
    gt_o2c = _to_np(data_gt.get("o2c"))  # (M, 4, 4) GT object-to-camera
    gt_K_raw = _to_np(data_gt.get("K"))
    gt_K = gt_K_raw.reshape(3, 3) if gt_K_raw is not None else None  # single (3,3) for whole seq
    rgb_dir = Path(data_dir) / "rgb"
    DUMMY_THRESH = -500  # gt.py sets invalid verts to -1000


    # Load and log SAM3D mesh as static (orange)
    if sam3d_data is not None and sam3d_data["mesh_path"].exists() and vis_space == "camera":
        sam3d_mesh = trimesh.load(str(sam3d_data["mesh_path"]), force='mesh')
        sam3d_verts = np.array(sam3d_mesh.vertices, dtype=np.float32)
        sam3d_faces = np.array(sam3d_mesh.faces, dtype=np.uint32)
        sam3d_colors = None
        if sam3d_mesh.visual is not None and hasattr(sam3d_mesh.visual, 'vertex_colors'):
            sam3d_colors = np.array(sam3d_mesh.visual.vertex_colors)[:, :3]
        sam3d_to_cond_cam = sam3d_data["sam3d_to_cond_cam"]

        # Transform SAM3D mesh vertices to condition camera space
        verts_homo = np.hstack([sam3d_verts, np.ones((len(sam3d_verts), 1), dtype=np.float32)])
        sam3d_verts = (sam3d_to_cond_cam @ verts_homo.T).T[:, :3]

        mesh_kwargs = dict(
            vertex_positions=sam3d_verts,
            triangle_indices=sam3d_faces,
        )
        if sam3d_colors is not None:
            mesh_kwargs["vertex_colors"] = sam3d_colors
        else:
            mesh_kwargs["mesh_material"] = rr.Material(albedo_factor=[255, 165, 0])
        rr.log("world/sam3d_mesh", rr.Mesh3D(**mesh_kwargs), static=True)

    for i, fid in enumerate(valid_frame_indices):
        fid = int(fid)
        rr.set_time_sequence("frame", i)

        valid = bool(gt_is_valid[i]) if gt_is_valid is not None and i < len(gt_is_valid) else False

        # Compute GT c2o (camera-to-object) for this frame
        gt_c2o = None
        if valid and gt_o2c is not None and i < len(gt_o2c):
            gt_c2o = np.linalg.inv(gt_o2c[i]).astype(np.float32)

        # Compute pred c2o for this frame
        pred_c2o = None
        if pred_o2c is not None and i < len(pred_o2c):
            pred_c2o = np.linalg.inv(pred_o2c[i]).astype(np.float32)

        if vis_space == "object":
            # Log camera poses in object/world space
            if gt_c2o is not None:
                rr.log("world/gt_camera", rr.Transform3D(
                    translation=gt_c2o[:3, 3],
                    mat3x3=gt_c2o[:3, :3],
                ))
            if pred_c2o is not None:
                rr.log("world/pred_camera", rr.Transform3D(
                    translation=pred_c2o[:3, 3],
                    mat3x3=pred_c2o[:3, :3],
                ))

        # Load image from data_dir/rgb/
        img = None
        for ext in (".jpg", ".png", ".jpeg"):
            img_path = rgb_dir / f"{fid:04d}{ext}"
            if img_path.exists():
                from PIL import Image as PILImage
                img = np.array(PILImage.open(img_path).convert("RGB"))
                break

        # Get intrinsics from GT data (single K for whole sequence)
        K = gt_K

        # Log pinhole + image on GT camera
        if img is not None and K is not None:
            H, W = img.shape[:2]
            pinhole = rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(K[0, 0]), float(K[1, 1])],
                principal_point=[float(K[0, 2]), float(K[1, 2])],
                image_plane_distance=1.0,
            )
            rr.log("world/gt_camera/cam", pinhole)
            rr.log("world/gt_camera/cam", rr.Image(img))

        if img is not None and K is not None:
            H, W = img.shape[:2]
            rr.log("world/pred_camera/cam", rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(K[0, 0]), float(K[1, 1])],
                principal_point=[float(K[0, 2]), float(K[1, 2])],
                image_plane_distance=1.0,
            ))
            rr.log("world/pred_camera/cam", rr.Image(img))

        # GT hand mesh (green)
        if valid and gt_v3d_h is not None and gt_faces_h is not None and i < len(gt_v3d_h):
            verts = gt_v3d_h[i].astype(np.float32)
            if verts.min() > DUMMY_THRESH:
                if vis_space == "object" and gt_c2o is not None:
                    verts = (gt_c2o[:3, :3] @ verts.T).T + gt_c2o[:3, 3]
                rr.log("world/gt_hand", rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=gt_faces_h.astype(np.uint32),
                    mesh_material=rr.Material(albedo_factor=[120, 220, 120]),
                ))

        # GT object mesh (gray)
        if valid and gt_v3d_o is not None and gt_faces_o is not None and i < len(gt_v3d_o):
            verts_o = gt_v3d_o[i].astype(np.float32)
            if verts_o.min() > DUMMY_THRESH:
                if vis_space == "object" and gt_c2o is not None:
                    verts_o = (gt_c2o[:3, :3] @ verts_o.T).T + gt_c2o[:3, 3]
                rr.log("world/gt_object", rr.Mesh3D(
                    vertex_positions=verts_o,
                    triangle_indices=gt_faces_o.astype(np.uint32),
                    mesh_material=rr.Material(albedo_factor=[200, 200, 200]),
                ))

        # Predicted hand mesh (blue)
        if pred_v3d_h is not None and pred_faces_h is not None and i < len(pred_v3d_h):
            verts_pred = pred_v3d_h[i].astype(np.float32)
            if vis_space == "object" and pred_c2o is not None:
                verts_pred = (pred_c2o[:3, :3] @ verts_pred.T).T + pred_c2o[:3, 3]
            rr.log("world/pred_hand", rr.Mesh3D(
                vertex_positions=verts_pred,
                triangle_indices=pred_faces_h.astype(np.uint32),
                mesh_material=rr.Material(albedo_factor=[120, 120, 220]),
            ))

    print(f"[hand_vis] Logged {len(valid_frame_indices)} frames to Rerun")


def load_sam3d_data(sam3d_dir: Path, cond_index: int):
    """Load SAM3D mesh path and transform data.

    Returns dict with 'sam3d_to_cond_cam', 'scale', 'mesh_path', or None if not found.
    """
    sam3d_mesh_path = sam3d_dir / f"{cond_index:04d}" / "mesh.obj"
    try:
        sam3d_transform = load_sam3d_transform(sam3d_dir, cond_index)
        sam3d_data = {
            "sam3d_to_cond_cam": sam3d_transform["sam3d_to_cond_cam"],
            "scale": sam3d_transform["scale"],
            "mesh_path": sam3d_mesh_path,
        }
        print(f"[hand_vis] Loaded SAM3D transform from {sam3d_dir}, scale={sam3d_data['scale']:.4f}")
        return sam3d_data
    except FileNotFoundError as e:
        print(f"[hand_vis] SAM3D transform not found: {e}")
        return None


def main(args):

    data_dir = Path(args.data_dir)
    seq_name = data_dir.name
    SAM3D_dir = data_dir / "SAM3D_aligned_post_process" 

    # Auto-detect total frames from rgb directory and cap end
    rgb_dir = data_dir / "rgb"
    total_frames = len([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    end = min(args.end, total_frames)

    # Generate frame indices from args
    frame_indices = np.arange(args.begin, end, args.interval)
    # Ensure cond_index is included                                                                   
    if args.cond_index not in frame_indices and args.cond_index < total_frames:                       
        frame_indices = np.sort(np.append(frame_indices, args.cond_index))    

    print(f"[hand_vis] {total_frames} frames in {rgb_dir}, using {len(frame_indices)} frames "
          f"(begin={args.begin}, end={end}, interval={args.interval})")

    # Load hand predictions
    hand_data = load_hand_predictions(data_dir, args.hand_mode, frame_indices)

    # Build minimal data_pred dict for GT filtering
    valid_frame_indices = frame_indices.copy()
    data_pred = {
        "valid_frame_indices": frame_indices.copy(),
        "is_valid": np.ones(len(valid_frame_indices), dtype=np.float32),
    }
    if hand_data is not None:
        data_pred["v3d_c.right"] = hand_data["v3d_c.right"]
        data_pred["faces.right"] = hand_data["faces.right"]

    def get_image_fids():
        return valid_frame_indices.tolist()

    data_gt = gt.load_data(seq_name, get_image_fids)

    # Filter out frames that are invalid in GT from both data_gt and data_pred
    data_gt, data_pred = filter_invalid_gt_frames(data_gt, data_pred)

    sam3d_data = load_sam3d_data(SAM3D_dir, args.cond_index)

    # Visualize GT and predicted hand meshes in Rerun
    visualize_hand_in_rerun(
        data_gt, hand_data, data_pred["valid_frame_indices"], data_dir,
        vis_space=args.vis_space,
        pred_align=args.pred_align,
        cond_index=args.cond_index,
        sam3d_data=sam3d_data,
    )
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="HO3D sequence directory (e.g. ho3d_v3/train/MC1)")
    parser.add_argument("--cond_index", type=int, default=0)
    parser.add_argument("--begin", type=int, default=0,
                        help="Start frame index")
    parser.add_argument("--end", type=int, default=10000,
                        help="End frame index (exclusive)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Frame sampling interval")
    parser.add_argument("--hand_mode", type=str, default="trans",
                         help="Hand fit mode for HandDataProvider (e.g. 'rot', 'trans', 'intrinsic')")
    parser.add_argument("--vis_space", type=str, default="camera", choices=["object", "camera"],
                         help="Visualization space: 'object' transforms meshes to object space, "
                              "'camera' keeps meshes in camera space")
    parser.add_argument("--pred_align", type=str, default="SAM3D", choices=["GT", "SAM3D"],
                         help="Align predicted poses to 'GT' or 'SAM3D' reference")

    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    main(args)
