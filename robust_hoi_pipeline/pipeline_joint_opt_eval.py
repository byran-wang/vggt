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
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform, load_preprocessed_frame
from viewer.viewer_step import HandDataProvider
from utils_simba.geometry import transform_points
device = "cuda:0"

eval_fn_dict = {
    "add": eval_m.eval_add_object,
    "add_s": eval_m.eval_add_s_object,
    "add_auc": eval_m.eval_add_auc_object,
    "add_s_auc": eval_m.eval_add_s_auc_object,
    "image_info": eval_m.eval_image_info,
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    "cd_f_right": eval_m.eval_cd_f_right,
}


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


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="")
    parser.add_argument("--SAM3D_dir", type=str, required=True, help="Path to SAM3D_aligned_post_process directory")
    parser.add_argument("--cond_index", type=int, required=True,help="Condition frame index")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--mesh_type", type=str, default="sam3d",
                        help="Type of mesh for evaluation: hy_omni or sam3d")
    parser.add_argument("--eval_mesh_chamfer", action="store_true", default=False,
                         help="Whether to evaluate Chamfer/F-score metrics for available meshes")
    parser.add_argument("--vis_gt_pred", action="store_true", default=False,
                         help="Whether to visualize GT and predicted poses with rotated 3D points in rerun")
    parser.add_argument("--hand_mode", type=str, default="trans",
                         help="Hand fit mode for HandDataProvider (e.g. 'rot', 'trans', 'intrinsic')")
    
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


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


def eval_mesh_chamfer(
    metric_dict, data_gt, results_dir, SAM3D_dir, args,
    frame_indices, valid_extrinsics, c2o, scale, sam3d_to_cond_cam,
):
    """Evaluate Chamfer/F-score metrics for available meshes.

    Args:
        metric_dict: Dictionary to update with prefixed Chamfer metrics (modified in-place)
        data_gt: Ground truth data dictionary
        results_dir: Path to pipeline_joint_opt results directory
        SAM3D_dir: Path to SAM3D_aligned_post_process directory
        args: Parsed arguments (needs cond_index, out_dir)
        frame_indices: (N,) frame index array
        valid_extrinsics: (M, 4, 4) object-to-camera matrices for valid frames
        c2o: (N, 4, 4) camera-to-object matrices
        scale: SAM3D-to-metric scale factor
        sam3d_to_cond_cam: (4, 4) SAM3D-to-condition-camera transform
    """
    hy_omni_mesh = results_dir.parent / "pipeline_HY_to_SAM3D" / "HY_omni_in_sam3d.obj"
    sam3d_mesh = SAM3D_dir.parent / "SAM3D" / f"{args.cond_index:04d}" / "scene.glb"
    joint_opt_ckpt = (
        results_dir.parent / "pipeline_neus_init" / "neus_training" / "joint_opt" / "ckpt" / "epoch=0-step=10000.ckpt"
    )
    joint_opt_mesh = find_joint_opt_mesh_from_ckpt(joint_opt_ckpt)

    mesh_specs = [
        ("sam3d", sam3d_mesh),
        ("joint_opt", joint_opt_mesh),
        ("hy_omni", hy_omni_mesh),
    ]

    for prefix, mesh_path in mesh_specs:
        if mesh_path is None or not mesh_path.exists():
            if prefix == "joint_opt" and joint_opt_ckpt.exists():
                print(f"Joint-opt checkpoint exists at {joint_opt_ckpt}, but no exported mesh found under save/")
            else:
                print(f"[{prefix}] Mesh not found, skip Chamfer evaluation")
            continue

        pred_mesh_data = build_mesh_object_predictions(
            mesh_path=mesh_path,
            frame_indices=frame_indices,
            valid_extrinsics=valid_extrinsics,
            c2o=c2o,
            scale=scale,
            sam3d_to_cond_cam=sam3d_to_cond_cam,
            cond_index=args.cond_index,
        )
        if pred_mesh_data is None:
            print(f"[{prefix}] Failed to prepare predicted mesh vertices, skip Chamfer evaluation")
            continue

        pred_mesh_data["out_dir"] = args.out_dir
        local_metric_dict = {}
        try:
            local_metric_dict = eval_m.eval_icp_first_frame(pred_mesh_data, data_gt, local_metric_dict)
        except Exception as e:
            print(f"[{prefix}] ICP evaluation failed ({e}), continue with Chamfer only")
        local_metric_dict = eval_m.eval_cd_f_right(pred_mesh_data, data_gt, local_metric_dict)
        for k, v in local_metric_dict.items():
            metric_dict[f"{prefix}_{k}"] = v
        print(f"[{prefix}] Added Chamfer metrics from {mesh_path}")


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

    # Root-aligned canonical joints
    j3d_ra_right = hand_jnts_can - hand_jnts_can[:, 0:1, :]  # (N, 21, 3)

    # Hand joints in camera space
    if h2c_transls_np.ndim == 2 and h2c_transls_np.shape[1] == 3:
        h2c_transforms = np.repeat(np.eye(4)[None], h2c_transls_np.shape[0], axis=0)
        h2c_transforms[:, :3, 3] = h2c_transls_np
    else:
        h2c_transforms = h2c_transls_np
    hand_jnts_c = transform_points(hand_jnts_can, h2c_transforms)  # (N, 21, 3)
    root_right = hand_jnts_c[:, 0, :]  # (N, 3)

    # Select valid frames
    j3d_ra_right = j3d_ra_right[valid_flags]
    root_right = root_right[valid_flags]

    return {
        "j3d_ra.right": torch.from_numpy(j3d_ra_right).float(),
        "root.right": root_right.astype(np.float32),
    }


def main():
    from tqdm import tqdm
    args = parse_args()

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

    def get_image_fids():
        return data_pred["valid_frame_indices"].tolist()

    data_gt = gt.load_data(seq_name, get_image_fids)

    # Filter out frames that are invalid in GT from both data_gt and data_pred
    data_gt, data_pred = filter_invalid_gt_frames(data_gt, data_pred)

    gt_o2c_all = data_gt["o2c"].numpy() if torch.is_tensor(data_gt["o2c"]) else np.array(data_gt["o2c"])
    aligned_pred_extrinsics = align_pred_to_gt(
        data_pred["extrinsics"], gt_o2c_all, data_pred["valid_frame_indices"],
        args.cond_index, register_indices,
    )

    data_pred["extrinsics"] = aligned_pred_extrinsics

    # Compute v3d_right.object (object verts relative to hand root) if hand data available
    if "root.right" in data_pred:
        v3d_can = data_gt.get("v3d_can.object")
        if v3d_can is not None:
            v3d_can_np = v3d_can.numpy() if torch.is_tensor(v3d_can) else np.array(v3d_can)
            root_right = data_pred["root.right"]
            if torch.is_tensor(root_right):
                root_right = root_right.numpy()
            v3d_right_list = []
            for i in range(len(aligned_pred_extrinsics)):
                o2c_i = aligned_pred_extrinsics[i]
                v_can_i = v3d_can_np[0] if v3d_can_np.ndim == 3 else v3d_can_np
                v_cam = (o2c_i[:3, :3] @ v_can_i.T).T + o2c_i[:3, 3]
                v_right = v_cam - root_right[i]
                v3d_right_list.append(torch.from_numpy(v_right.astype(np.float32)))
            data_pred["v3d_right.object"] = v3d_right_list

    if args.vis_gt_pred:
        visualize_gt_and_pred_in_rerun(
            data_gt, aligned_pred_extrinsics, data_pred["valid_frame_indices"], SAM3D_dir,
        )
    

    out_p = args.out_dir
    os.makedirs(out_p, exist_ok=True)


    print("------------------")
    print("Involving the following eval_fn:")
    active_eval_fns = {}
    hand_eval_keys = {"mpjpe_ra_r", "cd_f_right"}
    for eval_fn_name, eval_fn in eval_fn_dict.items():
        if eval_fn_name in hand_eval_keys and "j3d_ra.right" not in data_pred:
            print(f"  {eval_fn_name} (SKIPPED - no hand data)")
            continue
        active_eval_fns[eval_fn_name] = eval_fn
        print(f"  {eval_fn_name}")
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(active_eval_fns.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Optional Chamfer/F-score metrics from available meshes, with prefixed keys.
    if args.eval_mesh_chamfer:
        eval_mesh_chamfer(
            metric_dict, data_gt, results_dir, SAM3D_dir, args,
            frame_indices, data_pred["extrinsics"], c2o_pred, scale, sam3d_to_cond_cam,
        )

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + "/metric.json"
    npy_path = out_p + "/metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm), F-score (percentage), MPJPE (mm)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()
