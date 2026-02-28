import numpy as np

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

import vggt.utils.gt as gt
import torch
import trimesh
import smplx
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform
from viewer.viewer_step import HandDataProvider
from utils_simba.geometry import transform_points
from utils_simba.depth import get_depth


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
    # Filter to frames that are within the hand data range
    max_fid = max(int(np.max(frame_indices)), 0)
    hand_len = len(hand_poses)
    if hand_len <= max_fid:
        valid_mask = frame_indices < hand_len
        print(f"[hand] Hand data length {hand_len} < max frame index {max_fid}, "
              f"using {valid_mask.sum()}/{len(frame_indices)} frames")
        frame_indices = frame_indices[valid_mask]
        if len(frame_indices) == 0:
            print("[hand] No valid frames within hand data range, skip")
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
        "frame_indices": frame_indices.copy(),
    }


def visualize_hand_in_rerun(data_gt, hand_pred_data, valid_frame_indices, data_dir,
                            vis_space="object", pred_align="GT", cond_index=0, sam3d_data=None,
                            jpeg_quality=85):
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

    # Build frame_id → hand data index mapping (hand data may cover fewer frames)
    hand_fid_to_idx = {}
    if hand_pred_data is not None and "frame_indices" in hand_pred_data:
        for hi, hfid in enumerate(hand_pred_data["frame_indices"]):
            hand_fid_to_idx[int(hfid)] = hi

    # Align pred_o2c to gt_o2c at the first valid frame that has hand data
    if pred_align == "GT" and pred_o2c is not None and gt_o2c is not None:
        anchor = None
        if gt_is_valid is not None:
            valid_indices = np.where(gt_is_valid.astype(bool))[0]
        else:
            valid_indices = np.arange(len(valid_frame_indices))
        for ai in valid_indices:
            anchor_fid = int(valid_frame_indices[ai])
            if anchor_fid in hand_fid_to_idx:
                anchor = ai
                break
        if anchor is not None:
            hand_anchor = hand_fid_to_idx[int(valid_frame_indices[anchor])]
            align_tf = np.linalg.inv(pred_o2c[hand_anchor]) @ gt_o2c[anchor]
            pred_o2c = pred_o2c @ align_tf
            print(f"[hand_vis] Aligned pred_o2c to gt_o2c at frame {anchor}")
        else:
            print("[hand_vis] No valid frame with hand data for alignment, skipping alignment")
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

    # Log GT object mesh once (rigid body — shape doesn't change, only pose)
    gt_obj_logged = False
    if gt_v3d_o is not None and gt_faces_o is not None:
        for first_i in range(len(gt_v3d_o)):
            if gt_is_valid is not None and not bool(gt_is_valid[first_i]):
                continue
            verts_o = gt_v3d_o[first_i].astype(np.float32)
            if verts_o.min() <= DUMMY_THRESH:
                continue
            # Compute canonical object-space vertices from first valid frame
            if gt_o2c is not None and first_i < len(gt_o2c):
                c2o = np.linalg.inv(gt_o2c[first_i]).astype(np.float32)
                verts_obj = (c2o[:3, :3] @ verts_o.T).T + c2o[:3, 3]
            else:
                verts_obj = verts_o
            rr.log("world/gt_object/mesh", rr.Mesh3D(
                vertex_positions=verts_obj,
                triangle_indices=gt_faces_o.astype(np.uint32),
                mesh_material=rr.Material(albedo_factor=[200, 200, 200]),
            ), static=True)
            gt_obj_logged = True
            break

    for i, fid in enumerate(valid_frame_indices):
        fid = int(fid)
        rr.set_time_sequence("frame", i)

        valid = bool(gt_is_valid[i]) if gt_is_valid is not None and i < len(gt_is_valid) else False

        # Compute GT c2o (camera-to-object) for this frame
        gt_c2o = None
        if valid and gt_o2c is not None and i < len(gt_o2c):
            gt_c2o = np.linalg.inv(gt_o2c[i]).astype(np.float32)

        # Compute pred c2o for this frame (hand data may cover fewer frames)
        pred_c2o = None
        hi = hand_fid_to_idx.get(fid)
        if pred_o2c is not None and hi is not None:
            pred_c2o = np.linalg.inv(pred_o2c[hi]).astype(np.float32)

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
            rr.log("world/gt_camera/cam", rr.Image(img).compress(jpeg_quality=jpeg_quality))

        if img is not None and K is not None:
            H, W = img.shape[:2]
            rr.log("world/pred_camera/cam", rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(K[0, 0]), float(K[1, 1])],
                principal_point=[float(K[0, 2]), float(K[1, 2])],
                image_plane_distance=1.0,
            ))
            rr.log("world/pred_camera/cam", rr.Image(img).compress(jpeg_quality=jpeg_quality))

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

        # GT object mesh pose (mesh geometry logged once before the loop)
        if gt_obj_logged and valid and gt_o2c is not None and i < len(gt_o2c):
            if vis_space == "camera":
                # Camera space: transform canonical mesh by o2c
                o2c = gt_o2c[i].astype(np.float32)
                rr.log("world/gt_object", rr.Transform3D(
                    translation=o2c[:3, 3], mat3x3=o2c[:3, :3],
                ))

        # Predicted hand mesh (blue)
        if pred_v3d_h is not None and pred_faces_h is not None and hi is not None:
            verts_pred = pred_v3d_h[hi].astype(np.float32)
            if vis_space == "object" and pred_c2o is not None:
                verts_pred = (pred_c2o[:3, :3] @ verts_pred.T).T + pred_c2o[:3, 3]
            rr.log("world/pred_hand", rr.Mesh3D(
                vertex_positions=verts_pred,
                triangle_indices=pred_faces_h.astype(np.uint32),
                mesh_material=rr.Material(albedo_factor=[120, 120, 220]),
            ))

        # Hand depth point cloud from depth + hand_mask
        depth_path = Path(data_dir) / "depth" / f"{fid:04d}.png"
        mask_hand_path = Path(data_dir) / "mask_hand" / f"{fid:04d}.png"
        if depth_path.exists() and mask_hand_path.exists() and K is not None:
            from PIL import Image as PILImage
            depth_map = get_depth(str(depth_path))
            mask_hand = np.array(PILImage.open(mask_hand_path).convert("L"))
            valid = (depth_map > 0) & (mask_hand > 0)
            ys, xs = np.where(valid)
            if len(ys) > 0:
                max_pts = 5000
                if len(ys) > max_pts:
                    sel = np.random.choice(len(ys), max_pts, replace=False)
                    ys, xs = ys[sel], xs[sel]
                zs = depth_map[ys, xs].astype(np.float32)
                xc = (xs.astype(np.float32) - K[0, 2]) * zs / K[0, 0]
                yc = (ys.astype(np.float32) - K[1, 2]) * zs / K[1, 1]
                pts_cam = np.stack([xc, yc, zs], axis=-1)

                # Camera space (magenta)
                rr.log("world/pred_camera/cam/depth_hand_cam", rr.Points3D(
                    pts_cam,
                    colors=np.broadcast_to(np.array([200, 0, 255], dtype=np.uint8), pts_cam.shape),
                    radii=0.0005,
                ), static=False)

                # World / object space (cyan)
                c2o_for_depth = gt_c2o if vis_space == "object" and gt_c2o is not None else np.eye(4, dtype=np.float32)
                if c2o_for_depth is not None:
                    pts_world = (c2o_for_depth[:3, :3] @ pts_cam.T).T + c2o_for_depth[:3, 3]
                    rr.log("world/depth_hand", rr.Points3D(
                        pts_world,
                        colors=np.broadcast_to(np.array([0, 200, 255], dtype=np.uint8), pts_world.shape),
                        radii=0.0005,
                    ), static=False)

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
        jpeg_quality=args.jpeg_quality,
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
    parser.add_argument("--pred_align", type=str, default="GT", choices=["GT", "SAM3D"],
                         help="Align predicted poses to 'GT' or 'SAM3D' reference")
    parser.add_argument("--jpeg_quality", type=int, default=30,
                         help="JPEG compression quality for camera images (0-100)")

    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    main(args)
