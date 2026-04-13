"""Debug helpers for pipeline_joint_opt.

Functions here are only used for saving debug artifacts (images, meshes) to
disk. They are no-ops when ``debug_dir`` is None.
"""

from pathlib import Path

import numpy as np
import torch


def _save_binary_mask_debug(debug_dir, frame_idx, pred_mask, target_mask, filename_prefix, it=None):
    if debug_dir is None or pred_mask is None:
        return

    import cv2 as _cv2

    def _to_u8(mask):
        if torch.is_tensor(mask):
            arr = mask.detach().float().cpu().numpy()
        else:
            arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim != 2:
            return None
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    pred_u8 = _to_u8(pred_mask)
    if pred_u8 is None:
        return

    H, W = pred_u8.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Pred mask contour in blue
    contours, _ = _cv2.findContours(pred_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    _cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)
    # Target mask contour in green
    if target_mask is not None:
        target_u8 = _to_u8(target_mask)
        if target_u8 is not None:
            contours_t, _ = _cv2.findContours(target_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            _cv2.drawContours(canvas, contours_t, -1, (0, 255, 0), 2)

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    if it is None:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}.png"
    else:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}_iter{it:03d}.png"
    _cv2.imwrite(str(_debug_dir / out_name), canvas)


def _save_hand_obj_meshes_in_cam_space(debug_dir, frame_idx, obj_mesh, hand_verts_in_cam, hand_faces,
                                       R_o2c, t_o2c, tag, depth_map=None, K=None):
    """Save hand mesh, object mesh, and depth 3D points in camera space.

    The hand mesh is already in camera space. The object mesh is transformed
    from object space to camera space using p_cam = R @ p_obj + t, where
    (R, t) is the object-to-camera pose. If ``depth_map`` and ``K`` are
    provided, valid depth pixels are backprojected to 3D and saved as a
    point cloud.

    Args:
        debug_dir: Output directory (created if missing). No-op if None.
        frame_idx: Frame index for filename suffix.
        obj_mesh: trimesh object in object space.
        hand_verts_in_cam: (1, V, 3) torch tensor of hand vertices in camera space.
        hand_faces: (F, 3) numpy array of hand faces.
        R_o2c: (3, 3) torch tensor, object-to-camera rotation.
        t_o2c: (3,) torch tensor, object-to-camera translation.
        tag: "before" or "after" (used in filename).
        depth_map: optional (H, W) numpy depth map.
        K: optional (3, 3) numpy intrinsic matrix.
    """
    if debug_dir is None:
        return
    import trimesh as _trimesh

    _dbg = Path(debug_dir)
    _dbg.mkdir(parents=True, exist_ok=True)

    # Transform object mesh: object space → camera space
    with torch.no_grad():
        obj_verts_t = torch.tensor(np.asarray(obj_mesh.vertices), dtype=torch.float32, device=R_o2c.device)
        obj_verts_cam = (obj_verts_t @ R_o2c.T + t_o2c[None, :]).cpu().numpy()
    _trimesh.Trimesh(
        vertices=obj_verts_cam.astype(np.float32),
        faces=np.asarray(obj_mesh.faces, dtype=np.int32),
        process=False,
    ).export(_dbg / f"obj_{tag}_frame_{frame_idx:04d}.obj")

    # Hand mesh: already in camera space
    with torch.no_grad():
        hand_verts_np = hand_verts_in_cam[0].cpu().numpy()
    _trimesh.Trimesh(
        vertices=hand_verts_np.astype(np.float32),
        faces=hand_faces,
        process=False,
    ).export(_dbg / f"hand_{tag}_frame_{frame_idx:04d}.obj")

    # Depth points: backproject to 3D in camera space
    if depth_map is not None and K is not None:
        vs, us = np.where(depth_map > 0.01)
        if len(vs) > 0:
            zs = depth_map[vs, us].astype(np.float64)
            xs = (us.astype(np.float64) - K[0, 2]) * zs / K[0, 0]
            ys = (vs.astype(np.float64) - K[1, 2]) * zs / K[1, 1]
            pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
            _trimesh.PointCloud(pts).export(_dbg / f"depth_pts_{tag}_frame_{frame_idx:04d}.ply")
