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


def _save_hand_obj_meshes_in_obj_space(debug_dir, frame_idx, obj_mesh, hand_verts_in_cam, hand_faces, R_o2c, t_o2c, tag):
    """Save hand and object meshes in object space to ``debug_dir``.

    The object mesh is already in object space. The hand mesh lives in camera
    space and is transformed using p_obj = R^T @ (p_cam - t), where (R, t) is
    the object-to-camera pose.

    Args:
        debug_dir: Output directory (created if missing). No-op if None.
        frame_idx: Frame index for filename suffix.
        obj_mesh: trimesh object in object space.
        hand_verts_in_cam: (1, V, 3) torch tensor of hand vertices in camera space.
        hand_faces: (F, 3) numpy array of hand faces.
        R_o2c: (3, 3) torch tensor, object-to-camera rotation.
        t_o2c: (3,) torch tensor, object-to-camera translation.
        tag: "before" or "after" (used in filename).
    """
    if debug_dir is None:
        return
    import trimesh as _trimesh

    _dbg = Path(debug_dir)
    _dbg.mkdir(parents=True, exist_ok=True)

    _trimesh.Trimesh(
        vertices=np.asarray(obj_mesh.vertices, dtype=np.float32),
        faces=np.asarray(obj_mesh.faces, dtype=np.int32),
        process=False,
    ).export(_dbg / f"obj_{tag}_frame_{frame_idx:04d}.obj")

    with torch.no_grad():
        hand_verts_obj = ((hand_verts_in_cam[0] - t_o2c[None, :]) @ R_o2c).cpu().numpy()
    _trimesh.Trimesh(
        vertices=hand_verts_obj.astype(np.float32),
        faces=hand_faces,
        process=False,
    ).export(_dbg / f"hand_{tag}_frame_{frame_idx:04d}.obj")
