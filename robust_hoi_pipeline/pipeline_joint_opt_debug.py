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


def _save_depth_points_debug(debug_dir, frame_idx, depth, K, rgb=None, o2c=None):
    """Back-project depth to 3D points and save as colored PLY in debug_dir.

    If o2c (4x4 object-to-camera) is provided, transforms points to object space.
    """
    if debug_dir is None or depth is None or K is None:
        return
    import trimesh as _trimesh

    _dbg = Path(debug_dir)
    _dbg.mkdir(parents=True, exist_ok=True)

    vs, us = np.where(depth > 0.01)
    if len(vs) == 0:
        return
    zs = depth[vs, us].astype(np.float64)
    xs = (us.astype(np.float64) - K[0, 2]) * zs / K[0, 0]
    ys = (vs.astype(np.float64) - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)

    # Transform to object space if o2c is provided
    if o2c is not None:
        c2o = np.linalg.inv(o2c)
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts = (c2o @ pts_h.T).T[:, :3].astype(np.float32)

    colors = None
    if rgb is not None and rgb.shape[:2] == depth.shape[:2]:
        colors = rgb[vs, us]  # (N, 3) uint8

    pc = _trimesh.PointCloud(pts, colors=colors)
    pc.export(_dbg / f"depth_pts_frame_{frame_idx:04d}.ply")


def _gather_frame_depth_assets(image_info_work, frame_idx):
    """Collect the per-frame tensors needed by the register debug helpers.

    Returns (depth_masked, K, rgb) where depth_masked has object-mask pixels
    zeroed out. Any of the three may be None if unavailable.
    """
    depth_priors = image_info_work.get("depth_priors")
    depth = depth_priors[frame_idx] if depth_priors is not None else None
    depth_masked = depth.copy() if depth is not None else None

    masks = image_info_work.get("image_masks")
    obj_mask = masks[frame_idx] if masks is not None else None
    if depth_masked is not None and obj_mask is not None:
        depth_masked[obj_mask == 0] = 0

    intrinsics = image_info_work.get("intrinsics")
    if intrinsics is None:
        K = None
    elif intrinsics.ndim == 3:
        K = intrinsics[frame_idx]
    else:
        K = intrinsics

    images = image_info_work.get("images")
    rgb = images[frame_idx] if images is not None else None
    return depth_masked, K, rgb


def _dump_register_frame_inputs(debug_dir, image_info_work, frame_idx,
                                sam3d_mesh, neus_mesh, pnp_pose):
    """Dump per-frame register inputs: SAM3D mesh, NeuS mesh, and PnP-pose depth.

    No-op when ``debug_dir`` is None.
    """
    if debug_dir is None:
        return
    _dbg = Path(debug_dir)
    _dbg.mkdir(parents=True, exist_ok=True)
    if sam3d_mesh is not None:
        sam3d_mesh.export(_dbg / "sam3d_mesh.obj")
    if neus_mesh is not None:
        neus_mesh.export(_dbg / "neus_mesh.obj")

    if pnp_pose is not None:
        depth_masked, K, rgb = _gather_frame_depth_assets(image_info_work, frame_idx)
        _save_depth_points_debug(_dbg / "pnp", frame_idx, depth_masked, K, rgb=rgb, o2c=pnp_pose)


def _dump_fp_iter_depth_points(debug_dir, image_info_work, frame_idx,
                               fp_pose_sam3d=None, fp_pose_neus=None):
    """Dump object-space depth point clouds for each FP candidate pose.

    Writes ``<debug_dir>/sam3d/`` and ``<debug_dir>/neus/`` PLYs. No-op when
    ``debug_dir`` is None.
    """
    if debug_dir is None:
        return
    depth_masked, K, rgb = _gather_frame_depth_assets(image_info_work, frame_idx)
    if fp_pose_sam3d is not None:
        _save_depth_points_debug(Path(debug_dir) / "sam3d", frame_idx, depth_masked, K,
                                 rgb=rgb, o2c=fp_pose_sam3d)
    if fp_pose_neus is not None:
        _save_depth_points_debug(Path(debug_dir) / "neus", frame_idx, depth_masked, K,
                                 rgb=rgb, o2c=fp_pose_neus)


def _dump_nearby_pts_obj(debug_dir, nearby_pts_obj, filename="nearby_pts_obj.ply"):
    """Save the nearby-frame depth points (object space) used for pose scoring."""
    if debug_dir is None or nearby_pts_obj is None or len(nearby_pts_obj) == 0:
        return
    import trimesh as _trimesh
    _dbg = Path(debug_dir)
    _dbg.mkdir(parents=True, exist_ok=True)
    _trimesh.PointCloud(np.asarray(nearby_pts_obj)).export(_dbg / filename)
