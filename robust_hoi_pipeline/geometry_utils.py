# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Geometry utility functions for the COLMAP pipeline.
"""

import copy
import numpy as np
import torch


def compute_normals_from_depth(depth_map, intrinsics):
    """Compute per-pixel normals from depth maps.

    Args:
        depth_map: Depth maps of shape [B, H, W]
        intrinsics: Camera intrinsic matrices of shape [B, 3, 3]

    Returns:
        Normal maps of shape [B, 3, H, W]
    """
    B, H, W = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    z = depth_map
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    pts = torch.stack([x, y, z], dim=-1)  # B,H,W,3

    dx = pts[:, :, 1:] - pts[:, :, :-1]  # B,H,W-1,3
    dy = pts[:, 1:, :] - pts[:, :-1, :]  # B,H-1,W,3

    dx_full = torch.empty_like(pts)
    dy_full = torch.empty_like(pts)
    dx_full[:, :, :-1] = dx
    dx_full[:, :, -1] = dx[:, :, -1]
    dy_full[:, :-1, :] = dy
    dy_full[:, -1, :] = dy[:, -1, :]

    normals = torch.cross(dx_full, dy_full, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    return normals.permute(0, 3, 1, 2)  # B,3,H,W


def compute_reproj_errors(pts_3d, pts_2d, ext, K):
    """Compute reprojection errors for 3D-2D correspondences.

    Args:
        pts_3d: (N, 3) 3D points in world/object frame.
        pts_2d: (N, 2) observed 2D points.
        ext: (3, 4) or (4, 4) extrinsic matrix (world-to-camera).
        K: (3, 3) intrinsic matrix.

    Returns:
        errs: (N,) reprojection errors (NaN for points behind camera).
        proj_2d: (N, 2) projected 2D points (NaN for points behind camera).
    """
    pts_3d = np.asarray(pts_3d, dtype=np.float64)
    pts_2d = np.asarray(pts_2d, dtype=np.float64)
    n = len(pts_3d)
    cam = (ext[:3, :3] @ pts_3d.T).T + ext[:3, 3]
    in_front = cam[:, 2] > 0

    errs = np.full(n, np.nan, dtype=np.float64)
    proj_2d = np.full((n, 2), np.nan, dtype=np.float64)
    if in_front.any():
        px = K[0, 0] * cam[in_front, 0] / cam[in_front, 2] + K[0, 2]
        py = K[1, 1] * cam[in_front, 1] / cam[in_front, 2] + K[1, 2]
        proj_2d[in_front] = np.stack([px, py], axis=1)
        errs[in_front] = np.linalg.norm(proj_2d[in_front] - pts_2d[in_front], axis=1)
    return errs, proj_2d


def axis_angle_to_matrix(rvecs):
    """Convert batched axis-angle vectors to rotation matrices.

    Args:
        rvecs: Axis-angle rotation vectors of shape [B, 3]

    Returns:
        Rotation matrices of shape [B, 3, 3]
    """
    device, dtype = rvecs.device, rvecs.dtype
    B = rvecs.shape[0]
    theta = torch.linalg.norm(rvecs, dim=1, keepdim=True).clamp(min=1e-9)  # (B,1)
    k = rvecs / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zero, -kz, ky], dim=-1),
            torch.stack([kz, zero, -kx], dim=-1),
            torch.stack([-ky, kx, zero], dim=-1),
        ],
        dim=1,
    )  # (B,3,3)
    I = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3)
    sin_t = torch.sin(theta).view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)
    k_outer = (k[:, :, None] * k[:, None, :])  # (B,3,3)
    R = cos_t * I + (1 - cos_t) * k_outer + sin_t * K
    return R


def adjust_intrinsic_for_new_image_size(intrinsic, original_coords, frame_idx=0):
    """Rescale/shift intrinsic to the padded+resized image frame using stored original coords.

    Args:
        intrinsic: Camera intrinsic matrix (3x3)
        original_coords: Original image coordinates tensor/array containing (x1, y1, x2, y2, width, height)
        frame_idx: Frame index to use for coordinates

    Returns:
        Adjusted intrinsic matrix
    """
    base_intr = np.asarray(intrinsic, dtype=np.float32)
    if base_intr.shape != (3, 3):
        return base_intr

    if original_coords is None:
        return base_intr

    if torch.is_tensor(original_coords):
        orig_np = original_coords.detach().cpu().numpy()
    else:
        orig_np = np.asarray(original_coords)

    if orig_np.ndim < 2 or orig_np.shape[1] < 6:
        return base_intr

    idx = int(frame_idx)
    if idx >= orig_np.shape[0]:
        idx = 0

    x1, y1, x2, y2, width, height = orig_np[idx]
    if width <= 0 or height <= 0:
        return base_intr

    scale_x = (x2 - x1) / float(width)
    scale_y = (y2 - y1) / float(height)

    adjusted = base_intr.copy()
    adjusted[0, 0] *= scale_x
    adjusted[1, 1] *= scale_y
    adjusted[0, 2] = adjusted[0, 2] * scale_x + x1
    adjusted[1, 2] = adjusted[1, 2] * scale_y + y1
    return adjusted


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    """Rename COLMAP reconstruction outputs and rescale camera parameters.

    Args:
        reconstruction: COLMAP reconstruction object
        image_paths: List of original image paths
        original_coords: Original image coordinates
        img_size: Target image size
        shift_point2d_to_original_res: Whether to shift 2D points to original resolution
        shared_camera: Whether all images share the same camera

    Returns:
        Modified reconstruction object
    """
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction
