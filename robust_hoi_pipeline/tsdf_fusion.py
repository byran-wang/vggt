# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TSDF fusion for fusing multiple depth frames into a 3D mesh.

Uses the KinectFusion TSDFVolumeTorch backend to integrate depth maps
with known camera poses into a TSDF volume, then extracts a mesh
via marching cubes.
"""

import os
import sys

import numpy as np
import torch
import trimesh

# Add KinectFusion to path for TSDFVolumeTorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "KinectFusion"))
from fusion import TSDFVolumeTorch


def select_keyframes(
    extrinsics,
    rot_thresh=5.0,
    trans_thresh=0.02,
    track_mask=None,
    depth_frames=None,
    track_inlier_thresh=10,
    depth_thresh=500,
):
    """Select keyframes from a sequence based on validity and pose deltas.

    First filters frames by validity (enough track inliers and valid depth),
    then greedily selects frames that are sufficiently different from all
    previously selected keyframes.

    Args:
        extrinsics: Camera w2c matrices (N, 4, 4) or (N, 3, 4), numpy array.
            Convention: p_cam = R @ p_world + t
        rot_thresh: Minimum rotation delta in degrees to qualify as a new keyframe.
        trans_thresh: Minimum translation delta in meters to qualify as a new keyframe.
        track_mask: Optional track visibility mask (N, P) where P is number of points.
            Used to filter frames with insufficient track inliers.
        depth_frames: Optional depth maps (N, H, W). Used to filter frames with
            insufficient valid depth pixels.
        track_inlier_thresh: Minimum number of track inliers for a valid frame.
        depth_thresh: Minimum number of valid depth pixels for a valid frame.

    Returns:
        List of keyframe indices into the original extrinsics array.
    """
    N = len(extrinsics)
    if N == 0:
        return []

    extr = np.asarray(extrinsics)

    # Step 1: Identify valid frames (enough track inliers + valid depth)
    valid_frames = []
    for i in range(N):
        # Check track inliers
        has_enough_inliers = True
        if track_mask is not None:
            mask = track_mask[i]
            if torch.is_tensor(mask):
                track_inliers = mask.sum().item()
            else:
                track_inliers = np.sum(mask)
            has_enough_inliers = track_inliers >= track_inlier_thresh

        # Check valid depth
        has_enough_depth = True
        if depth_frames is not None:
            depth = depth_frames[i]
            if torch.is_tensor(depth):
                valid_depth_count = (depth > 0).sum().item()
            else:
                valid_depth_count = np.sum(np.asarray(depth) > 0)
            has_enough_depth = valid_depth_count >= depth_thresh

        if has_enough_inliers and has_enough_depth:
            valid_frames.append(i)

    if len(valid_frames) == 0:
        print("[select_keyframes] No valid frames found, using all frames")
        valid_frames = list(range(N))

    print(f"[select_keyframes] Valid frames: {len(valid_frames)}/{N}")

    # Step 2: From valid frames, select keyframes based on rotation/translation thresholds
    keyframes = []
    for frame_idx in valid_frames:
        if len(keyframes) == 0:
            # First valid frame is always a keyframe
            keyframes.append(frame_idx)
            continue

        # Check rotation and translation delta with all existing keyframes
        R_curr = extr[frame_idx, :3, :3]
        t_curr = extr[frame_idx, :3, 3]

        is_keyframe = True
        for kf_idx in keyframes:
            R_kf = extr[kf_idx, :3, :3]
            t_kf = extr[kf_idx, :3, 3]

            # Rotation delta in degrees
            R_delta = R_curr @ R_kf.T
            cos_angle = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.rad2deg(np.arccos(cos_angle))

            # Translation delta
            trans = np.linalg.norm(t_curr - t_kf)

            # Reject if too close to any existing keyframe
            if angle < rot_thresh and trans < trans_thresh:
                is_keyframe = False
                break

        if is_keyframe:
            keyframes.append(frame_idx)

    return keyframes


def compute_volume_bounds(depth_frames, extrinsics, intrinsic, masks=None,
                          frame_indices=None, margin=0.1, subsample=20):
    """Compute axis-aligned volume bounds from depth frames and camera poses.

    Back-projects sparse depth samples to world coordinates across selected frames
    to determine the bounding box of the scene.

    Args:
        depth_frames: Depth maps (N, H, W), numpy or torch tensor. Units: meters.
        extrinsics: Camera w2c matrices (N, 4, 4) or (N, 3, 4), numpy array.
            Convention: p_cam = R @ p_world + t
        intrinsic: Camera intrinsic matrix (3, 3) or (N, 3, 3), numpy array.
        masks: Optional object masks (N, H, W). If provided, only masked pixels
            contribute to bounds computation.
        frame_indices: Optional list of frame indices to use. If None, all frames
            are used.
        margin: Padding in meters added to each side of the bounding box.
        subsample: Pixel stride for sparse sampling (higher = faster but coarser).

    Returns:
        vol_bounds: Array of shape (3, 2) with [min, max] for x, y, z.
    """
    all_points = []

    if frame_indices is None:
        frame_indices = range(len(depth_frames))

    for i in frame_indices:
        depth = depth_frames[i]
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)

        H, W = depth.shape[-2:]

        # Get intrinsic for this frame
        intr = np.asarray(intrinsic)
        if intr.ndim == 3:
            intr = intr[i]
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]

        # Build w2c -> c2w
        ext = np.asarray(extrinsics[i])
        if ext.shape == (3, 4):
            ext_4x4 = np.eye(4, dtype=np.float64)
            ext_4x4[:3, :] = ext
        else:
            ext_4x4 = ext.astype(np.float64)
        c2w = np.linalg.inv(ext_4x4)

        # Sparse grid
        u = np.arange(0, W, subsample)
        v = np.arange(0, H, subsample)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()

        z = depth[v, u]
        valid = z > 0

        # Apply mask if provided
        if masks is not None:
            mask = masks[i]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            valid = valid & (mask[v, u] > 0.5)

        if not valid.any():
            continue

        u, v, z = u[valid], v[valid], z[valid]

        # Back-project to camera coords
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (M, 4)

        # Transform to world coords
        pts_world = (c2w @ pts_cam.T).T[:, :3]
        all_points.append(pts_world)

    if len(all_points) == 0:
        raise ValueError("[compute_volume_bounds] No valid depth points found across all frames.")

    all_points = np.concatenate(all_points, axis=0)

    mins = all_points.min(axis=0) - margin
    maxs = all_points.max(axis=0) + margin

    vol_bounds = np.array([[mins[0], maxs[0]],
                           [mins[1], maxs[1]],
                           [mins[2], maxs[2]]])
    return vol_bounds


def fuse_depth_to_mesh(
    depth_frames,
    extrinsics,
    intrinsic,
    color_frames=None,
    masks=None,
    frame_indices=None,
    voxel_size=0.005,
    vol_bounds=None,
    margin=3,
    device="cuda",
    output_path=None,
):
    """Fuse multiple depth frames into a TSDF volume and extract a mesh.

    Args:
        depth_frames: Depth maps (N, H, W), tensor or numpy. Units: meters.
        extrinsics: Camera w2c matrices (N, 4, 4) or (N, 3, 4), numpy array.
            Convention: p_cam = R @ p_world + t
        intrinsic: Camera intrinsic matrix (3, 3) or (N, 3, 3), numpy array.
        color_frames: Optional RGB images (N, H, W, 3), uint8 or float [0,255].
            If None, the mesh will have no vertex colors.
        masks: Optional object masks (N, H, W). Depth outside the mask is zeroed
            before integration so only the object is reconstructed.
        frame_indices: Optional list of frame indices to integrate. If None, all
            frames are used. Use select_keyframes() to get keyframe indices.
        voxel_size: Size of each voxel in meters. Smaller = finer detail.
        vol_bounds: Volume bounds (3, 2) array with [min, max] per axis.
            If None, auto-computed from depth and poses.
        margin: SDF truncation margin in voxels (trunc_dist = margin * voxel_size).
        device: Torch device string.
        output_path: If provided, save mesh as PLY to this path.

    Returns:
        trimesh.Trimesh: The fused mesh, or None if fusion fails.
    """
    if frame_indices is None:
        frame_indices = list(range(len(depth_frames)))

    N = len(frame_indices)
    if N == 0:
        print("[fuse_depth_to_mesh] No frames to fuse.")
        return None

    print(f"[fuse_depth_to_mesh] Fusing {N} frames (indices: {frame_indices})")

    fuse_color = color_frames is not None

    # Auto-compute volume bounds using only the selected frames
    if vol_bounds is None:
        print("[fuse_depth_to_mesh] Auto-computing volume bounds...")
        vol_bounds = compute_volume_bounds(
            depth_frames, extrinsics, intrinsic, masks=masks, frame_indices=frame_indices
        )

    vol_bounds = np.asarray(vol_bounds, dtype=np.float64).reshape(3, 2)
    vol_dims = ((vol_bounds[:, 1] - vol_bounds[:, 0]) / voxel_size + 1).astype(int)
    vol_origin = vol_bounds[:, 0]

    total_voxels = int(np.prod(vol_dims))
    print(f"[fuse_depth_to_mesh] Volume dims: {vol_dims.tolist()}, "
          f"origin: [{vol_origin[0]:.3f}, {vol_origin[1]:.3f}, {vol_origin[2]:.3f}], "
          f"voxel_size: {voxel_size}, total voxels: {total_voxels}")

    if total_voxels > 500_000_000:
        print(f"[fuse_depth_to_mesh] Volume too large ({total_voxels} voxels). "
              "Consider increasing voxel_size or providing tighter vol_bounds.")
        return None

    # Initialize TSDF volume
    tsdf_volume = TSDFVolumeTorch(
        voxel_dim=vol_dims,
        origin=vol_origin,
        voxel_size=voxel_size,
        device=device,
        margin=margin,
        fuse_color=fuse_color,
    )

    # Integrate selected frames
    for count, i in enumerate(frame_indices):
        # Prepare depth
        depth = depth_frames[i]
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32).copy()

        # Apply mask: zero out depth outside the object
        if masks is not None:
            mask = masks[i]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            depth[mask < 0.5] = 0.0

        # Get intrinsic for this frame
        intr = np.asarray(intrinsic, dtype=np.float32)
        if intr.ndim == 3:
            intr = intr[i]

        # Convert w2c extrinsic to c2w for KinectFusion
        ext = np.asarray(extrinsics[i], dtype=np.float64)
        if ext.shape == (3, 4):
            ext_4x4 = np.eye(4, dtype=np.float64)
            ext_4x4[:3, :] = ext
        else:
            ext_4x4 = ext
        c2w = np.linalg.inv(ext_4x4).astype(np.float32)

        # Prepare color
        color = None
        if fuse_color and color_frames is not None:
            color = color_frames[i]
            if torch.is_tensor(color):
                color = color.cpu().numpy()
            color = np.asarray(color, dtype=np.float32)

        # Integrate into TSDF
        tsdf_volume.integrate(
            depth_im=depth,
            cam_intr=intr,
            cam_pose=c2w,
            obs_weight=1.0,
            color_img=color,
        )

        if (count + 1) % max(1, N // 5) == 0 or count == N - 1:
            print(f"[fuse_depth_to_mesh] Integrated frame {count + 1}/{N} (index {i})")

    # Extract mesh via marching cubes
    try:
        mesh_data = tsdf_volume.get_mesh()
    except Exception as e:
        print(f"[fuse_depth_to_mesh] Marching cubes failed: {e}")
        return None

    if fuse_color:
        verts, faces, norms, colors = mesh_data
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=norms,
            vertex_colors=colors,
        )
    else:
        verts, faces, norms = mesh_data
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=norms,
        )
    # remove disconnect part
    connected_comp = mesh.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    mesh = max_comp        

    print(f"[fuse_depth_to_mesh] Extracted mesh: {len(verts)} vertices, {len(faces)} faces")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        print(f"[fuse_depth_to_mesh] Saved mesh to {output_path}")

    return mesh


def visualize_tsdf_fusion_rerun(
    tsdf_mesh,
    extrinsics,
    intrinsic,
    color_frames,
    frame_indices,
    image_masks=None,
):
    """Visualize TSDF fusion results in Rerun: camera frustums, images, and fused mesh.

    Args:
        tsdf_mesh: trimesh.Trimesh of the fused mesh.
        extrinsics: Camera w2c matrices (N, 4, 4) numpy array.
        intrinsic: Camera intrinsic matrix (3, 3) or (N, 3, 3) numpy array.
        color_frames: RGB images (N, H, W, 3) numpy float [0,255].
        frame_indices: List of keyframe indices.
        image_masks: Optional (N, H, W) masks.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    # Build blueprint: vertical layout with 3D view on top, 2D image grid below
    image_views = [
        rrb.Spatial2DView(name=f"Camera {idx}", origin=f"world/camera_{idx}")
        for idx in range(len(frame_indices))
    ]
    blueprint = rrb.Vertical(
        rrb.Spatial3DView(name="3D View", origin="world"),
        rrb.Grid(*image_views, name="Images"),
        row_shares=[2, 1],
    )

    rr.init("tsdf_fusion", spawn=True, default_blueprint=blueprint)

    # Log world coordinate axes
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    extr = np.asarray(extrinsics)
    intr = np.asarray(intrinsic)

    for idx, frame_idx in enumerate(frame_indices):
        rr.set_time_sequence("frame", idx)

        # Compute c2w from w2c
        ext = extr[frame_idx].astype(np.float64)
        if ext.shape == (3, 4):
            ext_4x4 = np.eye(4, dtype=np.float64)
            ext_4x4[:3, :] = ext
        else:
            ext_4x4 = ext
        c2w = np.linalg.inv(ext_4x4)

        translation = c2w[:3, 3]
        mat3x3 = c2w[:3, :3]

        entity = f"world/camera_{idx}"

        # Log camera transform (c2w)
        rr.log(entity, rr.Transform3D(translation=translation, mat3x3=mat3x3))

        # Get intrinsic for this frame
        K = intr[frame_idx] if intr.ndim == 3 else intr

        color = color_frames[frame_idx]
        if torch.is_tensor(color):
            color = color.cpu().numpy()
        color = np.asarray(color)
        H, W = color.shape[:2]

        # Log pinhole camera
        rr.log(entity, rr.Pinhole(image_from_camera=K, resolution=[W, H]))

        # Log image
        rr.log(entity, rr.Image(color.astype(np.uint8)))

    # Log the TSDF mesh as static
    if tsdf_mesh is not None:
        kwargs = dict(
            vertex_positions=tsdf_mesh.vertices,
            triangle_indices=tsdf_mesh.faces,
        )
        if tsdf_mesh.vertex_normals is not None and len(tsdf_mesh.vertex_normals) > 0:
            kwargs["vertex_normals"] = tsdf_mesh.vertex_normals
        if tsdf_mesh.visual and hasattr(tsdf_mesh.visual, "vertex_colors"):
            kwargs["vertex_colors"] = np.asarray(tsdf_mesh.visual.vertex_colors)[:, :3]
        rr.log("world/mesh", rr.Mesh3D(**kwargs), static=True)

    print("[visualize_tsdf_fusion_rerun] Rerun visualization launched.")
