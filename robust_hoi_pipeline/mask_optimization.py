# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mask-based pose and intrinsic optimization using differentiable rendering.

Uses nvdiffrast to render silhouettes and optimizes aligned poses and camera
intrinsics by minimizing IoU loss between rendered and target masks.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import trimesh

# Add utils_simba to path
_THIRD_PARTY_UTILS_SIMBA = str(Path(__file__).resolve().parents[1] / "third_party" / "utils_simba")
if _THIRD_PARTY_UTILS_SIMBA not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_UTILS_SIMBA)

import nvdiffrast.torch as dr
from utils_simba.render import (
    diff_renderer,
    make_mesh_tensors,
    projection_matrix_from_intrinsics,
    projection_matrix_to_intrinsics,
)
from utils_simba.geometry import matrix_to_axis_angle_t, axis_angle_t_to_matrix


def compute_iou_loss(pred_mask, target_mask, eps=1e-6):
    """Compute IoU loss (1 - IoU) between predicted and target masks.

    Args:
        pred_mask: Predicted silhouette mask [H, W] in [0, 1]
        target_mask: Target mask [H, W] in [0, 1]
        eps: Small epsilon to prevent division by zero

    Returns:
        IoU loss (scalar tensor)
    """
    inter = (pred_mask * target_mask).sum()
    total = (pred_mask + target_mask).sum()
    union = total - inter
    iou = inter / (union + eps)
    return 1.0 - iou


def _optimize_single_keyframe(
    verts,
    tri,
    color_obj,
    target_mask,
    intrinsic,
    extrinsic,
    aligned_pose,
    glctx,
    device,
    num_iters=200,
    lr=1e-2,
    optimize_intrinsic=True,
    debug_dir=None,
    frame_name="",
):
    """Optimize pose and intrinsic for a single keyframe using mask IoU loss.

    Args:
        verts: Mesh vertices [1, N, 3]
        tri: Triangle faces [M, 3]
        color_obj: Vertex colors for rendering [1, N, 3]
        target_mask: Target mask tensor [H, W]
        intrinsic: Camera intrinsic matrix [3, 3] numpy
        extrinsic: Camera extrinsic matrix [4, 4] numpy
        aligned_pose: Alignment transform from gen_3d space to world [4, 4] numpy
        glctx: nvdiffrast context
        device: torch device
        num_iters: Number of optimization iterations
        lr: Learning rate
        optimize_intrinsic: Whether to optimize camera intrinsics
        debug_dir: Optional directory to save debug images
        frame_name: Frame identifier for debug output

    Returns:
        Tuple of (optimized_aligned_pose, optimized_intrinsic, final_iou_loss)
    """
    H, W = target_mask.shape
    resolution = (H, W)

    # Build projection matrix from intrinsic
    K = intrinsic.astype(np.float64)
    projection_orig = torch.tensor(
        projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100),
        dtype=torch.float32,
        device=device,
    )

    # Projection residual for intrinsic optimization (only fx, fy)
    projection_residual = torch.nn.Parameter(
        torch.zeros((4, 4), device=device, dtype=torch.float32)
    )
    projection_mask = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    # Compose the full world-to-camera transform: w2c = extrinsic @ aligned_pose
    # This transforms points from gen_3d space to camera space
    extrinsic_t = torch.tensor(extrinsic, dtype=torch.float32, device=device)
    aligned_pose_t = torch.tensor(aligned_pose, dtype=torch.float32, device=device)

    # Initial composed transform
    w2c_init = extrinsic_t @ aligned_pose_t

    # Decompose into axis-angle and translation for optimization
    c2ws_r_orig, c2ws_t_orig, c2ws_s_orig = matrix_to_axis_angle_t(w2c_init)

    # Pose residual: [3 for rotation, 3 for translation]
    c2ws_residual = torch.nn.Parameter(
        torch.zeros(6, device=device, dtype=torch.float32)
    )

    # Setup optimizer
    params = [c2ws_residual]
    if optimize_intrinsic:
        params.append(projection_residual)
    optimizer = torch.optim.Adam(params, lr=lr)

    # Optimization loop
    best_loss = float("inf")
    best_c2ws_residual = None
    best_projection_residual = None

    for it in range(num_iters):
        optimizer.zero_grad()

        # Build projection with residual
        if optimize_intrinsic:
            projection = projection_orig + projection_residual * projection_mask
        else:
            projection = projection_orig

        # Build pose with residual
        c2ws_r = c2ws_r_orig + c2ws_residual[:3] * 0.1  # Scale rotation step
        c2ws_t = c2ws_t_orig + c2ws_residual[3:]
        w2c = axis_angle_t_to_matrix(c2ws_r, c2ws_t, c2ws_s_orig)

        # Render silhouette
        rgb_rendered, _ = diff_renderer(
            verts, tri, color_obj, projection, w2c, resolution, glctx
        )
        sil_pred = rgb_rendered[..., 1]  # Green channel as silhouette

        # Compute IoU loss
        loss = compute_iou_loss(sil_pred, target_mask)

        # Save debug images periodically
        if debug_dir and it % 50 == 0:
            os.makedirs(debug_dir, exist_ok=True)
            sil_np = sil_pred.detach().cpu().numpy()
            tgt_np = target_mask.detach().cpu().numpy()
            Image.fromarray((sil_np * 255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"{frame_name}_sil_{it:04d}.png")
            )
            if it == 0:
                Image.fromarray((tgt_np * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, f"{frame_name}_gt.png")
                )

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_c2ws_residual = c2ws_residual.detach().clone()
            if optimize_intrinsic:
                best_projection_residual = projection_residual.detach().clone()

        # Backward and step
        loss.backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            print(
                f"[mask_opt] {frame_name} Iter {it+1:04d}: IoU loss={loss.item():.4f}"
            )

    # Reconstruct best results
    if best_c2ws_residual is not None:
        c2ws_r_best = c2ws_r_orig + best_c2ws_residual[:3] * 0.1
        c2ws_t_best = c2ws_t_orig + best_c2ws_residual[3:]
        w2c_best = axis_angle_t_to_matrix(c2ws_r_best, c2ws_t_best, c2ws_s_orig)

        # Recover optimized aligned_pose: aligned_pose_opt = extrinsic^{-1} @ w2c_best
        extrinsic_inv = torch.inverse(extrinsic_t)
        aligned_pose_opt = extrinsic_inv @ w2c_best
        aligned_pose_opt_np = aligned_pose_opt.detach().cpu().numpy()
    else:
        aligned_pose_opt_np = aligned_pose

    # Recover optimized intrinsic
    if optimize_intrinsic and best_projection_residual is not None:
        projection_best = projection_orig + best_projection_residual * projection_mask
        K_opt = projection_matrix_to_intrinsics(
            projection_best.detach().cpu().numpy(), W, H
        )
    else:
        K_opt = K.astype(np.float32)

    return aligned_pose_opt_np, K_opt, best_loss


def _optimize_multi_keyframes(
    verts,
    tri,
    color_obj,
    target_masks,
    intrinsics,
    extrinsics,
    aligned_pose,
    glctx,
    device,
    num_iters=200,
    lr=1e-2,
    optimize_intrinsic=True,
    debug_dir=None,
    frame_names=None,
):
    num_frames = len(target_masks)
    if frame_names is None:
        frame_names = [f"frame_{idx:04d}" for idx in range(num_frames)]

    projection_mask = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    projection_origs = []
    projection_residuals = []
    extrinsic_ts = []
    resolutions = []

    for idx in range(num_frames):
        mask = target_masks[idx]
        H, W = mask.shape
        resolutions.append((H, W))

        K = intrinsics[idx].astype(np.float64)
        projection_origs.append(
            torch.tensor(
                projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100),
                dtype=torch.float32,
                device=device,
            )
        )

        if optimize_intrinsic:
            projection_residuals.append(
                torch.nn.Parameter(torch.zeros((4, 4), device=device, dtype=torch.float32))
            )

        extrinsic_ts.append(torch.tensor(extrinsics[idx], dtype=torch.float32, device=device))

    aligned_pose_t = torch.tensor(aligned_pose, dtype=torch.float32, device=device)
    pose_r_orig, pose_t_orig, pose_s_orig = matrix_to_axis_angle_t(aligned_pose_t)
    pose_residual = torch.nn.Parameter(torch.zeros(6, device=device, dtype=torch.float32))

    params = [pose_residual]
    if optimize_intrinsic:
        params.extend(projection_residuals)
    optimizer = torch.optim.Adam(params, lr=lr)

    best_loss = float("inf")
    best_pose_residual = None
    best_projection_residuals = None

    for it in range(num_iters):
        optimizer.zero_grad()

        pose_r = pose_r_orig + pose_residual[:3] * 0.1
        pose_t = pose_t_orig + pose_residual[3:]
        aligned_pose_cur = axis_angle_t_to_matrix(pose_r, pose_t, pose_s_orig)

        total_loss = 0.0
        for idx in range(num_frames):
            if optimize_intrinsic:
                projection = projection_origs[idx] + projection_residuals[idx] * projection_mask
            else:
                projection = projection_origs[idx]

            w2c = extrinsic_ts[idx] @ aligned_pose_cur
            rgb_rendered, _ = diff_renderer(
                verts, tri, color_obj, projection, w2c, resolutions[idx], glctx
            )
            sil_pred = rgb_rendered[..., 1]
            total_loss = total_loss + compute_iou_loss(sil_pred, target_masks[idx])

            if debug_dir and it % 50 == 0:
                os.makedirs(debug_dir, exist_ok=True)
                sil_np = sil_pred.detach().cpu().numpy()
                tgt_np = target_masks[idx].detach().cpu().numpy()
                Image.fromarray((sil_np * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, f"{frame_names[idx]}_sil_{it:04d}.png")
                )
                if it == 0:
                    Image.fromarray((tgt_np * 255).astype(np.uint8)).save(
                        os.path.join(debug_dir, f"{frame_names[idx]}_gt.png")
                    )

        loss = total_loss / max(1, num_frames)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_pose_residual = pose_residual.detach().clone()
            if optimize_intrinsic:
                best_projection_residuals = [p.detach().clone() for p in projection_residuals]

        loss.backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            print(f"[mask_opt] Iter {it+1:04d}: avg IoU loss={loss.item():.4f}")

    if best_pose_residual is not None:
        pose_r_best = pose_r_orig + best_pose_residual[:3] * 0.1
        pose_t_best = pose_t_orig + best_pose_residual[3:]
        aligned_pose_opt = axis_angle_t_to_matrix(pose_r_best, pose_t_best, pose_s_orig)
        aligned_pose_opt_np = aligned_pose_opt.detach().cpu().numpy()
    else:
        aligned_pose_opt_np = aligned_pose

    K_opts = []
    for idx in range(num_frames):
        if optimize_intrinsic and best_projection_residuals is not None:
            projection_best = projection_origs[idx] + best_projection_residuals[idx] * projection_mask
            H, W = resolutions[idx]
            K_opt = projection_matrix_to_intrinsics(
                projection_best.detach().cpu().numpy(), W, H
            )
        else:
            K_opt = intrinsics[idx].astype(np.float32)
        K_opts.append(K_opt)

    return aligned_pose_opt_np, K_opts, best_loss


def optimize_pose_with_mask_loss(image_info, gen_3d, args):
    """Optimize aligned poses and intrinsics using mask IoU loss.

    For each keyframe, renders the mesh silhouette using nvdiffrast and
    optimizes the pose/intrinsic residuals to maximize IoU with target masks.

    Args:
        image_info: Dictionary containing:
            - keyframe: Boolean array marking keyframes
            - image_masks: Target masks [N, 1, H, W] or [N, H, W]
            - extrinsics: Camera extrinsic matrices [N, 4, 4]
            - intrinsics: Camera intrinsic matrices [N, 3, 3] or [3, 3]
        gen_3d: Generated 3D model object with:
            - mesh_path: Path to mesh file
            - get_aligned_pose(): Returns alignment transform
            - save_aligned_pose(): Saves updated alignment
        args: Arguments with:
            - mask_opt_iters: Number of optimization iterations
            - mask_opt_lr: Learning rate
            - optimize_intrinsic: Whether to optimize intrinsics
            - output_dir: Output directory for debug images

    Returns:
        Tuple of (updated image_info, updated gen_3d)
    """
    # Check if we have keyframes
    keyframes = image_info.get("keyframe")
    if keyframes is None:
        print("[optimize_pose_with_mask_loss] No keyframes found, skipping")
        return image_info, gen_3d

    keyframe_indices = np.where(keyframes)[0]
    if len(keyframe_indices) == 0:
        print("[optimize_pose_with_mask_loss] No keyframes to optimize")
        return image_info, gen_3d

    print(f"[optimize_pose_with_mask_loss] Optimizing {len(keyframe_indices)} keyframes")

    # Get mesh
    mesh_path = gen_3d.get_mesh_path() if hasattr(gen_3d, "get_mesh_path") else str(gen_3d.mesh_path)
    if not os.path.exists(mesh_path):
        print(f"[optimize_pose_with_mask_loss] Mesh not found at {mesh_path}, skipping")
        return image_info, gen_3d

    # Setup device and context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glctx = dr.RasterizeCudaContext()

    # Load mesh
    mesh = trimesh.load(mesh_path, process=False)
    mesh_tensors = make_mesh_tensors(mesh, device=device)
    verts = mesh_tensors["pos"].unsqueeze(0)  # [1, N, 3]
    tri = mesh_tensors["faces"]

    # Use green color for silhouette rendering
    color_obj = torch.FloatTensor([0, 1, 0]).repeat(verts.shape[1], 1)
    color_obj = color_obj.unsqueeze(0).to(device)

    # Get data from image_info
    image_masks = image_info.get("image_masks")
    extrinsics = image_info.get("extrinsics")
    intrinsics = image_info.get("intrinsics")

    if image_masks is None or extrinsics is None or intrinsics is None:
        print("[optimize_pose_with_mask_loss] Missing required data, skipping")
        return image_info, gen_3d

    # Get aligned pose from gen_3d
    aligned_pose = gen_3d.get_aligned_pose()
    if aligned_pose is None:
        print("[optimize_pose_with_mask_loss] No aligned pose found, skipping")
        return image_info, gen_3d

    # Get optimization parameters
    num_iters = getattr(args, "mask_opt_iters", 200)
    lr = getattr(args, "mask_opt_lr", 1e-2)
    optimize_intrinsic = getattr(args, "optimize_intrinsic", True)
    output_dir = getattr(args, "output_dir", "output")
    debug_dir = os.path.join(output_dir, "mask_opt")

    target_masks = []
    use_extrinsics = []
    use_intrinsics = []
    frame_names = []

    for kf_idx in keyframe_indices:
        print(f"[optimize_pose_with_mask_loss] Preparing keyframe {kf_idx}")

        mask = image_masks[kf_idx]
        if torch.is_tensor(mask):
            mask = mask.to(device)
        else:
            mask = torch.tensor(mask, dtype=torch.float32, device=device)

        if mask.ndim == 3:
            mask = mask.squeeze(0)
        target_masks.append((mask > 0.5).float())
        frame_names.append(f"frame_{kf_idx:04d}")

        extrinsic = extrinsics[kf_idx]
        if extrinsic.shape[0] == 3:
            extrinsic_4x4 = np.eye(4, dtype=np.float64)
            extrinsic_4x4[:3, :] = extrinsic
        else:
            extrinsic_4x4 = extrinsic.astype(np.float64)
        use_extrinsics.append(extrinsic_4x4)

        if intrinsics.ndim == 3:
            intrinsic = intrinsics[kf_idx].astype(np.float64)
        else:
            intrinsic = intrinsics.astype(np.float64)
        use_intrinsics.append(intrinsic)

    final_aligned_pose, _, avg_loss = _optimize_multi_keyframes(
        verts=verts,
        tri=tri,
        color_obj=color_obj,
        target_masks=target_masks,
        intrinsics=use_intrinsics,
        extrinsics=use_extrinsics,
        aligned_pose=aligned_pose,
        glctx=glctx,
        device=device,
        num_iters=num_iters,
        lr=lr,
        optimize_intrinsic=optimize_intrinsic,
        debug_dir=debug_dir,
        frame_names=frame_names,
    )

    gen_3d.save_aligned_pose(final_aligned_pose)

    print(f"[optimize_pose_with_mask_loss] Optimization complete. "
          f"Average IoU loss: {avg_loss:.4f}")

    return image_info, gen_3d
