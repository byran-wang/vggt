# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import json
import os
import numpy as np
import torch
import argparse
import trimesh
from PIL import Image
import pickle
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, "dependency/LightGlue")
from lightglue import LightGlue, SuperPoint
from lightglue.utils import match_pair

from third_party.utils_simba.utils_simba.depth import (
    get_depth,
)


def load_image(path: str) -> np.ndarray:
    """Load image as uint8 numpy array."""
    image = Image.open(path)
    image = np.array(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def load_mask(path: str) -> np.ndarray:
    """Load mask as boolean array."""
    mask = np.array(Image.open(path))
    if mask.ndim == 3 and mask.shape[-1] == 4:
        mask = mask[..., 3] > 0
    elif mask.ndim == 3:
        mask = mask.any(axis=-1) > 0
    else:
        mask = mask > 0
    return mask


def load_intrinsics_from_meta(meta_file: str) -> np.ndarray:
    """Load camera intrinsics from meta pickle file."""
    with open(meta_file, "rb") as f:
        meta_data = pickle.load(f)
    return np.array(meta_data["camMat"], dtype=np.float32)


def load_intrinsics_from_json(camera_json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics (K) and object-to-camera transform (o2c) from JSON."""
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)
    K = np.array(camera_data["K"], dtype=np.float32)
    o2c = np.array(camera_data["blw2cvc"], dtype=np.float32)
    return K, o2c


def load_mesh_from_glb(glb_path: str) -> trimesh.Trimesh:
    """Load mesh from GLB file.

    Returns:
        Combined trimesh mesh from all geometries in the GLB file.
    """
    loaded = trimesh.load(glb_path)

    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if len(meshes) == 1:
            return meshes[0]
        else:
            return trimesh.util.concatenate(meshes)
    else:
        return loaded





def backproject_points(
    points_2d: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project 2D points to 3D using depth and intrinsics.

    Args:
        points_2d: (N, 2) pixel coordinates (u, v)
        depth: (H, W) depth map in meters
        K: (3, 3) camera intrinsic matrix

    Returns:
        points_3d: (M, 3) 3D points in camera coordinates
        valid_mask: (N,) boolean mask for points with valid depth
    """
    N = len(points_2d)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = points_2d[:, 0].astype(int)
    v = points_2d[:, 1].astype(int)

    # Clamp to image bounds
    H, W = depth.shape
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    z = depth[v, u]
    valid_mask = z > 0

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_3d = np.stack([x, y, z], axis=1)

    return points_3d, valid_mask


def get_correspondences(
    image0: np.ndarray,
    image1: np.ndarray,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Get feature correspondences between two images using SuperPoint + LightGlue.

    Args:
        image0: First image (H, W, 3) uint8
        image1: Second image (H, W, 3) uint8
        device: torch device

    Returns:
        kpts0: (N, 2) keypoints in image0
        kpts1: (N, 2) matched keypoints in image1
    """
    # Initialize extractor and matcher
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Convert images to torch tensors (C, H, W) float [0, 1]
    img0_tensor = torch.from_numpy(image0).permute(2, 0, 1).float() / 255.0
    img1_tensor = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0

    # Add batch dimension
    img0_tensor = img0_tensor.unsqueeze(0).to(device)
    img1_tensor = img1_tensor.unsqueeze(0).to(device)

    # Match features
    feats0, feats1, matches01 = match_pair(extractor, matcher, img0_tensor, img1_tensor, device=device)

    # Get keypoints
    kpts0 = feats0["keypoints"].cpu().numpy()  # (M, 2)
    kpts1 = feats1["keypoints"].cpu().numpy()  # (N, 2)

    # Get matches - this is a (K, 2) array of matched indices [idx0, idx1]
    matches = matches01["matches"].cpu().numpy()  # (K, 2)

    if len(matches) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    # Extract matched keypoints
    kpts0_matched = kpts0[matches[:, 0]]
    kpts1_matched = kpts1[matches[:, 1]]

    return kpts0_matched, kpts1_matched


def rigid_transform_3d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rigid transform (rotation + translation) from A to B using SVD.

    Solves: B = R @ A + t

    Args:
        A: (N, 3) source points
        B: (N, 3) target points

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert A.shape == B.shape
    N = A.shape[0]

    # Centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


def scale_translation_3d(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute scale and translation from A to B (no rotation).

    Solves: B = s * A + t

    Args:
        A: (N, 3) source points
        B: (N, 3) target points

    Returns:
        s: scale factor
        t: (3,) translation vector
    """
    assert A.shape == B.shape

    # Centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute scale using least squares: s = sum(BB * AA) / sum(AA * AA)
    # This minimizes ||s * AA - BB||^2
    s = np.sum(BB * AA) / (np.sum(AA * AA) + 1e-8)

    # Compute translation
    t = centroid_B - s * centroid_A

    return s, t


def optimize_rigid_transform(
    pts_src: np.ndarray,
    pts_tgt: np.ndarray,
    num_iters: int = 100,
    inlier_thresh: float = 0.02,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """RANSAC-based scale + translation optimization (no rotation).

    Args:
        pts_src: (N, 3) source points
        pts_tgt: (N, 3) target points
        num_iters: Number of RANSAC iterations
        inlier_thresh: Inlier distance threshold in meters

    Returns:
        s: best scale factor
        t: (3,) best translation vector
        inlier_mask: (N,) boolean mask for inliers
    """
    N = len(pts_src)
    best_inliers = 0
    best_s = 1.0
    best_t = np.zeros(3)
    best_mask = np.zeros(N, dtype=bool)

    for _ in range(num_iters):
        # Sample 2 random points (minimum for scale + translation)
        idx = np.random.choice(N, min(2, N), replace=False)
        if len(idx) < 2:
            continue

        # Compute transform from sample
        s, t = scale_translation_3d(pts_src[idx], pts_tgt[idx])

        # Transform all source points
        pts_transformed = s * pts_src + t

        # Compute distances
        dists = np.linalg.norm(pts_transformed - pts_tgt, axis=1)
        inlier_mask = dists < inlier_thresh
        num_inliers = inlier_mask.sum()

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_s = s
            best_t = t
            best_mask = inlier_mask

    # Refine using all inliers
    if best_inliers >= 2:
        best_s, best_t = scale_translation_3d(pts_src[best_mask], pts_tgt[best_mask])

    print(f"RANSAC: {best_inliers}/{N} inliers, scale={best_s:.4f}")

    return best_s, best_t, best_mask


def save_points_to_ply(
    points: np.ndarray,
    output_path: str,
    color: Tuple[int, int, int, int] = (255, 255, 255, 255),
):
    """Save 3D points to PLY file.

    Args:
        points: (N, 3) array of 3D points
        output_path: Path to save PLY file
        color: RGBA color tuple for all points
    """
    pcd = trimesh.PointCloud(points, colors=np.tile(color, (len(points), 1)))
    pcd.export(output_path)
    print(f"Saved {len(points)} points to {output_path}")


def save_alignment_results(
    out_dir: str,
    s: float,
    t: np.ndarray,
    o2c_sam3d: np.ndarray,
    K_cond: np.ndarray,
    sam3d_pts_3d: np.ndarray,
    cond_kpts: np.ndarray,
    valid_mask: np.ndarray,
    inlier_mask: np.ndarray,
    errors: np.ndarray,
    SAM3D_mesh_file: str,
):
    """Save alignment results including transform, camera, and mesh.

    Args:
        out_dir: Output directory path
        s: Scale factor
        t: Translation vector (3,)
        o2c_sam3d: Original SAM3D object-to-camera transform (4, 4)
        K_cond: Condition camera intrinsics (3, 3)
        sam3d_pts_3d: SAM3D 3D points (N, 3)
        cond_kpts: Condition keypoints (M, 2)
        valid_mask: Valid mask (M,)
        inlier_mask: Inlier mask (M,)
        errors: Alignment errors for valid points
        SAM3D_mesh_file: Path to SAM3D mesh file
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build 4x4 transform matrix: B = s * A + t
    # Represented as [s*I, t; 0, 1]
    T = np.eye(4)
    T[:3, :3] = s * np.eye(3)
    T[:3, 3] = t

    # Save transformed points
    transformed_sam3d_pts = s * sam3d_pts_3d + t
    save_points_to_ply(transformed_sam3d_pts, os.path.join(out_dir, "sam3d_pts_3d_transformed.ply"), color=(0, 0, 255, 255))

    # Save alignment results
    results = {
        "scale": float(s),
        "t": t.tolist(),
        "T_sam3d_to_cond": T.tolist(),
        "num_correspondences": int(len(cond_kpts)),
        "num_valid_3d": int(valid_mask.sum()),
        "num_inliers": int(inlier_mask.sum()),
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
    }
    with open(os.path.join(out_dir, "alignment.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved alignment to {out_dir}/alignment.json")

    # Combine scale, T with original o2c_sam3d to get new camera pose
    # The transformation chain is:
    # p_cond = T @ p_sam3d = T @ o2c_sam3d @ p_obj
    # So new_o2c = T @ o2c_sam3d
    new_o2c = T @ o2c_sam3d

    # Save to camera.json with K_cond (since we're now in condition camera space)
    camera_data = {
        "K": K_cond.tolist(),
        "blw2cvc": new_o2c.tolist(),
    }
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(camera_data, f, indent=2)
    print(f"Saved camera to {out_dir}/camera.json")

    # Transform SAM3D mesh with new_o2c and save to mesh_aligned.ply
    if os.path.exists(SAM3D_mesh_file):
        sam3d_mesh = load_mesh_from_glb(SAM3D_mesh_file)
        # Apply new_o2c transform to mesh vertices: p_cond = new_o2c @ p_obj
        verts_homogeneous = np.hstack([sam3d_mesh.vertices, np.ones((len(sam3d_mesh.vertices), 1))])
        verts_transformed = (new_o2c @ verts_homogeneous.T).T[:, :3]
        sam3d_mesh.vertices = verts_transformed
        sam3d_mesh.export(os.path.join(out_dir, "mesh_aligned.ply"))
        print(f"Saved transformed mesh to {out_dir}/mesh_aligned.ply")
    else:
        print(f"SAM3D mesh file not found: {SAM3D_mesh_file}")

    return new_o2c


def visualize_final_alignment_rerun(
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    app_name: str = "align_SAM3D_final",
):
    """Visualize final alignment: image, depth pointcloud, and aligned mesh.

    Args:
        image: RGB image (H, W, 3) uint8.
        mask: Binary mask (H, W) bool.
        depth: Depth map (H, W) in meters.
        mesh: Trimesh mesh in camera coordinate system.
        K: Camera intrinsic matrix (3, 3).
        app_name: Name for the Rerun application.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    height, width = image.shape[:2]

    # Build blueprint
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D View", origin="world"),
        rrb.Horizontal(
            rrb.Spatial2DView(name="Masked Image", origin="world/camera"),
            rrb.Spatial2DView(name="Raw Image", origin="world/image_raw"),
        )
    )
    rr.init(app_name, spawn=True, default_blueprint=blueprint)

    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Log camera frustum at origin
    rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
    rr.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[width, height], image_plane_distance=1.0))

    # Mask the image
    masked_image = image.copy()
    masked_image[~mask] = 0
    rr.log("world/camera/image", rr.Image(masked_image))
    rr.log("world/image_raw", rr.Image(image))

    # Create pointcloud from depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    valid = (z > 0) & mask
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x[valid], y[valid], z[valid]], axis=1)
    colors = image[valid]

    rr.log("world/pointcloud", rr.Points3D(
        positions=points,
        colors=colors,
        radii=0.002,
    ), static=True)

    # Log mesh
    mesh_colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        mesh_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    rr.log("world/mesh", rr.Mesh3D(
        vertex_positions=mesh.vertices,
        triangle_indices=mesh.faces,
        vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
        vertex_colors=mesh_colors,
    ), static=True)

    # Log mesh points in blue
    blue_color = np.array([0, 0, 255], dtype=np.uint8)
    num_verts = len(mesh.vertices)
    max_verts = 10000
    if num_verts > max_verts:
        sample_idx = np.random.choice(num_verts, max_verts, replace=False)
        sampled_verts = mesh.vertices[sample_idx]
    else:
        sampled_verts = mesh.vertices
    rr.log("world/mesh_points", rr.Points3D(
        positions=sampled_verts,
        colors=np.tile(blue_color, (len(sampled_verts), 1)),
        radii=0.0003,
    ), static=True)

    print(f"Launched Rerun visualization: {app_name}")


def visualize_correspondences_rerun(
    cond_image: np.ndarray,
    sam3d_image: np.ndarray,
    cond_kpts: np.ndarray,
    sam3d_kpts: np.ndarray,
    cond_pts_3d: np.ndarray,
    sam3d_pts_3d: np.ndarray,
    valid_mask: np.ndarray,
    K_cond: np.ndarray,
    K_sam3d: np.ndarray,
    s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    inlier_mask: Optional[np.ndarray] = None,
    app_name: str = "align_SAM3D_pts",
):
    """Visualize 2D and 3D correspondences in Rerun.

    Args:
        cond_image: Condition image (H, W, 3)
        sam3d_image: SAM3D image (H, W, 3)
        cond_kpts: 2D keypoints in condition image (N, 2)
        sam3d_kpts: 2D keypoints in SAM3D image (N, 2)
        cond_pts_3d: 3D points from condition depth (N, 3)
        sam3d_pts_3d: 3D points from SAM3D depth (N, 3)
        valid_mask: Mask for points with valid depth in both views (N,)
        K_cond: Condition camera intrinsics (3, 3)
        K_sam3d: SAM3D camera intrinsics (3, 3)
        s: Optional scale factor from SAM3D to condition space
        t: Optional translation vector
        inlier_mask: Optional inlier mask after optimization
        app_name: Rerun application name
    """
    import rerun as rr
    import rerun.blueprint as rrb

    H_cond, W_cond = cond_image.shape[:2]
    H_sam3d, W_sam3d = sam3d_image.shape[:2]

    # Build blueprint
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D View", origin="world"),
        rrb.Vertical(
            rrb.Spatial2DView(name="Condition Image", origin="world/cond_camera"),
            rrb.Spatial2DView(name="SAM3D Image", origin="world/sam3d_camera"),
        ),
    )
    rr.init(app_name, spawn=True, default_blueprint=blueprint)


    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Log condition camera
    rr.log("world/cond_camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
    rr.log("world/cond_camera", rr.Pinhole(image_from_camera=K_cond, resolution=[W_cond, H_cond], image_plane_distance=1.0))
    rr.log("world/cond_camera/image", rr.Image(cond_image))

    # Log SAM3D camera (offset for visualization)
    sam3d_offset = np.array([0, 0, 0])
    rr.log("world/sam3d_camera", rr.Transform3D(translation=sam3d_offset, mat3x3=np.eye(3)))
    rr.log("world/sam3d_camera", rr.Pinhole(image_from_camera=K_sam3d, resolution=[W_sam3d, H_sam3d], image_plane_distance=1.0))
    rr.log("world/sam3d_camera/image", rr.Image(sam3d_image))

    # Log 2D keypoints on images
    rr.log("world/cond_camera/keypoints", rr.Points2D(
        positions=cond_kpts[valid_mask],
        colors=np.array([0, 255, 0]),
        radii=3,
    ))
    rr.log("world/sam3d_camera/keypoints", rr.Points2D(
        positions=sam3d_kpts[valid_mask],
        colors=np.array([0, 0, 255]),
        radii=3,
    ))

    # Log 3D points
    valid_cond_pts = cond_pts_3d[valid_mask]
    valid_sam3d_pts = sam3d_pts_3d[valid_mask]

    # Color by match index
    N_valid = valid_mask.sum()
    colors = np.zeros((N_valid, 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(0, 255, N_valid).astype(np.uint8)  # Red gradient
    colors[:, 2] = np.linspace(255, 0, N_valid).astype(np.uint8)  # Blue gradient

    rr.log("world/cond_pts", rr.Points3D(
        positions=valid_cond_pts,
        colors=colors,
        radii=0.003,
    ))

    # SAM3D points (optionally transformed)
    if s is not None and t is not None:
        # Transform SAM3D points to condition space: B = s * A + t
        sam3d_pts_transformed = s * valid_sam3d_pts + t
        rr.log("world/sam3d_pts_aligned", rr.Points3D(
            positions=sam3d_pts_transformed,
            colors=colors,
            radii=0.003,
        ))

        # Log inliers in green
        if inlier_mask is not None:
            valid_inlier_mask = inlier_mask[valid_mask]
            if valid_inlier_mask.any():
                rr.log("world/inliers_cond", rr.Points3D(
                    positions=valid_cond_pts[valid_inlier_mask],
                    colors=np.array([0, 255, 0]),
                    radii=0.005,
                ))
                rr.log("world/inliers_sam3d", rr.Points3D(
                    positions=sam3d_pts_transformed[valid_inlier_mask],
                    colors=np.array([0, 255, 255]),
                    radii=0.005,
                ))
    else:
        # Log SAM3D points with offset
        rr.log("world/sam3d_pts", rr.Points3D(
            positions=valid_sam3d_pts + sam3d_offset,
            colors=colors,
            radii=0.003,
        ))

    print(f"Visualized {N_valid} valid correspondences in Rerun")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_image_path = os.path.join(args.data_dir, "rgb", f"{args.cond_index:04d}.jpg")
    cond_mask_path = os.path.join(args.data_dir, "mask_object", f"{args.cond_index:04d}.png")
    cond_depth_file = os.path.join(args.data_dir, "depth", f"{args.cond_index:04d}.png")
    cond_meta_file = os.path.join(args.data_dir, "meta", f"{args.cond_index:04d}.pkl")
    SAM3D_dir = os.path.join(args.data_dir, "SAM3D_aligned_mask", f"{args.SAM3D_index:04d}")
    SAM3D_image_file = os.path.join(SAM3D_dir, "image.png")
    SAM3D_mask_file = os.path.join(SAM3D_dir, "mask.png")
    SAM3D_depth_file = os.path.join(SAM3D_dir, "depth_aligned.png")
    SAM3D_camera_file = os.path.join(SAM3D_dir, "camera.json")
    SAM3D_mesh_file = os.path.join(args.data_dir, "SAM3D", f"{args.SAM3D_index:04d}", "scene.glb")

    # Check required files
    for f in [cond_image_path, cond_depth_file, cond_meta_file, SAM3D_image_file, SAM3D_depth_file, SAM3D_camera_file]:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return

    # Load condition image, mask, depth and intrinsic
    print("=" * 50)
    print("Loading condition data...")
    print("=" * 50)
    cond_image = load_image(cond_image_path)
    cond_mask = load_mask(cond_mask_path) if os.path.exists(cond_mask_path) else np.ones(cond_image.shape[:2], dtype=bool)
    cond_depth = load_filtered_depth(cond_depth_file)
    K_cond = load_intrinsics_from_meta(cond_meta_file)
    if K_cond.shape != (3, 3):
        K_cond = load_intrinsics_from_meta(Path(cond_meta_file).parent / "0000.pkl")
        if K_cond.shape != (3, 3):
            raise ValueError(f"Invalid intrinsics shape in {cond_meta_file}: {K_cond.shape}")
    print(f"Condition image: {cond_image.shape}, depth: {cond_depth.shape}, K: {K_cond.shape}")

    # Load SAM3D depth and intrinsic
    print("=" * 50)
    print("Loading SAM3D data...")
    print("=" * 50)
    sam3d_image = load_image(SAM3D_image_file)
    sam3d_depth = get_depth(SAM3D_depth_file)  # Already aligned, no filtering needed
    K_sam3d, o2c_sam3d = load_intrinsics_from_json(SAM3D_camera_file)
    print(f"SAM3D image: {sam3d_image.shape}, depth: {sam3d_depth.shape}, K: {K_sam3d.shape}")

    # Get correspondences between condition image and SAM3D image
    print("=" * 50)
    print("Finding correspondences with SuperPoint + LightGlue...")
    print("=" * 50)
    cond_kpts, sam3d_kpts = get_correspondences(cond_image, sam3d_image, device=device)
    print(f"Found {len(cond_kpts)} correspondences")

    # Get 3D corresponding points from depth maps
    print("=" * 50)
    print("Back-projecting to 3D...")
    print("=" * 50)
    cond_pts_3d, cond_valid = backproject_points(cond_kpts, cond_depth, K_cond)
    sam3d_pts_3d, sam3d_valid = backproject_points(sam3d_kpts, sam3d_depth, K_sam3d)
    

    # Combined valid mask (valid in both views)
    valid_mask = cond_valid & sam3d_valid
    print(f"Valid 3D correspondences: {valid_mask.sum()}/{len(valid_mask)}")

    # Save the valid 3D points to PLY files for debugging
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        valid_cond_pts = cond_pts_3d[valid_mask]
        valid_sam3d_pts = sam3d_pts_3d[valid_mask]
        save_points_to_ply(valid_cond_pts, os.path.join(args.out_dir, "cond_pts_3d.ply"), color=(0, 255, 0, 255))
        save_points_to_ply(valid_sam3d_pts, os.path.join(args.out_dir, "sam3d_pts_3d.ply"), color=(255, 0, 0, 255))
    
    if valid_mask.sum() < 20:
        print("Not enough valid correspondences for alignment")
        return

    # Visualize before optimization
    if args.vis:
        print("=" * 50)
        print("Visualizing before optimization...")
        print("=" * 50)
        visualize_correspondences_rerun(
            cond_image, sam3d_image,
            cond_kpts, sam3d_kpts,
            cond_pts_3d, sam3d_pts_3d,
            valid_mask,
            K_cond, K_sam3d,
            app_name="align_SAM3D_corres_before"
        )
    # Optimize alignment between SAM3D points and condition points
    print("=" * 50)
    print("Optimizing scale + translation alignment (RANSAC)...")
    print("=" * 50)
    valid_cond_pts = cond_pts_3d[valid_mask]
    valid_sam3d_pts = sam3d_pts_3d[valid_mask]

    s, t, inlier_mask_valid = optimize_rigid_transform(
        valid_sam3d_pts, valid_cond_pts,
        num_iters=1000,
        inlier_thresh=0.01,
    )

    # Expand inlier mask back to full size
    inlier_mask = np.zeros(len(valid_mask), dtype=bool)
    inlier_mask[valid_mask] = inlier_mask_valid

    # Compute alignment error
    aligned_pts = s * valid_sam3d_pts + t
    errors = np.linalg.norm(aligned_pts - valid_cond_pts, axis=1)
    print(f"Mean alignment error: {errors.mean():.4f} m")
    print(f"Median alignment error: {np.median(errors):.4f} m")
    print(f"Max alignment error: {errors.max():.4f} m")

    # Visualize after optimization
    if args.vis:
        print("=" * 50)
        print("Visualizing after optimization...")
        print("=" * 50)
        visualize_correspondences_rerun(
            cond_image, sam3d_image,
            cond_kpts, sam3d_kpts,
            cond_pts_3d, sam3d_pts_3d,
            valid_mask,
            K_cond, K_sam3d,
            s=s, t=t,
            inlier_mask=inlier_mask,
            app_name="align_SAM3D_corres_after"
        )
    
    # Save results
    if args.out_dir:
        save_alignment_results(
            out_dir=args.out_dir,
            s=s,
            t=t,
            o2c_sam3d=o2c_sam3d,
            K_cond=K_cond,
            sam3d_pts_3d=sam3d_pts_3d,
            cond_kpts=cond_kpts,
            valid_mask=valid_mask,
            inlier_mask=inlier_mask,
            errors=errors,
            SAM3D_mesh_file=SAM3D_mesh_file,
        )

    # Visualize final results in rerun
    if args.vis and args.out_dir:
        mesh_aligned_path = os.path.join(args.out_dir, "mesh_aligned.ply")
        if os.path.exists(mesh_aligned_path):
            aligned_mesh = trimesh.load(mesh_aligned_path)
            visualize_final_alignment_rerun(
                image=cond_image,
                mask=cond_mask,
                depth=cond_depth,
                mesh=aligned_mesh,
                K=K_cond,
                app_name="align_SAM3D_final",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-optimization for SAM-3D layout refinement")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the input RGB image.",
    )

    parser.add_argument(
        "--hand-pose-suffix",
        type=str,
        default="rot",
        help="Suffix for hand pose files.",
    )
    parser.add_argument(
        "--cond-index",
        type=int,
        default=0,
        help="Index of condition image.",
    )
    parser.add_argument(
        "--SAM3D-index",
        type=int,
        default=0,
        help="Index of SAM3D.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save optimized outputs.",
    )

    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize in rerun instead of running optimization.",
    )

    args = parser.parse_args()
    main(args)
