# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Visualize rendered views from SAM3D_post_process.py output directory."""
import os
import argparse
import pickle
import numpy as np
import trimesh
import cv2
from glob import glob

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party/utils_simba"))
from utils_simba.depth import get_depth


def load_rendered_data(out_dir: str):
    """Load rendered data from SAM3D_post_process.py output directory.

    Args:
        out_dir: Path to output directory containing rgb/, depth/, meta/, mesh.obj

    Returns:
        mesh: trimesh.Trimesh object
        K: (3, 3) intrinsic matrix
        w2c_list: List of (4, 4) world-to-camera transforms
        colors: List of RGB images (H, W, 3)
        depths: List of depth maps (H, W)
    """
    # Load mesh
    mesh_path = os.path.join(out_dir, "mesh.obj")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Find all meta files
    meta_dir = os.path.join(out_dir, "meta")
    meta_files = sorted(glob(os.path.join(meta_dir, "*.pkl")))
    if not meta_files:
        raise FileNotFoundError(f"No meta files found in: {meta_dir}")
    print(f"Found {len(meta_files)} meta files")

    # Load camera parameters and images
    K = None
    w2c_list = []
    colors = []
    depths = []

    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")

    for meta_path in meta_files:
        # Load meta
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        if K is None:
            K = meta["camMat"]

        # Reconstruct w2c from objRot and objTrans
        from scipy.spatial.transform import Rotation
        objRot = meta["objRot"]
        objTrans = meta["objTrans"]
        R = Rotation.from_rotvec(objRot).as_matrix()
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = objTrans
        w2c_list.append(w2c)

        # Load RGB image
        basename = os.path.splitext(os.path.basename(meta_path))[0]
        rgb_path = os.path.join(rgb_dir, f"{basename}.jpg")
        if os.path.exists(rgb_path):
            color = cv2.imread(rgb_path)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            colors.append(color)
        else:
            print(f"Warning: RGB image not found: {rgb_path}")
            colors.append(None)

        # Load depth
        depth_path = os.path.join(depth_dir, f"{basename}.png")
        if os.path.exists(depth_path):
            depth = get_depth(depth_path)
            depths.append(depth)
        else:
            print(f"Warning: Depth not found: {depth_path}")
            depths.append(None)

    return mesh, K, w2c_list, colors, depths


def visualize_rendered_views_rerun(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    w2c_list: list,
    colors: list,
    depths: list,
    app_name: str = "rendered_views",
):
    """Visualize rendered views with camera frustums and point clouds in Rerun.

    Args:
        mesh: Trimesh mesh in object coordinates
        K: (3, 3) intrinsic matrix
        w2c_list: List of (4, 4) world-to-camera transforms
        colors: List of RGB images (H, W, 3)
        depths: List of depth maps (H, W)
        app_name: Rerun application name
    """
    import rerun as rr
    import rerun.blueprint as rrb

    # Get image dimensions from first valid color image
    height, width = None, None
    for color in colors:
        if color is not None:
            height, width = color.shape[:2]
            break

    if height is None:
        raise ValueError("No valid color images found")

    # Build blueprint
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D View", origin="world"),
        # rrb.Vertical(
        #     rrb.Spatial2DView(name="Current Image", origin="world/current_camera"),
        # ),
    )
    rr.init(app_name, spawn=True, default_blueprint=blueprint)

    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

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

    # Log each camera pose and rendered view
    for i, w2c in enumerate(w2c_list):
        if colors[i] is None or depths[i] is None:
            continue

        rr.set_time_sequence("frame", i)

        # Get camera pose (camera to world)
        c2w = np.linalg.inv(w2c)

        # Log camera frustum
        translation = c2w[:3, 3]
        mat3x3 = c2w[:3, :3]

        rr.log(f"world/camera_{i}", rr.Transform3D(translation=translation, mat3x3=mat3x3))
        rr.log(f"world/camera_{i}", rr.Pinhole(
            image_from_camera=K,
            resolution=[width, height],
            image_plane_distance=0.1,
        ))

        # Log current camera for 2D view
        rr.log("world/current_camera", rr.Transform3D(translation=translation, mat3x3=mat3x3))
        rr.log("world/current_camera", rr.Pinhole(
            image_from_camera=K,
            resolution=[width, height],
            image_plane_distance=0.1,
        ))
        rr.log(f"world/camera_{i}/image", rr.Image(colors[i]))

        # Create point cloud from depth
        depth = depths[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth
        valid = z > 0

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=1)

        # Transform to world coordinates
        pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (c2w @ pts_cam_h.T).T[:, :3]

        # Subsample for visualization
        max_points = 5000
        if len(pts_world) > max_points:
            idx = np.random.choice(len(pts_world), max_points, replace=False)
            pts_world = pts_world[idx]
            pts_colors = colors[i][valid][idx]
        else:
            pts_colors = colors[i][valid]

        rr.log(f"world/pointcloud_{i}", rr.Points3D(
            positions=pts_world,
            colors=pts_colors,
            radii=0.002,
        ))

    print(f"Launched Rerun visualization: {app_name}")


def main(args):
    print(f"Loading data from: {args.out_dir}")
    mesh, K, w2c_list, colors, depths = load_rendered_data(args.out_dir)

    print(f"Loaded {len(w2c_list)} views")
    print(f"Intrinsic matrix:\n{K}")

    visualize_rendered_views_rerun(
        mesh=mesh,
        K=K,
        w2c_list=w2c_list,
        colors=colors,
        depths=depths,
        app_name="rendered_views",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize rendered views from SAM3D_post_process.py output")
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to the output directory of SAM3D_post_process.py",
    )

    args = parser.parse_args()
    main(args)
