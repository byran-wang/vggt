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


sys.path.append("notebook")


from third_party.utils_simba.utils_simba.depth import (
    get_depth,
    depth2xyzmap,
    erode_depth_map_torch,
    bilateral_filter_depth,
    remove_depth_outliers,
)

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask

def load_intrinsics(meta_file):
    """Load camera intrinsics from meta pickle file."""
    with open(meta_file, "rb") as f:
        meta_data = pickle.load(f)
    return np.array(meta_data["camMat"], dtype=np.float32)

def load_mesh_from_glb(glb_path: str) -> trimesh.Trimesh:
    """Load mesh from GLB file.

    Returns:
        Combined trimesh mesh from all geometries in the GLB file.
    """
    loaded = trimesh.load(glb_path)

    if isinstance(loaded, trimesh.Scene):
        # Combine all meshes in the scene
        meshes = list(loaded.geometry.values())
        if len(meshes) == 1:
            return meshes[0]
        else:
            return trimesh.util.concatenate(meshes)
    else:
        return loaded


def load_pointmap_from_depth(depth_file, K, thresh_min=0.01, thresh_max=1.5):
    """Load depth and convert to pointmap using intrinsics K."""
    # Load depth
    depth = get_depth(depth_file)

    # Convert depth to pointmap (H, W, 3)
    pointmap = depth2xyzmap(depth, K)
    # if the depth of pointmap is less than thresh_min and greater than thresh_max meter set to nan
    pointmap[(pointmap[..., 2] <= thresh_min) | (pointmap[..., 2] >= thresh_max)] = np.nan


    # Convert to torch tensor
    pointmap = torch.from_numpy(pointmap).float()

    return pointmap

def _load_camera_data(camera_json_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics (K) and object-to-camera transform (o2c) from JSON."""
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)
    K = np.array(camera_data["K"], dtype=np.float32)
    o2c = np.array(camera_data["blw2cvc"], dtype=np.float32)
    return K, o2c




def visualize_alignment_rerun(
    image: np.ndarray,
    mask: np.ndarray,
    pointmap: torch.Tensor,
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    app_name: str = "align_SAM3D",
):
    """Visualize camera frustum, pointmap, and mesh in Rerun.

    Args:
        image: RGB image (H, W, 3) uint8.
        mask: Binary mask (H, W) bool.
        pointmap: Point cloud from depth (H, W, 3) torch tensor in camera coords.
        mesh: Trimesh mesh in camera coordinate system.
        K: Camera intrinsic matrix (3, 3).
        app_name: Name for the Rerun application.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    height, width = image.shape[:2]

    # Build blueprint: 3D view and 2D image view side by side
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D View", origin="world"),
        rrb.Spatial2DView(name="Image", origin="world/camera"),
    )
    rr.init(app_name, spawn=True, default_blueprint=blueprint)

    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Log camera frustum at origin (camera coordinate system)
    rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
    rr.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[width, height]))
    rr.log("world/camera", rr.Image(image))

    # Log pointmap as point cloud (filter out NaN values)
    pointmap_np = pointmap.numpy() if torch.is_tensor(pointmap) else pointmap
    valid_mask_pts = ~np.isnan(pointmap_np).any(axis=-1) & mask
    valid_points = pointmap_np[valid_mask_pts]
    valid_colors = image[valid_mask_pts]
    rr.log("world/pointmap", rr.Points3D(
        positions=valid_points,
        colors=valid_colors,
        radii=0.002,
    ), static=True)

    # Log mesh in camera coordinate system
    mesh_colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        mesh_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    rr.log("world/mesh", rr.Mesh3D(
        vertex_positions=mesh.vertices,
        triangle_indices=mesh.faces,
        vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
        vertex_colors=mesh_colors,
    ), static=True)

    print("Rerun visualization launched.")


def transform_mesh_to_camera(mesh: trimesh.Trimesh, o2c: np.ndarray) -> trimesh.Trimesh:
    """Transform mesh from object coordinate system to camera coordinate system.

    Args:
        mesh: Input mesh in object coordinates.
        o2c: Object-to-camera transform (4x4 matrix).

    Returns:
        Mesh transformed to camera coordinates.
    """
    mesh_vertices_homo = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    mesh_vertices_cam = (o2c @ mesh_vertices_homo.T).T[:, :3]

    vertex_normals = None
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        vertex_normals = mesh.vertex_normals @ o2c[:3, :3].T

    mesh_in_cam = trimesh.Trimesh(
        vertices=mesh_vertices_cam,
        faces=mesh.faces,
        vertex_normals=vertex_normals,
    )

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        mesh_in_cam.visual.vertex_colors = mesh.visual.vertex_colors

    return mesh_in_cam


def load_filtered_pointmap(
    depth_file: str,
    K: np.ndarray,
    device: str,
    thresh_min: float = 0.01,
    thresh_max: float = 1.5,
) -> torch.Tensor:
    """Load depth, apply filtering, and convert to pointmap tensor.

    Args:
        depth_file: Path to the depth file (PNG encoded)
        K: Camera intrinsics matrix (3x3)
        device: torch device
        thresh_min: Minimum depth threshold (meters)
        thresh_max: Maximum depth threshold (meters)

    Returns:
        pointmap_tensor: (H, W, 3) tensor in pytorch3d coordinate system
    """
    # Load raw depth
    depth = get_depth(depth_file)
    depth_tensor = torch.from_numpy(depth).float()

    # Filter the depth
    print("Filtering depth...")
    depth_tensor = erode_depth_map_torch(depth_tensor, structure_size=2, d_thresh=0.003, frac_req=0.5)
    depth_tensor = bilateral_filter_depth(depth_tensor, d=5, sigma_color=0.2, sigma_space=15)
    depth_tensor = remove_depth_outliers(depth_tensor, num_std=4.0, num_iterations=3)

    # Convert filtered depth to pointmap
    depth_filtered = depth_tensor.numpy()
    pointmap_filtered = depth2xyzmap(depth_filtered, K)

    # Apply depth thresholds and convert to pytorch3d coords
    pointmap_filtered[(pointmap_filtered[..., 2] <= thresh_min) | (pointmap_filtered[..., 2] >= thresh_max)] = np.nan

    pointmap_tensor = torch.from_numpy(pointmap_filtered).float().to(device)
    print(f"Filtered pointmap shape: {pointmap_tensor.shape}")

    return pointmap_tensor


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check required input files
    if not os.path.exists(args.image_path):
        print(f"Image {args.image_path} not found.")
        return
    if not os.path.exists(args.mask_path):
        print(f"Mask {args.mask_path} not found.")
        return
    if not os.path.exists(args.depth_file):
        print(f"Depth file {args.depth_file} not found.")
        return
    if not os.path.exists(args.meta_file):
        print(f"Meta file {args.meta_file} not found.")
        return

    # Check demo.py outputs
    camera_json_path = os.path.join(args.SAM3D_dir, "camera.json")
    scene_glb_path = os.path.join(args.SAM3D_dir, "scene.glb")


    # --- Load image, mask ---
    print(f"Loading image: {args.image_path}")
    image = load_image(args.image_path)
    print(f"Loading mask: {args.mask_path}")
    mask = load_mask(args.mask_path)
    height, width = image.shape[:2]

    # --- Load depth and intrinsics to create pointmap ---
    print(f"Loading depth: {args.depth_file}")
    print(f"Loading intrinsics from: {args.meta_file}")
    K = load_intrinsics(args.meta_file)
    pointmap = load_pointmap_from_depth(args.depth_file, K)
 

    # --- Load camera.json for initial pose ---
    print(f"Loading camera data from: {camera_json_path}")
    K_camera, o2c = _load_camera_data(camera_json_path)

    # --- Load mesh from GLB ---
    print(f"Loading mesh from: {scene_glb_path}")
    mesh = load_mesh_from_glb(scene_glb_path)
    print(f"Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

    # Convert the mesh from object coordinate system to camera coordinate system
    mesh_in_cam = transform_mesh_to_camera(mesh, o2c)
    print(f"Mesh transformed to camera coordinate system")

    # Visualize camera frustum, pointmap, and mesh in Rerun
    visualize_alignment_rerun(image, mask, pointmap, mesh_in_cam, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-optimization for SAM-3D layout refinement")
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to the input mask image.",
    )
    parser.add_argument(
        "--depth-file",
        type=str,
        required=True,
        help="Path to the depth file (PNG encoded).",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        required=True,
        help="Path to the meta pickle file containing intrinsics (camMat key).",
    )
    parser.add_argument(
        "--SAM3D-dir",
        type=str,
        required=True,
        help="Directory containing demo.py outputs (camera.json, scene.glb).",
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
