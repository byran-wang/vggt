# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import trimesh
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party/utils_simba"))
from utils_simba.render import nvdiffrast_render, make_mesh_tensors
from utils_simba.depth import save_depth


def load_camera_json(camera_json_path: str) -> np.ndarray:
    """Load transformation matrix from camera.json.

    Args:
        camera_json_path: Path to camera.json file

    Returns:
        blw2cvc: 4x4 transformation matrix (object to camera)
    """
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)
    blw2cvc = np.array(camera_data["blw2cvc"], dtype=np.float64)
    return blw2cvc


def decompose_transform(matrix: np.ndarray) -> dict:
    """Decompose 4x4 transformation matrix into rotation, translation, and scale.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        Dictionary with matrix, rotation, translation, scale
    """
    # Extract rotation and scale from upper-left 3x3
    R_scaled = matrix[:3, :3]

    # Compute scale as the average of column norms
    scale = np.mean([np.linalg.norm(R_scaled[:, i]) for i in range(3)])

    # Extract pure rotation by normalizing
    R = R_scaled / scale

    # Extract translation
    t = matrix[:3, 3]

    return {
        "matrix": matrix.tolist(),
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "scale": float(scale),
    }


def transform_mesh(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    """Apply 4x4 transformation to mesh vertices.

    Args:
        mesh: Input trimesh
        transform: 4x4 transformation matrix

    Returns:
        Transformed mesh
    """
    # Apply transform to vertices
    verts_homogeneous = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    verts_transformed = (transform @ verts_homogeneous.T).T[:, :3]

    # Create new mesh with transformed vertices
    transformed_mesh = trimesh.Trimesh(
        vertices=verts_transformed,
        faces=mesh.faces,
        vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
    )

    # Copy visual properties if available
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        transformed_mesh.visual.vertex_colors = mesh.visual.vertex_colors

    return transformed_mesh


def sample_camera_poses(
    num_views: int,
    elevation_range: tuple,
    azimuth_range: tuple,
    distance: float,
) -> list:
    """Sample camera poses uniformly on a sphere.

    Args:
        num_views: Number of camera views to sample
        elevation_range: (min_elev, max_elev) in degrees
        azimuth_range: (min_azim, max_azim) in degrees
        distance: Camera distance from origin

    Returns:
        List of (elevation, azimuth) tuples in degrees
    """
    elev_min, elev_max = elevation_range
    azim_min, azim_max = azimuth_range

    # Sample elevation uniformly on sphere (using sin distribution for uniform area)
    # Convert elevation to latitude for uniform sampling
    sin_elev_min = np.sin(np.deg2rad(elev_min))
    sin_elev_max = np.sin(np.deg2rad(elev_max))
    sin_elevs = np.random.uniform(sin_elev_min, sin_elev_max, num_views)
    elevations = np.rad2deg(np.arcsin(sin_elevs))

    # Sample azimuth uniformly within range
    azimuths = np.random.uniform(azim_min, azim_max, num_views)

    return list(zip(elevations, azimuths))


def spherical_to_camera_pose(
    elevation: float,
    azimuth: float,
    distance: float,
) -> np.ndarray:
    """Convert spherical coordinates to camera pose (world to camera transform).

    Camera looks at origin from the given spherical coordinates.
    Uses OpenCV convention (X right, Y down, Z forward).

    Args:
        elevation: Elevation angle in degrees (0 = horizontal, 90 = top)
        azimuth: Azimuth angle in degrees (0 = front, 90 = right)
        distance: Distance from origin

    Returns:
        w2c: 4x4 world-to-camera transformation matrix
    """
    # Convert to radians
    elev_rad = np.deg2rad(elevation)
    azim_rad = np.deg2rad(azimuth)

    # Camera position in world coordinates
    x = distance * np.cos(elev_rad) * np.sin(azim_rad)
    y = -distance * np.sin(elev_rad)  # Y is down in OpenCV
    z = distance * np.cos(elev_rad) * np.cos(azim_rad)
    cam_pos = np.array([x, y, z])

    # Camera looks at origin
    forward = -cam_pos / np.linalg.norm(cam_pos)  # Z axis (into scene)

    # Up vector (world Y down, so camera up is -Y world)
    world_up = np.array([0, -1, 0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Handle case when forward is parallel to up
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (camera axes in world frame)
    R = np.stack([right, -up, forward], axis=1)  # Columns are camera axes

    # World to camera: R.T @ (p - cam_pos)
    R_w2c = R.T
    t_w2c = -R_w2c @ cam_pos

    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c

    # flip y and z axis
    flip = np.diag([1, -1, -1, 1])
    w2c = w2c @ flip

    return w2c


def create_intrinsic_matrix(focal_length: float, width: int, height: int) -> np.ndarray:
    """Create camera intrinsic matrix.

    Args:
        focal_length: Focal length in pixels
        width: Image width
        height: Image height

    Returns:
        K: 3x3 intrinsic matrix
    """
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def render_mesh(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    w2c: np.ndarray,
    height: int,
    width: int,
    device: str = "cuda",
):
    """Render mesh from given camera pose.

    Args:
        mesh: Trimesh mesh in object coordinates
        K: 3x3 intrinsic matrix
        w2c: 4x4 world-to-camera transform
        height: Image height
        width: Image width
        device: Torch device

    Returns:
        color: (H, W, 3) RGB image [0, 255]
        depth: (H, W) depth map in meters
        mask: (H, W) binary mask
    """
    # Object to camera transform
    ob_in_cvcams = torch.from_numpy(w2c.astype(np.float32)).unsqueeze(0).to(device)

    # Render
    color, depth, _ = nvdiffrast_render(
        K=K,
        H=height,
        W=width,
        ob_in_cvcams=ob_in_cvcams,
        mesh=mesh,
    )

    # Convert to numpy
    color_np = (color[0].cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth[0].cpu().numpy()
    mask_np = (depth_np > 0).astype(np.uint8) * 255

    return color_np, depth_np, mask_np


def save_meta_pickle(
    filepath: str,
    K: np.ndarray,
    w2c: np.ndarray,
):
    """Save camera parameters in HO3D meta format.

    Args:
        filepath: Path to save pickle file
        K: 3x3 intrinsic matrix
        w2c: 4x4 world-to-camera transform (objRot, objTrans)
    """
    # Extract rotation and translation from w2c
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Convert rotation matrix to axis-angle (rodrigues)
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    objRot = rot.as_rotvec()

    meta = {
        "camMat": K,
        "objRot": objRot,
        "objTrans": t,
    }

    with open(filepath, "wb") as f:
        pickle.dump(meta, f)


def main(args):
    src_dir = args.src_dir  # Path to the output of align_SAM3D_pts.py
    dst_dir = args.dst_dir  # Path to save converted results
    elevation_range = tuple(args.elevation_range)  # for example [-30, 75]
    azimuth_range = tuple(args.azimuth_range)  # for example [-180, 180]
    distance = args.distance  # for example 1.5
    height, width = args.image_size  # for example [512, 512]
    focal_length = args.focal_length  # for example 500
    num_views = args.num_views  # number of views to render

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directories
    os.makedirs(dst_dir, exist_ok=True)
    rgb_dir = os.path.join(dst_dir, "rgb")
    mask_dir = os.path.join(dst_dir, "mask_object")
    depth_dir = os.path.join(dst_dir, "depth")
    meta_dir = os.path.join(dst_dir, "meta")
    for d in [rgb_dir, mask_dir, depth_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)

    # Load the transformation matrix from blw2cvc in src_dir/camera.json
    camera_json_path = os.path.join(src_dir, "camera.json")
    if not os.path.exists(camera_json_path):
        print(f"Camera file not found: {camera_json_path}")
        return

    print(f"Loading transformation from: {camera_json_path}")
    blw2cvc = load_camera_json(camera_json_path)
    print(f"Loaded blw2cvc matrix:\n{blw2cvc}")

    # Convert the transformation matrix to aligned_transform.json format
    transform_data = decompose_transform(blw2cvc)
    aligned_transform_path = os.path.join(dst_dir, "aligned_transform.json")
    with open(aligned_transform_path, "w") as f:
        json.dump(transform_data, f, indent=2)
    print(f"Saved aligned transform to: {aligned_transform_path}")

    # Load the transformed mesh from src_dir/mesh_aligned.ply
    mesh_aligned_path = os.path.join(src_dir, "mesh_aligned.ply")
    if not os.path.exists(mesh_aligned_path):
        print(f"Mesh file not found: {mesh_aligned_path}")
        return

    print(f"Loading mesh from: {mesh_aligned_path}")
    mesh_aligned = trimesh.load(mesh_aligned_path)
    print(f"Loaded mesh: {len(mesh_aligned.vertices)} vertices, {len(mesh_aligned.faces)} faces")

    # Transform the mesh to object coordinate system using inverse of blw2cvc
    # blw2cvc transforms from object to camera, so inverse transforms from camera to object
    cvc2blw = np.linalg.inv(blw2cvc)
    print(f"Transforming mesh to object coordinates...")
    mesh_object = transform_mesh(mesh_aligned, cvc2blw)

    # Save to dst_dir/mesh.obj
    mesh_obj_path = os.path.join(dst_dir, "mesh.obj")
    mesh_object.export(mesh_obj_path)
    print(f"Saved mesh to: {mesh_obj_path}")

    # Sample camera poses
    print(f"Sampling {num_views} camera poses...")
    print(f"  Elevation range: {elevation_range}")
    print(f"  Azimuth range: {azimuth_range}")
    print(f"  Distance: {distance}")
    camera_poses = sample_camera_poses(num_views, elevation_range, azimuth_range, distance)

    # Create intrinsic matrix
    K = create_intrinsic_matrix(focal_length, width, height)
    print(f"Intrinsic matrix:\n{K}")

    # Render from each camera pose
    print(f"Rendering {num_views} views...")
    for i, (elev, azim) in tqdm(enumerate(camera_poses), total=num_views, desc="Rendering"):
        # Get camera pose (world to camera)
        w2c = spherical_to_camera_pose(elev, azim, distance)

        # Render mesh
        color, depth, mask = render_mesh(mesh_object, K, w2c, height, width, device)

        # Save RGB image
        rgb_path = os.path.join(rgb_dir, f"{i:04d}.jpg")
        cv2.imwrite(rgb_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Save mask
        mask_path = os.path.join(mask_dir, f"{i:04d}.png")
        cv2.imwrite(mask_path, mask)

        # Save depth
        depth_path = os.path.join(depth_dir, f"{i:04d}.png")
        save_depth(depth, depth_path)

        # Save meta (camera parameters)
        meta_path = os.path.join(meta_dir, f"{i:04d}.pkl")
        save_meta_pickle(meta_path, K, w2c)


    print(f"\nSaved rendered data to: {dst_dir}")
    print(f"  - RGB images: {rgb_dir}")
    print(f"  - Masks: {mask_dir}")
    print(f"  - Depth maps: {depth_dir}")
    print(f"  - Camera parameters: {meta_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAM3D alignment results to standard format and render views")
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Path to the output directory of align_SAM3D_pts.py",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        required=True,
        help="Path to save converted results",
    )
    parser.add_argument(
        "--elevation-range",
        type=float,
        nargs=2,
        default=[-30, 75],
        help="Elevation range in degrees [min, max]",
    )
    parser.add_argument(
        "--azimuth-range",
        type=float,
        nargs=2,
        default=[-180, 180],
        help="Azimuth range in degrees [min, max]",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.0,
        help="Camera distance from origin",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size [height, width]",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=350,
        help="Focal length in pixels",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=100,
        help="Number of views to render",
    )

    args = parser.parse_args()
    main(args)
