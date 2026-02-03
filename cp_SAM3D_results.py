# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import json
import argparse
import numpy as np
import trimesh


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


def main(args):
    src_dir = args.src_dir  # Path to the output of align_SAM3D_pts.py
    dst_dir = args.dst_dir  # Path to save converted results

    # Create output directory
    os.makedirs(dst_dir, exist_ok=True)

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

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAM3D alignment results to standard format")
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

    args = parser.parse_args()
    main(args)
