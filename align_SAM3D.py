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


def decompose_o2c_to_pose(o2c: np.ndarray, device: str = "cuda"):
    """Decompose o2c matrix back to quaternion, translation, scale.

    The o2c in camera.json was computed as:
        o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP

    We need to reverse this to get back the original transform.

    Returns:
        quaternion: (1, 1, 4) tensor
        translation: (1, 3) tensor
        scale: (1, 3) tensor
    """
    # Reverse the coordinate transforms
    # o2c = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ M @ _R_ZUP_TO_YUP
    # M = _R_ZUP_TO_YUP @ _GL_TO_CV @ o2c @ _R_ZUP_TO_YUP.T
    transform_matrix = _R_ZUP_TO_YUP @ _GL_TO_CV @ o2c @ _R_ZUP_TO_YUP.T

    # Convert to torch and create Transform3d
    # Note: transform_matrix is in row-major form, need to transpose for Transform3d
    M = torch.from_numpy(transform_matrix.T.astype(np.float32)).to(device)
    tfm = Transform3d(matrix=M.unsqueeze(0), device=device)

    # Decompose into scale, rotation, translation
    decomposed = decompose_transform(tfm)
    scale = decomposed.scale  # (1, 3)
    rotation = decomposed.rotation  # (1, 3, 3)
    translation = decomposed.translation  # (1, 3)

    # Convert rotation matrix to quaternion
    quaternion = matrix_to_quaternion(rotation)  # (1, 4)
    quaternion = quaternion.unsqueeze(1)  # (1, 1, 4)

    return quaternion, translation, scale


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
    pointmap_filtered[..., 0] = -pointmap_filtered[..., 0]  # Flip x for pytorch3d
    pointmap_filtered[..., 1] = -pointmap_filtered[..., 1]  # Flip y for pytorch3d

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
    camera_json_path = os.path.join(args.demo_out_dir, "camera.json")
    scene_glb_path = os.path.join(args.demo_out_dir, "scene.glb")



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
    K_normalized = normalize_intrinsics(K, height, width)
    print(f"Pointmap shape: {pointmap.shape}")

    # --- Load camera.json for initial pose ---
    print(f"Loading camera data from: {camera_json_path}")
    K_camera, o2c = _load_camera_data(camera_json_path)

    # --- Load mesh from GLB ---
    print(f"Loading mesh from: {scene_glb_path}")
    mesh = load_mesh_from_glb(scene_glb_path)
    print(f"Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

    # --- Decompose o2c to get initial pose parameters ---
    print("Decomposing initial pose...")
    quaternion, translation, scale = decompose_o2c_to_pose(o2c, device=device)
    print(f"Initial quaternion: {quaternion.squeeze()}")
    print(f"Initial translation: {translation.squeeze()}")
    print(f"Initial scale: {scale.squeeze()}")

    # --- Prepare inputs for post-optimization ---
    # Convert mask to tensor (H, W)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)

    # Pointmap is already a tensor (H, W, 3)
    pointmap_tensor = pointmap.to(device)

    # Intrinsics normalized
    intrinsics_tensor = torch.from_numpy(K_normalized).to(device)

    # --- Run post-optimization ---
    if args.vis:
        # Visualization mode: show current state in rerun
        print("Visualizing in rerun...")
        # Flip pointmap for visualization (reverse pytorch3d coords)
        pointmap_vis = convert_pointmap_to_pytorch3d(pointmap.numpy().copy())
        visualize_in_rerun(
            image, mask, camera_json_path,
            scene_glb_path,
            pointmap=pointmap_vis
        )
        return

    print("Running post-optimization...")

    # Load and filter depth, convert to pointmap
    pointmap_tensor = load_filtered_pointmap(args.depth_file, K, device)

    result = layout_post_optimization(
        Mesh=mesh,
        quaternion=quaternion,
        translation=translation,
        scale=scale,
        mask=mask_tensor,
        point_map=pointmap_tensor,
        intrinsics=intrinsics_tensor,
        Enable_shape_ICP=args.enable_shape_icp,
        Enable_rendering_optimization=args.enable_rendering_optimization,
        min_size=image.shape[0],
        device=device,
    )

    # Unpack results
    (
        optimized_quaternion,
        optimized_translation,
        optimized_scale,
        final_iou,
        flag_manual,
        flag_icp,
    ) = result

    print(f"\n=== Post-optimization Results ===")
    print(f"Final IoU: {final_iou:.4f}")
    print(f"Manual alignment applied: {flag_manual}")
    print(f"ICP applied: {flag_icp}")
    print(f"Optimized quaternion: {optimized_quaternion.squeeze()}")
    print(f"Optimized translation: {optimized_translation.squeeze()}")
    print(f"Optimized scale: {optimized_scale.squeeze()}")

    # --- Save optimized results ---
    os.makedirs(args.out_dir, exist_ok=True)

    # Recompute o2c from optimized parameters
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    R_optimized = quaternion_to_matrix(optimized_quaternion.squeeze(1))
    l2c_transform = compose_transform(
        scale=optimized_scale,
        rotation=R_optimized,
        translation=optimized_translation,
    )
    transform_matrix = l2c_transform.get_matrix()[0].cpu().numpy().T
    o2c_optimized = _GL_TO_CV.T @ _R_ZUP_TO_YUP.T @ transform_matrix @ _R_ZUP_TO_YUP
    # Save optimized camera.json
    camera_data_optimized = {
        "K": K.tolist(),
        "blw2cvc": o2c_optimized.tolist(),
        "final_iou": final_iou,
    }
    optimized_camera_path = os.path.join(args.out_dir, "camera_optimized.json")
    with open(optimized_camera_path, "w") as f:
        json.dump(camera_data_optimized, f, indent=2)
    print(f"Saved optimized camera to: {optimized_camera_path}")

    # Visualize optimized result in rerun
    print("Visualizing optimized result in rerun...")
    # Flip pointmap for visualization (reverse pytorch3d coords)
    pointmap_vis = convert_pointmap_to_pytorch3d(pointmap.numpy().copy())
    visualize_in_rerun(
        image, mask, optimized_camera_path,
        scene_glb_path,
        pointmap=pointmap_vis
    )


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
        "--demo-out-dir",
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
