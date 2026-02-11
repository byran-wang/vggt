# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import json
import os
import shutil
import numpy as np
import torch
import argparse
import trimesh
from PIL import Image
import pickle
from pathlib import Path
from typing import Optional


sys.path.append("notebook")
sys.path.append("viewer")

from third_party.utils_simba.utils_simba.depth import (
    get_depth,
    depth2xyzmap,
    save_depth,
    load_filtered_pointmap,
)

from third_party.utils_simba.utils_simba.render import (
    diff_renderer,
    make_mesh_tensors,
    projection_matrix_from_intrinsics,
    projection_matrix_to_intrinsics,
    nvdiffrast_render,
)
from viewer_step import HandDataProvider, compute_vertex_normals

# Try to import seal_mano_mesh_np
try:
    from common.body_models import seal_mano_mesh_np
except Exception:
    seal_mano_mesh_np = None

LIGHT_RED = [200, 180, 220, 255]

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    # If RGBA, use alpha channel as mask
    if mask.ndim == 3 and mask.shape[-1] == 4:
        mask = mask[..., 3] > 0
    elif mask.ndim == 3:
        mask = mask.any(axis=-1) > 0
    else:
        mask = mask > 0
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




def visualize_optimization_rerun(
    image: np.ndarray,
    mask: np.ndarray,
    pointmap: torch.Tensor,
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    hand_verts: Optional[np.ndarray] = None,
    hand_faces: Optional[np.ndarray] = None,
    app_name: str = "align_SAM3D",
):
    """Visualize camera frustum, pointmap, mesh, and hand in Rerun.

    Args:
        image: RGB image (H, W, 3) uint8.
        mask: Binary mask (H, W) bool.
        pointmap: Point cloud from depth (H, W, 3) torch tensor in camera coords.
        mesh: Trimesh mesh in camera coordinate system.
        K: Camera intrinsic matrix (3, 3).
        hand_verts: Hand vertices in camera coordinates (N, 3), optional.
        hand_faces: Hand faces (F, 3), optional.
        app_name: Name for the Rerun application.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    height, width = image.shape[:2]

    # Build blueprint: 3D view and 2D image view side by side
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D View", origin="world"),
        rrb.Vertical(
        rrb.Spatial2DView(name="Image", origin="world/camera"),
        rrb.Spatial2DView(name="image_raw", origin="world/image_raw"),
        )
    )
    rr.init(app_name, spawn=True, default_blueprint=blueprint)

    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Log camera frustum at origin (camera coordinate system)
    rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
    rr.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[width, height], image_plane_distance=1.0))
    # Mask the image with the mask (set non-masked pixels to black)
    masked_image = image.copy()
    masked_image[~mask] = 0
    rr.log("world/camera/image", rr.Image(masked_image))
    rr.log("world/image_raw", rr.Image(image))



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
    rr.log("world/object/mesh", rr.Mesh3D(
        vertex_positions=mesh.vertices,
        triangle_indices=mesh.faces,
        vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
        vertex_colors=mesh_colors,
    ), static=True)

    # Log 3D mesh points with blue color (randomly sample up to 5000 for visualization)
    blue_color = np.array([0, 0, 255], dtype=np.uint8)
    num_verts = len(mesh.vertices)
    if num_verts > 5000:
        sample_idx = np.random.choice(num_verts, 5000, replace=False)
        sampled_verts = mesh.vertices[sample_idx]
    else:
        sampled_verts = mesh.vertices
    rr.log("world/object/points", rr.Points3D(
        positions=sampled_verts,
        colors=np.tile(blue_color, (len(sampled_verts), 1)),
        radii=0.0003,
    ), static=True)

    # Log hand mesh if available
    if hand_verts is not None and hand_faces is not None:
        hand_verts = np.asarray(hand_verts)
        hand_faces = np.asarray(hand_faces, dtype=np.int32)

        # Optionally seal the MANO mesh
        if seal_mano_mesh_np is not None:
            try:
                hand_verts, hand_faces = seal_mano_mesh_np(hand_verts[None], hand_faces, is_rhand=True)
                hand_verts = np.asarray(hand_verts)[0]
            except Exception as e:
                print(f"[visualize_alignment_rerun] seal_mano_mesh_np failed: {e}")

        # Compute vertex normals
        vertex_normals = compute_vertex_normals(hand_verts, hand_faces)

        # Log hand as points
        color_rgb = np.array(LIGHT_RED[:3], dtype=np.uint8)
        rr.log("world/hand/points", rr.Points3D(
            positions=hand_verts,
            colors=np.tile(color_rgb, (hand_verts.shape[0], 1)),
            radii=0.0005,
        ), static=True)

        # Log hand as mesh
        rr.log("world/hand/mesh", rr.Mesh3D(
            vertex_positions=hand_verts,
            triangle_indices=hand_faces,
            vertex_normals=vertex_normals,
            vertex_colors=np.tile(color_rgb, (hand_verts.shape[0], 1)),
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


def load_hand_pose(
    hand_pose_dir: str,
    hand_pose_suffix: str = "rot",
    hand_index: int = 0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load hand pose vertices and faces using HandDataProvider.

    Args:
        hand_pose_dir: Directory containing hand pose data files.
        hand_pose_suffix: Suffix for hand pose mode (e.g., "rot", "trans", "intrinsic", "pose", "all").
        hand_index: Index of the hand pose to use.

    Returns:
        Tuple of (hand_verts, hand_faces) or (None, None) if not found.
    """
    if not hand_pose_dir or not os.path.exists(hand_pose_dir):
        print(f"[load_hand_pose] Hand pose directory not found: {hand_pose_dir}")
        return None, None

    print(f"Loading hand pose from: {hand_pose_dir}")
    hand_provider = HandDataProvider(Path(hand_pose_dir))

    if not hand_provider.has_hand:
        print("[load_hand_pose] HandDataProvider has no hand data")
        return None, None

    hand_verts = hand_provider.get_hand_verts_cam(hand_pose_suffix, hand_index)
    hand_faces = hand_provider.get_hand_faces(hand_pose_suffix)

    if hand_verts is not None:
        print(f"Loaded hand pose: {len(hand_verts)} vertices, mode={hand_pose_suffix}, index={hand_index}")
    else:
        print(f"[load_hand_pose] No hand vertices found for mode={hand_pose_suffix}, index={hand_index}")

    return hand_verts, hand_faces


def optimize_o2c_with_mask(
    mesh: trimesh.Trimesh,
    hand_verts: Optional[np.ndarray],
    hand_faces: Optional[np.ndarray],
    target_mask: np.ndarray,
    intrinsic: np.ndarray,
    o2c_init: np.ndarray,
    device: str,
    num_iters: int = 200,
    lr: float = 1e-2,
    debug_dir: Optional[str] = None,
) -> tuple[np.ndarray, float]:
    """Optimize object-to-camera transform to fit rendered mesh silhouette to target mask.

    Args:
        mesh: Object mesh in object coordinate system.
        hand_verts: Hand vertices in camera coordinates (already transformed), optional.
        hand_faces: Hand faces, optional.
        target_mask: Target binary mask (H, W).
        intrinsic: Camera intrinsic matrix (3, 3).
        o2c_init: Initial object-to-camera transform (4, 4).
        device: Torch device.
        num_iters: Number of optimization iterations.
        lr: Learning rate.
        debug_dir: Optional directory to save debug images.

    Returns:
        Tuple of (optimized_o2c, final_iou_loss).
    """
    import nvdiffrast.torch as dr
    from utils_simba.render import (
        diff_renderer,
        make_mesh_tensors,
        projection_matrix_from_intrinsics,
    )
    from utils_simba.geometry import matrix_to_axis_angle_t, axis_angle_t_to_matrix

    H, W = target_mask.shape
    resolution = (H, W)

    # Prepare mesh tensors (object in object space)
    obj_verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)
    obj_faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)

    # Green color for silhouette rendering
    color_obj = torch.zeros_like(obj_verts)
    color_obj[..., 1] = 1.0  # Green channel

    # Prepare hand mesh if available (hand is already in camera space, so we don't transform it with o2c)
    hand_verts_t = None
    hand_faces_t = None
    hand_color = None
    if hand_verts is not None and hand_faces is not None:
        hand_verts_np = np.asarray(hand_verts)
        hand_faces_np = np.asarray(hand_faces, dtype=np.int32)

        # Seal hand mesh if possible
        if seal_mano_mesh_np is not None:
            try:
                hand_verts_np, hand_faces_np = seal_mano_mesh_np(hand_verts_np[None], hand_faces_np, is_rhand=True)
                hand_verts_np = np.asarray(hand_verts_np)[0]
            except Exception:
                pass

        hand_verts_t = torch.tensor(hand_verts_np, dtype=torch.float32, device=device).unsqueeze(0)
        hand_faces_t = torch.tensor(hand_faces_np, dtype=torch.int32, device=device)
        hand_color = torch.zeros_like(hand_verts_t)
        hand_color[..., 1] = 1.0  # Green channel

    # Build projection matrix
    K = intrinsic.astype(np.float64)
    projection = torch.tensor(
        projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.01, zfar=100),
        dtype=torch.float32,
        device=device,
    )

    # Prepare target mask
    target_mask_t = torch.tensor(target_mask.astype(np.float32), device=device)

    # Initialize o2c transform
    o2c_t = torch.tensor(o2c_init, dtype=torch.float32, device=device)

    # Decompose into axis-angle and translation for optimization
    o2c_r_orig, o2c_t_orig, o2c_s_orig = matrix_to_axis_angle_t(o2c_t)

    # Pose residual: [3 for rotation, 3 for translation]
    pose_residual = torch.nn.Parameter(torch.zeros(6, device=device, dtype=torch.float32))

    # Setup optimizer
    optimizer = torch.optim.Adam([pose_residual], lr=lr)

    # Create nvdiffrast context
    glctx = dr.RasterizeCudaContext() if 'cuda' in device else dr.RasterizeGLContext()

    # Identity transform for hand (already in camera space)
    identity = torch.eye(4, dtype=torch.float32, device=device)

    # Optimization loop
    best_loss = float("inf")
    best_pose_residual = None

    for it in range(num_iters):
        optimizer.zero_grad()

        # Build o2c with residual
        o2c_r = o2c_r_orig + pose_residual[:3] * 0.1  # Scale rotation step
        o2c_t_new = o2c_t_orig + pose_residual[3:]
        o2c_current = axis_angle_t_to_matrix(o2c_r, o2c_t_new, o2c_s_orig)

        # Render object silhouette (transform object verts by o2c)
        rgb_obj, _ = diff_renderer(
            obj_verts, obj_faces, color_obj, projection, o2c_current, resolution, glctx
        )
        sil_pred = rgb_obj[..., 1]  # Green channel as silhouette

        # Render hand silhouette if available (hand already in camera space, use identity)
        if hand_verts_t is not None:
            rgb_hand, _ = diff_renderer(
                hand_verts_t, hand_faces_t, hand_color, projection, identity, resolution, glctx
            )
            sil_hand = rgb_hand[..., 1]
            # Combine silhouettes (union)
            sil_pred = torch.clamp(sil_pred + sil_hand, 0, 1)

        # Compute IoU loss
        inter = (sil_pred * target_mask_t).sum()
        union = (sil_pred + target_mask_t).sum() - inter
        iou = inter / (union + 1e-6)
        loss = 1.0 - iou

        # Save debug images periodically
        if debug_dir and it % 50 == 0:
            os.makedirs(debug_dir, exist_ok=True)
            sil_np = sil_pred.detach().cpu().numpy()
            Image.fromarray((sil_np * 255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"sil_{it:04d}.png")
            )
            if it == 0:
                tgt_np = target_mask_t.detach().cpu().numpy()
                Image.fromarray((tgt_np * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, "target_mask.png")
                )

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_pose_residual = pose_residual.detach().clone()

        # Backward and step
        loss.backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            print(f"[optimize_o2c] Iter {it+1:04d}: IoU loss={loss.item():.4f}, IoU={iou.item():.4f}")

    # Reconstruct best o2c
    if best_pose_residual is not None:
        o2c_r_best = o2c_r_orig + best_pose_residual[:3] * 0.1
        o2c_t_best = o2c_t_orig + best_pose_residual[3:]
        o2c_optimized = axis_angle_t_to_matrix(o2c_r_best, o2c_t_best, o2c_s_orig)
        o2c_optimized_np = o2c_optimized.detach().cpu().numpy()
    else:
        o2c_optimized_np = o2c_init

    return o2c_optimized_np, best_loss


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = os.path.join(args.data_dir, "rgb", f"{args.cond_index:04d}.jpg")
    obj_mask_path = os.path.join(args.data_dir, "mask_object", f"{args.cond_index:04d}.png")
    inpaint_image_path = os.path.join(args.data_dir, "inpaint", f"{args.cond_index:04d}_rgba.png")
    inpaint_mask_path = os.path.join(args.data_dir, "inpaint", f"{args.cond_index:04d}_rgba.png")
    hand_mask_path = os.path.join(args.data_dir, "mask_hand", f"{args.cond_index:04d}.png")
    hand_mask_path = ""
    depth_file = os.path.join(args.data_dir, "depth", f"{args.cond_index:04d}.png")
    meta_file = os.path.join(args.data_dir, "meta", f"{args.cond_index:04d}.pkl")
    SAM3D_dir = os.path.join(args.data_dir, "SAM3D", f"{args.cond_index:04d}")
    hand_pose_dir = args.data_dir
    hand_pose_dir = ""

    # Check required input files
    if not os.path.exists(inpaint_image_path):
        print(f"Image {inpaint_image_path} not found.")
        return
    if not os.path.exists(inpaint_mask_path):
        print(f"Mask {inpaint_mask_path} not found.")
        return
    if not os.path.exists(depth_file):
        print(f"Depth file {depth_file} not found.")
        return
    if not os.path.exists(meta_file):
        print(f"Meta file {meta_file} not found.")
        return

    # Check demo.py outputs
    camera_json_path = os.path.join(SAM3D_dir, "camera.json")
    scene_glb_path = os.path.join(SAM3D_dir, "scene.glb")


    # --- Load image, mask ---
    print(f"Loading inpaint image: {inpaint_image_path}")
    inpaint_image = load_image(inpaint_image_path)
    print(f"Loading inpaint mask: {inpaint_mask_path}")
    inpaint_mask = load_mask(inpaint_mask_path)
    height, width = inpaint_image.shape[:2]

    # --- Load depth and intrinsics to create pointmap ---
    print(f"Loading depth: {depth_file}")
    print(f"Loading intrinsics from: {meta_file}")
    K = load_intrinsics(meta_file)
    if K.shape != (3, 3):
        from pathlib import Path
        K = load_intrinsics(Path(meta_file).parent / "0000.pkl")
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsics K has invalid shape: {K.shape}")
    # pointmap = load_pointmap_from_depth(depth_file, K)
    pointmap = load_filtered_pointmap(depth_file, K, device).cpu().numpy()
 

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

    # Load hand pose
    hand_pose_suffix = args.hand_pose_suffix if args.hand_pose_suffix else "rot"
    cond_index = args.cond_index if args.cond_index is not None else 0
    hand_verts, hand_faces = load_hand_pose(hand_pose_dir, hand_pose_suffix, cond_index)

    # Load hand mask
    print(f"Loading hand mask: {hand_mask_path}")
    if os.path.exists(hand_mask_path):
        hand_mask = load_mask(hand_mask_path)
    else:
        print(f"[WARNING] Hand mask not found: {hand_mask_path}, using empty mask")
        hand_mask = np.zeros_like(inpaint_mask)

    # Merge the object mask and hand mask to merged mask for optimization
    merged_mask = inpaint_mask | hand_mask
    print(f"Merged mask: obj={inpaint_mask.sum()} + hand={hand_mask.sum()} = merged={merged_mask.sum()} pixels")

    # Merge the object mesh and hand mesh to merged mesh for optimization
    merged_mesh = mesh_in_cam
    if hand_verts is not None and hand_faces is not None:
        hand_verts_np = np.asarray(hand_verts)
        hand_faces_np = np.asarray(hand_faces, dtype=np.int32)

        # Optionally seal the MANO mesh
        if seal_mano_mesh_np is not None:
            try:
                hand_verts_np, hand_faces_np = seal_mano_mesh_np(hand_verts_np[None], hand_faces_np, is_rhand=True)
                hand_verts_np = np.asarray(hand_verts_np)[0]
            except Exception as e:
                print(f"[WARNING] seal_mano_mesh_np failed: {e}")

        hand_mesh = trimesh.Trimesh(vertices=hand_verts_np, faces=hand_faces_np)
        merged_mesh = trimesh.util.concatenate([mesh_in_cam, hand_mesh])
        print(f"Merged mesh: obj={len(mesh_in_cam.vertices)} + hand={len(hand_verts_np)} = {len(merged_mesh.vertices)} vertices")

    # Visualize camera frustum, pointmap, mesh, and hand in Rerun (before optimization)
    if args.vis:
        visualize_optimization_rerun(inpaint_image, merged_mask, pointmap, mesh_in_cam, K, hand_verts, hand_faces,
                                    app_name="align_SAM3D_before_optimized")

    # Optimize the o2c to fit the rendered merged mesh to merged mask
    print("=" * 50)
    print("Starting mask-based pose optimization...")
    print("=" * 50)

    optimized_o2c, final_loss = optimize_o2c_with_mask(
        mesh=mesh,  # Original mesh in object space
        hand_verts=hand_verts,
        hand_faces=hand_faces,
        target_mask=merged_mask,
        intrinsic=K,
        o2c_init=o2c,
        device=device,
        num_iters=getattr(args, 'num_iters', 200),
        lr=getattr(args, 'lr', 1e-2),
        debug_dir=os.path.join(args.out_dir, "debug") if args.out_dir else None,
    )

    print(f"Optimization complete. Final IoU loss: {final_loss:.4f}")

    # Transform mesh with optimized pose
    mesh_in_cam_optimized = transform_mesh_to_camera(mesh, optimized_o2c)

    # Save optimized results
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        # Save optimized o2c
        optimized_camera_data = {
            "K": K.tolist(),
            "blw2cvc": optimized_o2c.tolist(),
        }
        with open(os.path.join(args.out_dir, "camera.json"), "w") as f:
            json.dump(optimized_camera_data, f, indent=2)
        print(f"Saved optimized camera to {args.out_dir}/camera.json")

        # Copy image to output directory
        shutil.copy(image_path, os.path.join(args.out_dir, "image.png"))
        print(f"Copied inpaint image to {args.out_dir}/image.png")

        # Copy mask to output directory
        shutil.copy(obj_mask_path, os.path.join(args.out_dir, "mask.png"))
        print(f"Copied object mask to {args.out_dir}/mask.png")

        # Save optimized mesh
        mesh_in_cam_optimized.export(os.path.join(args.out_dir, "mesh_aligned.ply"))
        print(f"Saved optimized mesh to {args.out_dir}/mesh_aligned.ply")

        # Render depth from optimized mesh and save
        ob_in_cvcams = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)  # Identity since mesh is already in camera space
        _, rendered_depth, _ = nvdiffrast_render(
            K=K,
            H=height,
            W=width,
            ob_in_cvcams=ob_in_cvcams,
            mesh=mesh_in_cam_optimized,
        )
        rendered_depth_np = rendered_depth[0].cpu().numpy()
        save_depth(rendered_depth_np, os.path.join(args.out_dir, "depth_aligned.png"))
        print(f"Saved rendered depth to {args.out_dir}/depth_aligned.png")

    # Visualize after optimization
    if args.vis:
        visualize_optimization_rerun(
            inpaint_image, merged_mask, pointmap, mesh_in_cam_optimized, K, hand_verts, hand_faces,
            app_name="align_SAM3D_after_optimized"
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
        required=True,
        help="Suffix for hand pose files.",
    )
    parser.add_argument(
        "--cond-index",
        type=int,
        help="Index of the hand pose to use.",
    )    

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save optimized outputs.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1000,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for optimization.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize in rerun instead of running optimization.",
    )


    args = parser.parse_args()
    main(args)
