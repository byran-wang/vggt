"""
Visualize hand-object distance for each frame using Rerun.
Similar to ARCTIC InterField visualization.

Usage:
    python viewer/viewer_distance.py --seq_name MC1
    python viewer/viewer_distance.py --seq_name MC1 --use_pred --result_folder output/MC1/results
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

# Add project paths
_CODE_DIR = Path(__file__).resolve().parents[1]
if _CODE_DIR.is_dir():
    sys.path = [str(_CODE_DIR)] + sys.path
    sys.path.append(str(_CODE_DIR / "third_party/utils_simba"))

from utils_simba.rerun import Visualizer, add_material
from common.body_models import seal_mano_mesh_np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize hand-object distance")
    parser.add_argument("--seq_name", type=str, required=True, help="Sequence name (e.g., MC1)")
    parser.add_argument("--use_pred", action="store_true", help="Use predicted hand/object instead of GT")
    parser.add_argument("--result_folder", type=str, default=None, help="Path to prediction results folder")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to visualize")
    parser.add_argument("--distance_threshold", type=float, default=0.05, help="Distance threshold for colormap (meters)")
    parser.add_argument("--colormap", type=str, default="plasma", help="Colormap name (plasma, viridis, jet, etc.)")
    parser.add_argument("--rrd_output_path", type=str, default=None, help="Save to .rrd file")
    return parser.parse_args()


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute vertex normals from mesh vertices and faces.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices

    Returns:
        normals: (N, 3) normalized vertex normals
    """
    # Initialize vertex normals
    vertex_normals = np.zeros_like(vertices)

    # Get vertices for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face normals (cross product of edges)
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Accumulate face normals to vertices
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    vertex_normals = vertex_normals / norms

    return vertex_normals


def compute_hand_object_distance(hand_verts: np.ndarray, object_verts: np.ndarray) -> np.ndarray:
    """
    Compute minimum distance from each hand vertex to object surface.

    Args:
        hand_verts: (N, 3) hand vertices
        object_verts: (M, 3) object vertices

    Returns:
        distances: (N,) distance for each hand vertex
    """
    tree = cKDTree(object_verts)
    distances, _ = tree.query(hand_verts)
    return distances


def compute_object_hand_distance(object_verts: np.ndarray, hand_verts: np.ndarray) -> np.ndarray:
    """
    Compute minimum distance from each object vertex to hand surface.

    Args:
        object_verts: (M, 3) object vertices
        hand_verts: (N, 3) hand vertices

    Returns:
        distances: (M,) distance for each object vertex
    """
    tree = cKDTree(hand_verts)
    distances, _ = tree.query(object_verts)
    return distances


def distance_to_color(distances: np.ndarray, cmap_name: str = "plasma", threshold: float = 0.05) -> np.ndarray:
    """
    Convert distances to RGB colors using colormap.
    Closer = brighter (yellow in plasma), farther = darker (purple in plasma).

    Args:
        distances: (N,) distances in meters
        cmap_name: matplotlib colormap name
        threshold: distance threshold for normalization

    Returns:
        colors: (N, 3) RGB colors in [0, 255]
    """
    cmap = cm.get_cmap(cmap_name)

    # Exponential mapping: closer = higher value = brighter color
    # exp(-20 * d) maps d=0 -> 1.0, d=0.05 -> ~0.37, d=0.1 -> ~0.14
    normalized = np.exp(-20.0 * distances / threshold)
    normalized = np.clip(normalized, 0, 1)

    colors = cmap(normalized)[:, :3]  # RGB only, no alpha
    colors = (colors * 255).astype(np.uint8)
    return colors


def load_gt_data(seq_name: str, max_frames: int = None):
    """Load ground truth hand and object data from HO3D."""
    from vggt.utils.gt import load_data

    # Load all frames
    def get_all_fids():
        data = torch.load(f"./ho3d_v3/processed/{seq_name}.pt")
        num_frames = data["hand_pose"].shape[0]
        fids = list(range(num_frames))
        if max_frames is not None:
            fids = fids[:max_frames]
        return np.array(fids)

    import torch
    gt_data = load_data(seq_name, get_all_fids)
    return gt_data


def load_pred_data(result_folder: str, max_frames: int = None):
    """Load predicted hand and object data from results folder."""
    result_path = Path(result_folder)

    # Find all step directories
    step_dirs = sorted(
        (d for d in result_path.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda p: int(p.name),
    )

    if max_frames is not None:
        step_dirs = step_dirs[:max_frames]

    pred_data = []
    for step_dir in step_dirs:
        results_file = step_dir / "results.pkl"
        if not results_file.exists():
            continue
        with open(results_file, "rb") as f:
            data = pickle.load(f)
        pred_data.append({"step": int(step_dir.name), "data": data, "path": step_dir})

    return pred_data


def build_blueprint(num_frames: int) -> rrb.BlueprintLike:
    """Build Rerun blueprint for visualization."""
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera Image", origin="/world/camera"),
            rrb.Spatial3DView(name="3D View", origin="/world"),
            column_shares=[1, 1],
        ),
        rrb.Horizontal(
            rrb.Spatial3DView(name="Hand Distance", origin="/world/hand_distance"),
            rrb.Spatial3DView(name="Object Distance", origin="/world/object_distance"),
            column_shares=[1, 1],
        ),
        row_shares=[2, 1],
    )


def log_camera(
    label: str,
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    image_path: str = None,
    image_plane_distance: float = 0.1,
    axis_length: float = 0.1,
):
    """
    Log camera with pose, intrinsics, and optional image to Rerun.

    Args:
        label: Rerun entity path (e.g., "/world/camera")
        c2w: (4, 4) camera-to-world transformation matrix
        K: (3, 3) camera intrinsic matrix
        width: image width
        height: image height
        image_path: optional path to image file
        image_plane_distance: distance to display image plane in 3D
        axis_length: length of camera axis visualization
    """
    # Log camera transform (camera-to-world)
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    # Log pinhole camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rr.log(
        label,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            image_plane_distance=image_plane_distance,
        ),
        static=False,
    )

    rr.log(
        label,
        rr.Transform3D(
            translation=translation,
            mat3x3=rotation,
        ),
        # rr.components.AxisLength(axis_length),
        static=False,
    )
    # Log image if available
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        rr.log(f"{label}", rr.Image(img_array), static=False)


def visualize_gt_distance(args):
    """Visualize hand-object distance using ground truth data."""
    print(f"Loading GT data for sequence: {args.seq_name}")
    gt_data = load_gt_data(args.seq_name, args.max_frames)

    v3d_hand = gt_data["v3d_c.right"]  # (num_frames, 778, 3)
    v3d_object = gt_data["v3d_c.object"]  # (num_frames, num_obj_verts, 3)
    faces_hand = gt_data["faces.right"]
    faces_object = gt_data["faces.object"]
    o2c = gt_data["o2c"]  # (num_frames, 4, 4)
    K = gt_data["K"]
    is_valid = gt_data["is_valid"]
    fnames = gt_data["fnames"]

    if hasattr(v3d_hand, 'numpy'):
        v3d_hand = v3d_hand.numpy()
    if hasattr(v3d_object, 'numpy'):
        v3d_object = v3d_object.numpy()
    if hasattr(faces_hand, 'numpy'):
        faces_hand = faces_hand.numpy()
    if hasattr(faces_object, 'numpy'):
        faces_object = faces_object.numpy()
    if hasattr(o2c, 'numpy'):
        o2c = o2c.numpy()
    if hasattr(K, 'numpy'):
        K = K.numpy()
    if hasattr(is_valid, 'numpy'):
        is_valid = is_valid.numpy()

    # Seal the hand mesh (close the wrist opening)
    v3d_hand, faces_hand = seal_mano_mesh_np(v3d_hand, faces_hand.astype(np.int64), is_rhand=True)

    num_frames = len(v3d_hand)
    print(f"Loaded {num_frames} frames")

    # Initialize Rerun
    visualizer = Visualizer(f"distance_{args.seq_name}")
    rr.send_blueprint(build_blueprint(num_frames))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # # Load object mesh for static display
    # mesh_path = gt_data.get("mesh_name.object")
    # if mesh_path:
    #     visualizer.log_mesh("/world/object_canonical", mesh_path, static=True)

    print("Visualizing frames...")
    for frame_idx in tqdm(range(num_frames)):
        if not is_valid[frame_idx]:
            continue

        rr.set_time_sequence("frame", frame_idx)

        hand_verts = v3d_hand[frame_idx]
        obj_verts = v3d_object[frame_idx]

        # Skip invalid frames
        if np.any(hand_verts < -100) or np.any(obj_verts < -100):
            continue

        # Compute distances
        hand_to_obj_dist = compute_hand_object_distance(hand_verts, obj_verts)
        obj_to_hand_dist = compute_object_hand_distance(obj_verts, hand_verts)

        # Convert to colors
        hand_colors = np.full((hand_verts.shape[0], 3), 180, dtype=np.uint8)  # Gray color for hand
        obj_colors = distance_to_color(obj_to_hand_dist, args.colormap, args.distance_threshold)

        # Compute vertex normals for better lighting
        hand_normals = compute_vertex_normals(hand_verts, faces_hand.astype(np.int32))
        obj_normals = compute_vertex_normals(obj_verts, faces_object.astype(np.int32))

        # Log hand mesh with distance colors and normals
        rr.log(
            "/world/hand_distance/mesh",
            rr.Mesh3D(
                vertex_positions=hand_verts,
                triangle_indices=faces_hand.astype(np.int32),
                vertex_colors=hand_colors,
                vertex_normals=hand_normals,
            ),
            static=False,
        )

        # Log object mesh with distance colors and normals
        rr.log(
            "/world/object_distance/mesh",
            rr.Mesh3D(
                vertex_positions=obj_verts,
                triangle_indices=faces_object.astype(np.int32),
                vertex_colors=obj_colors,
                vertex_normals=obj_normals,
            ),
            static=False,
        )

        # Log original meshes (gray)
        # rr.log(
        #     "/world/hand/mesh",
        #     rr.Mesh3D(
        #         vertex_positions=hand_verts,
        #         triangle_indices=faces_hand.astype(np.int32),
        #         mesh_material=add_material([200, 200, 200, 255]),
        #     ),
        #     static=False,
        # )

        # rr.log(
        #     "/world/object/mesh",
        #     rr.Mesh3D(
        #         vertex_positions=obj_verts,
        #         triangle_indices=faces_object.astype(np.int32),
        #         mesh_material=add_material([100, 150, 200, 255]),
        #     ),
        #     static=False,
        # )

        # # Log image if available
        # if frame_idx < len(fnames):
        #     fname = fnames[frame_idx]
        #     if isinstance(fname, (str, Path)) and os.path.exists(fname):
        #         visualizer.log_image("/camera/image", str(fname), static=False)

        # Log distance statistics as text
        min_hand_dist = np.min(hand_to_obj_dist)
        mean_hand_dist = np.mean(hand_to_obj_dist)
        contact_ratio = np.mean(hand_to_obj_dist < 0.01)  # vertices within 1cm

        rr.log(
            "/stats/distance",
            rr.TextLog(
                f"Frame {frame_idx}: min_dist={min_hand_dist:.4f}m, "
                f"mean_dist={mean_hand_dist:.4f}m, contact_ratio={contact_ratio:.2%}"
            ),
            static=False,
        )

        # Log camera with image and intrinsics
        # o2c is object-to-camera, we need camera-to-object (c2w in object frame)
        o2c_mat = o2c[frame_idx]
        c2w = np.eye(4)
        # c2w[:3, :3] = o2c_mat[:3, :3].T  # R^T
        # c2w[:3, 3] = -o2c_mat[:3, :3].T @ o2c_mat[:3, 3]  # -R^T * t

        # Get image path and dimensions
        image_path = None
        if frame_idx < len(fnames):
            fname = fnames[frame_idx]
            if isinstance(fname, (str, Path)) and os.path.exists(str(fname)):
                image_path = str(fname)

        # Get image dimensions
        if image_path:
            with Image.open(image_path) as img:
                width, height = img.size
        else:
            width, height = 640, 480  # default

        # Reshape K if needed
        K_mat = K.reshape(3, 3) if K.ndim == 1 else K

        log_camera(
            label="/world/camera",
            c2w=c2w,
            K=K_mat,
            width=width,
            height=height,
            image_path=image_path,
            image_plane_distance=1.0,
        )

    if args.rrd_output_path:
        rr.save(args.rrd_output_path)
        print(f"Saved to {args.rrd_output_path}")


def visualize_pred_distance(args):
    """Visualize hand-object distance using predicted data."""
    print(f"Loading predicted data from: {args.result_folder}")
    pred_data = load_pred_data(args.result_folder, args.max_frames)

    if not pred_data:
        print("No prediction data found!")
        return

    # Load hand data
    hand_base_dir = Path(args.result_folder).parent
    from viewer.viewer_step import HandDataProvider, ObjDataProvider
    hand_provider = HandDataProvider(hand_base_dir)
    obj_provider = ObjDataProvider(Path(args.result_folder))

    if not hand_provider.has_hand:
        print("No hand data found in results!")
        return

    # Load object mesh
    gen3d_dir = hand_base_dir / "gen_3d"
    mesh_path = gen3d_dir / "white_mesh_remesh.obj"
    if not mesh_path.exists():
        print(f"Object mesh not found: {mesh_path}")
        return

    obj_mesh = trimesh.load(mesh_path, force="mesh")
    obj_faces = np.array(obj_mesh.faces)

    num_frames = len(pred_data)
    print(f"Loaded {num_frames} frames")

    # Initialize Rerun
    seq_name = hand_base_dir.name
    visualizer = Visualizer(f"pred_distance_{seq_name}")
    rr.send_blueprint(build_blueprint(num_frames))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    print("Visualizing frames...")
    for step_info in tqdm(pred_data):
        step_idx = step_info["step"]
        data = step_info["data"]

        rr.set_time_sequence("frame", step_idx)

        extr = data.get("extrinsics")
        intr = data.get("intrinsics")
        aligned_pose = data.get("aligned_pose")

        if extr is None or step_idx >= len(extr):
            continue

        # Get hand vertices in world space
        hand_verts_cam = hand_provider.get_hand_verts_cam("rot", step_idx)
        hand_faces = hand_provider.get_hand_faces("rot")

        if hand_verts_cam is None or hand_faces is None:
            # Try other modes
            for mode in ["trans", "intrinsic"]:
                hand_verts_cam = hand_provider.get_hand_verts_cam(mode, step_idx)
                hand_faces = hand_provider.get_hand_faces(mode)
                if hand_verts_cam is not None:
                    break

        if hand_verts_cam is None:
            continue

        hand_verts_cam = np.asarray(hand_verts_cam)
        hand_faces = np.asarray(hand_faces, dtype=np.int64)

        # Seal the hand mesh (close the wrist opening)
        hand_verts_cam_sealed, hand_faces = seal_mano_mesh_np(
            hand_verts_cam[None], hand_faces, is_rhand=True
        )
        hand_verts_cam = hand_verts_cam_sealed[0]

        # Transform hand to world space
        w2c = np.eye(4)
        w2c[:3] = extr[step_idx]
        c2w = np.linalg.inv(w2c)
        hand_verts_world = (c2w[:3, :3] @ hand_verts_cam.T + c2w[:3, 3:4]).T

        # Transform object mesh using aligned pose
        obj_verts = np.array(obj_mesh.vertices)
        if aligned_pose:
            R = np.asarray(aligned_pose.get("rotation", np.eye(3)))
            t = np.asarray(aligned_pose.get("translation", np.zeros(3)))
            s = float(aligned_pose.get("scale", 1.0))
            obj_verts = (obj_verts @ R.T) * s + t

        # Compute distances
        hand_to_obj_dist = compute_hand_object_distance(hand_verts_world, obj_verts)
        obj_to_hand_dist = compute_object_hand_distance(obj_verts, hand_verts_world)

        # Convert to colors
        hand_colors = np.full((hand_verts_world.shape[0], 3), 180, dtype=np.uint8)  # Gray color for hand
        obj_colors = distance_to_color(obj_to_hand_dist, args.colormap, args.distance_threshold)

        # Compute vertex normals for better lighting
        hand_normals = compute_vertex_normals(hand_verts_world, hand_faces.astype(np.int32))
        obj_normals = compute_vertex_normals(obj_verts, obj_faces.astype(np.int32))

        # Log hand mesh with distance colors and normals
        rr.log(
            "/world/hand_distance/mesh",
            rr.Mesh3D(
                vertex_positions=hand_verts_world,
                triangle_indices=hand_faces.astype(np.int32),
                vertex_colors=hand_colors,
                vertex_normals=hand_normals,
            ),
            static=False,
        )

        # Log object mesh with distance colors and normals
        rr.log(
            "/world/object_distance/mesh",
            rr.Mesh3D(
                vertex_positions=obj_verts,
                triangle_indices=obj_faces.astype(np.int32),
                vertex_colors=obj_colors,
                vertex_normals=obj_normals,
            ),
            static=False,
        )

        # Log distance statistics
        min_hand_dist = np.min(hand_to_obj_dist)
        mean_hand_dist = np.mean(hand_to_obj_dist)
        contact_ratio = np.mean(hand_to_obj_dist < 0.01)

        rr.log(
            "/stats/distance",
            rr.TextLog(
                f"Frame {step_idx}: min_dist={min_hand_dist:.4f}m, "
                f"mean_dist={mean_hand_dist:.4f}m, contact_ratio={contact_ratio:.2%}"
            ),
            static=False,
        )

        # Log camera with image and intrinsics
        # Get image path
        image_path = None
        if step_idx < len(obj_provider.origin_images):
            image_path = str(obj_provider.origin_images[step_idx])
        elif step_idx < len(obj_provider.images):
            image_path = str(obj_provider.images[step_idx])

        # Get image dimensions
        if image_path and os.path.exists(image_path):
            with Image.open(image_path) as img:
                width, height = img.size
        else:
            width, height = 640, 480

        # Get intrinsics
        if intr is not None and step_idx < len(intr):
            K_mat = np.asarray(intr[step_idx])
            if K_mat.ndim == 1:
                K_mat = K_mat.reshape(3, 3)
        else:
            # Default intrinsics
            K_mat = np.array([
                [500, 0, width / 2],
                [0, 500, height / 2],
                [0, 0, 1]
            ])

        log_camera(
            label="/world/camera",
            c2w=c2w,
            K=K_mat,
            width=width,
            height=height,
            image_path=image_path,
            image_plane_distance=0.15,
        )

    if args.rrd_output_path:
        rr.save(args.rrd_output_path)
        print(f"Saved to {args.rrd_output_path}")


def main():
    args = parse_args()
    print(f"Arguments: {args}")

    if args.use_pred:
        if args.result_folder is None:
            args.result_folder = f"output/{args.seq_name}/results"
        visualize_pred_distance(args)
    else:
        visualize_gt_distance(args)


if __name__ == "__main__":
    main()
