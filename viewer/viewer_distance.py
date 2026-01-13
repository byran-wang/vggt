"""
Visualize hand-object distance for each frame using Rerun.
Similar to ARCTIC InterField visualization.

Usage:
    python viewer/viewer_distance.py --seq_name MC1
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
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
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to visualize")
    parser.add_argument("--frame_interval", type=int, default=3, help="Frame interval (e.g., 5 to show every 5th frame)")
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


def build_blueprint(num_frames: int) -> rrb.BlueprintLike:
    """Build Rerun blueprint for visualization."""
    white_bg = [255, 255, 255]
    return rrb.Horizontal(
        rrb.Spatial2DView(name="Camera Image", origin="/world/camera"),
        rrb.Spatial3DView(name="3D View", origin="/world/scene", background=white_bg),
        rrb.Spatial3DView(name="Hand Distance", origin="/world/hand_distance", background=white_bg),
        rrb.Spatial3DView(name="Object Distance", origin="/world/object_distance", background=white_bg),
        column_shares=[1, 1, 1, 1],
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
    v3d_hand_flat = gt_data["v3d_flat.right"]  # (num_frames, 778, 3)
    v3d_object = gt_data["v3d_c.object"]  # (num_frames, num_obj_verts, 3)
    v3d_object_can = gt_data["v3d_can.object"]  # (num_frames, num_obj_verts, 3) canonical object
    faces_hand = gt_data["faces.right"]
    faces_object = gt_data["faces.object"]
    o2c = gt_data["o2c"]  # (num_frames, 4, 4)
    K = gt_data["K"]
    is_valid = gt_data["is_valid"]
    fnames = gt_data["fnames"]

    if hasattr(v3d_hand, 'numpy'):
        v3d_hand = v3d_hand.numpy()
    if hasattr(v3d_hand_flat, 'numpy'):
        v3d_hand_flat = v3d_hand_flat.numpy()        
    if hasattr(v3d_object, 'numpy'):
        v3d_object = v3d_object.numpy()
    if hasattr(v3d_object_can, 'numpy'):
        v3d_object_can = v3d_object_can.numpy()
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
    v3d_hand_flat, faces_hand_flat = seal_mano_mesh_np(v3d_hand_flat, faces_hand.astype(np.int64), is_rhand=True)

    num_frames = len(v3d_hand)
    print(f"Loaded {num_frames} frames")

    # Initialize Rerun
    visualizer = Visualizer(f"distance_{args.seq_name}")
    rr.send_blueprint(build_blueprint(num_frames))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    print("Visualizing frames...")
    vis_frame_idx = 0
    for frame_idx in tqdm(range(0, num_frames, args.frame_interval)):
        if not is_valid[frame_idx]:
            continue

        rr.set_time_sequence("frame", vis_frame_idx)
        vis_frame_idx += 1

        hand_verts = v3d_hand[frame_idx]
        # TODO: hard code due to beta in different frame
        hand_verts_flat = v3d_hand_flat[0]
        obj_verts = v3d_object[frame_idx]
        obj_verts_can = v3d_object_can[frame_idx]  # Canonical object vertices (identity pose)

        # Skip invalid frames
        if np.any(hand_verts < -100) or np.any(obj_verts < -100):
            continue

        # Compute distances
        hand_to_obj_dist = compute_hand_object_distance(hand_verts, obj_verts)
        obj_to_hand_dist = compute_object_hand_distance(obj_verts, hand_verts)

        # Convert distances to colors
        hand_dist_colors = distance_to_color(hand_to_obj_dist, args.colormap, args.distance_threshold)
        obj_dist_colors = distance_to_color(obj_to_hand_dist, args.colormap, args.distance_threshold)

        # Gray colors for non-distance views
        hand_purple = np.full((hand_verts.shape[0], 3), [200, 180, 220], dtype=np.uint8)  # Light purple
        obj_blue = np.full((obj_verts.shape[0], 3), [180, 200, 230], dtype=np.uint8)  # Light blue

        # Compute vertex normals for better lighting
        hand_normals = compute_vertex_normals(hand_verts, faces_hand.astype(np.int32))
        hand_normals_flat = compute_vertex_normals(hand_verts_flat, faces_hand.astype(np.int32))
        obj_normals = compute_vertex_normals(obj_verts, faces_object.astype(np.int32))
        obj_normals_can = compute_vertex_normals(obj_verts_can, faces_object.astype(np.int32))

        # === View 2: 3D Scene (hand, object, camera - no distance colors) ===
        rr.log(
            "/world/scene/hand",
            rr.Mesh3D(
                vertex_positions=hand_verts,
                triangle_indices=faces_hand.astype(np.int32),
                vertex_colors=hand_purple,
                vertex_normals=hand_normals,
            ),
            static=False,
        )
        rr.log(
            "/world/scene/object",
            rr.Mesh3D(
                vertex_positions=obj_verts,
                triangle_indices=faces_object.astype(np.int32),
                vertex_colors=obj_blue,
                vertex_normals=obj_normals,
            ),
            static=False,
        )

        # === View 3: Hand Distance (hand in canonical space with distance colors) ===
        rr.log(
            "/world/hand_distance/hand",
            rr.Mesh3D(
                vertex_positions=hand_verts_flat,
                triangle_indices=faces_hand_flat.astype(np.int32),
                vertex_colors=hand_dist_colors,
                vertex_normals=hand_normals_flat,
            ),
            static=False,
        )
        # Also show object in canonical space (gray) for reference
        # rr.log(
        #     "/world/hand_distance/object",
        #     rr.Mesh3D(
        #         vertex_positions=obj_verts_can,
        #         triangle_indices=faces_object.astype(np.int32),
        #         vertex_colors=obj_gray,
        #         vertex_normals=obj_normals_can,
        #     ),
        #     static=False,
        # )

        # === View 4: Object Distance (object in canonical space with distance colors) ===
        rr.log(
            "/world/object_distance/object",
            rr.Mesh3D(
                vertex_positions=obj_verts_can,
                triangle_indices=faces_object.astype(np.int32),
                vertex_colors=obj_dist_colors,
                vertex_normals=obj_normals_can,
            ),
            static=False,
        )
        # Also show hand in canonical space (gray) for reference
        # rr.log(
        #     "/world/object_distance/hand",
        #     rr.Mesh3D(
        #         vertex_positions=hand_verts_can,
        #         triangle_indices=faces_hand.astype(np.int32),
        #         vertex_colors=hand_gray,
        #         vertex_normals=hand_normals_can,
        #     ),
        #     static=False,
        # )

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
        c2w = np.eye(4)

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

        # Also log camera in scene view with image
        log_camera(
            label="/world/scene/camera",
            c2w=c2w,
            K=K_mat,
            width=width,
            height=height,
            image_path=image_path,  # Show image in 3D scene view
            image_plane_distance=1.0,
        )

    if args.rrd_output_path:
        rr.save(args.rrd_output_path)
        print(f"Saved to {args.rrd_output_path}")


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    visualize_gt_distance(args)


if __name__ == "__main__":
    main()
