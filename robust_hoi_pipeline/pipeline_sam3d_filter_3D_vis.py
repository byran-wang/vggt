import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import depth2xyzmap, get_depth, load_filtered_depth
from utils_simba.rerun import load_mesh_as_trimesh, get_vertex_colors, stamp_frame_text, log_camera_frame


def init_rerun(app_name):
    """Initialize rerun with a standard 3D+camera blueprint."""
    rr.init(app_name, spawn=True)
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera", origin="world/camera"),
            ),
            column_shares=[2, 1],
        ),
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)


def load_camera_pose(camera_json_path):
    """Load camera.json and return K, c2o (rigid, unscaled), and scale.

    Returns (K, c2o, scale) or None if file doesn't exist.
    """
    if not camera_json_path.exists():
        return None
    with open(camera_json_path, "r") as f:
        camera = json.load(f)
    K = np.array(camera["K"], dtype=np.float64)           # (3, 3)
    o2c = np.array(camera["blw2cvc"], dtype=np.float64)   # (4, 4)
    R = o2c[:3, :3]
    scale = np.linalg.norm(R, axis=0)[0]  # per-column scale
    o2c_rigid = o2c.copy()
    o2c_rigid[:3, :3] = R / scale
    c2o = np.linalg.inv(o2c_rigid)
    c2o[:3, 3] = c2o[:3, 3] / scale
    return K, c2o, scale


def log_mesh(frame_dir, entity="world/sam3d_mesh", static=False):
    """Load and log a mesh from frame_dir. Returns the mesh or None."""
    mesh = load_mesh_as_trimesh(frame_dir)
    if mesh is not None:
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.uint32)
        vertex_colors = get_vertex_colors(mesh)
        mesh_kwargs = dict(vertex_positions=verts, triangle_indices=faces)
        if vertex_colors is not None:
            mesh_kwargs["vertex_colors"] = vertex_colors
        rr.log(entity, rr.Mesh3D(**mesh_kwargs), static=static)
    return mesh


def log_image(rgb_dir, fid, frame_idx, K, c2o, jpeg_quality=85):
    """Load and log camera image. Returns True if image was found."""
    img_path = rgb_dir / f"{fid}.jpg"
    if not img_path.exists():
        img_path = rgb_dir / f"{fid}.png"
    if not img_path.exists():
        return False
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
    img = stamp_frame_text(img, f"Frame {frame_idx:04d}")
    log_camera_frame(
        "world/camera", K, c2o, img,
        image_plane_distance=10.0,
        jpeg_quality=jpeg_quality,
        static=False,
    )
    return True


def log_depth_points(dataset_dir, scene_name, fid, c2o, scale):
    """Back-project filtered depth to 3D in SAM3D object space and log. Returns True if logged."""
    preprocess_dir = Path(f"{dataset_dir}/{scene_name}/pipeline_preprocess")
    depth_path = preprocess_dir / "../depth" / f"{fid}.png"
    mask_path = preprocess_dir / "mask_obj" / f"{fid}.png"
    meta_path = preprocess_dir / "meta" / f"{fid}.pkl"

    if not all(p.exists() for p in [depth_path, mask_path, meta_path]):
        return False

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    K_meta = np.array(meta["intrinsics"], dtype=np.float64)
    depth = load_filtered_depth(str(depth_path))
    depth /= scale
    mask_obj = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    xyz_cam = depth2xyzmap(depth, K_meta)  # (H, W, 3)
    valid = (mask_obj > 0) & (depth > 0.01)
    pts_cam = xyz_cam[valid]  # (N, 3)

    if len(pts_cam) == 0:
        return False

    pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_obj = (c2o @ pts_cam_h.T).T[:, :3]
    rr.log("world/depth_points", rr.Points3D(pts_obj, radii=0.001), static=False)
    return True


def main(args):
    sam3d_dir = Path(f"{args.dataset_dir}/{args.scene_name}/SAM3D")

    # Load frame list after 3D filter
    frame_list_file = sam3d_dir / "frame_list_after_3d_filtered.txt"
    if not frame_list_file.exists():
        print(f"Error: {frame_list_file} not found. Run ho3d_obj_SAM3D_filter_3D first.")
        return
    with open(frame_list_file, "r") as f:
        frame_indices = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(frame_indices)} frames from {frame_list_file}")

    init_rerun("sam3d_filter_3D_vis")

    rgb_dir = Path(f"{args.dataset_dir}/{args.scene_name}/rgb")

    for seq_i, frame_idx in enumerate(frame_indices):
        fid = f"{frame_idx:04d}"
        frame_dir = sam3d_dir / fid

        cam = load_camera_pose(frame_dir / "camera.json")
        if cam is None:
            print(f"  Frame {fid}: camera.json not found, skipping")
            continue
        K, c2o, scale = cam

        rr.set_time_sequence("frame", seq_i)

        mesh = log_mesh(frame_dir)
        has_img = log_image(rgb_dir, fid, frame_idx, K, c2o, args.jpeg_quality)
        has_pts = log_depth_points(args.dataset_dir, args.scene_name, fid, c2o, scale)

        print(f"  Frame {fid}: mesh={'yes' if mesh else 'no'}, image={'yes' if has_img else 'no'}, depth_pts={'yes' if has_pts else 'no'}")

    print(f"Done. Visualized {len(frame_indices)} frames in rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM3D 3D-filtered frames in Rerun")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--jpeg_quality", type=int, default=85)

    args = parser.parse_args()
    main(args)
