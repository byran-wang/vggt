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

from utils_simba.depth import depth2xyzmap, get_depth
from utils_simba.rerun import load_mesh_as_trimesh, get_vertex_colors, stamp_frame_text, log_camera_frame

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

    # Init rerun
    rr.init("sam3d_filter_3D_vis", spawn=True)
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

    rgb_dir = Path(f"{args.dataset_dir}/{args.scene_name}/rgb")

    for seq_i, frame_idx in enumerate(frame_indices):
        fid = f"{frame_idx:04d}"
        frame_dir = sam3d_dir / fid

        # Load camera.json
        camera_json = frame_dir / "camera.json"
        if not camera_json.exists():
            print(f"  Frame {fid}: camera.json not found, skipping")
            continue

        with open(camera_json, "r") as f:
            camera = json.load(f)

        K = np.array(camera["K"], dtype=np.float64)           # (3, 3)
        o2c = np.array(camera["blw2cvc"], dtype=np.float64)   # (4, 4)
        # Remove scale from rotation to get a rigid transform
        R = o2c[:3, :3]
        scale = np.linalg.norm(R, axis=0)  # per-column scale
        o2c_rigid = o2c.copy()
        o2c_rigid[:3, :3] = R / scale
        c2o = np.linalg.inv(o2c_rigid)                        # camera-to-object (rigid)
        c2o[:3, 3] = c2o[:3, 3] / scale
        rr.set_time_sequence("frame", seq_i)

        # Log mesh in SAM3D (object) space
        mesh = load_mesh_as_trimesh(frame_dir)
        if mesh is not None:
            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)
            vertex_colors = get_vertex_colors(mesh)
            mesh_kwargs = dict(vertex_positions=verts, triangle_indices=faces)
            if vertex_colors is not None:
                mesh_kwargs["vertex_colors"] = vertex_colors
            rr.log("world/sam3d_mesh", rr.Mesh3D(**mesh_kwargs), static=False)
        else:
            print(f"  Frame {fid}: no mesh found, skipping mesh log")

        # Log pinhole camera and image
        img_path = rgb_dir / f"{fid}.jpg"
        if not img_path.exists():
            img_path = rgb_dir / f"{fid}.png"
        if img_path.exists():
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
            img = stamp_frame_text(img, f"Frame {frame_idx:04d}")
            log_camera_frame(
                "world/camera", K, c2o, img,
                image_plane_distance=3.0,
                jpeg_quality=args.jpeg_quality,
                static=False,
            )

        # Back-project filtered depth to 3D points in SAM3D object space
        preprocess_dir = Path(f"{args.dataset_dir}/{args.scene_name}/pipeline_preprocess")
        depth_path = preprocess_dir / "depth_filtered" / f"{fid}.png"
        mask_path = preprocess_dir / "mask_obj" / f"{fid}.png"
        meta_path = preprocess_dir / "meta" / f"{fid}.pkl"

        has_pts = False
        if depth_path.exists() and mask_path.exists() and meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            K_meta = np.array(meta["intrinsics"], dtype=np.float64)
            depth = get_depth(str(depth_path))
            mask_obj = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            xyz_cam = depth2xyzmap(depth, K_meta)  # (H, W, 3)
            valid = (mask_obj > 0) & (depth > 0.01)
            pts_cam = xyz_cam[valid]  # (N, 3)

            if len(pts_cam) > 0:
                # Transform to SAM3D object space
                pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
                pts_obj = (c2o @ pts_cam_h.T).T[:, :3]
                rr.log("world/depth_points", rr.Points3D(pts_obj, radii=0.001), static=False)
                has_pts = True

        print(f"  Frame {fid}: mesh={'yes' if mesh else 'no'}, image={'yes' if img_path.exists() else 'no'}, depth_pts={'yes' if has_pts else 'no'}")

    print(f"Done. Visualized {len(frame_indices)} frames in rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM3D 3D-filtered frames in Rerun")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--jpeg_quality", type=int, default=85)

    args = parser.parse_args()
    main(args)
