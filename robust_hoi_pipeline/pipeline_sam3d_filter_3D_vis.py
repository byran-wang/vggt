import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import rerun as rr


def load_mesh(mesh_dir: Path):
    """Load mesh from SAM3D frame directory, trying scene.glb then mesh.obj."""
    candidates = [mesh_dir / "scene.glb", mesh_dir / "mesh.obj"]
    for p in candidates:
        if p.exists():
            loaded = trimesh.load(str(p), process=False)
            if isinstance(loaded, trimesh.Scene):
                meshes = []
                for node_name in loaded.graph.nodes_geometry:
                    transform, geom_name = loaded.graph[node_name]
                    geom = loaded.geometry[geom_name].copy()
                    geom.apply_transform(transform)
                    meshes.append(geom)
                if meshes:
                    return trimesh.util.concatenate(meshes)
                return None
            if isinstance(loaded, trimesh.Trimesh):
                return loaded
    return None


def get_vertex_colors(mesh):
    """Extract RGB vertex colors from a trimesh object, or None."""
    if mesh is None:
        return None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices) and vc.shape[1] >= 3:
            return vc[:, :3].astype(np.uint8)
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "to_color") and callable(mesh.visual.to_color):
        vc = np.asarray(mesh.visual.to_color().vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices) and vc.shape[1] >= 3:
            return vc[:, :3].astype(np.uint8)
    return None


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
        mesh = load_mesh(frame_dir)
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

        # Use a single fixed entity so each time step replaces the previous
        entity = "world/camera"

        # Log pinhole camera and image
        img_path = rgb_dir / f"{fid}.jpg"
        if not img_path.exists():
            img_path = rgb_dir / f"{fid}.png"
        if img_path.exists():
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((10, 10), f"Frame {frame_idx:04d}", fill=(255, 255, 0), font=font)
            img = np.array(pil_img)
            H, W = img.shape[:2]
            rr.log(
                entity,
                rr.Pinhole(
                    resolution=[W, H],
                    focal_length=[float(K[0, 0]), float(K[1, 1])],
                    principal_point=[float(K[0, 2]), float(K[1, 2])],
                    image_plane_distance=3.0,
                ),
                static=False,
            )
            rr.log(entity, rr.Transform3D(
                translation=c2o[:3, 3].astype(np.float32),
                mat3x3=c2o[:3, :3].astype(np.float32),
            ), static=False)
            rr.log(entity, rr.Image(img).compress(jpeg_quality=args.jpeg_quality), static=False)

        print(f"  Frame {fid}: logged mesh={'yes' if mesh else 'no'}, image={'yes' if img_path.exists() else 'no'}")

    print(f"Done. Visualized {len(frame_indices)} frames in rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM3D 3D-filtered frames in Rerun")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--jpeg_quality", type=int, default=85)

    args = parser.parse_args()
    main(args)
