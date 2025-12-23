import argparse
from cProfile import label
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import rerun as rr  # @manual
import rerun.blueprint as rrb
from PIL import Image
import trimesh
import sys
sys.path.append("../third_party/utils_simba")
from utils_simba.rerun import Visualizer
from utils_simba.depth import get_depth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Path to saved results (output/.../results or a specific step directory).",
    )
    parser.add_argument("--jpeg_quality", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rrd_output_path", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--image_plane_distance", type=float, default=0.05, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--vis_only_register",
        action="store_true",
        help="Only visualize frames marked as registered.",
    )
    return parser.parse_args()


class StepDataProvider:
    """Lightweight loader for demo_colmap.py outputs."""

    def __init__(self, result_dir: Path):
        self.result_dir = Path(result_dir).resolve()
        self.base_dir = (
            self.result_dir
            if (self.result_dir / "images").exists()
            else self.result_dir.parent
        )
        if not (self.base_dir / "images").exists():
            raise RuntimeError(
                f"Could not find preprocessed inputs under {self.result_dir}"
            )

        self.images = sorted((self.base_dir / "images").glob("*.png"))
        self.masks = sorted((self.base_dir / "masks").glob("*.png"))
        self.depths = sorted((self.base_dir / "depth_prior").glob("*.png"))

        # Discover per-step folders containing results.pkl
        step_dirs = sorted(
            (d for d in self.base_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda p: p.stat().st_mtime,
        )

        self.steps = []
        for step_idx, step_dir in enumerate(step_dirs):
            results_file = step_dir / "results.pkl"
            if not results_file.exists():
                continue
            with open(results_file, "rb") as f:
                data = pickle.load(f)
            gen_3d_mesh_aligned = step_dir / "white_mesh_remesh_aligned.obj"
            self.steps.append({"index": step_idx, "path": step_dir, "data": data, "gen_3d_mesh_aligned": gen_3d_mesh_aligned})

        gen3d_dir = self.base_dir / "gen_3d"
        self.mesh_path = gen3d_dir / "white_mesh_remesh.obj"
        self.cond_image = gen3d_dir / "image.png"
        self.cond_depth = gen3d_dir / "depth.png"
        self.camera_json = gen3d_dir / "camera.json"


def log_mesh_with_pose(label: str, mesh_path: Path, pose: Optional[dict]):
    if not mesh_path.exists():
        return
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    vertices = mesh.vertices
    faces = mesh.faces

    if pose:
        R = np.asarray(pose.get("rotation"))
        t = np.asarray(pose.get("translation"))
        s = float(pose.get("scale", 1.0))
        vertices = (vertices @ R.T) * s + t

    rr.log(
        label,
        rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces),
    )


def build_blueprint(num_images: int) -> rrb.BlueprintLike:
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(name="world", origin="/"),
            rrb.Spatial2DView(name="current_camera", origin="/camera/image"),
        ),
        rrb.Horizontal(
            rrb.Grid(
                *[
                    rrb.Spatial2DView(name=f"image_{i}", origin=f"/camera/image_{i}")
                    for i in range(num_images)
                ],
                grid_columns=20,
            ),
        ),
        row_shares=[3, 3],
    )


def main(args):
    provider = StepDataProvider(Path(args.result_folder))
    vis_name = provider.base_dir.parents[0].name
    visualizer = Visualizer(vis_name, jpeg_quality=args.jpeg_quality)

    rr.send_blueprint(build_blueprint(100))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    gen_3d = trimesh.load_mesh(provider.mesh_path)

    for step in provider.steps:
        step_idx = step["index"]
        data = step["data"]
        visualizer.set_time_sequence(step_idx)

        intr = data.get("intrinsics")
        extr = data.get("extrinsics")
        registered = data.get("registered")
        pred_tracks = data.get("pred_tracks")
        track_mask = data.get("track_mask")
        aligned_pose = data.get("aligned_pose")
        gen_3d_mesh_aligned_path = step["gen_3d_mesh_aligned"]
        points_3d = data.get("points_3d")
        points_rgb = data.get("points_rgb")
        points_conf_color = data.get("points_conf_color")
        keyframe_flags = data.get("keyframe")
        

        if intr is None or extr is None:
            continue

        intr = np.asarray(intr)
        extr = np.asarray(extr)

        for cam_idx in range(len(extr)):
            if args.vis_only_register and registered is not None:
                reg_flags = np.asarray(registered).astype(bool)
                if cam_idx >= len(reg_flags) or not reg_flags[cam_idx]:
                    continue

            visualizer.log_image(f"camera/image_{cam_idx}", str(provider.images[cam_idx]), static=False)
            with Image.open(provider.images[cam_idx]) as im:
                w, h = im.size
            visualizer.log_calibration(
                f"camera/image_{cam_idx}",
                resolution=[w, h],
                intrins=intr[cam_idx],
                image_plane_distance=args.image_plane_distance,
                static=False,
            )
            w2c = np.eye(4)
            w2c[:3] = extr[cam_idx]
            c2w = np.linalg.inv(w2c)
            visualizer.log_cam_pose(f"camera/image_{cam_idx}", c2w, static=False)

            is_keyframe = (
                keyframe_flags is not None
                and cam_idx < len(keyframe_flags)
                and bool(np.asarray(keyframe_flags)[cam_idx])
            )

            if is_keyframe:
                rr.log(
                    f"camera/image_{cam_idx}/keyframe_border",
                    rr.Boxes2D(
                        centers=[[w / 2, h / 2]],
                        sizes=[[w, h]],
                        colors=[[0, 255, 0]],
                        radii=2.0,
                    ),
                    static=False,
                )

            if pred_tracks is not None and track_mask is not None:
                tracks = np.asarray(pred_tracks)[cam_idx]
                mask = np.asarray(track_mask)[cam_idx].astype(bool)
                if tracks.shape[0] == mask.shape[0]:
                    if is_keyframe:
                        track_count = int(mask.sum())
                        rr.log(
                            f"camera/image_{cam_idx}/track_count",
                            rr.TextLog(f"track_count: {track_count}"),
                            static=False,
                        )
                    rr.log(
                        f"camera/image_{cam_idx}/keypoints",
                        rr.Points2D(tracks[mask], colors=[34, 138, 167]),
                        static=False,
                    )

            # log the current camera view
            cam_idx = step_idx
            visualizer.log_image("camera/image", str(provider.images[cam_idx]), static=False)
            visualizer.log_calibration(
                "camera/image",
                resolution=[w, h],
                intrins=intr[cam_idx],
                image_plane_distance=1,
                static=False,
            )
            w2c = np.eye(4)
            w2c[:3] = extr[cam_idx]
            c2w = np.linalg.inv(w2c)
            visualizer.log_cam_pose("camera/image", c2w, static=False)
            tracks = np.asarray(pred_tracks)[cam_idx]
            rr.log(
                f"camera/image/keypoints",
                rr.Points2D(tracks[mask], colors=[34, 138, 167]),
                static=False,
            )
            
        visualizer.log_mesh("aligned_mesh", gen_3d_mesh_aligned_path, static=False)
        # Log 3D points with color if available
        if points_3d is not None:
            pts = np.asarray(points_3d)
            visualizer.log_points("points_rgb", pts, colors=points_rgb, static=False)
            visualizer.log_points("points_conf", pts, colors=points_conf_color, static=False)

    if args.rrd_output_path:
        rr.save(args.rrd_output_path)


if __name__ == "__main__":
    args = parse_args()
    print(f"args provided: {args}")
    main(args)
