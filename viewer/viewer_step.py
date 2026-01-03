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
# sys.path.append("third_party/utils_simba")
_CODE_DIR = Path(__file__).resolve().parents[1] / "third_party/utils_simba"
if _CODE_DIR.is_dir():
    sys.path = [str(_CODE_DIR)] + sys.path
from utils_simba.rerun import Visualizer, add_material
from utils_simba.depth import get_depth

_CODE_DIR = Path(__file__).resolve().parents[1]
if _CODE_DIR.is_dir():
    sys.path = [str(_CODE_DIR)] + sys.path
try:
    from common.body_models import seal_mano_mesh_np  # type: ignore
except Exception:
    seal_mano_mesh_np = None

LIGHT_GRAY = [200, 200, 200, 255]
GREEN = [0, 255, 0, 255]
LIGHT_RED = [255, 128, 128, 255]


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
    parser.add_argument(
        "--num_frames",
        type=int,
        help="Number of frames to visualize in the grid view.",
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

    def get_reproj_error_vis_path(self, step_idx: int) -> Path:
        step_dir = self.base_dir / f"{step_idx:04d}"
        vis_path = step_dir / "reproj_error.png"
        return vis_path
    
class HandDataProvider:
    def __init__(self, base_dir: Path):
        self.base_dir =base_dir
        self.hand_fit_intrinsic = self._load_fit("intrinsic")
        self.hand_fit_trans = self._load_fit("trans")
        self.hand_fit_rot = self._load_fit("rot")
        self.hand_fit_pose = self._load_fit("pose")
        self.hand_fit_all = self._load_fit("all")

    def _load_fit(self, suffix: str):
        for prefix in ("hand_fit", "hold_fit"):
            path = self.base_dir / f"{prefix}.aligned_h_{suffix}.npy"
            if path.exists():
                try:
                    arr = np.load(path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                        return arr.item()
                    return arr
                except Exception as e:
                    print(f"[HandDataProvider] Failed to load {path}: {e}")
        return None

    def _get_fit(self, mode: str):
        return getattr(self, f"hand_{mode}", None)

    def _extract_from_fit(self, fit, key: str, idx: Optional[int] = None):
        if fit is None:
            return None
        if isinstance(fit, np.ndarray) and fit.dtype == object and fit.size == 1:
            try:
                fit = fit.item()
            except Exception:
                pass
        if isinstance(fit, dict):
            for hand_key in ("right", "rhand", "hand"):
                sub = fit.get(hand_key)
                if isinstance(sub, dict) and key in sub:
                    val = sub[key]
                    break
            else:
                val = fit.get(key)
        elif hasattr(fit, key):
            val = getattr(fit, key)
        else:
            val = None
        if val is None:
            return None
        if idx is None:
            return val
        if isinstance(val, np.ndarray) and val.shape[0] > idx:
            return val[idx]
        if isinstance(val, list) and len(val) > idx:
            return val[idx]
        return None

    def get_hand_faces(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "f3d")

    def get_hand_verts_cam(self, mode: str, i: int):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "v3d_cam", i)

    @property
    def has_hand(self) -> bool:
        return any(
            x is not None
            for x in [
                self.hand_fit_intrinsic,
                self.hand_fit_trans,
                self.hand_fit_rot,
                self.hand_fit_pose,
                self.hand_fit_all,
            ]
        )
def _get_original_resolution(original_coords, cam_idx, fallback_size):
    if original_coords is None:
        return fallback_size
    if hasattr(original_coords, "detach"):
        coords = original_coords.detach().cpu().numpy()
    else:
        coords = np.asarray(original_coords)
    if coords.ndim < 2 or cam_idx >= coords.shape[0] or coords.shape[1] < 6:
        return fallback_size
    _, _, _, _, width, height = coords[cam_idx]
    if width <= 0 or height <= 0:
        return fallback_size
    return int(width), int(height)


def _intrinsic_to_original(intrinsic, original_coords, cam_idx):
    intr = np.asarray(intrinsic, dtype=np.float32)
    if intr.shape != (3, 3):
        return intr
    if original_coords is None:
        return intr
    if hasattr(original_coords, "detach"):
        coords = original_coords.detach().cpu().numpy()
    else:
        coords = np.asarray(original_coords)
    if coords.ndim < 2 or cam_idx >= coords.shape[0] or coords.shape[1] < 6:
        return intr
    x1, y1, x2, y2, width, height = coords[cam_idx]
    if width <= 0 or height <= 0:
        return intr
    scale_x = (x2 - x1) / float(width)
    scale_y = (y2 - y1) / float(height)
    if scale_x == 0 or scale_y == 0:
        return intr
    intr_adj = intr.copy()
    intr_adj[0, 0] /= scale_x
    intr_adj[1, 1] /= scale_y
    intr_adj[0, 2] = (intr_adj[0, 2] - x1) / scale_x
    intr_adj[1, 2] = (intr_adj[1, 2] - y1) / scale_y
    return intr_adj


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
            column_shares=[3, 2],
        ),
        rrb.Horizontal(
            rrb.Grid(
                *[
                    rrb.Spatial2DView(name=f"image_{i}", origin=f"/camera/image_{i}")
                    for i in range(num_images)
                ],
                grid_columns=25,
            ),
        ),
        row_shares=[3, 3],
    )


def main(args):
    obj_provider = StepDataProvider(Path(args.result_folder))
    hand_provider = HandDataProvider(Path(Path(args.result_folder).parents[0]))
    vis_name = obj_provider.base_dir.parents[0].name
    visualizer = Visualizer(vis_name, jpeg_quality=args.jpeg_quality)

    rr.send_blueprint(build_blueprint(args.num_frames))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    gen_3d = trimesh.load_mesh(obj_provider.mesh_path)

    for step in obj_provider.steps:
        step_idx = step["index"]
        data = step["data"]
        visualizer.set_time_sequence(step_idx)

        intr = data.get("intrinsics")
        extr = data.get("extrinsics")
        original_coords = data.get("original_coords")
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

            visualizer.log_image(f"camera/image_{cam_idx}", str(obj_provider.images[cam_idx]), static=False)
            with Image.open(obj_provider.images[cam_idx]) as im:
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
        cam_idx = int(step['path'].name)
        w, h = _get_original_resolution(original_coords, cam_idx, (w, h))
        intr_cam = _intrinsic_to_original(intr[cam_idx], original_coords, cam_idx)    
        visualizer.log_calibration(
            "camera/image",
            resolution=[w, h],
            intrins=intr_cam,
            image_plane_distance=1,
            static=False,
        )
        w2c = np.eye(4)
        w2c[:3] = extr[cam_idx]
        c2w = np.linalg.inv(w2c)
        visualizer.log_cam_pose("camera/image", c2w, static=False)
        tracks = np.asarray(pred_tracks)[cam_idx]
        if 1:
            visualizer.log_image("camera/image", str(obj_provider.get_reproj_error_vis_path(cam_idx)), static=False)
        else:
            visualizer.log_image("camera/image", str(obj_provider.images[cam_idx]), static=False)
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

        if hand_provider.has_hand:
            hand_modes = [
                ("intrinsic", LIGHT_GRAY),
                ("trans", GREEN),
                ("rot", LIGHT_RED),
            ]
            for mode, color in hand_modes:
                verts_cam = hand_provider.get_hand_verts_cam(mode, cam_idx)
                faces = hand_provider.get_hand_faces(mode)
                if verts_cam is None or faces is None:
                    continue
                verts_cam = np.asarray(verts_cam)
                faces = np.asarray(faces, dtype=np.int32)

                if seal_mano_mesh_np is not None:
                    try:
                        verts_cam, faces = seal_mano_mesh_np(verts_cam[None], faces, is_rhand=True)
                        verts_cam = np.asarray(verts_cam)[0]
                    except Exception as e:
                        print(f"[HandDataProvider] seal_mano_mesh_np failed for {mode}: {e}")
                verts_world = (c2w[:3, :3] @ verts_cam.T + c2w[:3, 3:4]).T
                color_rgb = np.array(color[:3], dtype=np.uint8)
                rr.log(
                    f"hand/{mode}/points",
                    rr.Points3D(
                        positions=verts_world,
                        colors=np.tile(color_rgb, (verts_world.shape[0], 1)),
                        radii=0.0005,
                    ),
                    static=False,
                )
                mat = add_material(color)
                rr.log(
                    f"hand/{mode}/mesh",
                    rr.Mesh3D(
                        vertex_positions=verts_world,
                        triangle_indices=faces,
                        mesh_material=mat,
                    ),
                    static=False,
                )

    if args.rrd_output_path:
        rr.save(args.rrd_output_path)


if __name__ == "__main__":
    args = parse_args()
    print(f"args provided: {args}")
    main(args)
