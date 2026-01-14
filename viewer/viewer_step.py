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
    parser.add_argument(
        "--only_current_view",
        type=int,
        default=1,
        help="Only visualize the current camera view.",
    )
    parser.add_argument(
        "--gt_ho3d",
        type=int,
        default=0,
        help="Visualize HO3D ground-truth cameras and object meshes when available.",
    )
    parser.add_argument(
        "--aliggned_mesh",
        action="store_true",
        help="Visualize the aligned remeshed 3D mesh at each step.",
    )
    parser.add_argument(
        "--vis_only_keyframes",
        action="store_true",
        help="Only visualize keyframes (skip non-keyframe steps).",
    )
    return parser.parse_args()


class ObjDataProvider:
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
        self.origin_images = sorted((self.base_dir / "images_origin").glob("*.jpg"))
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

    def get_seq_name(self):
        return self.base_dir.parents[0].name

    def get_reproj_error_vis_path(self, step_idx: int) -> Path:
        step_dir = self.base_dir / f"{step_idx:04d}"
        vis_path = step_dir / "reproj_error.png"
        return vis_path
    
    def get_image_fids(self):
        with open(self.result_dir / "image_paths.txt", "r") as f:
            image_fids = [int(Path(line.strip()).stem) for line in f.readlines()]
        return image_fids
    
    def get_image_fs(self):
        with open(self.result_dir / "image_paths.txt", "r") as f:
            image_fs = [Path(line.strip()) for line in f.readlines()]
        return image_fs    
    
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
    
    def get_hand_verts(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "v3d_cam") 

    def get_hand_beta(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "hand_beta")  

    def get_hand_poses(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "hand_pose")  

    def get_hand_transls(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "hand_transl")  

    def get_hand_rots(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "hand_rot")                   

    def get_hand_scale(self, mode: str):
        fit = self._get_fit(f"fit_{mode}")
        return self._extract_from_fit(fit, "hand_scale")  

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


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


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


def build_blueprint(args) -> rrb.BlueprintLike:
    num_images = args.num_frames
    rows = [
        rrb.Horizontal(
            rrb.Spatial3DView(name="our_world", origin="/our"),
            rrb.Vertical(
                rrb.Spatial2DView(name="camera_current", origin="/our/camera_current/image"),
                rrb.Spatial2DView(name="camera_obj", origin="/our/camera_obj/image"),
                row_shares=[1, 3],
            ),
            column_shares=[6, 2],
        )
    ]
    row_shares = [3]
    if getattr(args, "gt_ho3d", False):
        rows.append(rrb.Spatial3DView(name="gt_world", origin="/gt"))
        row_shares.append(2)

    if not args.only_current_view:
        rows.append(
            rrb.Horizontal(
                rrb.Grid(
                    *[
                        rrb.Spatial2DView(name=f"image_{i}", origin=f"/camera/image_{i}")
                        for i in range(num_images)
                    ],
                    grid_columns=25,
                ),
            )
        )
        row_shares.append(3)

    return rrb.Vertical(*rows, row_shares=row_shares)


def log_all_frames(
    visualizer: Visualizer,
    obj_provider: ObjDataProvider,
    intr,
    extr,
    args,
    registered,
    pred_tracks,
    track_mask,
    keyframe_flags,
):
    reg_flags = (
        np.asarray(registered).astype(bool)
        if args.vis_only_register and registered is not None
        else None
    )
    for cam_idx in range(len(extr)):
        if reg_flags is not None:
            if cam_idx >= len(reg_flags) or not reg_flags[cam_idx]:
                continue

        visualizer.log_image(
            f"/our/camera/image_{cam_idx}", str(obj_provider.images[cam_idx]), static=False
        )
        with Image.open(obj_provider.images[cam_idx]) as im:
            w, h = im.size
        visualizer.log_calibration(
            f"/our/camera/image_{cam_idx}",
            resolution=[w, h],
            intrins=intr[cam_idx],
            image_plane_distance=args.image_plane_distance,
            static=False,
        )
        w2c = np.eye(4)
        w2c[:3] = extr[cam_idx]
        c2w = np.linalg.inv(w2c)
        visualizer.log_cam_pose(f"/our/camera/image_{cam_idx}", c2w, static=False)

        is_keyframe = (
            keyframe_flags is not None
            and cam_idx < len(keyframe_flags)
            and bool(np.asarray(keyframe_flags)[cam_idx])
        )

        if is_keyframe:
            rr.log(
                f"/our/camera/image_{cam_idx}/keyframe_border",
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
                        f"/our/camera/image_{cam_idx}/track_count",
                        rr.TextLog(f"track_count: {track_count}"),
                        static=False,
                    )
                rr.log(
                    f"/our/camera/image_{cam_idx}/keypoints",
                    rr.Points2D(tracks[mask], colors=[34, 138, 167]),
                    static=False,
                )


def log_current_frame(
    visualizer: Visualizer,
    obj_provider: ObjDataProvider,
    extr,
    intr,
    original_coords,
    cam_idx: int,
):
    with Image.open(obj_provider.images[cam_idx]) as im:
        w, h = im.size
    w2c = np.eye(4)
    w2c[:3] = extr[cam_idx]
    c2w = np.linalg.inv(w2c)
    visualizer.log_cam_pose("/our/camera_current/image", c2w, static=False)
    w, h = _get_original_resolution(original_coords, cam_idx, (w, h))
    intr_cam = _intrinsic_to_original(intr[cam_idx], original_coords, cam_idx)
    visualizer.log_calibration(
        "/our/camera_current/image",
        resolution=[w, h],
        intrins=intr_cam,
        image_plane_distance=1,
        static=False,
    )
    visualizer.log_image(
        "/our/camera_current/image", str(obj_provider.origin_images[cam_idx]), static=False
    )
    visualizer.log_image(
        "/our/camera_obj/image",
        str(obj_provider.get_reproj_error_vis_path(cam_idx)),
        static=False,
    )


def log_points_3d(
    visualizer: Visualizer, points_3d, points_rgb=None, points_conf_color=None
):
    if points_3d is None:
        return
    pts = np.asarray(points_3d)
    visualizer.log_points("/our/points_rgb", pts, colors=points_rgb, static=False)
    visualizer.log_points("/our/points_conf", pts, colors=points_conf_color, static=False)


def log_depth_points_3d(
    visualizer: Visualizer,
    obj_provider: ObjDataProvider,
    cam_idx: int,
    extr,
    intr,
    pred_tracks,
    track_mask,
):
    if cam_idx >= len(obj_provider.depths):
        return
    depth_path = obj_provider.depths[cam_idx]
    if not depth_path.exists():
        return
    if intr is None or cam_idx >= len(intr):
        return
    if pred_tracks is None or track_mask is None:
        return
    tracks = np.asarray(pred_tracks)[cam_idx]
    mask = np.asarray(track_mask)[cam_idx].astype(bool)
    if tracks.shape[0] != mask.shape[0]:
        return
    if not np.any(mask):
        return
    depth = get_depth(str(depth_path))
    intr_cam = intr[cam_idx]
    uvs = np.round(tracks[mask]).astype(int)
    h, w = depth.shape[:2]
    in_bounds = (
        (uvs[:, 0] >= 0)
        & (uvs[:, 0] < w)
        & (uvs[:, 1] >= 0)
        & (uvs[:, 1] < h)
    )
    if not np.any(in_bounds):
        return
    uvs = uvs[in_bounds]
    z = depth[uvs[:, 1], uvs[:, 0]]
    valid_depth = z > 0
    if not np.any(valid_depth):
        return
    uvs = uvs[valid_depth]
    z = z[valid_depth]
    fx, fy = intr_cam[0, 0], intr_cam[1, 1]
    cx, cy = intr_cam[0, 2], intr_cam[1, 2]
    x = (uvs[:, 0] - cx) * z / fx
    y = (uvs[:, 1] - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    w2c = np.eye(4)
    w2c[:3] = extr[cam_idx]
    c2w = np.linalg.inv(w2c)
    pts_world = (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T
    visualizer.log_points("/our/depth_points", pts_world, static=False)


def log_gt_frame(
    visualizer: Visualizer,
    gt_data,
    obj_provider: ObjDataProvider,
    cam_idx: int,
):
    # breakpoint()
    c2o = _to_numpy(gt_data.get("o2c"))
    c2o = np.linalg.inv(c2o)

    if cam_idx >= len(c2o):
        print(f"[WARN][log_gt_frame] cam_idx {cam_idx} out of range for c2o with length {len(c2o)}")
        return

    is_valid = gt_data.get("is_valid")
    if is_valid is not None:
        valid_flags = _to_numpy(is_valid)
        if cam_idx < len(valid_flags) and not bool(valid_flags[cam_idx]):
            return


    with Image.open(obj_provider.origin_images[cam_idx]) as im:
        w, h = im.size
    intr = gt_data.get("K")
    if intr is not None:
        visualizer.log_calibration(
            "gt/camera/image",
            resolution=[w, h],
            intrins=_to_numpy(intr),
            image_plane_distance=1,
            static=False,
    )
    visualizer.log_cam_pose("gt/camera/image", c2o[cam_idx], static=False)
    visualizer.log_image("gt/camera/image", str(obj_provider.origin_images[cam_idx]), static=False)

    # Log GT hand mesh/points in object/world space
    verts_cam = gt_data.get("v3d_c.right")
    faces_hand = gt_data.get("faces.right")
    if verts_cam is not None and faces_hand is not None and cam_idx < len(verts_cam):
        verts_cam_np = _to_numpy(verts_cam[cam_idx])
        R = c2o[cam_idx][:3, :3]
        t = c2o[cam_idx][:3, 3]
        verts_world = (R @ verts_cam_np.T + t[:, None]).T
        color_rgb = np.array([0, 255, 0], dtype=np.uint8)
        rr.log(
            "gt/hand/points",
            rr.Points3D(
                positions=verts_world,
                colors=np.tile(color_rgb, (verts_world.shape[0], 1)),
                radii=0.0005,
            ),
            static=False,
        )
        rr.log(
            "gt/hand/mesh",
            rr.Mesh3D(
                vertex_positions=verts_world,
                triangle_indices=_to_numpy(faces_hand).astype(np.int32),
                mesh_material=add_material([0, 255, 0, 255]),
            ),
            static=False,
        )

    # v3d_object = gt_data.get("v3d_c.object")
    # faces_object = gt_data.get("faces.object")
    # if v3d_object is None or faces_object is None:
    #     return
    # verts_cam = _to_numpy(v3d_object[cam_idx])
    # faces = _to_numpy(faces_object).astype(np.int32)
    # colors_obj = gt_data.get("colors.object")
    # colors = _to_numpy(colors_obj) if colors_obj is not None else None    
    # rr.log(
    #     "gt/object_mesh",
    #     rr.Mesh3D(
    #         vertex_positions=verts_cam,
    #         triangle_indices=faces,
    #         vertex_colors=colors,
    #     ),
    #     static=False,
    # )


def log_hands(hand_provider: HandDataProvider, extr, cam_idx: int):
    if not hand_provider.has_hand:
        return

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
        w2c = np.eye(4)
        w2c[:3] = extr[cam_idx]
        c2w = np.linalg.inv(w2c)
        verts_world = (c2w[:3, :3] @ verts_cam.T + c2w[:3, 3:4]).T
        color_rgb = np.array(color[:3], dtype=np.uint8)
        rr.log(
            f"our/hand/{mode}/points",
            rr.Points3D(
                positions=verts_world,
                colors=np.tile(color_rgb, (verts_world.shape[0], 1)),
                radii=0.0005,
            ),
            static=False,
        )
        mat = add_material(color)
        rr.log(
            f"our/hand/{mode}/mesh",
            rr.Mesh3D(
                vertex_positions=verts_world,
                triangle_indices=faces,
                mesh_material=mat,
            ),
            static=False,
        )      

def main(args):
 
    obj_provider = ObjDataProvider(Path(args.result_folder))
    hand_provider = HandDataProvider(Path(Path(args.result_folder).parents[0]))
    gt_data = None
    if args.gt_ho3d:
        _CODE_DIR = Path(__file__).resolve().parents[1]
        if _CODE_DIR.is_dir():
            sys.path = [str(_CODE_DIR)] + sys.path
        from vggt.utils.gt import load_data as load_gt_data        
        seq_name = obj_provider.get_seq_name()
        gt_data = load_gt_data(seq_name, obj_provider.get_image_fids)
    
    vis_name = obj_provider.base_dir.parents[0].name
    visualizer = Visualizer(vis_name, jpeg_quality=args.jpeg_quality)

    rr.send_blueprint(build_blueprint(args))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    gen_3d = trimesh.load_mesh(obj_provider.mesh_path)

    if args.gt_ho3d and gt_data is not None:
        mesh_path = gt_data.get("mesh_name.object")
        visualizer.log_mesh(
            "gt/object_mesh/static",
            mesh_path,
            static=True,
        )


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

        # Skip non-keyframe steps if vis_only_keyframes is enabled
        if args.vis_only_keyframes:
            cam_idx = int(step["path"].name)
            is_keyframe = (
                keyframe_flags is not None
                and cam_idx < len(keyframe_flags)
                and bool(np.asarray(keyframe_flags)[cam_idx])
            )
            if not is_keyframe:
                continue
        

        if not args.only_current_view:
            log_all_frames(
                visualizer=visualizer,
                obj_provider=obj_provider,
                intr=intr,
                extr=extr,
                args=args,
                registered=registered,
                pred_tracks=pred_tracks,
                track_mask=track_mask,
                keyframe_flags=keyframe_flags,
            )

        # log the current camera view
        cam_idx = int(step["path"].name)
        log_current_frame(
            visualizer=visualizer,
            obj_provider=obj_provider,
            extr=extr,
            intr=intr,
            original_coords=original_coords,
            cam_idx=cam_idx,
        )

        if args.aliggned_mesh:
            visualizer.log_mesh("/our/aligned_mesh", gen_3d_mesh_aligned_path, colors=np.array([255, 255, 255]), static=False)
        
        log_points_3d(
            visualizer=visualizer,
            points_3d=points_3d,
            points_rgb=points_rgb,
            points_conf_color=points_conf_color,
        )
        log_depth_points_3d(
            visualizer=visualizer,
            obj_provider=obj_provider,
            cam_idx=cam_idx,
            extr=extr,
            intr=intr,
            pred_tracks=pred_tracks,
            track_mask=track_mask,
        )

        log_hands(hand_provider=hand_provider, extr=extr, cam_idx=cam_idx)

        if args.gt_ho3d:
            log_gt_frame(
                visualizer=visualizer,
                gt_data=gt_data,
                obj_provider=obj_provider,
                cam_idx=cam_idx,
            )

    if args.rrd_output_path:
        rr.save(args.rrd_output_path)


if __name__ == "__main__":
    args = parse_args()
    print(f"args provided: {args}")
    main(args)
