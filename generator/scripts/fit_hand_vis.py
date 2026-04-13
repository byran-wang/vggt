"""Visualize hand fitting results from fit_hand.py in Rerun.

Displays per-frame hand mesh (gray + vertex normals), camera intrinsics,
and the original RGB image.

Usage:
    cd generator && python scripts/fit_hand_vis.py \
        --data_dir /path/to/dataset/scene_name --mode h_trans
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.logger import get_logger

logger = get_logger(__name__)


def _load_fit(data_dir, mode):
    """Load hand fitting .npy, trying both hold_fit and hand_fit prefixes."""
    data_dir = Path(data_dir)
    for prefix in ("hold_fit", "hand_fit"):
        path = data_dir / f"{prefix}.aligned_{mode}.npy"
        if path.exists():
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                return arr.item()
            return arr
    return None


def _extract(fit, key):
    """Extract a key from the fit dict, handling nested 'right' sub-dict."""
    if fit is None:
        return None
    if isinstance(fit, dict):
        for hand_key in ("right", "rhand", "hand"):
            sub = fit.get(hand_key)
            if isinstance(sub, dict) and key in sub:
                return sub[key]
        return fit.get(key)
    return None


def _load_intrinsics(data_dir, fid):
    """Load camera intrinsics (3x3) from meta pickle."""
    meta_path = Path(data_dir) / "meta" / f"{fid:04d}.pkl"
    if not meta_path.exists():
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    K = meta.get("intrinsics", meta.get("camMat"))
    if K is not None:
        return np.array(K, dtype=np.float64).reshape(3, 3)
    return None


def _load_image(data_dir, fid):
    """Load RGB image for a frame."""
    from PIL import Image

    rgb_dir = Path(data_dir) / "rgb"
    for ext in (".png", ".jpg", ".jpeg"):
        p = rgb_dir / f"{fid:04d}{ext}"
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    return None


def main(args):
    data_dir = Path(args.data_dir)

    # Load hand fit data
    fit = _load_fit(data_dir, args.mode)
    if fit is None:
        logger.error(f"No hand fit data found for mode '{args.mode}' in {data_dir}")
        return

    v3d_cam = _extract(fit, "v3d_cam")
    f3d = _extract(fit, "f3d")
    if v3d_cam is None or f3d is None:
        logger.error(f"Missing v3d_cam or f3d in fit data")
        return

    v3d_cam = np.asarray(v3d_cam, dtype=np.float32)
    f3d = np.asarray(f3d, dtype=np.int32)

    # Seal the wrist opening
    from common.body_models import seal_mano_mesh_np
    v3d_cam, f3d = seal_mano_mesh_np(v3d_cam, f3d, is_rhand=True)

    logger.info(f"Loaded hand fit: {v3d_cam.shape[0]} frames, "
                f"{v3d_cam.shape[1]} verts, {f3d.shape[0]} faces (sealed)")

    # Determine frame indices
    if args.frame_indices is not None:
        frame_indices = args.frame_indices
    else:
        frame_list_path = data_dir / "hands" / "frame_list.txt"
        if frame_list_path.exists():
            with open(frame_list_path, "r") as f:
                frame_indices = [int(line.strip()) for line in f if line.strip()]
            logger.info(f"Loaded {len(frame_indices)} frames from {frame_list_path}")
        else:
            frame_indices = list(range(v3d_cam.shape[0]))
            logger.warning(f"No frame_list.txt found, using indices 0..{len(frame_indices)-1}")

    if len(frame_indices) != v3d_cam.shape[0]:
        logger.warning(f"Frame list length ({len(frame_indices)}) != vertex data length ({v3d_cam.shape[0]}), "
                       f"using min of both")
        n = min(len(frame_indices), v3d_cam.shape[0])
        frame_indices = frame_indices[:n]
        v3d_cam = v3d_cam[:n]

    # Init Rerun
    rr.init("fit_hand_vis", spawn=True)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Spatial2DView(name="Camera", origin="world/camera"),
            column_shares=[2, 1],
        ),
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Load intrinsics once (same for all frames typically)
    K = _load_intrinsics(data_dir, frame_indices[0])

    for i, fid in enumerate(frame_indices):
        rr.set_time_sequence("frame", i)

        verts = v3d_cam[i]  # (778, 3)

        # Skip frames with NaN vertices
        if np.isnan(verts).any():
            continue

        # Compute vertex normals via trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=f3d, process=False)
        vnormals = np.array(mesh.vertex_normals, dtype=np.float32)

        # Log hand mesh (gray + normals)
        rr.log("world/hand_mesh", rr.Mesh3D(
            vertex_positions=verts,
            triangle_indices=f3d.astype(np.uint32),
            vertex_normals=vnormals,
            mesh_material=rr.Material(albedo_factor=[180, 180, 180]),
        ))

        # Log camera image with intrinsics
        img = _load_image(data_dir, fid)
        frame_K = _load_intrinsics(data_dir, fid)
        if frame_K is None:
            frame_K = K
        if img is not None and frame_K is not None:
            H, W = img.shape[:2]
            rr.log("world/camera", rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(frame_K[0, 0]), float(frame_K[1, 1])],
                principal_point=[float(frame_K[0, 2]), float(frame_K[1, 2])],
                image_plane_distance=1.0,
            ))
            rr.log("world/camera", rr.Image(img).compress(jpeg_quality=args.jpeg_quality))

        # Log filtered depth as 3D points (masked by hand mask)
        if frame_K is not None:
            depth_path = data_dir / "depth" / f"{fid:04d}.png"
            mask_hand_path = data_dir / "mask_hand" / f"{fid:04d}.png"
            if depth_path.exists():
                from utils_simba.depth import load_filtered_depth
                from PIL import Image as _PILImage
                depth = load_filtered_depth(str(depth_path))
                valid = depth > 0.01
                if mask_hand_path.exists():
                    mask_hand = np.array(_PILImage.open(mask_hand_path).convert("L"))
                    valid = valid & (mask_hand > 0)
                ys, xs = np.where(valid)
                if len(ys) > 0:
                    zs = depth[ys, xs].astype(np.float32)
                    xc = (xs.astype(np.float32) - frame_K[0, 2]) * zs / frame_K[0, 0]
                    yc = (ys.astype(np.float32) - frame_K[1, 2]) * zs / frame_K[1, 1]
                    pts = np.stack([xc, yc, zs], axis=-1)
                    rr.log("world/depth_points", rr.Points3D(pts, radii=0.0005))

    logger.info(f"Logged {len(frame_indices)} frames to Rerun (mode={args.mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand fitting results in Rerun")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory (e.g. ho3d_v3/train/MC1)")
    parser.add_argument("--mode", type=str, default="h_trans",
                        help="Hand fit mode (e.g. h_intrinsic, h_trans, h_rot, h_pose, h_all)")
    parser.add_argument("--frame_indices", type=int, nargs="+", default=None,
                        help="Explicit frame indices (overrides frame_list.txt)")
    parser.add_argument("--jpeg_quality", type=int, default=30,
                        help="JPEG compression quality for camera images")
    args = parser.parse_args()
    main(args)
