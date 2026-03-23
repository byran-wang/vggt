import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from third_party.utils_simba.utils_simba.depth import depth2xyzmap, get_depth
from tqdm import tqdm


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_intrinsic(meta_file: Path) -> np.ndarray:
    with open(meta_file, "rb") as f:
        try:
            meta = pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
            f.seek(0)
            meta = _NumpyCompatUnpickler(f).load()
    if "camMat" not in meta:
        raise KeyError(f"'camMat' not found in {meta_file}")
    cam_mat = np.asarray(meta["camMat"], dtype=np.float32)
    if cam_mat.shape != (3, 3):
        raise ValueError(f"camMat in {meta_file} must be 3x3, got {cam_mat.shape}")
    return cam_mat


def _depth_file_index(depth_file: Path) -> int | None:
    try:
        return int(depth_file.stem)
    except ValueError:
        return None


def _save_point_cloud_to_ply(points: np.ndarray, filepath: Path, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if len(colors) != len(points):
            raise ValueError("colors and points must have the same length")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is None:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main(args):
    input_dir = Path(args.input_dir)
    depth_dir = input_dir / args.depth_dir
    meta_dir = input_dir / args.meta_dir
    rgb_dir = input_dir / args.rgb_dir
    output_dir = input_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise FileNotFoundError(f"No depth png found in {depth_dir}")

    default_meta = None
    if args.meta_file:
        default_meta = Path(args.meta_file)
        if not default_meta.is_file():
            raise FileNotFoundError(f"Meta file not found: {default_meta}")
    else:
        meta_files = sorted(meta_dir.glob("*.pkl"))
        if not meta_files:
            raise FileNotFoundError(f"No meta pkl found in {meta_dir}")
        default_meta = meta_files[0]

    default_cam_mat = _load_intrinsic(default_meta)
    saved_count = 0

    for depth_file in tqdm(depth_files, desc="Processing depth files"):
        frame_idx = _depth_file_index(depth_file)
        if frame_idx is not None and args.ply_interval > 1 and frame_idx % args.ply_interval != 0:
            continue

        meta_file = meta_dir / f"{depth_file.stem}.pkl"
        cam_mat = _load_intrinsic(meta_file) if meta_file.is_file() else default_cam_mat

        depth = get_depth(str(depth_file), zfar=args.zfar)
        valid = depth > args.min_depth
        if np.isfinite(args.zfar):
            valid &= depth < args.zfar
        if not np.any(valid):
            continue

        xyz_map = depth2xyzmap(depth, cam_mat)
        points = xyz_map[valid]
        colors = None

        if args.use_rgb:
            rgb_file = next(
                (rgb_dir / f"{depth_file.stem}{ext}" for ext in (".jpg", ".png")
                 if (rgb_dir / f"{depth_file.stem}{ext}").is_file()),
                None
            )
            if rgb_file is not None and rgb_file.is_file():
                rgb = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
                if rgb is not None and rgb.shape[:2] == depth.shape:
                    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[valid]

        _save_point_cloud_to_ply(points, output_dir / f"{depth_file.stem}.ply", colors=colors)
        saved_count += 1

    print(f"Saved {saved_count} ply files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ZED exported depth maps to point-cloud PLY files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Scene root directory.")
    parser.add_argument("--depth_dir", type=str, default="depth_ZED", help="Depth folder name under input_dir.")
    parser.add_argument("--meta_dir", type=str, default="meta", help="Meta folder name under input_dir.")
    parser.add_argument("--rgb_dir", type=str, default="rgb", help="RGB folder name under input_dir.")
    parser.add_argument("--output_dir", type=str, default="ply_zed", help="Output folder name under input_dir.")
    parser.add_argument("--meta_file", type=str, default="", help="Optional fixed meta pkl path.")
    parser.add_argument("--ply_interval", type=int, default=1, help="Save one ply every N frames by frame index.")
    parser.add_argument("--min_depth", type=float, default=0.01, help="Minimum valid depth in meters.")
    parser.add_argument("--zfar", type=float, default=np.inf, help="Maximum valid depth in meters.")
    parser.add_argument("--use_rgb", action="store_true", help="If set, attach rgb colors from rgb/*.jpg.")
    main(parser.parse_args())
