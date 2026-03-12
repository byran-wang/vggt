"""
Convert HOI4D_processed depth images to PLY point clouds.

Layout assumed (output of preprocess.py):
  {processed_root}/train/{seq_name}/
    depth/*.png    (uint16, mm)
    meta/*.pkl     (camMat)
    rgb/*.jpg      (optional, for colored point clouds)

Output:
  {processed_root}/train/{seq_name}/ply/*.ply

Usage:
  # Single sequence
  python run_hoi4d/depth_to_ply_hoi4d.py \
      --seq ZY20210800002_H2_C7_N41_S57_s04_T1

  # All sequences
  python run_hoi4d/depth_to_ply_hoi4d.py

  # With RGB color and every 5th frame
  python run_hoi4d/depth_to_ply_hoi4d.py --use_rgb --ply_interval 5
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ── Intrinsic loader (reused from depth_to_ply.py) ───────────────────────────

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
        raise ValueError(f"camMat must be 3x3, got {cam_mat.shape}")
    return cam_mat


# ── PLY writer (reused from depth_to_ply.py) ─────────────────────────────────

def _save_point_cloud_to_ply(points: np.ndarray, filepath: Path,
                              colors: np.ndarray = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)

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


# ── Depth → XYZ ──────────────────────────────────────────────────────────────

def depth2xyzmap(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject depth (H,W) in metres to XYZ map (H,W,3) in camera frame."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    xyz = np.zeros((H, W, 3), dtype=np.float32)
    xyz[..., 0] = (uu - cx) / fx * depth
    xyz[..., 1] = (vv - cy) / fy * depth
    xyz[..., 2] = depth
    return xyz


# ── Per-sequence processing ───────────────────────────────────────────────────

def process_sequence(seq_dir: Path, args) -> int:
    depth_dir = seq_dir / "depth"
    meta_dir  = seq_dir / "meta"
    rgb_dir   = seq_dir / "rgb"
    out_dir   = seq_dir / "ply"

    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        print(f"  [WARN] no depth files: {depth_dir}")
        return 0

    meta_files = sorted(meta_dir.glob("*.pkl"))
    if not meta_files:
        print(f"  [WARN] no meta files: {meta_dir}")
        return 0

    default_K = _load_intrinsic(meta_files[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for depth_file in tqdm(depth_files, desc=seq_dir.name, leave=False):
        # interval filter
        try:
            fidx = int(depth_file.stem)
        except ValueError:
            fidx = None
        if fidx is not None and args.ply_interval > 1 and fidx % args.ply_interval != 0:
            continue

        out_ply = out_dir / f"{depth_file.stem}.ply"
        if out_ply.exists() and not args.overwrite:
            saved += 1
            continue

        # per-frame intrinsics if available
        meta_file = meta_dir / f"{depth_file.stem}.pkl"
        K = _load_intrinsic(meta_file) if meta_file.is_file() else default_K

        # HO3D packed 3-channel format: depth_m = (R + G*256) * scale
        _DEPTH_SCALE = 0.00012498664727900177
        raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        depth = (raw[..., 2].astype(np.float32) + raw[..., 1].astype(np.float32) * 256) * _DEPTH_SCALE

        valid = (depth > args.min_depth)
        if np.isfinite(args.zfar):
            valid &= (depth < args.zfar)
        if not np.any(valid):
            continue

        xyz_map = depth2xyzmap(depth, K)
        points  = xyz_map[valid]
        colors  = None

        if args.use_rgb:
            rgb_file = rgb_dir / f"{depth_file.stem}.jpg"
            if rgb_file.is_file():
                rgb = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
                if rgb is not None and rgb.shape[:2] == depth.shape:
                    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[valid]

        _save_point_cloud_to_ply(points, out_ply, colors=colors)
        saved += 1

    return saved


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Convert HOI4D_processed depth to PLY")
    p.add_argument("--processed_root", default="/mnt/hdd_volume/datasets/HOI4D_ori/HOI4D_processed",
                   help="Root of HOI4D_processed (contains train/)")
    p.add_argument("--seq", default="",
                   help="Single sequence name. If empty, process all sequences.")
    p.add_argument("--ply_interval", type=int, default=1,
                   help="Save one PLY every N frames (by frame index)")
    p.add_argument("--min_depth", type=float, default=0.01,
                   help="Minimum valid depth in metres")
    p.add_argument("--zfar", type=float, default=float("inf"),
                   help="Maximum valid depth in metres")
    p.add_argument("--use_rgb", action="store_true",
                   help="Attach RGB colour from rgb/*.jpg")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate PLY files that already exist")
    args = p.parse_args()

    train_root = Path(args.processed_root) / "train"
    if not train_root.is_dir():
        print(f"[ERROR] train dir not found: {train_root}")
        sys.exit(1)

    if args.seq:
        seq_dirs = [train_root / args.seq]
    else:
        seq_dirs = sorted(d for d in train_root.iterdir() if d.is_dir())

    print(f"Processing {len(seq_dirs)} sequence(s)...")
    total = 0
    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            print(f"[WARN] not found: {seq_dir}")
            continue
        n = process_sequence(seq_dir, args)
        print(f"  {seq_dir.name}: {n} PLY saved")
        total += n

    print(f"\nDone. Total PLY saved: {total}")


if __name__ == "__main__":
    main()
