import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch


def _setup_paths() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _setup_paths()


def _load_image_paths(image_dir: Path, frame_list_path: Path) -> List[Path]:
    image_paths: List[Path] = []
    if frame_list_path.exists():
        frames = [line.strip() for line in frame_list_path.read_text().splitlines() if line.strip()]
        for frame in frames:
            for ext in (".png", ".jpg", ".jpeg"):
                p = image_dir / f"{frame}{ext}"
                if p.exists():
                    image_paths.append(p)
                    break
        if image_paths:
            return image_paths

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda x: x.stem,
    )
    return image_paths


def _resolve_condition_pos(image_paths, cond_index: int) -> int:
    for i, p in enumerate(image_paths):
        try:
            if int(p.stem) == cond_index:
                return i
        except ValueError:
            continue
    return max(0, min(cond_index, len(image_paths) - 1))


def _draw_correspondences(
    img0_path: Path,
    img1_path: Path,
    pts0: np.ndarray,
    pts1: np.ndarray,
    out_path: Path,
    max_draw: int = 200,
) -> None:
    img0 = cv2.imread(str(img0_path), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        return

    if len(pts0) == 0:
        canvas = np.concatenate([img0, img1], axis=1)
        cv2.imwrite(str(out_path), canvas)
        return

    n = min(max_draw, len(pts0))
    ids = np.linspace(0, len(pts0) - 1, n, dtype=np.int32)
    pts0_s = pts0[ids]
    pts1_s = pts1[ids]

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas_h = max(h0, h1)
    canvas = np.zeros((canvas_h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    for i in range(n):
        c = tuple(int(v) for v in colors[i].tolist())
        x0, y0 = int(round(float(pts0_s[i, 0]))), int(round(float(pts0_s[i, 1])))
        x1, y1 = int(round(float(pts1_s[i, 0]))), int(round(float(pts1_s[i, 1])))
        x1_shifted = x1 + w0
        cv2.circle(canvas, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1_shifted, y1), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x0, y0), (x1_shifted, y1), c, 1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def main(args):
    data_dir = Path(args.data_dir)
    corres_dir = Path(args.corres_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = data_dir / "rgb"
    frame_list_path = data_dir / "frame_list.txt"
    image_paths = _load_image_paths(image_dir, frame_list_path)

    if len(image_paths) < 2:
        print(f"Need at least 2 images in {image_dir}, found {len(image_paths)}")
        return

    cond_pos = _resolve_condition_pos(image_paths, args.cond_index)
    print(f"Condition frame: index={args.cond_index}, position={cond_pos}, file={image_paths[cond_pos].name}")

    # Load saved VGGSfM tracking results
    tracks_path = corres_dir / "corres" / "vggsfm_tracks.npz"
    if not tracks_path.exists():
        print(f"Tracks file not found: {tracks_path}")
        return

    data = np.load(tracks_path, allow_pickle=True)
    pred_tracks = data["tracks"]           # (S, N, 2)
    pred_vis_scores = data["vis_scores"]   # (S, N)
    pred_tracks_mask = data["tracks_mask"]  # (S, N)
    saved_image_paths = data["image_paths"] # (S,)

    num_frames = pred_tracks.shape[0]
    num_tracks = pred_tracks.shape[1]
    print(f"Loaded tracks: {num_frames} frames, {num_tracks} tracks")

    if cond_pos >= num_frames:
        print(f"Condition position {cond_pos} out of range (num_frames={num_frames})")
        return

    # Draw correspondences from condition frame to every other frame
    total_pairs = 0
    for j in tqdm(range(num_frames), desc="Drawing correspondences"):
        if j == cond_pos:
            continue

        keep = pred_tracks_mask[cond_pos] & pred_tracks_mask[j]
        # Apply visibility threshold filter
        vis_keep = (pred_vis_scores[cond_pos] >= args.vis_thresh) & (pred_vis_scores[j] >= args.vis_thresh)
        keep = keep & vis_keep
        num_valid = int(keep.sum())
        if num_valid == 0:
            print(f"  No valid correspondences between cond({image_paths[cond_pos].stem}) and {image_paths[j].stem}")
            continue

        pts_cond = pred_tracks[cond_pos][keep].astype(np.float32)
        pts_j = pred_tracks[j][keep].astype(np.float32)

        pair_name = f"{image_paths[cond_pos].stem}_{image_paths[j].stem}"
        vis_path = out_dir / f"{pair_name}.png"
        _draw_correspondences(
            image_paths[cond_pos], image_paths[j],
            pts_cond, pts_j,
            vis_path, args.max_vis_matches,
        )
        total_pairs += 1
        print(f"  {pair_name}: {num_valid} valid correspondences")

    print(f"Done. Drew {total_pairs} pair visualizations to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize correspondences from pipeline_get_corres.py")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Pipeline preprocess directory (contains rgb/, frame_list.txt)")
    parser.add_argument("--corres_dir", type=str, required=True,
                        help="Correspondence output directory from pipeline_get_corres.py (contains corres/vggsfm_tracks.npz)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for correspondence visualizations")
    parser.add_argument("--cond_index", type=int, required=True,
                        help="Condition frame index to visualize correspondences from")
    parser.add_argument("--max_vis_matches", type=int, default=2000,
                        help="Maximum number of correspondences to draw per pair")
    parser.add_argument("--vis_thresh", type=float, default=0.5,
                        help="Minimum visibility score threshold for valid correspondences")

    main(parser.parse_args())
