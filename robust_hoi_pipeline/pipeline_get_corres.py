import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch


def _setup_paths() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))
    return project_root


PROJECT_ROOT = _setup_paths()

from vggt.dependency.track_predict import predict_tracks
from robust_hoi_pipeline.pipeline_utils import (
    compute_vggsfm_foreground_mask,
    compute_vggsfm_depth_mask,
)


def _load_image_paths(image_dir: Path, frame_list_path: Path) -> List[Path]:
    image_paths: List[Path] = []
    if frame_list_path.exists():
        frames = [line.strip() for line in frame_list_path.read_text().splitlines() if line.strip()]
        for frame in frames:
            png = image_dir / f"{frame}.png"
            jpg = image_dir / f"{frame}.jpg"
            jpeg = image_dir / f"{frame}.jpeg"
            if png.exists():
                image_paths.append(png)
            elif jpg.exists():
                image_paths.append(jpg)
            elif jpeg.exists():
                image_paths.append(jpeg)
        if image_paths:
            return image_paths

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda x: x.stem,
    )
    return image_paths


def _resolve_condition_pos(image_paths: Sequence[Path], cond_index: int) -> int:
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


def _load_vggsfm_sequence_images(
    image_paths: Sequence[Path],
    device: str,
    mask_dir: Path = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load images for VGGSfM tracking, optionally masking with black background.

    Args:
        image_paths: Sequence of image file paths
        device: Torch device string
        mask_dir: Optional directory containing masks (same stem as images, .png format)

    Returns:
        images: (S, 3, H, W) tensor of images
        masks: (S, 1, H, W) tensor of binary masks (1 for foreground, 0 for background)
    """
    images = []
    masks = []
    ref_hw = None
    for path in image_paths:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if ref_hw is None:
            ref_hw = img.shape[:2]
        elif img.shape[:2] != ref_hw:
            raise ValueError(
                f"VGGSfM sequence requires same resolution, got {ref_hw} and {img.shape[:2]} "
                f"for {path.name}"
            )

        # Load and apply mask if mask_dir is provided
        if mask_dir is not None:
            mask_path = mask_dir / f"{path.stem}.png"
            if mask_path.exists():
                mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
                # Ensure mask has same resolution as image
                if mask.shape[:2] != ref_hw:
                    mask = cv2.resize(mask, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_NEAREST)
                # Apply mask: set background to black
                img = img * mask[..., None]
                masks.append(torch.from_numpy(mask).unsqueeze(0))  # (1, H, W)
            else:
                print(f"Warning: Mask not found for {path.name}, using full image")
                masks.append(torch.ones((1, ref_hw[0], ref_hw[1]), dtype=torch.float32))
        else:
            masks.append(torch.ones((1, ref_hw[0], ref_hw[1]), dtype=torch.float32))

        images.append(torch.from_numpy(img).permute(2, 0, 1))

    images_tensor = torch.stack(images, dim=0).to(device)
    masks_tensor = torch.stack(masks, dim=0).to(device)
    return images_tensor, masks_tensor


def _build_anchor_sequence(cond_pos: int, num_frames: int, interval: int) -> List[int]:
    """Build anchor frame indices: forward from cond_pos, then backward from cond_pos.

    Args:
        cond_pos: Condition frame position (first anchor).
        num_frames: Total number of frames.
        interval: Step size between anchors.

    Returns:
        Ordered list of unique anchor indices.
    """
    anchors = [cond_pos]
    # Move forward from cond_pos
    pos = cond_pos + interval
    while pos < num_frames:
        anchors.append(pos)
        pos += interval
    # Move backward from cond_pos
    pos = cond_pos - interval
    while pos >= 0:
        anchors.append(pos)
        pos -= interval
    return anchors


def _get_window_indices(anchor: int, window_size: int, num_frames: int) -> List[int]:
    """Get frame indices for a window centered on anchor, clamped to [0, num_frames)."""
    half = window_size // 2
    start = max(0, anchor - half)
    end = min(num_frames, start + window_size)
    # Adjust start if end hit the boundary
    start = max(0, end - window_size)
    return list(range(start, end))


def main(args):
    data_dir = Path(args.data_dir)  # the out_dir of pipeline_data_preprocess.py
    out_dir = Path(args.out_dir)
    image_dir = data_dir / "rgb"
    mask_dir = data_dir / "mask_obj"
    frame_list_path = data_dir / "frame_list.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    corres_dir = out_dir / "corres"
    corres_vis_dir = out_dir / "corres_vis"
    corres_dir.mkdir(parents=True, exist_ok=True)
    corres_vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _load_image_paths(image_dir, frame_list_path)
    if len(image_paths) < 2:
        print(f"Need at least 2 images in {image_dir}, found {len(image_paths)}")
        return

    cond_pos = _resolve_condition_pos(image_paths, args.cond_index)
    num_frames = len(image_paths)

    # Build anchor sequence and sliding windows
    anchors = _build_anchor_sequence(cond_pos, num_frames, args.anchor_interval)
    print(f"Sliding window tracking: {len(anchors)} anchors, window_size={args.window_size}, interval={args.anchor_interval}")
    print(f"Anchor positions: {anchors}")

    # Save anchor indices and their window ranges
    anchor_window_path = corres_dir / "anchor_window.txt"
    with open(anchor_window_path, "w") as f:
        f.write("# anchor_frame window_start_frame window_end_frame\n")
        for anchor in anchors:
            win = _get_window_indices(anchor, args.window_size, num_frames)
            f.write(f"{image_paths[anchor].stem} {image_paths[win[0]].stem} {image_paths[win[-1]].stem}\n")
    print(f"Saved anchor/window info to {anchor_window_path}")

    # Load all images once (shared across windows)
    print("Loading all images (with masked background)...")
    images_all, image_masks = _load_vggsfm_sequence_images(image_paths, args.device, mask_dir)

    # Accumulate tracks from all windows
    # Global arrays: pred_tracks (num_frames, N_total, 2), pred_vis_scores (num_frames, N_total)
    # pred_tracks_mask (num_frames, N_total) - only valid within window frames
    all_tracks = []       # list of (num_frames, N_w, 2) arrays
    all_vis_scores = []   # list of (num_frames, N_w) arrays
    all_window_masks = [] # list of (num_frames, N_w) bool arrays

    for anchor in tqdm(anchors, desc="Sliding window tracking"):
        win_indices = _get_window_indices(anchor, args.window_size, num_frames)
        win_size = len(win_indices)
        if win_size < 2:
            continue

        # Find anchor position within the window
        anchor_in_win = win_indices.index(anchor)

        # Extract window subset of images and masks
        images_win = images_all[win_indices]   # (win_size, 3, H, W)
        masks_win = image_masks[win_indices]    # (win_size, 1, H, W)

        print(f"  Anchor {anchor} (frames {win_indices[0]}-{win_indices[-1]}), "
              f"query_frame_index={anchor_in_win}")

        with torch.inference_mode():
            win_tracks, win_vis, _, _, _ = predict_tracks(
                images_win,
                image_masks=masks_win,
                max_query_pts=args.vggsfm_max_query_pts,
                query_frame_num=1,
                keypoint_extractor=args.vggsfm_keypoint_extractor,
                max_points_num=args.vggsfm_max_points_num,
                fine_tracking=args.vggsfm_fine_tracking,
                complete_non_vis=False,
                query_frame_indexes=[anchor_in_win],
            )
        # win_tracks: (win_size, N_w, 2), win_vis: (win_size, N_w)
        n_w = win_tracks.shape[1]

        # Expand to global frame dimension, fill non-window frames with zeros
        global_tracks = np.zeros((num_frames, n_w, 2), dtype=win_tracks.dtype)
        global_vis = np.zeros((num_frames, n_w), dtype=win_vis.dtype)
        global_win_mask = np.zeros((num_frames, n_w), dtype=bool)

        for local_idx, global_idx in enumerate(win_indices):
            global_tracks[global_idx] = win_tracks[local_idx]
            global_vis[global_idx] = win_vis[local_idx]
            global_win_mask[global_idx] = True  # only valid within window

        all_tracks.append(global_tracks)
        all_vis_scores.append(global_vis)
        all_window_masks.append(global_win_mask)

    if not all_tracks:
        print("No tracks produced from any window.")
        return

    # Concatenate tracks from all windows along the track dimension
    pred_tracks = np.concatenate(all_tracks, axis=1)       # (num_frames, N_total, 2)
    pred_vis_scores = np.concatenate(all_vis_scores, axis=1)  # (num_frames, N_total)
    window_valid = np.concatenate(all_window_masks, axis=1)   # (num_frames, N_total)

    print(f"Total tracks from all windows: {pred_tracks.shape[1]}")

    # Check if each track point is in foreground (mask > 0)
    in_foreground = compute_vggsfm_foreground_mask(pred_tracks, image_paths, mask_dir)
    depth_dir = data_dir / "depth_filtered"
    # depth_valid = compute_vggsfm_depth_mask(pred_tracks, image_paths, depth_dir)

    vis_valid = pred_vis_scores > args.vggsfm_vis_thresh
    
    # Combine: must be in window, in foreground, have valid depth, and pass visibility threshold
    pred_tracks_mask = window_valid & in_foreground & vis_valid #& depth_valid

    print(f"Track validity stats: {pred_tracks_mask.sum()} / {pred_tracks_mask.size} valid track-frame pairs")
    min_tracks = 5
    num_tracks = (pred_tracks_mask.sum(axis=0) >= min_tracks).sum()
    print(f"Number of tracks with at least {min_tracks} valid observation: {num_tracks} / {pred_tracks.shape[1]}")

    # Save tracking results
    tracks_path = corres_dir / "vggsfm_tracks.npz"
    np.savez_compressed(
        tracks_path,
        tracks=pred_tracks,
        vis_scores=pred_vis_scores,
        tracks_mask=pred_tracks_mask,
        image_paths=[str(p) for p in image_paths],
    )
    print(f"Saved VGGSfM tracking results to {tracks_path}")

    print("Exporting VGGSfM correspondences...")
    vggsfm_pairs = [(cond_pos, j) for j in range(num_frames) if j != cond_pos]

    for i, j in tqdm(vggsfm_pairs, desc="VGGSfM matching"):
        image0_path = image_paths[i]
        image1_path = image_paths[j]
        # Use combined validity mask (visibility + foreground + window)
        keep = pred_tracks_mask[i] & pred_tracks_mask[j]
        if int(keep.sum()) == 0:
            print(f"No matches between {image0_path.name} and {image1_path.name}, skipping.")
            continue

        xy1_orig = pred_tracks[i][keep].astype(np.float32)
        xy2_orig = pred_tracks[j][keep].astype(np.float32)
        conf_np = np.minimum(pred_vis_scores[i][keep], pred_vis_scores[j][keep]).astype(np.float32)
        if len(conf_np) == 0:
            print("No matches after filtering, skipping.")
            continue

        name0 = image0_path.stem
        name1 = image1_path.stem

        pair_name = f"{name0}_{name1}"
        vis_path = corres_vis_dir / f"{pair_name}.png"
        _draw_correspondences(image0_path, image1_path, xy1_orig, xy2_orig, vis_path, args.max_vis_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get image correspondences using VGGSfM")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory from pipeline_data_preprocess.py")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for correspondences")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pair_mode", type=str, default="condition_to_all",
                        choices=["condition_to_all", "consecutive", "all"])
    parser.add_argument("--cond_index", type=int, default=0, help="Condition frame index used by condition_to_all mode")
    parser.add_argument("--vggsfm_vis_thresh", type=float, default=0.0, help="Minimum VGGSfM visibility score")
    parser.add_argument("--vggsfm_max_query_pts", type=int, default=512)
    parser.add_argument("--vggsfm_keypoint_extractor", type=str, default="aliked+sp")
    parser.add_argument("--vggsfm_max_points_num", type=int, default=163840)
    parser.add_argument("--vggsfm_fine_tracking", action="store_true", default=False)
    parser.add_argument("--max_vis_matches", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=41, help="Number of frames per sliding window")
    parser.add_argument("--anchor_interval", type=int, default=5, help="Interval between anchor frames")

    main(parser.parse_args())
