import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch


def _setup_paths() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "third_party" / "mast3r"))
    sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))
    return project_root


PROJECT_ROOT = _setup_paths()

from mast3r.fast_nn import extract_correspondences_nonsym
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import inference
from dust3r_visloc.datasets.utils import get_resize_function
from vggt.dependency.track_predict import predict_tracks
from robust_hoi_pipeline.pipeline_utils import (
    load_mask,
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


def _build_pair_indices(num_images: int, pair_mode: str, cond_pos: int) -> List[Tuple[int, int]]:
    if num_images < 2:
        return []
    if pair_mode == "consecutive":
        return [(i, i + 1) for i in range(num_images - 1)]
    if pair_mode == "all":
        return [(i, j) for i in range(num_images) for j in range(i + 1, num_images)]
    if pair_mode == "condition_to_all":
        return [(cond_pos, j) for j in range(num_images) if j != cond_pos]
    raise ValueError(f"Unknown pair_mode: {pair_mode}")


def _prepare_mast3r_image(image_path: Path, idx: int, maxdim: int, patch_size: int) -> Dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resize_func, _, to_orig = get_resize_function(maxdim, patch_size, height, width)
    rgb_tensor = resize_func(ImgNorm(image))
    return {
        "img": rgb_tensor.unsqueeze(0),
        "true_shape": np.int32([rgb_tensor.shape[1:]]),
        "to_orig": to_orig.astype(np.float32),
        "idx": idx,
        "instance": str(image_path),
        "orig_shape": np.int32([height, width]),
    }


def _to_original_coords(points_xy: np.ndarray, to_orig: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points_xy, np.ones((len(points_xy), 1), dtype=np.float32)], axis=1)
    out = (to_orig @ homo.T).T
    return out[:, :2]


def _filter_background_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pts0) == 0:
        return pts0, pts1, conf

    h0, w0 = mask0.shape[:2]
    h1, w1 = mask1.shape[:2]

    x0 = np.clip(np.round(pts0[:, 0]).astype(np.int32), 0, w0 - 1)
    y0 = np.clip(np.round(pts0[:, 1]).astype(np.int32), 0, h0 - 1)
    x1 = np.clip(np.round(pts1[:, 0]).astype(np.int32), 0, w1 - 1)
    y1 = np.clip(np.round(pts1[:, 1]).astype(np.int32), 0, h1 - 1)

    keep = (mask0[y0, x0] > 0) & (mask1[y1, x1] > 0)
    return pts0[keep], pts1[keep], conf[keep]


def _geometry_verify_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    reproj_thresh: float,
    confidence: float,
    max_iters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(pts0) < 8:
        return pts0, pts1, conf, int(len(pts0))

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
    _, inlier_mask = cv2.findFundamentalMat(
        pts0,
        pts1,
        method=method,
        ransacReprojThreshold=reproj_thresh,
        confidence=confidence,
        maxIters=max_iters,
    )
    if inlier_mask is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    keep = inlier_mask.reshape(-1).astype(bool)
    return pts0[keep], pts1[keep], conf[keep], int(keep.sum())


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
    pair_indices = _build_pair_indices(len(image_paths), args.pair_mode, cond_pos)
    if not pair_indices:
        print("No image pairs to process.")
        return

    pair_meta = []
    total_matches = 0
    saved_pairs = 0

    if args.matching_backend == "mast3r":
        print(f"Loading MASt3R model: {args.model_name}")
        model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)
        model.eval()
        image_dicts = [
            _prepare_mast3r_image(path, idx=i, maxdim=args.max_dim, patch_size=args.patch_size)
            for i, path in enumerate(image_paths)
        ]
        pairs = [(image_dicts[i], image_dicts[j]) for i, j in pair_indices]

        for start in tqdm(range(0, len(pairs), args.pair_batch_size), desc="MASt3R matching"):
            chunk_pairs = pairs[start:start + args.pair_batch_size]
            output = inference(chunk_pairs, model, args.device, batch_size=1, verbose=False)
            pred1 = output["pred1"]
            pred2 = output["pred2"]

            for k, pair in enumerate(chunk_pairs):
                desc1 = pred1["desc"][k]
                desc2 = pred2["desc"][k]
                conf1 = pred1["desc_conf"][k]
                conf2 = pred2["desc_conf"][k]

                xy1, xy2, conf = extract_correspondences_nonsym(
                    desc1,
                    desc2,
                    conf1,
                    conf2,
                    subsample=args.subsample,
                    device=args.device,
                    pixel_tol=args.pixel_tol,
                )

                keep = conf >= args.conf_thr
                if int(keep.sum()) == 0:
                    continue

                xy1_np = xy1[keep].detach().cpu().numpy().astype(np.float32)
                xy2_np = xy2[keep].detach().cpu().numpy().astype(np.float32)
                conf_np = conf[keep].detach().cpu().numpy().astype(np.float32)

                xy1_orig = _to_original_coords(xy1_np, pair[0]["to_orig"])
                xy2_orig = _to_original_coords(xy2_np, pair[1]["to_orig"])
                image0_path = Path(pair[0]["instance"])
                image1_path = Path(pair[1]["instance"])

                # Remove correspondences falling on object-mask background (mask value == 0).
                name0 = image0_path.stem
                name1 = image1_path.stem
                mask0_path = mask_dir / f"{name0}.png"
                mask1_path = mask_dir / f"{name1}.png"
                mask0 = load_mask(mask0_path)
                mask1 = load_mask(mask1_path)
                xy1_orig, xy2_orig, conf_np = _filter_background_matches(
                    xy1_orig, xy2_orig, conf_np, mask0, mask1
                )
                if len(conf_np) == 0:
                    continue

                geom_inliers = None
                geom_inlier_ratio = None
                if args.mast3r_geom_verify:
                    num_before_geom = len(conf_np)
                    xy1_orig, xy2_orig, conf_np, geom_inliers = _geometry_verify_matches(
                        xy1_orig,
                        xy2_orig,
                        conf_np,
                        reproj_thresh=args.mast3r_geom_reproj_thresh,
                        confidence=args.mast3r_geom_confidence,
                        max_iters=args.mast3r_geom_max_iters,
                    )
                    geom_inlier_ratio = float(geom_inliers) / max(num_before_geom, 1)
                    if geom_inlier_ratio < args.mast3r_min_inlier_ratio:
                        continue
                    if len(conf_np) == 0:
                        continue

                pair_name = f"{name0}_{name1}"
                corres_path = corres_dir / f"{pair_name}.npz"
                vis_path = corres_vis_dir / f"{pair_name}.png"

                np.savez_compressed(
                    corres_path,
                    image0=str(image0_path),
                    image1=str(image1_path),
                    points0=xy1_orig,
                    points1=xy2_orig,
                    confidence=conf_np,
                )
                _draw_correspondences(image0_path, image1_path, xy1_orig, xy2_orig, vis_path, args.max_vis_matches)

                num_match = int(len(conf_np))
                total_matches += num_match
                saved_pairs += 1
                pair_meta.append(
                    {
                        "pair_name": pair_name,
                        "image0": str(image0_path),
                        "image1": str(image1_path),
                        "num_matches": num_match,
                        "num_geom_inliers": geom_inliers,
                        "geom_inlier_ratio": geom_inlier_ratio,
                        "corres_file": str(corres_path.relative_to(out_dir)),
                        "vis_file": str(vis_path.relative_to(out_dir)),
                    }
                )
    else:
        if args.pair_mode != "condition_to_all":
            print("VGGSfM backend uses one query (condition image). Overriding pair_mode to condition_to_all.")
        vggsfm_pairs = [(cond_pos, j) for j in range(len(image_paths)) if j != cond_pos]
        print("Running VGGSfM tracking once on all images (with masked background)...")
        images_all, image_masks = _load_vggsfm_sequence_images(image_paths, args.device, mask_dir)
        with torch.inference_mode():
            pred_tracks, pred_vis_scores, _, _, _ = predict_tracks(
                images_all,
                image_masks=image_masks,
                max_query_pts=args.vggsfm_max_query_pts,
                query_frame_num=1,
                keypoint_extractor=args.vggsfm_keypoint_extractor,
                max_points_num=args.vggsfm_max_points_num,
                fine_tracking=args.vggsfm_fine_tracking,
                complete_non_vis=True,
                query_frame_indexes=[cond_pos],
            )


        # Check if each track point is in foreground (mask > 0)
        in_foreground = compute_vggsfm_foreground_mask(pred_tracks, image_paths, mask_dir)
        depth_dir = data_dir / "depth_filtered"
        depth_valid = compute_vggsfm_depth_mask(pred_tracks, image_paths, depth_dir)

        vis_valid = pred_vis_scores > args.vggsfm_vis_thresh
        pred_tracks_mask = in_foreground & depth_valid & vis_valid

        print(f"Track validity stats: {pred_tracks_mask.sum()} / {pred_tracks_mask.size} valid track-frame pairs")
        # print track number
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
        
        for i, j in tqdm(vggsfm_pairs, desc="VGGSfM matching"):
            image0_path = image_paths[i]
            image1_path = image_paths[j]
            # Use combined validity mask (visibility + foreground)
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
    parser = argparse.ArgumentParser(description="Get image correspondences using MASt3R")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory from pipeline_data_preprocess.py")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for correspondences")
    parser.add_argument("--model_name", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--matching_backend", type=str, default="mast3r", choices=["mast3r", "vggsfm"])
    parser.add_argument("--pair_mode", type=str, default="condition_to_all",
                        choices=["condition_to_all", "consecutive", "all"])
    parser.add_argument("--cond_index", type=int, default=0, help="Condition frame index used by condition_to_all mode")
    parser.add_argument("--pair_batch_size", type=int, default=2, help="Number of image pairs per inference chunk")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--pixel_tol", type=int, default=0)
    parser.add_argument("--conf_thr", type=float, default=0.8, help="Minimum MASt3R correspondence confidence")
    parser.add_argument("--mast3r_geom_verify", action="store_true", default=True,
                        help="Enable geometric verification for MASt3R matches using F-matrix RANSAC")
    parser.add_argument("--mast3r_geom_reproj_thresh", type=float, default=1.5,
                        help="RANSAC reprojection threshold (pixels) for MASt3R geometric verification")
    parser.add_argument("--mast3r_geom_confidence", type=float, default=0.999,
                        help="RANSAC confidence for MASt3R geometric verification")
    parser.add_argument("--mast3r_geom_max_iters", type=int, default=10000,
                        help="Maximum RANSAC iterations for MASt3R geometric verification")
    parser.add_argument("--mast3r_min_inlier_ratio", type=float, default=0.3,
                        help="Minimum inlier ratio after MASt3R geometric verification; pairs below this are dropped")
    parser.add_argument("--vggsfm_vis_thresh", type=float, default=0.3, help="Minimum VGGSfM visibility score")
    parser.add_argument("--vggsfm_max_query_pts", type=int, default=2048)
    parser.add_argument("--vggsfm_keypoint_extractor", type=str, default="aliked+sp")
    parser.add_argument("--vggsfm_max_points_num", type=int, default=163840)
    parser.add_argument("--vggsfm_fine_tracking", action="store_true", default=False)
    parser.add_argument("--max_vis_matches", type=int, default=200)

    main(parser.parse_args())
