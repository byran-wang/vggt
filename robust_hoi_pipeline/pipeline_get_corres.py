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
    return project_root


PROJECT_ROOT = _setup_paths()

from mast3r.fast_nn import extract_correspondences_nonsym
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import inference
from dust3r_visloc.datasets.utils import get_resize_function


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


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return mask


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

    print(f"Loading MASt3R model: {args.model_name}")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)
    model.eval()

    image_dicts = [
        _prepare_mast3r_image(path, idx=i, maxdim=args.max_dim, patch_size=args.patch_size)
        for i, path in enumerate(image_paths)
    ]

    pair_meta = []
    total_matches = 0
    saved_pairs = 0
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

            # Remove correspondences falling on object-mask background (mask value == 0).
            name0 = Path(pair[0]["instance"]).stem
            name1 = Path(pair[1]["instance"]).stem
            mask0_path = mask_dir / f"{name0}.png"
            mask1_path = mask_dir / f"{name1}.png"
            mask0 = _load_mask(mask0_path)
            mask1 = _load_mask(mask1_path)
            xy1_orig, xy2_orig, conf_np = _filter_background_matches(
                xy1_orig, xy2_orig, conf_np, mask0, mask1
            )
            if len(conf_np) == 0:
                continue

            pair_name = f"{name0}_{name1}"
            corres_path = corres_dir / f"{pair_name}.npz"
            vis_path = corres_vis_dir / f"{pair_name}.png"

            np.savez_compressed(
                corres_path,
                image0=str(pair[0]["instance"]),
                image1=str(pair[1]["instance"]),
                points0=xy1_orig,
                points1=xy2_orig,
                confidence=conf_np,
            )
            _draw_correspondences(Path(pair[0]["instance"]), Path(pair[1]["instance"]), xy1_orig, xy2_orig, vis_path, args.max_vis_matches)

            num_match = int(len(conf_np))
            total_matches += num_match
            saved_pairs += 1
            pair_meta.append(
                {
                    "pair_name": pair_name,
                    "image0": str(pair[0]["instance"]),
                    "image1": str(pair[1]["instance"]),
                    "num_matches": num_match,
                    "corres_file": str(corres_path.relative_to(out_dir)),
                    "vis_file": str(vis_path.relative_to(out_dir)),
                }
            )

    summary = {
        "data_dir": str(data_dir),
        "num_images": len(image_paths),
        "pair_mode": args.pair_mode,
        "num_candidate_pairs": len(pair_indices),
        "num_saved_pairs": saved_pairs,
        "total_matches": total_matches,
        "mean_matches_per_saved_pair": (float(total_matches) / max(saved_pairs, 1)),
        "pairs": pair_meta,
    }
    summary_path = out_dir / "correspondences_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved correspondences for {saved_pairs}/{len(pair_indices)} pairs to {corres_dir}")
    print(f"Saved visualizations to {corres_vis_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get image correspondences using MASt3R")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory from pipeline_data_preprocess.py")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for correspondences")
    parser.add_argument("--model_name", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--pair_mode", type=str, default="condition_to_all",
                        choices=["condition_to_all", "consecutive", "all"])
    parser.add_argument("--cond_index", type=int, default=0, help="Condition frame index used by condition_to_all mode")
    parser.add_argument("--pair_batch_size", type=int, default=2, help="Number of image pairs per inference chunk")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--pixel_tol", type=int, default=0)
    parser.add_argument("--conf_thr", type=float, default=0.3, help="Minimum MASt3R correspondence confidence")
    parser.add_argument("--max_vis_matches", type=int, default=200)

    main(parser.parse_args())
