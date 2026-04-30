import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "../" / "third_party" / "utils_simba"))
sys.path.insert(0, str(project_root / "../" / "dependency" / "LightGlue"))

from utils_simba.depth import depth2xyzmap, get_depth, load_filtered_depth
from utils_simba.logger import get_logger
from lightglue import SuperPoint

sys.path.insert(0, str(project_root / ".."))
from robust_hoi_pipeline.pipeline_data_preprocess import prepare_image_list

logger = get_logger(__name__)


def _load_intrinsics(meta_path: str) -> np.ndarray:
    """Load camera intrinsics from meta pickle file."""
    import io

    class _NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with open(meta_path, "rb") as f:
        try:
            meta = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            meta = _NumpyCompatUnpickler(f).load()
    K = meta.get("intrinsics", meta.get("camMat"))
    return np.array(K, dtype=np.float32)


def _geometry_quality(depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> float:
    """Compute geometry quality via SVD of masked object point cloud.

    Returns the ratio of the smallest to largest singular value (0-1).
    A value close to 1 means the points are well-spread in 3D (good geometry);
    a value close to 0 means degenerate/planar/sparse.
    """
    xyz = depth2xyzmap(depth, K)  # (H, W, 3)
    valid = (mask > 0) & (depth > 0.01)
    pts = xyz[valid]  # (N, 3)
    if len(pts) < 10:
        return 0.0, np.zeros(3)
    pts_centered = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts_centered, full_matrices=False)
    if s[0] < 1e-8:
        return 0.0, s
    return float(s[2] / s[0]), s  # min / max singular value, singular values


def _count_feature_points(image: np.ndarray, mask: np.ndarray, extractor) -> int:
    """Count SuperPoint feature points within the object mask.

    Args:
        image: (H, W, 3) uint8 RGB image
        mask: (H, W) uint8 mask (>0 = object)
        extractor: SuperPoint model on device

    Returns:
        Number of keypoints inside the mask.
    """
    device = next(extractor.parameters()).device
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feats = extractor.extract(img_tensor)
    kpts = feats["keypoints"][0].cpu().numpy()  # (N, 2) x,y
    # Count keypoints inside mask
    kpts_int = kpts.astype(int)
    h, w = mask.shape
    in_bounds = (kpts_int[:, 0] >= 0) & (kpts_int[:, 0] < w) & (kpts_int[:, 1] >= 0) & (kpts_int[:, 1] < h)
    kpts_valid = kpts_int[in_bounds]
    in_mask = mask[kpts_valid[:, 1], kpts_valid[:, 0]] > 0
    return int(in_mask.sum())


def _save_frame_list(path, frame_indices):
    """Write a list of frame indices to a text file."""
    with open(path, "w") as f:
        for idx in frame_indices:
            f.write(f"{idx}\n")


def _filter_by_boundary(frame_indices, mask_obj_dir, border_px=5):
    """Filter out frames where the object mask touches the image boundary.

    Returns list of frames whose mask does not touch any edge.
    """
    kept = []
    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        mask_path = mask_obj_dir / f"{fid}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Object mask not found: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        edges_hit = []
        if (mask[:border_px, :] > 0).any():
            edges_hit.append("top")
        if (mask[h - border_px:, :] > 0).any():
            edges_hit.append("bottom")
        if (mask[:, :border_px] > 0).any():
            edges_hit.append("left")
        if (mask[:, w - border_px:] > 0).any():
            edges_hit.append("right")
        if edges_hit:
            logger.info(f"  Frame {fid}: mask touches boundary ({', '.join(edges_hit)}), dropping")
        else:
            kept.append(frame_idx)
    return kept


def _filter_by_dino_similarity(frame_indices, rgb_dir, mask_obj_dir, similarity_threshold=0.9, image_size=336):
    """Filter out visually similar frames using DINOv2 CLS token cosine similarity.

    Uses threshold-and-drop: sequentially scan frames, drop any frame whose
    max cosine similarity to already-kept frames exceeds the threshold.
    Returns list of kept frame indices.
    """
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    model = model.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Load and preprocess all frames into a batch
    tensors = []
    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        rgb_path = rgb_dir / f"{fid}.jpg"
        mask_path = mask_obj_dir / f"{fid}.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        image = cv2.imread(str(rgb_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image[mask == 0] = 0
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensors.append(img_tensor)

    images = torch.stack(tensors).to(device)  # (N, 3, H, W)
    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)
    images = (images - mean) / std

    # Extract CLS token features
    with torch.no_grad():
        feats = model(images, is_training=True)
    feat_norm = F.normalize(feats["x_norm_clstoken"], p=2, dim=1)  # (N, D)

    # Threshold-and-drop
    kept = []
    kept_indices = []
    for i, frame_idx in enumerate(frame_indices):
        if not kept_indices:
            kept.append(frame_idx)
            kept_indices.append(i)
            logger.info(f"  Frame {frame_idx:04d}: kept (first frame)")
            continue
        sims = (feat_norm[kept_indices] @ feat_norm[i]).cpu().numpy()  # (K,)
        max_sim = float(sims.max())
        most_similar_idx = kept[int(sims.argmax())]
        if max_sim > similarity_threshold:
            logger.info(f"  Frame {frame_idx:04d}: dropped (sim={max_sim:.3f} with frame {most_similar_idx:04d})")
        else:
            kept.append(frame_idx)
            kept_indices.append(i)
            logger.info(f"  Frame {frame_idx:04d}: kept (max_sim={max_sim:.3f})")
    return kept


def _filter_by_mask(frame_indices, mask_obj_dir, max_frames):
    """Filter frames by object mask pixel count, keep top max_frames.

    Returns (sorted frame list, mask_filtered set, pixel counts dict).
    """
    mask_pixel_counts = {}
    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        mask_path = mask_obj_dir / f"{fid}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Object mask not found: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_pixels = int((mask > 0).sum()) if mask is not None else 0
        mask_pixel_counts[frame_idx] = mask_pixels
        logger.info(f"  Frame {fid}: mask_pixels={mask_pixels}")

    sorted_by_mask = sorted(mask_pixel_counts.keys(), key=lambda i: mask_pixel_counts[i], reverse=True)
    filtered_ordered = sorted_by_mask[:max_frames]
    filtered_set = set(filtered_ordered)
    return filtered_ordered, filtered_set, mask_pixel_counts


def _filter_by_geometry(frame_indices, depth_dir, meta_dir, mask_obj_dir, max_frames):
    """Filter frames by geometry quality (SVD ratio), keep top max_frames.

    Returns sorted frame list.
    """
    geometry_qualities = {}
    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        depth_path = depth_dir / f"{fid}.png"
        meta_path = meta_dir / "0000.pkl"
        mask_path = mask_obj_dir / f"{fid}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Object mask not found: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        K = _load_intrinsics(str(meta_path))
        depth = load_filtered_depth(str(depth_path))
        quality, svs = _geometry_quality(depth, K, mask)
        geometry_qualities[frame_idx] = quality
        logger.info(f"  Frame {fid}: geometry_quality={quality:.4f}, singular_values=({svs[0]:.4f}, {svs[1]:.4f}, {svs[2]:.4f})")

    sorted_by_geom = sorted(geometry_qualities.keys(), key=lambda i: geometry_qualities[i], reverse=True)
    return sorted_by_geom[:max_frames]


def _filter_by_feature_points(frame_indices, rgb_dir, mask_obj_dir, max_frames):
    """Filter frames by SuperPoint feature point count, keep top max_frames.

    Returns sorted frame list.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    ftp_counts = {}
    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        rgb_path = rgb_dir / f"{fid}.jpg"
        mask_path = mask_obj_dir / f"{fid}.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Object mask not found: {mask_path}")
        image = cv2.imread(str(rgb_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        count = _count_feature_points(image, mask, extractor)
        ftp_counts[frame_idx] = count
        logger.info(f"  Frame {fid}: feature_points={count}")

    sorted_by_ftp = sorted(ftp_counts.keys(), key=lambda i: ftp_counts[i], reverse=True)
    return sorted_by_ftp[:max_frames]


def main(args):
    rgb_dir = Path(f"{args.dataset_dir}/{args.scene_name}/rgb")
    depth_dir = Path(f"{args.dataset_dir}/{args.scene_name}/depth")
    meta_dir = Path(f"{args.dataset_dir}/{args.scene_name}/meta")
    mask_obj_dir = Path(f"{args.dataset_dir}/{args.scene_name}/mask_object")
    out_dir = Path(f"{args.dataset_dir}/{args.scene_name}/SAM3D")

    # Set up file logging
    log_path = out_dir / "filter_2D.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_path}")

    # Load frame index list
    frame_indices = prepare_image_list(
        image_dir=rgb_dir,
        start=args.frame_start,
        end=args.frame_end,
        interval=args.frame_interval,
        cond_index=args.cond_idx,
    )
    logger.info(f"Loaded {len(frame_indices)} frames from {rgb_dir}")

    # Stage 1: Filter by mask pixel count
    mask_filtered_ordered, mask_filtered_set, _ = _filter_by_mask(
        frame_indices, mask_obj_dir, args.max_mask_filtered_frames,
    )
    _save_frame_list(out_dir / "frame_list_after_mask_filtered.txt", mask_filtered_ordered)
    logger.info(f"Saved mask-filtered frame list ({len(mask_filtered_ordered)} frames)")

    # Stage 2: Filter by boundary
    boundary_filtered = _filter_by_boundary(
        mask_filtered_ordered, mask_obj_dir, border_px=args.border_px,
    )
    _save_frame_list(out_dir / "frame_list_after_boundary_filtered.txt", boundary_filtered)
    logger.info(f"Saved boundary-filtered frame list ({len(boundary_filtered)} frames)")

    # Stage 3: Filter by DINO similarity
    dino_filtered = _filter_by_dino_similarity(
        boundary_filtered, rgb_dir, mask_obj_dir, similarity_threshold=args.dino_similarity_threshold,
    )
    _save_frame_list(out_dir / "frame_list_after_dino_filtered.txt", dino_filtered)
    logger.info(f"Saved DINO-filtered frame list ({len(dino_filtered)} frames)")

    # Stage 4: Filter by geometry quality
    depth_filtered_ordered = _filter_by_geometry(
        dino_filtered, depth_dir, meta_dir, mask_obj_dir, args.max_geometry_filtered_frames,
    )
    _save_frame_list(out_dir / "frame_list_after_depth_filtered.txt", depth_filtered_ordered)
    logger.info(f"Saved depth-filtered frame list ({len(depth_filtered_ordered)} frames)")

    # Stage 5: Filter by feature point count
    ftp_filtered_ordered = _filter_by_feature_points(
        depth_filtered_ordered, rgb_dir, mask_obj_dir, args.max_ftp_filter_frames,
    )
    _save_frame_list(out_dir / "frame_list_after_ftp_filtered.txt", ftp_filtered_ordered)
    logger.info(f"Saved feature-point-filtered frame list ({len(ftp_filtered_ordered)} frames)")

    # Keep only frames that pass all filters, preserving feature-point order
    filtered = [i for i in ftp_filtered_ordered if i in mask_filtered_set]

    # Ensure condition frame is always included
    if args.cond_idx not in filtered:
        logger.info(f"Condition frame {args.cond_idx:04d} not in filtered list, prepending it")
        filtered = filtered + [args.cond_idx]

    _save_frame_list(out_dir / "frame_list_filtered.txt", filtered)
    logger.info(f"Saved final filtered frame list ({len(filtered)} frames)")

    logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter frames by mask size and geometry quality for SAM3D")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frame_end", type=int, default=-1, help="End frame index (-1 for all)")
    parser.add_argument("--frame_interval", type=int, default=5, help="Frame sampling interval")
    parser.add_argument("--cond_idx", type=int, default=0, help="Condition frame index (always included)")
    parser.add_argument("--max_mask_filtered_frames", type=int, default=100, help="Max frames to keep after mask filtering")
    parser.add_argument("--border_px", type=int, default=5, help="Border pixel width for boundary filter")
    parser.add_argument("--dino_similarity_threshold", type=float, default=0.96, help="Cosine similarity threshold for DINO dedup")
    parser.add_argument("--max_geometry_filtered_frames", type=int, default=30, help="Max frames to keep after geometry filtering")
    parser.add_argument("--max_ftp_filter_frames", type=int, default=20, help="Max frames to keep after feature point filtering")

    args = parser.parse_args()
    main(args)
