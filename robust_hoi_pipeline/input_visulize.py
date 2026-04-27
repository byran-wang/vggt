import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from matplotlib import cm
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt_eval_vis_nvdiffrast import (
    ensure_sealed_right_hand_mesh,
)
from robust_hoi_pipeline.pipeline_utils import load_intrinsics_from_meta
from utils_simba.depth import get_depth
from utils_simba.eval_vis import ensure_cuda_available, normalize_intrinsics
from utils_simba.logger import get_logger
from utils_simba.render import make_mesh_tensors, nvdiffrast_render

import nvdiffrast.torch as dr

try:
    from viewer.viewer_step import HandDataProvider
except Exception:
    HandDataProvider = None

logger = get_logger(__name__)

_OBJ_COLOR = np.array([144, 210, 236], dtype=np.float32)   # light blue
_HAND_COLOR = np.array([200, 180, 220], dtype=np.float32)  # light purple
_WHITE = np.array([255.0, 255.0, 255.0], dtype=np.float32)


def _load_inputs_from_data_dir(data_dir: Path, frame_idx: int) -> dict:
    """Load RGB / masks / depth / intrinsics directly from the raw data_dir layout
    (mirrors :func:`pipeline_data_preprocess.pipeline_data_preprocess`)."""
    image_path = data_dir / "rgb" / f"{frame_idx:04d}.jpg"
    if not image_path.exists():
        image_path = data_dir / "rgb" / f"{frame_idx:04d}.png"
    image = np.array(Image.open(image_path).convert("RGB")) if image_path.exists() else None

    mask_obj_path = data_dir / "mask_object" / f"{frame_idx:04d}.png"
    mask_obj = np.array(Image.open(mask_obj_path).convert("L")) if mask_obj_path.exists() else None

    mask_hand_path = data_dir / "mask_hand" / f"{frame_idx:04d}.png"
    mask_hand = np.array(Image.open(mask_hand_path).convert("L")) if mask_hand_path.exists() else None

    depth_path = data_dir / "depth" / f"{frame_idx:04d}.png"
    depth = get_depth(str(depth_path)) if depth_path.exists() else None

    meta_path = data_dir / "meta" / f"{frame_idx:04d}.pkl"
    intrinsics = load_intrinsics_from_meta(str(meta_path)) if meta_path.exists() else None

    return {
        "image": image,
        "mask_obj": mask_obj,
        "mask_hand": mask_hand,
        "depth": depth,
        "intrinsics": intrinsics,
    }




def _merge_masks_rgba(shape_hw, layers) -> np.ndarray:
    """Merge masks into an RGBA image with transparent background.

    ``layers`` is a list of ``(mask, color)`` tuples; later layers overwrite
    earlier ones on overlap. Output: (H, W, 4) uint8, alpha=0 where no mask, 255 elsewhere."""
    h, w = shape_hw
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for mask, color in layers:
        if mask is None:
            continue
        mask_bool = np.asarray(mask) > 127 if np.asarray(mask).dtype != bool else np.asarray(mask)
        out[mask_bool, :3] = color.astype(np.uint8)
        out[mask_bool, 3] = 255
    return out


def _normalize_depth(depth: np.ndarray, valid: np.ndarray,
                     pct_low: float, pct_high: float, gamma: float) -> np.ndarray:
    vmin = float(np.percentile(depth[valid], pct_low))
    vmax = float(np.percentile(depth[valid], pct_high))
    norm = np.zeros_like(depth, dtype=np.float32)
    if vmax > vmin:
        norm[valid] = ((depth[valid] - vmin) / (vmax - vmin)).clip(0.0, 1.0)
        norm[valid] = norm[valid] ** float(gamma)
    return norm


def _depth_colormap(depth: np.ndarray, cmap_name: str = "coolwarm",
                    pct_low: float = 2.0, pct_high: float = 98.0,
                    gamma: float = 0.7) -> np.ndarray:
    """Blue→red gradient depth colormap. Invalid depth (<=0) is rendered black.

    - Clips dynamic range to ``[pct_low, pct_high]`` percentiles so outliers
      don't squash the visible range.
    - ``coolwarm`` (default) gives a clean blue→red gradient (blue=near, red=far).
    - ``gamma`` < 1.0 expands mid-range contrast."""
    h, w = depth.shape[:2]
    valid = depth > 0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)  # black bg for invalid pixels
    if not valid.any():
        return rgb
    norm = _normalize_depth(depth, valid, pct_low, pct_high, gamma)
    rgba = cm.get_cmap(cmap_name)(norm)
    rgb[valid] = (rgba[valid, :3] * 255).astype(np.uint8)
    return rgb


def _depth_graymap(depth: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0,
                   gamma: float = 0.7) -> np.ndarray:
    """Grayscale depth map: white=near, black=far. Invalid depth (<=0) is also black."""
    h, w = depth.shape[:2]
    valid = depth > 0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if not valid.any():
        return rgb
    norm = _normalize_depth(depth, valid, pct_low, pct_high, gamma)
    intensity = ((1.0 - norm) * 255.0).clip(0, 255).astype(np.uint8)
    rgb[valid] = intensity[valid, None]
    return rgb




def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = int(args.frame_index)

    data = _load_inputs_from_data_dir(data_dir, frame_idx)
    image = data.get("image")
    if image is None:
        raise FileNotFoundError(f"No RGB image for frame {frame_idx} in {data_dir}/rgb")

    mask_rgba = _merge_masks_rgba(
        image.shape[:2],
        [(data.get("mask_obj"), _OBJ_COLOR), (data.get("mask_hand"), _HAND_COLOR)],
    )
    mask_path = out_dir / f"{frame_idx:04d}_mask.png"
    Image.fromarray(mask_rgba, mode="RGBA").save(mask_path)
    logger.info(f"Saved merged masks (RGBA) -> {mask_path}")

    depth = data.get("depth")
    if depth is not None:
        depth_arr = np.asarray(depth, dtype=np.float32)
        depth_color_path = out_dir / f"{frame_idx:04d}_depth.png"
        Image.fromarray(_depth_colormap(depth_arr)).save(depth_color_path)
        logger.info(f"Saved depth colormap -> {depth_color_path}")
        depth_gray_path = out_dir / f"{frame_idx:04d}_depth_gray.png"
        Image.fromarray(_depth_graymap(depth_arr)).save(depth_gray_path)
        logger.info(f"Saved depth graymap -> {depth_gray_path}")
    else:
        logger.warning(f"No depth for frame {frame_idx}")



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-frame inputs: RGB with hand/object mask overlays, depth colormap, "
            "and the hand mesh (cam space) rendered and overlaid on the RGB."
        ),
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Raw sequence dir with rgb/, mask_object/, mask_hand/, depth/, meta/, hands/ "
                             "(same layout consumed by pipeline_data_preprocess.py)")
    parser.add_argument("--frame_index", type=int, required=True, help="Frame index to visualize")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Background fade alpha (0=keep RGB, 1=fully white) for the masked overlays")

    parser.add_argument("--hand_mode", type=str, default="trans",
                        choices=["trans", "intrinsic", "rot"],
                        help="HandDataProvider mode (matches pipeline_data_preprocess.py --hand_mode)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
