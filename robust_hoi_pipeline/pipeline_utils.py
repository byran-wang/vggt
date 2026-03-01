import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image
import json

from utils_simba.depth import get_depth, get_normal


def load_frame_list(data_preprocess_dir: Path) -> List[int]:
    """Load frame list from preprocessed data directory."""
    frame_list_path = data_preprocess_dir / "frame_list.txt"
    if not frame_list_path.exists():
        raise FileNotFoundError(f"Frame list not found: {frame_list_path}")

    frames = []
    with open(frame_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(int(line))
    return frames


def load_preprocessed_frame(data_preprocess_dir: Path, frame_idx: int) -> Dict:
    """Load preprocessed data for a single frame."""
    data: Dict = {}

    rgb_path = data_preprocess_dir / "rgb" / f"{frame_idx:04d}.png"
    data["image"] = np.array(Image.open(rgb_path).convert("RGB")) if rgb_path.exists() else None

    mask_obj_path = data_preprocess_dir / "mask_obj" / f"{frame_idx:04d}.png"
    data["mask_obj"] = np.array(Image.open(mask_obj_path).convert("L")) if mask_obj_path.exists() else None

    mask_hand_path = data_preprocess_dir / "mask_hand" / f"{frame_idx:04d}.png"
    data["mask_hand"] = np.array(Image.open(mask_hand_path).convert("L")) if mask_hand_path.exists() else None

    depth_path = data_preprocess_dir / "depth_filtered" / f"{frame_idx:04d}.png"
    data["depth"] = get_depth(str(depth_path)) if depth_path.exists() else None

    normal_path = data_preprocess_dir / "normal" / f"{frame_idx:04d}.png"
    data["normal"] = get_normal(str(normal_path)) if normal_path.exists() else None

    meta_path = data_preprocess_dir / "meta" / f"{frame_idx:04d}.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        data["intrinsics"] = meta.get("intrinsics")
        data["hand_pose"] = meta.get("hand_pose")
    else:
        data["intrinsics"] = None
        data["hand_pose"] = None

    return data


def _normalize_intrinsics_array(raw_value) -> np.ndarray:
    """Convert metadata intrinsics to a strict (3, 3) float32 matrix."""
    K = np.asarray(raw_value, dtype=np.float32)

    if K.ndim == 0:
        K = np.asarray(K.item(), dtype=np.float32)

    if K.shape == (1, 3, 3):
        K = K[0]
    elif K.shape == (9,):
        K = K.reshape(3, 3)
    elif K.shape == (4,):
        fx, fy, cx, cy = K.tolist()
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    if K.shape != (3, 3):
        raise ValueError(f"Invalid intrinsics shape: {K.shape} (expected (3, 3))")
    return K


def load_intrinsics_from_meta(meta_file: str) -> np.ndarray:
    """Load camera intrinsics from a meta pickle file."""
    with open(meta_file, "rb") as f:
        meta_data = pickle.load(f)

    if isinstance(meta_data, dict):
        for key in ("camMat", "intrinsics", "K", "camera_matrix"):
            if key in meta_data:
                return _normalize_intrinsics_array(meta_data[key])
        raise KeyError(f"No intrinsics key found. Available keys: {list(meta_data.keys())}")

    return _normalize_intrinsics_array(meta_data)


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return mask


def compute_vggsfm_foreground_mask(
    pred_tracks: np.ndarray,
    image_paths: Sequence[Path],
    mask_dir: Path,
) -> np.ndarray:
    """Return per-frame/per-track foreground validity from object masks."""
    num_frames, num_tracks = pred_tracks.shape[:2]
    in_foreground = np.ones((num_frames, num_tracks), dtype=bool)

    for frame_idx, img_path in enumerate(image_paths):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape[:2]
        track_coords = pred_tracks[frame_idx]
        x_coords = np.clip(np.round(track_coords[:, 0]).astype(np.int32), 0, w - 1)
        y_coords = np.clip(np.round(track_coords[:, 1]).astype(np.int32), 0, h - 1)
        in_foreground[frame_idx] = mask[y_coords, x_coords] > 0

    return in_foreground


def compute_vggsfm_depth_mask(
    pred_tracks: np.ndarray,
    image_paths: Sequence[Path],
    depth_dir: Path,
) -> np.ndarray:
    """Return per-frame/per-track validity from depth > 0."""
    num_frames, num_tracks = pred_tracks.shape[:2]
    depth_valid = np.ones((num_frames, num_tracks), dtype=bool)

    for frame_idx, img_path in enumerate(image_paths):
        depth_path = depth_dir / f"{img_path.stem}.png"
        if not depth_path.exists():
            continue
        depth = get_depth(str(depth_path))
        h, w = depth.shape[:2]
        track_coords = pred_tracks[frame_idx]
        x_coords = np.clip(np.round(track_coords[:, 0]).astype(np.int32), 0, w - 1)
        y_coords = np.clip(np.round(track_coords[:, 1]).astype(np.int32), 0, h - 1)
        depth_valid[frame_idx] = depth[y_coords, x_coords] > 0

    return depth_valid


def load_sam3d_transform(sam3d_dir: Path, cond_idx: int) -> Dict:
    """Load transformation from SAM3D post-processing.

    Args:
        sam3d_dir: Path to SAM3D_aligned_post_process directory
        cond_idx: Condition frame index

    Returns:
        Dictionary containing:
        - scale: scalar scale factor
        - rotation: (3, 3) rotation matrix
        - translation: (3,) translation vector
        - matrix: (4, 4) full transformation matrix
        - obj2cam: (4, 4) object-to-camera transformation
        - cam2obj: (4, 4) camera-to-object transformation
    """
    transform_path = sam3d_dir / f"{cond_idx:04d}" / "aligned_transform.json"
    # transform_path = sam3d_dir / "../SAM3D" / f"{cond_idx:04d}" / "camera.json"
    if transform_path == sam3d_dir / f"{cond_idx:04d}" / "aligned_transform.json":
        if not transform_path.exists():
            raise FileNotFoundError(f"SAM3D transform not found: {transform_path}")        
        with open(transform_path, "r") as f:
            transform_data = json.load(f)

        sam3d_to_cam_scale = float(transform_data["scale"])
        sam3d_to_cond_cam = np.array(transform_data["matrix"], dtype=np.float32)  # (4, 4)
        cond_cam_to_sam3d = np.linalg.inv(sam3d_to_cond_cam)
    elif transform_path == sam3d_dir / "../SAM3D" / f"{cond_idx:04d}" / "camera.json":
        if not transform_path.exists():
            raise FileNotFoundError(f"SAM3D transform not found: {transform_path}")
        with open(transform_path, "r") as f:
            transform_data = json.load(f)
            sam3d_to_cond_cam = np.array(transform_data["blw2cvc"], dtype=np.float32)  # (4, 4)
            cond_cam_to_sam3d = np.linalg.inv(sam3d_to_cond_cam)
            # get the scale from the rotation part of the camera_to_world matrix
            sam3d_to_cam_scale = np.linalg.norm(sam3d_to_cond_cam[:3, 0])
        


    return {
        'scale': sam3d_to_cam_scale,
        'sam3d_to_cond_cam': sam3d_to_cond_cam,
        "cond_cam_to_sam3d": cond_cam_to_sam3d,
    }