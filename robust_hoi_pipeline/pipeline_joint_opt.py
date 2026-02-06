import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import get_depth, depth2xyzmap


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


def load_preprocessed_data(data_preprocess_dir: Path, frame_indices: List[int]) -> Dict:
    """Load preprocessed data from pipeline_data_preprocess.py output.

    Args:
        data_preprocess_dir: Path to preprocessed data directory
        frame_indices: List of frame indices to load

    Returns:
        Dictionary containing:
        - images: list of (H, W, 3) RGB images
        - masks_obj: list of (H, W) object masks
        - masks_hand: list of (H, W) hand masks
        - depths: list of (H, W) filtered depth maps (in object space)
        - normals: list of (H, W, 3) normal maps
        - intrinsics: list of (3, 3) camera intrinsic matrices
        - hand_poses: list of hand pose data (or None)
    """
    from PIL import Image
    from utils_simba.depth import get_depth, get_normal

    data = {
        'frame_indices': frame_indices,
        'images': [],
        'masks_obj': [],
        'masks_hand': [],
        'depths': [],
        'normals': [],
        'intrinsics': [],
        'hand_poses': [],
    }

    for frame_idx in frame_indices:
        # Load RGB image
        rgb_path = data_preprocess_dir / "rgb" / f"{frame_idx:04d}.png"
        if rgb_path.exists():
            img = np.array(Image.open(rgb_path).convert("RGB"))
            data['images'].append(img)
        else:
            data['images'].append(None)

        # Load object mask
        mask_obj_path = data_preprocess_dir / "mask_obj" / f"{frame_idx:04d}.png"
        if mask_obj_path.exists():
            mask = np.array(Image.open(mask_obj_path).convert("L"))
            data['masks_obj'].append(mask)
        else:
            data['masks_obj'].append(None)

        # Load hand mask
        mask_hand_path = data_preprocess_dir / "mask_hand" / f"{frame_idx:04d}.png"
        if mask_hand_path.exists():
            mask = np.array(Image.open(mask_hand_path).convert("L"))
            data['masks_hand'].append(mask)
        else:
            data['masks_hand'].append(None)

        # Load filtered depth (already in object space from preprocessing)
        depth_path = data_preprocess_dir / "depth_filtered" / f"{frame_idx:04d}.png"
        if depth_path.exists():
            depth = get_depth(str(depth_path))
            data['depths'].append(depth)
        else:
            data['depths'].append(None)

        # Load normal map
        normal_path = data_preprocess_dir / "normal" / f"{frame_idx:04d}.png"
        if normal_path.exists():
            normal = get_normal(str(normal_path))
            data['normals'].append(normal)
        else:
            data['normals'].append(None)

        # Load metadata (intrinsics + hand pose)
        meta_path = data_preprocess_dir / "meta" / f"{frame_idx:04d}.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            data['intrinsics'].append(meta.get('intrinsics'))
            data['hand_poses'].append(meta.get('hand_pose'))
        else:
            data['intrinsics'].append(None)
            data['hand_poses'].append(None)

    return data


def load_tracks(tracks_dir: Path) -> Dict:
    """Load VGGSfM tracking results from pipeline_get_corres.py output.

    Args:
        tracks_dir: Path to correspondence output directory

    Returns:
        Dictionary containing:
        - tracks: (S, N, 2) predicted track coordinates
        - vis_scores: (S, N) visibility scores
        - tracks_mask: (S, N) combined validity mask (visibility + foreground)
        - image_paths: list of image path strings
    """
    tracks_path = tracks_dir / "corres" / "vggsfm_tracks.npz"
    if not tracks_path.exists():
        raise FileNotFoundError(f"VGGSfM tracks not found: {tracks_path}")

    data = np.load(tracks_path, allow_pickle=True)
    return {
        'tracks': data['tracks'],  # (S, N, 2)
        'vis_scores': data['vis_scores'],  # (S, N)
        'tracks_mask': data['tracks_mask'],  # (S, N)
        'image_paths': list(data['image_paths']),
    }


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
    if not transform_path.exists():
        raise FileNotFoundError(f"SAM3D transform not found: {transform_path}")

    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    rotation = np.array(transform_data["rotation"], dtype=np.float32)  # (3, 3)
    translation = np.array(transform_data["translation"], dtype=np.float32)  # (3,)
    scale = float(transform_data["scale"])
    sam3d_to_cond_cam = np.array(transform_data["matrix"], dtype=np.float32)  # (4, 4)
    cond_cam_to_sam3d = np.linalg.inv(sam3d_to_cond_cam)


    return {
        'scale': scale,
        'rotation': rotation,
        'translation': translation,
        'sam3d_to_cond_cam': sam3d_to_cond_cam,
        "cond_cam_to_sam3d": cond_cam_to_sam3d,
    }


def lift_tracks_to_3d(
    tracks: np.ndarray,
    tracks_mask: np.ndarray,
    depths: List[np.ndarray],
    intrinsics: List[np.ndarray],
    cam2obj: np.ndarray,
) -> np.ndarray:
    """Lift 2D tracks to 3D points using depth and transformation.

    Args:
        tracks: (S, N, 2) track coordinates in pixel space
        tracks_mask: (S, N) validity mask
        depths: List of (H, W) depth maps (in object space)
        intrinsics: List of (3, 3) camera intrinsic matrices
        cam2obj: (4, 4) camera-to-object transformation

    Returns:
        points_3d: (S, N, 3) 3D points in object space (NaN for invalid)
    """
    num_frames, num_tracks = tracks.shape[:2]
    points_3d = np.full((num_frames, num_tracks, 3), np.nan, dtype=np.float32)

    for frame_idx in range(num_frames):
        if depths[frame_idx] is None or intrinsics[frame_idx] is None:
            continue

        depth = depths[frame_idx]
        K = intrinsics[frame_idx]
        H, W = depth.shape[:2]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for track_idx in range(num_tracks):
            if not tracks_mask[frame_idx, track_idx]:
                continue

            # Get pixel coordinates
            x, y = tracks[frame_idx, track_idx]
            u = int(round(x))
            v = int(round(y))

            # Bounds check
            if u < 0 or u >= W or v < 0 or v >= H:
                continue

            # Get depth value
            z = depth[v, u]
            if z <= 0:
                continue

            # Unproject to camera space
            x_cam = (x - cx) * z / fx
            y_cam = (y - cy) * z / fy
            z_cam = z

            # Transform to object space
            pt_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
            pt_obj = cam2obj @ pt_cam
            points_3d[frame_idx, track_idx] = pt_obj[:3]

    return points_3d


def build_image_info(
    frame_idx: int,
    local_idx: int,
    tracks: np.ndarray,
    vis_scores: np.ndarray,
    tracks_mask: np.ndarray,
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    o2c: np.ndarray,
    is_keyframe: bool,
) -> Dict:
    """Build image info structure for a single frame.

    Args:
        frame_idx: Original frame index
        local_idx: Local index in the sequence
        tracks: (N, 2) track coordinates for this frame
        vis_scores: (N,) visibility scores for this frame
        tracks_mask: (N,) validity mask for this frame
        points_3d: (N, 3) 3D points for this frame
        intrinsics: (3, 3) camera intrinsic matrix
        o2c: (4, 4) object-to-camera transformation
        is_keyframe: Whether this frame is a keyframe

    Returns:
        Dictionary with image info
    """
    return {
        'frame_idx': frame_idx,
        'local_idx': local_idx,
        'tracks': tracks.astype(np.float32),  # (N, 2)
        'vis_scores': vis_scores.astype(np.float32),  # (N,)
        'tracks_mask': tracks_mask.astype(bool),  # (N,)
        'points_3d': points_3d.astype(np.float32),  # (N, 3)
        'intrinsics': intrinsics.astype(np.float32),  # (3, 3)
        'o2c': o2c.astype(np.float32),  # (4, 4)
        'is_keyframe': is_keyframe,
    }


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir / "pipeline_preprocess"
    tracks_dir = out_dir / "pipeline_corres"

    # 1. Load the preprocessed data from data_preprocess_dir
    print("Loading preprocessed data...")
    frame_indices = load_frame_list(data_preprocess_dir)
    preprocessed = load_preprocessed_data(data_preprocess_dir, frame_indices)
    print(f"Loaded {len(frame_indices)} frames: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")

    # 2. Load the tracks from tracks_dir
    print("Loading VGGSfM tracks...")
    track_data = load_tracks(tracks_dir)
    tracks = track_data['tracks']  # (S, N, 2)
    vis_scores = track_data['vis_scores']  # (S, N)
    tracks_mask = track_data['tracks_mask']  # (S, N)
    print(f"Loaded tracks: {tracks.shape[0]} frames, {tracks.shape[1]} tracks")

    # 3. Load the transformation from SAM3D_dir
    print("Loading SAM3D transformation...")
    sam3d_transform = load_sam3d_transform(SAM3D_dir, cond_idx)
    cond_cam_to_obj = np.eye(4, dtype=np.float32)
    cond_cam_to_obj[:3, :3] = sam3d_transform['scale'] * sam3d_transform['cond_cam_to_sam3d'][:3, :3] # Scale rotation to be SO3
    cond_cam_to_obj[:3, 3] = sam3d_transform['cond_cam_to_sam3d'][:3, 3]


    scale = sam3d_transform['scale']
    print(f"SAM3D scale: {scale}")

    # 4. Find condition frame position
    try:
        cond_local_idx = frame_indices.index(cond_idx)
    except ValueError:
        raise ValueError(f"Condition index {cond_idx} not found in frame list: {frame_indices}")
    print(f"Condition frame {cond_idx} is at local index {cond_local_idx}")

    # 5. Lift 2D tracks to 3D points using depth and transformation
    print("Lifting tracks to 3D (condition frame only)...")
    points_3d = np.full(tracks.shape[:2] + (3,), np.nan, dtype=np.float32)
    cond_points_3d = lift_tracks_to_3d(
        tracks[cond_local_idx:cond_local_idx + 1],
        tracks_mask[cond_local_idx:cond_local_idx + 1],
        [preprocessed['depths'][cond_local_idx]],
        [preprocessed['intrinsics'][cond_local_idx]],
        cond_cam_to_obj,  # Use condition frame's camera-to-object transform
    )
    points_3d[cond_local_idx] = cond_points_3d[0]
    valid_3d_count = np.isfinite(points_3d[cond_local_idx]).all(axis=-1).sum()
    cond_mask_count = int(tracks_mask[cond_local_idx].sum())
    print(f"Lifted {valid_3d_count} valid 3D points out of {cond_mask_count} masked track observations")
    

    # 6. Build image info for each frame and save
    print("Building and saving image info...")
    results_dir = out_dir / "pipeline_joint_opt"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize o2c for each frame
    # The condition frame has known o2c from SAM3D
    # Other frames will be initialized to identity (to be optimized later)
    c2o_per_frame = []
    for i, frame_idx in enumerate(frame_indices):
        if i == cond_local_idx:
            # Condition frame: use SAM3D transformation
            c2o_per_frame.append(cond_cam_to_obj)
        else:
            # Other frames: initialize to identity (to be optimized later)
            c2o_per_frame.append(np.eye(4, dtype=np.float32))

    c2o_per_frame = np.stack(c2o_per_frame, axis=0)  # (S, 4, 4)

    # Determine keyframes (for now, only condition frame is keyframe)
    keyframe_flags = [i == cond_local_idx for i in range(len(frame_indices))]
    # Save image info for each frame
    image_info = {
        'frame_indices': frame_indices,
        'cond_idx': cond_idx,
        "tracks": tracks.astype(np.float32),
        "vis_scores": vis_scores.astype(np.float32),
        "tracks_mask": tracks_mask.astype(bool),
        "keyframes": keyframe_flags,
        "points_3d": points_3d.astype(np.float32),
    }


    # Save image info
    info_path = results_dir / "image_info.npy"
    np.save(info_path, image_info)

    print(f"Saved image info for {len(frame_indices)} frames to {results_dir}")

    # Also save a summary file
    summary = {
        'frame_indices': frame_indices,
        'cond_idx': cond_idx,
        'cond_local_idx': cond_local_idx,
        'num_tracks': tracks.shape[1],
        'points_3d_valid_count': int(np.isfinite(points_3d).all(axis=-1).sum()),
        'keyframe_indices': [frame_indices[i] for i, is_kf in enumerate(keyframe_flags) if is_kf],
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint optimization pipeline for HOI reconstruction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")

    args = parser.parse_args()
    main(args)
