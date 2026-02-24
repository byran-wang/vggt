import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace
from tqdm import tqdm

import numpy as np

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import get_depth, depth2xyzmap
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_sam3d_transform


class TeeStream:
    """Duplicate writes to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


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


def prepare_joint_opt_inputs(
    data_preprocess_dir: Path,
    tracks_dir: Path,
    sam3d_dir: Path,
    cond_idx: int,
) -> Tuple[List[int], Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load preprocessing, tracks, and SAM3D transform for joint optimization."""
    print("Loading preprocessed data...")
    frame_indices = load_frame_list(data_preprocess_dir)
    preprocessed = load_preprocessed_data(data_preprocess_dir, frame_indices)
    print(f"Loaded {len(frame_indices)} frames: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")

    print("Loading VGGSfM tracks...")
    track_data = load_tracks(tracks_dir)
    tracks = track_data['tracks']
    vis_scores = track_data['vis_scores']
    tracks_mask = track_data['tracks_mask']
    # Mask out tracks with low visibility scores
    tracks_mask = tracks_mask & (vis_scores >= 0.5)
    print(f"Loaded tracks: {tracks.shape[0]} frames, {tracks.shape[1]} tracks")

    print("Loading SAM3D transformation...")
    sam3d_transform = load_sam3d_transform(sam3d_dir, cond_idx)
    cond_cam_to_obj = np.eye(4, dtype=np.float32)
    cond_cam_to_obj[:3, :3] = sam3d_transform['scale'] * sam3d_transform['cond_cam_to_sam3d'][:3, :3]
    cond_cam_to_obj[:3, 3] = sam3d_transform['cond_cam_to_sam3d'][:3, 3]

    try:
        cond_local_idx = frame_indices.index(cond_idx)
    except ValueError:
        raise ValueError(f"Condition index {cond_idx} not found in frame list: {frame_indices}")
    print(f"Condition frame {cond_idx} is at local index {cond_local_idx}")

    return (
        frame_indices,
        preprocessed,
        tracks,
        vis_scores,
        tracks_mask,
        cond_cam_to_obj,
        cond_local_idx,
    )



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


def register_first_frame(
    tracks: np.ndarray,
    tracks_mask: np.ndarray,
    preprocessed: Dict,
    frame_indices: List[int],
    cond_local_idx: int,
    cond_cam_to_obj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift condition-frame tracks and initialize per-frame poses/keyframes."""
    print("Lifting tracks to 3D (condition frame only)...")
    cond_points_3d = lift_tracks_to_3d(
        tracks[cond_local_idx:cond_local_idx + 1],
        tracks_mask[cond_local_idx:cond_local_idx + 1],
        [preprocessed['depths'][cond_local_idx]],
        [preprocessed['intrinsics'][cond_local_idx]],
        cond_cam_to_obj,
    )
    points_3d = cond_points_3d[0]
    valid_3d_count = np.isfinite(points_3d).all(axis=-1).sum()
    cond_mask_count = int(tracks_mask[cond_local_idx].sum())
    print(f"Lifted {valid_3d_count} valid 3D points out of {cond_mask_count} masked track observations")

    c2o_per_frame = []
    for i in range(len(frame_indices)):
        if i == cond_local_idx:
            c2o_per_frame.append(cond_cam_to_obj)
        else:
            c2o_per_frame.append(np.eye(4, dtype=np.float32))
    c2o_per_frame = np.stack(c2o_per_frame, axis=0)

    return points_3d, c2o_per_frame


def save_reproj_errors(image_info: Dict, register_idx: int, image: np.ndarray, results_dir: Path) -> None:
    """Compute and save reprojection errors for a registered frame."""
    frame_indices = image_info.get("frame_indices", [])
    if register_idx not in frame_indices:
        return

    local_idx = frame_indices.index(register_idx)
    tracks = image_info["tracks"]
    tracks_mask = image_info["tracks_mask"]
    points_3d = image_info["points_3d"]
    c2o = image_info["c2o"]

    frame_mask = np.asarray(tracks_mask[local_idx]).astype(bool)
    finite_mask = np.isfinite(points_3d).all(axis=-1)
    valid = frame_mask & finite_mask

    if not valid.any():
        return

    o2c = np.linalg.inv(c2o[local_idx])
    K = image_info.get("intrinsics")

    if K.ndim == 3:
        K = K[local_idx]

    pts_3d = points_3d[valid].astype(np.float64)
    pts_2d = tracks[local_idx][valid].astype(np.float64)

    cam = (o2c[:3, :3] @ pts_3d.T).T + o2c[:3, 3]
    in_front = cam[:, 2] > 0

    errs = np.full(valid.sum(), np.nan, dtype=np.float64)
    proj_2d_all = np.full((valid.sum(), 2), np.nan, dtype=np.float64)
    if in_front.any():
        proj_x = K[0, 0] * cam[in_front, 0] / cam[in_front, 2] + K[0, 2]
        proj_y = K[1, 1] * cam[in_front, 1] / cam[in_front, 2] + K[1, 2]
        proj_2d = np.stack([proj_x, proj_y], axis=1)
        errs[in_front] = np.linalg.norm(proj_2d - pts_2d[in_front], axis=1)
        proj_2d_all[in_front] = proj_2d

    valid_errs = errs[np.isfinite(errs)]

    # Save reprojection error as image
    import cv2
    from PIL import Image



    vis_img = image.copy()

    finite_errs = np.isfinite(errs)
    if finite_errs.any():
        start_pts = pts_2d[finite_errs]
        end_pts = proj_2d_all[finite_errs]
        errors_vis = errs[finite_errs]

        for s, e, err in zip(start_pts, end_pts, errors_vis):
            start = tuple(np.round(s).astype(int))
            end = tuple(np.round(e).astype(int))
            color = (255, 0, 0) if err >= 2.0 else (0, 0, 255)
            cv2.arrowedLine(
                vis_img,
                start,
                end,
                color=color,
                thickness=1,
                tipLength=0.2,
            )

    img_path = results_dir / "reproj_error.png"
    # Draw stats text on image
    if len(valid_errs) > 0:
        text = (f"mean={valid_errs.mean():.2f}px max={valid_errs.max():.2f}px "
                f"val_n/pts_n {len(valid_errs)}/{points_3d.shape[0]}")
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    Image.fromarray(vis_img).save(img_path)
    print(f"Saved reproj error image to {img_path}")


def save_results(image_info: Dict, register_idx, preprocessed_data, results_dir: Path, only_save_register_order=False
) -> None:
    """Save image info for joint optimization outputs."""
    results_dir = results_dir / f"{register_idx:04d}"
    from robust_hoi_pipeline.frame_management import save_register_order
    save_register_order(results_dir / "../" , register_idx)
    if only_save_register_order:
        return
    
    results_dir.mkdir(parents=True, exist_ok=True)
    info_path = results_dir / "image_info.npy"
    np.save(info_path, image_info)
    print(f"Saved image info to {results_dir}")



    #Load the image from preprocessed data for the registered frame
    frame_indices = image_info.get("frame_indices", [])
    image = None
    if register_idx in frame_indices:
        local_idx = frame_indices.index(register_idx)
        images = preprocessed_data.get("images")
        if images is not None and local_idx < len(images):
            image = images[local_idx]

    # Save reprojection errors for the registered frame
    if image is not None:
        save_reproj_errors(image_info, register_idx, image, results_dir)
    


def _build_default_joint_opt_args(output_dir: Path, cond_index: int) -> SimpleNamespace:
    """Create a minimal args namespace for frame management helpers."""
    return SimpleNamespace(
        output_dir=str(output_dir),
        cond_index=cond_index,
        max_query_pts=512,
        query_frame_num=0,
        fine_tracking=True,
        # thresholds
        vis_thresh=0.4,
        max_reproj_error=3.0,
        min_inlier_per_frame=50,
        min_inlier_per_track=4,
        min_depth_pixels=500,
        min_track_number=3,
        kf_rot_thresh=5.0,
        kf_trans_thresh=0.02,
        kf_depth_thresh=500,
        kf_inlier_thresh=10,
        run_ba_on_keyframe=0,
        unc_thresh=4.0,
        duplicate_track_thresh=3.0,
        pnp_reproj_thresh=4.0,
        joint_opt_reproj_thresh=4.0,
        no_optimize_with_point_to_plane=False,
        only_save_register_order=True,
    )


def _stack_intrinsics(intrinsics_list: List[np.ndarray]) -> np.ndarray:
    """Stack intrinsics, filling missing entries with the first valid matrix."""
    valid = [K for K in intrinsics_list if K is not None]
    if not valid:
        raise ValueError("No valid intrinsics found.")
    fallback = valid[0]
    stacked = [K if K is not None else fallback for K in intrinsics_list]
    return np.stack(stacked, axis=0)


def mask_track_for_outliers(image_info, frame_idx, reproj_thresh, min_track_number=1):
    """Mask tracks whose reprojection error exceeds a threshold for a given frame.

    After a frame is registered via PnP, this reprojects 3D points onto the frame
    and sets track_mask to 0 for tracks with reprojection error > reproj_thresh.
    Only tracks whose 3D points are tracked by at least min_track_number keyframes
    are considered for masking.

    Args:
        min_track_number: Minimum number of keyframes a track must be visible in
            to be considered for outlier masking.
    """
    track_mask = image_info["track_mask"]
    pred_tracks = image_info["pred_tracks"]
    points_3d = image_info.get("points_3d")

    frame_mask = np.asarray(track_mask[frame_idx]).astype(bool)
    # Only consider tracks visible in >= min_track_number keyframes
    kf_indices = np.where(np.asarray(image_info["keyframe"]).astype(bool))[0]
    if len(kf_indices) > 0:
        track_vis_count = np.asarray(track_mask)[kf_indices].astype(bool).sum(axis=0)
        well_observed = track_vis_count >= min_track_number
        frame_mask = frame_mask & well_observed
    if points_3d is None or not frame_mask.any():
        return

    ext = image_info["extrinsics"][frame_idx]
    K = image_info["intrinsics"][frame_idx] if image_info["intrinsics"].ndim == 3 else image_info["intrinsics"]

    pts_3d = np.asarray(points_3d)[frame_mask].astype(np.float64)
    pts_2d = np.asarray(pred_tracks[frame_idx])[frame_mask].astype(np.float64)

    finite = np.isfinite(pts_3d).all(axis=1)
    if not finite.any():
        return

    cam = (ext[:3, :3] @ pts_3d[finite].T).T + ext[:3, 3]
    in_front = cam[:, 2] > 0

    errs = np.zeros(finite.sum(), dtype=np.float64)
    if in_front.any():
        proj_x = K[0, 0] * cam[in_front, 0] / cam[in_front, 2] + K[0, 2]
        proj_y = K[1, 1] * cam[in_front, 1] / cam[in_front, 2] + K[1, 2]
        proj_2d = np.stack([proj_x, proj_y], axis=1)
        errs[in_front] = np.linalg.norm(proj_2d - pts_2d[finite][in_front], axis=1)

    visible_idx = np.where(frame_mask)[0]
    finite_idx = visible_idx[finite]
    outlier_idx = finite_idx[errs > reproj_thresh]
    if len(outlier_idx) > 0:
        track_mask[frame_idx][outlier_idx] = 0
        print(f"[mask_reproj_outliers] Frame {frame_idx}: masked {len(outlier_idx)} tracks "
              f"with reproj error > {reproj_thresh}px")


def _save_depth_alignment_debug(image_info_work, frame_idx, depth_map, K, masks, ys, xs, debug_dir):
    if debug_dir is None:
        return

    from PIL import Image
    import trimesh as _trimesh

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)

    img_dbg = None
    images = image_info_work.get("images")
    if images is not None and images[frame_idx] is not None:
        img_dbg = np.asarray(images[frame_idx])
        if img_dbg.ndim == 2:
            Image.fromarray(img_dbg.astype(np.uint8), mode="L").save(
                _debug_dir / f"image_frame_{frame_idx:04d}.png")
        elif img_dbg.ndim == 3 and img_dbg.shape[2] >= 3:
            Image.fromarray(img_dbg[:, :, :3].astype(np.uint8), mode="RGB").save(
                _debug_dir / f"image_frame_{frame_idx:04d}.png")

    if masks is not None and masks[frame_idx] is not None:
        mask_u8 = (masks[frame_idx] > 0).astype(np.uint8) * 255
        Image.fromarray(mask_u8, mode="L").save(_debug_dir / f"mask_frame_{frame_idx:04d}.png")

    if len(ys) == 0:
        return

    z_dbg = depth_map[ys, xs].astype(np.float64)
    x_dbg = (xs.astype(np.float64) - K[0, 2]) * z_dbg / K[0, 0]
    y_dbg = (ys.astype(np.float64) - K[1, 2]) * z_dbg / K[1, 1]
    pts_dbg = np.stack([x_dbg, y_dbg, z_dbg], axis=-1)

    colors_dbg = None
    if img_dbg is not None:
        if img_dbg.ndim == 2:
            img_rgb = np.stack([img_dbg, img_dbg, img_dbg], axis=-1)
        elif img_dbg.ndim == 3 and img_dbg.shape[2] >= 3:
            img_rgb = img_dbg[:, :, :3]
        else:
            img_rgb = None

        if img_rgb is not None:
            colors_dbg = img_rgb[ys, xs]
            if np.issubdtype(colors_dbg.dtype, np.floating) and colors_dbg.max(initial=0.0) <= 1.0:
                colors_dbg = colors_dbg * 255.0
            colors_dbg = np.clip(colors_dbg, 0, 255).astype(np.uint8)

    if colors_dbg is None:
        colors_dbg = np.zeros((len(pts_dbg), 3), dtype=np.uint8)
        colors_dbg[:, 2] = 255

    ply_path = _debug_dir / f"depth_frame_{frame_idx:04d}.ply"
    _trimesh.PointCloud(pts_dbg.astype(np.float32), colors=colors_dbg).export(ply_path)


def _save_icp_iteration_debug(debug_dir, it, pts_obj, p, c):
    if debug_dir is None:
        return

    import trimesh as _trimesh

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)

    # all object-space depth points (blue)
    colors_all = np.zeros((len(pts_obj), 4), dtype=np.uint8)
    colors_all[:, 2] = 255
    colors_all[:, 3] = 255
    _trimesh.PointCloud(pts_obj.astype(np.float32), colors=colors_all).export(
        _debug_dir / f"pts_obj_iter{it:03d}.ply")

    # inlier points (green)
    colors_p = np.zeros((len(p), 4), dtype=np.uint8)
    colors_p[:, 1] = 255
    colors_p[:, 3] = 255
    _trimesh.PointCloud(p.astype(np.float32), colors=colors_p).export(
        _debug_dir / f"inlier_iter{it:03d}.ply")

    # closest mesh surface points (red)
    colors_c = np.zeros((len(c), 4), dtype=np.uint8)
    colors_c[:, 0] = 255
    colors_c[:, 3] = 255
    _trimesh.PointCloud(c.astype(np.float32), colors=colors_c).export(
        _debug_dir / f"closest_iter{it:03d}.ply")


def _save_obj_depth_points_debug(debug_dir, frame_idx, pts_obj, filename_prefix, color_rgba):
    if debug_dir is None or pts_obj is None or len(pts_obj) == 0:
        return

    import trimesh as _trimesh

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    colors_obj = np.zeros((len(pts_obj), 4), dtype=np.uint8)
    colors_obj[:, :] = np.array(color_rgba, dtype=np.uint8)
    _trimesh.PointCloud(pts_obj.astype(np.float32), colors=colors_obj).export(
        _debug_dir / f"{filename_prefix}_frame_{frame_idx:04d}.ply"
    )


def _save_normal_map_debug(debug_dir, frame_idx, normal_map, filename_prefix, it=None):
    if debug_dir is None or normal_map is None:
        return

    from PIL import Image

    normal_np = np.asarray(normal_map)
    if normal_np.ndim != 3 or normal_np.shape[2] != 3:
        return

    normal_u8 = np.clip((normal_np + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    if it is None:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}.png"
    else:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}_iter{it:03d}.png"
    Image.fromarray(normal_u8, mode="RGB").save(_debug_dir / out_name)


def _align_frame_with_mesh_depth(image_info_work, frame_idx, mesh, max_pts=2000, num_iters=60, inlier_thresh=0.3, debug_dir=None):
    """Align a frame by optimizing pose with rendered-vs-observed depth/normal losses."""
    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr
    from scipy.spatial.transform import Rotation as ScipyRotation
    from utils_simba.render import diff_renderer, projection_matrix_from_intrinsics

    del inlier_thresh  # kept in signature for compatibility

    depth_priors = image_info_work.get("depth_priors")
    normal_priors = image_info_work.get("normal_priors")
    intrinsics = image_info_work.get("intrinsics")
    extrinsics = image_info_work.get("extrinsics")

    d = depth_priors[frame_idx]
    if d is None:
        print(f"[align_depth] Frame {frame_idx}: no depth, skipping")
        return False

    K = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics
    n_obs = None
    if normal_priors is not None and frame_idx < len(normal_priors):
        n_obs = normal_priors[frame_idx]
        _save_normal_map_debug(
            debug_dir=debug_dir,
            frame_idx=frame_idx,
            normal_map=n_obs,
            filename_prefix="normal_prior_raw",
        )

    vmask = d > 0.01
    masks = image_info_work.get("image_masks")
    if masks is not None and masks[frame_idx] is not None:
        vmask = vmask & (masks[frame_idx] > 0)
    if n_obs is not None:
        vmask = vmask & np.isfinite(n_obs).all(axis=-1) & (np.linalg.norm(n_obs, axis=-1) > 1e-6)

    ys, xs = np.where(vmask)
    if debug_dir is not None:
        _save_depth_alignment_debug(
            image_info_work=image_info_work,
            frame_idx=frame_idx,
            depth_map=d,
            K=K,
            masks=masks,
            ys=ys,
            xs=xs,
            debug_dir=debug_dir,
        )

    if len(ys) < 50:
        print(f"[align_depth] Frame {frame_idx}: only {len(ys)} depth pixels, skipping")
        return False

    zs = d[ys, xs].astype(np.float64)
    xc = (xs.astype(np.float64) - K[0, 2]) * zs / K[0, 0]
    yc = (ys.astype(np.float64) - K[1, 2]) * zs / K[1, 1]
    pts_cam = np.stack([xc, yc, zs], axis=-1)

    ext0 = extrinsics[frame_idx].astype(np.float64)
    R0 = ext0[:3, :3]
    t0 = ext0[:3, 3]
    pts_obj0 = (R0.T @ (pts_cam - t0).T).T
    _save_obj_depth_points_debug(
        debug_dir=debug_dir,
        frame_idx=frame_idx,
        pts_obj=pts_obj0,
        filename_prefix="depth_obj",
        color_rgba=(255, 255, 0, 255),
    )

    bbox_min = np.array([-0.8, -0.8, -0.8], dtype=np.float64)
    bbox_max = np.array([0.8, 0.8, 0.8], dtype=np.float64)
    in_bbox = np.all((pts_obj0 >= bbox_min) & (pts_obj0 <= bbox_max), axis=1)
    ys = ys[in_bbox]
    xs = xs[in_bbox]

    if len(ys) < max_pts:
        print(
            f"[align_depth] Frame {frame_idx}: only {len(ys)} points in object bbox, "
            f"need at least {max_pts}, skipping"
        )
        return False

    if len(ys) > max_pts:
        sel = np.random.choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]

    zs_sel = d[ys, xs].astype(np.float64)
    xc_sel = (xs.astype(np.float64) - K[0, 2]) * zs_sel / K[0, 0]
    yc_sel = (ys.astype(np.float64) - K[1, 2]) * zs_sel / K[1, 1]
    pts_cam_sel = np.stack([xc_sel, yc_sel, zs_sel], axis=-1)

    if not torch.cuda.is_available():
        print(f"[align_depth] Frame {frame_idx}: CUDA unavailable, skipping depth-normal optimization")
        return False

    H, W = d.shape
    device = "cuda"
    glctx = dr.RasterizeCudaContext()

    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)[None]
    tri = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    vnormals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)[None]
    if mesh.visual.vertex_colors is not None:
        colors_np = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    else:
        colors_np = np.full((len(mesh.vertices), 3), 0.5, dtype=np.float32)
    color = torch.tensor(colors_np, dtype=torch.float32, device=device)[None]

    proj_np = projection_matrix_from_intrinsics(K.astype(np.float64), height=H, width=W, znear=0.1, zfar=100.0)
    projection = torch.tensor(proj_np, dtype=torch.float32, device=device)

    obs_depth = torch.tensor(d.astype(np.float32), dtype=torch.float32, device=device)
    obs_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    obs_mask[torch.tensor(ys, device=device), torch.tensor(xs, device=device)] = True

    obs_normal = None
    if n_obs is not None:
        n_obs_f = n_obs.astype(np.float32)
        obs_normal = torch.tensor(n_obs_f, dtype=torch.float32, device=device)
        obs_normal = F.normalize(obs_normal, dim=-1)

    init_rot = ScipyRotation.from_matrix(ext0[:3, :3]).as_rotvec().astype(np.float32)
    rotvec = torch.tensor(init_rot, dtype=torch.float32, device=device, requires_grad=True)
    trans = torch.tensor(ext0[:3, 3].astype(np.float32), dtype=torch.float32, device=device, requires_grad=True)

    def rodrigues(aa):
        aa_b = aa[None]
        th = aa_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        k = aa_b / th
        Kx = torch.zeros(1, 3, 3, device=device, dtype=torch.float32)
        Kx[:, 0, 1] = -k[:, 2]
        Kx[:, 0, 2] = k[:, 1]
        Kx[:, 1, 0] = k[:, 2]
        Kx[:, 1, 2] = -k[:, 0]
        Kx[:, 2, 0] = -k[:, 1]
        Kx[:, 2, 1] = k[:, 0]
        c = torch.cos(th).unsqueeze(-1)
        s = torch.sin(th).unsqueeze(-1)
        I3 = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
        return (c * I3 + s * Kx + (1 - c) * k.unsqueeze(-1) @ k.unsqueeze(-2))[0]

    optimizer = torch.optim.Adam([rotvec, trans], lr=2e-3)
    best = {"loss": float("inf"), "R": ext0[:3, :3].copy(), "t": ext0[:3, 3].copy(), "valid": 0}

    for it in range(num_iters):
        optimizer.zero_grad()
        R = rodrigues(rotvec)
        o2c = torch.eye(4, dtype=torch.float32, device=device)
        o2c[:3, :3] = R
        o2c[:3, 3] = trans

        if obs_normal is not None:
            _, depth_r, normal_r = diff_renderer(
                verts=verts,
                tri=tri,
                color=color,
                projection=projection,
                ob_in_cvcams=o2c,
                resolution=np.asarray([H, W]),
                glctx=glctx,
                get_normal=True,
                vnormals=vnormals,
            )
            normal_r = torch.flip(normal_r[0], dims=[0])
            normal_r = F.normalize(normal_r, dim=-1)
        else:
            _, depth_r = diff_renderer(
                verts=verts,
                tri=tri,
                color=color,
                projection=projection,
                ob_in_cvcams=o2c,
                resolution=np.asarray([H, W]),
                glctx=glctx,
            )
            normal_r = None

        depth_r = torch.flip(depth_r[0], dims=[0])
        valid = obs_mask & (depth_r > 0) & torch.isfinite(depth_r)
        valid_count = int(valid.sum().item())
        if valid_count < 100:
            print(f"[align_depth] Frame {frame_idx}: iter {it}, only {valid_count} valid rendered pixels")
            continue

        depth_res = depth_r[valid] - obs_depth[valid]
        depth_abs = depth_res.abs()
        loss_depth = torch.where(depth_abs < 0.02, 0.5 * depth_res ** 2, 0.02 * (depth_abs - 0.01)).mean()

        if obs_normal is not None and normal_r is not None:
            dot = (normal_r[valid] * obs_normal[valid]).sum(dim=-1).clamp(-1.0, 1.0)
            loss_normal = (1.0 - dot).mean()
        else:
            loss_normal = torch.tensor(0.0, device=device)

        loss = loss_depth + 0.2 * loss_normal
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()

        loss_val = float(loss.detach().item())
        if loss_val < best["loss"]:
            best["loss"] = loss_val
            best["R"] = R.detach().cpu().numpy().astype(np.float64)
            best["t"] = trans.detach().cpu().numpy().astype(np.float64)
            best["valid"] = valid_count

        if it == 0 or (it + 1) % 5 == 0 or it == num_iters - 1:
            if normal_r is not None:
                _save_normal_map_debug(
                    debug_dir=debug_dir,
                    frame_idx=frame_idx,
                    normal_map=normal_r.detach().cpu().numpy(),
                    filename_prefix="normal_rendered",
                    it=it + 1,
                )
            print(
                f"[align_depth] Frame {frame_idx}: iter {it+1}/{num_iters}, "
                f"loss_d={loss_depth.item():.6f}, loss_n={loss_normal.item():.6f}, valid={valid_count}"
            )

    if not np.isfinite(best["loss"]) or best["valid"] < 100:
        print(f"[align_depth] Frame {frame_idx}: optimization failed (best_valid={best['valid']})")
        return False

    pts_obj_opt = (best["R"].T @ (pts_cam_sel - best["t"]).T).T
    _save_obj_depth_points_debug(
        debug_dir=debug_dir,
        frame_idx=frame_idx,
        pts_obj=pts_obj_opt,
        filename_prefix="depth_obj_optimized",
        color_rgba=(0, 255, 0, 255),
    )

    extrinsics[frame_idx, :3, :3] = best["R"].astype(np.float32)
    extrinsics[frame_idx, :3, 3] = best["t"].astype(np.float32)
    print(
        f"[align_depth] Frame {frame_idx}: alignment done, "
        f"best_loss={best['loss']:.6f}, valid={best['valid']}"
    )
    return True


def _joint_optimize_keyframes(
    image_info_work, neus_mesh_path, cond_local_idx,
    num_iters=30, lr_pose=5e-4, lr_points=1e-4,
    lambda_reproj=1.0, lambda_p2plane=10000., lambda_depth=300.,
    max_depth_pts=2000,
    min_track_number=None, cauchy_c=3.0, depth_huber_delta=0.01,
):
    """Jointly refine keyframe poses and 3D track points.

    Minimizes two losses:
      - Reprojection: projected 3D points vs 2D track observations on keyframes.
      - Point-to-plane: depth-map points transformed to object space vs NeuS mesh surface.

    The condition frame pose is held fixed.
    Modifies image_info_work["extrinsics"] and image_info_work["points_3d"] in-place.
    """
    import torch

    try:
        import trimesh
    except ImportError:
        print("[joint_opt] trimesh not installed, skipping")
        return

    if neus_mesh_path is None or not Path(neus_mesh_path).exists():
        mesh = None
    else:
        mesh = trimesh.load(neus_mesh_path, process=False)
        if len(mesh.vertices) < 3:
            print("[joint_opt] Mesh too small, disabling point-to-plane")
            mesh = None

    use_p2p = mesh is not None

    kf_mask = image_info_work["keyframe"].astype(bool)
    kf_indices = np.where(kf_mask)[0]
    if len(kf_indices) < 2:
        return

    mesh_info = ""
    if use_p2p:
        mesh_face_normals = np.array(mesh.face_normals, dtype=np.float32)
        mesh_info = f", mesh={len(mesh.vertices)} verts"
    print(f"[joint_opt] Starting: {len(kf_indices)} keyframes{mesh_info}, p2p={'on' if use_p2p else 'off'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extrinsics = image_info_work["extrinsics"]       # (N, 4, 4) o2c
    intrinsics_np = image_info_work["intrinsics"]     # (N, 3, 3)
    tracks_np = image_info_work["pred_tracks"]        # (N, M, 2)
    track_mask_np = image_info_work["track_mask"]      # (N, M)
    points_3d_np = image_info_work["points_3d"]       # (M, 3)
    depth_priors = image_info_work["depth_priors"]

    n_kf = len(kf_indices)
    opt_mask = np.array([idx != cond_local_idx for idx in kf_indices])
    if opt_mask.sum() == 0:
        return

    # --- optimisation variables: delta rotation (axis-angle) + translation ---
    delta_aa = torch.zeros(n_kf, 3, device=device, requires_grad=True)
    delta_t = torch.zeros(n_kf, 3, device=device, requires_grad=True)

    finite_mask = np.isfinite(points_3d_np).all(axis=-1)
    # Replace NaN with 0 to prevent NaN gradient contamination through einsum backward
    # (valid mask already excludes these points from the loss)
    pts3d_init = points_3d_np.copy()
    pts3d_init[~finite_mask] = 0.0
    pts3d = torch.tensor(pts3d_init, dtype=torch.float32,
                         device=device, requires_grad=True)

    # --- fixed data on device ---
    base_R = torch.tensor(extrinsics[kf_indices, :3, :3], dtype=torch.float32, device=device)
    base_t = torch.tensor(extrinsics[kf_indices, :3, 3], dtype=torch.float32, device=device)
    K = torch.tensor(intrinsics_np[kf_indices], dtype=torch.float32, device=device)
    trk = torch.tensor(tracks_np[kf_indices], dtype=torch.float32, device=device)
    tmask = torch.tensor(track_mask_np[kf_indices].astype(bool), device=device)
    fin_t = torch.tensor(finite_mask, device=device)
    # Only use 3D points visible in >= min_track_number keyframes for reprojection
    vis_count = tmask.sum(dim=0)  # (M,) how many keyframes each track is visible in
    valid = tmask & fin_t.unsqueeze(0) & (vis_count >= min_track_number).unsqueeze(0)
    opt_t = torch.tensor(opt_mask, device=device)

    # --- subsample depth point clouds (camera space) per keyframe ---
    dclouds = []
    if use_p2p:
        for ki, kf_idx in enumerate(kf_indices):
            d = depth_priors[kf_idx]
            if d is None:
                dclouds.append(None)
                continue
            Kn = intrinsics_np[kf_idx]
            vmask = d > 0
            masks = image_info_work.get("image_masks")
            if masks is not None and masks[kf_idx] is not None:
                vmask = vmask & (masks[kf_idx] > 0)
            ys, xs = np.where(vmask)
            if len(ys) == 0:
                dclouds.append(None)
                continue
            if len(ys) > max_depth_pts:
                sel = np.random.choice(len(ys), max_depth_pts, replace=False)
                ys, xs = ys[sel], xs[sel]
            zs = d[ys, xs]
            xc = (xs.astype(np.float32) - Kn[0, 2]) * zs / Kn[0, 0]
            yc = (ys.astype(np.float32) - Kn[1, 2]) * zs / Kn[1, 1]
            dclouds.append(torch.tensor(np.stack([xc, yc, zs], -1), dtype=torch.float32, device=device))

    # --- pre-compute observed depth at 2D track locations for each keyframe ---
    num_tracks = points_3d_np.shape[0]
    depth_at_tracks = torch.zeros(n_kf, num_tracks, dtype=torch.float32, device=device)
    for ki, kf_idx in enumerate(kf_indices):
        d = depth_priors[kf_idx]
        if d is None:
            continue
        H_d, W_d = d.shape[:2]
        coords = tracks_np[kf_idx]  # (M, 2)
        us = np.round(coords[:, 0]).astype(int)
        vs = np.round(coords[:, 1]).astype(int)
        in_bounds = (us >= 0) & (us < W_d) & (vs >= 0) & (vs < H_d)
        valid_pix = np.where(in_bounds)[0]
        depth_vals = np.zeros(num_tracks, dtype=np.float32)
        depth_vals[valid_pix] = d[vs[valid_pix], us[valid_pix]]
        depth_at_tracks[ki] = torch.tensor(depth_vals, dtype=torch.float32, device=device)
    has_obs_depth = depth_at_tracks > 0

    # --- helpers ---
    def rodrigues(aa):
        """Axis-angle (B,3) -> rotation matrix (B,3,3)."""
        th = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        k = aa / th
        Kx = torch.zeros(aa.shape[0], 3, 3, device=device)
        Kx[:, 0, 1] = -k[:, 2]; Kx[:, 0, 2] = k[:, 1]
        Kx[:, 1, 0] = k[:, 2];  Kx[:, 1, 2] = -k[:, 0]
        Kx[:, 2, 0] = -k[:, 1]; Kx[:, 2, 1] = k[:, 0]
        c = torch.cos(th).unsqueeze(-1)
        s = torch.sin(th).unsqueeze(-1)
        I3 = torch.eye(3, device=device).unsqueeze(0)
        return c * I3 + s * Kx + (1 - c) * k.unsqueeze(-1) @ k.unsqueeze(-2)

    def current_Rt():
        """Apply delta to base o2c: new_R = dR @ R, new_t = dR @ t + dt."""
        dR = rodrigues(delta_aa)
        R = torch.where(opt_t[:, None, None], dR @ base_R, base_R)
        t = torch.where(opt_t[:, None],
                        torch.einsum('bij,bj->bi', dR, base_t) + delta_t,
                        base_t)
        return R, t

    cond_ki = list(kf_indices).index(cond_local_idx) if cond_local_idx in kf_indices else -1

    optimizer = torch.optim.Adam([
        {"params": [delta_aa, delta_t], "lr": lr_pose},
        {"params": [pts3d], "lr": lr_points},
    ])

    nn_cache = [None] * n_kf

    for it in range(num_iters):
        optimizer.zero_grad()
        R_o2c, t_o2c = current_Rt()

        # === reprojection loss (COLMAP-style: Cauchy kernel on ||r||²) ===
        # r_ij = u_ij - π(K_i, R_i X_j + t_i),  ρ(s) = c² log(1 + s/c²)
        cam = torch.einsum('bij,mj->bmi', R_o2c, pts3d) + t_o2c[:, None, :]
        z = cam[:, :, 2:3].clamp(min=1e-6)
        px = K[:, 0:1, 0:1] * cam[:, :, 0:1] / z + K[:, 0:1, 2:3]
        py = K[:, 1:2, 1:2] * cam[:, :, 1:2] / z + K[:, 1:2, 2:3]
        proj = torch.cat([px, py], dim=-1)

        residual_sq = ((proj - trk) ** 2).sum(-1)  # ||r_ij||² (n_kf, M)
        front = cam[:, :, 2] > 0
        m = valid & front

        cauchy_c_sq = cauchy_c ** 2
        if m.any():
            s = residual_sq[m]
            loss_r = (cauchy_c_sq * torch.log1p(s / cauchy_c_sq)).mean()
        else:
            loss_r = torch.tensor(0.0, device=device)

        # === point-to-plane loss ===
        loss_p = torch.tensor(0.0, device=device)
        if use_p2p:
            update_nn = (it % 5 == 0)
            np2p = 0

            for ki in range(n_kf):
                if dclouds[ki] is None:
                    continue
                Ri, ti = R_o2c[ki], t_o2c[ki]
                RiT = Ri.T
                tic = -(RiT @ ti)
                pts_obj = (RiT @ dclouds[ki].T).T + tic

                # Filter out non-finite points before mesh query
                fin_mask = torch.isfinite(pts_obj).all(dim=-1)
                if fin_mask.sum() == 0:
                    continue
                pts_obj_fin = pts_obj[fin_mask]

                if update_nn or nn_cache[ki] is None:
                    pnp = pts_obj_fin.detach().cpu().numpy()
                    closest, _, tri_ids = mesh.nearest.on_surface(pnp)
                    nn_cache[ki] = (
                        torch.tensor(closest, dtype=torch.float32, device=device),
                        torch.tensor(mesh_face_normals[tri_ids],
                                     dtype=torch.float32, device=device),
                        fin_mask.detach().clone(),
                    )

                cp, cn, cached_mask = nn_cache[ki]
                # Re-filter with cached mask on non-update iterations
                if not update_nn:
                    pts_obj_fin = pts_obj[cached_mask]
                d = ((pts_obj_fin - cp) * cn).sum(-1)      # signed point-to-plane
                da = d.abs()
                loss_p = loss_p + torch.where(
                    da < 0.05, 0.5 * d ** 2, 0.05 * (da - 0.025)
                ).mean()
                np2p += 1

            if np2p > 0:
                loss_p = loss_p / np2p

        # === point-to-depth loss (predicted cam-z vs observed depth map) ===
        loss_d = torch.tensor(0.0, device=device)
        depth_valid = valid & front & has_obs_depth
        if depth_valid.any():
            pred_z = cam[:, :, 2][depth_valid]
            obs_z = depth_at_tracks[depth_valid]
            depth_res = pred_z - obs_z
            abs_res = depth_res.abs()
            loss_d = torch.where(
                abs_res < depth_huber_delta,
                0.5 * depth_res ** 2,
                depth_huber_delta * (abs_res - 0.5 * depth_huber_delta),
            ).mean()

        loss = lambda_reproj * loss_r + lambda_p2plane * loss_p + lambda_depth * loss_d
        loss.backward()

        # keep condition frame
        with torch.no_grad():
            if cond_ki >= 0 and delta_aa.grad is not None:
                delta_aa.grad[cond_ki] = 0
                delta_t.grad[cond_ki] = 0
            # if pts3d.grad is not None:
            #     pts3d.grad[~fin_t] = 0

        optimizer.step()

        if it == 0 or (it + 1) % 5 == 0:
            print(f"[joint_opt] {it+1}/{num_iters}  reproj={loss_r.item():.3f}  "
                  f"p2plane={loss_p.item():.5f}  p2depth={loss_d.item():.5f}  total={loss.item():.3f}")

    # --- write back ---
    with torch.no_grad():
        Rf, tf = current_Rt()
        Rf_np = Rf.cpu().numpy().astype(np.float32)
        tf_np = tf.cpu().numpy().astype(np.float32)
        for ki, kf_idx in enumerate(kf_indices):
            if opt_mask[ki]:
                image_info_work["extrinsics"][kf_idx, :3, :3] = Rf_np[ki]
                image_info_work["extrinsics"][kf_idx, :3, 3] = tf_np[ki]
        pts3d_out = pts3d.detach().cpu().numpy().astype(np.float32)
        pts3d_out[~finite_mask] = np.nan  # restore NaN for originally invalid points
        image_info_work["points_3d"] = pts3d_out

    print(f"[joint_opt] Done. Refined {int(opt_mask.sum())} poses, "
          f"{int(finite_mask.sum())} 3D points.")
    
def print_image_info_stats(image_info, invalid_cnt):
    print(
        f"stats {np.array(image_info['registered']).sum() + np.array(image_info['invalid']).sum() - 1}/{len(image_info['frame_indices'])} :, " # -1 for the first condition frame
        f"registered: {np.array(image_info['registered']).sum()}, "
        f"keyframes: {np.array(image_info['keyframe']).sum()}, "
        f"invalid: {np.array(image_info['invalid']).sum()}"
        f"(insuf_pixel: {invalid_cnt['insufficient_pixel']}, "
        f"3d_3d_corr: {invalid_cnt['3d_3d_corr']}, "
        f"reproj_err: {invalid_cnt['reproj_err']})"
    )   

def register_remaining_frames(image_info, preprocessed_data, output_dir: Path, cond_idx: int,
                               neus_ckpt=None, neus_total_steps=0, sam3d_root_dir=None,
                               neus_init_mesh=None, no_optimize_with_point_to_plane=False):

    from robust_hoi_pipeline.frame_management import (
        find_next_frame,
        check_frame_invalid,
        check_reprojection_error,
        check_key_frame,
        process_key_frame,
        _refine_frame_pose_3d,
        save_keyframe_indices,
    )
    from robust_hoi_pipeline.optimization import register_new_frame_by_PnP
    from robust_hoi_pipeline.neus_integration import prepare_neus_data, run_neus_training, save_neus_mesh

    args = _build_default_joint_opt_args(output_dir, cond_idx)
    args.no_optimize_with_point_to_plane = no_optimize_with_point_to_plane
    neus_data_dir = output_dir / "pipeline_joint_opt" / "neus_data"

    frame_indices = image_info["frame_indices"]
    cond_local_idx = frame_indices.index(cond_idx)
    c2o = image_info.get("c2o")
    if c2o is None:
        c2o = np.tile(np.eye(4, dtype=np.float32), (len(frame_indices), 1, 1))
    extrinsics = np.linalg.inv(c2o).astype(np.float32)

    intrinsics = _stack_intrinsics(preprocessed_data["intrinsics"])
    depth_priors = preprocessed_data["depths"]
    points_3d_global = image_info["points_3d"].astype(np.float32)
    invalid_cnt = {
        "insufficient_pixel": 0,
        "3d_3d_corr": 0,
        "reproj_err": 0,
    }

    image_info_work = {
        "frame_indices": image_info["frame_indices"],
        "pred_tracks": image_info["tracks"],
        "track_mask": image_info["tracks_mask"],
        "points_3d": points_3d_global,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depth_priors": depth_priors,
        "normal_priors": preprocessed_data.get("normals"),
        "images": preprocessed_data["images"],
        "image_masks": preprocessed_data.get("masks_obj"),
        "keyframe": np.array(image_info["keyframe"], dtype=bool),
        "registered": np.array(image_info["register"], dtype=bool),
        "invalid": np.array(image_info["invalid"], dtype=bool),
    }

    num_frames = len(frame_indices)
    latest_neus_mesh = neus_init_mesh

    sam_3d_mesh_file = f"{sam3d_root_dir}/mesh.obj"

    # Load the SAM3D mesh for depth-based alignment
    sam3d_mesh = None
    if sam3d_root_dir is not None and Path(sam_3d_mesh_file).exists():
        import trimesh
        sam3d_mesh = trimesh.load(str(sam_3d_mesh_file), process=False)
        print(f"Loaded SAM3D mesh: {len(sam3d_mesh.vertices)} vertices")


    while image_info_work["registered"].sum() + image_info_work["invalid"].sum() < num_frames:
        next_frame_idx = find_next_frame(image_info_work)
        if next_frame_idx is None:
            break

        print("+" * 50)
        print(f"Next frame to register: {image_info['frame_indices'][next_frame_idx]} (local idx {next_frame_idx})")

        if check_frame_invalid(
            image_info_work,
            next_frame_idx,
            min_inlier_per_frame=args.min_inlier_per_frame,
            min_depth_pixels=args.min_depth_pixels,
        ):
            image_info["invalid"][next_frame_idx] = image_info_work["invalid"][next_frame_idx] = True  
            print(f"[register_remaining_frames] Frame {next_frame_idx} marked as invalid due to insufficient inliers/depth pixels")
            invalid_cnt["insufficient_pixel"] += 1
            save_results(image_info=image_info, register_idx= image_info['frame_indices'][next_frame_idx], preprocessed_data=preprocessed_data, results_dir=output_dir / "pipeline_joint_opt", only_save_register_order=args.only_save_register_order)
            print_image_info_stats(image_info_work, invalid_cnt)
            continue

        register_new_frame_by_PnP(image_info_work, next_frame_idx, args)
        mask_track_for_outliers(image_info_work, next_frame_idx, args.pnp_reproj_thresh, min_track_number=1)
        

        # if not _refine_frame_pose_3d(image_info_work, next_frame_idx, args):
        #     image_info["invalid"][next_frame_idx] = image_info_work["invalid"][next_frame_idx] = True     
        #     print(f"[register_remaining_frames] Frame {next_frame_idx} marked as invalid due to 3D-3D correspondences refinement failure")
        #     invalid_cnt["3d_3d_corr"] += 1
        #     save_results(image_info=image_info, register_idx= image_info['frame_indices'][next_frame_idx], preprocessed_data=preprocessed_data, results_dir=output_dir / "pipeline_joint_opt", only_save_register_order=args.only_save_register_order)
        #     print_image_info_stats(image_info_work, invalid_cnt)
        #     continue
        
        sucess, mean_error = check_reprojection_error(image_info_work, next_frame_idx, args)
        if not sucess:
            print(f"[register_remaining_frames] Frame {next_frame_idx} align depth to mesh due to high reprojection error")
            # Reset pose to nearest registered frame
            registered = np.asarray(image_info_work["registered"]).astype(bool)
            invalid = np.asarray(image_info_work["invalid"]).astype(bool)
            valid_reg = registered & (~invalid)
            reg_idx = np.where(valid_reg)[0]
            if reg_idx.size > 0:
                nearest_idx = reg_idx[np.argmin(np.abs(reg_idx - next_frame_idx))]
                image_info_work["extrinsics"][next_frame_idx] = image_info_work["extrinsics"][nearest_idx].copy()
                print(f"[reproj_recovery] Reset frame {next_frame_idx} pose to nearest registered frame {nearest_idx}")

            # # Mask tracks without valid (finite) 3D points
            # finite_3d = np.isfinite(image_info_work["points_3d"]).all(axis=-1)
            # image_info_work["track_mask"][next_frame_idx] = (
            #     np.asarray(image_info_work["track_mask"][next_frame_idx]).astype(bool) & finite_3d
            # ).astype(image_info_work["track_mask"].dtype)

            # Align the frame with SAM3D mesh using depth with outlier rejection
            if sam3d_mesh is not None:
                print(f"[register_remaining_frames] Aligning frame {next_frame_idx} with SAM3D mesh using depth")
                sucess = _align_frame_with_mesh_depth(image_info_work, next_frame_idx, sam3d_mesh, 
                                             debug_dir=output_dir / "pipeline_joint_opt" / f"debug_frame_{image_info_work['frame_indices'][next_frame_idx]:04d}_{image_info_work['registered'].sum():04d}"
                                             )
                if not sucess:
                    image_info["invalid"][next_frame_idx] = image_info_work["invalid"][next_frame_idx] = True
                    print(f"[register_remaining_frames] Frame {next_frame_idx} marked as invalid due to large reprojection error and failed depth-mesh alignment")
                    invalid_cnt["reproj_err"] += 1
                    save_results(image_info=image_info, register_idx= image_info['frame_indices'][next_frame_idx], preprocessed_data=preprocessed_data, results_dir=output_dir / "pipeline_joint_opt", only_save_register_order=args.only_save_register_order)
                    print_image_info_stats(image_info_work, invalid_cnt)
                    continue
                else:
                    sucess, mean_error = check_reprojection_error(image_info_work, next_frame_idx, args, skip_check=True)
                    print(f"[register_remaining_frames] After depth-mesh alignment, reprojection error for frame {next_frame_idx}: {mean_error:.2f}")


        
                # mask_track_for_outliers(image_info_work, next_frame_idx, args.pnp_reproj_thresh, min_track_number=1)


        image_info_work["registered"][next_frame_idx] = True
        print(f"Successfully registered frame {image_info['frame_indices'][next_frame_idx]}")
        key_frame_min_reproj_thresh = 2.0
        if mean_error <= key_frame_min_reproj_thresh:
            if check_key_frame(
                image_info_work,
                next_frame_idx,
                rot_thresh=args.kf_rot_thresh,
                trans_thresh=args.kf_trans_thresh,
                depth_thresh=args.kf_depth_thresh,
                frame_inliner_thresh=args.kf_inlier_thresh,
            ):
                try:
                    image_info_work = process_key_frame(image_info_work, next_frame_idx, args)
                except Exception as exc:
                    print(f"[register_remaining_frames] process_key_frame failed: {exc}")

                # Resume NeuS optimization with new keyframe
                # if neus_ckpt is not None:
                if 0:
                    try:
                        kf_mask = image_info_work["keyframe"].astype(bool)
                        kf_local_indices = np.where(kf_mask)[0]
                        prepare_neus_data(
                            keyframe_indices=kf_local_indices.tolist(),
                            images=[preprocessed_data["images"][i] for i in kf_local_indices],
                            masks=[preprocessed_data["masks_obj"][i] for i in kf_local_indices],
                            depths=[preprocessed_data["depths"][i] for i in kf_local_indices],
                            extrinsics_o2c=image_info_work["extrinsics"][kf_local_indices],
                            intrinsics=image_info_work["intrinsics"][kf_local_indices],
                            neus_data_dir=neus_data_dir,
                        )
                        neus_total_steps += 300
                        neus_ckpt, neus_mesh = run_neus_training(
                            neus_data_dir,
                            config_path="configs/neus-pipeline.yaml",
                            max_steps=neus_total_steps,
                            checkpoint_path=neus_ckpt,
                            output_dir=output_dir / "pipeline_joint_opt" / "neus_training",
                            sam3d_root_dir=sam3d_root_dir,
                            robust_hoi_weight=1.0,
                            sam3d_weight=0.03,
                        )
                        frame_id = image_info['frame_indices'][next_frame_idx]
                        save_neus_mesh(neus_mesh, output_dir / "pipeline_joint_opt" / f"{frame_id:04d}")
                        latest_neus_mesh = neus_mesh
                    except Exception as exc:
                        print(f"[register_remaining_frames] NeuS resume failed: {exc}")
        else:
            print(f"Frame {image_info['frame_indices'][next_frame_idx]} not marked as keyframe due to high reprojection error ({mean_error:.2f} > {key_frame_min_reproj_thresh})")     



        # Joint optimize keyframe poses + 3D points against NeuS mesh
        can_joint_opt = (latest_neus_mesh is not None or args.no_optimize_with_point_to_plane)
        if can_joint_opt and image_info_work["keyframe"].sum() >= args.min_track_number:
            try:
                mesh_path = None if args.no_optimize_with_point_to_plane else latest_neus_mesh
                _joint_optimize_keyframes(
                    image_info_work, mesh_path, cond_local_idx,
                    min_track_number=args.min_track_number,
                )
                # Mask tracks with reprojection error > joint_opt_reproj_thresh
                kf_indices_arr = np.where(image_info_work["keyframe"].astype(bool))[0]
                for ki in kf_indices_arr:
                    mask_track_for_outliers(image_info_work, ki, args.joint_opt_reproj_thresh,
                                           min_track_number=args.min_track_number)
            except Exception as exc:
                print(f"[register_remaining_frames] joint optimization failed: {exc}")

        image_info["register"] = image_info_work["registered"].tolist()
        image_info["invalid"] = image_info_work["invalid"].tolist()
        image_info["keyframe"] = image_info_work["keyframe"].tolist()
        image_info["c2o"] = np.linalg.inv(image_info_work["extrinsics"]).astype(np.float32)
        image_info["points_3d"] = image_info_work["points_3d"].astype(np.float32)
        print_image_info_stats(image_info_work, invalid_cnt)
        save_results(image_info=image_info, register_idx= image_info['frame_indices'][next_frame_idx], preprocessed_data=preprocessed_data, results_dir=output_dir / "pipeline_joint_opt", only_save_register_order=args.only_save_register_order)

    save_results(image_info=image_info, register_idx= image_info['frame_indices'][next_frame_idx], preprocessed_data=preprocessed_data, results_dir=output_dir / "pipeline_joint_opt")


def main(args):
    log_file = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        log_dir = Path(args.output_dir) / "pipeline_joint_opt"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "log.txt"
        log_file = open(log_path, "a", buffering=1)
        sys.stdout = TeeStream(orig_stdout, log_file)
        sys.stderr = TeeStream(orig_stderr, log_file)
        print(f"[logging] Writing console output to {log_path}")

        data_dir = Path(args.data_dir)
        out_dir = Path(args.output_dir)
        cond_idx = args.cond_index

        SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
        data_preprocess_dir = data_dir / "pipeline_preprocess"
        tracks_dir = data_dir / "pipeline_corres"

        (
            frame_indices,
            preprocessed_data,
            tracks,
            vis_scores,
            tracks_mask,
            cond_cam_to_obj,
            cond_local_idx,
        ) = prepare_joint_opt_inputs(
            data_preprocess_dir=data_preprocess_dir,
            tracks_dir=tracks_dir,
            sam3d_dir=SAM3D_dir,
            cond_idx=cond_idx,
        )

        # 5. Lift 2D tracks to 3D points using depth and transformation
        points_3d, c2o_per_frame = register_first_frame(
            tracks=tracks,
            tracks_mask=tracks_mask,
            preprocessed=preprocessed_data,
            frame_indices=frame_indices,
            cond_local_idx=cond_local_idx,
            cond_cam_to_obj=cond_cam_to_obj,
        )

        # mark the condition frame as keyframe and register frame
        keyframe_flags = [i == cond_local_idx for i in range(len(frame_indices))]
        register_flags = keyframe_flags.copy()
        invalid_flags = [False] * len(frame_indices)


        # 6. Build image info
        image_info = {
            'frame_indices': frame_indices,
            'cond_idx': cond_idx,
            "tracks": tracks.astype(np.float32),
            "vis_scores": vis_scores.astype(np.float32),
            "tracks_mask": tracks_mask.astype(bool),
            "keyframe": keyframe_flags,
            "register": register_flags,
            "invalid": invalid_flags,
            "points_3d": points_3d.astype(np.float32),
            "c2o": c2o_per_frame.astype(np.float32),
            "intrinsics": _stack_intrinsics(preprocessed_data["intrinsics"]),
        }

        # 6. Save image info
        print("Building and saving image info...")
        save_results(image_info=image_info, register_idx=cond_idx, preprocessed_data=preprocessed_data, results_dir=out_dir / "pipeline_joint_opt")

        # 7. Load NeuS checkpoint from pipeline_neus_init.py output
        neus_ckpt = None
        neus_init_mesh = None
        neus_total_steps = args.neus_init_steps
        sam3d_root_dir = SAM3D_dir / f"{cond_idx:04d}"

        if not args.no_optimize_with_point_to_plane:
            from robust_hoi_pipeline.neus_integration import _find_latest_checkpoint, _find_latest_mesh
            neus_training_dir = out_dir / "pipeline_neus_init" / "neus_training"
            neus_ckpt = _find_latest_checkpoint(neus_training_dir)
            neus_init_mesh = _find_latest_mesh(neus_training_dir)

            if neus_ckpt is None:
                print(f"[WARNING] No NeuS checkpoint found in {neus_training_dir}. "
                      "Run pipeline_neus_init.py first. NeuS resume will be skipped.")
        else:
            print("[INFO] Point-to-plane disabled. Skipping NeuS mesh/checkpoint loading.")

        # 8. Register remaining frames with incremental NeuS
        register_remaining_frames(
            image_info, preprocessed_data, out_dir, cond_idx,
            neus_ckpt=neus_ckpt, neus_total_steps=neus_total_steps,
            sam3d_root_dir=sam3d_root_dir, neus_init_mesh=neus_init_mesh,
            no_optimize_with_point_to_plane=args.no_optimize_with_point_to_plane,
        )
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if log_file is not None:
            log_file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint optimization pipeline for HOI reconstruction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")
    parser.add_argument("--neus_init_steps", type=int, default=10000,
                        help="Number of NeuS training steps used in pipeline_neus_init.py (for resuming)")
    parser.add_argument("--no_optimize_with_point_to_plane", action="store_true", default=True,
                        help="Disable point-to-plane loss and skip NeuS mesh loading")

    args = parser.parse_args()
    main(args)
