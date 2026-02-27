import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as ScipyRotation


import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import get_depth, depth2xyzmap
from utils_simba.render import diff_renderer, projection_matrix_from_intrinsics
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_sam3d_transform
from robust_hoi_pipeline.geometry_utils import compute_reproj_errors
import os
RUN_ON_PC = os.getenv("RUN_ON_PC", "").lower() == "true"
device = "cuda"

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
        - hand_meshes: list of hand meshes {'vertices': (V,3), 'faces': (F,3)} or None
    """
    from PIL import Image
    from utils_simba.depth import get_depth, get_normal
    import trimesh

    data = {
        'frame_indices': frame_indices,
        'images': [],
        'masks_obj': [],
        'masks_hand': [],
        'depths': [],
        'normals': [],
        'intrinsics': [],
        'hand_meshes_right': [],
        'hand_meshes_left': [],
        'hand_poses': [],
        'hand_c2o': [],
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
        else:
            data['intrinsics'].append(None)

        # Load sealed hand mesh generated in preprocessing.
        for key in ["right", "left"]:
            hand_mesh_path = data_preprocess_dir / "hand" / f"{frame_idx:04d}_{key}.obj"
            if hand_mesh_path.exists():
                try:
                    hand_mesh = trimesh.load(str(hand_mesh_path), process=False, force="mesh")
                    data[f'hand_meshes_{key}'].append(
                        {
                            "vertices": np.asarray(hand_mesh.vertices, dtype=np.float32),
                            "faces": np.asarray(hand_mesh.faces, dtype=np.int32),
                        }
                    )
                except Exception:
                    data[f'hand_meshes_{key}'].append(None)
            else:
                data[f'hand_meshes_{key}'].append(None)

        # Load hand pose parameters (hand_pose, hand_rot, hand_trans) from preprocessing.
        hand_pose_path = data_preprocess_dir / "hand_pose" / f"{frame_idx:04d}.npz"
        if hand_pose_path.exists():
            try:
                hp = np.load(hand_pose_path)
                hand_rot = hp['hand_rot'].astype(np.float32)
                hand_o2c = np.eye(4, dtype=np.float32)
                if hand_rot.shape == (3,):
                    hand_o2c[:3, :3] = ScipyRotation.from_rotvec(hand_rot).as_matrix().astype(np.float32)
                else:
                    hand_o2c[:3, :3] = hand_rot
                hand_o2c[:3, 3] = hp['hand_trans'].astype(np.float32)
                hand_c2o = np.linalg.inv(hand_o2c)
                data['hand_c2o'].append(hand_c2o)
            except Exception:
                data['hand_c2o'].append(None)
        else:
            data['hand_c2o'].append(None)
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
    vis_thresh: float,
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
    tracks_mask = tracks_mask & (vis_scores >= vis_thresh)
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

    # Align hand c2o poses with cond_cam_to_obj at condition frame (right-multiplication)
    hand_c2o = preprocessed.get('hand_c2o')
    hand_o2c = np.linalg.inv(np.array(hand_c2o, dtype=np.float64))
    cond_obj_to_cam = np.linalg.inv(cond_cam_to_obj.astype(np.float64))
    if hand_o2c is not None and cond_local_idx < len(hand_o2c) and hand_o2c[cond_local_idx] is not None:
        align_tf = np.linalg.inv(hand_o2c[cond_local_idx].astype(np.float64)) @ cond_obj_to_cam.astype(np.float64)
        for i in range(len(hand_o2c)):
            if hand_o2c[i] is not None:
                hand_o2c[i] = (hand_o2c[i].astype(np.float64) @ align_tf).astype(np.float32)
        print(f"Aligned hand o2c poses with cond_cam_to_obj at condition frame {cond_idx}")
    preprocessed['hand_c2o'] = np.linalg.inv(hand_o2c)

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

    errs, proj_2d_all = compute_reproj_errors(pts_3d, pts_2d, o2c, K)

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
        text1 = f"frame={register_idx} local={local_idx}"
        text2 = (f"mean={valid_errs.mean():.2f}px max={valid_errs.max():.2f}px "
                 f"val_n/pts_n {len(valid_errs)}/{points_3d.shape[0]}")
        cv2.putText(vis_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(vis_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
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
    only_save_register_order = True
    if RUN_ON_PC:
        only_save_register_order = False
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
        only_save_register_order=only_save_register_order,
    )


def _stack_intrinsics(intrinsics_list: List[np.ndarray]) -> np.ndarray:
    """Stack intrinsics, filling missing entries with the first valid matrix."""
    valid = [K for K in intrinsics_list if K is not None]
    if not valid:
        raise ValueError("No valid intrinsics found.")
    fallback = valid[0]
    stacked = [K if K is not None else fallback for K in intrinsics_list]
    return np.stack(stacked, axis=0)


def print_frame_reproj_error(image_info_work, frame_idx, tag="joint_opt"):
    """Print reprojection error stats for a single frame."""
    fm = np.asarray(image_info_work["track_mask"][frame_idx]).astype(bool)
    pts_3d = image_info_work.get("points_3d")
    if pts_3d is None or not fm.any():
        return
    ext = image_info_work["extrinsics"][frame_idx]
    K = image_info_work["intrinsics"][frame_idx] if image_info_work["intrinsics"].ndim == 3 else image_info_work["intrinsics"]
    p3 = np.asarray(pts_3d)[fm].astype(np.float64)
    p2 = np.asarray(image_info_work["pred_tracks"][frame_idx])[fm].astype(np.float64)
    fin = np.isfinite(p3).all(axis=1)
    if not fin.any():
        return
    errs, _ = compute_reproj_errors(p3[fin], p2[fin], ext, K)
    valid_errs = errs[np.isfinite(errs)]
    if len(valid_errs) > 0:
        print(f"[{tag}] Frame {frame_idx}: reproj_error mean={valid_errs.mean():.2f} "
              f"median={np.median(valid_errs):.2f} max={valid_errs.max():.2f} "
              f"({len(valid_errs)}/{fm.sum()} pts)")


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

    errs, _ = compute_reproj_errors(pts_3d[finite], pts_2d[finite], ext, K)
    errs = np.nan_to_num(errs, nan=0.0)

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


def _save_obj_depth_points_debug(debug_dir, frame_idx, pts_obj, filename_prefix, color_rgba, it=None):
    if debug_dir is None or pts_obj is None or len(pts_obj) == 0:
        return

    import trimesh as _trimesh

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    colors_obj = np.zeros((len(pts_obj), 4), dtype=np.uint8)
    colors_obj[:, :] = np.array(color_rgba, dtype=np.uint8)
    suffix = f"_iter_{it:03d}" if it is not None else ""
    _trimesh.PointCloud(pts_obj.astype(np.float32), colors=colors_obj).export(
        _debug_dir / f"{filename_prefix}_frame_{frame_idx:04d}{suffix}.ply"
    )


def _backproject_depth_torch(depth, K, mask):
    """Back-project masked depth pixels to 3D camera-space points (differentiable)."""
    vs, us = torch.where(mask)
    zs = depth[vs, us]
    xs = (us.float() - K[0, 2]) * zs / K[0, 0]
    ys = (vs.float() - K[1, 2]) * zs / K[1, 1]
    return torch.stack([xs, ys, zs], dim=-1)  # (N, 3)


def _depth_map_to_obj_points(depth, K, R, trans, mask=None):
    """Back-project a depth map to 3D points in object space.

    Args:
        depth: (H, W) depth map (torch tensor).
        K: (3, 3) intrinsic matrix (numpy).
        R: (3, 3) rotation from object to camera (torch tensor).
        trans: (3,) translation from object to camera (torch tensor).
        mask: optional (H, W) boolean mask of valid pixels (torch tensor).

    Returns:
        pts_obj: (N, 3) numpy array of points in object space.
    """
    if mask is None:
        mask = (depth > 0) & torch.isfinite(depth)
    vs, us = torch.where(mask)
    zs = depth[vs, us]
    xs = (us.float() - K[0, 2]) * zs / K[0, 0]
    ys = (vs.float() - K[1, 2]) * zs / K[1, 1]
    pts_cam = torch.stack([xs, ys, zs], dim=-1)  # (N, 3)
    # cam_to_obj: p_obj = R^T @ (p_cam - t)
    pts_obj = (R.T @ (pts_cam - trans[None, :]).T).T
    return pts_obj.detach().cpu().numpy()


def _save_colored_points_debug(debug_dir, frame_idx, pts, image, ys, xs, filename_prefix):
    """Save a colored point cloud using pixel colors from the image."""
    if debug_dir is None or pts is None or len(pts) == 0:
        return

    import trimesh as _trimesh

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)

    if image is not None and image.ndim == 3 and image.shape[2] >= 3:
        colors = image[ys, xs, :3].copy()
        if np.issubdtype(colors.dtype, np.floating) and colors.max(initial=0.0) <= 1.0:
            colors = (colors * 255.0).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        colors_rgba = np.zeros((len(pts), 4), dtype=np.uint8)
        colors_rgba[:, :3] = colors
        colors_rgba[:, 3] = 255
    else:
        colors_rgba = np.full((len(pts), 4), [128, 128, 128, 255], dtype=np.uint8)

    _trimesh.PointCloud(pts.astype(np.float32), colors=colors_rgba).export(
        _debug_dir / f"{filename_prefix}_frame_{frame_idx:04d}.ply"
    )


def _save_normal_map_debug(debug_dir, frame_idx, normal_map, filename_prefix, it=None):
    if debug_dir is None or normal_map is None:
        return

    from PIL import Image
    import torch

    if torch.is_tensor(normal_map):
        normal_np = normal_map.detach().cpu().numpy()
    else:
        normal_np = np.asarray(normal_map)
    if normal_np.ndim != 3 or normal_np.shape[2] != 3:
        return
    normal_np = (normal_np + 1.0) * 0.5
    normal_u8 = np.clip(normal_np * 255.0, 0, 255).astype(np.uint8)
    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    if it is None:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}.png"
    else:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}_iter{it:03d}.png"
    Image.fromarray(normal_u8, mode="RGB").save(_debug_dir / out_name)


def _save_binary_mask_debug(debug_dir, frame_idx, mask, filename_prefix, it=None):
    if debug_dir is None or mask is None:
        return

    from PIL import Image

    if torch.is_tensor(mask):
        mask_np = mask.detach().float().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    if mask_np.ndim != 2:
        return

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    if it is None:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}.png"
    else:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}_iter{it:03d}.png"
    Image.fromarray((np.clip(mask_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(_debug_dir / out_name)


def _save_mesh_debug(debug_dir, frame_idx, vertices, faces, filename_prefix, it=None):
    if debug_dir is None or vertices is None or faces is None:
        return
    import trimesh as _trimesh

    if torch.is_tensor(vertices):
        verts_np = vertices.detach().cpu().numpy()
    else:
        verts_np = np.asarray(vertices)
    if torch.is_tensor(faces):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = np.asarray(faces)

    if verts_np.ndim == 3:
        verts_np = verts_np[0]
    if faces_np.ndim != 2 or verts_np.ndim != 2:
        return

    _debug_dir = Path(debug_dir)
    _debug_dir.mkdir(parents=True, exist_ok=True)
    if it is None:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}.obj"
    else:
        out_name = f"{filename_prefix}_frame_{frame_idx:04d}_iter{it:03d}.obj"
    _trimesh.Trimesh(vertices=verts_np.astype(np.float32), faces=faces_np.astype(np.int32), process=False).export(
        _debug_dir / out_name
    )


def _smooth_normal_map_masked(normal_map, valid_mask, num_iters=2, kernel_size=3, eps=1e-6):
    """Smooth an HxWx3 normal map with masked local averaging and renormalization."""
    import torch
    import torch.nn.functional as F

    if normal_map is None:
        return None

    normal_valid = torch.as_tensor(valid_mask, dtype=torch.float32, device=normal_map.device)[None, None]
    normal_chw = normal_map.permute(2, 0, 1)[None]

    for _ in range(int(num_iters)):
        smoothed_sum = F.avg_pool2d(normal_chw * normal_valid, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        smoothed_w = F.avg_pool2d(normal_valid, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        normal_chw = torch.where(
            smoothed_w > eps,
            smoothed_sum / (smoothed_w + eps),
            normal_chw,
        )
        normal_chw = F.normalize(normal_chw, dim=1, eps=eps)
        normal_chw = torch.where(normal_valid > 0, normal_chw, torch.zeros_like(normal_chw))

    return normal_chw[0].permute(1, 2, 0)


def _depth_pixels_to_cam_points(depth_map, K, ys, xs):
    """Back-project selected depth pixels to camera-space points (N, 3)."""
    zs = depth_map[ys, xs].astype(np.float64)
    xc = (xs.astype(np.float64) - K[0, 2]) * zs / K[0, 0]
    yc = (ys.astype(np.float64) - K[1, 2]) * zs / K[1, 1]
    return np.stack([xc, yc, zs], axis=-1)


def _cam_points_to_object_points(pts_cam, extrinsic):
    """Transform camera-space points to object space using o2c extrinsic."""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return (R.T @ (pts_cam - t).T).T


def _build_observation_mask(height, width, ys, xs, device):
    """Create boolean HxW observation mask from selected pixel indices."""
    import torch

    obs_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    obs_mask[torch.tensor(ys, device=device), torch.tensor(xs, device=device)] = True
    return obs_mask


def _filter_pixels_in_object_bbox(depth_map, K, extrinsics, frame_idx, ys, xs, bbox_min=None, bbox_max=None):
    """Keep only depth pixels whose object-space points are inside a 3D bbox."""
    ext0 = extrinsics[frame_idx].astype(np.float64)
    pts_cam = _depth_pixels_to_cam_points(depth_map, K, ys, xs)
    pts_obj0 = _cam_points_to_object_points(pts_cam, ext0)

    if bbox_min is None:
        bbox_min = np.array([-0.8, -0.8, -0.8], dtype=np.float64)
    else:
        bbox_min = np.asarray(bbox_min, dtype=np.float64)
    if bbox_max is None:
        bbox_max = np.array([0.8, 0.8, 0.8], dtype=np.float64)
    else:
        bbox_max = np.asarray(bbox_max, dtype=np.float64)

    in_bbox = np.all((pts_obj0 >= bbox_min) & (pts_obj0 <= bbox_max), axis=1)
    return ys[in_bbox], xs[in_bbox], ext0


def _filter_depth_by_object_bbox(image_info_work, frame_idx, bbox_min=[-0.8, -0.8, -0.8], bbox_max=[0.8, 0.8, 0.8]):
    """Zero out depth pixels whose object-space 3D points fall outside a bounding box."""
    depth_priors = image_info_work.get("depth_priors")
    if depth_priors is None or depth_priors[frame_idx] is None:
        return

    d = depth_priors[frame_idx]
    intrinsics = image_info_work.get("intrinsics")
    extrinsics = image_info_work.get("extrinsics")
    K = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics

    vmask = d > 0
    masks = image_info_work.get("image_masks")
    if masks is not None and masks[frame_idx] is not None:
        vmask = vmask & (masks[frame_idx] > 0)

    ys, xs = np.where(vmask)
    if len(ys) == 0:
        return

    pts_cam = _depth_pixels_to_cam_points(d, K, ys, xs)
    ext = extrinsics[frame_idx].astype(np.float64)
    pts_obj = _cam_points_to_object_points(pts_cam, ext)


    bbox_min = np.array(bbox_min, dtype=np.float64)

    bbox_max = np.array(bbox_max, dtype=np.float64)


    outside = ~np.all((pts_obj >= bbox_min) & (pts_obj <= bbox_max), axis=1)
    if outside.any():
        d[ys[outside], xs[outside]] = 0
        print(f"[filter_depth_bbox] Frame {frame_idx}: zeroed {int(outside.sum())}/{len(ys)} depth pixels outside object bbox")


def _prepare_frame_observations(
    image_info_work,
    frame_idx,
    torch_device=None,
    normal_smooth_iters=5,
    normal_kernel_size=5,
    normal_eps=1e-6,
    debug_dir=None,
):
    """Collect per-frame observations and optionally prepare smoothed normal tensor."""
    depth_priors = image_info_work.get("depth_priors")
    normal_priors = image_info_work.get("normal_priors")
    intrinsics = image_info_work.get("intrinsics")
    extrinsics = image_info_work.get("extrinsics")

    d = depth_priors[frame_idx]
    if d is None:
        print(f"[align_depth] Frame {frame_idx}: no depth, skipping")
        return None

    K = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics
    n_obs = normal_priors[frame_idx] if (normal_priors is not None and frame_idx < len(normal_priors)) else None

    vmask = d > 0.01
    masks = image_info_work.get("image_masks")
    if masks is not None and masks[frame_idx] is not None:
        vmask = vmask & (masks[frame_idx] > 0)
    if n_obs is not None:
        vmask = vmask & np.isfinite(n_obs).all(axis=-1) & (np.linalg.norm(n_obs, axis=-1) > 1e-6)

    ys, xs = np.where(vmask)

    obs_normal = None
    if n_obs is not None and torch_device is not None:
        n_obs_f = n_obs.astype(np.float32)
        obs_normal = torch.tensor(n_obs_f, dtype=torch.float32, device=torch_device)
        obs_normal = F.normalize(obs_normal, dim=-1)
        obs_normal = _smooth_normal_map_masked(
            obs_normal,
            vmask,
            num_iters=normal_smooth_iters,
            kernel_size=normal_kernel_size,
            eps=normal_eps,
        )
        _save_normal_map_debug(
            debug_dir=debug_dir,
            frame_idx=frame_idx,
            normal_map=obs_normal,
            filename_prefix="normal_prior_smoothed",
        )

    return d, K, n_obs, vmask, masks, ys, xs, extrinsics, obs_normal


def _prepare_render_inputs(mesh, K, H, W, device):
    """Prepare renderer context, mesh tensors, and camera projection tensor."""
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
    return glctx, verts, tri, vnormals, color, projection





def _build_observed_hoi_mask(image_info_work, frame_idx, H, W, device, debug_dir=None):
    """Build observed HOI mask as union of object/hand masks."""
    mask_union = np.zeros((H, W), dtype=bool)
    masks_obj = image_info_work.get("image_masks")
    masks_hand = image_info_work.get("image_masks_hand")

    if masks_obj is not None and frame_idx < len(masks_obj) and masks_obj[frame_idx] is not None:
        mask_union |= (np.asarray(masks_obj[frame_idx]) > 0)
    if masks_hand is not None and frame_idx < len(masks_hand) and masks_hand[frame_idx] is not None:
        mask_union |= (np.asarray(masks_hand[frame_idx]) > 0)

    if not mask_union.any():
        return None

    if debug_dir is not None:
        from PIL import Image

        _debug_dir = Path(debug_dir)
        _debug_dir.mkdir(parents=True, exist_ok=True)
        mask_u8 = (mask_union.astype(np.uint8) * 255)
        Image.fromarray(mask_u8, mode="L").save(_debug_dir / f"hoi_mask_union_frame_{frame_idx:04d}.png")

    return torch.tensor(mask_union.astype(np.float32), dtype=torch.float32, device=device)


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

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    # loss_val = (sigma_squared * x_squared) / (sigma_squared + x_squared + 1e-8)
    loss_val = (x_squared) / (sigma_squared + x_squared + 1e-8)
    return loss_val

def _compute_depth_loss(depth_r, obs_depth, obs_mask, K, sigma=0.1,
                        max_pts=2000, debug_ctx=None):
    """KNN-based depth loss between rendered and observed 3D point clouds.

    Each depth map is independently masked by its own valid pixels,
    back-projected to 3D camera space using intrinsics, then a bidirectional
    chamfer distance with GMoF robustifier is computed.

    Returns (loss, valid_count) or (None, 0) when too few valid pixels.
    """
    render_mask = (depth_r > 0) & torch.isfinite(depth_r)
    obs_valid = obs_mask & (obs_depth > 0) & torch.isfinite(obs_depth)

    render_count = int(render_mask.sum().item())
    obs_count = int(obs_valid.sum().item())
    valid_count = min(render_count, obs_count)

    if valid_count < 100:
        return None, valid_count

    # Back-project to 3D camera space
    pts_r = _backproject_depth_torch(depth_r, K, render_mask)
    pts_o = _backproject_depth_torch(obs_depth, K, obs_valid)


    # # Subsample for memory efficiency
    # if len(pts_r) > max_pts:
    #     idx = torch.randperm(len(pts_r), device=pts_r.device)[:max_pts]
    #     pts_r = pts_r[idx]
    # if len(pts_o) > max_pts:
    #     idx = torch.randperm(len(pts_o), device=pts_o.device)[:max_pts]
    #     pts_o = pts_o[idx]

    # Debug: save subsampled point clouds in object space
    if debug_ctx is not None:
        R = debug_ctx["R"]
        trans = debug_ctx["trans"]
        debug_dir = debug_ctx["debug_dir"]
        frame_idx = debug_ctx["frame_idx"]
        it = debug_ctx["it"]
        with torch.no_grad():
            pts_r_obj = (R.T @ (pts_r - trans[None, :]).T).T.cpu().numpy()
            pts_o_obj = (R.T @ (pts_o - trans[None, :]).T).T.cpu().numpy()
        _save_obj_depth_points_debug(
            debug_dir, frame_idx, pts_r_obj,
            "depth_rendered_obj", color_rgba=(0, 0, 255, 255), it=it,
        )
        _save_obj_depth_points_debug(
            debug_dir, frame_idx, pts_o_obj,
            "depth_observed_obj", color_rgba=(0, 255, 0, 255), it=it,
        )

    # Bidirectional chamfer with GMoF robustifier
    dists = torch.cdist(pts_r, pts_o)           # (Nr, No)
    nn_r2o = dists.min(dim=1).values             # rendered → observed
    nn_o2r = dists.min(dim=0).values             # observed → rendered
    # loss = gmof(nn_r2o, sigma=sigma).mean() + gmof(nn_o2r, sigma=sigma).mean()
    loss = nn_r2o.mean() + nn_o2r.mean()

    # # KNN correspondences: for each observed point find nearest rendered point,
    # # filter by max distance, then compute L2 loss on matched pairs.
    # dists = torch.cdist(pts_o, pts_r)              # (No, Nr)
    # nn_dist, nn_idx = dists.min(dim=1)             # (No,) nearest rendered for each observed
    # close = nn_dist < 0.05
    # if close.sum() == 0:
    #     return torch.tensor(0.0, device=depth_r.device), valid_count
    # corr_r = pts_r[nn_idx[close]]                  # matched rendered points
    # corr_dist = (pts_o[close] - corr_r).norm(dim=-1)
    # loss = gmof(corr_dist, sigma=sigma).mean()

    return loss, valid_count


def _compute_normal_loss(normal_r, obs_normal, valid):
    """Cosine-distance normal loss on valid pixels."""
    if obs_normal is None or normal_r is None:
        return torch.tensor(0.0, device=normal_r.device if normal_r is not None else "cuda")
    dot = (normal_r[valid] * obs_normal[valid]).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - dot).mean()


def _compute_iou_loss(render_union, obs_hoi_mask):
    """Silhouette IoU loss between rendered union mask and observed HOI mask."""
    if obs_hoi_mask is None:
        return torch.tensor(0.0, device=render_union.device)
    inter = (render_union * obs_hoi_mask).sum()
    union = (render_union + obs_hoi_mask - render_union * obs_hoi_mask).sum()
    iou = inter / (union + 1e-6)
    return 1.0 - iou


def _compute_reproj_loss(R, trans, trk_pts3d, trk_pts2d, K, sigma=4.0):
    """Reprojection loss from 3D track points projected via current pose.

    Returns scalar loss (0 when inputs are None or no points in front of camera).
    """
    if trk_pts3d is None:
        return torch.tensor(0.0, device=R.device)
    cam_trk = (R @ trk_pts3d.T).T + trans[None, :]
    front = cam_trk[:, 2] > 1e-6
    if front.sum() == 0:
        return torch.tensor(0.0, device=R.device)
    z = cam_trk[front, 2:3]
    proj_x = K[0, 0] * cam_trk[front, 0:1] / z + K[0, 2]
    proj_y = K[1, 1] * cam_trk[front, 1:2] / z + K[1, 2]
    proj = torch.cat([proj_x, proj_y], dim=-1)
    err = (proj - trk_pts2d[front]).norm(dim=-1)
    return gmof(err, sigma=sigma).mean()


def _prepare_track_reproj_data(image_info_work, frame_idx, K, device):
    """Extract valid 3D track points and their 2D observations for reprojection loss.

    Returns (trk_pts3d, trk_pts2d, K_tensor) or (None, None, None) if insufficient data.
    """
    pred_tracks_all = image_info_work.get("pred_tracks")
    track_mask_all = image_info_work.get("track_mask")
    points_3d_all = image_info_work.get("points_3d")

    if pred_tracks_all is None or track_mask_all is None or points_3d_all is None:
        return None, None, None

    frame_tmask = np.asarray(track_mask_all[frame_idx]).astype(bool)
    finite_mask = np.isfinite(points_3d_all).all(axis=-1)
    # Only use tracks with well-established 3D points (visible in multiple keyframes)
    well_observed = np.ones_like(finite_mask)
    kf_mask_arr = image_info_work.get("keyframe")
    if kf_mask_arr is not None:
        kf_indices_arr = np.where(np.asarray(kf_mask_arr).astype(bool))[0]
        if len(kf_indices_arr) > 0:
            trk_vis_count = np.asarray(track_mask_all)[kf_indices_arr].astype(bool).sum(axis=0)
            well_observed = trk_vis_count >= 2
    trk_valid = frame_tmask & finite_mask & well_observed
    if trk_valid.sum() < 10:
        return None, None, None

    trk_pts3d = torch.tensor(points_3d_all[trk_valid], dtype=torch.float32, device=device)
    trk_pts2d = torch.tensor(pred_tracks_all[frame_idx][trk_valid], dtype=torch.float32, device=device)
    K_tensor = torch.tensor(K.astype(np.float32), dtype=torch.float32, device=device)
    print(f"[align_depth] Frame {frame_idx}: using {int(trk_valid.sum())} tracks for reprojection loss")
    return trk_pts3d, trk_pts2d, K_tensor


def _save_depth_debug_stages(debug_dir, frame_idx, d, K, extrinsics, ys, xs, dbg_image):
    """Save 3D point clouds at raw and masked depth filtering stages for debugging."""
    # Raw 3D points (all depth > 0, before mask) in camera and object space
    raw_mask = d > 0
    raw_ys, raw_xs = np.where(raw_mask)
    if len(raw_ys) > 0:
        ext_raw = extrinsics[frame_idx].astype(np.float64)
        raw_pts_cam = _depth_pixels_to_cam_points(d, K, raw_ys, raw_xs)
        raw_pts_obj = _cam_points_to_object_points(raw_pts_cam, ext_raw)
        _save_colored_points_debug(debug_dir, frame_idx, raw_pts_cam, dbg_image, raw_ys, raw_xs, "depth_raw_in_cam")
        _save_colored_points_debug(debug_dir, frame_idx, raw_pts_obj, dbg_image, raw_ys, raw_xs, "depth_raw_in_obj")

    # Masked 3D points (after vmask, before bbox filter) in object space
    if len(ys) > 0:
        ext_masked = extrinsics[frame_idx].astype(np.float64)
        masked_pts_cam = _depth_pixels_to_cam_points(d, K, ys, xs)
        masked_pts_obj = _cam_points_to_object_points(masked_pts_cam, ext_masked)
        _save_colored_points_debug(debug_dir, frame_idx, masked_pts_obj, dbg_image, ys, xs, "depth_after_masked_in_obj")


def _align_frame_with_mesh_depth(image_info_work, frame_idx, obj_mesh, max_pts=2000, num_iters=100, inlier_thresh=0.3, debug_dir=None):
    """Align a frame by optimizing pose with rendered-vs-observed depth/normal losses."""


    del inlier_thresh  # kept in signature for compatibility

    frame_obs = _prepare_frame_observations(
        image_info_work,
        frame_idx,
        debug_dir=debug_dir,
    )
    if frame_obs is None:
        return False

    d, K, n_obs, vmask, masks, ys, xs, extrinsics, obs_normal = frame_obs

    # Debug: save depth points at various filtering stages
    _dbg_image = None
    if debug_dir is not None:
        images = image_info_work.get("images")
        if images is not None and frame_idx < len(images):
            _dbg_image = images[frame_idx]
        _save_depth_debug_stages(debug_dir, frame_idx, d, K, extrinsics, ys, xs, _dbg_image)

    ys, xs, ext0 = _filter_pixels_in_object_bbox(d, K, extrinsics, frame_idx, ys, xs)

    if debug_dir is not None and len(ys) > 0:
        bbox_pts_cam = _depth_pixels_to_cam_points(d, K, ys, xs)
        bbox_pts_obj = _cam_points_to_object_points(bbox_pts_cam, ext0)
        _save_colored_points_debug(debug_dir, frame_idx, bbox_pts_obj, _dbg_image, ys, xs, "depth_after_bbox_in_obj")

    if len(ys) < max_pts:
        print(
            f"[align_depth] Frame {frame_idx}: only {len(ys)} points in object bbox, "
            f"need at least {max_pts}, skipping"
        )
        return False

    if len(ys) > max_pts:
        sel = np.random.choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]

    pts_cam_sel = _depth_pixels_to_cam_points(d, K, ys, xs)

    if not torch.cuda.is_available():
        print(f"[align_depth] Frame {frame_idx}: CUDA unavailable, skipping depth optimization")
        return False

    H, W = d.shape
    glctx, obj_verts, obj_tri, vnormals, obj_color, projection = _prepare_render_inputs(obj_mesh, K, H, W, device)
    obs_hoi_mask = _build_observed_hoi_mask(image_info_work, frame_idx, H, W, device, debug_dir=debug_dir)
    obj_nv = int(obj_verts.shape[1])

    trk_pts3d_t, trk_pts2d_t, K_t = _prepare_track_reproj_data(image_info_work, frame_idx, K, device)

    # Load hand mesh in camera space (prefer preprocessed right-hand mesh, fallback to pose payload).
    hand_verts_in_cam = hand_tri = hand_color = None
    hand_meshes_right = image_info_work.get("hand_meshes_right")
    hv_np = hf_np = None
    if hand_meshes_right is not None and frame_idx < len(hand_meshes_right) and hand_meshes_right[frame_idx] is not None:
        hand_mesh_data = hand_meshes_right[frame_idx]
        hv_np = np.asarray(hand_mesh_data.get("vertices"), dtype=np.float32)
        hf_np = np.asarray(hand_mesh_data.get("faces"), dtype=np.int32)


    if hv_np is not None and hf_np is not None and hv_np.ndim == 2 and hf_np.ndim == 2:
        hand_verts_in_cam = torch.tensor(hv_np, dtype=torch.float32, device=device)[None]  # (1, Nh, 3), camera space
        hand_tri = torch.tensor(hf_np, dtype=torch.int32, device=device)  # (Fh, 3)
        hand_color = torch.ones((1, hv_np.shape[0], 3), dtype=torch.float32, device=device)


    obs_depth = torch.tensor(d.astype(np.float32), dtype=torch.float32, device=device)
    obs_mask = _build_observation_mask(H, W, ys, xs, device)

    init_rot = ScipyRotation.from_matrix(ext0[:3, :3]).as_rotvec().astype(np.float32)
    rotvec = torch.tensor(init_rot, dtype=torch.float32, device=device, requires_grad=True)
    trans = torch.tensor(ext0[:3, 3].astype(np.float32), dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([rotvec, trans], lr=2e-3)
    best = {"loss": float("inf"), "R": ext0[:3, :3].copy(), "t": ext0[:3, 3].copy(), "valid": 0}

    for it in range(num_iters):
        optimizer.zero_grad()
        R = rodrigues(rotvec)
        o2c = torch.eye(4, dtype=torch.float32, device=device)
        o2c[:3, :3] = R
        o2c[:3, 3] = trans


        obj_img, depth_r = diff_renderer(
            verts=obj_verts,
            tri=obj_tri,
            color=obj_color,
            projection=projection,
            ob_in_cvcams=o2c,
            resolution=np.asarray([H, W]),
            glctx=glctx,
        )

        # Render merged foreground mask from object mesh + hand mesh.
        hand_verts_in_obj = None

        # Transform hand vertices from camera space to object space using current pose.
        # For row vectors: v_obj = (v_cam - t) @ R, where o2c = [R|t].
        hand_verts_in_obj = ((hand_verts_in_cam[0] - trans[None, :]) @ R).unsqueeze(0)
        fg_verts = torch.cat([obj_verts, hand_verts_in_obj], dim=1)
        fg_tri = torch.cat([obj_tri, hand_tri + obj_nv], dim=0)
        fg_color = torch.cat([torch.ones_like(obj_color), hand_color], dim=1)
        fg_img, _ = diff_renderer(
            verts=fg_verts,
            tri=fg_tri,
            color=fg_color,
            projection=projection,
            ob_in_cvcams=o2c,
            resolution=np.asarray([H, W]),
            glctx=glctx,
        )
        sil_pred = fg_img[..., 1]  # Green channel as silhouette
        render_union = sil_pred


        depth_r = torch.flip(depth_r[0], dims=[0])
        _debug_ctx = None
        if debug_dir is not None and (it == 0 or (it + 1) % 5 == 0 or it == num_iters - 1):
            _debug_ctx = {"K": K, "R": R, "trans": trans, "debug_dir": debug_dir,
                          "frame_idx": frame_idx, "it": it + 1}
        loss_depth, valid_count = _compute_depth_loss(depth_r, obs_depth, obs_mask, K, debug_ctx=_debug_ctx)
        if loss_depth is None:
            print(f"[align_depth] Frame {frame_idx}: iter {it}, only {valid_count} valid rendered pixels")
            continue

        loss_iou = _compute_iou_loss(render_union, obs_hoi_mask)
        loss_reproj = _compute_reproj_loss(R, trans, trk_pts3d_t, trk_pts2d_t, K_t)

        w_depth = 1.0
        w_mask = 20.0
        w_reproj = 0.0

        loss = w_depth * loss_depth + w_mask * loss_iou + w_reproj * loss_reproj

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
            _save_binary_mask_debug(
                debug_dir=debug_dir,
                frame_idx=frame_idx,
                mask=render_union,
                filename_prefix="mask_render_union",
                it=it + 1,
            )
            if hand_verts_in_obj is not None:
                _save_mesh_debug(
                    debug_dir=debug_dir,
                    frame_idx=frame_idx,
                    vertices=hand_verts_in_obj,
                    faces=hand_tri,
                    filename_prefix="hand_mesh_obj",
                    it=it + 1,
                )
            print(
                f"[align_depth] Frame {frame_idx}: iter {it+1}/{num_iters}, total {loss.item():.3f} "
                f"loss_d={loss_depth.item():.3f}, "
                f"loss_iou={loss_iou.item():.3f}, loss_reproj={loss_reproj.item():.3f}, valid={valid_count}"
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

    # Store optimized depth points in object space
    depth_pts_obj = image_info_work.get("depth_points_obj")
    if depth_pts_obj is not None:
        depth_pts_obj[frame_idx] = pts_obj_opt.astype(np.float32)

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
    
def _reset_pose_to_nearest_registered(image_info_work, frame_idx):
    """Reset a frame's pose to that of the nearest valid registered frame."""
    registered = np.asarray(image_info_work["registered"]).astype(bool)
    invalid = np.asarray(image_info_work["invalid"]).astype(bool)
    valid_reg = registered & (~invalid)
    reg_idx = np.where(valid_reg)[0]
    if reg_idx.size > 0:
        nearest_idx = reg_idx[np.argmin(np.abs(reg_idx - frame_idx))]
        image_info_work["extrinsics"][frame_idx] = image_info_work["extrinsics"][nearest_idx].copy()
        print(f"[reproj_recovery] Reset frame {frame_idx} pose to nearest registered frame {nearest_idx}")


def _init_pose_from_hand(image_info_work, frame_idx):
    """Initialize frame pose using hand pose for frame_idx frame.


    Returns True if the pose was successfully initialized, False otherwise
    (caller should fall back to _reset_pose_to_nearest_registered).
    """
    hand_c2o = image_info_work.get("hand_c2o")
    if hand_c2o is None:
        return False

    hand_o2c = np.linalg.inv(hand_c2o[frame_idx])
    image_info_work["extrinsics"][frame_idx] = hand_o2c

    return True


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
        "hand_meshes_right": preprocessed_data.get("hand_meshes_right"),
        "hand_meshes_left": preprocessed_data.get("hand_meshes_left"),
        "hand_c2o": preprocessed_data.get("hand_c2o"),
        "images": preprocessed_data["images"],
        "image_masks": preprocessed_data.get("masks_obj"),
        "image_masks_hand": preprocessed_data.get("masks_hand"),
        "keyframe": np.array(image_info["keyframe"], dtype=bool),
        "registered": np.array(image_info["register"], dtype=bool),
        "invalid": np.array(image_info["invalid"], dtype=bool),
        "depth_points_obj": image_info.get("depth_points_obj", [None] * len(frame_indices)),
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
        # _filter_depth_by_object_bbox(image_info_work, next_frame_idx)
        

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
            # _reset_pose_to_nearest_registered(image_info_work, next_frame_idx)
            _init_pose_from_hand(image_info_work, next_frame_idx)
            mask_track_for_outliers(image_info_work, next_frame_idx, args.pnp_reproj_thresh, min_track_number=1)

            # # Mask tracks without valid (finite) 3D points
            # finite_3d = np.isfinite(image_info_work["points_3d"]).all(axis=-1)
            # image_info_work["track_mask"][next_frame_idx] = (
            #     np.asarray(image_info_work["track_mask"][next_frame_idx]).astype(bool) & finite_3d
            # ).astype(image_info_work["track_mask"].dtype)

            # Align the frame with SAM3D mesh using depth with outlier rejection
            debug_dir = None
            if RUN_ON_PC:
                debug_dir = output_dir / "pipeline_joint_opt" / f"debug_frame_{image_info_work['frame_indices'][next_frame_idx]:04d}_{image_info_work['registered'].sum():04d}"
            # if sam3d_mesh is not None:
            if 0:
                print(f"[register_remaining_frames] Aligning frame {next_frame_idx} with SAM3D mesh using depth")
                sucess = _align_frame_with_mesh_depth(image_info_work, next_frame_idx, sam3d_mesh, 
                                             debug_dir=debug_dir
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
        # if mean_error <= key_frame_min_reproj_thresh:
        if 1:
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
                # Print reprojection error after joint optimization for the newly registered frame
                kf_indices_arr = np.where(image_info_work["keyframe"].astype(bool))[0]
                print_frame_reproj_error(image_info_work, next_frame_idx, tag="joint_opt")
                # Mask tracks with reprojection error > joint_opt_reproj_thresh
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
        image_info["depth_points_obj"] = image_info_work["depth_points_obj"]
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
            vis_thresh=args.vis_thresh,
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
            "depth_points_obj": [None] * len(frame_indices),
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
    parser.add_argument("--vis_thresh", type=float, default=0.3,
                        help="Visibility score threshold for filtering tracks in the condition frame")

    args = parser.parse_args()
    main(args)
