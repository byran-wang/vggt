import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import get_depth, depth2xyzmap
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_sam3d_transform


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


def save_results(image_info: Dict, register_idx, preprocessed_data, results_dir: Path
) -> None:
    """Save image info for joint optimization outputs."""
    results_dir = results_dir / f"{register_idx:04d}"
    results_dir.mkdir(parents=True, exist_ok=True)
    info_path = results_dir / "image_info.npy"
    np.save(info_path, image_info)
    print(f"Saved image info to {results_dir}")

    from robust_hoi_pipeline.frame_management import save_register_order
    save_register_order(results_dir / "../" , register_idx)

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
        min_track_number=5,
        kf_rot_thresh=5.0,
        kf_trans_thresh=0.02,
        kf_depth_thresh=500,
        kf_inlier_thresh=10,
        run_ba_on_keyframe=0,
        unc_thresh=4.0,
        duplicate_track_thresh=3.0,
        pnp_reproj_thresh=3.0,
    )


def _stack_intrinsics(intrinsics_list: List[np.ndarray]) -> np.ndarray:
    """Stack intrinsics, filling missing entries with the first valid matrix."""
    valid = [K for K in intrinsics_list if K is not None]
    if not valid:
        raise ValueError("No valid intrinsics found.")
    fallback = valid[0]
    stacked = [K if K is not None else fallback for K in intrinsics_list]
    return np.stack(stacked, axis=0)


def mask_track_for_outliers(image_info, frame_idx, reproj_thresh):
    """Mask tracks whose reprojection error exceeds a threshold for a given frame.

    After a frame is registered via PnP, this reprojects 3D points onto the frame
    and sets track_mask to 0 for tracks with reprojection error > reproj_thresh.
    """
    track_mask = image_info["track_mask"]
    pred_tracks = image_info["pred_tracks"]
    points_3d = image_info.get("points_3d")

    frame_mask = np.asarray(track_mask[frame_idx]).astype(bool)
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


def register_remaining_frames(image_info, preprocessed_data, output_dir: Path, cond_idx: int):

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

    args = _build_default_joint_opt_args(output_dir, cond_idx)

    frame_indices = image_info["frame_indices"]
    cond_local_idx = frame_indices.index(cond_idx)
    c2o = image_info.get("c2o")
    if c2o is None:
        c2o = np.tile(np.eye(4, dtype=np.float32), (len(frame_indices), 1, 1))
    extrinsics = np.linalg.inv(c2o).astype(np.float32)

    intrinsics = _stack_intrinsics(preprocessed_data["intrinsics"])
    depth_priors = preprocessed_data["depths"]
    points_3d_global = image_info["points_3d"].astype(np.float32)

    image_info_work = {
        "pred_tracks": image_info["tracks"],
        "track_mask": image_info["tracks_mask"],
        "points_3d": points_3d_global,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depth_priors": depth_priors,
        "images": preprocessed_data["images"],
        "image_masks": preprocessed_data.get("masks_obj"),
        "keyframe": np.array(image_info["keyframe"], dtype=bool),
        "registered": np.array(image_info["register"], dtype=bool),
        "invalid": np.array(image_info["invalid"], dtype=bool),
    }

    num_frames = len(frame_indices)

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
            image_info_work["invalid"][next_frame_idx] = True
            continue

        register_new_frame_by_PnP(image_info_work, next_frame_idx, args)
        mask_track_for_outliers(image_info_work, next_frame_idx, args.pnp_reproj_thresh)
        image_info_work["registered"][next_frame_idx] = True

        if not _refine_frame_pose_3d(image_info_work, next_frame_idx, args):
            image_info_work["invalid"][next_frame_idx] = True
            print(f"[register_remaining_frames] Pose refinement failed for frame {next_frame_idx}")
        
        if check_reprojection_error(image_info_work, next_frame_idx, args):
            image_info_work["invalid"][next_frame_idx] = True
            print(f"[register_remaining_frames] High reprojection error, marking frame {next_frame_idx} as invalid")


        if not image_info_work["invalid"][next_frame_idx]:
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

        print(
            f"registered: {image_info_work['registered'].sum()}, "
            f"keyframes: {image_info_work['keyframe'].sum()}, "
            f"invalid: {image_info_work['invalid'].sum()}"
        )

        image_info["register"] = image_info_work["registered"].tolist()
        image_info["invalid"] = image_info_work["invalid"].tolist()
        image_info["keyframe"] = image_info_work["keyframe"].tolist()
        image_info["c2o"] = np.linalg.inv(image_info_work["extrinsics"]).astype(np.float32)
        image_info["points_3d"] = image_info_work["points_3d"].astype(np.float32)
        save_results(image_info=image_info, register_idx= image_info['frame_indices'][next_frame_idx], preprocessed_data=preprocessed_data, results_dir=output_dir / "pipeline_joint_opt")

def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir / "pipeline_preprocess"
    tracks_dir = out_dir / "pipeline_corres"

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
    
    register_remaining_frames(image_info, preprocessed_data, out_dir, cond_idx)




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
