# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Frame registration and keyframe management functions for the COLMAP pipeline.
"""

import numpy as np
import torch


def find_next_frame(image_info):
    """Select next unregistered frame to process.

    Args:
        image_info: Dictionary containing registered and invalid frame flags

    Returns:
        Index of next frame to process, or -1 if all processed
    """
    registered = image_info.get("registered", np.array([]))
    invalid = image_info.get("invalid", np.array([]))

    for i in range(len(registered)):
        if not registered[i] and not invalid[i]:
            return i
def find_next_frame(image_info):
    track_mask = image_info.get("track_mask")
    registered = image_info.get("registered")
    invalid = image_info.get("invalid")
    if track_mask is None or registered is None or invalid is None:
        return None

    track_mask = np.asarray(track_mask)
    registered = np.asarray(registered)
    invalid = np.asarray(invalid)

    if registered.ndim != 1:
        registered = registered.reshape(-1)
    if invalid.ndim != 1:
        invalid = invalid.reshape(-1)

    num_frames = track_mask.shape[0]
    registered_mask = registered & (~invalid)
    if not np.any(registered_mask):
        return None

    # tracks visible in any registered frame
    vis_in_registered = track_mask[registered_mask].any(axis=0)

    best_idx = None
    best_count = -1
    for idx in range(num_frames):
        if registered[idx] or invalid[idx]:
            continue
        count = np.count_nonzero(track_mask[idx] & vis_in_registered)
        if count > best_count:
            best_count = count
            best_idx = idx
    return best_idx


def check_frame_invalid(image_info, frame_idx, min_inlier_per_frame=10, min_depth_pixels=500):
    """Check if frame has insufficient inliers or depth data.

    Args:
        image_info: Dictionary containing track_mask and depth_priors
        frame_idx: Frame index to check
        min_inlier_per_frame: Minimum required track inliers
        min_depth_pixels: Minimum required valid depth pixels

    Returns:
        True if frame is invalid, False otherwise
    """
    track_mask = image_info.get("track_mask")
    depth_priors = image_info.get("depth_priors")

    if track_mask is not None:
        track_inliers = int(np.count_nonzero(track_mask[frame_idx]))
        if track_inliers < min_inlier_per_frame:
            print(f"[check_frame_invalid] Frame {frame_idx} invalid: insufficient track inliers ({track_inliers} < {min_inlier_per_frame})")
            return True

    if depth_priors is not None:
        depth_map = depth_priors[frame_idx]
        if torch.is_tensor(depth_map):
            depth_map = depth_map.detach().cpu().numpy()
        valid_depth = int(np.count_nonzero(np.asarray(depth_map) > 0))
        if valid_depth < min_depth_pixels:
            print(f"[check_frame_invalid] Frame {frame_idx} invalid: insufficient depth pixels ({valid_depth} < {min_depth_pixels})")
            return True

    return False


def check_key_frame(image_info, frame_idx, rot_thresh, trans_thresh, depth_thresh, frame_inliner_thresh):
    """Heuristically decide if frame should become a keyframe based on validity + pose delta.

    Args:
        image_info: Dictionary containing extrinsics, keyframes, depth_priors, track_mask
        frame_idx: Frame index to check
        rot_thresh: Minimum rotation delta threshold (degrees)
        trans_thresh: Minimum translation delta threshold
        depth_thresh: Minimum depth pixel count
        frame_inliner_thresh: Minimum track inlier count

    Returns:
        True if frame should be a keyframe, False otherwise
    """
    registered = image_info.get("registered")
    extrinsics = image_info.get("extrinsics")
    keyframes = image_info.get("keyframe")
    depth_priors = image_info.get("depth_priors")
    track_mask = image_info.get("track_mask")

    registered = np.asarray(registered).astype(bool)
    if not registered[frame_idx]:
        print(f"[check_key_frame] Frame {frame_idx} is not registered; cannot be keyframe.")
        return False

    # Basic validity checks
    if depth_priors is not None:
        depth_map = depth_priors[frame_idx]
        if torch.is_tensor(depth_map):
            depth_map = depth_map.detach().cpu().numpy()
        valid_depth = int(np.count_nonzero(np.asarray(depth_map) > 0))
        if valid_depth < depth_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient depth pixels ({valid_depth} < {depth_thresh}).")
            return False

    if track_mask is not None:
        tm = np.asarray(track_mask)
        if frame_idx < tm.shape[0]:
            if int(np.count_nonzero(tm[frame_idx])) < frame_inliner_thresh:
                print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient track inliers.")
                return False

    if keyframes is None:
        return True  # no keyframes tracked yet

    keyframes = np.asarray(keyframes).astype(bool)
    past_keys = np.where(keyframes & registered & (np.arange(len(keyframes)) < frame_idx))[0]
    if len(past_keys) == 0:
        print(f"[check_key_frame] Frame {frame_idx} accepted as first keyframe.")
        return True

    T_curr = extrinsics[frame_idx]
    R_curr, t_curr = T_curr[:3, :3], T_curr[:3, 3]

    for kf_idx in past_keys:
        T_prev = extrinsics[kf_idx]
        R_prev, t_prev = T_prev[:3, :3], T_prev[:3, 3]
        R_delta = R_curr @ R_prev.T
        angle = np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)))
        trans = np.linalg.norm(t_curr - t_prev)

        if angle < rot_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient rotation delta ({angle:.2f} < {rot_thresh}).")
            return False

        if trans < trans_thresh:
            print(f"[check_key_frame] Frame {frame_idx} rejected: insufficient translation delta ({trans:.3f} < {trans_thresh}).")
            return False

    return True


def process_key_frame(image_info, frame_idx, args):
    """Process frame designated as keyframe (trigger BA if needed).

    Args:
        image_info: Dictionary containing reconstruction data
        frame_idx: Frame index to process
        args: Arguments with configuration

    Returns:
        Updated image_info
    """
    # Import here to avoid circular dependency
    from .optimization import bundle_adjust_keyframes

    print(f"[process_key_frame] Processing keyframe at frame {frame_idx}")
    image_info["keyframe"][frame_idx] = True

    # Run bundle adjustment on all keyframes
    image_info = bundle_adjust_keyframes(
        image_info,
        ref_frame_idx=args.cond_index,
        iters=30,
        lr=1e-3,
    )

    return image_info


def register_remaining_frames(image_info, gen_3d, args):
    """Register all remaining frames in the sequence.

    Args:
        image_info: Dictionary containing reconstruction data
        gen_3d: Generated 3D model object
        args: Arguments with configuration

    Returns:
        Updated image_info with all frames registered
    """
    # Import here to avoid circular dependency
    from .visualization_io import save_results
    from .optimization import register_new_frame

    num_images = len(image_info["images"])
    image_info["registered"] = np.array([False] * num_images)
    image_info["registered"][args.cond_index] = True

    image_info["invalid"] = np.array([False] * num_images)

    image_info["keyframe"] = np.array([False] * num_images)
    image_info["keyframe"][args.cond_index] = True

    save_results(image_info, gen_3d, out_dir=f"{args.output_dir}/results/{args.cond_index:04d}/")

    while image_info["registered"].sum() + image_info["invalid"].sum() < num_images:
        next_frame_idx = find_next_frame(image_info)
        print("+" * 50)
        print(f"Next frame to register: {next_frame_idx}, registered: {image_info['registered'].sum()}, "
              f"keyframes: {image_info['keyframe'].sum()}, invalid: {image_info['invalid'].sum()}")

        if check_frame_invalid(
            image_info, next_frame_idx,
            min_inlier_per_frame=args.min_inlier_per_frame,
            min_depth_pixels=args.min_depth_pixels
        ):
            image_info["invalid"][next_frame_idx] = True
            continue

        # Register the frame
        register_new_frame(
            image_info, gen_3d, next_frame_idx, args,
            out_dir=f"{args.output_dir}/results/{next_frame_idx:04d}/"
        )
        image_info["registered"][next_frame_idx] = True

        # Check if this frame should be a keyframe
        if check_key_frame(
            image_info, next_frame_idx,
            rot_thresh=args.kf_rot_thresh,
            trans_thresh=args.kf_trans_thresh,
            depth_thresh=args.kf_depth_thresh,
            frame_inliner_thresh=args.kf_inlier_thresh
        ):
            image_info = process_key_frame(image_info, next_frame_idx, args)

        save_results(image_info, gen_3d, out_dir=f"{args.output_dir}/results/{next_frame_idx:04d}/")

    return image_info
