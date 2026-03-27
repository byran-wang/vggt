import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt import (
    load_preprocessed_data,
    _stack_intrinsics,
)
from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform


def _load_joint_opt_image_info(joint_opt_dir: Path, excluded_global=False) -> dict:
    """Load the latest saved joint-opt image info.

    `pipeline_joint_opt.py` currently writes split compressed outputs:
    `shared_info.pkl.gz` plus `<register_idx>/image_info.pkl.gz`. Keep legacy
    `.npy` fallback for older runs.
    """
    register_indices = load_register_indices(joint_opt_dir)
    if not register_indices:
        raise RuntimeError(f"No register indices found in {joint_opt_dir / 'register_order.txt'}")

    shared_gz = joint_opt_dir / "shared_info.pkl.gz"
    shared_npy = joint_opt_dir / "shared_info.npy"

    for register_idx in reversed(register_indices):
        if excluded_global and register_idx >= 9900:
            continue
        frame_dir = joint_opt_dir / f"{register_idx:04d}"
        gz_path = frame_dir / "image_info.pkl.gz"
        npy_path = frame_dir / "image_info.npy"

        if gz_path.exists():
            with gzip.open(gz_path, "rb") as f:
                image_info = pickle.load(f)
            image_info_path = gz_path
        elif npy_path.exists():
            image_info = np.load(npy_path, allow_pickle=True).item()
            image_info_path = npy_path
        else:
            continue

        if shared_gz.exists():
            with gzip.open(shared_gz, "rb") as f:
                shared_info = pickle.load(f)
            shared_info.update(image_info)
            image_info = shared_info
        elif shared_npy.exists():
            shared_info = np.load(shared_npy, allow_pickle=True).item()
            shared_info.update(image_info)
            image_info = shared_info

        return image_info, register_idx, image_info_path

    raise FileNotFoundError(
        f"No saved image info found under {joint_opt_dir}. "
        "Expected one of <frame>/image_info.pkl.gz or <frame>/image_info.npy."
    )


def _select_registered_frame_subset(image_info, joint_opt_dir: Path, max_registered_frames: int):
    frame_indices_all = list(image_info["frame_indices"])
    selected_local_indices = np.arange(len(frame_indices_all), dtype=np.int64)

    if max_registered_frames > 0:
        register_order = load_register_indices(joint_opt_dir)
        frame_to_local = {frame_idx: i for i, frame_idx in enumerate(frame_indices_all)}
        registered_flags = np.asarray(image_info.get("register", []), dtype=bool)
        seen = set()
        selected_local_list = []
        for frame_idx in register_order:
            if frame_idx in seen:
                continue
            local_idx = frame_to_local.get(frame_idx)
            if local_idx is None:
                continue
            if registered_flags.size and not registered_flags[local_idx]:
                continue
            selected_local_list.append(local_idx)
            seen.add(frame_idx)
            if len(selected_local_list) >= max_registered_frames:
                break
        if selected_local_list:
            selected_local_indices = np.asarray(selected_local_list, dtype=np.int64)

    frame_indices = [frame_indices_all[i] for i in selected_local_indices]
    keyframe_flags = np.asarray(image_info["keyframe"], dtype=bool)[selected_local_indices]
    keyframe_local_indices = np.where(keyframe_flags)[0]
    return frame_indices, selected_local_indices, keyframe_local_indices


def _load_gt_o2c(data_dir: Path, frame_indices):
    """Load GT object-to-camera extrinsics from ho3d_v3 processed data.

    Args:
        data_dir: HO3D sequence directory (e.g., ho3d_v3/train/MC1).
        frame_indices: List of frame indices to extract.

    Returns:
        gt_o2c: (N, 4, 4) float32 GT object-to-camera matrices.
        is_valid: (N,) bool validity flags per frame.
    """
    seq_name = data_dir.name
    # Look for processed .pt relative to ho3d_v3 root
    # data_dir is e.g. .../ho3d_v3/train/MC1, so ho3d_root = data_dir.parent.parent
    ho3d_root = data_dir.parent.parent
    pt_path = ho3d_root / "processed" / f"{seq_name}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"GT data not found at {pt_path}")

    data = torch.load(pt_path, map_location="cpu")
    obj_rot = data["obj_rot"]      # (total_frames, 3, 3)
    obj_trans = data["obj_trans"]   # (total_frames, 3)
    is_valid_all = data["is_valid"] # (total_frames,)

    num_total = obj_rot.shape[0]
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int64)

    # Build o2c: eye(4), set rot+trans, then flip y,z (OpenGL→OpenCV)
    o2c_all = torch.eye(4).unsqueeze(0).repeat(num_total, 1, 1)
    o2c_all[:, :3, :3] = obj_rot
    o2c_all[:, :3, 3] = obj_trans
    o2c_all[:, 1:3] *= -1

    gt_o2c = o2c_all[frame_indices_arr].numpy().astype(np.float32)
    is_valid = is_valid_all[frame_indices_arr].numpy().astype(bool)
    return gt_o2c, is_valid


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    result_dir = Path(args.result_dir)
    cond_idx = args.cond_index

    sam3d_root_dir = data_dir / "SAM3D_aligned_post_process" / f"{cond_idx:04d}"
    data_preprocess_dir = data_dir / "pipeline_preprocess"
    joint_opt_dir = result_dir / "pipeline_joint_opt"

    print("Loading latest image info from pipeline_joint_opt...")
    image_info, last_register_idx, image_info_path = _load_joint_opt_image_info(joint_opt_dir)
    print(f"Loaded image info from {image_info_path} (register_idx={last_register_idx:04d})")

    if args.gt_pose:
        # Use all frames from preprocessed data (not just registered ones from joint-opt)
        from robust_hoi_pipeline.pipeline_utils import load_frame_list
        frame_indices = load_frame_list(data_preprocess_dir)
        print(f"GT pose mode: using all {len(frame_indices)} preprocessed frames")

        preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)

        # Use GT extrinsics from HO3D processed data
        print("Using GT poses for NeuS initialization...")
        gt_o2c, gt_valid = _load_gt_o2c(data_dir, frame_indices)

        # Filter to only valid GT poses
        valid_mask = gt_valid.astype(bool)
        if not np.any(valid_mask):
            raise RuntimeError("No valid GT poses found.")
        o2c_keyframes = gt_o2c[valid_mask]
        print(f"GT poses: {valid_mask.sum()}/{len(valid_mask)} frames have valid GT poses")

        # Get scale from aligned_transform.json in sam3d_root_dir and apply to o2c translations
        from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform
        sam3d_transform = load_sam3d_transform(sam3d_root_dir.parent, cond_idx)
        sam3d_scale = sam3d_transform['scale']
        # GT o2c translations are in camera-space meters; pipeline works in SAM3D-normalized space
        # where cam_translation = sam3d_translation * scale, so divide by scale
        o2c_keyframes[:, :3, 3] /= sam3d_scale
        print(f"Applied SAM3D scale={sam3d_scale:.6f} to GT o2c translations")

        intrinsics_all = _stack_intrinsics(preprocessed_data["intrinsics"])
        K_keyframes = intrinsics_all[valid_mask]

        images_keyframes = [preprocessed_data["images"][i] for i, v in enumerate(valid_mask) if v]
        masks_keyframes = [preprocessed_data["masks_obj"][i] for i, v in enumerate(valid_mask) if v]
        masks_hand_keyframes = [preprocessed_data["masks_hand"][i] for i, v in enumerate(valid_mask) if v] if "masks_hand" in preprocessed_data else None
        depths_keyframes = [preprocessed_data["depths"][i] for i, v in enumerate(valid_mask) if v]
    else:
        print("Loading preprocessed data for image_info frames...")
        frame_indices, selected_local_indices, keyframe_local_indices = _select_registered_frame_subset(
            image_info,
            joint_opt_dir,
            args.max_registered_frames,
        )
        if keyframe_local_indices.size == 0:
            raise RuntimeError("No keyframes found in latest image_info.")

        preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)

        # Use keyframe extrinsics from latest joint-opt image_info.
        if "c2o" not in image_info:
            raise KeyError("Latest image_info is missing 'c2o'.")
        c2o_all = np.asarray(image_info["c2o"], dtype=np.float32)[selected_local_indices]
        o2c_all = np.linalg.inv(c2o_all).astype(np.float32)
        o2c_keyframes = o2c_all[keyframe_local_indices]
        valid_mask = None

        if "intrinsics" in image_info:
            intrinsics_all = np.asarray(image_info["intrinsics"], dtype=np.float32)[selected_local_indices]
        else:
            intrinsics_all = _stack_intrinsics(preprocessed_data["intrinsics"])
        K_keyframes = intrinsics_all[keyframe_local_indices]

        images_keyframes = [preprocessed_data["images"][i] for i in keyframe_local_indices]
        masks_keyframes = [preprocessed_data["masks_obj"][i] for i in keyframe_local_indices]
        masks_hand_keyframes = [preprocessed_data["masks_hand"][i] for i in keyframe_local_indices] if "masks_hand" in preprocessed_data else None
        depths_keyframes = [preprocessed_data["depths"][i] for i in keyframe_local_indices]

        # Filter to valid GT poses if using gt_pose
        if valid_mask is not None:
            K_keyframes = K_keyframes[valid_mask]
            images_keyframes = [img for img, v in zip(images_keyframes, valid_mask) if v]
            masks_keyframes = [m for m, v in zip(masks_keyframes, valid_mask) if v]
            if masks_hand_keyframes is not None:
                masks_hand_keyframes = [m for m, v in zip(masks_hand_keyframes, valid_mask) if v]
            depths_keyframes = [d for d, v in zip(depths_keyframes, valid_mask) if v]

    neus_data_dir = out_dir / "neus_data"
    

    from robust_hoi_pipeline.neus_integration import prepare_neus_data, run_neus_training, save_neus_mesh

    prepare_neus_data(
        images=images_keyframes,
        masks=masks_keyframes,
        depths=depths_keyframes,
        extrinsics_o2c=o2c_keyframes,
        intrinsics=K_keyframes,
        neus_data_dir=neus_data_dir,
        masks_hand=masks_hand_keyframes,
    )

    neus_ckpt, neus_mesh, _ = run_neus_training(
        neus_data_dir,
        config_path="configs/neus-pipeline.yaml",
        max_steps=args.max_steps,
        checkpoint_path=None,
        output_dir=out_dir / "neus_training",
        sam3d_root_dir=sam3d_root_dir,
        robust_hoi_weight=args.robust_hoi_weight,
        sam3d_weight=args.sam3d_weight,
        export_only=args.export_only,
    )

    # if neus_mesh:
    #     save_neus_mesh(neus_mesh, out_dir / "pipeline_joint_opt" / f"{cond_idx:04d}")

    print(f"NeuS init complete. Checkpoint: {neus_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuS initialization with condition frame")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2/ which includes SAM3D_aligned_post_process/)")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing latest pipeline_joint_opt results (used to load latest image_info)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Number of NeuS training steps")
    parser.add_argument("--robust_hoi_weight", type=float, default=1.0,
                        help="Weight for robust HOI loss")
    parser.add_argument("--sam3d_weight", type=float, default=0.5,
                        help="Weight for SAM3D loss")
    parser.add_argument("--max_registered_frames", type=int, default=-1,
                        help="If > -1, only use the first N valid registered frames when preparing NeuS data")
    parser.add_argument("--gt_pose", action="store_true", default=False,
                        help="Use GT object poses from ho3d_v3/processed/{seq_name}.pt instead of joint-opt poses")
    parser.add_argument("--export_only", action="store_true", default=False,
                        help="Skip training, load latest checkpoint and export mesh only")

    args = parser.parse_args()
    main(args)
