import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt import (
    load_preprocessed_data,
    _stack_intrinsics,
)
from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir.parent / "pipeline_preprocess"
    joint_opt_dir = out_dir.parent / "pipeline_joint_opt"

    print("Loading latest image info from pipeline_joint_opt...")
    register_indices = load_register_indices(joint_opt_dir)
    if not register_indices:
        raise RuntimeError(f"No register indices found in {joint_opt_dir / 'register_order.txt'}")
    last_register_idx = register_indices[-1]
    image_info_path = joint_opt_dir / f"{last_register_idx:04d}" / "image_info.npy"
    if not image_info_path.exists():
        raise FileNotFoundError(f"Latest image info not found: {image_info_path}")
    image_info = np.load(image_info_path, allow_pickle=True).item()
    print(f"Loaded image info from {image_info_path}")

    frame_indices = list(image_info["frame_indices"])
    keyframe_flags = np.array(image_info["keyframe"], dtype=bool)
    keyframe_local_indices = np.where(keyframe_flags)[0]
    if keyframe_local_indices.size == 0:
        raise RuntimeError("No keyframes found in latest image_info.")

    print("Loading preprocessed data for image_info frames...")
    preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)

    print("Loading SAM3D transformation...")
    sam3d_transform = load_sam3d_transform(SAM3D_dir, cond_idx)
    cond_cam_to_obj = np.eye(4, dtype=np.float32)
    cond_cam_to_obj[:3, :3] = sam3d_transform['scale'] * sam3d_transform['cond_cam_to_sam3d'][:3, :3]
    cond_cam_to_obj[:3, 3] = sam3d_transform['cond_cam_to_sam3d'][:3, 3]

    # Use keyframe extrinsics/intrinsics from latest joint-opt image_info.
    if "c2o" not in image_info:
        raise KeyError("Latest image_info is missing 'c2o'.")
    c2o_all = np.asarray(image_info["c2o"], dtype=np.float32)
    o2c_all = np.linalg.inv(c2o_all).astype(np.float32)
    o2c_keyframes = o2c_all[keyframe_local_indices]

    if "intrinsics" in image_info:
        intrinsics_all = np.asarray(image_info["intrinsics"], dtype=np.float32)
    else:
        intrinsics_all = _stack_intrinsics(preprocessed_data["intrinsics"])
    K_keyframes = intrinsics_all[keyframe_local_indices]

    images_keyframes = [preprocessed_data["images"][i] for i in keyframe_local_indices]
    masks_keyframes = [preprocessed_data["masks_obj"][i] for i in keyframe_local_indices]
    depths_keyframes = [preprocessed_data["depths"][i] for i in keyframe_local_indices]

    neus_data_dir = out_dir / "neus_data"
    sam3d_root_dir = SAM3D_dir / f"{cond_idx:04d}"

    from robust_hoi_pipeline.neus_integration import prepare_neus_data, run_neus_training, save_neus_mesh

    prepare_neus_data(
        keyframe_indices=keyframe_local_indices.tolist(),
        images=images_keyframes,
        masks=masks_keyframes,
        depths=depths_keyframes,
        extrinsics_o2c=o2c_keyframes,
        intrinsics=K_keyframes,
        neus_data_dir=neus_data_dir,
    )

    neus_ckpt, neus_mesh = run_neus_training(
        neus_data_dir,
        config_path="configs/neus-pipeline.yaml",
        max_steps=args.max_steps,
        checkpoint_path=None,
        output_dir=out_dir / "neus_training",
        sam3d_root_dir=sam3d_root_dir,
        robust_hoi_weight=0.1,
        sam3d_weight=1.0,
    )

    # if neus_mesh:
    #     save_neus_mesh(neus_mesh, out_dir / "pipeline_joint_opt" / f"{cond_idx:04d}")

    print(f"NeuS init complete. Checkpoint: {neus_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuS initialization with condition frame")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2/ which includes SAM3D_aligned_post_process/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Number of NeuS training steps")

    args = parser.parse_args()
    main(args)
