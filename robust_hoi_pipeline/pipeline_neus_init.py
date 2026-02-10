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
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir.parent / "pipeline_preprocess"

    frame_indices = [cond_idx]
    cond_local_idx = 0
    print("Loading preprocessed data...")
    preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)

    print("Loading SAM3D transformation...")
    sam3d_transform = load_sam3d_transform(SAM3D_dir, cond_idx)
    cond_cam_to_obj = np.eye(4, dtype=np.float32)
    cond_cam_to_obj[:3, :3] = sam3d_transform['scale'] * sam3d_transform['cond_cam_to_sam3d'][:3, :3]
    cond_cam_to_obj[:3, 3] = sam3d_transform['cond_cam_to_sam3d'][:3, 3]

    # o2c = inv(c2o), shape (1, 4, 4) for prepare_neus_data
    o2c_cond = np.linalg.inv(cond_cam_to_obj).astype(np.float32)[np.newaxis]
    K_cond = _stack_intrinsics(preprocessed_data["intrinsics"])  # (1, 3, 3)

    neus_data_dir = out_dir / "neus_data"
    sam3d_root_dir = SAM3D_dir / f"{cond_idx:04d}"

    from robust_hoi_pipeline.neus_integration import prepare_neus_data, run_neus_training, save_neus_mesh

    prepare_neus_data(
        keyframe_indices=[cond_local_idx],
        images=[preprocessed_data["images"][cond_local_idx]],
        masks=[preprocessed_data["masks_obj"][cond_local_idx]],
        depths=[preprocessed_data["depths"][cond_local_idx]],
        extrinsics_o2c=o2c_cond,
        intrinsics=K_cond,
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
