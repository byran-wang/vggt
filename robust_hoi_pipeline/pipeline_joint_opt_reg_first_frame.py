import argparse
from pathlib import Path

import numpy as np

from robust_hoi_pipeline.pipeline_joint_opt_utils import (
    prepare_joint_opt_inputs,
    register_first_frame,
    save_results,
    _stack_intrinsics,
)


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

    # 7. Run initial NeuS optimization on condition frame
    from robust_hoi_pipeline.neus_integration import prepare_neus_data, run_neus_training, save_neus_mesh

    neus_data_dir = out_dir / "neus_data"
    sam3d_root_dir = SAM3D_dir / f"{cond_idx:04d}"
    kf_indices = [cond_local_idx]
    o2c_cond = np.linalg.inv(c2o_per_frame[kf_indices]).astype(np.float32)
    K_cond = _stack_intrinsics(preprocessed_data["intrinsics"])[kf_indices]

    prepare_neus_data(
        keyframe_indices=kf_indices,
        images=[preprocessed_data["images"][i] for i in kf_indices],
        masks=[preprocessed_data["masks_obj"][i] for i in kf_indices],
        depths=[preprocessed_data["depths"][i] for i in kf_indices],
        extrinsics_o2c=o2c_cond,
        intrinsics=K_cond,
        neus_data_dir=neus_data_dir,
    )
    neus_ckpt, neus_mesh = run_neus_training(
        neus_data_dir,
        config_path="configs/neus-pipeline.yaml",
        max_steps=10000,
        checkpoint_path=None,
        output_dir=out_dir / "pipeline_joint_opt",
        sam3d_root_dir=sam3d_root_dir,
    )

    save_neus_mesh(neus_mesh, out_dir / "pipeline_joint_opt" / f"{cond_idx:04d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint optimization pipeline - register first frame")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")

    args = parser.parse_args()
    main(args)
