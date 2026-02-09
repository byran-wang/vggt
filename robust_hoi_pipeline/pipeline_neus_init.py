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
from robust_hoi_pipeline.pipeline_joint_opt import prepare_joint_opt_inputs, _stack_intrinsics, load_preprocessed_data, load_sam3d_transform



def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_idx = args.cond_index

    SAM3D_dir = data_dir / "SAM3D_aligned_post_process"
    data_preprocess_dir = out_dir.parent / "pipeline_preprocess"
    tracks_dir = out_dir / "pipeline_corres"
 


    
    
    frame_indices = [cond_idx]
    cond_local_idx = frame_indices.index(cond_idx)
    print("Loading preprocessed data...")
    preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)
    

    print("Loading SAM3D transformation...")
    sam3d_transform = load_sam3d_transform(SAM3D_dir, cond_idx)
    cond_cam_to_obj = np.eye(4, dtype=np.float32)
    cond_cam_to_obj[:3, :3] = sam3d_transform['scale'] * sam3d_transform['cond_cam_to_sam3d'][:3, :3]
    cond_cam_to_obj[:3, 3] = sam3d_transform['cond_cam_to_sam3d'][:3, 3]

    neus_data_dir = out_dir / "neus_data"
    sam3d_root_dir = SAM3D_dir / f"{cond_idx:04d}"
    kf_indices = [cond_local_idx]
    o2c_cond = cond_cam_to_obj
    K_cond = _stack_intrinsics(preprocessed_data["intrinsics"])[kf_indices]

    from robust_hoi_pipeline.neus_integration import prepare_neus_data, run_neus_training, save_neus_mesh

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint optimization pipeline for HOI reconstruction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2/ which includes SAM3D_aligned_post_process/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")    

    args = parser.parse_args()
    main(args)
