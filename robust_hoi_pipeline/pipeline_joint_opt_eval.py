import numpy as np

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

import vggt.utils.eval_modules as eval_m
import vggt.utils.gt as gt
from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform
device = "cuda:0"

eval_fn_dict = {
    "add": eval_m.eval_add_object,
    "add_s": eval_m.eval_add_s_object,
    "add_auc": eval_m.eval_add_auc_object,
    "add_s_auc": eval_m.eval_add_s_auc_object,
    # "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
    # "cd_f_ra": eval_m.eval_cd_f_ra,
    # "cd_f_right": eval_m.eval_cd_f_right,
}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="")
    parser.add_argument("--SAM3D_dir", type=str, required=True, help="Path to SAM3D_aligned_post_process directory")
    parser.add_argument("--cond_index", type=int, required=True,help="Condition frame index")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--debug", default=False, action="store_true")
    
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def main():
    from tqdm import tqdm
    args = parse_args()
    import vggt.utils.ours as ours

    # Load image info from the last register step determined by register_order.txt
    results_dir = Path(args.result_folder)
    register_indices = load_register_indices(results_dir)
    last_register_idx = register_indices[-1]
    image_info = np.load(
        results_dir / f"{last_register_idx:04d}" / "image_info.npy",
        allow_pickle=True,
    ).item()

    
    SAM3D_dir = Path(args.SAM3D_dir)
    sam3d_transform = load_sam3d_transform(SAM3D_dir, args.cond_index)
    sam3d_to_cond_cam = sam3d_transform['sam3d_to_cond_cam']
    scale = sam3d_transform['scale']

    # Reconstruct data_pred from image info
    frame_indices = np.array(image_info["frame_indices"])
    register_flags = np.array(image_info["register"], dtype=bool)
    invalid_flags = np.array(image_info["invalid"], dtype=bool)
    valid_flags = register_flags & ~invalid_flags

    c2o = np.array(image_info["c2o"])  # (N, 4, 4) camera-to-object (SAM3D scaled)
    # Transform from SAM3D object space to scene space (condition camera frame)
    R_s2c = sam3d_to_cond_cam[:3, :3]  # (3, 3)
    t_s2c = sam3d_to_cond_cam[:3, 3]   # (3,)
    extrinsics = np.zeros_like(c2o)
    extrinsics[:, :3, 3] = (R_s2c @ c2o[:, :3, 3, None]).squeeze(-1) + t_s2c
    extrinsics[:, :3, :3] = (R_s2c / scale) @ c2o[:, :3, :3]

    valid_extrinsics = extrinsics[valid_flags]
    valid_frame_indices = frame_indices[valid_flags]

    seq_name = results_dir.parent.name

    data_pred = {
        "extrinsics": valid_extrinsics,
        "is_valid": np.ones(len(valid_frame_indices), dtype=np.float32),
        "full_seq_name": seq_name,
    }

    # Return registered & valid frame indices for GT data selection
    def get_image_fids():
        return valid_frame_indices.tolist()

    data_gt = gt.load_data(seq_name, get_image_fids)        
    
    
    out_p = args.out_dir
    os.makedirs(out_p, exist_ok=True)


    print("------------------")
    print("Involving the following eval_fn:")
    for eval_fn_name in eval_fn_dict.keys():
        print(eval_fn_name)
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(eval_fn_dict.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + "/metric.json"
    npy_path = out_p + "/metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm), F-score (percentage), MPJPE (mm)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()
