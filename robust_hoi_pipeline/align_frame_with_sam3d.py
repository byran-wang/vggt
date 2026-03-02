"""Offline runner for _align_frame_with_sam3d.

Load saved inputs (pickle) and run _align_frame_with_sam3d for debugging.

Usage:
    python robust_hoi_pipeline/align_frame_with_sam3d.py \
        --input_pkl output/MC1/pipeline_joint_opt/debug_align_frame_inputs/align_frame_inputs_0001.pkl \
        [--frame_idx 1] \
        [--debug_dir output/MC1/pipeline_joint_opt/debug_offline]
"""
import argparse
import pickle
import sys
from pathlib import Path

# Setup paths (same as pipeline_joint_opt.py)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt import _align_frame_with_sam3d


class _DictBackedList:
    """Wrap a {int: value} dict so it supports list-style indexing and len()."""
    def __init__(self, d, total_len):
        self._d = d
        self._len = total_len

    def __getitem__(self, idx):
        return self._d.get(idx)

    def __len__(self):
        return self._len


def _rebuild_image_info_work(slim_info, frame_idx):
    """Convert slim (dict-keyed per-frame) format back to list-indexable format."""
    n_frames = len(slim_info["frame_indices"])
    per_frame_keys = [
        "depth_priors", "normal_priors", "images",
        "image_masks", "image_masks_hand",
        "hand_meshes_right", "depth_points_obj",
    ]
    rebuilt = dict(slim_info)
    for key in per_frame_keys:
        val = slim_info.get(key)
        if isinstance(val, dict):
            rebuilt[key] = _DictBackedList(val, n_frames)
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description="Offline runner for _align_frame_with_sam3d")
    parser.add_argument("--input_pkl", type=str, required=True,
                        help="Path to saved inputs pickle (e.g. debug_align_frame_inputs/align_frame_inputs_0001.pkl)")
    parser.add_argument("--frame_idx", type=int, default=None,
                        help="Override frame index. If not set, uses the one saved in pickle.")
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="Override debug output directory. If not set, uses the one saved in pickle.")
    args = parser.parse_args()

    print(f"Loading inputs from {args.input_pkl}")
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    frame_idx = args.frame_idx if args.frame_idx is not None else data["frame_idx"]
    image_info_work = _rebuild_image_info_work(data["image_info_work"], frame_idx)
    obj_mesh = data["obj_mesh"]
    debug_dir = args.debug_dir if args.debug_dir is not None else data.get("debug_dir")

    print(f"Running _align_frame_with_sam3d for frame_idx={frame_idx}, debug_dir={debug_dir}")
    success = _align_frame_with_sam3d(
        image_info_work, frame_idx, obj_mesh,
        debug_dir=debug_dir,
    )
    print(f"Result: success={success}")


if __name__ == "__main__":
    main()
