import argparse
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.logger import get_logger

logger = get_logger(__name__)


def main(args):
    sam3d_dir = Path(args.dataset_dir) / args.scene_name / "SAM3D"
    filter_file = Path(args.dataset_dir) / args.scene_name / "SAM3D_align_filter" / "frame_list_align_filter.txt"

    if not filter_file.exists():
        logger.error(f"{filter_file} not found. Run pipeline_sam3d_align_filter first.")
        return

    with open(filter_file, "r") as f:
        keep_frames = {int(line.strip()) for line in f if line.strip()}
    # Always keep the condition frame
    keep_frames.add(args.cond_idx)
    logger.info(f"Loaded {len(keep_frames)} kept frames from {filter_file} (including cond_idx={args.cond_idx})")

    # Enumerate all numeric frame folders under SAM3D/
    all_folders = sorted(
        p for p in sam3d_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    logger.info(f"Found {len(all_folders)} SAM3D frame folders")

    deleted = []
    for folder in all_folders:
        frame_idx = int(folder.name)
        if frame_idx not in keep_frames:
            logger.info(f"  Deleting {folder.name}")
            shutil.rmtree(folder)
            deleted.append(frame_idx)

    logger.info(f"Deleted {len(deleted)} unused SAM3D folders, kept {len(all_folders) - len(deleted)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete SAM3D folders not in frame_list_align_filter.txt")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--cond_idx", type=int, default=0, help="Condition frame index (always kept)")
    args = parser.parse_args()
    main(args)
