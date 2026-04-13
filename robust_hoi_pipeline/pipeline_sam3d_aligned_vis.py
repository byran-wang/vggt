import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.logger import get_logger
from pipeline_sam3d_filter_3D_vis import init_rerun, load_camera_pose, log_mesh, log_image, log_depth_points, render_mesh_contour

logger = get_logger(__name__)


def load_frame_indices(frame_list_file):
    """Load frame indices from a file. Each line may be just an index
    (e.g. "0123") or "{idx} {n}/6 {names}" — only the first token is parsed.
    Returns list of ints, or None if the file is missing.
    """
    frame_list_file = Path(frame_list_file)
    if not frame_list_file.exists():
        logger.error(f"{frame_list_file} not found.")
        return None
    indices = []
    with open(frame_list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(line.split()[0]))
    logger.info(f"Loaded {len(indices)} frames from {frame_list_file}")
    return indices


def save_contour_image(mesh, rgb_dir, fid, K, c2o, scene_name):
    """Render mesh contour, overlay on the RGB image, and save to /tmp.

    Returns:
        Path to the tmp directory containing the saved image, or None if
        the contour could not be rendered.
    """
    from PIL import Image as _Image

    img_path = rgb_dir / f"{fid}.jpg"
    if not img_path.exists():
        img_path = rgb_dir / f"{fid}.png"
    if not img_path.exists():
        return None

    img = np.array(_Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    o2c = np.linalg.inv(c2o)
    contour_img = render_mesh_contour(mesh, K, o2c, H, W)
    if contour_img is None:
        return None

    contour_mask = contour_img.any(axis=-1)
    img_with_contour = img.copy()
    img_with_contour[contour_mask] = contour_img[contour_mask]

    tmp_dir = Path("/tmp/sam3d_contour") / scene_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _Image.fromarray(img_with_contour).save(tmp_dir / f"{fid}.png")
    return tmp_dir


def main(args):
    dataset_dir = Path(args.dataset_dir)
    sam3d_dir = dataset_dir / args.scene_name / "SAM3D"

    # Aligned camera source: SAM3D_aligned_mask or SAM3D_aligned_pts
    aligned_dir = dataset_dir / args.scene_name / f"SAM3D_aligned_{args.align_method}"
    if not aligned_dir.exists():
        logger.error(f"{aligned_dir} not found. Run ho3d_align_SAM3D_{args.align_method} first.")
        return

    # Load frame list — either after 3D filter or after depth-coverage filter
    frame_list_file = (Path(args.frame_list_file) if args.frame_list_file is not None
                       else dataset_dir / args.scene_name / "SAM3D_aligned_pts" / "frame_list_after_aligned_pts.txt")
    frame_indices = load_frame_indices(frame_list_file)
    if frame_indices is None:
        return

    # init_rerun(f"sam3d_aligned_{args.align_method}_vis")
    init_rerun(f"sam3d_aligned_vis")

    rgb_dir = dataset_dir / args.scene_name / "rgb"

    for seq_i, frame_idx in enumerate(frame_indices):
        fid = f"{frame_idx:04d}"

        # Load camera pose from the aligned directory
        camera_json = aligned_dir / fid / "camera.json"
        cam = load_camera_pose(camera_json)
        if cam is None:
            logger.warning(f"  Frame {fid}: camera.json not found in {aligned_dir / fid}, skipping")
            continue
        K, c2o, scale = cam

        rr.set_time_sequence("frame", seq_i)

        # Log mesh from original SAM3D directory
        mesh = log_mesh(sam3d_dir / fid)
        has_pts = log_depth_points(args.dataset_dir, args.scene_name, fid, c2o, scale)

        # Render mesh contour overlay and log image
        contour_dir = save_contour_image(mesh, rgb_dir, fid, K, c2o, args.scene_name) if mesh is not None else None
        has_contour = contour_dir is not None
        has_img = log_image(contour_dir or rgb_dir, fid, frame_idx, K, c2o, args.jpeg_quality)

        logger.info(f"  Frame {fid}: mesh={'yes' if mesh else 'no'}, image={'yes' if has_img else 'no'}, "
                     f"depth_pts={'yes' if has_pts else 'no'}, contour={'yes' if has_contour else 'no'}")

    logger.info(f"Done. Visualized {len(frame_indices)} frames with {args.align_method} alignment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM3D aligned frames in Rerun")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--align_method", type=str, default="pts", choices=["mask", "pts"],
                        help="Alignment method: 'mask' for SAM3D_aligned_mask, 'pts' for SAM3D_aligned_pts")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--frame_list_file", type=str, default=None,
                        help="Optional explicit path to a frame list file (e.g. frame_list_faces_coverage.txt)")

    args = parser.parse_args()
    main(args)
