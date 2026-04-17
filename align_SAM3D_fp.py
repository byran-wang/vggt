import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "third_party" / "utils_simba"))
sys.path.insert(0, str(Path(__file__).parent / "third_party" / "FoundationPose"))

from utils_simba.depth import get_depth, load_filtered_depth
from utils_simba.rerun import load_mesh_as_trimesh
from utils_simba.logger import get_logger
from robust_hoi_pipeline.pipeline_joint_opt_debug import _save_depth_points_debug
from robust_hoi_pipeline.pipeline_sam3d_filter_3D_vis import load_camera_pose

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_logging_format, set_seed
import nvdiffrast.torch as dr

logger = get_logger(__name__)

# Lazy cache for FoundationPose shared resources
_fp_cache = {}


def _get_foundation_pose(mesh, debug_dir="/tmp/fp_debug"):
    """Return a cached FoundationPose estimator for the given mesh."""
    mesh_id = id(mesh)
    per_mesh = _fp_cache.get("per_mesh", {})
    if mesh_id in per_mesh:
        return per_mesh[mesh_id]

    if "scorer" not in _fp_cache:
        _fp_cache["scorer"] = ScorePredictor()
        _fp_cache["refiner"] = PoseRefinePredictor()
        _fp_cache["glctx"] = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=_fp_cache["scorer"],
        refiner=_fp_cache["refiner"],
        glctx=_fp_cache["glctx"],
        debug=0,
        debug_dir=debug_dir,
    )
    per_mesh[mesh_id] = est
    _fp_cache["per_mesh"] = per_mesh
    logger.info(f"Initialized FoundationPose estimator ({len(mesh.vertices)} verts)")
    return est




def _save_camera_json(path, K, o2c, scale):
    """Save camera.json in the same format as align_SAM3D_pts.py.

    Re-embeds the scale into the rotation columns of blw2cvc.
    """
    o2c_scaled = o2c.copy()
    o2c_scaled[:3, :3] = o2c[:3, :3] * scale
    o2c_scaled[:3, 3] = o2c[:3, 3] * scale

    camera_data = {
        "K": K.tolist(),
        "blw2cvc": o2c_scaled.tolist(),
    }
    with open(path, "w") as f:
        json.dump(camera_data, f, indent=2)


def _run_fp_track(est, rgb, depth, K, current_o2c):
    """Run FoundationPose tracking from current_o2c and return refined 4x4 o2c."""
    tf_to_center = est.get_tf_to_centered_mesh()
    tf_from_center = torch.eye(4, device="cuda", dtype=torch.float)
    tf_from_center[:3, 3] = -tf_to_center[:3, 3]

    o2c_t = torch.as_tensor(current_o2c, device="cuda", dtype=torch.float)
    est.pose_last = (o2c_t @ tf_from_center).reshape(1, 4, 4)

    pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=5)
    if pose is None or not np.isfinite(pose).all():
        return None
    return pose.astype(np.float64)


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sam3d_dir = data_dir / "SAM3D"
    aligned_dir = data_dir / "SAM3D_aligned_pts"

    set_logging_format()
    set_seed(0)

    # Load frame list from SAM3D_aligned_pts
    frame_list_file = aligned_dir / "frame_list_after_aligned_pts.txt"
    if not frame_list_file.exists():
        logger.error(f"{frame_list_file} not found. Run ho3d_align_SAM3D_pts first.")
        return
    with open(frame_list_file, "r") as f:
        frame_indices = [int(line.strip()) for line in f if line.strip()]
    logger.info(f"Loaded {len(frame_indices)} frames from {frame_list_file}")

    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        logger.info(f"Processing frame {fid}")

        # 1. Load intrinsics, camera pose and scale from camera.json
        camera_json = aligned_dir / fid / "camera.json"
        if not camera_json.exists():
            logger.warning(f"  Frame {fid}: camera.json not found, skipping")
            continue
        cam = load_camera_pose(camera_json)
        if cam is None:
            logger.warning(f"  Frame {fid}: failed to load camera.json, skipping")
            continue
        K, c2o, scale = cam
        o2c = np.linalg.inv(c2o)

        # 2. Load depth and scale
        depth_path = data_dir / "depth" / f"{fid}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        depth = load_filtered_depth(str(depth_path))
        depth /= scale

        # 3. Load SAM3D mesh
        mesh = load_mesh_as_trimesh(sam3d_dir / fid)
        if mesh is None:
            logger.warning(f"  Frame {fid}: no SAM3D mesh found, skipping")
            continue

        # Load RGB and mask
        rgb_path = data_dir / "rgb" / f"{fid}.jpg"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

        mask_path = data_dir / "mask_object" / f"{fid}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Object mask not found: {mask_path}")
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

        # Resize depth to match RGB if needed
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Mask depth by object mask
        depth[mask == 0] = 0

        # 4. Update pose by FoundationPose tracker
        debug_dir = str(out_dir / fid / "fp_debug")
        os.makedirs(debug_dir, exist_ok=True)
        est = _get_foundation_pose(mesh, debug_dir=debug_dir)
        
        _save_depth_points_debug(debug_dir, frame_idx, depth, K, rgb=rgb, o2c=o2c)
        refined_o2c = _run_fp_track(est, rgb, depth, K, o2c)
        if refined_o2c is None:
            logger.warning(f"  Frame {fid}: FoundationPose tracking failed, keeping original pose")
            refined_o2c = o2c

        # 5. Save camera.json in the same format as align_SAM3D_pts.py
        frame_out_dir = out_dir / fid
        frame_out_dir.mkdir(parents=True, exist_ok=True)
        _save_camera_json(str(frame_out_dir / "camera.json"), K, refined_o2c, scale)
        logger.info(f"  Frame {fid}: saved refined pose to {frame_out_dir / 'camera.json'}")

        # cam = load_camera_pose(frame_out_dir / "camera.json")

    logger.info(f"Done. Processed {len(frame_indices)} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine SAM3D alignment using FoundationPose tracking")
    parser.add_argument("--data_dir", type=str, required=True, help="Sequence root dir (rgb/, depth/, mask_object/, SAM3D/, SAM3D_aligned_pts/)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for refined camera.json per frame")
    args = parser.parse_args()
    main(args)
