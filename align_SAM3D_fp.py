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

from utils_simba.depth import get_depth, load_filtered_depth, depth2xyzmap
from utils_simba.rerun import load_mesh_as_trimesh
from utils_simba.render import nvdiffrast_render
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


def _compute_depth_chamfer(mesh, K, o2c, depth_obs, mask):
    """Render mesh depth at the given pose and compute chamfer distance to observed depth.

    Returns mean bidirectional point-to-point distance (in meters), or None on failure.
    """
    H, W = depth_obs.shape[:2]
    ob_in_cvcams = torch.tensor(o2c, dtype=torch.float32, device="cuda")[None]
    _, depth_render, _ = nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cvcams=ob_in_cvcams, mesh=mesh,
    )
    depth_render = depth_render[0].cpu().numpy()  # (H, W)

    # Back-project both depth maps to 3D
    valid_obs = (depth_obs > 0.01) & (mask > 0)
    valid_render = depth_render > 0.01
    if valid_obs.sum() < 10 or valid_render.sum() < 10:
        return None

    xyz_obs = depth2xyzmap(depth_obs, K)
    xyz_render = depth2xyzmap(depth_render, K)
    pts_obs = xyz_obs[valid_obs]        # (N, 3)
    pts_render = xyz_render[valid_render]  # (M, 3)

    # Chamfer: mean of nearest-neighbor distances in both directions
    from scipy.spatial import cKDTree
    tree_obs = cKDTree(pts_obs)
    tree_render = cKDTree(pts_render)
    d_obs_to_render, _ = tree_render.query(pts_obs)
    d_render_to_obs, _ = tree_obs.query(pts_render)
    chamfer = (d_obs_to_render.mean() + d_render_to_obs.mean()) / 2.0
    return float(chamfer)


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

    scores = {}

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

        # Evaluate: chamfer distance between rendered depth and observed depth
        cd_before = _compute_depth_chamfer(mesh, K, o2c, depth, mask)
        cd_after = _compute_depth_chamfer(mesh, K, refined_o2c, depth, mask)
        logger.info(f"  Frame {fid}: chamfer before={cd_before:.4f}, after={cd_after:.4f}" if cd_before is not None and cd_after is not None
                     else f"  Frame {fid}: chamfer computation skipped")

        # 5. Save camera.json in the same format as align_SAM3D_pts.py
        frame_out_dir = out_dir / fid
        frame_out_dir.mkdir(parents=True, exist_ok=True)
        _save_camera_json(str(frame_out_dir / "camera.json"), K, refined_o2c, scale)
        logger.info(f"  Frame {fid}: saved refined pose to {frame_out_dir / 'camera.json'}")

        if cd_before is not None and cd_after is not None:
            scores[fid] = {"cd_before": cd_before, "cd_after": cd_after}

    # Sort by cd_after (increasing = best first)
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]["cd_after"]))

    # Save scores.json
    scores_path = out_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(sorted_scores, f, indent=2)
    logger.info(f"Saved scores to {scores_path}")

    # Save sorted frame list
    frame_list_path = out_dir / "frame_list_after_aligned_fp.txt"
    with open(frame_list_path, "w") as f:
        for sid in sorted_scores:
            f.write(f"{int(sid)}\n")
    logger.info(f"Saved sorted frame list ({len(sorted_scores)} frames) to {frame_list_path}")

    # Print ranking
    logger.info("Ranking (best to worst):")
    for rank, (sid, s) in enumerate(sorted_scores.items()):
        logger.info(f"  {rank+1}. frame={sid} cd_before={s['cd_before']:.4f} cd_after={s['cd_after']:.4f}")

    logger.info(f"Done. Processed {len(frame_indices)} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine SAM3D alignment using FoundationPose tracking")
    parser.add_argument("--data_dir", type=str, required=True, help="Sequence root dir (rgb/, depth/, mask_object/, SAM3D/, SAM3D_aligned_pts/)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for refined camera.json per frame")
    args = parser.parse_args()
    main(args)
