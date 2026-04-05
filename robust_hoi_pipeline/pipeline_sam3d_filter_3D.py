import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.logger import get_logger
from utils_simba.rerun import load_mesh_as_trimesh

logger = get_logger(__name__)


def _get_mesh_pca(mesh, frame_idx=None):
    """Return aligned axis index and eigenvalue ratios for the mesh.

    Returns:
        aligned: int (0=X, 1=Y, 2=Z) — which axis the principal eigenvector points along
        ratios: np.array of shape (3,) — eigenvalues normalized by their sum
    """
    verts = np.array(mesh.vertices)
    cov = np.cov(verts, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue (last column from eigh)
    principal = eigenvectors[:, -1]
    abs_principal = np.abs(principal)
    aligned = int(np.argmax(abs_principal))
    # Eigenvalue ratios (normalized so they sum to 1)
    ratios = eigenvalues / eigenvalues.sum()
    axis_names = ["X", "Y", "Z"]
    fid = f"Frame {frame_idx:04d}" if frame_idx is not None else ""
    logger.debug(f"  {fid}: abs principal = {abs_principal} -> aligned={axis_names[aligned]}, ratios = {ratios}")
    return aligned, ratios


def _filter_by_axis_alignment(frame_indices, sam3d_dir):
    """Filter frames whose principal axis doesn't match the majority.

    Returns frame_ratios dict for use by subsequent filters.
    Modifies frame_indices in place.
    """
    axis_names = ["X", "Y", "Z"]
    frame_axes = {}   # frame_idx -> aligned axis (0=X, 1=Y, 2=Z)
    frame_ratios = {} # frame_idx -> eigenvalue ratios (3,)
    for frame_idx in list(frame_indices):
        fid = f"{frame_idx:04d}"
        frame_dir = sam3d_dir / fid
        mesh = load_mesh_as_trimesh(frame_dir)
        if mesh is None:
            logger.warning(f"  Frame {fid}: no mesh found, keeping frame")
            continue
        aligned, ratios = _get_mesh_pca(mesh, frame_idx)
        frame_axes[frame_idx] = aligned
        frame_ratios[frame_idx] = ratios

    if not frame_axes:
        return frame_ratios

    axis_counts = np.bincount(list(frame_axes.values()), minlength=3)
    dominant_axis = int(np.argmax(axis_counts))
    logger.info(f"Axis counts: X={axis_counts[0]}, Y={axis_counts[1]}, Z={axis_counts[2]} -> dominant={axis_names[dominant_axis]}")

    filtered_out = []
    for frame_idx, axis in frame_axes.items():
        if axis != dominant_axis:
            logger.info(f"  Frame {frame_idx:04d}: aligned={axis_names[axis]}, expected={axis_names[dominant_axis]}, filtering out")
            filtered_out.append(frame_idx)
            frame_indices.remove(frame_idx)
    if filtered_out:
        logger.info(f"Filtered {len(filtered_out)} frames with non-{axis_names[dominant_axis]}-aligned meshes: {filtered_out}")

    return frame_ratios


def _filter_by_eigenvalue_ratios(frame_indices, frame_ratios):
    """Filter frames whose eigenvalue ratios deviate from the median.

    Uses MAD (median absolute deviation) for robust outlier detection.
    Modifies frame_indices in place.
    """
    remaining_ratios = {k: v for k, v in frame_ratios.items() if k in frame_indices}
    if len(remaining_ratios) < 3:
        return

    ratio_matrix = np.array(list(remaining_ratios.values()))  # (N, 3)
    median_ratios = np.median(ratio_matrix, axis=0)
    distances_to_median = np.linalg.norm(ratio_matrix - median_ratios, axis=1)
    median_dist = np.median(distances_to_median)
    mad = np.median(np.abs(distances_to_median - median_dist))
    threshold = median_dist + 3.0 * max(mad, 1e-6)

    logger.info(f"Eigenvalue ratio median: {median_ratios}, dist median: {median_dist:.4f}, MAD: {mad:.4f}, threshold: {threshold:.4f}")

    filtered_out = []
    for (frame_idx, _), dist in zip(remaining_ratios.items(), distances_to_median):
        if dist > threshold:
            logger.info(f"  Frame {frame_idx:04d}: ratio dist={dist:.4f} > {threshold:.4f}, filtering out")
            filtered_out.append(frame_idx)
            frame_indices.remove(frame_idx)
    if filtered_out:
        logger.info(f"Filtered {len(filtered_out)} frames with deviant eigenvalue ratios: {filtered_out}")


def _filter_by_projection_distance(frame_indices, sam3d_dir, max_frame):
    """Filter frames by distance of projected mesh origin to optical center.

    Keeps the top max_frame frames closest to the optical center.
    Returns (selected, distances, projected_points).
    """
    distances = {}
    projected_points = {}

    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"
        camera_json = sam3d_dir / fid / "camera.json"
        if not camera_json.exists():
            logger.warning(f"  Frame {fid}: camera.json not found, skipping")
            continue

        with open(camera_json, "r") as f:
            camera = json.load(f)

        K = np.array(camera["K"], dtype=np.float64)       # (3, 3)
        o2c = np.array(camera["blw2cvc"], dtype=np.float64)  # (4, 4)

        origin_world = np.array([0.0, 0.0, 0.0, 1.0])
        origin_cam = o2c @ origin_world  # (4,)
        origin_cam_3d = origin_cam[:3]

        if origin_cam_3d[2] <= 0:
            logger.warning(f"  Frame {fid}: origin behind camera, skipping")
            continue

        p_2d = K @ origin_cam_3d
        u = p_2d[0] / p_2d[2]
        v = p_2d[1] / p_2d[2]

        cx = K[0, 2]
        cy = K[1, 2]
        dist = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)

        distances[frame_idx] = dist
        projected_points[frame_idx] = (u, v)
        logger.debug(f"  Frame {fid}: projected=({u:.1f}, {v:.1f}), optical_center=({cx:.1f}, {cy:.1f}), dist={dist:.1f}")

    # Save debug info
    debug_path = sam3d_dir / "frame_list_3d_filter_debug.txt"
    with open(debug_path, "w") as f:
        f.write("frame_idx\tproj_u\tproj_v\tdistance\n")
        for idx in sorted(distances, key=lambda i: distances[i]):
            u, v = projected_points[idx]
            f.write(f"{idx:04d}\t{u:.1f}\t{v:.1f}\t{distances[idx]:.1f}\n")
    logger.info(f"Saved debug info to {debug_path}")

    sorted_by_dist = sorted(distances, key=lambda i: distances[i])
    selected = sorted_by_dist[:max_frame]
    return selected


def main(args):
    sam3d_dir = Path(f"{args.dataset_dir}/{args.scene_name}/SAM3D")

    # Load frame list after 2D filter generated by pipeline_sam3d_filter_2D.py
    frame_list_file = sam3d_dir / "frame_list_after_depth_filtered.txt"
    with open(frame_list_file, "r") as f:
        frame_indices = [int(line.strip()) for line in f if line.strip()]
    logger.info(f"Loaded {len(frame_indices)} frames from {frame_list_file}")

    # Filter 1: remove frames not aligned with the dominant (majority) axis
    frame_ratios = _filter_by_axis_alignment(frame_indices, sam3d_dir)

    # Filter 2: remove frames whose eigenvalue ratios deviate from the median
    _filter_by_eigenvalue_ratios(frame_indices, frame_ratios)

    # Filter 3: keep frames closest to optical center
    selected = _filter_by_projection_distance(frame_indices, sam3d_dir, args.max_frame)

    out_path = sam3d_dir / "frame_list_after_3d_filtered.txt"
    with open(out_path, "w") as f:
        for idx in selected:
            f.write(f"{idx}\n")
    logger.info(f"Saved 3D-filtered frame list ({len(selected)} frames) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter frames by optical center for SAM3D")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--max_frame", type=int, default=20, help="Max frames to keep after 3D filtering")

    args = parser.parse_args()
    main(args)
