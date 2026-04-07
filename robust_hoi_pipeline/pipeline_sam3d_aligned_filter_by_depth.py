import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.depth import depth2xyzmap, get_depth
from utils_simba.logger import get_logger
from utils_simba.rerun import load_mesh_as_trimesh
from pipeline_sam3d_filter_3D_vis import load_camera_pose

logger = get_logger(__name__)


def _backproject_depth_to_object_space(fid, dataset_dir, scene_name, c2o, scale):
    """Back-project filtered depth to 3D points in SAM3D object space.

    Returns pts_obj (N, 3) or None if data is missing.
    """
    preprocess_dir = Path(f"{dataset_dir}/{scene_name}/pipeline_preprocess")
    depth_path = preprocess_dir / "../depth" / f"{fid}.png"
    mask_path = preprocess_dir / "mask_obj" / f"{fid}.png"
    meta_path = preprocess_dir / "meta" / f"{fid}.pkl"

    if not all(p.exists() for p in [depth_path, mask_path, meta_path]):
        return None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    K = np.array(meta["intrinsics"], dtype=np.float64)
    depth = get_depth(str(depth_path))
    depth /= scale
    mask_obj = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    xyz_cam = depth2xyzmap(depth, K)  # (H, W, 3)
    valid = (mask_obj > 0) & (depth > 0.01)
    pts_cam = xyz_cam[valid]  # (N, 3)

    if len(pts_cam) < 10:
        return None

    pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_obj = (c2o @ pts_cam_h.T).T[:, :3]
    return pts_obj


def _filter_outliers_by_mesh(mesh, pts_obj, dist_threshold):
    """Remove points that are far from the mesh surface.

    Returns inlier points (M, 3).
    """
    _, distances, _ = trimesh.proximity.closest_point(mesh, pts_obj)
    inlier_mask = distances < dist_threshold
    return pts_obj[inlier_mask]



def _rasterize_to_grid(points_2d, global_min, global_max, resolution=128):
    """Rasterize 2D points into a boolean grid using shared global bounds."""
    span = global_max - global_min
    span[span < 1e-8] = 1e-8
    normed = ((points_2d - global_min) / span * (resolution - 1)).astype(int)
    normed = np.clip(normed, 0, resolution - 1)
    grid = np.zeros((resolution, resolution), dtype=bool)
    grid[normed[:, 0], normed[:, 1]] = True
    return grid


def _check_faces_coverage(pts, mesh, ratio_threshold, debug_dir=None):
    """Check coverage by projecting points and mesh onto XY, XZ, YZ planes.

    For each plane, computes the ratio of projected point cloud area to
    projected mesh area. A face is covered if the ratio >= ratio_threshold.

    Returns (num_faces_covered, face_names_covered).
    """
    mesh_verts = mesh.vertices  # (V, 3)
    # 6 faces: filter by sign of the normal axis, then project onto the other two
    # (proj_axes, filter_axis, sign, name)
    faces = [
        ((0, 1), 2, +1, "XY_Z+"),
        ((0, 1), 2, -1, "XY_Z-"),
        ((0, 2), 1, +1, "XZ_Y+"),
        ((0, 2), 1, -1, "XZ_Y-"),
        ((1, 2), 0, +1, "YZ_X+"),
        ((1, 2), 0, -1, "YZ_X-"),
    ]

    # Compute ratio for each of the 6 faces
    face_ratios = {}
    for (a0, a1), filt_axis, sign, name in faces:
        if sign > 0:
            mesh_sel = mesh_verts[mesh_verts[:, filt_axis] >= 0]
            pts_sel = pts[pts[:, filt_axis] >= 0]
        else:
            mesh_sel = mesh_verts[mesh_verts[:, filt_axis] <= 0]
            pts_sel = pts[pts[:, filt_axis] <= 0]

        if len(mesh_sel) == 0:
            face_ratios[name] = 0.0
            continue

        mesh_2d = mesh_sel[:, [a0, a1]]
        pts_2d = pts_sel[:, [a0, a1]] if len(pts_sel) > 0 else np.empty((0, 2))

        global_min = np.array([-0.5, -0.5])
        global_max = np.array([0.5, 0.5])

        mesh_grid = _rasterize_to_grid(mesh_2d, global_min, global_max)
        pts_grid = _rasterize_to_grid(pts_2d, global_min, global_max) if len(pts_2d) > 0 else np.zeros_like(mesh_grid)

        mesh_cells = mesh_grid.sum()
        if mesh_cells == 0:
            face_ratios[name] = 0.0
            continue
        pts_cells = pts_grid.sum()
        ratio = pts_cells / mesh_cells
        face_ratios[name] = ratio
        logger.info(f"    Face {name}: pts_cells={pts_cells}, mesh_cells={mesh_cells}, ratio={ratio:.2f}")

        if debug_dir is not None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            img[mesh_grid, 0] = 255
            img[pts_grid, 1] = 255
            img_bgr = cv2.flip(img, 0)
            cv2.imwrite(str(Path(debug_dir) / f"coverage_{name}.png"), img_bgr)

    # For each plane pair, keep only the side with the higher ratio
    plane_pairs = [
        ("XY_Z+", "XY_Z-", "XY"),
        ("XZ_Y+", "XZ_Y-", "XZ"),
        ("YZ_X+", "YZ_X-", "YZ"),
    ]
    faces_covered = []
    for pos_name, neg_name, plane_name in plane_pairs:
        best_name = pos_name if face_ratios.get(pos_name, 0) >= face_ratios.get(neg_name, 0) else neg_name
        best_ratio = face_ratios.get(best_name, 0)
        logger.info(f"    Plane {plane_name}: best={best_name}, ratio={best_ratio:.2f}")
        if best_ratio >= ratio_threshold:
            faces_covered.append(best_name)

    return len(faces_covered), faces_covered


def main(args):
    dataset_dir = Path(args.dataset_dir)
    sam3d_dir = dataset_dir / args.scene_name / "SAM3D"
    aligned_dir = dataset_dir / args.scene_name / "SAM3D_aligned_pts"

    # Load frame list from SAM3D_aligned_pts/frame_list_after_aligned_pts.txt
    frame_list_file = aligned_dir / "frame_list_after_aligned_pts.txt"
    if not frame_list_file.exists():
        logger.error(f"{frame_list_file} not found. Run ho3d_align_SAM3D_pts first.")
        return
    with open(frame_list_file, "r") as f:
        frame_indices = [int(line.strip()) for line in f if line.strip()]
    logger.info(f"Loaded {len(frame_indices)} frames from {frame_list_file}")

    filtered = []
    filtered_out = []
    coverage_info = []
    for frame_idx in frame_indices:
        fid = f"{frame_idx:04d}"

        # Load SAM3D mesh
        mesh = load_mesh_as_trimesh(sam3d_dir / fid)
        if mesh is None:
            logger.warning(f"  Frame {fid}: no mesh found, skipping")
            continue

        # Load aligned camera pose
        camera_json = aligned_dir / fid / "camera.json"
        cam = load_camera_pose(camera_json)
        if cam is None:
            logger.warning(f"  Frame {fid}: camera.json not found in aligned dir, skipping")
            continue
        K, c2o, scale = cam

        # Back-project depth to object space
        pts_obj = _backproject_depth_to_object_space(
            fid, args.dataset_dir, args.scene_name, c2o, scale
        )
        if pts_obj is None:
            logger.warning(f"  Frame {fid}: depth back-projection failed, skipping")
            continue

        logger.info(f"  Frame {fid}: {len(pts_obj)} depth points in object space")

        # Remove outlier points far from mesh surface
        pts_inlier = _filter_outliers_by_mesh(mesh, pts_obj, args.dist_threshold)
        logger.info(f"  Frame {fid}: {len(pts_inlier)} inlier points (threshold={args.dist_threshold})")

        if len(pts_inlier) < 10:
            logger.warning(f"  Frame {fid}: too few inlier points, filtering out")
            filtered_out.append(frame_idx)
            continue

        # Check 3-face coverage
        debug_dir = aligned_dir / fid / "coverage_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        faces_covered, face_names = _check_faces_coverage(
            pts_inlier, mesh, args.face_ratio, debug_dir=str(debug_dir)
        )
        logger.info(
            f"  Frame {fid}: faces_covered={faces_covered}/3, "
            f"covered_faces={face_names}"
        )

        coverage_info.append((frame_idx, faces_covered, face_names))

        if faces_covered < 3:
            logger.info(f"  Frame {fid}: insufficient face coverage ({face_names}), filtering out")
            filtered_out.append(frame_idx)
        else:
            filtered.append(frame_idx)

    logger.info(f"Kept {len(filtered)} frames, filtered out {len(filtered_out)}: {filtered_out}")

    # Save coverage info sorted by number of covered faces (descending)
    coverage_info.sort(key=lambda x: x[1], reverse=True)
    coverage_path = aligned_dir / "frame_list_faces_coverage.txt"
    with open(coverage_path, "w") as f:
        for frame_idx, num_covered, names in coverage_info:
            f.write(f"{frame_idx:04d} {num_covered}/3 {','.join(names)}\n")
    logger.info(f"Saved faces coverage info ({len(coverage_info)} frames) to {coverage_path}")

    # Save filtered frame list
    out_path = aligned_dir / "frame_list_after_depth_filtered.txt"
    with open(out_path, "w") as f:
        for idx in filtered:
            f.write(f"{idx}\n")
    logger.info(f"Saved depth-filtered frame list ({len(filtered)} frames) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter SAM3D aligned frames by depth coverage")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--dist_threshold", type=float, default=0.05,
                        help="Max distance from mesh surface to keep a depth point")
    parser.add_argument("--face_ratio", type=float, default=0.2,
                        help="Min fraction of points near a bbox face to consider it covered")

    args = parser.parse_args()
    main(args)
