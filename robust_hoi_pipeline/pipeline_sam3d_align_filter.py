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

import torch

from utils_simba.depth import depth2xyzmap, load_filtered_depth
from utils_simba.logger import get_logger
from utils_simba.rerun import load_mesh_as_trimesh
from utils_simba.render import nvdiffrast_render
from utils_simba.visibility_mesh import get_visibility_mesh, get_visibility_occ
from utils_simba.visibility_test import (
    mesh_to_voxel_grid, voxel_centers, get_camera_pose,
)
from pipeline_sam3d_filter_3D_vis import load_camera_pose

logger = get_logger(__name__)


def _backproject_depth_to_object_space(fid, dataset_dir, scene_name, c2o, scale):
    """Back-project filtered depth to 3D points in SAM3D object space.

    Returns pts_obj (N, 3) or None if data is missing.
    """
    preprocess_dir = Path(f"{dataset_dir}/{scene_name}")
    depth_path = preprocess_dir / "depth" / f"{fid}.png"
    mask_path = preprocess_dir / "mask_object" / f"{fid}.png"
    meta_path = preprocess_dir / "meta" / f"0000.pkl"
    
    if not all(p.exists() for p in [depth_path, mask_path, meta_path]):
        return None
    
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    K = np.array(meta["camMat"], dtype=np.float64)
    depth = load_filtered_depth(str(depth_path))
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


def _make_six_cameras(mesh):
    """Create 6 axis-aligned cameras looking at mesh center.

    Returns list of (name, c2w_4x4) tuples.
    """
    radius = mesh.bounding_sphere.primitive.radius * 2
    center = mesh.bounding_sphere.primitive.center
    configs = [
        ("X+", np.array([1.0, 0, 0]), np.array([0, 0, 1])),
        ("X-", np.array([-1.0, 0, 0]), np.array([0, 0, 1])),
        ("Y+", np.array([0, 1.0, 0]), np.array([0, 0, 1])),
        ("Y-", np.array([0, -1.0, 0]), np.array([0, 0, 1])),
        ("Z+", np.array([0, 0, 1.0]), np.array([0, 1, 0])),
        ("Z-", np.array([0, 0, -1.0]), np.array([0, 1, 0])),
    ]
    cameras = []
    for name, direction, up in configs:
        eye = center + direction * radius
        c2w = get_camera_pose(eye, center, up)
        # flip Y and flip Z
        c2w[:3, 1] = -c2w[:3, 1]
        c2w[:3, 2] = -c2w[:3, 2]
        cameras.append((name, c2w))
    return cameras


def _visualize_cameras_rerun(mesh, cameras, K):
    """Visualize mesh and cameras in rerun."""
    import rerun as rr
    from utils_simba.rerun import log_camera_frame
    rr.init("faces_coverage", spawn=True)
    rr.log("mesh", rr.Mesh3D(
        vertex_positions=mesh.vertices,
        triangle_indices=mesh.faces,
        vertex_colors=mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, 'vertex_colors') else None,
    ), static=True)
    for name, c2w in cameras:
        log_camera_frame(f"cameras/{name}", K=K, c2w=c2w, image_plane_distance=0.05, static=True)


def _check_faces_coverage(mesh, c2o, ratio_threshold, debug_dir=None, vis_cam_in_rerun=False):
    """Check face coverage by rendering mesh and depth occupancy from 6 views.


    Returns (num_faces_covered, face_names_covered).
    """
    # Get visibility mesh from the actual camera pose
    if debug_dir is not None:
        debug_path = Path(debug_dir)
        # c2w = np.linalg.inv(np.linalg.inv(c2o))  # c2o is already c2w in object space
        visibility_mesh = get_visibility_occ(mesh, c2o, str(debug_path))

    # Create 6 axis-aligned cameras
    cameras = _make_six_cameras(visibility_mesh)

    # Synthetic intrinsics for rendering
    render_size = 256
    focal = render_size * 0.8
    K_synth = np.array([
        [focal, 0, render_size / 2],
        [0, focal, render_size / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    # Render visibility_mesh (red=visible, blue=occluded) from each of the 6 cameras
    faces_covered = []
    if vis_cam_in_rerun:
        _visualize_cameras_rerun(visibility_mesh, cameras, K_synth)

    for name, c2w in cameras:
        o2c = np.linalg.inv(c2w)
        ob_in_cvcams = torch.tensor(o2c, dtype=torch.float32, device="cuda")[None]
        color_render, depth_render, _ = nvdiffrast_render(
            K=K_synth, H=render_size, W=render_size,
            ob_in_cvcams=ob_in_cvcams, mesh=visibility_mesh,
        )
        # color_render: (1, H, W, 3) float [0,1], depth_render: (1, H, W)
        color_img = color_render[0].cpu().numpy()  # (H, W, 3) RGB float
        depth_img = depth_render[0].cpu().numpy()   # (H, W)

        # Save debug rendering
        if debug_dir is not None:
            debug_img = (color_img * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(debug_path / f"render_{name}.png"), debug_img[:, :, ::-1])

        # Mask of rendered pixels (anything with depth > 0)
        rendered = depth_img > 0.01

        if rendered.sum() < 10:
            logger.info(f"    Face {name}: not visible ({rendered.sum()} px), skipping")
            continue

        # Red channel > blue channel means visible (red voxels)
        red_pixels = rendered & (color_img[:, :, 0] > color_img[:, :, 2])
        num_red = red_pixels.sum()
        num_total = rendered.sum()
        ratio = num_red / num_total

        logger.info(f"    Face {name}: visible_px={num_red}, total_px={num_total}, ratio={ratio:.2f}")



        if ratio >= ratio_threshold:
            faces_covered.append(name)
    return len(faces_covered), faces_covered


def main(args):
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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


        # Check 3-face coverage
        debug_dir = out_dir / fid / "coverage_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        faces_covered, face_names = _check_faces_coverage(
            mesh, c2o, args.face_ratio, debug_dir=str(debug_dir)
        )
        logger.info(
            f"  Frame {fid}: faces_covered={faces_covered}/6, "
            f"covered_faces={face_names}"
        )

        coverage_info.append((frame_idx, faces_covered, face_names))

        # Save per-frame coverage info
        frame_coverage_path = out_dir / fid / "coverage_info.txt"
        frame_coverage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(frame_coverage_path, "w") as f:
            f.write(f"faces_covered: {faces_covered}/6\n")
            f.write(f"covered_faces: {','.join(face_names)}\n")

        if faces_covered < 3:
            logger.info(f"  Frame {fid}: insufficient face coverage ({face_names}), filtering out")
            filtered_out.append(frame_idx)
        else:
            filtered.append(frame_idx)

    logger.info(f"Kept {len(filtered)} frames, filtered out {len(filtered_out)}: {filtered_out}")

    # Save coverage info sorted by number of covered faces (descending)
    coverage_info.sort(key=lambda x: x[1], reverse=True)
    coverage_path = out_dir / "frame_list_faces_coverage.txt"
    with open(coverage_path, "w") as f:
        for frame_idx, num_covered, names in coverage_info:
            f.write(f"{frame_idx:04d} {num_covered}/6 {','.join(names)}\n")
    logger.info(f"Saved faces coverage info ({len(coverage_info)} frames) to {coverage_path}")

    # Save filtered frame list
    out_path = out_dir / "frame_list_align_filter.txt"
    with open(out_path, "w") as f:
        for idx in filtered:
            f.write(f"{idx}\n")
    logger.info(f"Saved depth-filtered frame list ({len(filtered)} frames) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter SAM3D aligned frames by depth coverage")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for filtered frames and coverage info")    
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--dist_threshold", type=float, default=0.05,
                        help="Max distance from mesh surface to keep a depth point")
    parser.add_argument("--face_ratio", type=float, default=0.2,
                        help="Min fraction of points near a bbox face to consider it covered")


    args = parser.parse_args()
    main(args)
