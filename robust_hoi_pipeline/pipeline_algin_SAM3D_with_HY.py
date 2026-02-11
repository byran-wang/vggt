import json
import shutil
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree as KDTree


def load_mesh_from_glb(glb_path: str) -> trimesh.Trimesh:
    """Load mesh from GLB file, concatenating all geometries if needed."""
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if len(meshes) == 1:
            return meshes[0]
        else:
            return trimesh.util.concatenate(meshes)
    else:
        return loaded


def chamfer_distance(pts_a, pts_b):
    """Compute bidirectional mean Chamfer distance between two point sets."""
    tree_a = KDTree(pts_a)
    tree_b = KDTree(pts_b)
    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)
    return np.mean(dist_a2b) + np.mean(dist_b2a)


def refine_translation_icp(source_pts, target_pts, num_iters=50):
    """Refine translation only via ICP (nearest-neighbor + mean offset).

    At each iteration, find closest target point for each source point,
    then shift source by the mean displacement vector. Scale and rotation
    are kept fixed.

    Returns:
        delta_t: (3,) refined translation offset.
    """
    pts = source_pts.copy()
    delta_t = np.zeros(3, dtype=np.float64)

    for it in range(num_iters):
        tree = KDTree(target_pts)
        dists, idx = tree.query(pts)
        displacement = target_pts[idx] - pts  # (N, 3)
        shift = displacement.mean(axis=0)
        pts += shift
        delta_t += shift

        if np.linalg.norm(shift) < 1e-7:
            print(f"  ICP converged at iteration {it + 1}")
            break

    cd = chamfer_distance(pts, target_pts)
    print(f"  ICP refined: CD = {cd:.6f} after {min(it + 1, num_iters)} iters")
    return delta_t


def rotation_matrix_y(angle_deg):
    """Create a 3x3 rotation matrix around the Y axis."""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    cond_index = args.cond_index

    SAM_3D_dir = data_dir / "SAM3D" / f"{cond_index:04d}"
    HY_dir = out_dir / "pipeline_hunyuan_3d"

    result_dir = out_dir / "pipeline_algin_SAM3D_with_HY"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load meshes
    sam3d_glb_path = SAM_3D_dir / "scene.glb"
    hy_obj_path = HY_dir / "white_mesh_remesh.obj"

    sam3d_mesh = load_mesh_from_glb(str(sam3d_glb_path))
    hy_mesh = trimesh.load(str(hy_obj_path), process=False)

    print(f"SAM3D mesh: {len(sam3d_mesh.vertices)} vertices, {len(sam3d_mesh.faces)} faces")
    print(f"Hunyuan mesh: {len(hy_mesh.vertices)} vertices, {len(hy_mesh.faces)} faces")

    # Align SAM3D to Hunyuan: fixed scale=2.0, Y-axis rotation in [0, 90, 180, 270], translation
    fixed_scale = 2.0
    candidate_angles = [0, 90, 180, 270]

    sam3d_verts = np.array(sam3d_mesh.vertices, dtype=np.float64)
    hy_verts = np.array(hy_mesh.vertices, dtype=np.float64)
    hy_centroid = hy_verts.mean(axis=0)

    best_cd = float("inf")
    best_angle = 0
    best_translation = np.zeros(3)

    for angle in candidate_angles:
        R = rotation_matrix_y(angle)
        transformed = fixed_scale * (R @ sam3d_verts.T).T
        # Optimal translation: align centroids
        translation = hy_centroid - transformed.mean(axis=0)
        transformed += translation

        cd = chamfer_distance(transformed, hy_verts)
        print(f"  Y-rot {angle:3d} deg: CD = {cd:.6f}")

        if cd < best_cd:
            best_cd = cd
            best_angle = angle
            best_translation = translation

    print(f"Best candidate: Y-rot={best_angle} deg, CD={best_cd:.6f}")

    # Refine translation with ICP (rotation and scale held fixed)
    R_best = rotation_matrix_y(best_angle)
    initial_transformed = fixed_scale * (R_best @ sam3d_verts.T).T + best_translation
    print("Running translation-only ICP refinement...")
    icp_delta_t = refine_translation_icp(initial_transformed, hy_verts)
    best_translation = best_translation + icp_delta_t

    # Build final 4x4 transformation matrix
    transform_4x4 = np.eye(4, dtype=np.float64)
    transform_4x4[:3, :3] = fixed_scale * R_best
    transform_4x4[:3, 3] = best_translation

    # Save transformation parameters
    transform_params = {
        "scale": fixed_scale,
        "rotation_y_deg": best_angle,
        "rotation_matrix": R_best.tolist(),
        "translation": best_translation.tolist(),
        "transform_4x4": transform_4x4.tolist(),
        "chamfer_distance": float(best_cd),
    }
    json_path = result_dir / "SAM3D_aligned_with_HY.json"
    with open(json_path, "w") as f:
        json.dump(transform_params, f, indent=4)
    print(f"Saved transformation to {json_path}")

    # Copy original SAM3D mesh (export .glb as .obj)
    sam3d_mesh.export(str(result_dir / "SAM3D_origin.obj"))
    print(f"Saved SAM3D original mesh to {result_dir / 'SAM3D_origin.obj'}")

    # Copy original Hunyuan mesh
    shutil.copy2(str(hy_obj_path), str(result_dir / "Hunyuan_origin.obj"))
    print(f"Copied Hunyuan mesh to {result_dir / 'Hunyuan_origin.obj'}")

    # Save aligned SAM3D mesh
    aligned_mesh = sam3d_mesh.copy()
    aligned_verts = (transform_4x4[:3, :3] @ np.array(aligned_mesh.vertices).T).T + transform_4x4[:3, 3]
    aligned_mesh.vertices = aligned_verts
    aligned_mesh.export(str(result_dir / "SAM3D_aligned.obj"))
    print(f"Saved aligned SAM3D mesh to {result_dir / 'SAM3D_aligned.obj'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align SAM3D mesh with Hunyuan 3D mesh")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/GSF13)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index")
    args = parser.parse_args()
    main(args)
