import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

# Add FoundationPose to path so estimater/Utils are importable
sys.path.insert(0, str(Path(__file__).parent / "third_party" / "FoundationPose"))

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_logging_format, set_seed
import nvdiffrast.torch as dr


def _load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
            f.seek(0)
            class _Compat(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("numpy._core"):
                        module = module.replace("numpy._core", "numpy.core", 1)
                    return super().find_class(module, name)
            return _Compat(f).load()


def load_depth(depth_path):
    """24-bit encoded depth PNG → metres."""
    depth_scale = 0.00012498664727900177
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    if raw.ndim == 3:
        depth = (raw[..., 0] * 256.0 * 256.0 + raw[..., 1] * 256.0 + raw[..., 2]) * depth_scale
    else:
        depth = raw.astype(np.float32) * depth_scale
    return depth.astype(np.float32)


def load_intrinsics(meta_path):
    meta = _load_pickle(meta_path)
    for key in ("camMat", "intrinsics", "cam_K"):
        if key in meta:
            K = np.array(meta[key], dtype=np.float64)
            if K.shape == (9,):
                K = K.reshape(3, 3)
            return K
    raise ValueError(f"No intrinsics key found in {meta_path}. Keys: {list(meta.keys())}")


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fid = f"{args.cond_index:04d}"

    # Load RGB
    rgb_path = data_dir / "rgb" / f"{fid}.jpg"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

    # Load depth
    depth = load_depth(data_dir / "depth" / f"{fid}.png")

    # Resize depth to match rgb if needed
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Load intrinsics
    K = load_intrinsics(str(data_dir / "meta" / f"{fid}.pkl"))

    # Load object mask
    mask_path = data_dir / "mask_object" / f"{fid}.png"
    if mask_path.exists():
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    else:
        print(f"Warning: mask not found at {mask_path}, using full image")
        mask = np.ones(rgb.shape[:2], dtype=np.uint8)

    # Load GT mesh
    gt_mesh_dir = data_dir / "gt_mesh"
    mesh = None
    for name in ("model.obj", "model.ply", "textured.obj"):
        candidate = gt_mesh_dir / name
        if candidate.exists():
            mesh = trimesh.load(str(candidate), force="mesh")
            print(f"Loaded mesh: {candidate} ({len(mesh.vertices)} verts)")
            break
    if mesh is None:
        raise FileNotFoundError(f"No mesh found in {gt_mesh_dir}")

    # Run FoundationPose
    set_logging_format()
    set_seed(0)
    debug_dir = str(out_dir / "fp_debug")
    os.makedirs(debug_dir, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    print("FoundationPose estimator initialized")

    pose = est.register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
    )  # (4,4) ob_in_cam

    # Save pose
    pose_path = out_dir / f"ob_in_cam_{fid}.txt"
    np.savetxt(str(pose_path), pose.reshape(4, 4))
    print(f"Saved pose to {pose_path}")
    print(f"ob_in_cam:\n{pose.reshape(4, 4)}")

    if args.vis_in_rerun:
        _vis_in_rerun(rgb, depth, mask.astype(bool), K, mesh, pose.reshape(4, 4))


def _vis_in_rerun(rgb, depth, mask, K, mesh, ob_in_cam):
    import rerun as rr
    import rerun.blueprint as rrb

    h, w = rgb.shape[:2]
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D", origin="world"),
        rrb.Spatial2DView(name="Camera", origin="world/camera"),
    )
    rr.init("align_by_foundation_pose", spawn=True, default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Camera
    rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
    rr.log("world/camera", rr.Pinhole(image_from_camera=K, resolution=[w, h], image_plane_distance=1.0))
    rr.log("world/camera/image", rr.Image(rgb))

    # Depth point cloud (masked)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = (z > 0) & mask
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)[valid]
    rr.log("world/pointcloud", rr.Points3D(positions=pts, colors=rgb[valid], radii=0.002), static=True)

    # Mesh transformed to camera space by estimated pose
    verts_cam = (ob_in_cam[:3, :3] @ mesh.vertices.T).T + ob_in_cam[:3, 3]
    normals_cam = (ob_in_cam[:3, :3] @ mesh.vertex_normals.T).T if mesh.vertex_normals is not None else None
    mesh_colors = None
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        mesh_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    rr.log("world/mesh", rr.Mesh3D(
        vertex_positions=verts_cam,
        triangle_indices=mesh.faces,
        vertex_normals=normals_cam,
        vertex_colors=mesh_colors,
    ), static=True)
    print("Rerun visualization launched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate object pose with FoundationPose for a single condition frame")
    parser.add_argument("--data_dir", type=str, required=True, help="Sequence root dir (contains rgb/, depth/, meta/, mask_object/, gt_mesh/)")
    parser.add_argument("--cond_index", type=int, required=True, help="Condition frame index")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for pose txt")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--vis_in_rerun", action="store_true", help="Visualize pose + depth + mesh in Rerun")
    args = parser.parse_args()
    main(args)
