import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import rerun as rr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

import vggt.utils.gt as gt
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_preprocessed_frame


def _to_numpy(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _load_frame_indices(data_preprocess_dir: Path):
    frame_list_file = data_preprocess_dir / "frame_list.txt"
    if frame_list_file.exists():
        return load_frame_list(data_preprocess_dir)

    rgb_dir = data_preprocess_dir / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Cannot find frame list or rgb dir under {data_preprocess_dir}")

    fids = []
    for p in sorted(rgb_dir.glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            fids.append(int(p.stem))
        except ValueError:
            continue
    if len(fids) == 0:
        raise RuntimeError(f"No frame ids found in {rgb_dir}")
    return fids


def compute_frustum_lines(K, H, W, c2w, depth=0.05):
    """Compute camera frustum line segments in world space."""
    K_inv = np.linalg.inv(K[:3, :3])
    corners_px = np.array([
        [0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1],
    ], dtype=np.float64)
    corners_cam = (K_inv @ corners_px.T).T * depth
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    origin_w = t.copy()
    corners_w = (R @ corners_cam.T).T + t

    segments = []
    for c in corners_w:
        segments.append(np.stack([origin_w, c]))
    for j in range(4):
        segments.append(np.stack([corners_w[j], corners_w[(j + 1) % 4]]))
    return segments


def main(args):
    data_dir = Path(args.data_dir)
    data_preprocess_dir = data_dir / "pipeline_preprocess"

    frame_indices = _load_frame_indices(data_preprocess_dir)
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]

    seq_name = data_dir.name

    def get_image_fids():
        return frame_indices

    print(f"Loading GT data for {seq_name} with {len(frame_indices)} frames...")
    data_gt = gt.load_data(seq_name, get_image_fids)

    # Extract GT arrays
    gt_o2c = _to_numpy(data_gt["o2c"]).astype(np.float64)       # (N, 4, 4)
    gt_is_valid = _to_numpy(data_gt["is_valid"]).astype(bool)    # (N,)
    gt_K = _to_numpy(data_gt["K"]).astype(np.float64)            # (3,3) or (N,3,3)

    # Load GT object mesh
    gt_mesh_path = data_gt["mesh_name.object"]
    gt_mesh = None
    if gt_mesh_path and Path(gt_mesh_path).exists():
        gt_mesh = trimesh.load(str(gt_mesh_path), force="mesh", process=False)
    gt_verts_can = np.array(gt_mesh.vertices, dtype=np.float32) if gt_mesh is not None else None
    gt_faces = np.array(gt_mesh.faces, dtype=np.uint32) if gt_mesh is not None else None

    gt_vertex_colors = None
    if gt_mesh is not None:
        if hasattr(gt_mesh, "visual") and hasattr(gt_mesh.visual, "vertex_colors"):
            vc = np.asarray(gt_mesh.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == len(gt_verts_can) and vc.shape[1] >= 3:
                gt_vertex_colors = vc[:, :3].astype(np.uint8)
        if gt_vertex_colors is None and hasattr(gt_mesh, "visual") and hasattr(gt_mesh.visual, "to_color"):
            vc = np.asarray(gt_mesh.visual.to_color().vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == len(gt_verts_can) and vc.shape[1] >= 3:
                gt_vertex_colors = vc[:, :3].astype(np.uint8)

    # Load GT hand mesh
    gt_hand_verts_cam = None
    gt_hand_faces = None
    if args.render_hand and "v3d_c.right" in data_gt and "faces.right" in data_gt:
        gt_hand_verts_cam = _to_numpy(data_gt["v3d_c.right"]).astype(np.float32)
        gt_hand_faces = _to_numpy(data_gt["faces.right"]).astype(np.int64)
        try:
            from common.body_models import seal_mano_mesh_np
            gt_hand_verts_cam, gt_hand_faces = seal_mano_mesh_np(
                gt_hand_verts_cam, gt_hand_faces, is_rhand=True)
            gt_hand_faces = gt_hand_faces.astype(np.int32)
        except Exception as exc:
            print(f"[warn] Failed to seal GT hand mesh: {exc}")
            gt_hand_faces = gt_hand_faces.astype(np.int32)

    # Init rerun
    rr.init("gt_vis", spawn=True)
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera", origin="world/camera"),
                rrb.Spatial2DView(name="Depth", origin="world/depth"),
                rrb.Spatial2DView(name="Depth (Object)", origin="world/depth_obj"),
                rrb.Spatial2DView(name="Depth (Hand)", origin="world/depth_hand"),
            ),
            column_shares=[2, 1],
        ),
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Log GT mesh at identity (canonical space) as static
    if gt_verts_can is not None and gt_faces is not None:
        mesh_kwargs = dict(
            vertex_positions=gt_verts_can,
            triangle_indices=gt_faces,
        )
        if gt_vertex_colors is not None:
            mesh_kwargs["vertex_colors"] = gt_vertex_colors
        rr.log("world/gt_mesh", rr.Mesh3D(**mesh_kwargs), static=True)

        # Also log GT mesh vertices as 3D points
        pts_kwargs = dict(positions=gt_verts_can, radii=0.001)
        if gt_vertex_colors is not None:
            pts_kwargs["colors"] = gt_vertex_colors
        rr.log("world/gt_mesh_points", rr.Points3D(**pts_kwargs), static=True)

    print(f"Visualizing {len(frame_indices)} frames...")
    for i, fid in enumerate(frame_indices):
        if i >= len(gt_o2c) or i >= len(gt_is_valid) or not gt_is_valid[i]:
            continue

        rr.set_time_sequence("frame", i)

        o2c_i = gt_o2c[i]                                    # (4, 4) obj-to-cam
        c2o_i = np.linalg.inv(o2c_i).astype(np.float32)      # (4, 4) cam-to-obj
        K_i = gt_K if gt_K.ndim == 2 else gt_K[i]

        # Load preprocessed image and intrinsics
        preprocess_data = load_preprocessed_frame(data_preprocess_dir, int(fid))
        img = preprocess_data.get("image")
        K_img = preprocess_data.get("intrinsics")
        if K_img is not None:
            K_i = np.asarray(K_img, dtype=np.float64)

        # Log camera transform and pinhole
        entity = "world/camera"
        rr.log(entity, rr.Transform3D(
            translation=c2o_i[:3, 3],
            mat3x3=c2o_i[:3, :3],
        ), static=False)

        if img is not None:
            H, W = img.shape[:2]
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((10, 10), f"Frame {fid:04d}", fill=(255, 255, 0), font=font)
            img = np.array(pil_img)

            rr.log(entity, rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(K_i[0, 0]), float(K_i[1, 1])],
                principal_point=[float(K_i[0, 2]), float(K_i[1, 2])],
                image_plane_distance=args.image_plane_distance,
            ), static=False)
            rr.log(entity, rr.Image(img).compress(jpeg_quality=args.jpeg_quality), static=False)

        # Log depth views
        depth = preprocess_data.get("depth")
        mask_obj = preprocess_data.get("mask_obj")
        mask_hand = preprocess_data.get("mask_hand")

        if depth is not None:
            depth_f32 = depth.astype(np.float32)
            rr.log("world/depth", rr.DepthImage(depth_f32), static=False)

            # Object-masked depth
            if mask_obj is not None:
                obj_depth = depth_f32.copy()
                obj_depth[mask_obj == 0] = 0.0
                rr.log("world/depth_obj", rr.DepthImage(obj_depth), static=False)

            # Hand-masked depth
            if mask_hand is not None:
                hand_depth = depth_f32.copy()
                hand_depth[mask_hand == 0] = 0.0
                rr.log("world/depth_hand", rr.DepthImage(hand_depth), static=False)

            # Backproject depth to 3D point cloud in object space
            if img is not None:
                H, W = img.shape[:2]
                K_inv = np.linalg.inv(K_i[:3, :3])
                uu, vv = np.meshgrid(np.arange(W), np.arange(H))
                pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3).astype(np.float64)
                valid = depth.ravel() > 0
                rays_cam = (K_inv @ pixels.T).T  # (H*W, 3)
                pts_cam = rays_cam * depth.ravel()[:, None]

                def _log_points(entity, pt_mask, color_src):
                    pts = pts_cam[pt_mask]
                    pts_obj = (c2o_i[:3, :3] @ pts.T).T + c2o_i[:3, 3]
                    cs = color_src.reshape(-1, 3)[pt_mask] if color_src is not None else None
                    if len(pts_obj) > 50000:
                        sel = np.random.choice(len(pts_obj), 50000, replace=False)
                        pts_obj = pts_obj[sel]
                        cs = cs[sel] if cs is not None else None
                    kw = dict(positions=pts_obj.astype(np.float32), radii=0.001)
                    if cs is not None:
                        kw["colors"] = cs
                    rr.log(entity, rr.Points3D(**kw), static=False)

                # Full depth point cloud
                _log_points("world/depth_points", valid, img)

                # Object-masked point cloud
                if mask_obj is not None:
                    obj_valid = valid & (mask_obj.ravel() > 0)
                    _log_points("world/depth_points_obj", obj_valid, img)

                # Hand-masked point cloud
                if mask_hand is not None:
                    hand_valid = valid & (mask_hand.ravel() > 0)
                    _log_points("world/depth_points_hand", hand_valid, img)

        # Log hand mesh in camera space (transform to object space for visualization)
        if gt_hand_verts_cam is not None and gt_hand_faces is not None and i < len(gt_hand_verts_cam):
            hv = gt_hand_verts_cam[i]
            if np.isfinite(hv).all() and hv.min() > -500:
                # Transform hand verts from camera space to object (world) space
                hv_obj = (c2o_i[:3, :3] @ hv.T).T + c2o_i[:3, 3]
                rr.log("world/gt_hand", rr.Mesh3D(
                    vertex_positions=hv_obj.astype(np.float32),
                    triangle_indices=gt_hand_faces,
                    mesh_material=rr.Material(albedo_factor=[225, 186, 160]),
                ), static=False)

        print(f"  Frame {fid:04d}: valid")

    print(f"Done. Visualized GT for {seq_name}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GT mesh, pose, and images in Rerun")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Sequence directory, e.g., HO3D_v3/train/MC1")
    parser.add_argument("--render_hand", action="store_true", default=True,
                        help="Render GT hand mesh")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Maximum number of frames to visualize")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--image_plane_distance", type=float, default=3.0,
                        help="Image plane distance for camera frustum size")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
