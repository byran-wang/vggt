import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt_eval_vis_nvdiffrast import (
    _load_gt_valid_flags,
    load_image_info,
    get_sam3d_mesh_path,
    build_mesh_in_object_space,
    _mesh_vertex_colors,
    load_hand_mesh_for_frame,
    load_hand_mesh_from_hand_object_alignment,
    ensure_sealed_right_hand_mesh,
)
from robust_hoi_pipeline.pipeline_utils import load_preprocessed_frame, load_sam3d_transform
from utils_simba.logger import get_logger
from utils_simba.rerun import (
    log_camera_frame,
    stamp_frame_text,
    compute_vertex_normals,
    backproject_depth_to_points,
)

logger = get_logger(__name__)


def _build_camera_c2w(c2o_local: np.ndarray, scale: float) -> np.ndarray:
    """Build a valid SO3+scaled-translation camera-to-world (=object) transform."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = c2o_local[:3, :3]
    c2w[:3, 3] = c2o_local[:3, 3] * scale
    return c2w


def _hand_verts_to_object_space(
    hand_verts_cam: np.ndarray, c2o_local: np.ndarray, scale: float
) -> np.ndarray:
    """Transform hand vertices from camera space to scaled object space."""
    R = c2o_local[:3, :3]
    t = c2o_local[:3, 3]
    return (scale * ((R @ hand_verts_cam.T).T + t)).astype(np.float32)


def main(args):
    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    neus_dir = results_dir.parent / "pipeline_neus_global" / "neus_training"

    image_info = load_image_info(results_dir)
    sam3d_tf = load_sam3d_transform(sam3d_dir, args.cond_index)
    sam3d_to_cond_cam = sam3d_tf["sam3d_to_cond_cam"]
    scale = float(sam3d_tf["scale"])

    frame_indices = np.asarray(image_info["frame_indices"])
    register_flags = np.asarray(image_info["register"], dtype=bool)
    invalid_flags = np.asarray(image_info["invalid"], dtype=bool)
    valid_flags = register_flags & (~invalid_flags)
    gt_valid_flags = None
    if bool(args.vis_gt):
        seq_name = sam3d_dir.parent.name
        gt_valid_flags = _load_gt_valid_flags(seq_name, frame_indices)
        valid_flags = valid_flags & gt_valid_flags
        logger.info(
            f"Using GT-valid + registered frames: {int(valid_flags.sum())}/{len(valid_flags)} "
            f"(registered={int((register_flags & (~invalid_flags)).sum())}, gt_valid={int(gt_valid_flags.sum())})"
        )
    else:
        logger.info(
            f"Using registered frames only: {int(valid_flags.sum())}/{len(valid_flags)} "
            f"(GT filtering disabled)"
        )

    c2o = np.asarray(image_info["c2o"], dtype=np.float64)

    if args.mesh_type == "sam3d":
        mesh_path = get_sam3d_mesh_path(sam3d_dir, args.cond_index)
    elif args.mesh_type == "neus":
        obj_files = sorted(neus_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime)
        if not obj_files:
            raise FileNotFoundError(f"No .obj file found under {neus_dir}")
        mesh_path = obj_files[-1]
    else:
        raise ValueError(f"Unsupported mesh type: {args.mesh_type}")
    logger.info(f"Using mesh: {mesh_path}")

    mesh_obj = build_mesh_in_object_space(
        mesh_path=mesh_path,
        frame_indices=frame_indices,
        c2o=c2o,
        scale=scale,
        sam3d_to_cond_cam=sam3d_to_cond_cam,
        cond_index=args.cond_index,
    )
    obj_verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
    obj_faces = np.asarray(mesh_obj.faces, dtype=np.int32)
    obj_colors = _mesh_vertex_colors(mesh_obj)

    data_preprocess_dir = sam3d_dir.parent / "pipeline_preprocess"

    # --- Init rerun ---
    save_to_file = args.rrd_output_path is not None
    rr.init("hoi_eval_vis", spawn=not save_to_file)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera", origin="world/camera"),
            ),
            column_shares=[2, 1],
        ),
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # --- Log static object mesh ---
    obj_normals = compute_vertex_normals(obj_verts, obj_faces)
    rr.log(
        "world/object_mesh",
        rr.Mesh3D(
            vertex_positions=obj_verts,
            triangle_indices=obj_faces,
            vertex_normals=obj_normals,
            vertex_colors=obj_colors,
        ),
        static=True,
    )

    # --- Per-frame loop ---
    logger.info(f"Logging {int(valid_flags.sum())} valid frames to rerun...")
    for local_idx, frame_idx in enumerate(frame_indices):
        if not bool(valid_flags[local_idx]):
            continue

        preprocess_data = load_preprocessed_frame(data_preprocess_dir, int(frame_idx))
        image = preprocess_data.get("image")
        K = preprocess_data.get("intrinsics")
        if image is None or K is None:
            logger.warning(f"Frame {frame_idx}: missing image or intrinsics, skipping")
            continue

        rr.set_time_sequence("frame", int(frame_idx))

        c2w = _build_camera_c2w(c2o[local_idx], scale)
        image_stamped = stamp_frame_text(image, f"Frame {frame_idx:04d}")
        log_camera_frame(
            "world/camera",
            K,
            c2w,
            image_stamped,
            image_plane_distance=args.image_plane_distance,
            jpeg_quality=args.jpeg_quality,
            static=False,
        )

        # --- Hand mesh ---
        if args.render_hand:
            if args.hand_mode == "trans":
                hand_mesh_cam = load_hand_mesh_for_frame(data_preprocess_dir, int(frame_idx))
            elif args.hand_mode in ("h", "o", "ho", "all"):
                hand_mesh_cam = load_hand_mesh_from_hand_object_alignment(
                    results_dir, data_preprocess_dir, args.hand_mode, int(frame_idx)
                )
            else:
                raise ValueError(f"Unsupported hand_mode: {args.hand_mode}")

            if hand_mesh_cam is not None:
                hand_verts, hand_faces = ensure_sealed_right_hand_mesh(
                    hand_mesh_cam["vertices"], hand_mesh_cam["faces"]
                )
                hand_verts_obj = _hand_verts_to_object_space(hand_verts, c2o[local_idx], scale)
                hand_normals = compute_vertex_normals(hand_verts_obj, hand_faces)
                rr.log(
                    "world/hand_mesh",
                    rr.Mesh3D(
                        vertex_positions=hand_verts_obj,
                        triangle_indices=hand_faces,
                        vertex_normals=hand_normals,
                        mesh_material=rr.Material(albedo_factor=[150, 80, 220, 255]),
                    ),
                    static=False,
                )
            else:
                rr.log("world/hand_mesh", rr.Clear(recursive=False), static=False)

    if save_to_file:
        out_path = Path(args.rrd_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(out_path))
        logger.info(f"Saved rerun recording to {out_path}")
        logger.info(f"  View with: rerun {out_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HOI pipeline results in Rerun")
    parser.add_argument("--result_folder", type=str, required=True,
                        help="Path to output/<seq>/pipeline_joint_opt")
    parser.add_argument("--SAM3D_dir", type=str, required=True,
                        help="Path to <seq>/SAM3D_aligned_post_process")
    parser.add_argument("--cond_index", type=int, required=True,
                        help="Condition frame index")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (unused; kept for API compatibility)")
    parser.add_argument("--rrd_output_path", type=str, default=None,
                        help="Save rerun recording to this .rrd file path")
    parser.add_argument("--vis_gt", type=int, default=1,
                        help="Use GT-valid filtering (1) or not (0)")
    parser.add_argument("--render_hand", dest="render_hand", action="store_true", default=True,
                        help="Render sealed right-hand mesh together with the object mesh")
    parser.add_argument("--mesh_type", type=str, default="neus", choices=["sam3d", "neus"],
                        help="Mesh source: 'neus' (default) or 'sam3d'")
    parser.add_argument("--hand_mode", type=str, default="ho",
                        choices=["trans", "h", "o", "ho", "all"],
                        help="Hand fitting mode")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--image_plane_distance", type=float, default=5.0,
                        help="Camera frustum image plane distance in rerun")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
