import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))
sys.path.insert(0, str(project_root / "third_party" / "BlenderToolbox"))

from robust_hoi_pipeline.pipeline_hand_object_mesh import _hand_mesh_in_obj
from robust_hoi_pipeline.pipeline_joint_opt_eval_vis_nvdiffrast import (
    _load_gt_valid_flags,
    build_mesh_in_object_space,
    ensure_sealed_right_hand_mesh,
    get_sam3d_mesh_path,
    load_hand_mesh_for_frame,
    load_hand_mesh_from_hand_object_alignment,
    load_image_info,
)
from third_party.utils_simba.utils_simba.eval_vis import (
    load_mesh_as_trimesh,
)
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform
from utils_simba.eval_vis import create_video
from utils_simba.logger import get_logger

logger = get_logger(__name__)

# OpenCV (+X right, +Y down, +Z forward) -> Blender (+X right, +Y up, -Z forward)
_CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def _save_hand_cam_space(hand_cam: dict, out_path: Path) -> None:
    verts, faces = ensure_sealed_right_hand_mesh(hand_cam["vertices"], hand_cam["faces"])
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(out_path))


def _save_hand_in_obj_space(hand_cam: dict, c2o_with_scale_i: np.ndarray, out_path: Path) -> None:
    verts, faces = ensure_sealed_right_hand_mesh(hand_cam["vertices"], hand_cam["faces"])
    verts_h = np.hstack([verts, np.ones((len(verts), 1), dtype=np.float32)])
    verts_obj = (c2o_with_scale_i @ verts_h.T).T[:, :3].astype(np.float32)
    trimesh.Trimesh(vertices=verts_obj, faces=faces, process=False).export(str(out_path))


def _render_with_camera_pose(*, image_resolution, num_samples, obj_mesh_path, hand_mesh_path,
                             output_path, camera_pose_obj, light_angle, focal_length,
                             obj_rgb, hand_rgb, shading="smooth", subdivision_iteration=0,
                             shadow_brightness=0.9, save_blend=False):
    """Render object (in obj space) + hand (in cam space) by placing the Blender camera
    and the hand mesh's object transform at ``camera_pose_obj`` (4x4, c2o_with_scale).

    Mesh vertex data is *not* modified: the hand mesh's cam-space verts are placed in
    obj-space via Blender's object transform, which equals the camera pose."""
    import bpy
    import mathutils

    from blendertoolbox.blenderInit import blenderInit
    from blendertoolbox.invisibleGround import invisibleGround
    from blendertoolbox.readMesh import readMesh
    from blendertoolbox.render_mesh_default import colorObj
    from blendertoolbox.renderImage import renderImage
    from blendertoolbox.setLight_ambient import setLight_ambient
    from blendertoolbox.setLight_sun import setLight_sun
    from blendertoolbox.setMat_plastic import setMat_plastic
    from blendertoolbox.shadowThreshold import shadowThreshold
    from blendertoolbox.subdivision import subdivision

    blenderInit(image_resolution[0], image_resolution[1], num_samples, exposure=1.5, use_GPU=True)

    obj_mesh = readMesh(str(obj_mesh_path), (0, 0, 0), (0, 0, 0), (1, 1, 1))
    subdivision(obj_mesh, level=subdivision_iteration)
    setMat_plastic(obj_mesh, colorObj((obj_rgb[0], obj_rgb[1], obj_rgb[2], 1), 0.5, 1.0, 1.0, 0.0, 2.0))

    hand_mesh = readMesh(str(hand_mesh_path), (0, 0, 0), (0, 0, 0), (1, 1, 1))
    subdivision(hand_mesh, level=subdivision_iteration)
    setMat_plastic(hand_mesh, colorObj((hand_rgb[0], hand_rgb[1], hand_rgb[2], 1), 0.5, 1.0, 1.0, 0.0, 2.0))

    if shading == "smooth":
        bpy.ops.object.shade_smooth()
    elif shading == "flat":
        bpy.ops.object.shade_flat()
    else:
        raise ValueError(f"shading must be 'smooth' or 'flat', got {shading}")

    invisibleGround(location=(0, 0, -0.6), shadowBrightness=shadow_brightness)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.object
    cam.data.lens = float(focal_length)
    cam.matrix_world = mathutils.Matrix((camera_pose_obj @ _CV_TO_BLENDER).tolist())
    bpy.context.view_layer.update()

    setLight_sun(light_angle, 2, 0.3)
    setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if save_blend:
        blend_path = Path(output_path).with_suffix(".blend")
        bpy.ops.wm.save_mainfile(filepath=str(blend_path))
    renderImage(str(output_path), cam)


def main(args):
    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    out_dir = Path(args.out_dir)
    render_dir = out_dir / "renders"
    mesh_cache_dir = out_dir / "mesh_cache"
    render_dir.mkdir(parents=True, exist_ok=True)
    mesh_cache_dir.mkdir(parents=True, exist_ok=True)
    neus_dir = results_dir / "neus_training"

    image_info = load_image_info(results_dir)
    sam3d_tf = load_sam3d_transform(sam3d_dir, args.cond_index)
    sam3d_to_cond_cam = sam3d_tf["sam3d_to_cond_cam"]
    scale = float(sam3d_tf["scale"])

    frame_indices = np.asarray(image_info["frame_indices"])
    register_flags = np.asarray(image_info["register"], dtype=bool)
    invalid_flags = np.asarray(image_info["invalid"], dtype=bool)
    valid_flags = register_flags & (~invalid_flags)
    if bool(args.vis_gt):
        seq_name = sam3d_dir.parent.name
        gt_valid_flags = _load_gt_valid_flags(seq_name, frame_indices)
        valid_flags = valid_flags & gt_valid_flags
        logger.info(
            f"Using GT-valid + registered frames: {int(valid_flags.sum())}/{len(valid_flags)}"
        )
    else:
        logger.info(
            f"Using registered frames only: {int(valid_flags.sum())}/{len(valid_flags)}"
        )

    c2o = np.asarray(image_info["c2o"], dtype=np.float64)
    c2o_with_scale = c2o.copy()
    c2o_with_scale[:, :3, :3] *= scale
    c2o_with_scale[:, :3, 3] *= scale

    if args.mesh_type == "sam3d":
        mesh_path = get_sam3d_mesh_path(sam3d_dir, args.cond_index)
    elif args.mesh_type == "neus":
        obj_files = sorted(neus_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime)
        if not obj_files:
            raise FileNotFoundError(f"No .obj file found under {neus_dir}")
        mesh_path = obj_files[-1]
    else:
        raise ValueError(f"Unsupported mesh_type: {args.mesh_type}")
    logger.info(f"Using object mesh ({args.mesh_type}): {mesh_path}")

    object_mesh = build_mesh_in_object_space(
        mesh_path=mesh_path,
        frame_indices=frame_indices,
        c2o=c2o,
        scale=scale,
        sam3d_to_cond_cam=sam3d_to_cond_cam,
        cond_index=args.cond_index,
    )
    object_mesh = load_mesh_as_trimesh(mesh_path)
    object_mesh_path = mesh_cache_dir / "object.obj"
    object_mesh.export(str(object_mesh_path))

    data_preprocess_dir = sam3d_dir.parent / "pipeline_preprocess"
    light_angle = eval(args.light_angle)

    rendered_count = 0
    frame_list = np.asarray(image_info["frame_indices"]).tolist()
    iter_frames = args.frame_list if args.frame_list else frame_list
    for frame_idx in iter_frames:
        frame_idx = int(frame_idx)
        if frame_idx not in frame_list:
            logger.warning(f"[skip] frame {frame_idx}: not in frame_indices")
            continue
        local_idx = frame_list.index(frame_idx)
        if not bool(valid_flags[local_idx]):
            continue

        if args.hand_mode == "trans":
            hand_cam = load_hand_mesh_for_frame(data_preprocess_dir, int(frame_idx))
        elif args.hand_mode in ("h", "o", "ho"):
            hand_cam = load_hand_mesh_from_hand_object_alignment(
                results_dir, data_preprocess_dir, args.hand_mode, int(frame_idx)
            )
        else:
            raise ValueError(f"Unsupported hand_mode: {args.hand_mode}")
        if hand_cam is None:
            logger.warning(f"[skip] frame {frame_idx}: hand mesh not available")
            continue
        hand_mesh_path = mesh_cache_dir / f"hand_{int(frame_idx):04d}.obj"
        hand_obj = _hand_mesh_in_obj(hand_cam, c2o[local_idx])
        hand_obj.export(str(hand_mesh_path))

        png_path = render_dir / f"{frame_idx:04d}.png"
        _render_with_camera_pose(
            image_resolution=args.image_resolution,
            num_samples=args.number_of_samples,
            obj_mesh_path=object_mesh_path,
            hand_mesh_path=hand_mesh_path,
            output_path=png_path,
            camera_pose_obj=c2o[local_idx],
            light_angle=light_angle,
            focal_length=args.focal_length,
            obj_rgb=args.obj_mesh_RGB,
            hand_rgb=args.hand_mesh_RGB,
            save_blend=bool(args.debug),
        )
        logger.info(f"Rendered frame {frame_idx} -> {png_path}")
        rendered_count += 1

    if rendered_count == 0:
        raise RuntimeError("No frames rendered")

    if args.fps > 0:
        video_path = out_dir / "hand_object_mesh.mp4"
        create_video(render_dir, video_path, fps=args.fps)
        logger.info(f"Saved video -> {video_path}")
    logger.info(f"Saved {rendered_count} renders to {render_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render hand+object meshes per-frame with Blender. The hand mesh stays in "
            "camera space and the object mesh in object space; per-frame variation comes "
            "from setting the Blender camera pose to c2o_with_scale (and using that same "
            "matrix as the hand mesh's Blender object transform, so its cam-space verts "
            "are placed correctly without modifying mesh data)."
        ),
    )
    parser.add_argument("--result_folder", type=str, required=True,
                        help="Path to output/<seq>/pipeline_joint_opt")
    parser.add_argument("--SAM3D_dir", type=str, required=True,
                        help="Path to <seq>/SAM3D_aligned_post_process")
    parser.add_argument("--cond_index", type=int, required=True, help="Condition frame index")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mesh_type", type=str, default="neus", choices=["sam3d", "neus"])
    parser.add_argument("--hand_mode", type=str, default="ho", choices=["trans", "h", "o", "ho"])
    parser.add_argument("--vis_gt", type=int, default=1, help="Use GT-valid filtering (1) or not (0)")
    parser.add_argument("--fps", type=int, default=6, help="Output video FPS (0 to disable)")
    parser.add_argument("--image_resolution", type=int, nargs=2, default=[1080, 1080])
    parser.add_argument("--number_of_samples", type=int, default=200)
    parser.add_argument("--focal_length", type=float, default=35.0,
                        help="Blender camera focal length (mm equivalent)")
    parser.add_argument("--light_angle", type=str, default="(6, -30, -155)")
    parser.add_argument("--obj_mesh_RGB", type=float, nargs=3,
                        default=[144.0 / 255, 210.0 / 255, 236.0 / 255])
    parser.add_argument("--hand_mesh_RGB", type=float, nargs=3,
                        default=[200.0 / 255, 180.0 / 255, 220.0 / 255])
    parser.add_argument("--debug", action="store_true",
                        help="Save the per-frame .blend file alongside each rendered PNG")
    parser.add_argument("--frame_list", type=int, nargs="+", default=None,
                        help="Optional explicit list of frame indices to render such as --frame_list 290 295 300"
                             "If omitted, iterates over all frames from image_info[\"frame_indices\"].")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
