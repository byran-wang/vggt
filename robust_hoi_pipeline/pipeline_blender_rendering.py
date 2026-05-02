import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

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


def _set_camera_intrinsics(cam_data, K, image_w, image_h, sensor_width=36.0):
    """Set Blender camera (cam_data) lens/shift_x/shift_y from a 3x3 OpenCV K
    matrix and target image size. Blender uses sensor_width-based fov, so:

        lens (mm)      = fx * sensor_width / image_w
        shift_x        = -(cx - image_w/2) / max(image_w, image_h)
        shift_y        =  (cy - image_h/2) / max(image_w, image_h)
        sensor_height  =  sensor_width * image_h / image_w  (matches aspect)
    """
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_width * image_h / image_w
    cam_data.lens = fx * sensor_width / image_w
    max_dim = max(image_w, image_h)
    cam_data.shift_x = -(cx - image_w / 2.0) / max_dim
    cam_data.shift_y = (cy - image_h / 2.0) / max_dim


def _build_animated_scene(*, image_resolution, num_samples, obj_mesh_path,
                          hand_mesh_paths, frame_indices, object_poses_world,
                          light_angle, K, obj_rgb, hand_rgb, plane_y, plane_z,
                          light_strength, light_size, ambient_color,
                          shading="smooth", subdivision_iteration=0,
                          shadow_brightness=0.9):
    """Build a Blender scene with a *fixed* camera and animated hand+object.

    World is the OpenCV cam_cond local coord system (camera.matrix_world is
    set to ``_CV_TO_BLENDER`` so Blender world == OpenCV cam coords by
    construction). The camera sits at world origin and renders the scene.

    Animation:
      - Object: single mesh at obj-space verts, world transform animated to
        ``object_poses_world[i]`` (= o2c[f]) keyframed per ``frame_indices``.
      - Hand: ``len(frame_indices)`` separate Blender objects. Each is loaded
        with the corresponding frame's *cam-space* verts (no pre-transform);
        hand mesh data IS its world position. Visibility is keyframed so only
        the active frame's hand renders.

    Plane: a horizontal floor below the scene. ``plane_y`` is the OpenCV +Y
    coord (= "down") of the plane center; ``plane_z`` is the +Z (forward)
    coord. The plane is rotated 90° around X so its normal is the world Y
    axis (matches OpenCV "down" direction).

    Returns: (cam, obj_mesh, hand_obj_by_frame)."""
    import bpy
    import mathutils

    from blendertoolbox.blenderInit import blenderInit
    from blendertoolbox.invisibleGround import invisibleGround
    from blendertoolbox.readMesh import readMesh
    from blendertoolbox.render_mesh_default import colorObj
    from blendertoolbox.setLight_ambient import setLight_ambient
    from blendertoolbox.setLight_sun import setLight_sun
    from blendertoolbox.setMat_plastic import setMat_plastic
    from blendertoolbox.shadowThreshold import shadowThreshold
    from blendertoolbox.subdivision import subdivision

    blenderInit(image_resolution[0], image_resolution[1], num_samples,
                exposure=1.5, use_GPU=True)

    # --- object mesh (animated world transform) ---
    obj_mesh = readMesh(str(obj_mesh_path), (0, 0, 0), (0, 0, 0), (1, 1, 1))
    obj_mesh.name = "object"
    obj_mesh.rotation_mode = "XYZ"
    subdivision(obj_mesh, level=subdivision_iteration)
    setMat_plastic(obj_mesh, colorObj((obj_rgb[0], obj_rgb[1], obj_rgb[2], 1),
                                      0.5, 1.0, 1.0, 0.0, 2.0))
    if shading == "smooth":
        bpy.ops.object.shade_smooth()
    elif shading == "flat":
        bpy.ops.object.shade_flat()
    else:
        raise ValueError(f"shading must be 'smooth' or 'flat', got {shading}")

    # --- per-frame hand meshes (each at world identity, vary by visibility) ---
    # All hands share a single material (created once on the first hand,
    # reused by appending the shared material datablock to the rest) and live
    # in a dedicated "hands" collection so the Outliner stays tidy.
    hand_collection = bpy.data.collections.new("hands")
    bpy.context.scene.collection.children.link(hand_collection)

    hand_obj_by_frame = {}
    shared_hand_mat = None
    for f, path in zip(frame_indices, hand_mesh_paths):
        f = int(f)
        h = readMesh(str(path), (0, 0, 0), (0, 0, 0), (1, 1, 1))
        h.name = f"hand_{f:04d}"
        subdivision(h, level=subdivision_iteration)
        if shared_hand_mat is None:
            setMat_plastic(h, colorObj((hand_rgb[0], hand_rgb[1], hand_rgb[2], 1),
                                       0.5, 1.0, 1.0, 0.0, 2.0))
            shared_hand_mat = h.active_material
            shared_hand_mat.name = "hand_material"
        else:
            h.data.materials.append(shared_hand_mat)
            h.active_material = shared_hand_mat
        if shading == "smooth":
            bpy.ops.object.shade_smooth()
        # Move object from the default Scene Collection into the hands collection.
        for coll in list(h.users_collection):
            coll.objects.unlink(h)
        hand_collection.objects.link(h)
        hand_obj_by_frame[f] = h

    # --- horizontal floor (perpendicular to OpenCV +Y = "down") ---
    import math
    invisibleGround(location=(0.0, float(plane_y), float(plane_z)),
                    shadowBrightness=shadow_brightness)
    plane_obj = bpy.context.object
    plane_obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)

    # GUI-tuned plane material tweaks (matched to debug7 scene): a small white
    # emission lifts the floor's apparent color and a tiny coat roughness gives
    # the shadow edge a softer highlight. Without these, plain shadow-catcher
    # comes out too gray.
    if plane_obj.active_material and plane_obj.active_material.use_nodes:
        bsdf = plane_obj.active_material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 0.25
            if "Coat Roughness" in bsdf.inputs:
                bsdf.inputs["Coat Roughness"].default_value = 0.03

    # --- camera fixed at world origin (Blender world ≡ OpenCV cam_cond) ---
    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.object
    cam.rotation_mode = "XYZ"
    if K is not None:
        _set_camera_intrinsics(cam.data, K, image_resolution[0], image_resolution[1])
    cam.matrix_world = mathutils.Matrix(_CV_TO_BLENDER.tolist())

    setLight_sun(light_angle, light_strength, light_size)
    setLight_ambient(color=ambient_color)
    try:
        shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")
    except (AttributeError, IndexError):
        # bpy 5.0+ removed Scene.node_tree (compositor moved); harmless skip.
        pass

    # --- timeline ---
    fmin = int(min(frame_indices))
    fmax = int(max(frame_indices))
    bpy.context.scene.frame_start = fmin
    bpy.context.scene.frame_end = fmax
    bpy.context.scene.camera = cam

    # --- object animation: location + rotation_euler keyframes ---
    for i, f in enumerate(frame_indices):
        f = int(f)
        pose = np.asarray(object_poses_world[i], dtype=np.float64)
        mat = mathutils.Matrix(pose.tolist())
        loc, rot_quat, _scale = mat.decompose()
        obj_mesh.location = loc
        obj_mesh.rotation_euler = rot_quat.to_euler("XYZ")
        obj_mesh.keyframe_insert(data_path="location", frame=f)
        obj_mesh.keyframe_insert(data_path="rotation_euler", frame=f)

    # --- hand visibility: only the matching frame's hand is visible ---
    # bpy auto-sets CONSTANT interpolation on boolean property keyframes, so
    # the visibility steps cleanly without explicit interp management.
    for f, h in hand_obj_by_frame.items():
        kfs = ((fmin, True), (f, False), (f + 1, True))
        for tgt_frame, tgt_hide in kfs:
            if tgt_frame > fmax + 1:
                continue
            h.hide_render = tgt_hide
            h.hide_viewport = tgt_hide
            h.keyframe_insert(data_path="hide_render", frame=tgt_frame)
            h.keyframe_insert(data_path="hide_viewport", frame=tgt_frame)

    return cam, obj_mesh, hand_obj_by_frame


def _render_animated_scene(*, frame_indices, render_dir):
    """Render each ``frame_indices`` entry of the current Blender scene to
    <render_dir>/<frame:04d>.png. Caller must have built the scene already
    (via ``_build_animated_scene``) and the scene's active camera must be set."""
    import bpy
    Path(render_dir).mkdir(parents=True, exist_ok=True)
    for f in frame_indices:
        f = int(f)
        bpy.context.scene.frame_set(f)
        bpy.context.scene.render.filepath = str(Path(render_dir) / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)


def main(args):
    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    out_dir = Path(args.out_dir)
    render_dir = out_dir / "renders"
    mesh_cache_dir = out_dir / "mesh_cache"
    render_dir.mkdir(parents=True, exist_ok=True)
    mesh_cache_dir.mkdir(parents=True, exist_ok=True)
    neus_dir = results_dir.parent / "pipeline_neus_global" / "neus_training"

    seq_dir = sam3d_dir.parent
    meta_pkl = seq_dir / "meta" / f"{int(args.cond_index):04d}.pkl"
    if not meta_pkl.exists():
        meta_pkl = seq_dir / "meta" / "0000.pkl"
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    K_orig = np.asarray(meta["camMat"], dtype=np.float64)

    rgb_sample = next(iter(sorted((seq_dir / "rgb").glob("*.jpg"))))
    rgb_W, rgb_H = Image.open(rgb_sample).size
    if args.image_resolution is None:
        image_resolution = [int(rgb_W), int(rgb_H)]
    else:
        image_resolution = [int(args.image_resolution[0]), int(args.image_resolution[1])]
    # Scale K if the requested render resolution differs from the source RGB.
    K = K_orig.copy()
    if image_resolution[0] != rgb_W or image_resolution[1] != rgb_H:
        sx = image_resolution[0] / float(rgb_W)
        sy = image_resolution[1] / float(rgb_H)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
    logger.info(
        f"RGB {rgb_W}x{rgb_H}; render {image_resolution[0]}x{image_resolution[1]}; "
        f"K=fx,fy={K[0,0]:.2f},{K[1,1]:.2f} cx,cy={K[0,2]:.2f},{K[1,2]:.2f}"
    )

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

    frame_list = np.asarray(image_info["frame_indices"]).tolist()

    candidate_frames = [int(f) for f in (args.frame_list if args.frame_list else frame_list)]
    if args.start_frame is not None:
        candidate_frames = [f for f in candidate_frames if f >= int(args.start_frame)]
    if args.end_frame is not None:
        candidate_frames = [f for f in candidate_frames if f <= int(args.end_frame)]

    # World = OpenCV cam_cond coords; camera at world origin.
    # Object world pose @ frame f = o2c[f] (camera local frame == world).
    # Hand mesh data IS its world position: cam-space verts loaded as-is (no pre-transform).
    sel_frames, sel_obj_poses, sel_hand_paths = [], [], []
    for f in candidate_frames:
        if f not in frame_list:
            logger.warning(f"[skip] frame {f}: not in frame_indices")
            continue
        local_idx = frame_list.index(f)
        if not bool(valid_flags[local_idx]):
            continue

        if args.hand_mode == "trans":
            hand_cam = load_hand_mesh_for_frame(data_preprocess_dir, f)
        elif args.hand_mode in ("h", "o", "ho"):
            hand_cam = load_hand_mesh_from_hand_object_alignment(
                results_dir, data_preprocess_dir, args.hand_mode, f
            )
        else:
            raise ValueError(f"Unsupported hand_mode: {args.hand_mode}")
        if hand_cam is None:
            logger.warning(f"[skip] frame {f}: hand mesh not available")
            continue

        hand_mesh_path = mesh_cache_dir / f"hand_{f:04d}.obj"
        _save_hand_cam_space(hand_cam, hand_mesh_path)

        sel_frames.append(f)
        sel_obj_poses.append(np.linalg.inv(c2o[local_idx]))  # o2c[f]
        sel_hand_paths.append(hand_mesh_path)

    if not sel_frames:
        raise RuntimeError("No frames selected after filtering")

    # --- horizontal floor placement (perpendicular to OpenCV +Y) ---
    # Plane center: Y = scene's max world Y + margin (so plane sits just below
    # the lowest visible content in cam-space; +Y is "down"). Z = mean Z of
    # the scene's world content (puts the plane around the action's depth).
    obj_verts = np.asarray(object_mesh.vertices, dtype=np.float64)
    obj_verts_h = np.hstack([obj_verts, np.ones((len(obj_verts), 1))])
    obj_world_y = []
    obj_world_z = []
    for P in sel_obj_poses:
        wv = (P @ obj_verts_h.T).T[:, :3]
        obj_world_y.append(float(wv[:, 1].max()))
        obj_world_z.append(float(wv[:, 2].mean()))
    hand_max_y = float("-inf")
    hand_mean_z = []
    for path in sel_hand_paths:
        try:
            hv = np.asarray(trimesh.load(str(path), process=False).vertices)
            if len(hv):
                hand_max_y = max(hand_max_y, float(hv[:, 1].max()))
                hand_mean_z.append(float(hv[:, 2].mean()))
        except Exception:
            pass

    if args.plane_y is not None:
        plane_y = float(args.plane_y)
    else:
        scene_max_y = max(max(obj_world_y), hand_max_y)
        plane_y = scene_max_y + float(args.plane_margin)
    if args.plane_distance is not None:
        plane_z_center = float(args.plane_distance)
    else:
        zs = obj_world_z + hand_mean_z
        plane_z_center = float(np.mean(zs)) if zs else 0.5

    logger.info(
        f"Floor plane: y={plane_y:.4f} (cam +Y), z_center={plane_z_center:.4f} (cam +Z)"
    )
    logger.info(f"Selected {len(sel_frames)} frames "
                f"[{sel_frames[0]}..{sel_frames[-1]}], cond={args.cond_index}")

    # --- build the animated scene ONCE ---
    _build_animated_scene(
        image_resolution=image_resolution,
        num_samples=args.number_of_samples,
        obj_mesh_path=object_mesh_path,
        hand_mesh_paths=sel_hand_paths,
        frame_indices=sel_frames,
        object_poses_world=sel_obj_poses,
        light_angle=light_angle,
        K=K,
        obj_rgb=args.obj_mesh_RGB,
        hand_rgb=args.hand_mesh_RGB,
        plane_y=plane_y,
        plane_z=plane_z_center,
        light_strength=float(args.light_strength),
        light_size=float(args.light_size),
        ambient_color=tuple(args.ambient_color) + (1.0,),
    )

    if args.debug:
        import bpy
        blend_path = out_dir / "scene.blend"
        bpy.ops.wm.save_mainfile(filepath=str(blend_path))
        logger.info(f"Saved debug scene -> {blend_path} "
                    f"(frames {sel_frames[0]}..{sel_frames[-1]}, no render)")
        return

    # --- render all selected frames ---
    _render_animated_scene(frame_indices=sel_frames, render_dir=render_dir)
    logger.info(f"Rendered {len(sel_frames)} frames to {render_dir}")

    if args.fps > 0:
        video_path = out_dir / "hand_object_mesh.mp4"
        create_video(render_dir, video_path, fps=args.fps)
        logger.info(f"Saved video -> {video_path}")


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
    parser.add_argument("--image_resolution", type=int, nargs=2, default=None,
                        help="Render WxH. If unset, defaults to the source RGB image size "
                             "(K is scaled accordingly when set).")
    parser.add_argument("--number_of_samples", type=int, default=200)
    parser.add_argument("--focal_length", type=float, default=35.0,
                        help="Blender camera focal length (mm equivalent). "
                             "Ignored when intrinsics K from meta/*.pkl is used (the default).")
    parser.add_argument("--light_angle", type=str, default="(0, 244, 0)",
                        help="Sun light Euler angles in degrees, format: \"(rx, ry, rz)\". "
                             "Default tuned in cam_cond world (see debug7 scene).")
    parser.add_argument("--light_strength", type=float, default=0.75,
                        help="Sun light strength (Emission node Strength).")
    parser.add_argument("--light_size", type=float, default=0.3,
                        help="Sun light angular size in radians (controls shadow softness).")
    parser.add_argument("--ambient_color", type=float, nargs=3, default=[0.1, 0.1, 0.1],
                        help="RGB ambient (world) light color, 0..1 each.")
    parser.add_argument("--obj_mesh_RGB", type=float, nargs=3,
                        default=[144.0 / 255, 210.0 / 255, 236.0 / 255])
    parser.add_argument("--hand_mesh_RGB", type=float, nargs=3,
                        default=[200.0 / 255, 180.0 / 255, 220.0 / 255])
    parser.add_argument("--debug", action="store_true",
                        help="Build the animated scene then save it as <out_dir>/scene.blend "
                             "and exit without rendering. Open in Blender GUI to scrub the "
                             "timeline / tune lights.")
    parser.add_argument("--frame_list", type=int, nargs="+", default=None,
                        help="Optional explicit list of frame indices, e.g. --frame_list 290 295 300. "
                             "If omitted, iterates over all frames in image_info[\"frame_indices\"].")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Lower bound (inclusive) on frame index. Combine with --debug "
                             "and --end_frame to bake a small subset into scene.blend.")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="Upper bound (inclusive) on frame index. Typical debug usage: "
                             "--debug --end_frame 30 to bake only the first 31 frames.")
    parser.add_argument("--plane_y", type=float, default=None,
                        help="Floor plane Y in world (= OpenCV cam_cond coords). +Y is 'down'. "
                             "If unset, auto-set to (scene's max world Y) + --plane_margin so "
                             "the plane sits just below the visible content.")
    parser.add_argument("--plane_distance", type=float, default=None,
                        help="Floor plane Z (cam +Z = forward / in front of camera). If unset, "
                             "auto-set to the mean world Z of the scene.")
    parser.add_argument("--plane_margin", type=float, default=0.05,
                        help="Margin below the scene used when --plane_y is auto-computed.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
