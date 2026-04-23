import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt_eval_vis_nvdiffrast import (
    _mesh_vertex_colors,
    build_mesh_in_object_space,
    ensure_sealed_right_hand_mesh,
    get_sam3d_mesh_path,
    load_hand_mesh_for_frame,
    load_hand_mesh_from_hand_object_alignment,
    load_image_info,
)
from robust_hoi_pipeline.pipeline_utils import load_sam3d_transform
from utils_simba.eval_vis import load_mesh_as_trimesh
from utils_simba.logger import get_logger

logger = get_logger(__name__)


_HAND_RGB = np.array([200, 180, 220], dtype=np.uint8)


def _transform_verts(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    verts_h = np.hstack([verts, np.ones((len(verts), 1), dtype=np.float32)])
    return (T @ verts_h.T).T[:, :3].astype(np.float32)


def _hand_mesh_in_obj(hand_mesh_cam: dict, c2o: np.ndarray) -> trimesh.Trimesh:
    verts, faces = ensure_sealed_right_hand_mesh(
        hand_mesh_cam["vertices"], hand_mesh_cam["faces"]
    )
    verts_obj = _transform_verts(np.asarray(verts, dtype=np.float32), c2o)
    mesh = trimesh.Trimesh(vertices=verts_obj, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(_HAND_RGB[None], (verts_obj.shape[0], 1))
    return mesh


def _merge_meshes(object_mesh: trimesh.Trimesh, hand_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    obj_v = np.asarray(object_mesh.vertices, dtype=np.float32)
    obj_f = np.asarray(object_mesh.faces, dtype=np.int32)
    obj_c = _mesh_vertex_colors(object_mesh)

    hand_v = np.asarray(hand_mesh.vertices, dtype=np.float32)
    hand_f = np.asarray(hand_mesh.faces, dtype=np.int32)
    hand_c = np.tile(_HAND_RGB[None], (hand_v.shape[0], 1))

    verts = np.concatenate([obj_v, hand_v], axis=0)
    faces = np.concatenate([obj_f, hand_f + obj_v.shape[0]], axis=0)
    colors = np.concatenate([obj_c, hand_c], axis=0)

    merged = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    merged.visual.vertex_colors = colors
    return merged


def main(args):
    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neus_dir = results_dir / "neus_training"

    image_info = load_image_info(results_dir)
    sam3d_tf = load_sam3d_transform(sam3d_dir, args.cond_index)
    sam3d_to_cond_cam = sam3d_tf["sam3d_to_cond_cam"]
    scale = float(sam3d_tf["scale"])

    frame_indices = np.asarray(image_info["frame_indices"])
    frame_list = frame_indices.tolist()
    frame_idx = int(args.frame_index if args.frame_index >= 0 else args.cond_index)
    if frame_idx not in frame_list:
        raise ValueError(
            f"frame_index {frame_idx} not found in frame_indices (range {frame_list[0]}..{frame_list[-1]})"
        )
    local_idx = frame_list.index(frame_idx)

    register_flags = np.asarray(image_info["register"], dtype=bool)
    invalid_flags = np.asarray(image_info["invalid"], dtype=bool)
    if not (register_flags[local_idx] and not invalid_flags[local_idx]):
        logger.warning(f"frame {frame_idx} is not registered or marked invalid — exporting anyway")

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
    logger.info(f"Using object mesh ({args.mesh_type}): {mesh_path}")

    object_obj = load_mesh_as_trimesh(mesh_path)

    data_preprocess_dir = sam3d_dir.parent / "pipeline_preprocess"
    if args.hand_mode == "trans":
        hand_cam_data = load_hand_mesh_for_frame(data_preprocess_dir, frame_idx)
    elif args.hand_mode in ("h", "o", "ho"):
        hand_cam_data = load_hand_mesh_from_hand_object_alignment(
            results_dir, data_preprocess_dir, args.hand_mode, frame_idx
        )
    else:
        raise ValueError(f"Unsupported hand_mode: {args.hand_mode}")
    if hand_cam_data is None:
        raise RuntimeError(
            f"Failed to load hand mesh for frame {frame_idx} (mode={args.hand_mode})"
        )
    hand_obj = _hand_mesh_in_obj(hand_cam_data, c2o[local_idx])

    merged_obj = _merge_meshes(object_obj, hand_obj)

    hand_path = out_dir / f"hand_{frame_idx:04d}.obj"
    object_path = out_dir / f"object_{frame_idx:04d}.obj"
    merged_path = out_dir / f"hand_object_{frame_idx:04d}.obj"

    hand_obj.export(str(hand_path))
    object_obj.export(str(object_path))
    merged_obj.export(str(merged_path))

    logger.info(f"Saved hand mesh    -> {hand_path}")
    logger.info(f"Saved object mesh  -> {object_path}")
    logger.info(f"Saved merged mesh  -> {merged_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the hand mesh, object mesh, and merged hand+object mesh "
            "for a given frame (all in that frame's camera space)."
        )
    )
    parser.add_argument("--result_folder", type=str, required=True, help="Path to output/<seq>/pipeline_joint_opt")
    parser.add_argument("--SAM3D_dir", type=str, required=True, help="Path to <seq>/SAM3D_aligned_post_process")
    parser.add_argument("--cond_index", type=int, required=True, help="Condition frame index")
    parser.add_argument(
        "--frame_index",
        type=int,
        default=-1,
        help="Frame index to export meshes for (-1 means use --cond_index)",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for mesh files")
    parser.add_argument("--mesh_type", type=str, default="neus", choices=["sam3d", "neus"])
    parser.add_argument("--hand_mode", type=str, default="ho", choices=["trans", "h", "o", "ho"])
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
