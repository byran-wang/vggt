import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_preprocessed_frame, load_sam3d_transform
from third_party.utils_simba.utils_simba.eval_vis import (
    ensure_cuda_available,
    load_mesh_as_trimesh,
    render_frames_with_nvdiffrast,
)
from third_party.utils_simba.utils_simba.render import make_mesh_tensors


def _load_gt_valid_flags(seq_name: str, frame_indices: np.ndarray):
    import vggt.utils.gt as gt

    frame_ids = [int(x) for x in np.asarray(frame_indices).tolist()]

    def get_image_fids():
        return frame_ids

    data_gt = gt.load_data(seq_name, get_image_fids)
    gt_is_valid = data_gt["is_valid"]
    if torch.is_tensor(gt_is_valid):
        gt_is_valid = gt_is_valid.detach().cpu().numpy()
    gt_is_valid = np.asarray(gt_is_valid).astype(bool)
    if len(gt_is_valid) != len(frame_ids):
        raise RuntimeError(
            f"GT validity length mismatch: {len(gt_is_valid)} vs {len(frame_ids)}"
        )
    return gt_is_valid


def load_image_info(results_dir: Path):
    register_indices = load_register_indices(results_dir)
    if len(register_indices) == 0:
        raise RuntimeError(f"No register_order found in {results_dir}")
    last_register_idx = register_indices[-1]

    gz_path = results_dir / f"{last_register_idx:04d}" / "image_info.pkl.gz"
    npy_path = results_dir / f"{last_register_idx:04d}" / "image_info.npy"
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as f:
            image_info = pickle.load(f)
    elif npy_path.exists():
        image_info = np.load(npy_path, allow_pickle=True).item()
    else:
        raise FileNotFoundError(
            f"Per-frame image info not found in {results_dir / f'{last_register_idx:04d}'}"
        )

    shared_gz = results_dir / "shared_info.pkl.gz"
    shared_npy = results_dir / "shared_info.npy"
    if shared_gz.exists():
        with gzip.open(shared_gz, "rb") as f:
            shared = pickle.load(f)
        shared.update(image_info)
        image_info = shared
    elif shared_npy.exists():
        shared = np.load(shared_npy, allow_pickle=True).item()
        shared.update(image_info)
        image_info = shared

    return image_info


def get_sam3d_mesh_path(sam3d_dir: Path, cond_index: int):
    candidates = [
        sam3d_dir.parent / "SAM3D" / f"{cond_index:04d}" / "scene.glb",
        sam3d_dir / f"{cond_index:04d}" / "mesh.obj",
        sam3d_dir / f"{cond_index:04d}" / "scene.glb",
        sam3d_dir / f"{cond_index:04d}" / "mesh_aligned.obj",
    ]
    for mesh_path in candidates:
        if mesh_path.exists():
            return mesh_path
    raise FileNotFoundError(f"No SAM3D mesh found for condition frame {cond_index}")


def build_mesh_in_object_space(mesh_path: Path, frame_indices, c2o, scale, sam3d_to_cond_cam, cond_index):
    mesh = load_mesh_as_trimesh(mesh_path)
    if mesh is None:
        raise RuntimeError(f"Failed to load mesh geometry from {mesh_path}")

    frame_list = frame_indices.tolist()
    if cond_index not in frame_list:
        raise ValueError(f"Condition index {cond_index} not found in frame indices")
    cond_local = frame_list.index(cond_index)

    c2o_cond_scaled = c2o[cond_local].copy()
    c2o_cond_scaled[:3, 3] *= scale
    sam3d_to_obj = c2o_cond_scaled @ sam3d_to_cond_cam

    verts_sam3d = np.asarray(mesh.vertices, dtype=np.float32)
    verts_h = np.hstack([verts_sam3d, np.ones((len(verts_sam3d), 1), dtype=np.float32)])
    verts_obj = (sam3d_to_obj @ verts_h.T).T[:, :3]

    mesh_obj = trimesh.Trimesh(vertices=verts_obj, faces=np.asarray(mesh.faces), process=False)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        mesh_obj.visual.vertex_colors = np.asarray(mesh.visual.vertex_colors)
    return mesh_obj


def _mesh_vertex_colors(mesh: trimesh.Trimesh, default_rgb=(190, 190, 190)):
    num_vertices = len(mesh.vertices)
    colors = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vertex_colors = np.asarray(mesh.visual.vertex_colors)
        if vertex_colors.ndim == 2 and vertex_colors.shape[0] == num_vertices and vertex_colors.shape[1] >= 3:
            colors = vertex_colors[:, :3].astype(np.uint8)
    if colors is None:
        colors = np.tile(np.asarray(default_rgb, dtype=np.uint8)[None], (num_vertices, 1))
    return colors


def load_hand_mesh_for_frame(data_preprocess_dir: Path, frame_idx: int):
    hand_mesh_path = data_preprocess_dir / "hand" / f"{frame_idx:04d}_right.obj"
    if not hand_mesh_path.exists():
        return None
    try:
        hand_mesh = trimesh.load(str(hand_mesh_path), process=False, force="mesh")
    except Exception:
        return None

    verts = np.asarray(hand_mesh.vertices, dtype=np.float32)
    faces = np.asarray(hand_mesh.faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return None
    return {"vertices": verts, "faces": faces}

_hand_alignment_cache = {}

def load_hand_mesh_from_hand_object_alignment(results_dir: Path, data_preprocess_dir: Path, hand_mode: str, frame_idx: int):
    align_dir = results_dir.parent / "align_hand_object"
    npy_path = align_dir / f"hold_fit.aligned_{hand_mode}.npy"
    cache_key = str(npy_path)

    if cache_key not in _hand_alignment_cache:
        if not npy_path.exists():
            print(f"[warn] alignment file not found: {npy_path}")
            _hand_alignment_cache[cache_key] = None
        else:
            data = np.load(str(npy_path), allow_pickle=True).item()
            if "right" not in data:
                print(f"[warn] no 'right' hand in alignment data: {npy_path}")
                _hand_alignment_cache[cache_key] = None
            else:
                hand_data = data["right"]
                # Build frame_id -> local_idx mapping
                from glob import glob as _glob
                rgb_dir = data_preprocess_dir / "rgb"
                if rgb_dir.exists():
                    im_ps = sorted(_glob(str(rgb_dir / "*.jpg")) + _glob(str(rgb_dir / "*.png")))
                    frame_id_to_local = {int(Path(p).stem): i for i, p in enumerate(im_ps)}
                else:
                    frame_id_to_local = None
                _hand_alignment_cache[cache_key] = {
                    "v3d_cam": hand_data["v3d_cam"],
                    "faces": np.asarray(hand_data["f3d"], dtype=np.int32),
                    "frame_id_to_local": frame_id_to_local,
                }

    cached = _hand_alignment_cache[cache_key]
    if cached is None:
        return None

    v3d_cam = cached["v3d_cam"]
    faces = cached["faces"]
    frame_id_to_local = cached["frame_id_to_local"]

    if frame_id_to_local is not None:
        if frame_idx not in frame_id_to_local:
            print(f"[warn] frame_idx {frame_idx} not in alignment frames")
            return None
        local_idx = frame_id_to_local[frame_idx]
    else:
        local_idx = frame_idx

    if local_idx >= len(v3d_cam):
        print(f"[warn] frame_idx {frame_idx} (local {local_idx}) out of range (max {len(v3d_cam) - 1})")
        return None
    verts = np.asarray(v3d_cam[local_idx], dtype=np.float32)
    if 0:
        debug_dir = align_dir / f"debug_mesh"
        debug_dir.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
            str(debug_dir / f"{frame_idx:04d}_{hand_mode}.obj")
        )

    return {"vertices": verts, "faces": faces}


def ensure_sealed_right_hand_mesh(verts: np.ndarray, faces: np.ndarray):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return verts, faces.astype(np.int32)

    if verts.shape[0] >= 779 and int(faces.max()) >= 778:
        return verts, faces.astype(np.int32)

    if verts.shape[0] == 778:
        try:
            from common.body_models import seal_mano_mesh_np

            sealed_verts, sealed_faces = seal_mano_mesh_np(verts[None], faces, is_rhand=True)
            return sealed_verts[0].astype(np.float32), sealed_faces.astype(np.int32)
        except Exception as exc:
            print(f"[warn] Failed to seal hand mesh for current frame: {exc}")

    return verts, faces.astype(np.int32)


def build_merged_object_hand_mesh(object_mesh: trimesh.Trimesh, hand_mesh_cam: dict, c2o_with_scale: np.ndarray):
    obj_verts = np.asarray(object_mesh.vertices, dtype=np.float32)
    obj_faces = np.asarray(object_mesh.faces, dtype=np.int32)
    obj_colors = _mesh_vertex_colors(object_mesh)

    hand_verts_cam, hand_faces = ensure_sealed_right_hand_mesh(
        hand_mesh_cam["vertices"], hand_mesh_cam["faces"]
    )
    hand_verts_obj = (c2o_with_scale[:3, :3] @ hand_verts_cam.T).T + c2o_with_scale[:3, 3]
    hand_verts_obj = hand_verts_obj.astype(np.float32)

    merged_verts = np.concatenate([obj_verts, hand_verts_obj], axis=0)
    merged_faces = np.concatenate([obj_faces, hand_faces + obj_verts.shape[0]], axis=0)
    hand_colors = np.tile(np.array([[225, 186, 160]], dtype=np.uint8), (hand_verts_obj.shape[0], 1))
    merged_colors = np.concatenate([obj_colors, hand_colors], axis=0)

    merged_mesh = trimesh.Trimesh(vertices=merged_verts, faces=merged_faces, process=False)
    merged_mesh.visual.vertex_colors = merged_colors
    return merged_mesh


def _build_render_frames(
    *,
    frame_indices,
    valid_flags,
    gt_valid_flags,
    data_preprocess_dir: Path,
    results_dir: Path,
    o2c_with_scale,
    mesh_obj: trimesh.Trimesh,
    c2o_with_scale,
    render_hand: bool,
    hand_mode: str,
):
    frames = []
    for local_idx, frame_idx in enumerate(frame_indices):
        if not bool(valid_flags[local_idx]):
            if gt_valid_flags is not None:
                print(f"[skip] frame {frame_idx}: invalid for evaluation according to GT")
            else:
                print(f"[skip] frame {frame_idx}: unregistered/invalid")
            continue

        preprocess_data = load_preprocessed_frame(data_preprocess_dir, int(frame_idx))
        image = preprocess_data.get("image")
        K = preprocess_data.get("intrinsics")
        if image is None or K is None:
            print(f"[skip] frame {frame_idx}: missing image or intrinsics")
            continue

        frame = {
            "frame_idx": int(frame_idx),
            "image": image,
            "K": K,
            "pose_o2c": o2c_with_scale[local_idx],
        }
        if render_hand:
            if hand_mode == "trans":
                hand_mesh_cam = load_hand_mesh_for_frame(data_preprocess_dir, int(frame_idx))
            elif hand_mode == "h" or hand_mode=="o" or hand_mode=="ho" or hand_mode=="all":
                hand_mesh_cam = load_hand_mesh_from_hand_object_alignment(results_dir, data_preprocess_dir, hand_mode, int(frame_idx))
            else:
                raise ValueError(f"Unsupported hand_mode: {hand_mode}")
            if hand_mesh_cam is not None:
                merged_mesh = build_merged_object_hand_mesh(mesh_obj, hand_mesh_cam, c2o_with_scale[local_idx])
                frame["mesh_tensors"] = make_mesh_tensors(merged_mesh, device="cuda")
        frames.append(frame)
    return frames


def main(args):
    ensure_cuda_available()

    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    output_dir = Path(args.out_dir)
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
        print(
            f"Using GT-valid + registered frames: {int(valid_flags.sum())}/{len(valid_flags)} "
            f"(registered={int((register_flags & (~invalid_flags)).sum())}, gt_valid={int(gt_valid_flags.sum())})"
        )
    else:
        print(
            f"Using registered frames only: {int(valid_flags.sum())}/{len(valid_flags)} "
            f"(GT filtering disabled)"
        )

    c2o = np.asarray(image_info["c2o"], dtype=np.float64)
    c2o_with_scale = c2o.copy()
    c2o_with_scale[:, :3, :3] *= scale
    c2o_with_scale[:, :3, 3] *= scale
    o2c_with_scale = np.linalg.inv(c2o_with_scale)
    
    if args.mesh_type == "sam3d":
        mesh_path = get_sam3d_mesh_path(sam3d_dir, args.cond_index)
    elif args.mesh_type == "neus":
        obj_files = sorted(neus_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime)
        if not obj_files:
            raise FileNotFoundError(f"No .obj file found under {neus_dir}")
        mesh_path = obj_files[-1]
    else:
        raise ValueError(f"Unsupported mesh type: {args.mesh_type}")
    print(f"Using mesh: {mesh_path}")
    mesh_obj = build_mesh_in_object_space(
        mesh_path=mesh_path,
        frame_indices=frame_indices,
        c2o=c2o,
        scale=scale,
        sam3d_to_cond_cam=sam3d_to_cond_cam,
        cond_index=args.cond_index,
    )
    default_mesh_tensors = make_mesh_tensors(mesh_obj, device="cuda")

    data_preprocess_dir = sam3d_dir.parent / "pipeline_preprocess"
    frames = _build_render_frames(
        frame_indices=frame_indices,
        valid_flags=valid_flags,
        gt_valid_flags=gt_valid_flags,
        data_preprocess_dir=data_preprocess_dir,
        results_dir=results_dir,
        o2c_with_scale=o2c_with_scale,
        mesh_obj=mesh_obj,
        c2o_with_scale=c2o_with_scale,
        render_hand=bool(args.render_hand),
        hand_mode=args.hand_mode,
    )

    result = render_frames_with_nvdiffrast(
        frames=frames,
        out_dir=output_dir,
        alpha=float(args.alpha),
        fps=args.fps,
        default_mesh_tensors=default_mesh_tensors,
        desc="Rendering frames with nvdiffrast",
        vis_in_rerun=False,
    )

    print(f"Saved {len(result['records'])} overlay frames to {result['overlay_dir']}")
    if args.fps > 0:
        print(f"Saved video to {result['video_path']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, required=True, help="Path to output/<seq>/pipeline_joint_opt")
    parser.add_argument("--SAM3D_dir", type=str, required=True, help="Path to <seq>/SAM3D_aligned_post_process")
    parser.add_argument("--cond_index", type=int, required=True, help="Condition frame index")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for visualization")
    parser.add_argument("--fps", type=int, default=6, help="Output video FPS")
    parser.add_argument("--alpha", type=float, default=0.8, help="Overlay weight for rendered normals")
    parser.add_argument("--vis_gt", type=int, default=1, help="Use GT-valid filtering (1) or not (0)")
    parser.add_argument("--render_hand", dest="render_hand", action="store_true", default=True, help="Render sealed right-hand mesh together with the object mesh")
    parser.add_argument("--mesh_type", type=str, default="neus", choices=["sam3d", "neus"], help="Mesh source to use: 'neus' (default) or 'sam3d'")
    parser.add_argument("--hand_mode", type=str, default="trans", choices=["trans", "h","o", "ho", "all"], help="Hand fitting mode: trans, h, o, ho, ")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
