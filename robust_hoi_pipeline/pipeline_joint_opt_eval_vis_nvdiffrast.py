
import argparse
import gzip
import json
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import trimesh
from PIL import Image

import nvdiffrast.torch as dr

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.frame_management import load_register_indices
from robust_hoi_pipeline.pipeline_utils import load_preprocessed_frame, load_sam3d_transform
from third_party.utils_simba.utils_simba.render import make_mesh_tensors, nvdiffrast_render




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


def _normalize_intrinsics(K):
    K = np.asarray(K, dtype=np.float32)
    if K.shape == (3, 3):
        return K
    if K.shape == (1, 3, 3):
        return K[0]
    if K.shape == (9,):
        return K.reshape(3, 3)
    if K.shape == (4,):
        fx, fy, cx, cy = K.tolist()
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported intrinsics shape: {K.shape}")


def load_mesh_as_trimesh(mesh_path: Path):
    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(transform)
            meshes.append(geom)
        if len(meshes) == 0:
            return None
        return trimesh.util.concatenate(meshes)
    return None


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
        raise FileNotFoundError(f"Per-frame image info not found in {results_dir / f'{last_register_idx:04d}'}")

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
    for p in candidates:
        if p.exists():
            return p
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
    """Return per-vertex RGB colors for mesh; fallback to a uniform color."""
    n = len(mesh.vertices)
    colors = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == n and vc.shape[1] >= 3:
            colors = vc[:, :3].astype(np.uint8)
    if colors is None:
        colors = np.tile(np.asarray(default_rgb, dtype=np.uint8)[None], (n, 1))
    return colors


def load_hand_mesh_for_frame(data_preprocess_dir: Path, frame_idx: int):
    """Load right-hand mesh for one frame from preprocessing output."""
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


def ensure_sealed_right_hand_mesh(verts: np.ndarray, faces: np.ndarray):
    """Seal MANO right-hand mesh if needed (close wrist opening)."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return verts, faces.astype(np.int32)

    # Already sealed MANO mesh contains wrist-center index 778.
    if verts.shape[0] >= 779 and int(faces.max()) >= 778:
        return verts, faces.astype(np.int32)

    # Canonical MANO right hand is 778 verts. Seal it explicitly.
    if verts.shape[0] == 778:
        try:
            from common.body_models import seal_mano_mesh_np

            sealed_verts, sealed_faces = seal_mano_mesh_np(verts[None], faces, is_rhand=True)
            return sealed_verts[0].astype(np.float32), sealed_faces.astype(np.int32)
        except Exception as exc:
            print(f"[warn] Failed to seal hand mesh for current frame: {exc}")

    return verts, faces.astype(np.int32)


def build_merged_object_hand_mesh(object_mesh: trimesh.Trimesh, hand_mesh_cam: dict, c2o_scaled: np.ndarray):
    """Build merged object+hand mesh in object space for one frame."""
    obj_verts = np.asarray(object_mesh.vertices, dtype=np.float32)
    obj_faces = np.asarray(object_mesh.faces, dtype=np.int32)
    obj_colors = _mesh_vertex_colors(object_mesh)

    hand_verts_cam, hand_faces = ensure_sealed_right_hand_mesh(
        hand_mesh_cam["vertices"], hand_mesh_cam["faces"]
    )
    # Hand meshes are stored in camera space. Convert to object space for merging.
    hand_verts_obj = (c2o_scaled[:3, :3] @ hand_verts_cam.T).T + c2o_scaled[:3, 3]
    hand_verts_obj = hand_verts_obj.astype(np.float32)

    merged_verts = np.concatenate([obj_verts, hand_verts_obj], axis=0)
    merged_faces = np.concatenate([obj_faces, hand_faces + obj_verts.shape[0]], axis=0)
    hand_colors = np.tile(np.array([[225, 186, 160]], dtype=np.uint8), (hand_verts_obj.shape[0], 1))
    merged_colors = np.concatenate([obj_colors, hand_colors], axis=0)

    merged_mesh = trimesh.Trimesh(vertices=merged_verts, faces=merged_faces, process=False)
    merged_mesh.visual.vertex_colors = merged_colors
    return merged_mesh


def overlay_normal(raw_img, normal_tensor, depth_tensor, alpha):
    normal = normal_tensor[0].detach().cpu().numpy()  # (H, W, 3), [-1, 1]
    depth = depth_tensor[0].detach().cpu().numpy()  # (H, W)
    mask = depth > 1e-6

    normal_vis = 1 - ((normal + 1.0) * 0.5).clip(0.0, 1.0)
    normal_vis = (normal_vis * 255.0).astype(np.uint8)

    out = raw_img.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * normal_vis[mask]
    return out.clip(0, 255).astype(np.uint8), normal_vis


def create_video(frame_dir: Path, output_video: Path, fps: int):
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_video),
    ]
    print(f"Running command:\n{shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast visualization")

    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    output_dir = Path(args.out_dir)
    frame_dir = output_dir / "nvdiffrast_overlay_frames"
    normal_dir = output_dir / "nvdiffrast_normal_frames"
    video_path = output_dir / "nvdiffrast_overlay.mp4"

    if args.rebuild and output_dir.exists():
        shutil.rmtree(output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

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
    c2o_scaled = c2o.copy()
    c2o_scaled[:, :3, 3] *= scale
    o2c_all = np.linalg.inv(c2o_scaled)

    mesh_path = get_sam3d_mesh_path(sam3d_dir, args.cond_index)
    print(f"Using mesh: {mesh_path}")
    mesh_obj = build_mesh_in_object_space(
        mesh_path=mesh_path,
        frame_indices=frame_indices,
        c2o=c2o,
        scale=scale,
        sam3d_to_cond_cam=sam3d_to_cond_cam,
        cond_index=args.cond_index,
    )
    mesh_tensors = make_mesh_tensors(mesh_obj, device="cuda")
    glctx = dr.RasterizeCudaContext()

    data_preprocess_dir = sam3d_dir.parent / "pipeline_preprocess"
    records = []
    saved_idx = 0

    for local_idx, frame_idx in tqdm(enumerate(frame_indices), total=len(frame_indices), desc="Rendering frames with nvdiffrast"):
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

        K = _normalize_intrinsics(K)
        H, W = image.shape[:2]
        o2c = torch.as_tensor(o2c_all[local_idx][None], dtype=torch.float32, device="cuda")
        mesh_tensors_frame = mesh_tensors
        if args.render_hand:
            hand_mesh_cam = load_hand_mesh_for_frame(data_preprocess_dir, int(frame_idx))
            if hand_mesh_cam is not None:
                merged_mesh = build_merged_object_hand_mesh(mesh_obj, hand_mesh_cam, c2o_scaled[local_idx])
                mesh_tensors_frame = make_mesh_tensors(merged_mesh, device="cuda")

        _, depth, normal = nvdiffrast_render(
            K=K,
            H=H,
            W=W,
            ob_in_cvcams=o2c,
            glctx=glctx,
            context="cuda",
            get_normal=True,
            mesh_tensors=mesh_tensors_frame,
            output_size=(H, W),
            use_light=False,
            extra={},
        )

        overlay_img, normal_img = overlay_normal(image, normal, depth, alpha=float(args.alpha))
        overlay_path = frame_dir / f"{saved_idx:06d}.png"
        normal_path = normal_dir / f"{saved_idx:06d}.png"
        Image.fromarray(overlay_img).save(overlay_path)
        Image.fromarray(normal_img).save(normal_path)
        records.append(
            {
                "render_index": saved_idx,
                "frame_idx": int(frame_idx),
                "overlay_path": str(overlay_path.name),
                "normal_path": str(normal_path.name),
            }
        )
        saved_idx += 1

    if saved_idx == 0:
        raise RuntimeError("No valid frames were rendered")

    with open(output_dir / "frame_map.json", "w") as f:
        json.dump(records, f, indent=2)

    create_video(frame_dir, video_path, fps=args.fps)
    print(f"Saved {saved_idx} overlay frames to {frame_dir}")
    print(f"Saved video to {video_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, required=True, help="Path to output/<seq>/pipeline_joint_opt")
    parser.add_argument("--SAM3D_dir", type=str, required=True, help="Path to <seq>/SAM3D_aligned_post_process")
    parser.add_argument("--cond_index", type=int, required=True, help="Condition frame index")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for visualization")
    parser.add_argument("--fps", type=int, default=6, help="Output video FPS")
    parser.add_argument("--alpha", type=float, default=0.8, help="Overlay weight for rendered normals")
    parser.add_argument("--vis_gt", type=int, default=1, help="Use GT-valid filtering (1) or not (0)")
    parser.add_argument(
        "--render_hand",
        action="store_true",
        default=True,
        help="Render sealed right-hand mesh together with the object mesh",
    )
    parser.add_argument("--rebuild", action="store_true", help="Clear previous visualization outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
