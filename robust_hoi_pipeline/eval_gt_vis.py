import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

import nvdiffrast.torch as dr

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

import vggt.utils.gt as gt
from robust_hoi_pipeline.pipeline_utils import load_frame_list, load_preprocessed_frame
from third_party.utils_simba.utils_simba.render import make_mesh_tensors, nvdiffrast_render


def _to_numpy(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


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


def _overlay_normal(raw_img, normal_tensor, depth_tensor, alpha):
    normal = normal_tensor[0].detach().cpu().numpy()
    depth = depth_tensor[0].detach().cpu().numpy()
    mask = depth > 1e-6
    normal_vis = 1.0 - ((normal + 1.0) * 0.5).clip(0.0, 1.0)
    normal_vis = (normal_vis * 255.0).astype(np.uint8)

    out = raw_img.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * normal_vis[mask]
    return out.clip(0, 255).astype(np.uint8), normal_vis


def _create_video(frame_dir: Path, output_video: Path, fps: int):
    cmd = [
        "ffmpeg",
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


def _build_gt_payload(data_gt, render_hand=False):
    gt_o2c = _to_numpy(data_gt["o2c"]).astype(np.float32)
    gt_is_valid = _to_numpy(data_gt["is_valid"]).astype(bool)

    v3d_can_obj = _to_numpy(data_gt["v3d_can.object"])
    if v3d_can_obj.ndim == 3:
        v3d_can_obj = v3d_can_obj[0]
    obj_faces = _to_numpy(data_gt["faces.object"]).astype(np.int32)

    obj_colors = _to_numpy(data_gt.get("colors.object"))
    if obj_colors is not None:
        if obj_colors.ndim == 3:
            obj_colors = obj_colors[0]
        if obj_colors.shape[1] >= 3:
            obj_colors = obj_colors[:, :3].astype(np.uint8)
        else:
            obj_colors = None
    if obj_colors is None:
        obj_colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (v3d_can_obj.shape[0], 1))

    hand_verts_cam = None
    hand_faces = None
    if render_hand and "v3d_c.right" in data_gt and "faces.right" in data_gt:
        hand_verts_cam = _to_numpy(data_gt["v3d_c.right"]).astype(np.float32)
        hand_faces = _to_numpy(data_gt["faces.right"]).astype(np.int64)
        try:
            from common.body_models import seal_mano_mesh_np

            hand_verts_cam, hand_faces = seal_mano_mesh_np(hand_verts_cam, hand_faces, is_rhand=True)
            hand_faces = hand_faces.astype(np.int32)
        except Exception as exc:
            print(f"[warn] Failed to seal GT hand mesh, using unsealed hand mesh: {exc}")
            hand_faces = hand_faces.astype(np.int32)

    return {
        "o2c": gt_o2c,
        "is_valid": gt_is_valid,
        "obj_verts_can": v3d_can_obj.astype(np.float32),
        "obj_faces": obj_faces,
        "obj_colors": obj_colors,
        "hand_verts_cam": hand_verts_cam,
        "hand_faces": hand_faces,
    }


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    data_preprocess_dir = data_dir / "pipeline_preprocess"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast visualization")

    if args.rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    overlay_dir = out_dir / "gt_overlay_frames"
    normal_dir = out_dir / "gt_normal_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = _load_frame_indices(data_preprocess_dir)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    seq_name = data_dir.name

    def get_image_fids():
        return frame_indices

    data_gt = gt.load_data(seq_name, get_image_fids)
    gt_payload = _build_gt_payload(data_gt, render_hand=args.render_hand)

    glctx = dr.RasterizeCudaContext()
    identity_pose = torch.eye(4, dtype=torch.float32, device="cuda")[None]

    records = []
    saved_idx = 0
    for i, fid in enumerate(tqdm(frame_indices, desc="Rendering GT normals with nvdiffrast")):
        preprocess_data = load_preprocessed_frame(data_preprocess_dir, int(fid))
        image = preprocess_data.get("image")
        if image is None:
            print(f"[skip] frame {fid}: missing image")
            continue

        K = preprocess_data.get("intrinsics")
        if K is None:
            K = _to_numpy(data_gt["K"])
        K = _normalize_intrinsics(K)

        if i >= len(gt_payload["o2c"]) or i >= len(gt_payload["is_valid"]) or not bool(gt_payload["is_valid"][i]):
            print(f"[skip] frame {fid}: GT invalid")
            continue

        o2c = gt_payload["o2c"][i].astype(np.float32)
        obj_verts_cam = (o2c[:3, :3] @ gt_payload["obj_verts_can"].T).T + o2c[:3, 3]
        obj_faces = gt_payload["obj_faces"]
        obj_colors = gt_payload["obj_colors"]

        verts = obj_verts_cam
        faces = obj_faces
        colors = obj_colors

        if args.render_hand and gt_payload["hand_verts_cam"] is not None and i < len(gt_payload["hand_verts_cam"]):
            hand_verts_cam = gt_payload["hand_verts_cam"][i]
            # Skip dummy invalid vertices used in GT files.
            if np.isfinite(hand_verts_cam).all() and hand_verts_cam.min() > -500:
                hand_faces = gt_payload["hand_faces"]
                hand_colors = np.tile(np.array([[225, 186, 160]], dtype=np.uint8), (hand_verts_cam.shape[0], 1))
                verts = np.concatenate([obj_verts_cam, hand_verts_cam], axis=0)
                faces = np.concatenate([obj_faces, hand_faces + obj_verts_cam.shape[0]], axis=0)
                colors = np.concatenate([obj_colors, hand_colors], axis=0)

        frame_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        frame_mesh.visual.vertex_colors = colors
        mesh_tensors = make_mesh_tensors(frame_mesh, device="cuda")

        h, w = image.shape[:2]
        _, depth, normal = nvdiffrast_render(
            K=K,
            H=h,
            W=w,
            ob_in_cvcams=identity_pose,
            glctx=glctx,
            context="cuda",
            get_normal=True,
            mesh_tensors=mesh_tensors,
            output_size=(h, w),
            use_light=False,
            extra={},
        )

        overlay_img, normal_img = _overlay_normal(image, normal, depth, alpha=float(args.alpha))
        overlay_path = overlay_dir / f"{saved_idx:06d}.png"
        normal_path = normal_dir / f"{saved_idx:06d}.png"
        Image.fromarray(overlay_img).save(overlay_path)
        Image.fromarray(normal_img).save(normal_path)
        records.append(
            {
                "render_index": saved_idx,
                "frame_idx": int(fid),
                "overlay_path": overlay_path.name,
                "normal_path": normal_path.name,
            }
        )
        saved_idx += 1

    if saved_idx == 0:
        raise RuntimeError("No valid GT frames were rendered")

    with open(out_dir / "frame_map.json", "w") as f:
        json.dump(records, f, indent=2)

    if args.fps > 0:
        video_path = out_dir / "gt_overlay.mp4"
        _create_video(overlay_dir, video_path, fps=args.fps)
        print(f"Saved video to {video_path}")
    print(f"Saved {saved_idx} GT overlay frames to {overlay_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Sequence directory, e.g., HO3D_v3/train/SM2")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for GT visualization")
    parser.add_argument("--render_hand", action="store_true", default=True, help="Render GT right-hand mesh with the GT object mesh")
    parser.add_argument("--fps", type=int, default=6, help="Output video FPS; set <=0 to skip video")
    parser.add_argument("--alpha", type=float, default=0.8, help="Overlay weight for rendered normals")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of frames to render")
    parser.add_argument("--rebuild", action="store_true", help="Clear output directory before rendering")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
