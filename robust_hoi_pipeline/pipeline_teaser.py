"""Teaser image generator.

Renders the object + hand normals for each selected frame on a transparent
background, then composites them all into a single RGBA PNG: earlier frames
are more transparent, the last frame is fully opaque. Per-frame RGBA renders
are also saved alongside the merged teaser image.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

import nvdiffrast.torch as dr

from robust_hoi_pipeline.pipeline_joint_opt_eval_vis_nvdiffrast import (
    build_mesh_in_object_space,
    build_merged_object_hand_mesh,
    get_sam3d_mesh_path,
    load_hand_mesh_for_frame,
    load_hand_mesh_from_hand_object_alignment,
    load_image_info,
)
from robust_hoi_pipeline.pipeline_utils import load_preprocessed_frame, load_sam3d_transform
from third_party.utils_simba.utils_simba.eval_vis import (
    ensure_cuda_available,
    normalize_intrinsics,
)
from third_party.utils_simba.utils_simba.render import make_mesh_tensors, nvdiffrast_render


def _parse_frame_list_arg(frame_list_arg):
    """Parse --frame_list as a comma-separated list or a path to a text file."""
    if frame_list_arg is None:
        return None
    path = Path(frame_list_arg)
    if path.exists():
        with open(path, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
    else:
        tokens = [t.strip() for t in frame_list_arg.split(",") if t.strip()]
    return [int(t) for t in tokens]


def _select_render_frames(frame_indices, valid_flags, *, start, end, interval, frame_list):
    """Pick the subset of frame_indices to render, preserving original order."""
    all_valid = [int(fi) for fi, ok in zip(frame_indices, valid_flags) if bool(ok)]

    if frame_list is not None:
        wanted = set(int(x) for x in frame_list)
        return [fi for fi in all_valid if fi in wanted]

    end_val = end if end >= 0 else (max(all_valid) + 1 if all_valid else 0)
    return [fi for fi in all_valid if start <= fi < end_val and ((fi - start) % max(interval, 1) == 0)]


def _render_frame_rgba(frame, glctx, default_mesh_tensors):
    """Render one frame's mesh normals into an RGBA image (transparent bg)."""
    image = frame["image"]
    h, w = image.shape[:2]
    pose = torch.as_tensor(frame["pose_o2c"][None], dtype=torch.float32, device="cuda")
    mesh_tensors = frame.get("mesh_tensors", default_mesh_tensors)
    if mesh_tensors is None:
        raise RuntimeError("No mesh tensors provided for rendering")

    _, depth, normal = nvdiffrast_render(
        K=normalize_intrinsics(frame["K"]),
        H=h,
        W=w,
        ob_in_cvcams=pose,
        glctx=glctx,
        context="cuda",
        get_normal=True,
        mesh_tensors=mesh_tensors,
        output_size=(h, w),
        use_light=False,
        extra={},
    )

    normal_np = normal[0].detach().cpu().numpy()
    depth_np = depth[0].detach().cpu().numpy()
    mask = (depth_np > 1e-6).astype(np.float32)

    # normal [-1, 1] -> RGB [0, 255] (same mapping as overlay_normal)
    rgb = 1.0 - ((normal_np + 1.0) * 0.5).clip(0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)

    alpha = (mask * 255.0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return rgba


def _alpha_over(accum: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Composite ``layer`` on top of ``accum`` using the premul 'over' operator.

    Inputs and outputs are (H, W, 4) float32 arrays with RGB in [0, 255] and
    alpha in [0, 1].
    """
    a_src = layer[..., 3:4]
    a_dst = accum[..., 3:4]
    a_out = a_src + a_dst * (1.0 - a_src)
    rgb_out = layer[..., :3] * a_src + accum[..., :3] * a_dst * (1.0 - a_src)
    safe = np.where(a_out > 1e-6, a_out, 1.0)
    rgb_out = rgb_out / safe
    return np.concatenate([rgb_out, a_out], axis=-1)


def main(args):
    ensure_cuda_available()

    results_dir = Path(args.result_folder)
    sam3d_dir = Path(args.SAM3D_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    neus_dir = results_dir / "neus_training"

    # Load image info + SAM3D transform (for scale and sam3d→cond-cam mapping)
    image_info = load_image_info(results_dir)
    sam3d_tf = load_sam3d_transform(sam3d_dir, args.cond_index)
    sam3d_to_cond_cam = sam3d_tf["sam3d_to_cond_cam"]
    scale = float(sam3d_tf["scale"])

    frame_indices = np.asarray(image_info["frame_indices"])
    register_flags = np.asarray(image_info["register"], dtype=bool)
    invalid_flags = np.asarray(image_info["invalid"], dtype=bool)
    valid_flags = register_flags & (~invalid_flags)

    c2o = np.asarray(image_info["c2o"], dtype=np.float64)
    c2o_with_scale = c2o.copy()
    c2o_with_scale[:, :3, :3] *= scale
    c2o_with_scale[:, :3, 3] *= scale
    o2c_with_scale = np.linalg.inv(c2o_with_scale)

    # Load object mesh (SAM3D or NeuS) — same logic as eval_vis_nvdiffrast
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

    # Build render list (start/end/interval or explicit frame_list)
    render_frame_ids = _select_render_frames(
        frame_indices, valid_flags,
        start=args.start, end=args.end, interval=args.interval,
        frame_list=_parse_frame_list_arg(args.frame_list),
    )
    

    if not render_frame_ids:
        raise RuntimeError("No frames selected for rendering")
    print(f"Rendering {len(render_frame_ids)} frames: {render_frame_ids[:10]}{'...' if len(render_frame_ids) > 10 else ''}")

    data_preprocess_dir = sam3d_dir.parent / "pipeline_preprocess"
    frame_id_to_local = {int(fi): i for i, fi in enumerate(frame_indices)}

    # Build per-frame render payloads (merge hand into mesh if requested)
    frames = []
    for frame_id in render_frame_ids:
        local_idx = frame_id_to_local[int(frame_id)]
        preprocess_data = load_preprocessed_frame(data_preprocess_dir, int(frame_id))
        image = preprocess_data.get("image")
        K = preprocess_data.get("intrinsics")
        if image is None or K is None:
            print(f"[skip] frame {frame_id}: missing image or intrinsics")
            continue

        frame = {
            "frame_idx": int(frame_id),
            "image": image,
            "K": K,
            "pose_o2c": o2c_with_scale[local_idx],
        }
        if args.render_hand:
            if args.hand_mode == "trans":
                hand_mesh_cam = load_hand_mesh_for_frame(data_preprocess_dir, int(frame_id))
            elif args.hand_mode in ("h", "o", "ho"):
                hand_mesh_cam = load_hand_mesh_from_hand_object_alignment(
                    results_dir, data_preprocess_dir, args.hand_mode, int(frame_id),
                )
            else:
                raise ValueError(f"Unsupported hand_mode: {args.hand_mode}")
            if hand_mesh_cam is not None:
                merged = build_merged_object_hand_mesh(mesh_obj, hand_mesh_cam, c2o_with_scale[local_idx])
                frame["mesh_tensors"] = make_mesh_tensors(merged, device="cuda")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No frames were loadable for rendering")

    # Output dirs
    rgba_dir = output_dir / "teaser_rgba_frames"
    rgba_dir.mkdir(parents=True, exist_ok=True)

    # Render loop: each frame's alpha is scaled by a fade weight; earliest is
    # most transparent and the last frame is fully opaque. Composite all
    # rendered frames (earliest first) into a single RGBA canvas.
    glctx = dr.RasterizeCudaContext()
    records = []
    n = len(frames)
    merged = None  # (H, W, 4) float32 accumulator with alpha in [0, 1]

    for i, frame in enumerate(tqdm(frames, desc="Rendering teaser")):
        rgba = _render_frame_rgba(frame, glctx, default_mesh_tensors)
        fade = (i + 1) / n  # earlier → more transparent, last → 1.0

        # Per-frame RGBA (fade already baked into alpha)
        faded = rgba.copy()
        faded[..., 3] = (faded[..., 3].astype(np.float32) * fade).clip(0, 255).astype(np.uint8)
        Image.fromarray(faded, mode="RGBA").save(rgba_dir / f"{i:06d}.png")

        # Accumulate onto the merged canvas
        layer = np.empty((*rgba.shape[:2], 4), dtype=np.float32)
        layer[..., :3] = rgba[..., :3].astype(np.float32)
        layer[..., 3] = (rgba[..., 3].astype(np.float32) / 255.0) * fade
        if merged is None:
            merged = np.zeros_like(layer)
        elif merged.shape[:2] != layer.shape[:2]:
            raise RuntimeError(
                f"Frame {frame['frame_idx']} shape {layer.shape[:2]} does not match "
                f"earlier frames {merged.shape[:2]}; all frames must share resolution"
            )
        merged = _alpha_over(merged, layer)

        records.append({
            "render_index": i,
            "frame_idx": int(frame["frame_idx"]),
            "fade_alpha": float(fade),
            "rgba_path": f"{i:06d}.png",
        })

    with open(output_dir / "teaser_frame_map.json", "w") as f:
        json.dump(records, f, indent=2)

    # Write merged teaser image (transparent background)
    merged_rgb = np.clip(merged[..., :3], 0, 255).astype(np.uint8)
    merged_alpha = np.clip(merged[..., 3] * 255.0, 0, 255).astype(np.uint8)
    merged_img = np.dstack([merged_rgb, merged_alpha])
    teaser_path = output_dir / "teaser.png"
    Image.fromarray(merged_img, mode="RGBA").save(teaser_path)

    print(f"Saved {len(records)} RGBA frames to {rgba_dir}")
    print(f"Saved merged teaser image to {teaser_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, required=True,
                        help="Path to output/<seq>/pipeline_joint_opt")
    parser.add_argument("--SAM3D_dir", type=str, required=True,
                        help="Path to <seq>/SAM3D_aligned_post_process")
    parser.add_argument("--cond_index", type=int, required=True, help="Condition frame index")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for teaser")
    parser.add_argument("--mesh_type", type=str, default="neus", choices=["sam3d", "neus"])
    parser.add_argument("--render_hand", action="store_true", default=True)
    parser.add_argument("--hand_mode", type=str, default="ho", choices=["trans", "h", "o", "ho"])
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=-1000, help="End frame index (exclusive; -1 for all)")
    parser.add_argument("--interval", type=int, default=100, help="Frame sampling interval")
    parser.add_argument("--frame_list", type=str, default=None,
                        help="Comma-separated frame indices OR path to a text file of frame indices "
                             "(overrides start/end/interval when set)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
