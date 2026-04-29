import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _list_frames(frame_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    frames = [p for p in sorted(frame_dir.iterdir()) if p.suffix in exts]
    return frames


def _record_key(record):
    if "frame_idx" in record:
        return ("frame_idx", int(record["frame_idx"]))
    if "frame_id" in record:
        return ("frame_idx", int(record["frame_id"]))
    if "render_index" in record:
        return ("frame_idx", int(record["render_index"]))
    return None


def _load_stream(name: str, base_dir: Path, frame_dir: Path):
    frame_map_path = base_dir / "frame_map.json"
    frames = _list_frames(frame_dir)
    stream = {
        "name": name,
        "base_dir": base_dir,
        "frame_dir": frame_dir,
        "frames": frames,
        "count": len(frames),
    }

    if not frame_map_path.exists():
        return stream

    with open(frame_map_path, "r") as f:
        records = json.load(f)

    key_to_frame = {}
    ordered_keys = []
    for record in records:
        key = _record_key(record)
        overlay_rel = record.get("overlay_path")
        if key is None or not overlay_rel:
            continue
        frame_path = frame_dir / overlay_rel
        if not frame_path.exists():
            continue
        if key not in key_to_frame:
            ordered_keys.append(key)
        key_to_frame[key] = frame_path

    if key_to_frame:
        stream["key_to_frame"] = key_to_frame
        stream["ordered_keys"] = ordered_keys
        stream["count"] = len(key_to_frame)
    return stream


def _resolve_frame_dir(base_dir: Path, candidates):
    for rel in candidates:
        p = base_dir / rel
        if p.exists() and p.is_dir():
            return p
    return None


def _resize_to_height(img: np.ndarray, h: int):
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * (h / float(img.shape[0]))))
    w = max(2, w)
    if w % 2 == 1:
        w += 1
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _stream_label(name: str):
    labels = {
        "foundation": "SAM3D + FoundationPose",
        "bundle_sdf": "BundleSDF",
        "hold": "HOLD",
        "joint_opt": "Ours",
        "gt": "Ground Truth",
    }
    return labels.get(name, name)


def _annotate_panel(img: np.ndarray, label: str):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(out.shape[1], out.shape[0]) / 900.0)
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad_x = max(10, int(round(font_scale * 14)))
    pad_y = max(8, int(round(font_scale * 10)))
    x0 = pad_x
    y0 = pad_y
    x1 = min(out.shape[1] - pad_x, x0 + text_w + pad_x * 2)
    y1 = min(out.shape[0] - pad_y, y0 + text_h + baseline + pad_y * 2)

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0.0, dst=out)
    cv2.putText(
        out,
        label,
        (x0 + pad_x, y1 - pad_y - baseline),
        font,
        font_scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
    return out


def _create_video(frame_dir: Path, out_video: Path, fps: int):
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
        str(out_video),
    ]
    print(f"Running command:\n{shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def _resolve_streams(streams):
    if not streams:
        raise RuntimeError("No valid frame directories found; cannot merge.")

    if all("key_to_frame" in stream for stream in streams):
        common_keys = set(streams[0]["ordered_keys"])
        for stream in streams[1:]:
            common_keys &= set(stream["ordered_keys"])
        ordered_keys = [key for key in streams[0]["ordered_keys"] if key in common_keys]
        if not ordered_keys:
            raise RuntimeError(
                "No shared frames found across frame_map.json files: "
                + ", ".join(f"{stream['name']}={stream['count']}" for stream in streams)
            )
        return ordered_keys, True

    n = min(stream["count"] for stream in streams)
    if n == 0:
        raise RuntimeError(
            "Empty input frames: " + ", ".join(f"{stream['name']}={stream['count']}" for stream in streams)
        )
    return list(range(n)), False


def _load_stream_frame(stream, frame_ref, use_frame_map: bool):
    if use_frame_map:
        frame_path = stream["key_to_frame"][frame_ref]
    else:
        frame_path = stream["frames"][frame_ref]
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    return img


def main(args):
    foundation_dir = Path(args.foundation_dir)
    bundle_sdf_dir = Path(args.bundle_sdf_dir) if args.bundle_sdf_dir is not None else None
    hold_dir = Path(args.hold_dir) if args.hold_dir is not None else None
    joint_opt_dir = Path(args.joint_opt_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    seq_name = out_dir.name

    foundation_frames_dir = _resolve_frame_dir(
        foundation_dir, ["nvdiffrast_overlay_frames", "overlay_frames"]
    )
    bundle_sdf_frames_dir = None
    if bundle_sdf_dir is not None:
        bundle_sdf_frames_dir = _resolve_frame_dir(
            bundle_sdf_dir, ["nvdiffrast_overlay_frames", "overlay_frames"]
        )
    hold_frames_dir = None
    if hold_dir is not None:
        hold_frames_dir = _resolve_frame_dir(
            hold_dir, ["nvdiffrast_overlay_frames", "overlay_frames"]
        )
    joint_opt_frames_dir = _resolve_frame_dir(
        joint_opt_dir, ["nvdiffrast_overlay_frames", "overlay_frames"]
    )
    gt_frames_dir = _resolve_frame_dir(
        gt_dir, ["gt_overlay_frames", "nvdiffrast_overlay_frames", "overlay_frames"]
    )

    if foundation_frames_dir is None:
        print(f"WARNING: No overlay frame dir found under {foundation_dir}, skipping.")
    if bundle_sdf_dir is not None and bundle_sdf_frames_dir is None:
        print(f"WARNING: No overlay frame dir found under {bundle_sdf_dir}, skipping.")
    if hold_dir is not None and hold_frames_dir is None:
        print(f"WARNING: No overlay frame dir found under {hold_dir}, skipping.")
    if joint_opt_frames_dir is None:
        print(f"WARNING: No overlay frame dir found under {joint_opt_dir}, skipping.")
    if gt_frames_dir is None:
        print(f"WARNING: No overlay frame dir found under {gt_dir}, skipping.")

    streams = []
    if foundation_frames_dir is not None:
        streams.append(_load_stream("foundation", foundation_dir, foundation_frames_dir))
    if bundle_sdf_frames_dir is not None:
        streams.append(_load_stream("bundle_sdf", bundle_sdf_dir, bundle_sdf_frames_dir))
    if hold_frames_dir is not None:
        streams.append(_load_stream("hold", hold_dir, hold_frames_dir))
    if joint_opt_frames_dir is not None:
        streams.append(_load_stream("joint_opt", joint_opt_dir, joint_opt_frames_dir))
    if gt_frames_dir is not None:
        streams.append(_load_stream("gt", gt_dir, gt_frames_dir))

    frame_refs, use_frame_map = _resolve_streams(streams)

    if args.rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    merge_frames_dir = out_dir / "eval_sum_vis_frames"
    merge_frames_dir.mkdir(parents=True, exist_ok=True)

    merge_mode = "frame_map alignment" if use_frame_map else "render-order alignment"
    stream_summary = ", ".join(f"{stream['name']}={stream['count']}" for stream in streams)
    print(f"Merging {len(frame_refs)} frames using {merge_mode} ({stream_summary})")

    for i, frame_ref in enumerate(tqdm(frame_refs, desc="Merging eval videos")):
        imgs = [_load_stream_frame(stream, frame_ref, use_frame_map) for stream in streams]

        target_h = min(im.shape[0] for im in imgs)
        imgs = [_resize_to_height(im, target_h) for im in imgs]
        if args.vis_method_name:
            imgs = [_annotate_panel(im, _stream_label(stream["name"])) for im, stream in zip(imgs, streams)]

        sep = np.full((target_h, args.line_width, 3), args.line_gray, dtype=np.uint8)
        parts = []
        for idx, im in enumerate(imgs):
            if idx > 0:
                parts.append(sep)
            parts.append(im)
        merged = np.concatenate(parts, axis=1)

        out_p = merge_frames_dir / f"{i:06d}.png"
        cv2.imwrite(str(out_p), merged)
    
    out_video = out_dir / f"../eval_sum_{seq_name}.mp4"
    if args.fps > 0:
        _create_video(merge_frames_dir, out_video, args.fps)
        print(f"Saved merged video: {out_video}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation_dir", type=str, required=True, help="FoundationPose eval vis output directory")
    parser.add_argument("--bundle_sdf_dir", type=str, default=None, help="BundleSDF eval vis output directory")
    parser.add_argument("--hold_dir", type=str, default=None, help="HOLD eval vis output directory")
    parser.add_argument("--joint_opt_dir", type=str, required=True, help="Joint-opt eval vis output directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT eval vis output directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for merged visualization")
    parser.add_argument("--fps", type=int, default=6, help="Output video fps; set <=0 to skip video")
    parser.add_argument("--line_width", type=int, default=8, help="Separator line width in pixels")
    parser.add_argument("--line_gray", type=int, default=160, help="Separator gray value [0,255]")
    parser.add_argument("--vis_method_name", dest="vis_method_name", action="store_true", help="Overlay method names on each merged panel")
    parser.add_argument("--no_vis_method_name", dest="vis_method_name", action="store_false", help="Do not overlay method names on merged panels")
    parser.set_defaults(vis_method_name=True)
    parser.add_argument("--rebuild", action="store_true", help="Clear output directory before writing")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
