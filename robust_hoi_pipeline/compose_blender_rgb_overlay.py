"""Alpha-composite Blender RGBA renders onto the source RGB images.

Inputs (per seq):
  --render_dir   directory of RGBA PNGs from pipeline_blender_rendering (e.g. <out>/renders)
  --rgb_dir      directory of source RGB jpgs (e.g. ~/data/rhoi_zed/01/<seq>/rgb)
  --out_dir      output directory for composed PNGs (created if missing)
  --out_video    optional output mp4 path (encoded with ffmpeg if available)

Frames are matched by filename stem (e.g. "0042.png" <-> "0042.jpg").

The render is placed onto the RGB at pixel offset (--x_offset, --y_offset). When the
render and RGB have the same size, the default (0, 0) gives a full-frame overlay.
The render is alpha-blended pixel-wise; a sub-rectangle outside the RGB canvas is
clipped silently.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def _alpha_composite(rgb_uint8, render_rgba_uint8, x_offset, y_offset):
    """Return rgb_uint8 with render_rgba_uint8 alpha-composited at (x_offset, y_offset).

    Both arrays are uint8 numpy. rgb is HxWx3, render is hxwx4. Out-of-canvas
    region of the render is clipped."""
    H, W, _ = rgb_uint8.shape
    h, w, _ = render_rgba_uint8.shape

    # Destination rect (clipped to canvas)
    x0 = max(x_offset, 0)
    y0 = max(y_offset, 0)
    x1 = min(x_offset + w, W)
    y1 = min(y_offset + h, H)
    if x1 <= x0 or y1 <= y0:
        return rgb_uint8  # render fully off-canvas

    # Source rect (mirrors the dest clip)
    sx0 = x0 - x_offset
    sy0 = y0 - y_offset
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    src = render_rgba_uint8[sy0:sy1, sx0:sx1]
    src_rgb = src[..., :3].astype(np.float32)
    alpha = (src[..., 3:4].astype(np.float32)) / 255.0

    out = rgb_uint8.copy()
    dst_rgb = out[y0:y1, x0:x1].astype(np.float32)
    blended = src_rgb * alpha + dst_rgb * (1.0 - alpha)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=str, required=True)
    parser.add_argument("--rgb_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory for composed PNGs.")
    parser.add_argument("--out_video", type=str, default=None,
                        help="Optional mp4 path; encoded via ffmpeg if available.")
    parser.add_argument("--x_offset", type=int, default=0,
                        help="Horizontal pixel offset of the render on the RGB canvas.")
    parser.add_argument("--y_offset", type=int, default=0,
                        help="Vertical pixel offset of the render on the RGB canvas.")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--rgb_ext", type=str, default="jpg",
                        help="RGB extension (jpg or png).")
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_paths = sorted(render_dir.glob("*.png"))
    if not render_paths:
        raise FileNotFoundError(f"No *.png under {render_dir}")

    n_done = 0
    n_skip = 0
    for render_path in render_paths:
        stem = render_path.stem
        rgb_path = rgb_dir / f"{stem}.{args.rgb_ext}"
        if not rgb_path.exists():
            print(f"[skip] no RGB for {stem} ({rgb_path})")
            n_skip += 1
            continue
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        render = np.asarray(Image.open(render_path).convert("RGBA"))
        composed = _alpha_composite(rgb, render, args.x_offset, args.y_offset)
        out_path = out_dir / f"{stem}.png"
        Image.fromarray(composed).save(out_path)
        n_done += 1
    print(f"composed {n_done} frames; skipped {n_skip}")

    if args.out_video and n_done > 0:
        if shutil.which("ffmpeg") is None:
            print("[warn] ffmpeg not on PATH; skipping mp4 encode")
            return
        out_video = Path(args.out_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-pattern_type", "glob",
            "-i", f"{out_dir}/*.png",
            "-c:v", "libx264", "-profile:v", "high", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_video),
        ]
        subprocess.run(cmd, check=True)
        print(f"saved {out_video}")


if __name__ == "__main__":
    main()
