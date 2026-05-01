import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_frame_or_black(cap, fallback_shape):
    ok, frame = cap.read()
    if ok:
        return frame, True
    return np.zeros(fallback_shape, dtype=np.uint8), False


def _fit_frame(frame, panel_w, panel_h):
    h, w = frame.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _draw_caption(frame, text):
    out = frame.copy()
    band_h = 44
    cv2.rectangle(out, (0, 0), (out.shape[1], band_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main(args):
    left_video_f = Path(args.left_video)
    right_video_f = Path(args.right_video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_f = out_dir / "merged_video.mp4"

    left_cap = cv2.VideoCapture(str(left_video_f))
    right_cap = cv2.VideoCapture(str(right_video_f))
    if not left_cap.isOpened():
        raise FileNotFoundError(f"Cannot open left video: {left_video_f}")
    if not right_cap.isOpened():
        raise FileNotFoundError(f"Cannot open right video: {right_video_f}")

    left_w = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_h = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    right_w = int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    right_h = int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frames = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = args.fps
    if fps is None:
        fps = left_cap.get(cv2.CAP_PROP_FPS) or right_cap.get(cv2.CAP_PROP_FPS) or 30
    fps = float(fps)

    panel_w = max(left_w, right_w)
    panel_h = max(left_h, right_h)
    out_w = panel_w * 2
    out_h = panel_h
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1

    writer = cv2.VideoWriter(
        str(out_video_f),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {out_video_f}")

    total_frames = min(left_frames, right_frames)
    if total_frames <= 0:
        total_frames = int(args.max_frames) if args.max_frames is not None else 10**12
    if args.max_frames is not None:
        total_frames = min(total_frames, int(args.max_frames))

    left_fallback = (left_h, left_w, 3)
    right_fallback = (right_h, right_w, 3)
    written = 0
    for _ in range(total_frames):
        left_frame, left_ok = _read_frame_or_black(left_cap, left_fallback)
        right_frame, right_ok = _read_frame_or_black(right_cap, right_fallback)
        if not left_ok or not right_ok:
            break

        left_panel = _draw_caption(_fit_frame(left_frame, panel_w, panel_h), "before")
        right_panel = _draw_caption(_fit_frame(right_frame, panel_w, panel_h), "after")
        merged = np.concatenate([left_panel, right_panel], axis=1)
        if merged.shape[1] != out_w or merged.shape[0] != out_h:
            merged = cv2.copyMakeBorder(
                merged,
                0,
                out_h - merged.shape[0],
                0,
                out_w - merged.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        writer.write(merged)
        written += 1

    left_cap.release()
    right_cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError("No frames were written to the merged video")

    print(f"Saved merged video to {out_video_f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge two videos side-by-side with before/after captions.")
    parser.add_argument("--left_video", type=str, required=True)
    parser.add_argument("--right_video", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
