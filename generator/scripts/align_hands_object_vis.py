"""
Visualize predicted (optimized) and GT hand meshes from align_hands_object.py results.
Usage:
    cd generator && python scripts/align_hands_object_vis.py \
        --pred_path ./data/{seq}/hold_fit.aligned_h_all.npy \
        --seq_name {seq} \
        --show_images
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pickle
from glob import glob

import rerun as rr
import rerun.blueprint as rrb

sys.path = [".", "..", "../code"] + sys.path
from common.rerun_utils import compute_vertex_normals
from src.utils.io.gt import load_data as gt_load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize aligned hands and object results")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to predicted .npy file (e.g., out_dir/hold_fit.aligned_h_all.npy)")
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset_type", type=str, default="zed", choices=["zed", "ho3d"])
    parser.add_argument("--show_object", action="store_true")
    parser.add_argument("--show_images", action="store_true")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--min_frame_num", type=int, default=0)
    parser.add_argument("--frame_interval", type=int, default=1)
    return parser.parse_args()


def load_pred_data(pred_path):
    """Load optimized result .npy file."""
    return np.load(pred_path, allow_pickle=True).item()


def get_image_fids_from_rgb(data_dir, seq_name, dataset_type):
    """Extract frame indices from RGB image filenames."""
    if dataset_type == "zed":
        im_ps = sorted(glob(f"{data_dir}/{seq_name}/rgb/*.jpg"))
    else:
        im_ps = sorted(glob(f"{data_dir}/train/{seq_name}/rgb/*.jpg"))
    fids = []
    for p in im_ps:
        fname = os.path.basename(p).split(".")[0]
        fids.append(int(fname))
    return np.array(fids)


def load_gt_data(seq_name, data_dir, dataset_type):
    """Load GT hand/object data using gt.load_data with image fids from RGB files."""
    image_fids = get_image_fids_from_rgb(data_dir, seq_name, dataset_type)

    def get_image_fids(full_seq_name):
        return image_fids.tolist()

    data_gt = gt_load_data(seq_name, get_image_fids)
    return data_gt


def load_metadata(seq_name, data_dir, dataset_type, num_frames, min_frame_num, frame_interval):
    """Load image paths and camera intrinsics."""
    if dataset_type == "zed":
        im_ps = sorted(glob(f"{data_dir}/{seq_name}/rgb/*.jpg"))[:num_frames]
        im_ps = im_ps[min_frame_num::frame_interval]
        intrinsic_file = sorted(glob(f"{data_dir}/{seq_name}/meta/*.pkl"))[0]
    else:
        im_ps = sorted(glob(f"{data_dir}/train/{seq_name}/rgb/*.jpg"))
        im_ps = im_ps[min_frame_num:num_frames:frame_interval]
        intrinsic_file = sorted(glob(f"{data_dir}/train/{seq_name}/meta/*.pkl"))[0]

    with open(intrinsic_file, "rb") as f:
        K = np.array(pickle.load(f)["camMat"])

    return {"im_paths": im_ps, "K": K}


def visualize(pred_data, gt_data, metadata, args):
    """Main visualization loop."""
    # Color constants (RGBA as 0-255 for albedo_factor)
    pred_blue = [76, 128, 255, 230]
    gt_green = [0, 255, 0, 204]
    pred_left_color = [143, 237, 255, 230]
    gt_left_color = [143, 237, 143, 204]
    obj_color = [5, 144, 201, 128]

    # Determine frame count from pred data
    num_frames = None
    for side in ["right", "left"]:
        if side in pred_data and "v3d_cam" in pred_data[side]:
            num_frames = pred_data[side]["v3d_cam"].shape[0]
            break
    if num_frames is None:
        print("No hand data found in pred_path")
        return

    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    # GT data fields
    gt_v3d_h = None
    gt_faces_h = None
    gt_v3d_o = None
    gt_faces_o = None
    gt_colors_o = None
    if gt_data is not None:
        if "v3d_c.right" in gt_data:
            gt_v3d_h = gt_data["v3d_c.right"].numpy() if hasattr(gt_data["v3d_c.right"], 'numpy') else np.array(gt_data["v3d_c.right"])
            gt_faces_h = gt_data["faces.right"].numpy() if hasattr(gt_data["faces.right"], 'numpy') else np.array(gt_data["faces.right"])
        if "v3d_c.object" in gt_data:
            gt_v3d_o = gt_data["v3d_c.object"].numpy() if hasattr(gt_data["v3d_c.object"], 'numpy') else np.array(gt_data["v3d_c.object"])
            gt_faces_o = gt_data["faces.object"].numpy() if hasattr(gt_data["faces.object"], 'numpy') else np.array(gt_data["faces.object"])
        if "colors.object" in gt_data:
            gt_colors_o = gt_data["colors.object"].numpy().astype(np.uint8) if hasattr(gt_data["colors.object"], 'numpy') else np.array(gt_data["colors.object"]).astype(np.uint8)

    # Blueprint
    blueprint = rrb.Vertical(
        rrb.Spatial3DView(
            name="3D View",
            origin="/",
        ),
        rrb.Horizontal(
            rrb.Spatial2DView(name="image", origin="/image"),
        ) if args.show_images else rrb.Spatial3DView(name="3D View 2", origin="/"),
        row_shares=[5, 2] if args.show_images else [1],
    )
    rr.init("align_hands_vis", spawn=True)
    rr.send_blueprint(blueprint)

    K = metadata.get("K")
    im_paths = metadata.get("im_paths", [])

    for frame_id in range(num_frames):
        rr.set_time("frame_id", sequence=frame_id)

        # --- Predicted hands ---
        for side, color in [("right", pred_blue), ("left", pred_left_color)]:
            if side in pred_data and "v3d_cam" in pred_data[side]:
                v = pred_data[side]["v3d_cam"][frame_id].astype(np.float32)
                f = pred_data[side]["f3d"].astype(np.int32)
                normals = compute_vertex_normals(v, f)
                rr.log(
                    f"pred/{side}_hand",
                    rr.Mesh3D(
                        vertex_positions=v,
                        triangle_indices=f,
                        vertex_normals=normals,
                        albedo_factor=color,
                    ),
                )

        # --- GT hand ---
        if gt_v3d_h is not None and frame_id < gt_v3d_h.shape[0]:
            v = gt_v3d_h[frame_id].astype(np.float32)
            f = gt_faces_h.astype(np.int32)
            normals = compute_vertex_normals(v, f)
            rr.log(
                "gt/right_hand",
                rr.Mesh3D(
                    vertex_positions=v,
                    triangle_indices=f,
                    vertex_normals=normals,
                    albedo_factor=gt_green,
                ),
            )

        # --- GT object ---
        if args.show_object and gt_v3d_o is not None and frame_id < gt_v3d_o.shape[0]:
            obj_v = gt_v3d_o[frame_id].astype(np.float32)
            obj_f = gt_faces_o.astype(np.int32)
            if gt_colors_o is not None:
                rr.log(
                    "gt/object",
                    rr.Mesh3D(
                        vertex_positions=obj_v,
                        triangle_indices=obj_f,
                        vertex_colors=gt_colors_o,
                    ),
                )
            else:
                rr.log(
                    "gt/object",
                    rr.Mesh3D(
                        vertex_positions=obj_v,
                        triangle_indices=obj_f,
                        albedo_factor=obj_color,
                    ),
                )

        # --- Predicted object ---
        if args.show_object and "object" in pred_data:
            obj = pred_data["object"]
            if "v3d_cam" in obj:
                obj_v = obj["v3d_cam"][frame_id].astype(np.float32)
                if "f3d" in obj and obj["f3d"].ndim == 2:
                    obj_f = obj["f3d"].astype(np.int32)
                    obj_normals = compute_vertex_normals(obj_v, obj_f)
                    rr.log(
                        "pred/object",
                        rr.Mesh3D(
                            vertex_positions=obj_v,
                            triangle_indices=obj_f,
                            vertex_normals=obj_normals,
                            albedo_factor=obj_color,
                        ),
                    )
                else:
                    rr.log("pred/object", rr.Points3D(obj_v, radii=0.001))

        # --- Image ---
        if args.show_images and frame_id < len(im_paths) and K is not None:
            img_path = im_paths[frame_id]
            if os.path.exists(img_path):
                bgr = cv2.imread(img_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                rr.log(
                    "image",
                    rr.Pinhole(
                        resolution=[w, h],
                        focal_length=[float(K[0, 0]), float(K[1, 1])],
                        principal_point=[float(K[0, 2]), float(K[1, 2])],
                    ),
                )
                rr.log("image", rr.Image(rgb.astype(np.uint8)))


def main():
    args = parse_args()

    print(f"Loading predicted data from {args.pred_path}")
    pred_data = load_pred_data(args.pred_path)

    # Determine max frame num
    max_frame_num = args.max_frames if args.max_frames > 0 else None
    if max_frame_num is None:
        for side in ["right", "left"]:
            if side in pred_data and "v3d_cam" in pred_data[side]:
                max_frame_num = pred_data[side]["v3d_cam"].shape[0]
                break

    # Load GT data via gt.load_data with image fids from RGB files
    gt_data = None
    try:
        print("Loading GT data...")
        gt_data = load_gt_data(args.seq_name, args.data_dir, args.dataset_type)
    except Exception as e:
        print(f"Could not load GT data: {e}")

    # Load metadata (images, intrinsics)
    metadata = {}
    if args.show_images:
        metadata = load_metadata(
            args.seq_name, args.data_dir, args.dataset_type,
            max_frame_num, args.min_frame_num, args.frame_interval,
        )

    visualize(pred_data, gt_data, metadata, args)


if __name__ == "__main__":
    main()
