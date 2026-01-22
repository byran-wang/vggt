import numpy as np

import json
import vggt.utils.eval_modules as eval_m
import vggt.utils.gt as gt
import os
from pathlib import Path
from viewer.viewer_step import ObjDataProvider
import torch
device = "cuda:0"



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="")
    parser.add_argument("--hand_fit_mode", type=str, default="intrinsic", help="choices: intrinsic, trans, rot") 
    
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _build_pose_4x4(pose):
    pose = np.array(pose)
    if pose.shape == (4, 4):
        return pose
    out = np.eye(4)
    out[:3] = pose
    return out

def _get_model_pts(model_pts, idx):
    if model_pts.ndim == 3:
        return model_pts[min(idx, model_pts.shape[0] - 1)]
    return model_pts

def to_homo(pts):
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)

def main():
    from tqdm import tqdm
    args = parse_args()
    import vggt.utils.ours as ours

    data_pred = ours.load_data(args)

    obj_provider = ObjDataProvider(Path(args.result_folder))
    seq_name = data_pred["full_seq_name"]
    data_gt = gt.load_data(seq_name, obj_provider.get_image_fids)        
    
    pred_o2c = data_pred.get("extrinsics")
    gt_o2c = data_gt.get("o2c")
    model_pts = data_gt.get("v3d_can.object")
    is_valid = data_gt.get("is_valid") * data_pred.get("is_valid")

    if pred_o2c is None or gt_o2c is None or model_pts is None:
        print("[WARN][eval_add_object] missing extrinsics/gt poses/model points; skipping ADD.")
        return

    pred_o2c = _to_numpy(pred_o2c)
    gt_o2c = _to_numpy(gt_o2c)
    model_pts = _to_numpy(model_pts)
    valid_flags = _to_numpy(is_valid) if is_valid is not None else None

    # Align predicted sequence to GT using first valid frame (match BundleSDF)
    pred_poses = [_build_pose_4x4(p) for p in pred_o2c]
    gt_poses = [np.array(g) for g in gt_o2c]
    n = min(len(gt_poses), len(pred_poses))
    if n > 0:
        align_tf = np.linalg.inv(pred_poses[0]) @ gt_poses[0]
        pred_poses = [p @ align_tf for p in pred_poses]

    # Init rerun
    import rerun as rr
    rr.init("eval_add_object", spawn=True)

    # Get image paths and intrinsic from data_gt
    fnames = data_gt.get("fnames")
    K = _to_numpy(data_gt.get("K"))

    for i in range(n):
        if valid_flags is not None and not bool(valid_flags[i]):
            continue
        pred_pose = pred_poses[i]
        gt_pose = gt_poses[i]
        cur_model_pts = _get_model_pts(model_pts, i)

        # Transform model points to camera space using pred_pose and gt_pose
        pred_pts = (pred_pose @ to_homo(cur_model_pts).T).T[:, :3]
        gt_pts = (gt_pose @ to_homo(cur_model_pts).T).T[:, :3]

        # Visualize pred_pts and gt_pts, image, intrinsic in rerun
        rr.set_time_sequence("frame", i)
        rr.log("eval/pred_pts", rr.Points3D(pred_pts, colors=[0, 0, 255], radii=0.0005))
        rr.log("eval/gt_pts", rr.Points3D(gt_pts, colors=[0, 255, 0], radii=0.0005))

        # Log image if available
        import cv2

        if fnames is not None and i < len(fnames):
            image = cv2.imread(fnames[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rr.log("eval/camera", rr.Image(image).compress(jpeg_quality=75), static=False)

        # Log camera with intrinsic (identity pose since points are in camera space)
        if K is not None:
            rr.log("eval/camera", rr.Pinhole(
                image_from_camera=K,
                image_plane_distance=0.5,
            ))

if __name__ == "__main__":
    main()
