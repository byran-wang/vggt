import os
import torch
from glob import glob
from PIL import Image
import numpy as np
import trimesh
from tqdm import tqdm
import os.path as op
import cv2
from smplx import MANO
from pytorch3d.transforms import matrix_to_axis_angle

import sys

sys.path = [".."] + sys.path
sys.path = ["."] + sys.path
import common.transforms as tf
from src.building.build_utils import save_visualizations
from src.building.build_utils import normalize_cameras
from third_party.utils_simba.utils_simba.geometry import transform_points
device = "cuda:0"


mano_models = {
    'right': MANO(
"../code/body_models", is_rhand=True, flat_hand_mean=False, use_pca=False
).to(device),
    'left': MANO(
    "../code/body_models", is_rhand=False, flat_hand_mean=False, use_pca=False
).to(device)}



    
def get_out_dir(seq_name):
    out_dir = f"./data/{seq_name}/build"
    return out_dir

def copy_images(rgb_ps, mask_ps, out_dir):
    num_frames = len(rgb_ps)
    remap_old2new = {}

    for idx in tqdm(range(num_frames)):
        rgb_p = rgb_ps[idx]
        mask_p = mask_ps[idx]
        remap_old2new[rgb_p.split("/")[-1].split(".")[0]] = idx

        image = Image.open(rgb_p)
        mask = Image.open(mask_p)

        out_mask_p = op.join(out_dir, "mask", f"{idx:04}.png")
        os.makedirs(op.dirname(out_mask_p), exist_ok=True)
        mask.save(out_mask_p)

        out_image_p = op.join(out_dir, "image", f"{idx:04}.png")
        os.makedirs(op.dirname(out_image_p), exist_ok=True)
        image.save(out_image_p)
        
        
    corres_out_p = op.join(out_dir, "corres.txt")
    # write rgb_ps into corres.txt
    base_ps = [op.basename(p) for p in rgb_ps]
    with open(corres_out_p, "w") as f:
        for rgb_p in base_ps:
            f.write(rgb_p + "\n")
    
def convert_parameters(mano_fit, K, o2w_all, num_frames, T_hip):
    output_trans_r = []
    output_pose_r = []
    output_P = {}
    output_object_trans = []
    output_object_pose = []
    
    trans_list = []
    for idx in range(num_frames):
        # prepare right hand
        mano_poses = np.concatenate(
            (mano_fit["hand_rot"][idx], mano_fit["hand_pose"][idx]), axis=0
        )
        mano_trans_r = mano_fit["hand_transl"][idx]
        mano_rot_r = mano_poses[:3]
        _, mano_trans_r = tf.cv2gl_mano(np.copy(mano_rot_r), np.copy(mano_trans_r), T_hip)
        trans_list.append(mano_trans_r)
    normalize_shift = np.stack(trans_list).mean(axis=0)
    
    
    for idx in range(num_frames):
        # prepare right hand
        mano_poses = np.concatenate(
            (mano_fit["hand_rot"][idx], mano_fit["hand_pose"][idx]), axis=0
        )
        mano_trans_r = mano_fit["hand_transl"][idx]
        mano_rot_r = mano_poses[:3]
        mano_rot_r, mano_trans_r = tf.cv2gl_mano(mano_rot_r, mano_trans_r, T_hip)
        mano_poses[:3] = mano_rot_r
        
        obj_rot = o2w_all[idx, :3, :3].numpy()
        obj_trans = o2w_all[idx, :3, 3]

        Rt_o = np.eye(4)
        Rt_o[:3, :3] = obj_rot
        Rt_o[:3, 3] = obj_trans
        Rt_o[1:3] *= -1

        obj_rot = matrix_to_axis_angle(torch.FloatTensor(Rt_o[:3, :3])).numpy()
        obj_trans = Rt_o[:3, 3]

        ## CVPR shift
        normalize_shift = np.array(
            [-0.0085238, -0.01372686, 0.42570806]
        )  # average mano root across time
        # # import pdb; pdb.set_trace()
        trans_r = mano_trans_r + normalize_shift
        trans_obj_normalized = obj_trans + normalize_shift

        # static camera position
        target_extrinsic = np.eye(4)
        target_extrinsic[1:3] *= -1
        target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (
            target_extrinsic[:3, :3] @ normalize_shift
        )

        # view matrix
        K_pad = np.eye(4)
        K_pad[:3, :3] = K.numpy()
        P = K_pad @ target_extrinsic

        output_trans_r.append(trans_r)
        output_pose_r.append(mano_poses)
        
        output_object_trans.append(trans_obj_normalized)
        output_object_pose.append(obj_rot)
        output_P[f"cam_{idx}"] = P

    hand_poses_r = np.array(output_pose_r)
    hand_trans_r = np.array(output_trans_r)
    return output_P,output_object_trans,output_object_pose,hand_poses_r,hand_trans_r

def convert_parameters_object_center(mano_fit, K, o2w_all, num_frames, T_hip):
    output_trans_r = []
    output_pose_r = []
    output_P = {}
    output_object_trans = []
    output_object_pose = []
       
    for idx in range(num_frames):
        # prepare right hand
        mano_poses = np.concatenate(
            (mano_fit["hand_rot"][idx], mano_fit["hand_pose"][idx]), axis=0
        )
        mano_trans_r = mano_fit["hand_transl"][idx]
        mano_rot_r = mano_poses[:3]
        # mano_rot_r, mano_trans_r = tf.cv2gl_mano(mano_rot_r, mano_trans_r, T_hip)
        mano_poses[:3] = mano_rot_r
        
        obj_rot = o2w_all[idx, :3, :3].numpy()
        obj_trans = o2w_all[idx, :3, 3]

        Rt_o = np.eye(4)
        Rt_o[:3, :3] = obj_rot
        Rt_o[:3, 3] = obj_trans
        # Rt_o[1:3] *= -1
        obj_rot = matrix_to_axis_angle(torch.FloatTensor(Rt_o[:3, :3])).numpy()
        obj_trans = Rt_o[:3, 3]

        # # import pdb; pdb.set_trace()
        trans_r = mano_trans_r

        # view matrix
        K_pad = np.eye(4)
        K_pad[:3, :3] = K.numpy()   

        output_trans_r.append(trans_r)
        output_pose_r.append(mano_poses)
        
        output_object_trans.append(obj_trans)
        output_object_pose.append(obj_rot)
        output_P[f"cam_{idx}"] = K_pad

    hand_poses_r = np.array(output_pose_r)
    hand_trans_r = np.array(output_trans_r)
    return output_P,output_object_trans,output_object_pose,hand_poses_r,hand_trans_r

def process_seq_hand_center(seq_name, rebuild, scene_bounding_sphere, max_radius_ratio, no_vis, colmap_path="", data_path=""):
    
    out_dir = get_out_dir(seq_name)
    print("Starting new build hand center dataset")
    print(out_dir)
    if rebuild:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    if op.exists(out_dir):
        print("Output directory already exists, skipping; Use --rebuild to remove and rebuild")
        return
    if data_path == "":
        rgb_ps = sorted(glob(f"./data/{seq_name}/processed/images/*.png"))
        mask_ps = sorted(glob(f"./data/{seq_name}/processed/masks/*"))
    else:
        rgb_ps = sorted(glob(f"{data_path}/images/*.png"))
        mask_ps = sorted(glob(f"{data_path}/masks/*"))

    assert len(rgb_ps) == len(mask_ps)
    if colmap_path == "":
        normalize_mat = torch.FloatTensor(
            np.load(f"./data/{seq_name}/processed/colmap/normalization_mat.npy")
        )
        mesh = trimesh.load(
            f"./data/{seq_name}/processed/colmap/sparse_points_normalized.obj",
            process=False,
        )

    else:
        # normalize_mat = torch.FloatTensor(
        #     np.load(f"{colmap_path}/normalization_mat.npy")
        # )
        normalize_mat = torch.eye(4)
        mesh = trimesh.load(
            f"{colmap_path}/sfm_superpoint+superglue/mvs/sparse_points_normalized_aligned.ply",
            process=False,
        )
        
    if data_path == "":
        hold_fit = np.load(
            f"./data/{seq_name}/processed/hold_fit.aligned.npy", allow_pickle=True
        ).item()
    else:
        hold_fit = np.load(
            f"{data_path}/hold_fit.aligned.npy", allow_pickle=True
        ).item()
    
    obj_fit = hold_fit['object']
    K = torch.FloatTensor(obj_fit['K'])

    o2w_all = torch.FloatTensor(obj_fit["o2w_all"])
    obj_scale = float(obj_fit["obj_scale"])
    num_frames = len(mask_ps)
    denormalize_mat = normalize_mat.inverse()

    # prepare cano pts
    cano_pts = torch.FloatTensor(mesh.vertices)
    cano_pts = torch.cat((cano_pts, torch.ones(cano_pts.shape[0], 1)), dim=1)
    cano_pts = denormalize_mat @ cano_pts.T
    cano_pts = cano_pts[None, :, :].repeat(num_frames, 1, 1)
    cano_pts[:, :3, :] = cano_pts[:, :3, :] * obj_scale

    # cano -> pix
    pts_w = torch.bmm(o2w_all, cano_pts).permute(0, 2, 1)
    pts_w = pts_w[:, :, :3]
    pts_2d = tf.project2d_batch(K[None, :, :].repeat(num_frames, 1, 1), pts_w).numpy()
    
    copy_images(rgb_ps, mask_ps, out_dir)
    entities = {}
    
    for hand in ['right', 'left']:  
        if hand not in hold_fit:
            continue
        hand_fit = hold_fit[hand]
        if not no_vis:
            save_visualizations(rgb_ps, mask_ps, pts_2d, hand_fit, num_frames, out_dir, flag=hand)
        print("Normalizing")
        mano_shape = hand_fit["hand_beta"]
        T_hip = (
            mano_models[hand].get_T_hip(betas=torch.tensor(mano_shape)[None].float().to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        # output_P is camera project matrix
        output_P, output_object_trans, output_object_pose, hand_poses, hand_trans = convert_parameters(
            hand_fit, K, o2w_all, num_frames, T_hip)
        myhand = {}
        myhand["hand_poses"] = hand_poses
        myhand["hand_trans"] = hand_trans
        myhand["mean_shape"] = mano_shape
        entities[hand] = myhand
    
    object_poses = np.concatenate(
        (np.array(output_object_pose), np.array(output_object_trans)), axis=1
    )
    
    cameras = output_P
    cameras_normalized = normalize_cameras(
        cameras, scene_bounding_sphere, max_radius_ratio
    )

    obj = {}
    obj["obj_scale"] = obj_scale
    obj["pts.cano"] = np.array(mesh.vertices)
    obj["norm_mat"] = normalize_mat
    obj["object_poses"] = object_poses
    entities['object'] = obj
    
    out = {}
    out["seq_name"] = seq_name
    out["cameras"] = cameras_normalized
    out["scene_bounding_sphere"] = scene_bounding_sphere
    out["max_radius_ratio"] = max_radius_ratio
    out['entities'] = entities
    out['data_type'] = 'hand_center'

    out_f = op.join(out_dir, "data_hand_center.npy")
    np.save(out_f, out)
    print(f"Exported {out_f}")



def process_seq_object_center(seq_name, rebuild, scene_bounding_sphere, max_radius_ratio, no_vis, colmap_path="", data_path=""):
    
    out_dir = get_out_dir(seq_name)
    print("Starting new build for object center dataset")
    print(out_dir)

    if data_path == "":
        rgb_ps = sorted(glob(f"./data/{seq_name}/processed/images/*.png"))
        mask_ps = sorted(glob(f"./data/{seq_name}/processed/masks/*"))
    else:
        rgb_ps = sorted(glob(f"{data_path}/images/*.png"))
        mask_ps = sorted(glob(f"{data_path}/masks/*"))

    assert len(rgb_ps) == len(mask_ps)

    normalize_mat = torch.eye(4)
    mesh = trimesh.load(
        f"{colmap_path}/sfm_superpoint+superglue/mvs/sparse_points_normalized_aligned.ply",
        process=False,
    )
    
    o2w_all = np.load(f"{colmap_path}/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy")
    pts_o = mesh.vertices    
    pts_o = np.tile(pts_o, (o2w_all.shape[0], 1, 1))
    pts_w = torch.FloatTensor(transform_points(pts_o, o2w_all))
    o2w_all = torch.FloatTensor(o2w_all)
    

    hold_fit = np.load(
        f"{data_path}/hold_fit.aligned.npy", allow_pickle=True
    ).item()
    
    obj_fit = hold_fit['object']
    K = torch.FloatTensor(obj_fit['K'])

    obj_scale = float(obj_fit["obj_scale"])
    num_frames = len(mask_ps)
    
    pts_2d = tf.project2d_batch(K[None, :, :].repeat(num_frames, 1, 1), pts_w).numpy()
    
    entities = {}
    
    for hand in ['right', 'left']:  
        if hand not in hold_fit:
            continue
        hand_fit = hold_fit[hand]
        # if not no_vis:
        if 0:
            save_visualizations(rgb_ps, mask_ps, pts_2d, hand_fit, num_frames, out_dir, flag=hand)
        print("Normalizing")
        mano_shape = hand_fit["hand_beta"]
        T_hip = (
            mano_models[hand].get_T_hip(betas=torch.tensor(mano_shape)[None].float().to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        # output_P is camera project matrix
        output_P, output_object_trans, output_object_pose, hand_poses, hand_trans = convert_parameters_object_center(
            hand_fit, K, o2w_all, num_frames, T_hip)
        myhand = {}
        myhand["hand_poses"] = hand_poses
        myhand["hand_trans"] = hand_trans
        myhand["mean_shape"] = mano_shape
        entities[hand] = myhand
    
    object_poses = np.concatenate(
        (np.array(output_object_pose), np.array(output_object_trans)), axis=1
    )
    
    cameras = output_P

    obj = {}
    obj["obj_scale"] = obj_scale
    obj["pts.cano"] = np.array(mesh.vertices)
    obj["norm_mat"] = normalize_mat
    obj["object_poses"] = object_poses
    entities['object'] = obj
    
    out = {}
    out["seq_name"] = seq_name
    out["cameras"] = cameras
    out["scene_bounding_sphere"] = scene_bounding_sphere
    out["max_radius_ratio"] = max_radius_ratio
    out['entities'] = entities
    out['data_type'] = 'object_center'

    out_f = op.join(out_dir, "data_object_center.npy")
    np.save(out_f, out)
    print(f"Exported {out_f}")
    
def zip_seq(seq_name):
    import os
    import zipfile

    def zip_directory_with_exclusions(source_dir, zip_path, exclusions):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                # Filter out directories that should be excluded
                dirs[:] = [d for d in dirs if not any(excl in d for excl in exclusions)]
                for file in files:
                    if not any(excl in file for excl in exclusions):
                        file_path = os.path.join(root, file)
                        # Store the file relative path in the zip
                        zipf.write(file_path, os.path.relpath(file_path, start=source_dir))
    # Example usage
    directory_to_zip = f"./data/{seq_name}/"
    output_zip_file = f'./data/{seq_name}.zip'
    exclusion_keywords = ['processed', 'video.mp4', 'vis', 'images.zip', 'images']
    zip_directory_with_exclusions(directory_to_zip, output_zip_file, exclusion_keywords)
    print("Exported zipfile to", output_zip_file)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--colmap_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--no_zip", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max_radius_ratio", type=float, default=3.0)
    parser.add_argument("--scene_bounding_sphere", type=float, default=6.0)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument('--dataset_type', 
        choices=[
                "hand_center", # HOLD code
                "object_center",       
                ], 
        default=["object_center"],
        nargs='+',  # To accept multiple values in a list
        required=False  # This makes the argument mandatory
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset_type = args.dataset_type[0]
    process_seq_hand_center(args.seq_name, args.rebuild, args.scene_bounding_sphere, args.max_radius_ratio, args.no_vis, colmap_path=args.colmap_path, data_path=args.data_path)
    process_seq_object_center(args.seq_name, args.rebuild, args.scene_bounding_sphere, args.max_radius_ratio, args.no_vis, colmap_path=args.colmap_path, data_path=args.data_path)
    out_dir = get_out_dir(args.seq_name)
    if args.dataset_type == "hand_center":
        ln_f = "data_hand_center.npy"
        
    elif args.dataset_type == "object_center":
        ln_f = "data_object_center.npy"
    cmd = f"cd {out_dir} && rm data.npy || true && ln -s {ln_f} data.npy"
    os.system(cmd)
    print(cmd)

    if not args.no_zip:
        zip_seq(args.seq_name)


if __name__ == "__main__":
    main()
