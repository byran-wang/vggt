from glob import glob
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import sys


from third_party.utils_simba.utils_simba.hand import initialize_mano_model
from third_party.utils_simba.utils_simba.depth import depth2xyzmap, get_depth
from third_party.utils_simba.utils_simba.mask import load_mask_bool
import cv2
import pickle
sys.path = [".."] + sys.path
from common.xdict import xdict
import trimesh
import os
from tqdm import tqdm


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle_compat(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()

def get_hand_pc(depth_ps, mask_hand_ps, meta, device="cuda"):
    hand_pc = []
    # show the progress bar with description
    for depth_p, mask_hand_p in tqdm(zip(depth_ps, mask_hand_ps), total=len(depth_ps), desc="Getting hand point map"):
        if not os.path.exists(depth_p) or not os.path.exists(mask_hand_p):
            print(f"Depth or mask hand not found: {depth_p} or {mask_hand_p}")
            continue
        if depth_p.split('/')[-1] != mask_hand_p.split('/')[-1]:
            print(f"Depth or mask hand not match: {depth_p} or {mask_hand_p}")
            continue
        
        depth = get_depth(depth_p)
        xyz_map = depth2xyzmap(torch.FloatTensor(depth).to(device), torch.FloatTensor(meta['K']).to(device))

        mask_hand = torch.from_numpy(load_mask_bool(mask_hand_p)).to(device)
        xyz_map = xyz_map[mask_hand]
        hand_pc.append(xyz_map)
    return hand_pc

def read_hand_data(data, args, num_total_frames):
    mydata = {}
    for k, v in data.items():
        mydata[k] = torch.FloatTensor(v)
        assert mydata[k].shape[0] == num_total_frames, f"Data length mismatch for {k}: expected {num_total_frames}, got {mydata[k].shape[0]}"
        mydata[k] = mydata[k][args.min_frame_num:args.max_frame_num:args.frame_interval]
    return mydata

def read_j2d_right_data(j2d_p, args, num_total_frames):
    j2d_data = np.load(j2d_p, allow_pickle=True).item()
    j2d_right = j2d_data['j2d.right']
    assert j2d_right.shape[0] == num_total_frames, f"j2d length mismatch: expected {num_total_frames}, got {j2d_right.shape[0]}"
    j2d_right = j2d_right[args.min_frame_num:args.max_frame_num:args.frame_interval]
    return j2d_right

def get_hand_mask(mask_hand_ps):
    mask_hands = []
    for i in range(len(mask_hand_ps)):
        mask_hand = torch.from_numpy(load_mask_bool(mask_hand_ps[i]))
        mask_hands.append(mask_hand)
    mask_hands = torch.stack(mask_hands, dim=0)    
    return mask_hands

def get_hand_depth(depth_ps, mask_hand_ps, meta, device="cuda"):

    hand_depth = []

    for depth_p, mask_hand_p in tqdm(zip(depth_ps, mask_hand_ps), total=len(depth_ps), desc="Getting hand depth"):
        if not (os.path.exists(depth_p) and os.path.exists(mask_hand_p)):
            print(f"Depth or mask hand not found: {depth_p} or {mask_hand_p}")
            continue

        if os.path.basename(depth_p) != os.path.basename(mask_hand_p):
            print(f"Depth or mask hand not match: {depth_p} or {mask_hand_p}")
            continue

        # Load depth once and compute in-place
        depth = get_depth(depth_p)
        depth = torch.from_numpy(depth.astype(np.float32)).to(device)

        # Load mask alpha channel directly and convert to boolean
        
        mask_hand = torch.from_numpy(load_mask_bool(mask_hand_p)).to(device)

        # Mask the depth in-place
        depth[~mask_hand] = -1

        hand_depth.append(depth)

    return torch.stack(hand_depth, dim=0)
    


def read_data(args):
    seq_name = args.seq_name
    # load data
    if args['dataset_type'] == "ho3d":
        data_dir = f"./data/train/{seq_name}"
    elif args['dataset_type'] == "zed":
        data_dir = args['out_dir']
    else:
        raise NotImplementedError(f"Dataset type {args['dataset_type']} not implemented.")

    im_ps = sorted(
        glob(f"{data_dir}/rgb/*.jpg")
        + glob(f"{data_dir}/rgb/*.png")
    )
    mask_obj_ps = sorted(glob(f"{data_dir}/mask_object/*.png"))
    mask_hand_ps = sorted(glob(f"{data_dir}/mask_hand/*.png"))
    depth_ps = sorted(glob(f"{data_dir}/depth/*.png"))
    
    assert len(im_ps) == len(mask_hand_ps) == len(mask_obj_ps) == len(depth_ps), "Number of images, hand masks, object masks, and depth maps must be equal."
    num_total_frames = len(im_ps)

    im_ps = im_ps[args.min_frame_num:args.max_frame_num:args.frame_interval]
    mask_hand_ps = mask_hand_ps[args.min_frame_num:args.max_frame_num:args.frame_interval]
    mask_obj_ps = mask_obj_ps[args.min_frame_num:args.max_frame_num:args.frame_interval]
    depth_ps = depth_ps[args.min_frame_num:args.max_frame_num:args.frame_interval]

    im0 = Path(im_ps[0])
    intrinsic_file = str(im0.parent.parent / "meta" / f"{im0.stem}.pkl")

    # Print image file names (not full paths) to quickly verify ordering
    im_file_names = [os.path.basename(p) for p in im_ps]
    print(f"Loaded images: {im_file_names}")
    

    meta = {}

    meta['K'] = np.array(_load_pickle_compat(intrinsic_file)['camMat'])
    meta['im_paths'] = im_ps
    meta['mask_obj_paths'] = mask_obj_ps
    meta['mask_hand_paths'] = mask_hand_ps
    meta['depth_paths'] = depth_ps
    # meta['object_cfg_f'] = args.object_cfg_f
    # meta['object_ckpt_f'] = args.object_ckpt_f
    # meta['object_mesh_f'] = args.object_mesh_f
    # o2w_all = torch.FloatTensor(np.load(f"{colmap_path}/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy"))
    # meta['o2c'] = o2w_all
    meta['im_H_W'] = cv2.imread(im_ps[0]).shape[:2] # H, W
    
    # data_o = {}
    # data_o['j2d.gt'] = torch.FloatTensor(
    #     np.load(f"./data/{seq_name}/colmap_2d/keypoints.npy")
    # )
    
    entities  = {}

    j2d_p = f"{data_dir}/hands/j2d.full.npy"
    data = np.load(
        f"{data_dir}/hands/hold_fit.slerp.npy", allow_pickle=True
    ).item()
    # get the hand mask

    if 'right' in data:
        data_r = read_hand_data(data['right'], args, num_total_frames)
        j2d_right = read_j2d_right_data(j2d_p, args, num_total_frames)
        right_valid_1 = (~np.isnan(j2d_right.reshape(-1, 21*2).mean(axis=1))).astype(np.float32) # num_frames
        right_valid = np.repeat(right_valid_1[:, np.newaxis], 21, axis=1)
        j2d_right_pad = torch.FloatTensor(np.concatenate([j2d_right, right_valid[:, :, None]], axis=2))
        mask_hands = get_hand_mask(meta['mask_hand_paths'])
        depth_gt = get_hand_depth(meta['depth_paths'], meta['mask_hand_paths'], meta)
        # assert and print error information if not equal
        assert depth_gt.shape[0] == j2d_right_pad.shape[0], f"depth_gt length: {depth_gt.shape[0]} != j2d_right_pad length: {j2d_right_pad.shape[0]}"
        data_r['j2d.gt'] = j2d_right_pad
        # data_r['v3d.gt'] = get_hand_pc(meta['depth_paths'], meta['mask_hand_paths'], meta)
        data_r['v3d.gt'] = None
        data_r['depth.gt'] = depth_gt
        data_r['valid'] = right_valid_1
        data_r['mask_hand.gt'] = mask_hands
        entities['right'] = data_r
    
    if 'left' in data:
        data_l = read_hand_data(data['left'], num_frames=num_frames)        
        j2d_left = j2d_data['j2d.left'][:num_frames]
        left_valid = (~np.isnan(j2d_left.reshape(-1, 21*2).mean(axis=1))).astype(np.float32)
        left_valid = np.repeat(left_valid[:, np.newaxis], 21, axis=1)
        j2d_left_pad = torch.FloatTensor(np.concatenate([j2d_left, left_valid[:, :, None]], axis=2))
        
        data_l['j2d.gt'] = j2d_left_pad
        entities['left'] = data_l
    
    mydata = xdict()
    mydata['entities'] = entities
    mydata['meta'] = meta
    return mydata


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        return idx
