from glob import glob
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import sys


from third_party.utils_simba.utils_simba.hand import initialize_mano_model
from third_party.utils_simba.utils_simba.depth import depth2xyzmap, load_filtered_depth
from third_party.utils_simba.utils_simba.mask import load_mask_bool
import cv2
import pickle
sys.path = [".."] + sys.path
from common.xdict import xdict
from robust_hoi_pipeline.pipeline_neus_init import _load_joint_opt_image_info
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


def read_hand_data(data, hand_indices):
    mydata = {}
    for k, v in data.items():
        v_tensor = torch.FloatTensor(v)
        mydata[k] = v_tensor[hand_indices]
    return mydata

def read_j2d_right_data(j2d_p, hand_indices):
    j2d_data = np.load(j2d_p, allow_pickle=True).item()
    j2d_right = j2d_data['j2d.right']
    j2d_right = j2d_right[hand_indices]
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
        depth = load_filtered_depth(depth_p)
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
    data_dir = f"{args['data_dir']}"

    im_ps = sorted(
        glob(f"{data_dir}/rgb/*.jpg")
        + glob(f"{data_dir}/rgb/*.png")
    )
    mask_obj_ps = [p.replace("/rgb/", "/mask_object/").replace(".jpg", ".png") for p in im_ps]
    mask_hand_ps = [p.replace("/rgb/", "/mask_hand/").replace(".jpg", ".png") for p in im_ps]
    depth_ps = [p.replace("/rgb/", "/depth/").replace(".jpg", ".png") for p in im_ps]

    for name, paths in [("mask_object", mask_obj_ps), ("mask_hand", mask_hand_ps), ("depth", depth_ps)]:
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"[WARNING] {len(missing)}/{len(paths)} {name} files missing, e.g.: {missing[0]}")


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
    hand_indices = [int(Path(p).stem) for p in im_ps]

    if 'right' in data:
        data_r = read_hand_data(data['right'], hand_indices)
        j2d_right = read_j2d_right_data(j2d_p, hand_indices)
        right_valid_1 = (~np.isnan(j2d_right.reshape(-1, 21*2).mean(axis=1))).astype(np.float32) # num_frames
        right_valid = np.repeat(right_valid_1[:, np.newaxis], 21, axis=1)
        j2d_right_pad = torch.FloatTensor(np.concatenate([j2d_right, right_valid[:, :, None]], axis=2))
        mask_hands = get_hand_mask(meta['mask_hand_paths'])
        depth_gt = get_hand_depth(meta['depth_paths'], meta['mask_hand_paths'], meta)
        # assert and print error information if not equal
        assert depth_gt.shape[0] == j2d_right_pad.shape[0], f"depth_gt length: {depth_gt.shape[0]} != j2d_right_pad length: {j2d_right_pad.shape[0]}"
        data_r['j2d.gt'] = j2d_right_pad
        data_r['v3d.gt'] = None
        data_r['depth.gt'] = depth_gt
        data_r['valid'] = right_valid_1
        data_r['mask_hand.gt'] = mask_hands
        entities['right'] = data_r
    
    
    mydata = xdict()
    mydata['entities'] = entities
    mydata['meta'] = meta
    return mydata

def read_data_after_object_reconstruction(args):
    seq_name = args.seq_name
    # load data
    data_dir = f"{args['data_dir']}/pipeline_preprocess"

    im_ps = sorted(
        glob(f"{data_dir}/rgb/*.jpg")
        + glob(f"{data_dir}/rgb/*.png")
    )
    mask_obj_ps = [p.replace("/rgb/", "/mask_obj/").replace(".png", ".png") for p in im_ps]
    mask_hand_ps = [p.replace("/rgb/", "/mask_hand/").replace(".png", ".png") for p in im_ps]
    depth_ps = [p.replace("/rgb/", "/depth_filtered/").replace(".png", ".png") for p in im_ps]

    for name, paths in [("mask_object", mask_obj_ps), ("mask_hand", mask_hand_ps), ("depth", depth_ps)]:
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"[WARNING] {len(missing)}/{len(paths)} {name} files missing, e.g.: {missing[0]}")

    im0 = Path(im_ps[0])
    intrinsic_file = str(im0.parent.parent / "meta" / f"{im0.stem}.pkl")

    # Print image file names (not full paths) to quickly verify ordering
    im_file_names = [os.path.basename(p) for p in im_ps]
    print(f"Loaded images: {im_file_names}")
    

    meta = {}
    
    meta['K'] = np.array(_load_pickle_compat(intrinsic_file)['intrinsics'])
    meta['im_paths'] = im_ps
    meta['mask_obj_paths'] = mask_obj_ps
    meta['mask_hand_paths'] = mask_hand_ps
    meta['depth_paths'] = depth_ps
    # load object latest checkpoint and mesh from pipeline_neus_global
    result_dir = Path(args['result_dir'])
    neus_training_dir = result_dir.parent / "pipeline_neus_global" / "neus_training"
    # find the latest subdirectory in neus_training
    subdirs = [d for d in neus_training_dir.iterdir() if d.is_dir()]
    assert subdirs, f"No subdirectory found in {neus_training_dir}"
    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    neus_training_dir = latest_subdir 
    ckpt_files = sorted((neus_training_dir / "ckpt").rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    mesh_files = sorted((neus_training_dir / "save").rglob("*.obj"), key=lambda p: p.stat().st_mtime)
    assert ckpt_files, f"No checkpoint found in {neus_training_dir / 'ckpt'}"
    assert mesh_files, f"No mesh found in {neus_training_dir / 'save'}"
    meta['object_ckpt_f'] = str(ckpt_files[-1])
    meta['object_mesh_f'] = str(mesh_files[-1])
    # load o2c from latest image_info (c2o -> o2c via inverse)
    image_info, _, _ = _load_joint_opt_image_info(result_dir)
    c2o = image_info["c2o"]  # (N, 4, 4)
    o2c_all = torch.FloatTensor(np.linalg.inv(c2o).astype(np.float32))
    assert len(o2c_all) == len(im_ps), f"Number of o2c ({len(o2c_all)}) must match number of images ({len(im_ps)})"
    # select only the frames matching hand_indices
    frame_indices = list(image_info["frame_indices"])
    frame_to_local = {fid: i for i, fid in enumerate(frame_indices)}
    hand_indices = [int(Path(p).stem) for p in im_ps]
    o2c_selected = o2c_all[[frame_to_local[idx] for idx in hand_indices]]
    meta['o2c'] = o2c_selected
    meta['im_H_W'] = cv2.imread(im_ps[0]).shape[:2] # H, W
    
    # data_o = {}
    # data_o['j2d.gt'] = torch.FloatTensor(
    #     np.load(f"./data/{seq_name}/colmap_2d/keypoints.npy")
    # )
    
    entities  = {}
    entities['object'] = xdict() # dummy object entity to be filled in ObjectParameters
    j2d_p = f"{data_dir}/../hands/j2d.full.npy"
    data = np.load(
        f"{data_dir}/../hands/hold_fit.slerp.npy", allow_pickle=True
    ).item()
    # get the hand mask
    hand_indices = [int(Path(p).stem) for p in im_ps]

    if 'right' in data:
        data_r = read_hand_data(data['right'], hand_indices)
        j2d_right = read_j2d_right_data(j2d_p, hand_indices)
        right_valid_1 = (~np.isnan(j2d_right.reshape(-1, 21*2).mean(axis=1))).astype(np.float32) # num_frames
        right_valid = np.repeat(right_valid_1[:, np.newaxis], 21, axis=1)
        j2d_right_pad = torch.FloatTensor(np.concatenate([j2d_right, right_valid[:, :, None]], axis=2))
        mask_hands = get_hand_mask(meta['mask_hand_paths'])
        depth_gt = get_hand_depth(meta['depth_paths'], meta['mask_hand_paths'], meta)
        # assert and print error information if not equal
        assert depth_gt.shape[0] == j2d_right_pad.shape[0], f"depth_gt length: {depth_gt.shape[0]} != j2d_right_pad length: {j2d_right_pad.shape[0]}"
        data_r['j2d.gt'] = j2d_right_pad
        data_r['v3d.gt'] = None
        data_r['depth.gt'] = depth_gt
        data_r['valid'] = right_valid_1
        data_r['mask_hand.gt'] = mask_hands
        entities['right'] = data_r
    

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
