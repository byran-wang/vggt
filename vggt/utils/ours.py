import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import torch
from common.viewer import ViewerData
from PIL import Image
from common.xdict import xdict
from pathlib import Path

from vggt.utils.eval_modules import compute_bounding_box_centers
import trimesh
from utils_simba.geometry import transform_points
from utils_simba.hand import initialize_mano_model
import smplx
from viewer.viewer_step import ObjDataProvider, HandDataProvider, _intrinsic_to_original






def load_data(args):
    device = "cuda:0"
    print("Loading data")

    obj_provider = ObjDataProvider(Path(args.result_folder))    
    last_step = obj_provider.steps[-1]
    data = last_step["data"]
    
    intrinsic =  data.get("intrinsics")      # square image.
    intrinsic = _intrinsic_to_original(intrinsic[0], data.get("original_coords"), 0) # origin image

    img_fs = obj_provider.origin_images
    seq_name = obj_provider.get_seq_name()    
    mesh_file = last_step["gen_3d_mesh_aligned"]


    hand_fit_mode = args.hand_fit_mode
    hand_provider = HandDataProvider(Path(Path(args.result_folder).parents[0]))
    hand_poses = hand_provider.get_hand_poses(hand_fit_mode)
    beta = hand_provider.get_hand_beta(hand_fit_mode)
    h2c_transls = hand_provider.get_hand_transls(hand_fit_mode)
    h2c_rots = hand_provider.get_hand_rots(hand_fit_mode)

    # Convert inputs to torch tensors on the target device for MANO
    hand_poses = torch.as_tensor(hand_poses, device=device, dtype=torch.float32)
    beta = torch.as_tensor(beta, device=device, dtype=torch.float32)
    betas = beta.unsqueeze(0).repeat(hand_poses.shape[0], 1)
    h2c_transls_np = np.asarray(h2c_transls)
    h2c_transls = torch.as_tensor(h2c_transls_np, device=device, dtype=torch.float32)
    h2c_rots = torch.as_tensor(h2c_rots, device=device, dtype=torch.float32)

    # Extract and transform right hand vertices
    mano_layer = smplx.create(
            model_path='./body_models/MANO_RIGHT.pkl', model_type="mano", use_pca=False, is_rhand=True
    )
    mano_layer.to(torch.device(device))

    hand_out_can = mano_layer(
        betas=betas,
        hand_pose=hand_poses,
        transl=torch.zeros_like(h2c_transls),
        global_orient=h2c_rots,
    )

    hand_v_c = hand_provider.get_hand_verts(hand_fit_mode)
    hand_jnts_can = hand_out_can.joints.cpu().numpy()
    # Build homogeneous transforms if only translation vectors are provided
    if h2c_transls_np.ndim == 2 and h2c_transls_np.shape[1] == 3:
        h2c_transforms = np.repeat(np.eye(4)[None], h2c_transls_np.shape[0], axis=0)
        h2c_transforms[:, :3, 3] = h2c_transls_np
    else:
        h2c_transforms = h2c_transls_np
    hand_jnts_c = transform_points(hand_jnts_can, h2c_transforms)
    obj_mesh_c = trimesh.load(mesh_file, process=False)
    obj_v_c = obj_mesh_c.vertices
    
    f3d_r = hand_provider.get_hand_faces(hand_fit_mode)

    out = xdict()
    out['verts.right'] = hand_v_c    # B, 778, 3
    out['jnts.right'] = hand_jnts_c  # B, 21, 3
    out['root.right'] = hand_jnts_c[:,0,:] # B, 3
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :] # B, 21, 3
    out['verts.object'] = obj_v_c[None]  # B, N, 3
    out['v3d_c.object'] = out['verts.object'] # B, N, 3
    out['root.object'] = compute_bounding_box_centers(out['verts.object']) # B, 3
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:] # B, N, 3
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :] # B, N, 3

    faces = {
        'object' : np.array(obj_mesh_c.faces),   # M,3          
        'right' : np.array(f3d_r), # 1538, 3

    }
    out["faces"] = faces
        
    print("Done loading data")
    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)

    out_dict = xdict()
    out_dict["fnames"] = img_fs#.tolist() # B

    out_dict["K"] = intrinsic[None] # 1, 3, 3
    out_dict["full_seq_name"] = seq_name    
    out_dict.merge(out)
    return out_dict

def load_viewer_data(args):
    data = load_data(args.ckpt_p)
    faces = xdict(data["faces"]).to_np()
    K = data["K"].numpy().reshape(3, 3)
    fnames = data["fnames"]

    color_dict = {"right": "gray_white", "left": "white", "object": "green-blue"}
    vis_dict = {}
    pred = data.search("v3d_c.").to_np()
    for v3d_key in pred.keys():
        node_id = v3d_key.split(".")[1]
        vis_dict[node_id] = {
            "v3d": pred[v3d_key],
            "f3d": faces[node_id],
            "vc": None,
            "name": node_id,
            "color": color_dict[node_id],
        }
    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )
    num_frames = len(fnames)
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(fnames[0])
    cols, rows = im.size

    images = [Image.open(im_p) for im_p in fnames]
    data = ViewerData(Rt, K, cols, rows, images)
    return meshes, data
