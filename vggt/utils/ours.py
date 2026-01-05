import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import torch
from common.viewer import ViewerData
from PIL import Image
from common.xdict import xdict
from src.model.mano.server import MANOServer
from src.model.obj.server import ObjectServer
from src.utils.eval_modules import compute_bounding_box_centers
import trimesh
from utils_simba.geometry import get_revert_tvec, transform_points
from utils_simba.hand import initialize_mano_model
import smplx
from common.body_models import seal_mano_mesh
from utils_simba.geometry import get_intrinsic_matrix_from_projection






def load_data(args):
    device = "cuda:0"
    print("Loading data")
    
    # from src.utils.io.optim import load_data
    # ckpt_p='logs/hold_GPMF14_ho3d_pre_train/checkpoints/last.pose_ref'
    # ckpt_p='logs/' + seq_name.split('.')[0] + '_pre_train/checkpoints/last.pose_ref'
    # "data/hold_MC1_ho3d/processed//colmap_hold_MC1_ho3d.80/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy",
    # out, ckpt = load_data(ckpt_p, pose_p=f"{mvs_root}/o2w_normalized_aligned.npy", object_mesh_f=object_mesh_f)
    # breakpoint()
    result_dir = args.result_folder
    vis_p = f"{result_dir}/hold_fit.{args.hand_fit_prefix}.npy"
    data = np.load(vis_p, allow_pickle=True).item()
    intrinsic = data['object']["K"]        # Camera intrinsics
    img_fs = data['object']["im_paths"]

    ckpt = torch.load(ckpt_p, map_location='cpu')
    sd = xdict(ckpt["state_dict"])
    # param_dict = sd.search(".params.")
    hand_scale = np.array(list(sd.search(".hand_scale").values())[0]) 
    betas = sd['models.right.hand_beta'].clone().cuda()
    betas = torch.tile(betas, (len(img_fs), 1))
    h2c_rot = sd["models.right.hand_rot"].clone().cuda()
    h2c_transl = sd["models.right.hand_transl"].clone().cuda()
    hand_pose = sd["models.right.hand_pose"].clone().cuda()


    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)    
    
    # # K = data['object']["K"].to(device).view(1, 4, 4)[:, :3, :3]
    # scale = 1 / data['object']['obj_scale']
    # scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = trimesh.load(object_mesh_f, process=False)

    # ## log object ##
    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)

    # Extract object data
    # pts_w = data['object']["j3d"]          # Shape: (B, number_3d_points, 3)
    # o2w_all = data['object']["o2w_all"]    # Shape: (B, 4, 4)
    o2c_mat = np.load(f"{mvs_root}/o2w_normalized_aligned.npy")
    c2o_mat = np.linalg.inv(o2c_mat) # camera to object
    # obj_3d_o = trimesh.load(f"{mvs_root}/sparse_points_normalized_aligned.ply", process=False).vertices
    # obj_3d_o = np.tile(obj_3d_o, (o2c_mat.shape[0], 1, 1))
    # img_fs = data['object']["im_paths"]
    # intrinsic = data['object']["K"]        # Camera intrinsics
    # obj_scale = data['object']['obj_scale']

    ###### log hand ########
    # Initialize MANO model
    f3d_r = initialize_mano_model(MANO_f)

    # betas = data['right']["hand_beta"]
    # betas = torch.tensor(np.tile(betas, (o2c_mat.shape[0], 1))).cuda()
    # h2c_rot = torch.tensor(data['right']["hand_rot"]).cuda()
    # h2c_transl = torch.tensor(data['right']["hand_transl"]).cuda()
    # hand_pose = torch.tensor(data['right']["hand_pose"]).cuda()
    

    # Extract and transform right hand vertices
    mano_layer = smplx.create(
            model_path=MANO_f, model_type="mano", use_pca=False, is_rhand=True
    )
    mano_layer.to(torch.device("cuda"))
    ### Note: Hand coordinates to object coordinates only involves translation, not rotation
    # hand vertices in canonical coordinates
    # breakpoint()
    hand_out_can = mano_layer(
        betas=betas,
        hand_pose=hand_pose,
        transl=torch.zeros_like(h2c_transl),
        global_orient=h2c_rot,
    )
    hand_v_can = hand_out_can.vertices.cpu().numpy()
    hand_jnts_can = hand_out_can.joints.cpu().numpy()
    h2c_mat = np.tile(np.eye(4), (o2c_mat.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = h2c_transl.cpu().numpy()
    h2c_mat = h2c_mat * hand_scale # scale the hand
    h2c_mat[:, 3, 3] = 1
    h2o_mat = c2o_mat @ h2c_mat
    # hand vertices in camera coordinates 
    # hand vertices in object coordinates
    hand_v_o = transform_points(hand_v_can, h2o_mat)
    hand_jnts_o = transform_points(hand_jnts_can, h2o_mat)


    hand_v_c = transform_points(hand_v_o, o2c_mat)
    hand_jnts_c = transform_points(hand_jnts_o, o2c_mat)

    object_v_c = transform_points( mesh_c_o.vertices, o2c_mat)

    hand_v_c /= hand_scale
    hand_jnts_c /= hand_scale    

    object_v_c = object_v_c / hand_scale



    out = xdict()
    out['verts.right'] = hand_v_c
    out['jnts.right'] = hand_jnts_c
    out['root.right'] = hand_jnts_c[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = object_v_c
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]

    faces = {
        'object' : np.array(mesh_c_o.faces),        
        'right' : np.array(f3d_r),

    }
    out["faces"] = faces
        
    print("Done loading data")
    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)

    out_dict = xdict()
    out_dict["fnames"] = img_fs#.tolist()

    out_dict["K"] = intrinsic[None]
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
