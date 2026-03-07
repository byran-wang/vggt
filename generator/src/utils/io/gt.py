import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import torch
import trimesh
from common.body_models import build_mano_aa
from common.transforms import project2d_batch, rigid_tf_torch_batch
from common.viewer import ViewerData
from PIL import Image

# from src_data.preprocessing_utils import tf.cv2gl_mano
import common.transforms as tf

# from src_data.smplx import MANO
from common.xdict import xdict
from src.utils.eval_modules import compute_bounding_box_centers
from src.utils.const import SEGM_IDS

import rerun as rr
import rerun.blueprint as rrb
from rerun import ImageFormat
from third_party.utils_simba.utils_simba.vis import log_asset_axis, log_point_cloud, rotation_matrix_to_quaternion, rr_show_trajectory, log_obj_model, log_sphere
from third_party.utils_simba.utils_simba.vis import log_asset_3D
from third_party.utils_simba.utils_simba.img import compress_image
from third_party.utils_simba.utils_simba.geometry import transform_points, save_point_cloud
import cv2
import trimesh


def add_material(color: list) -> rr.Material:
    """
    Creates a ReRun material with the specified color.

    Parameters:
        color (list): RGBA color list.

    Returns:
        rr.Material: ReRun material instance.
    """
    return rr.Material(albedo_factor=color)

def load_and_compress_image(img_path):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width = bgr.shape[:2]
    if bgr is None:
        raise FileNotFoundError(f"Image file not found: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    compressed_rgb = compress_image(rgb, format="JPEG", quality=75)
    return compressed_rgb, height, width

def compute_vertex_normals(vertices, faces):
    # Initialize normals to zero
    normals = np.zeros(vertices.shape, dtype=np.float32)
    
    # Compute normals for each face
    for face in faces:
        idx0, idx1, idx2 = face
        v0 = vertices[idx0]
        v1 = vertices[idx1]
        v2 = vertices[idx2]
        
        # Compute the normal of the face
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Add the face normal to each vertex normal
        normals[idx0] += face_normal
        normals[idx1] += face_normal
        normals[idx2] += face_normal
    
    # Normalize the normals
    norm = np.linalg.norm(normals, axis=1)
    norm[norm == 0] = 1  # Avoid division by zero
    normals /= norm[:, np.newaxis]
    
    return normals

def log_frame_data(
    frame_id: int,
    cam_w_mat: np.ndarray, # Transformation matrix to world coordinates. World coordinates could be camera, object or hand
    intrinsic: np.ndarray,
    img_f: str,
    obj_w: np.ndarray,
    hand_w: np.ndarray = None,
    hand_f: np.ndarray = None,
    red_material: rr.Material = None,
):
    """
    Logs data for a single frame to ReRun.

    Parameters:
        frame_id (int): Current frame identifier.
        cam_w_mat (np.ndarray): Camera transform matrix in world coordinates.
        intrinsic (np.ndarray): Camera intrinsic matrix.
        img_f (str): Image file path.
        obj_w (np.ndarray): Object points in world space.
        hand_w (np.ndarray): Right hand vertices in world space.
        hand_f (np.ndarray): Faces of the right hand mesh.
        red_material (rr.Material): Material for the hand mesh.
    """
    # Set time sequence
    rr.set_time_sequence("frame_id", frame_id)

    # Extract translation and rotation
    tvec = cam_w_mat[:3, 3]
    quat_xyzw = rotation_matrix_to_quaternion(cam_w_mat[:3, :3])

    # Log camera transform
    rr.log(
        f"/cameras/cameras_{frame_id}",
        rr.Transform3D(
            translation=tvec,
            rotation=rr.Quaternion(xyzw=quat_xyzw),
            axis_length=0.05,
            from_parent=False
        ),
        timeless=True
    )

    # Log view coordinates
    rr.log(
        f"/cameras/cameras_{frame_id}",
        rr.ViewCoordinates.RDF,
        static=True,
        timeless=True
    )
    # Load and compress image
    rgb, height, width = load_and_compress_image(img_f)

    # Log image to camera
    rr.log(
        f"/cameras/cameras_{frame_id}/image",
        rr.ImageEncoded(contents=rgb, format=ImageFormat.JPEG),
        timeless=True
    )

    # Log camera intrinsics
    rr.log(
        f"/cameras/cameras_{frame_id}",
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[intrinsic[0, 0], intrinsic[1, 1]],
            principal_point=[intrinsic[0, 2], intrinsic[1, 2]],
        ),
        timeless=True
    )

    # Log object points
    rr.log(
        "object",
        rr.Points3D(obj_w, radii=0.003)
    )

    # Log image to Spatial2DView
    rr.log(
        "/image",
        rr.ImageEncoded(contents=rgb, format=ImageFormat.JPEG)
    )

    # Log image transform
    rr.log(
        "/image",
        rr.Transform3D(
            translation=tvec,
            rotation=rr.Quaternion(xyzw=quat_xyzw),
            axis_length=0.05,
            from_parent=False
        )
    )

    # Log image view coordinates
    rr.log(
        "/image",
        rr.ViewCoordinates.RDF,
        static=True
    )

    # Log camera intrinsics for image view
    rr.log(
        "/image",
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[intrinsic[0, 0], intrinsic[1, 1]],
            principal_point=[intrinsic[0, 2], intrinsic[1, 2]],
        )
    )
    if hand_w is None or hand_f is None or red_material is None:
        return
    # Compute vertex normals
    normals = compute_vertex_normals(hand_w, hand_f)

    # Log hand mesh
    rr.log(
        "hand",
        rr.Mesh3D(
            vertex_positions=hand_w,
            triangle_indices=hand_f,
            vertex_normals=normals,
            mesh_material=red_material
        )
    )

def load_data_diff_hoi(
        seq_name,
        image_id,
        mano_path="./assets/mano/models/",
        debug = False,
):

    from smplx import MANO
    # load in opencv format
    device = "cuda:0"
    human_model = MANO(
        mano_path, is_rhand=True, flat_hand_mean=False, use_pca=False
    ).to(device)

    data = torch.load(f"../hold-private/generator/assets/ho3d_v3/processed/{seq_name}.pt")
    mano_layer = MANO(
        mano_path,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=True,
    )

    fnames = data["fnames"]
    fnames = [fname.replace("./", "../hold-private/") for fname in fnames]
    hand_pose = data["hand_pose"]
    hand_beta = data["hand_beta"]
    hand_transl = data["hand_transl"]
    K = data["K"]
    obj_trans = data["obj_trans"]
    obj_rot = data["obj_rot"]
    obj_name = data["obj_name"]
    is_valid = data["is_valid"]
    not_valid = (1 - is_valid).bool()
    assert len(fnames) == obj_trans.shape[0]
    assert len(fnames) == obj_rot.shape[0]
    obj_mesh_f = f"../hold-private/generator/assets/ho3d_v3/models/{obj_name}/textured_simple.obj"
    obj_mesh = trimesh.load(obj_mesh_f, process=False,)
    obj_3d_o = np.tile(np.array(obj_mesh.vertices), (obj_rot.shape[0], 1, 1))
    
    c2o = torch.tile(torch.eye(4), (obj_trans.shape[0], 1, 1))
    c2o[:, :3, :3] = obj_rot
    c2o[:, :3, 3] = obj_trans
    c2o[:, 1:3] *= -1 # gl to cv
    o2c = torch.inverse(c2o) # o2c -> c2o


    # OpenGL to OpenCV
    num_frames = hand_pose.shape[0]
    hand_rot = hand_pose[:, :3].numpy()
    hand_pose = hand_pose[:, 3:].numpy()
    hand_transl = hand_transl.numpy()
    hand_rot_all = []
    hand_transl_all = []
    for idx in range(num_frames):
        T_hip = (
            human_model.get_T_hip(betas=hand_beta[idx : idx + 1].to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        hand_rot_cv, hand_transl_cv = tf.cv2gl_mano(
            hand_rot[idx], hand_transl[idx], T_hip.reshape(3)
        )
        hand_rot_all.append(hand_rot_cv)
        hand_transl_all.append(hand_transl_cv)
    hand_rot_all = np.array(hand_rot_all)
    hand_transl_all = np.array(hand_transl_all)

    hand_pose = torch.FloatTensor(hand_pose)
    hand_rot = torch.FloatTensor(hand_rot_all)
    hand_transl = torch.FloatTensor(hand_transl_all)

    with torch.no_grad():
        mano_output = mano_layer(
            betas=hand_beta,
            hand_pose=hand_pose,
            global_orient=torch.zeros_like(hand_rot),
            transl=torch.zeros_like(hand_transl),
        )

        v3d_h_c = mano_output.vertices.cpu().numpy()  # in camera_space
        j3d_h_c = mano_output.joints.cpu().numpy()  # in camera_sapce

        num_frames = hand_transl.shape[0]
    h2c_mat = np.tile(np.eye(4), (o2c.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = hand_transl.cpu().numpy()
    h2c_mat[:, 3, 3] = 1
    h2o_mat = o2c.cpu().numpy() @ h2c_mat        
    
    v3d_h_o = transform_points(v3d_h_c, h2o_mat) # in object space
    j3d_h_o = transform_points(j3d_h_c, h2o_mat) # in object space

    DUMMPY_VAL = -1000
    v3d_h_o[not_valid, :] = DUMMPY_VAL
    # v3d_o_cam[not_valid, :] = DUMMPY_VAL
    j3d_h_o[not_valid, :] = DUMMPY_VAL
    # v2d_h[not_valid, :] = DUMMPY_VAL
    # v2d_o[not_valid, :] = DUMMPY_VAL


    if type(image_id) == list:
        selected_fnames = image_id
    else:
        selected_fnames = [image_id]
    assert len(selected_fnames) > 0
    # Get selected file IDs
    selected_fids = np.array(
    [int(op.basename(fname).split(".")[0])*5 for fname in selected_fnames]
    )
    assert len(selected_fids) > 0

    out = xdict()
    hand_v_o = v3d_h_o[selected_fids]
    hand_jnts_o = j3d_h_o[selected_fids]
    hand_v_can = v3d_h_c[selected_fids]
    hand_jnts_can = j3d_h_c[selected_fids]
    object_verts_o = obj_3d_o[selected_fids]
    out['verts.right'] = hand_v_o
    out['jnts.right'] = hand_jnts_o
    out['root.right'] = hand_jnts_o[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = object_verts_o
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    out['is_valid'] = is_valid[selected_fids]
    out["faces.object"] = np.array(obj_mesh.faces)
    print("Done loading data")

    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)


    if 0:
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(name="cameras_hands", 
                            defaults=[rr.components.ImagePlaneDistance(0.2)],
                            origin="/"),                         
            rrb.Horizontal(
                rrb.Spatial2DView(name="image", origin="/image"),
            ),
            row_shares=[5, 2],
        )
        rr.init(
            application_id=seq_name,
            default_blueprint=blueprint,
            spawn=True)
        log_asset_axis(log_path_prefix="", scale=0.2)
        log_asset_3D([obj_mesh_f]) 

        img_fs = fnames
        cam_w_mats = o2c.cpu().numpy()
        obj_ws = obj_3d_o
        intrinsic = K.cpu().numpy()[0]

        f3d_r = np.array(human_model.faces)
        hand_ws = v3d_h_o
        if 1:
            obj_mesh = trimesh.Trimesh(vertices=out['verts.object'][0], faces=obj_mesh.faces)
            obj_mesh.export("object_mesh_o_gt.obj")

            hand_mesh = trimesh.Trimesh(vertices=out['verts.right'][0], faces=f3d_r)
            hand_mesh.export("hand_mesh_o_gt.obj")


            hand_mesh = trimesh.Trimesh(vertices=hand_v_can[0], faces=f3d_r)
            hand_mesh.export("hand_mesh_can_gt.obj")
            
            hand_mesh = trimesh.Trimesh(vertices=(hand_v_can - hand_jnts_can[:,0:1, :])[0], faces=f3d_r)
            hand_mesh.export("hand_mesh_can_ra_gt.obj")
            colors = np.random.rand(out['j3d_ra.right'][0].shape[0], 3)
            colors = np.array(
                    [[0.87933823, 0.57164265, 0.51228683],
                    [0.61563337, 0.87884668, 0.75571479],
                    [0.61316533, 0.57664449, 0.74287663],
                    [0.748819  , 0.49999368, 0.27068597],
                    [0.3225945 , 0.16445378, 0.85036565],
                    [0.55646468, 0.90872404, 0.84165958],
                    [0.59780829, 0.08921893, 0.55948613],
                    [0.84240964, 0.2194021 , 0.04895059],
                    [0.37947888, 0.77235679, 0.62044855],
                    [0.7429536 , 0.0447016 , 0.73697736],
                    [0.4443045 , 0.42474416, 0.78575367],
                    [0.14269189, 0.34107265, 0.89379201],
                    [0.77566054, 0.92529184, 0.24586288],
                    [0.0957135 , 0.02743424, 0.12720011],
                    [0.60559555, 0.25561199, 0.84445288],
                    [0.92320857, 0.94816419, 0.77933839],
                    [0.7646481 , 0.60660361, 0.28126962],
                    [0.04167282, 0.7317444 , 0.83515505],
                    [0.51410243, 0.65174915, 0.5335095 ],
                    [0.64208175, 0.18345901, 0.54004745],
                    [0.41742055, 0.46962938, 0.33586929]])
            if 1:
                save_point_cloud(out['j3d_ra.right'][0].numpy(), "./hand_joints_can_ra_gt.ply", colors)
            else:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(out['j3d_ra.right'][0].numpy())
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([pcd])
                o3d.io.write_point_cloud("hand_joints_can_gt.ply", pcd)



        # Create material for hand mesh
        red_color = [1.0, 0.0, 0.0, 1.0]
        red_material = add_material(red_color)        
        for frame_id in np.linspace(0, 100, 4):
            frame_id = int(frame_id)
            log_frame_data(
                frame_id=frame_id,
                cam_w_mat=cam_w_mats[frame_id],
                intrinsic=intrinsic,
                img_f=img_fs[frame_id],
                obj_w=obj_ws[frame_id],
                hand_w=hand_ws[frame_id],
                hand_f=f3d_r,
                red_material=red_material
            )
    return out

def load_data_easy_hoi(
        seq_name,
        image_id,
        mano_path="./assets/mano/models/",
        debug = False,
):

    from smplx import MANO
    # load in opencv format
    device = "cuda:0"
    human_model = MANO(
        mano_path, is_rhand=True, flat_hand_mean=False, use_pca=False
    ).to(device)

    data = torch.load(f"../hold-private/generator/assets/ho3d_v3/processed/{seq_name}.pt")
    mano_layer = MANO(
        mano_path,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=True,
    )

    fnames = data["fnames"]
    fnames = [fname.replace("./", "../hold-private/") for fname in fnames]
    hand_pose = data["hand_pose"]
    hand_beta = data["hand_beta"]
    hand_transl = data["hand_transl"]
    K = data["K"]
    obj_trans = data["obj_trans"]
    obj_rot = data["obj_rot"]
    obj_name = data["obj_name"]
    is_valid = data["is_valid"]
    not_valid = (1 - is_valid).bool()
    assert len(fnames) == obj_trans.shape[0]
    assert len(fnames) == obj_rot.shape[0]
    obj_mesh_f = f"../hold-private/generator/assets/ho3d_v3/models/{obj_name}/textured_simple.obj"
    obj_mesh = trimesh.load(obj_mesh_f, process=False,)
    obj_3d_o = np.tile(np.array(obj_mesh.vertices), (obj_rot.shape[0], 1, 1))
    
    c2o = torch.tile(torch.eye(4), (obj_trans.shape[0], 1, 1))
    c2o[:, :3, :3] = obj_rot
    c2o[:, :3, 3] = obj_trans
    c2o[:, 1:3] *= -1 # gl to cv
    o2c = torch.inverse(c2o) # o2c -> c2o


    # OpenGL to OpenCV
    num_frames = hand_pose.shape[0]
    hand_rot = hand_pose[:, :3].numpy()
    hand_pose = hand_pose[:, 3:].numpy()
    hand_transl = hand_transl.numpy()
    hand_rot_all = []
    hand_transl_all = []
    for idx in range(num_frames):
        T_hip = (
            human_model.get_T_hip(betas=hand_beta[idx : idx + 1].to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        hand_rot_cv, hand_transl_cv = tf.cv2gl_mano(
            hand_rot[idx], hand_transl[idx], T_hip.reshape(3)
        )
        hand_rot_all.append(hand_rot_cv)
        hand_transl_all.append(hand_transl_cv)
    hand_rot_all = np.array(hand_rot_all)
    hand_transl_all = np.array(hand_transl_all)

    hand_pose = torch.FloatTensor(hand_pose)
    hand_rot = torch.FloatTensor(hand_rot_all)
    hand_transl = torch.FloatTensor(hand_transl_all)

    with torch.no_grad():
        mano_output = mano_layer(
            betas=hand_beta,
            hand_pose=hand_pose,
            global_orient=hand_rot,
            transl=torch.zeros_like(hand_transl),
        )

        v3d_h_c = mano_output.vertices.cpu().numpy()  # in camera_space
        j3d_h_c = mano_output.joints.cpu().numpy()  # in camera_sapce

        num_frames = hand_transl.shape[0]
    h2c_mat = np.tile(np.eye(4), (o2c.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = hand_transl.cpu().numpy()
    h2c_mat[:, 3, 3] = 1
    h2o_mat = o2c.cpu().numpy() @ h2c_mat        
    
    v3d_h_o = transform_points(v3d_h_c, h2o_mat) # in object space
    j3d_h_o = transform_points(j3d_h_c, h2o_mat) # in object space

    DUMMPY_VAL = -1000
    v3d_h_o[not_valid, :] = DUMMPY_VAL
    # v3d_o_cam[not_valid, :] = DUMMPY_VAL
    j3d_h_o[not_valid, :] = DUMMPY_VAL
    # v2d_h[not_valid, :] = DUMMPY_VAL
    # v2d_o[not_valid, :] = DUMMPY_VAL


    if type(image_id) == list:
        selected_fnames = image_id
    else:
        selected_fnames = [image_id]
    assert len(selected_fnames) > 0
    # Get selected file IDs
    selected_fids = np.array(
    [int(op.basename(fname).split(".")[0])*5 for fname in selected_fnames]
    )
    assert len(selected_fids) > 0

    out = xdict()
    hand_v_o = v3d_h_o[selected_fids]
    hand_jnts_o = j3d_h_o[selected_fids]
    hand_v_can = v3d_h_c[selected_fids]
    hand_jnts_can = j3d_h_c[selected_fids]
    object_verts_o = obj_3d_o[selected_fids]
    out['verts.right'] = hand_v_o
    out['jnts.right'] = hand_jnts_o
    out['root.right'] = hand_jnts_o[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = object_verts_o
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    out['is_valid'] = is_valid[selected_fids]
    out["faces.object"] = np.array(obj_mesh.faces)
    print("Done loading data")

    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)


    if 0:
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(name="cameras_hands", 
                            defaults=[rr.components.ImagePlaneDistance(0.2)],
                            origin="/"),                         
            rrb.Horizontal(
                rrb.Spatial2DView(name="image", origin="/image"),
            ),
            row_shares=[5, 2],
        )
        rr.init(
            application_id=seq_name,
            default_blueprint=blueprint,
            spawn=True)
        log_asset_axis(log_path_prefix="", scale=0.2)
        log_asset_3D([obj_mesh_f]) 

        img_fs = fnames
        cam_w_mats = o2c.cpu().numpy()
        obj_ws = obj_3d_o
        intrinsic = K.cpu().numpy()[0]

        f3d_r = np.array(human_model.faces)
        hand_ws = v3d_h_o
        if 1:
            obj_mesh = trimesh.Trimesh(vertices=out['verts.object'][0], faces=obj_mesh.faces)
            obj_mesh.export("object_mesh_o_gt.obj")

            hand_mesh = trimesh.Trimesh(vertices=out['verts.right'][0], faces=f3d_r)
            hand_mesh.export("hand_mesh_o_gt.obj")

            hand_mesh = trimesh.Trimesh(vertices=(hand_v_can - hand_jnts_can[:,0:1, :])[0], faces=f3d_r)
            hand_mesh.export("hand_mesh_can_gt.obj")
            colors = np.random.rand(out['j3d_ra.right'][0].shape[0], 3)
            colors = np.array(
                    [[0.87933823, 0.57164265, 0.51228683],
                    [0.61563337, 0.87884668, 0.75571479],
                    [0.61316533, 0.57664449, 0.74287663],
                    [0.748819  , 0.49999368, 0.27068597],
                    [0.3225945 , 0.16445378, 0.85036565],
                    [0.55646468, 0.90872404, 0.84165958],
                    [0.59780829, 0.08921893, 0.55948613],
                    [0.84240964, 0.2194021 , 0.04895059],
                    [0.37947888, 0.77235679, 0.62044855],
                    [0.7429536 , 0.0447016 , 0.73697736],
                    [0.4443045 , 0.42474416, 0.78575367],
                    [0.14269189, 0.34107265, 0.89379201],
                    [0.77566054, 0.92529184, 0.24586288],
                    [0.0957135 , 0.02743424, 0.12720011],
                    [0.60559555, 0.25561199, 0.84445288],
                    [0.92320857, 0.94816419, 0.77933839],
                    [0.7646481 , 0.60660361, 0.28126962],
                    [0.04167282, 0.7317444 , 0.83515505],
                    [0.51410243, 0.65174915, 0.5335095 ],
                    [0.64208175, 0.18345901, 0.54004745],
                    [0.41742055, 0.46962938, 0.33586929]])
            if 1:
                save_point_cloud(out['j3d_ra.right'][0].numpy(), "hand_joints_can_gt.ply", colors)
            else:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(out['j3d_ra.right'][0].numpy())
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([pcd])
                o3d.io.write_point_cloud("hand_joints_can_gt.ply", pcd)



        # Create material for hand mesh
        red_color = [1.0, 0.0, 0.0, 1.0]
        red_material = add_material(red_color)        
        for frame_id in np.linspace(0, 100, 4):
            frame_id = int(frame_id)
            log_frame_data(
                frame_id=frame_id,
                cam_w_mat=cam_w_mats[frame_id],
                intrinsic=intrinsic,
                img_f=img_fs[frame_id],
                obj_w=obj_ws[frame_id],
                hand_w=hand_ws[frame_id],
                hand_f=f3d_r,
                red_material=red_material
            )
        breakpoint()
    return out

def load_data_diff_object(
        full_seq_name = "",
        mvs_root = "", 
        debug = False,
):

    seq_name = full_seq_name.split("_")[1]
    from smplx import MANO

    # load in opencv format
    device = "cuda:0"
    human_model = MANO(
        "../code/body_models", is_rhand=True, flat_hand_mean=False, use_pca=False
    ).to(device)

    data = torch.load(f"../generator/assets/ho3d_v3/processed/{seq_name}.pt")
    mano_layer = build_mano_aa(True, flat_hand=False)

    fnames = data["fnames"]
    fnames = [fname.replace("./", "../") for fname in fnames]
    hand_pose = data["hand_pose"]
    hand_beta = data["hand_beta"]
    hand_transl = data["hand_transl"]
    K = data["K"]
    obj_trans = data["obj_trans"]
    obj_rot = data["obj_rot"]
    obj_name = data["obj_name"]
    is_valid = data["is_valid"]
    not_valid = (1 - is_valid).bool()
    assert len(fnames) == obj_trans.shape[0]
    assert len(fnames) == obj_rot.shape[0]
    obj_mesh_f = f"../generator/assets/ho3d_v3/models/{obj_name}/textured_simple.obj"
    obj_mesh = trimesh.load(obj_mesh_f, process=False,)
    obj_3d_o = np.tile(np.array(obj_mesh.vertices), (obj_rot.shape[0], 1, 1))
    
    c2o = torch.tile(torch.eye(4), (obj_trans.shape[0], 1, 1))
    c2o[:, :3, :3] = obj_rot
    c2o[:, :3, 3] = obj_trans
    c2o[:, 1:3] *= -1 # gl to cv
    o2c = torch.inverse(c2o) # o2c -> c2o


    # OpenGL to OpenCV
    num_frames = hand_pose.shape[0]
    hand_rot = hand_pose[:, :3].numpy()
    hand_pose = hand_pose[:, 3:].numpy()
    hand_transl = hand_transl.numpy()
    hand_rot_all = []
    hand_transl_all = []
    for idx in range(num_frames):
        T_hip = (
            human_model.get_T_hip(betas=hand_beta[idx : idx + 1].to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        hand_rot_cv, hand_transl_cv = tf.cv2gl_mano(
            hand_rot[idx], hand_transl[idx], T_hip.reshape(3)
        )
        hand_rot_all.append(hand_rot_cv)
        hand_transl_all.append(hand_transl_cv)
    hand_rot_all = np.array(hand_rot_all)
    hand_transl_all = np.array(hand_transl_all)

    hand_pose = torch.FloatTensor(hand_pose)
    hand_rot = torch.FloatTensor(hand_rot_all)
    hand_transl = torch.FloatTensor(hand_transl_all)

    with torch.no_grad():
        mano_output = mano_layer(
            betas=hand_beta,
            hand_pose=hand_pose,
            global_orient=hand_rot,
            transl=torch.zeros_like(hand_transl),
        )

        v3d_h_c = mano_output.vertices.cpu().numpy()  # in camera_space
        j3d_h_c = mano_output.joints.cpu().numpy()  # in camera_sapce

        num_frames = hand_transl.shape[0]
    h2c_mat = np.tile(np.eye(4), (o2c.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = hand_transl.cpu().numpy()
    h2c_mat[:, 3, 3] = 1
    h2o_mat = o2c.cpu().numpy() @ h2c_mat        
    
    v3d_h_o = transform_points(v3d_h_c, h2o_mat) # in object space
    j3d_h_o = transform_points(j3d_h_c, h2o_mat) # in object space

    DUMMPY_VAL = -1000
    v3d_h_o[not_valid, :] = DUMMPY_VAL
    # v3d_o_cam[not_valid, :] = DUMMPY_VAL
    j3d_h_o[not_valid, :] = DUMMPY_VAL
    # v2d_h[not_valid, :] = DUMMPY_VAL
    # v2d_o[not_valid, :] = DUMMPY_VAL


    
    selected_fnames = sorted(glob(f"{mvs_root}/../../HO3D/cameras/*.json"))
    assert len(selected_fnames) > 0
    # Get selected file IDs
    selected_fids = np.array(
    [int(op.basename(fname).split(".")[0])*5 for fname in selected_fnames]
    )
    assert len(selected_fids) > 0

    out = xdict()
    hand_v_o = v3d_h_o[selected_fids]
    hand_jnts_o = j3d_h_o[selected_fids]
    hand_jnts_can = j3d_h_c[selected_fids]
    object_verts_o = obj_3d_o[selected_fids]
    out['verts.right'] = hand_v_o
    out['jnts.right'] = hand_jnts_o
    out['root.right'] = hand_jnts_o[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = object_verts_o
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    out['is_valid'] = is_valid[selected_fids]
    out["faces.object"] = np.array(obj_mesh.faces)
    print("Done loading data")

    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)


    if debug:
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(name="cameras_hands", 
                            defaults=[rr.components.ImagePlaneDistance(0.2)],
                            origin="/"),                         
            rrb.Horizontal(
                rrb.Spatial2DView(name="image", origin="/image"),
            ),
            row_shares=[5, 2],
        )
        rr.init(
            application_id=seq_name,
            default_blueprint=blueprint,
            spawn=True)
        log_asset_axis(log_path_prefix="", scale=0.2)
        log_asset_3D([obj_mesh_f]) 

        img_fs = fnames
        cam_w_mats = o2c.cpu().numpy()
        obj_ws = obj_3d_o
        intrinsic = K.cpu().numpy()[0]

        f3d_r = np.array(human_model.faces)
        hand_ws = v3d_h_o

        # Create material for hand mesh
        red_color = [1.0, 0.0, 0.0, 1.0]
        red_material = add_material(red_color)        
        for frame_id in np.linspace(0, 100, 4):
            frame_id = int(frame_id)
            log_frame_data(
                frame_id=frame_id,
                cam_w_mat=cam_w_mats[frame_id],
                intrinsic=intrinsic,
                img_f=img_fs[frame_id],
                obj_w=obj_ws[frame_id],
                hand_w=hand_ws[frame_id],
                hand_f=f3d_r,
                red_material=red_material
            )
        breakpoint()
    return out

def load_data(full_seq_name, get_selected_fids_fn=None):
    from smplx import MANO

    # load in opencv format

    seq_name = full_seq_name.split("_")[1]

    device = "cuda:0"
    human_model = MANO(
        "../code/body_models", is_rhand=True, flat_hand_mean=False, use_pca=False
    ).to(device)

    data = torch.load(f"../generator/assets/ho3d_v3/processed/{seq_name}.pt")
    mano_layer = build_mano_aa(True, flat_hand=False)

    fnames = data["fnames"]
    fnames = [fname.replace("./generator", "../generator/") for fname in fnames]
    hand_pose = data["hand_pose"]
    hand_beta = data["hand_beta"]
    hand_transl = data["hand_transl"]
    K = data["K"]
    obj_trans = data["obj_trans"]
    obj_rot = data["obj_rot"]
    obj_name = data["obj_name"]
    is_valid = data["is_valid"]
    not_valid = (1 - is_valid).bool()
    assert len(fnames) == obj_trans.shape[0]
    assert len(fnames) == obj_rot.shape[0]
    # get all png files in the directory and sort them
    if get_selected_fids_fn is None:
        # Default behavior
        glob_path = f"./data/{full_seq_name}/processed/images/*.png"
        selected_fnames = sorted(glob(glob_path))
        selected_fids = np.array(
            [(int(op.basename(fname).split(".")[0])) * 5 for fname in selected_fnames]
        )
    else:
        # Use callback function
        selected_fids = get_selected_fids_fn(full_seq_name)
    assert len(selected_fids) > 0
    obj_mesh = trimesh.load(
        f"../generator/assets/ho3d_v3/models/{obj_name}/textured_simple.obj",
        process=False,
    )

    # OpenGL to OpenCV
    num_frames = hand_pose.shape[0]
    hand_rot = hand_pose[:, :3].numpy()
    hand_pose = hand_pose[:, 3:].numpy()
    hand_transl = hand_transl.numpy()
    hand_rot_all = []
    hand_transl_all = []
    for idx in range(num_frames):
        T_hip = (
            human_model.get_T_hip(betas=hand_beta[idx : idx + 1].to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        hand_rot_cv, hand_transl_cv = tf.cv2gl_mano(
            hand_rot[idx], hand_transl[idx], T_hip.reshape(3)
        )
        hand_rot_all.append(hand_rot_cv)
        hand_transl_all.append(hand_transl_cv)
    hand_rot_all = np.array(hand_rot_all)
    hand_transl_all = np.array(hand_transl_all)

    hand_pose = torch.FloatTensor(hand_pose)
    hand_rot = torch.FloatTensor(hand_rot_all)
    hand_transl = torch.FloatTensor(hand_transl_all)

    with torch.no_grad():
        mano_output = mano_layer(
            betas=hand_beta,
            hand_pose=hand_pose,
            global_orient=hand_rot,
            transl=hand_transl,
        )

        v3d_h = mano_output.vertices  # .numpy()
        j3d_h = mano_output.joints  # .numpy()

        num_frames = hand_transl.shape[0]
    v_cano_o = torch.FloatTensor(obj_mesh.vertices).repeat(num_frames, 1, 1)
    c_cano_o = torch.FloatTensor(obj_mesh.visual.to_color().vertex_colors)

    Rt_o = torch.eye(4)[None, :, :].repeat(num_frames, 1, 1)
    Rt_o[:, :3, :3] = obj_rot
    Rt_o[:, :3, 3] = obj_trans
    Rt_o[:, 1:3] *= -1

    v3d_o_cam = rigid_tf_torch_batch(v_cano_o, Rt_o[:, :3, :3], Rt_o[:, :3, 3:])
    v2d_o = project2d_batch(K, v3d_o_cam)
    v2d_h = project2d_batch(K, v3d_h)

    # masks_ps = sorted(glob(f"./data/{full_seq_name}/build/mask/*"))
    # masks_gt = np.stack([Image.open(mask_p) for mask_p in masks_ps], axis=0)
    # boxes = np.load(f"./data/{full_seq_name}/build/boxes.npy")
    # from src.fitting.utils import crop_masks

    hand_id = SEGM_IDS["right"]
    obj_id = SEGM_IDS["object"]
    # masks_gt = crop_masks(masks_gt, boxes, hand_id, obj_id, scale=0.7)
    # masks_gt = torch.FloatTensor(masks_gt)

    DUMMPY_VAL = -1000
    v3d_h[not_valid, :] = DUMMPY_VAL
    v3d_o_cam[not_valid, :] = DUMMPY_VAL
    j3d_h[not_valid, :] = DUMMPY_VAL
    v2d_h[not_valid, :] = DUMMPY_VAL
    v2d_o[not_valid, :] = DUMMPY_VAL

    # Select ground truth data based on selected file IDs
    v3d_h = v3d_h[selected_fids]
    v3d_o_cam = v3d_o_cam[selected_fids]
    j3d_h = j3d_h[selected_fids]
    v2d_h = v2d_h[selected_fids]
    v2d_o = v2d_o[selected_fids]
    fnames = np.array(fnames)[selected_fids]
    is_valid = is_valid[selected_fids]

    out = {}
    out["fnames"] = fnames
    out["v3d_c.right"] = v3d_h.detach().numpy()
    out["v3d_c.object"] = v3d_o_cam.detach().numpy()
    out["j3d_c.right"] = j3d_h.detach().numpy()
    out["v2d_h"] = v2d_h.detach().numpy()
    out["v2d_o"] = v2d_o.detach().numpy()
    out["faces.object"] = np.array(obj_mesh.faces)
    out["faces.right"] = np.array(human_model.faces)
    out["K"] = K[0].numpy()
    out["c2o"] = Rt_o[selected_fids].detach().numpy()
    out["colors.object"] = c_cano_o.detach().numpy()
    # out["masks_gt"] = masks_gt.detach().numpy()
    out["is_valid"] = is_valid.detach().numpy()

    # rh
    root_j3d = j3d_h[:, :1]
    root_o = torch.FloatTensor(
        compute_bounding_box_centers(v3d_o_cam.detach().numpy())[:, None]
    )
    out["v3d_right.object"] = torch.stack(
        [verts - rj3d for verts, rj3d in zip(v3d_o_cam, root_j3d)], dim=0
    ).numpy()
    out["j3d_ra.right"] = j3d_h - root_j3d
    out["v3d_ra.object"] = v3d_o_cam - root_o
    out["root.object"] = root_o[:, 0]
    out = xdict(out).to_torch()
    return out


def load_viewer_data(args, get_selected_fids_fn=None):
    full_seq_name = args.seq_name
    data = load_data(full_seq_name, get_selected_fids_fn)

    v3d_h_c = data["v3d_c.right"].numpy()
    v3d_o_c = data["v3d_c.object"].numpy()

    faces_o = data["faces.object"].numpy()
    faces_h = data["faces.right"].numpy()
    K = data["K"].numpy().reshape(3, 3)
    
    fnames = data["fnames"]
    from common.body_models import seal_mano_mesh_np

    v3d_h_c, faces_h = seal_mano_mesh_np(v3d_h_c, faces_h, is_rhand=True)

    vis_dict = {}
    vis_dict["right-gt"] = {
        "v3d": v3d_h_c,
        "f3d": faces_h,
        "vc": None,
        "name": "right-gt",
        "color": "gray_white",
        "flat_shading": True,
    }

    vis_dict["obj-gt"] = {
        "v3d": v3d_o_c,
        "f3d": faces_o,
        "vc": None,
        "name": "object-gt",
        "color": "green-blue",
        # "color": "red",
        "flat_shading": False,
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
