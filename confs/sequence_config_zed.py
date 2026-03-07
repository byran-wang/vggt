sequence_name_list = [
    ###### in-the-wild captured by ZED ##########
    "CUB1",
    "CUB2",
    "DUC1",
    "DUC2",
    "TC3",
    "TC4",
    "WC3",
    "WC4", 
]
sequences = {
    # {
    # # reconstruction fail after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    # # object is stationary in [0, 17]
    # "id": "hold_ABF12_ho3d.0",
    # "cond_select_strategy": "manual", # object_hand_ratio or object_pixel_mask or manual
    # "cond_image": 0,  
    # "consecutive_frame_star": 0,
    # "consecutive_frame_num": 30,
    # "consecutive_frame_interval": 1,
    # "data_path": HO3D_v3_HOLD_data_path,
    # "cond_pose_from_selected_cam": False,
    # },
    "default":
    {
    "geometry_poor_frames": [],
    "cond_idx": 0,
    "cond_select_strategy": "manual", # manual or object_hand_ratio    
    "frame_star": 0,
    "frame_end": 9999,
    "frame_interval": 1,
    "frame_number": 1000,   
    "obj_num": 6,
    "obj_1_cond_idx": 0,
    "obj_2_cond_idx": 0,
    "obj_3_cond_idx": 0,
    "obj_4_cond_idx": 0,    
    },
                                                 
}

