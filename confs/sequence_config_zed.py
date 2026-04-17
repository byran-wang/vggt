sequence_name_list = [
    ###### in-the-wild captured by ZED ##########
    "AG1",
    "CUB1",
    "CUP3",
    "DUC1",
    "GT1",
    "HAM1",
    "TG1",
    "WC3",
    "KNI1",
    "MEC1",
    "MED1",
    "MOU1",
    "PIN1",
    "SCI1",
    "SHP1",
    "SPA1",
    "SPN1",
    "TAB1",
    "TC3", 
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
    "AG1":
    {
    "cond_idx": 7,
    }, 
    "CUB1":
    {
    "cond_idx": 70,
    }, 
    "CUP3":
    {
    "cond_idx": 9,
    }, 
    "DUC1":
    {
    "cond_idx": 3,
    }, 
    "GT1":
    {
    "cond_idx": 13,
    }, 
    "TG1":
    {
    "cond_idx": 8,
    }, 
    "WC3":
    {
    "cond_idx": 193,
    }, 
    "KNI1":
    {
    "cond_idx": 24,
    }, 
    "MEC1":
    {
    "cond_idx": 84,
    }, 
    "MED1":
    {
    "cond_idx": 108,
    }, 
    "MOU1":
    {
    "cond_idx": 124,
    }, 
    "PIN1":
    {
    "cond_idx": 120,
    }, 
    "SCI1":
    {
    "cond_idx": 8,
    }, 
    "SHP1":
    {
    "cond_idx": 3,
    },                                                             
    "SPA1":
    {
    "cond_idx": 46,
    }, 
    "SPN1":
    {
    "cond_idx": 126,
    }, 
    "TAB1":
    {
    "cond_idx": 9999,
    }, 
    "TC3":
    {
    "cond_idx": 63,
    },                                                         
}

