sequence_name_list = [
    "ABF12",
    "ABF14",
    "GPMF12",
    "GPMF14",    
    "MC1",
    "MC4",
    "MDF12",
    "MDF14",
    "ShSu10",
    "ShSu12",
    # "ShSu14",
    "SM2",
    "SM4",
    "SMu1",
    "SMu40",
    "BB12",
    "BB13", 
    "GSF12",
    "GSF13",       
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
    "frame_interval": 5,
    "frame_number": 1000,   
    "obj_num": 6,
    "obj_1_cond_idx": 0,
    "obj_2_cond_idx": 0,
    "obj_3_cond_idx": 0,
    "obj_4_cond_idx": 0,    
    },
                    
    ########## HO3D Sequences ##########
    "ABF12":
        {
        "cond_idx": 1015,
        },     
    "ABF14":
        {
        "cond_idx": 1345,
        },
    "GPMF12":
        {
        "cond_idx": 245,
        }, 
    "GPMF14":
        {
        "cond_idx": 940,
        },
    "MC1":
        {
        "cond_idx": 690,
        }, 
    "MC4":
        {
        "cond_idx": 520,
        },
    "MDF12":
        {
        "cond_idx": 1470,
        }, 
    "MDF14":
        {
        "cond_idx": 1265,
        },         
    "ShSu10":
        {
        "cond_idx": 1680,
        },
    "ShSu12":
        {
        "cond_idx": 550,
        },          
    "ShSu14":
        {
        "cond_idx": 870,
        },          
    "SM2":
        {
        "cond_idx": 470,
        },
    "SM4":
        {
        "cond_idx": 665,
        },          
     "SMu1":
        {
        "cond_idx": 660,
        }, 
    "SMu40":
        {
        "cond_idx": 465,
        },
    "BB12":
        {
        "cond_idx": 510,
        },
    "BB13":
        {
        "cond_idx": 1185,
        },                                      
    "GSF12":
        {
        "cond_idx": 125,
        },
    "GSF13":
        {
        "cond_idx": 340,
        },        

}
