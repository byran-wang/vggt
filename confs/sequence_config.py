import os
RUN_ON_SERVER = os.getenv("RUN_ON_SERVER", "").lower() == "true"

if RUN_ON_SERVER:
    home_dir = "/data1/shibo/"
    conda_dir = "/home/shibo/.conda/"
else:
    home_dir = os.path.expanduser("~")
    conda_dir = f"{home_dir}/miniconda3"

vggt_code_dir = f"{home_dir.rstrip('/')}/Documents/project/vggt_wenxuan"
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

    ########## HOT3D Sequences ##########
    "P0002_2ea9af5b":
        {
        "frame_interval": 5,
        },
    "P0003_c701bd11":
        {
        "frame_star": 200,
        "frame_end": 800,
        "frame_interval": 5,
        },
    "P0004_a59ab32e":
        {
        "frame_interval": 5,
        },
    "P0005_f493e970":
        {
        "frame_interval": 5,
        },        
   
    "P0011_451a7734":
        {
        "frame_interval": 5,
        }, 
    "P0012_ca1f6626":
        {
        "frame_interval": 5,
        # obj_id = 4
        # "frame_star": 1950,
        # "frame_end": 2380,
        # obj_id = 5
        "frame_star": 2390,
        "frame_end": 2750,
        "cond_idx": 5,
        "obj_id_to_instance": {1: "hand_left", 2: "hand_right", 3: "obj_xxx", 4: "obj_carton_milk_125863066770940", 5: "obj_birdhouse_toy_195041665639898"},
        },   
    "P0020_1cb55e1b":
        {
        "frame_interval": 5,
        },              
                    
    ########## HO3D Sequences ##########
    "ABF12":
        {
        "cond_idx": 120,
        },     
    "ABF14":
        {
        "cond_idx": 0,
        },
    "GPMF12":
        {
        "cond_idx": 239,
        }, 
    "GPMF14":
        {
        "cond_idx": 1060,
        },
    "MC1":
        {
        "cond_idx": 487,
        }, 
    "MC4":
        {
        "cond_idx": 155,
        },
    "MDF12":
        {
        "cond_idx": 1485,
        }, 
    "MDF14":
        {
        "cond_idx": 605,
        },         
    "ShSu10":
        {
        "cond_idx": 518,
        },  
    "ShSu14":
        {
        "cond_idx": 870,
        },          
    "SM2":
        {
        "cond_idx": 18,
        },
    "SM4":
        {
        "cond_idx": 0,
        },          
     "SMu1":
        {
        "cond_idx": 1738,
        }, 
    "SMu40":
        {
        "cond_idx": 400,
        },
    "BB12":
        {
        "cond_idx": 770,
        },
    "BB13":
        {
        "cond_idx": 1024,
        },                                      
    "GSF12":
        {
        "cond_idx": 385,
        },
    "GSF13":
        {
        "cond_idx": 755,
        },        
                       
                     




    ########## WonderHOI Sequences ##########
    "cooking_shovel":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 59,
        "frame_end": 10000,      
        },
    "cup2":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 0,
        "frame_end": 10000,
        },
    "spoon":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 344,
        "frame_end": 10000,
        },
    "hammer":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 59,
        "frame_end": 10000,
        },
    "clamp":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 43,
        "frame_end": 10000,
        },
    "scisors_1":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 23,
        "frame_end": 10000,
        },
    "scisors":
        {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "frame_star": 22,
        "frame_end": 10000,
        }
}

if RUN_ON_SERVER:
    sequences["default"]["frame_interval"] = 5
    sequences["default"]["frame_number"] = 1000
