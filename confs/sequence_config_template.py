sequence_name_list = [
    ###### in-the-wild captured by ZED ##########
    "seq_name", # modify it
]

sequences = {
    "default":
    {
    "geometry_poor_frames": [],
    "cond_idx": 0,
    "cond_select_strategy": "auto", # manual or auto    
    "frame_star": 0,
    "frame_end": 9999,
    "frame_interval": 1,
    "frame_number": 1000,    
    },
    "seq_name":
    {
    "cond_idx": 7, # only apply when cond_select_strategy is manual
    },                            
}

