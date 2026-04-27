# Self-captured ZED data under scene "01" (DATASET=zed_xy).
# dataset_dir already points at {home}/data/rhoi_zed/01, so seq names are
# just the ordinal "NNN" (1-indexed by recording timestamp).
# The original HD720_SN10027197_*.svo2 filenames are preserved in
# {dataset_dir}/filename_mapping.tsv

sequence_name_list = [f"{i:03d}" for i in range(1, 56)]

sequences = {
    "default": {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "cond_select_strategy": "auto",  # manual or auto
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
# Auto-populate per-seq entries (cond_idx to be tuned manually later).
for _i in range(1, 56):
    sequences[f"{_i:03d}"] = {"cond_idx": 0}
