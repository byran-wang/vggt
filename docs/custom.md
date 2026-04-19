
# create a new config
- reference confs/seqence_config_template.py to create a new file confs/seqence_config_{your_name}.py. 
- and add a new dataset data_{your_name} in sequence_config.py

#### Note: following steps run on local pc, since they need the monitor.

```bash
# record ZED raw data
python run_wonder_hoi.py --execute_list data_read --process_list ZED_read_data  --seq_list $seq_list --rebuild


# in the terminal, export new dataset data_xxx by 
export DATASET=data_xxx

# parse left image, right image, intrinsic and zed depth from raw data with downsample 3
python run_wonder_hoi.py --execute_list data_convert --process_list ZED_parse_data  --seq_list $seq_list --rebuild --downsample 3

# Remember to check the depth *.ply files in ply_zed by Meshlab after convert_zed_depth_to_ply.
python run_wonder_hoi.py --execute_list data_convert --process_list convert_zed_depth_to_ply --seq_list $seq_list --rebuild # only for zed dataset

# get the hand and object mask by sam3
python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_get_obj_mask ho3d_get_hand_mask --seq_list $seq_list --rebuild

# Remember to check the depth *.ply files in ply_fs by Meshlab after get_depth_from_foundation_stereo.
python run_wonder_hoi.py --execute_list data_convert --process_list get_depth_from_foundation_stereo soft_link_depth --seq_list $seq_list --rebuild # only for zed dataset
```

#### Note: following steps run on local pc with 32 GB RAM.

```bash
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_filter_2D --seq_list $seq_list
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_filter_3D --seq_list $seq_list
```

#### Note: following steps can be run on server, after rsync the local dataset to server
```bash
export DATASET=data_{your_name}
cd /data1/shibo/Documents/project/vggt_in_the_wild
run_parallel_zed.sh # execute in tmux

# the final result can be find in output/metrics_summary

```
