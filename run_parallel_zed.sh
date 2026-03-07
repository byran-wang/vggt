#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=true
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

declare -A device_sequences=(
  [0]="CUB1 CUB2 DUC1 DUC2"
  [1]="TC3 TC4 WC3 WC4"
)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (
    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list data_convert \
      --process_list get_depth_from_foundation_stereo soft_link_depth \
      --seq_list $sequences --rebuild

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list data_convert \
      --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer \
      --seq_list $sequences --rebuild 

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list hand_pose_postprocess \
      --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot \
      --seq_list $sequences --rebuild            

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts ho3d_SAM3D_post_process \
      --seq_list $sequences --rebuild 


    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
      --seq_list $sequences --rebuild 

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_joint_opt \
      --seq_list $sequences --rebuild 

    
    echo "Running fit_hand on CUDA device $device with sequences: $sequences"
    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_joint_opt_eval_vis eval_sum_vis \
      --seq_list $sequences --rebuild



  ) &
done

wait

echo "All processes have completed successfully."
