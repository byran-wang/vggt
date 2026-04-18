#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=true
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

declare -A device_sequences=(
  [0]="AG1 CUB1 CUB2 CUP3 CUP4 DUC1 DUC2 GT1 HAM1 TG1 WC3 WC4"
  [1]="KNI1 MEC1 MED1 MOU1 PIN1 SCI1 SHP1 SPA1 SPN1 TAB1 TC3 TC4"  
  # [0]="CUB1 DUC1"
  # [1]="TC3 WC3"
  # [2]="CUP3 SCI1"
  # [3]="CUP4 SHP1"
  # [4]="GT1 SPA1"
  # [5]="KNI1 MEC1" 
  # [6]="MED1"
  # [7]="MOU1"    
)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (
    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list data_convert \
      --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer \
      --seq_list $sequences --rebuild 

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list hand_pose_postprocess \
      --process_list fit_hand_intrinsic fit_hand_trans \
      --seq_list $sequences --rebuild

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list ho3d_obj_SAM3D_filter_2D ho3d_obj_SAM3D_filter_3D \
      --seq_list $sequences --rebuild                   

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts ho3d_align_SAM3D_fp \
      --seq_list $sequences --rebuild

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list pipeline_sam3d_align_filter pipeline_sam3d_best_id \
      --seq_list $sequences --rebuild        

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list pipeline_sam3d_delete_unused \
      --seq_list $sequences --rebuild    

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list ho3d_SAM3D_post_process \
      --seq_list $sequences --rebuild        


    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
      --seq_list $sequences --rebuild 

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_joint_opt \
      --seq_list $sequences --rebuild 

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
      --seq_list $sequences --rebuild            

    
    echo "Running fit_hand on CUDA device $device with sequences: $sequences"
    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_eval_vis eval_sum_vis \
      --seq_list $sequences --rebuild



  ) &
done

wait

echo "All processes have completed successfully."
