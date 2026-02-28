#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=true
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

declare -A device_sequences=(
  [0]="ABF12 BB12"
  [1]="ABF14 BB13"
  [2]="GPMF12 GPMF14"
  [3]="MC1 MDF12"
  [4]="MC4 MDF14"
  [5]="ShSu10 ShSu14"
  [6]="SM2 SM4 GSF12"
  [7]="SMu1 SMu40 GSF13"     

)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (
    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list hand_pose_postprocess \
    #   --process_list fit_hand_intrinsic fit_hand_trans  \
    #   --seq_list $sequences --rebuild --dataset_type ho3d

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list ho3d_obj_SAM3D_gen ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts \
    #   --seq_list $sequences --rebuild 


    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list ho3d_SAM3D_post_process \
    #   --seq_list $sequences --rebuild 

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
    #   --seq_list $sequences --rebuild 

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_data_preprocess_sam3d_neus \
    #   --seq_list $sequences --rebuild


    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_joint_opt \
      --seq_list $sequences --rebuild 

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_neus_init \
    #   --seq_list $sequences --rebuild  



    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_HY_gen hoi_pipeline_align_SAM3D_with_HY hoi_pipeline_3D_points_align_with_HY hoi_pipeline_HY_omni_gen hoi_pipeline_HY_to_SAM3D \
    #   --seq_list $sequences --rebuild  


    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_joint_opt \
      --seq_list $sequences --eval     
    
    echo "Running fit_hand on CUDA device $device with sequences: $sequences"
    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list eval_sum \
      --seq_list $sequences --rebuild



  ) &
done

wait

echo "All processes have completed successfully."
