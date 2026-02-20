#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists

export RUN_ON_SERVER=true

declare -A device_sequences=(
  [0]="BB12 BB13"
  [1]="ABF12 ABF14"
  [2]="GPMF12 GPMF14"
  [3]="MC1 MC4"
  [4]="MDF12 MDF14 "
  [5]="ShSu10 ShSu14 "
  [6]="SM2 SM4 GSF12"
  [7]="SMu1 SMu40 GSF13"     

)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres \
    #   --seq_list $sequences --rebuild 


    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_joint_opt \
      --seq_list $sequences --rebuild 

    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_neus_init \
      --seq_list $sequences --rebuild  



    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_HY_gen hoi_pipeline_align_SAM3D_with_HY hoi_pipeline_3D_points_align_with_HY hoi_pipeline_HY_omni_gen hoi_pipeline_HY_to_SAM3D \
      --seq_list $sequences --rebuild  


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
