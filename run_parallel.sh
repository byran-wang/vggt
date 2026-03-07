#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=true
export RU
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/data1/shibo/Documents/project/vggt_wenxuan/third_party/utils_simba:$PYTHONPATH"

declare -A device_sequences=(
  [2]="ABF12 GPMF12 GPMF14"
  [3]="ABF14 MC1 MDF12"
  [4]="MC4 MDF14 BB12"
  [5]="ShSu10 ShSu14 BB13"
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
    #   --process_list hoi_pipeline_data_preprocess \
    #   --seq_list $sequences --rebuild \
    #   --dataset_dir /data1/shibo/Documents/dataset/ZED_wenxuan

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_get_corres \
    #   --seq_list $sequences --rebuild \
    #   --dataset_dir /data1/shibo/Documents/dataset/ZED_wenxuan

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
      --process_list eval_sum hoi_pipeline_joint_opt_eval_vis eval_sum_vis \
      --seq_list $sequences --rebuild



    # 可視化
  ) &
done

wait

echo "All processes have completed successfully."
