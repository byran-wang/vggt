#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=true
export RU
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/data1/shibo/Documents/project/vggt_wenxuan/third_party/utils_simba:$PYTHONPATH"

# declare -A device_sequences=(
#   [0]="ABF12 BB12"
#   [1]="ABF14 BB13"
#   [2]="GPMF12 GPMF14"
#   [3]="MC1 MDF12"
#   [4]="MC4 MDF14"
#   [5]="ShSu10 ShSu14"
#   [6]="SM2 SM4 GSF12"
#   [7]="SMu1 SMu40 GSF13"     

# )

# PMC1 效果太差
declare -A device_sequences=(
  [0]="TSC1 TSC2"
  [1]="PMC2"
)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (                                                         
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
      --seq_list $sequences --rebuild  \
      --dataset_dir /data1/shibo/Documents/dataset/ZED_wenxuan

    # 可視化
  ) &
done

wait

echo "All processes have completed successfully."
