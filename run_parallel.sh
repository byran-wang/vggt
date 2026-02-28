#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=false
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export LD_LIBRARY_PATH="/home/simba/anaconda3/envs/vggsfm_tmp/lib:$LD_LIBRARY_PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# declare -A device_sequences=(
#   [0]="CUP1 CUP2"
#   [1]="FFC1 FFC2"
#   [2]="MEC1 MEC2"
#   [3]="MED1 MED2"
#   [4]="MOU1 MOU2"
#   [5]="SPA1 SPA2"
#   [6]="TC1 TC2"
#   [7]="WC1 WC2"     

# )

declare -A device_sequences=(
  [0]="CUP1 CUP2 FFC1 FFC2 MEC1 MEC2 MED1 MED2 MOU1 MOU2 SPA1 SPA2 TC1 TC2 WC1 WC2"   
)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (
    # export RUN_ON_PC=true   
      CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_joint_opt \
    --seq_list $sequences --rebuild  \
    --dataset_dir /home/simba/Documents/dataset/ZED_wenxuan

    # 可視化
    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list hoi_pipeline_joint_opt \
    #   --seq_list $sequences --vis \
    #   --dataset_dir /home/simba/Documents/dataset/ZED_wenxuan

  ) &
done

wait

echo "All processes have completed successfully."
