#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=false
export DATASET=hoi4d
export RUN_ON_PC=true 
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export LD_LIBRARY_PATH="/home/simba/anaconda3/envs/vggsfm_tmp/lib:$LD_LIBRARY_PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"


declare -A device_sequences=(
  [0]="ZY20210800002_H2_C7_N41_S57_s04_T1"
)


current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (
    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list data_convert \
    #   --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer \
    #   --seq_list $sequences --rebuild 


    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list hand_pose_postprocess \
    #   --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot \
    #   --seq_list $sequences --rebuild    

      CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
        --execute_list hand_pose_postprocess \
        --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot \
        --seq_list $sequences --rebuild    

  ) &
done

wait

echo "All processes have completed successfully."


