#!/bin/bash

# Declare an associative array mapping CUDA devices to their respective sequence lists
export RUN_ON_SERVER=true
# nerf acc need to export the cuda path
export PATH="/usr/local/cuda-11.8:/usr/local/cuda-11.8/bin/:$PATH"
export CUDA_PATH='/usr/local/cuda-11.8'
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

declare -A device_sequences=(
  [0]="MDF12 MC1"
  [1]="MDF14 MC4"
  [2]="BB12 ShSu10"
  [3]="BB13 ShSu12"
  [4]="GSF12 SMu40"
  [5]="GSF13 SMu1"
  [6]="GPMF12 ABF12 SM2"
  [7]="GPMF14 ABF14 SM4"   


  # [2]="MDF12 ShSu10"
  # [3]="MDF14 ShSu12"
  # [4]="SMu40 GSF12 GPMF14"
  # [5]="SMu1 GSF13 BB12"
  # [6]="ABF12 ABF14 SM2 SM4"
  # [7]="BB13 GPMF12 MC1 MC4"  


  # ┌───────┬────────────────────────────────────────┬──────────────┐                                                                                                                                                                                                                
  # │ Group │               Sequences                │ Total Frames │                                                                                                                                                                                                                
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G1    │ MDF12 (562) + MC4 (180)                │ 742          │
  # ├───────┼────────────────────────────────────────┼──────────────┤
  # │ G2    │ MDF14 (562) + MC4 (180)                │ 742          │
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G3    │ ShSu10 (371) + BB12 (322)              │ 693          │
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G4    │ ShSu12 (371) + BB13 (323)              │ 694          │
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G5    │ SMu40 (400) + GSF12 (279)              │ 679          │
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G6    │ SMu1 (360) + GSF13 (319)               │ 679          │
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G7    │ ABF12 (277) + GPMF12 (220) + SM2 (181) │ 678          │
  # ├───────┼────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                                
  # │ G8    │ ABF14 (277) + GPMF14 (219) + SM4 (181) │ 677          │
  # └───────┴────────────────────────────────────────┴──────────────┘                                                                                                                                                                                                                
                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                   
  # ┌───────┬───────────────────────────────────────────────────┬──────────────┐                                                                                                                                                                                                     
  # │ Group │                     Sequences                     │ Total Frames │                                                                                                                                                                                                     
  # ├───────┼───────────────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                     
  # │ G1    │ MDF12 (562) + ShSu10 (371)                        │ 933          │                                                                                                                                                                                                     
  # ├───────┼───────────────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                     
  # │ G2    │ MDF14 (562) + ShSu12 (371)                        │ 933          │                                                                                                                                                                                                     
  # ├───────┼───────────────────────────────────────────────────┼──────────────┤
  # │ G3    │ SMu40 (400) + GSF12 (279) + GPMF14 (219)          │ 898          │                                                                                                                                                                                                     
  # ├───────┼───────────────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                     
  # │ G4    │ SMu1 (360) + GSF13 (319) + BB12 (322)             │ 1001         │
  # ├───────┼───────────────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                     
  # │ G5    │ ABF12 (277) + ABF14 (277) + SM2 (181) + SM4 (180) │ 915          │
  # ├───────┼───────────────────────────────────────────────────┼──────────────┤                                                                                                                                                                                                     
  # │ G6    │ BB13 (323) + GPMF12 (220) + MC1 (181) + MC4 (180) │ 904          │
  # └───────┴───────────────────────────────────────────────────┴──────────────┘                                                                                                                                                                                                     
                  

)

current_dir=$(pwd)


# Iterate over each CUDA device and its corresponding sequences
for device in "${!device_sequences[@]}"; do
  sequences=${device_sequences[$device]}
  
  (

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list data_convert \
    #   --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer  \
    #   --seq_list $sequences --rebuild

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list hand_pose_postprocess \
    #   --process_list fit_hand_intrinsic fit_hand_trans  \
    #   --seq_list $sequences --rebuild

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list ho3d_obj_SAM3D_filter_3D \
    #   --seq_list $sequences --rebuild 

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts ho3d_align_SAM3D_fp \
    #   --seq_list $sequences --rebuild 


    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list pipeline_sam3d_align_filter pipeline_sam3d_best_id \
    #   --seq_list $sequences --rebuild 
   
    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list pipeline_sam3d_delete_unused \
    #   --seq_list $sequences --rebuild    

    # CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
    #   --execute_list obj_process \
    #   --process_list ho3d_SAM3D_post_process \
    #   --seq_list $sequences --rebuild  

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
      --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
      --seq_list $sequences --rebuild       



    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list hoi_pipeline_eval hoi_pipeline_eval_vis \
      --seq_list $sequences --rebuild     
    
    CUDA_VISIBLE_DEVICES=$device python run_wonder_hoi.py \
      --execute_list obj_process \
      --process_list eval_sum eval_sum_vis \
      --seq_list $sequences --rebuild



  ) &
done

wait

echo "All processes have completed successfully."
