eval "$(conda shell.bash hook)"
conda activate threestudio


##########################################################################
##################### BundleSDF Processing Pipeline ######################
##########################################################################
# seq_list="air_gun clamp cooking_shovel cube cup1 cup2 duck fire_fighting_car glass_cup hammer jep_car lufei mouse pitch plane scisors scisors_1 spoon sprayer wrench"
# seq_list="cooking_shovel"
# seq_list=all
# seq_list="cup2"
seq_list="MC1"
## data process.
python run_wonder_hoi.py --execute_list data_read --process_list realsense_read_data ZED_read_data --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list data_convert --process_list ZED_parse_data convert_depth_to_ply get_depth_from_foundation_stereo --seq_list $seq_list

## object process. Note: Run in the doccker container with the conda env py38
python run_wonder_hoi.py --execute_list obj_process --process_list obj_3D_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list align_mesh_image get_pts_observed align_corres align_pcs scale_3D_gen hunyuan_omni mesh2SDF --seq_list $seq_list --rebuild #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list estimate_obj_pose --seq_list $seq_list --rebuild #--vis

## hand process
python run_wonder_hoi.py --execute_list hand_pose_preprocess --process_list estimate_hand_pose --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_viewer --seq_list $seq_list --world_coordinate object --only_key_frame true

##########################################################################
##################### HOT3D Processing Pipeline ######################
##########################################################################
seq_list="P0002_2ea9af5b P0003_c701bd11 P0004_a59ab32e P0005_f493e970 P0011_451a7734 P0012_ca1f6626 P0020_1cb55e1b"
python run_wonder_hoi.py --execute_list data_convert --process_list hot3d_sync_to_local hot3d_get_undistorted_stereo hot3d_get_pose_in_cam hot3d_validate_pose_in_cam hot3d_cp_images hot3d_gen_meta hot3d_get_depth hot3d_get_mask obj_inpaint obj_3D_gen --seq_list $seq_list
python run_wonder_hoi.py --execute_list data_convert --process_list hot3d_estimate_hand_pose hot3d_interpolate_hamer --seq_list $seq_list
python run_wonder_hoi.py --execute_list obj_process --process_list hot3d_obj_pose --seq_list $seq_list

##########################################################################
##################### ZED Processing Pipeline ######################
##########################################################################
python run_wonder_hoi.py --execute_list obj_process --process_list obj_3D_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list align_mesh_image --seq_list $seq_list --rebuild #--vis
python run_wonder_hoi.py --execute_list data_convert --process_list zed_sync_data_to_local --seq_list $seq_list
python run_wonder_hoi.py --execute_list obj_process --process_list zed_joint_optimization --seq_list $seq_list #--vis

##########################################################################
##################### HO3D Processing Pipeline ######################
##########################################################################
python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_get_obj_mask ho3d_get_hand_mask --seq_list $seq_list
python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_inpaint --seq_list $seq_list
python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer --seq_list $seq_list # run on the compus 4090 pc
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot --seq_list $seq_list --rebuild --dataset_type ho3d # --vis
python run_wonder_hoi.py --execute_list data_convert --process_list hot3d_sync_hands_to_local --seq_list $seq_list # run on the compus 4090 pc

python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_3D_gen align_mesh_image --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_condition_id --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_gen --seq_list $seq_list --rebuild
# python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_post_opt_GS --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_SAM3D_post_process  --seq_list $seq_list --rebuild
# python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_keyframe_optimization --seq_list $seq_list --rebuild #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_eval_corres --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess_sam3d_neus --seq_list $seq_list --rebuild

# python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_get_corres --seq_list $seq_list --rebuild

python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_reg_remaining --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --vis
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --eval
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_neus_init --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum --seq_list $seq_list

python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_HY_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_align_SAM3D_with_HY --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_3D_points_align_with_HY --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_HY_omni_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_HY_to_SAM3D --seq_list $seq_list --rebuild

python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_gen_3d ho3d_align_gen_3d_omni --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_sdf_optimization --seq_list $seq_list --rebuild #--vis

python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_eval_intrinsic ho3d_eval_trans ho3d_eval_rot --seq_list $seq_list --vis
python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum_intrinsic eval_sum_trans eval_sum_rot --seq_list $seq_list

######################################## data transfer #########################################
cd /home/simba/Documents/dataset/BundleSDF && rsync -azvp --no-o --no-g -e "ssh -p 2026" HO3D_v3 root@180.184.148.133:/mnt/afs/shibo/Documents/dataset/BundleSDF/
