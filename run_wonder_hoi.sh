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

##########################################################################
##################### HO3D Processing Pipeline ######################
##########################################################################

####Note: following steps run on local pc, since they need the monitor to check.
# collect ZED raw data
python run_wonder_hoi.py --execute_list data_read --process_list ZED_read_data  --seq_list $seq_list --rebuild 
# pase left image, right image, intrinsic and zed depth from raw data with downsample 3
python run_wonder_hoi.py --execute_list data_convert --process_list ZED_parse_data  --seq_list $seq_list --rebuild --downsample 3
# Remember to check the depth *.ply files in ply_zed by Meshlab after convert_zed_depth_to_ply.
python run_wonder_hoi.py --execute_list data_convert --process_list convert_depth_to_ply --seq_list $seq_list --rebuild
# get the hand and object mask by sam3
python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_get_obj_mask ho3d_get_hand_mask --seq_list $seq_list --rebuild
####Note: following steps run on local pc with 32 GB RAM.
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_filter_2D --seq_list $seq_list #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_gen --seq_list $seq_list #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_filter_3D --seq_list $seq_list #--vis

# python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_post_opt_GS --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_mask --seq_list $seq_list #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_pts --seq_list $seq_list #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_fp --seq_list $seq_list --rebuild  #--vis

# Filter aligned frames by depth 3-axis coverage
python run_wonder_hoi.py --execute_list obj_process --process_list pipeline_sam3d_align_filter --seq_list $seq_list --rebuild  #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list pipeline_sam3d_delete_unused --seq_list $seq_list
python run_wonder_hoi.py --execute_list obj_process --process_list pipeline_sam3d_best_id --seq_list $seq_list  #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list pipeline_sam3d_best_id_sum --seq_list $seq_list

python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_SAM3D_post_process  --seq_list $seq_list --rebuild

####Note: following steps can be run on server, since they do not need the monitor. 
# Remember to check the depth *.ply files in ply_fs by Meshlab after get_depth_from_foundation_stereo.
python run_wonder_hoi.py --execute_list data_convert --process_list get_depth_from_foundation_stereo soft_link_depth --seq_list $seq_list --rebuild # only for zed dataset
# python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_inpaint --seq_list $seq_list
python run_wonder_hoi.py --execute_list data_convert --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic_vis --seq_list $seq_list
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_trans_vis --seq_list $seq_list
# python run_wonder_hoi.py --execute_list data_convert --process_list hot3d_sync_hands_to_local --seq_list $seq_list --rebuild




python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_by_foundation_pose --seq_list $seq_list --rebuild #--vis

# python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_keyframe_optimization --seq_list $seq_list --rebuild #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_eval_corres --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess_sam3d_neus --seq_list $seq_list --rebuild

# python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_get_corres --seq_list $seq_list --rebuild

python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt_global --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_reg_remaining --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --vis
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_neus_init --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_neus_global --seq_list $seq_list --rebuild #--export_only true


python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_HY_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_align_SAM3D_with_HY --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_3D_points_align_with_HY --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_HY_omni_gen --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_HY_to_SAM3D --seq_list $seq_list --rebuild

python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_gen_3d ho3d_align_gen_3d_omni --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_sdf_optimization --seq_list $seq_list --rebuild #--vis

python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_eval_intrinsic ho3d_eval_trans ho3d_eval_rot --seq_list $seq_list --vis
python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum_intrinsic eval_sum_trans eval_sum_rot --seq_list $seq_list

python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho --seq_list $seq_list --rebuild #--vis
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_eval --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_eval_vis --seq_list $seq_list --rebuild
# Visualize GT mesh, pose, and images in Rerun
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_eval_vis_gt --seq_list $seq_list

python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum --seq_list $seq_list
python run_wonder_hoi.py --execute_list obj_process --process_list eval_sum_vis --seq_list $seq_list --rebuild

######################################## baseline #########################################
python run_wonder_hoi.py --execute_list baseline --process_list foundation_pose_eval_vis --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list baseline --process_list bundle_sdf_eval_vis --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list baseline --process_list hold_eval_vis --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list baseline --process_list gt_eval_vis --seq_list $seq_list --rebuild

######################################## data transfer #########################################
cd /home/simba/Documents/dataset/BundleSDF && rsync -azvp --no-o --no-g -e "ssh -p 2026" HO3D_v3 root@180.184.148.133:/mnt/afs/shibo/Documents/dataset/BundleSDF/
