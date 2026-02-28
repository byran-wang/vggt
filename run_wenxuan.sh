eval "$(conda shell.bash hook)"
conda activate vggsfm_tmp

seq_list=all
python run_wonder_hoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans fit_hand_rot --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_gen --seq_list $seq_list --rebuild # on 48G 4090 computer
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_mask ho3d_align_SAM3D_pts --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list ho3d_SAM3D_post_process  --seq_list $seq_list --rebuild

on vggt Code  (interval take effect)
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --rebuild
python run_wonder_hoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --vis (optional)