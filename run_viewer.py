import os
# scene_dir=examples_ZED/fire_fighting_car

current_dir=os.path.dirname(os.path.abspath(__file__))
data_dir=os.path.join(current_dir, 'examples_ZED')
reconstruction_dir=os.path.join(current_dir, 'examples_ZED')
reconstruction_dir=os.path.join(current_dir, 'output_backup','[102][6_20][100][SfM]')
reconstruction_dir=os.path.join(current_dir, 'output_test_white_bg')
# reconstruction_dir=os.path.join(current_dir, 'output_backup','[104][6_20][102][PIN_HOLE][fixed_calibrated_intrinsic]')
# scenes=['fire_fighting_car']
# scenes=os.listdir(data_dir)
# scenes=["fire_fighting_car"]
scenes=["cooking_shovel"]

# Feedforward prediction only
# python demo_colmap.py --scene_dir=$scene_dir --conf_thres_value 3
# python demo_gradio.py --scene_dir=$scene_dir

# With bundle adjustment
# python crop_image.py --image_dir=$scene_dir/images_origin --output_dir=$scene_dir/images
# rm -rf $out_dir
# python demo_colmap.py --scene_dir=$scene_dir --use_ba --max_query_pts 200 --query_frame_num 10 --vis_thresh 0.20 --max_reproj_error 1000 --shared_camera --output_dir $out_dir --use_sfm --use_calibrated_intrinsic
python_path='/home/simba/anaconda3/envs/threestudio/bin/python'
for scene in scenes:
    scene_dir=f"{data_dir}/{scene}"
    out_dir=os.path.join(reconstruction_dir, scene)
    os.system(f'cd viewer && {python_path} viewer.py --sequence_folder {scene_dir} --reconstruction_folder {out_dir}/sfm/sparse/ --world_coordinate object')
    # os.system(f'cd viewer && {python_path} viewer.py --sequence_folder {scene_dir} --reconstruction_folder {out_dir}/colmap_{scene}.0/sfm_superpoint+superglue --world_coordinate object')
    


