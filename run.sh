eval "$(conda shell.bash hook)"
conda activate vggsfm_tmp
scene_dir=examples_ZED/fire_fighting_car
out_dir=output_test
# Feedforward prediction only
# python demo_colmap.py --scene_dir=$scene_dir --conf_thres_value 3
# python demo_gradio.py --scene_dir=$scene_dir

# With bundle adjustment
# python crop_image.py --image_dir=$scene_dir/images_origin --output_dir=$scene_dir/images
rm -rf $out_dir
python demo_colmap.py --scene_dir=$scene_dir --use_ba --max_query_pts 200 --query_frame_num 10 --vis_thresh 0.20 --max_reproj_error 1000 --shared_camera --output_dir $out_dir --use_sfm --use_calibrated_intrinsic

