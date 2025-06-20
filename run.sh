eval "$(conda shell.bash hook)"
conda activate vggsfm_tmp
scene_dir=examples/fire_fighting_car
# Feedforward prediction only
# python demo_colmap.py --scene_dir=$scene_dir

# With bundle adjustment
# python crop_image.py --image_dir=$scene_dir/images_origin --output_dir=$scene_dir/images
python demo_colmap.py --scene_dir=$scene_dir --use_ba --max_query_pts 200 --query_frame_num 10 --vis_thresh 0.20 --max_reproj_error 1000 --shared_camera