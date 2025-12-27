eval "$(conda shell.bash hook)"
conda activate vggsfm_tmp

current_dir=$(pwd)
data_dir=$current_dir/examples_ZED
# scenes=$(ls ${data_dir})
scenes="cup2"
# scenes="hammer"
# Feedforward prediction only
# python demo_colmap.py --scene_dir=$scene_dir --conf_thres_value 3
# python demo_gradio.py --scene_dir=$scene_dir

# With bundle adjustment

for scene in ${scenes}; do
    echo "Processing scene: $scene"
    scene_dir=$current_dir/examples_ZED/$scene
    out_dir=$current_dir/output/$scene
    raw_dir="/home/simba/Documents/dataset/WonderHOI/ZED/${scene}/"
    frame_interval=5
    # python cp_origin.py --data_path="${raw_dir}/mask_obj" --output_dir="${scene_dir}/mask_obj_origin" --frame_interval=${frame_interval}
    # python crop_image.py --image_dir=$scene_dir/mask_obj_origin --output_dir=$scene_dir/images --meta_path="${scene_dir}/meta_origin/0000.pkl"
    # python cp_origin.py --data_path="${raw_dir}/depth_fs" --output_dir="${scene_dir}/depth_fs_origin" --frame_interval=${frame_interval}
    # python crop_image.py --image_dir=$scene_dir/depth_fs_origin --output_dir=$scene_dir/depth_fs --meta_path="${scene_dir}/meta_origin/0000.pkl"
    # rm -rf $out_dir
    rm -rf ${out_dir}/results
    python demo_colmap.py --scene_dir=$scene_dir --max_query_pts 200 --query_frame_num 0 --vis_thresh 0.40 --max_reproj_error 3 --shared_camera \
            --output_dir $out_dir  --use_calibrated_intrinsic --max_frames 100 #--use_sfm
    # python mvs.py --data_dir ${scene_dir} --out_dir ${out_dir}
done
# rm -rf $out_dir
# python demo_colmap.py --scene_dir=$scene_dir --use_ba --max_query_pts 200 --query_frame_num 10 --vis_thresh 0.20 --max_reproj_error 1000 --shared_camera --output_dir $out_dir --use_sfm --use_calibrated_intrinsic
# python demo_colmap.py --scene_dir=$scene_dir --use_ba --max_query_pts 200 --query_frame_num 1 --vis_thresh 0.20 --max_reproj_error 1000 --shared_camera --output_dir $out_dir --use_sfm --use_calibrated_intrinsic

# cd viewer && python viewer.py --sequence_folder $scene_dir --reconstruction_folder $out_dir/sfm/ --world_coordinate object