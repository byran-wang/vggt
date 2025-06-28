seq_list="air_gun clamp cooking_shovel cube cup1 cup2 duck fire_fighting_car glass_cup hammer jep_car lufei mouse pitch plane scisors scisors_1 spoon sprayer wrench"
# seq_list="cube"
data_dir="/home/simba/Documents/project/vggt/examples_ZED"
out_dir="/home/simba/Documents/project/vggt/output"

# python run_vggt.py --execute_list data_process --process_list copy_data --seq_list $seq_list --dataset_dir $data_dir --src_dir /home/simba/Documents/dataset/WonderHOI/ZED/ --rebuild
# python run_vggt.py --execute_list data_process --process_list crop_image --seq_list $seq_list --dataset_dir $data_dir 
python run_vggt.py --execute_list vggt_process --process_list vggt_colmap --seq_list $seq_list --dataset_dir $data_dir --out_dir $out_dir --rebuild
