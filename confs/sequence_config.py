import os
RUN_ON_SERVER = os.getenv("RUN_ON_SERVER", "").lower() == "true"
dataset = os.getenv("DATASET", "").lower()

if RUN_ON_SERVER:
    home_dir = "/data1/shibo/"
    conda_dir = "/home/shibo/.conda/"
else:
    home_dir = os.path.expanduser("~")
    conda_dir = f"{home_dir}/miniconda3"

vggt_code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if dataset == "zed":
    dataset_dir = f"{home_dir}/Documents/dataset/ZED_wenxuan/"
    dataset_type = "zed"
    from confs.sequence_config_zed import sequences, sequence_name_list

elif dataset == "ho3d":
    dataset_dir = f"{home_dir}/Documents/dataset/BundleSDF/HO3D_v3/train/"
    dataset_type = "ho3d"
    from confs.sequence_config_ho3d import sequences, sequence_name_list
else:
    raise ValueError(f"Please 'export DATASET=zed or ho3d' at first")           

