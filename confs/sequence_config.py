import os
RUN_ON_SERVER = os.getenv("RUN_ON_SERVER", "").lower() == "true"
dataset = os.getenv("DATASET", "").lower()

if RUN_ON_SERVER:
    home_dir = "/data1/shibo/"
    conda_dir = "/home/shibo/.conda/"
else:
    home_dir = os.getenv("RHOI_HOME", os.path.expanduser("~"))
    conda_dir = os.getenv("CONDA_DIR", f"{home_dir}/miniconda3")

vggt_code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda_dir = os.getenv("CUDA_DIR", os.getenv("CUDA_HOME", "/usr/local/cuda"))
output_baseline_dir = os.getenv("OUTPUT_BASELINE_DIR", f"{vggt_code_dir}/output_baseline")

if dataset == "zed":
    dataset_dir = f"{home_dir}/Documents/dataset/ZED_wenxuan/"
    dataset_type = "zed"
    from confs.sequence_config_zed import sequences, sequence_name_list

elif dataset == "ho3d":
    dataset_dir = f"{home_dir}/Documents/dataset/BundleSDF/HO3D_v3/train/"
    dataset_type = "ho3d"
    from confs.sequence_config_ho3d import sequences, sequence_name_list
elif dataset == "rs_zijian":
    dataset_dir = f"{home_dir}/Documents/dataset/rs_zijian/"
    dataset_type = "rs_zijian"
    from confs.sequence_config_rs_zijian import sequences, sequence_name_list
elif dataset == "zed_zijian":
    dataset_dir = f"{home_dir}/Documents/dataset/zed_zijian/"
    dataset_type = "zed_zijian"
    from confs.sequence_config_zed_zijian import sequences, sequence_name_list   
elif dataset == "zed_xy":
    dataset_dir = os.getenv("DATASET_DIR", f"{home_dir}/data/rhoi_zed/01")
    dataset_type = "zed_xy"
    from confs.sequence_config_zed_xy import sequences, sequence_name_list
elif dataset == "xper1m":
    dataset_dir = os.getenv(
        "DATASET_DIR", "/mnt/afs/xinyuan/run/rhoi_xper1m"
    )
    dataset_type = "xper1m"
    from confs.sequence_config_xper1m import sequences, sequence_name_list
else:
    raise ValueError(f"Please 'export DATASET=zed or ho3d or xper1m' at first")

