import os
import shutil
import time
from pathlib import Path
import pycolmap

def colmap_mvs(data_path, out_path):
    image_dir = Path(f"{data_path}/images")
    output_path = Path(f"{out_path}/ba/sparse")
    mvs_path = Path(f"{out_path}/ba/mvs")
    if os.path.exists(mvs_path):
        shutil.rmtree(mvs_path)
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    time.sleep(5)
    cmd = ""
    cmd += (
        f"export COLMAP_EXE_PATH=/usr/local/bin/ && "
        f"cd {mvs_path} && "
        f"$COLMAP_EXE_PATH/colmap patch_match_stereo "
        f"--workspace_path . "
        f"--workspace_format COLMAP "
        f"--PatchMatchStereo.max_image_size 2000 "
        f"--PatchMatchStereo.geom_consistency true && "
        f"$COLMAP_EXE_PATH/colmap stereo_fusion "
        f"--workspace_path . "
        f"--workspace_format COLMAP "
        f"--input_type geometric "
        f"--output_path ./fused.ply"
    )
    print(f"cmd: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory containing images.")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory where COLMAP results are stored.")
    args = parser.parse_args()
    colmap_mvs(args.data_dir, args.out_dir)