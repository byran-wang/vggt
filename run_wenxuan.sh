cd /home/simba-1/Documents/project/vggt_wenxuan/generator && \
python scripts/align_hands_object.py --seq_name CUP2 --mode h_intrinsic \
--max_frame_num 9999 --frame_interval 3 \
--out_dir /home/simba/Documents/dataset/ZED_wenxuan/CUP2/

ln -sfn /home/simba/Documents/dataset/ZED_wenxuan /home/simba-1/Documents/project/vggt_wenxuan/generator/data

export sequences=CUP2
python run_wonder_hoi.py \
--execute_list hand_pose_postprocess \
--process_list fit_hand_trans fit_hand_rot \
--seq_list $sequences --rebuild \
--dataset_dir /home/simba/Documents/dataset/ZED_wenxuan\
--conda_type anaconda3

python -c "
import numpy as np
data = np.load('/home/simba/Documents/dataset/ZED_wenxuan/CUP2/hands/hold_fit.slerp.npy',
allow_pickle=True).item()
print(data.keys())
"