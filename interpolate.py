import numpy as np
import argparse
import os

import hand_pose.slerp as slerp

def interpolate_hand_pose(args):
    """
    Interpolates hand pose parameters for a given sequence using spherical linear interpolation (SLERP).

    """
    # Define paths for input data
    init_pose_path = f'{args.dataset_dir}/hold_fit.init.npy'
    j2d_path = f'{args.dataset_dir}/j2d.full.npy'

    # Load data
    init_pose = np.load(init_pose_path, allow_pickle=True).item()
    j2d = np.load(j2d_path, allow_pickle=True).item()

    # Prepare interpolated data dictionary
    data_interp = {}

    # Process each hand
    for hand in ['right']:
        not_valid = np.isnan(j2d[f'j2d.{hand}'].reshape(-1, 21*2).mean(axis=1))
        outliers = np.where(not_valid)[0]
        num_frames = j2d[f'j2d.{hand}'].shape[0]
        key_frames = np.where(~not_valid)[0]

        # Perform SLERP interpolation
        if len(key_frames) >= 2:
            hand_interp = slerp.slerp_mano_params(outliers, num_frames, key_frames, init_pose[hand])
            hand_interp['is_valid'] = (~not_valid).astype(np.float32)
            data_interp[hand] = hand_interp

    # Define output path and save interpolated data
    # out_p = init_pose_path.replace('.init.', '.slerp.')
    out_p = f'{args.out_dir}/hold_fit.slerp.npy'

    np.save(out_p, data_interp)
    
    # Print the location of exported files
    print(f'Interpolated data saved to {out_p}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interpolate Hand Pose Data')
    parser.add_argument('--dataset_dir', type=str, default='', help='Path to the dataset directory')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory for interpolated data')
    args = parser.parse_args()

    interpolate_hand_pose(args)