import os
import shutil
import argparse
import json
import time
import random
import numpy as np
from confs.sequence_config import sequences



class run_wonder_hoi:
    def __init__(self, args, extras):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.code_dir = os.path.join(self.current_dir, "..")
        self.seq_list = args.seq_list
        self.execute_list = args.execute_list
        self.process_list = args.process_list
        self.dataset_dir = args.dataset_dir
        self.out_dir = args.out_dir
        self.rebuild = args.rebuild
        self.vis = args.vis
        self.extras = extras
        self.process_mapping = {
            "data_process": {
                "copy_image": self.copy_image,
                "crop_image": self.crop_image,
            },
            "vggt_process": {
                "vggt_colmap": self.vggt_colmap,
            },
        }

    def run(self):
        for exe in self.execute_list:
            for process in self.process_list:
                for seq in self.seq_list:
                    self.process_mapping[exe][process](seq, **self.extras)

    def get_seq_config(self, seq):
        for seq_config in sequences:
            if seq_config['id'] == 'default':
                seq_config_default = seq_config
            if seq_config['id'] == seq:
                return seq_config
        return seq_config_default



    def copy_image(self, seq, **extras):
        self.print_header('copy_image for {}'.format(seq))
        src_dir = os.path.join(extras['src_dir'], seq, 'mask_obj')
        dst_dir = os.path.join(self.dataset_dir, seq, 'images_origin')
        
        seq_config = self.get_seq_config(seq)
        # remove the dst_dir if it exists
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)
        for i, file in enumerate(sorted(os.listdir(src_dir))):
            if i < seq_config['frame_star']:
                continue
            if i > seq_config['frame_end']:
                break
            if i % seq_config['frame_interval'] == 0:
                if os.path.exists(os.path.join(src_dir, file)) and (file.endswith('.png') or file.endswith('.jpg')):
                    shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

    def crop_image(self, seq, **extras):
        self.print_header('crop_image for {}'.format(seq))
        src_dir = os.path.join(self.dataset_dir, seq, 'images_origin')
        dst_dir = os.path.join(self.dataset_dir, seq, 'images')
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)

        cmd = f"python crop_image.py --image_dir={src_dir} --output_dir={dst_dir}"
        print(cmd)
        os.system(cmd)

    def vggt_colmap(self, seq, **extras):
        self.print_header('vggt_colmap for {}'.format(seq))
        scene_dir = os.path.join(self.dataset_dir, seq)
        if self.rebuild:
            os.remove(f"{self.out_dir}/{seq}")
        os.makedirs(f"{self.out_dir}/{seq}", exist_ok=True)
        cmd = f"python demo_colmap.py --scene_dir={scene_dir} --use_ba --max_query_pts 200 --query_frame_num 10 --vis_thresh 0.20 --max_reproj_error 1000 --shared_camera --output_dir={self.out_dir}/{seq}"
        print(cmd)
        os.system(cmd)

    def print_header(self, process):
        header = f"========== start: {process} =========="
        print("-"*len(header))
        print(header)
        print("-"*len(header))                                                                  

def main(args, extras):
    # Convert extras list to dictionary
    extras_dict = {}
    for i in range(0, len(extras), 2):
        if i + 1 < len(extras):
            key = extras[i].lstrip('-')  # Remove leading dashes
            value = extras[i + 1]
            # Try to convert value to int or float if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            extras_dict[key] = value
    run_wonder_hoi(args, extras_dict).run()

if __name__ == "__main__":
    all_sequences = [
        "air_gun",
        "clamp",
        "cooking_shovel",
        "cube",
        "cup1",
        "cup2",
        "duck",
        "fire_fighting_car",
        "glass_cup",
        "hammer",
        "jep_car",
        "lufei",
        "mouse",
        "pitch",
        "plane",
        "scisors",
        "scisors_1",
        "spoon",
        "sprayer",
        "wrench",
        "bottle1",
        "bottle2",
        "drug_box",
        "trans_bottle1"    
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_list',
        choices=all_sequences + ['all'],
        help="Specify the sequence list. Use 'all' to select all sequences.",
        nargs='+',  # To accept multiple values in a list
        required=True  # This makes the argument mandatory
    )

    parser.add_argument('--execute_list', 
        choices=[
                "data_process",
                "vggt_process",        
                ], 
        help="Specify the execution option.", 
        nargs='+',  # To accept multiple values in a list
        required=False  # This makes the argument mandatory
    )
    parser.add_argument('--process_list', 
        choices=["copy_image", 
                "crop_image", 
                "vggt_colmap",
                ],
        help="Specify the process option.", 
        nargs='+',  # To accept multiple values in a list
        required=True  # This makes the argument mandatory
    )
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the process')
    parser.add_argument('--vis', action='store_true', help='Visualize the process')
    parser.add_argument('--dataset_dir', type=str, default="/home/simba/Documents/project/vggt/examples_ZED", help='Dataset directory')
    parser.add_argument('--out_dir', type=str, default="/home/simba/Documents/project/vggt/output", help='Output directory')
  
    args, extras = parser.parse_known_args()
    if 'all' in args.seq_list:
        args.seq_list = all_sequences

    main(args, extras)