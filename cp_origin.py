import argparse
import os
import shutil

from tqdm import tqdm

def main(args):
    frame_star = args.frame_star
    frame_end = args.frame_end
    frame_interval = args.frame_interval

    origin_dir = args.data_path
    out_dir = args.output_dir

    for fidx in tqdm(range(frame_star, frame_end, frame_interval)):
        # Process each frame
        frame_name = f"{fidx:04d}"
        image_path = os.path.join(origin_dir, f"{frame_name}.png")
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist, skipping.")
            continue
        output_image_path = os.path.join(out_dir, f"{frame_name}.png")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        shutil.copy(image_path, output_image_path)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to data sequence",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to output directory",
        required=True,
    )
    parser.add_argument(
        "--frame_star",
        type=int,
        default=0,
        help="start frame index",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=10000,
        help="end frame index",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=20,
        help="frame interval",
    )

    args = parser.parse_args()
    main(args)