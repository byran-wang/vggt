import argparse
import os
import glob
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    
    parser.add_argument("--crop_size", type=int, default=700, help="Random seed for reproducibility")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the cropped images")
    return parser.parse_args()

def load_meta(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta

if __name__ == "__main__":
    args = parse_args()
    # Get image paths and preprocess them
    image_dir = args.image_dir
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    meta_origin_path = os.path.join(image_dir.replace("images_origin", "meta_origin"), "0000.pkl")
    meta_origin = load_meta(meta_origin_path)
    intrinsics_origin = np.array(meta_origin['camMat'])

    # crop the intrinsics
    cx, cy = intrinsics_origin[0, 2], intrinsics_origin[1, 2]

    # crop the image to the crop size from the principal point
    # show the cropped progress    
    for image_path in tqdm(image_path_list, desc="Cropping images", total=len(image_path_list)):
        image = Image.open(image_path)
        width, height = image.size
        start_x = (cx - args.crop_size / 2).round().astype(int)
        start_y = (cy - args.crop_size / 2).round().astype(int)
        end_x = start_x + args.crop_size
        end_y = start_y + args.crop_size
        # assert with error message
        assert width >= end_x and height >= end_y and start_x >= 0 and start_y >= 0, f"Image {image_path} with size {width}x{height} is too small to crop to {args.crop_size}x{args.crop_size}"

        image = image.crop((start_x, start_y, end_x, end_y))
        image.save(os.path.join(args.output_dir, os.path.basename(image_path)))

    # get the new cx and cy
    cx_new = cx - start_x
    cy_new = cy - start_y

    # crop the intrinsics
    intrinsics_new = intrinsics_origin.copy()
    intrinsics_new[0, 2] = cx_new
    intrinsics_new[1, 2] = cy_new

    meta_new = meta_origin.copy()
    meta_new['camMat'] = intrinsics_new.tolist()

    meta_new_path = meta_origin_path.replace("meta_origin", "meta")
    os.makedirs(os.path.dirname(meta_new_path), exist_ok=True)
    with open(meta_new_path, 'wb') as f:
        pickle.dump(meta_new, f)
    
    
    print(f"Cropped images in {args.image_dir} to {args.output_dir}")
