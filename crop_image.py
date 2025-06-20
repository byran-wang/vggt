import argparse
import os
import glob
from PIL import Image
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    
    parser.add_argument("--crop_size", type=int, default=720, help="Random seed for reproducibility")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the cropped images")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Get image paths and preprocess them
    image_dir = args.image_dir
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    
    # center crop the image to the crop size
    # show the cropped progress    
    for image_path in tqdm(image_path_list, desc="Cropping images", total=len(image_path_list)):
        image = Image.open(image_path)
        width, height = image.size
        assert width >= args.crop_size and height >= args.crop_size and width % 2 == 0 and height % 2 == 0
        start_x = (width - args.crop_size) // 2
        start_y = (height - args.crop_size) // 2
        image = image.crop((start_x, start_y, start_x + args.crop_size, start_y + args.crop_size))
        image.save(os.path.join(args.output_dir, os.path.basename(image_path)))
    
    print(f"Cropped images in {args.image_dir} to {args.output_dir}")
