import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial

# Add project directory to path
import sys
sys.path = ['../code'] + sys.path

class ImageProcessor:
    def __init__(self, seq_name, out_name, rgba_format):
        self.seq_name = seq_name
        self.out_name = out_name
        self.rgba_format = rgba_format
        from src.utils.const import SEGM_IDS
        self.SEGM_IDS = SEGM_IDS
        
    def process_batch(self, batch_files, process_type="object"):
        """Process a batch of images"""
        for rgb_p, mask_p in batch_files:
            # Load images
            rgb_img = Image.open(rgb_p)
            mask_img = Image.open(mask_p)
            
            # Convert to numpy arrays
            rgb_np = np.array(rgb_img)[:, :, :3]
            mask_np = np.array(mask_img)
            
            # Determine mask condition based on process type
            if process_type == "object":
                condition = mask_np == self.SEGM_IDS["object"]
            else:  # hand
                condition = (mask_np == self.SEGM_IDS["right"]) | (mask_np == self.SEGM_IDS["left"])
            
            # Process mask
            processed_mask = np.zeros_like(mask_np)
            processed_mask[condition] = 255
            if len(processed_mask.shape) == 3:
                processed_mask = processed_mask[:, :, 0]
            
            # Apply mask to RGB
            rgb_np[processed_mask == 0] = 255
            
            # Add alpha channel if needed
            if self.rgba_format:
                rgb_np = np.concatenate([rgb_np, processed_mask[:, :, None]], axis=2)
            
            # Determine output path
            if process_type == "object":
                out_dir = self.out_name
            else:
                out_dir = f"{self.out_name}_hand"
            
            out_p = rgb_p.replace("/images/", f"/{out_dir}/")
            Path(out_p).parent.mkdir(parents=True, exist_ok=True)
            
            # Save result
            Image.fromarray(rgb_np).save(out_p)

def object_mask_gen(args):
    seq_name = args.seq_name
    out_name = args.out_name
    rgba_format = args.rgba_format
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    print(f"Processing {seq_name}")
    
    # Get sorted file lists
    rgb_ps = sorted(glob(f"data/{seq_name}/processed/images/*.png"))
    mask_ps = sorted(glob(f"data/{seq_name}/processed/masks/*.png"))
    assert len(rgb_ps) == len(mask_ps)
    
    # Create processor instance
    processor = ImageProcessor(seq_name, out_name, rgba_format)
    
    # Create batches
    file_pairs = list(zip(rgb_ps, mask_ps))
    batches = [file_pairs[i:i + batch_size] for i in range(0, len(file_pairs), batch_size)]
    
    # Process object masks
    print("Processing objects...")
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(processor.process_batch, batches), total=len(batches)))
    
    # Process hand masks
    print("Processing hands...")
    with Pool(num_workers) as pool:
        # Use partial to specify the process_type parameter
        process_hand_batch = partial(processor.process_batch, process_type="hand")
        list(tqdm(pool.imap(process_hand_batch, batches), total=len(batches)))
    
    print('Done!')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, required=True, help="Sequence name to process")
    parser.add_argument("--out_name", type=str, required=True, help="Output directory name")
    parser.add_argument("--rgba_format", action="store_true", help="Generate images with RGBA format")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images to process in each batch")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Number of parallel workers for processing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    object_mask_gen(args)