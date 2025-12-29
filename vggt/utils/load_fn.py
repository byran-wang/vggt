# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import pickle
from pathlib import Path
import yaml
import sys
from tqdm import tqdm
_THIRD_PARTY_UTILS_SIMBA = str(Path(__file__).resolve().parents[2] / "third_party" / "utils_simba")
if _THIRD_PARTY_UTILS_SIMBA not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_UTILS_SIMBA)
from utils_simba.depth import get_depth, erode_depth_map_torch, gauss_filter_depth_map_torch

def load_intrinsics(intrinsics_path):
    try:
        with open(intrinsics_path, 'rb') as f:
            intrinsics = np.array(pickle.load(f)['camMat'])
        return intrinsics
    except FileNotFoundError as e:
        yaml_path = Path(intrinsics_path).with_name("intrinsics.yaml")
        if not yaml_path.exists():
            raise e

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        try:
            first_entry = next(iter(data.values()))
            fx, fy, cx, cy = first_entry["params"]
        except Exception as parse_error:
            raise ValueError(f"Invalid intrinsics format in {yaml_path}") from parse_error

        intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return intrinsics

class GEN_3D:
    def __init__(self, gen_3D_dir):
        self.gen_3D_path = Path(gen_3D_dir)
        self.mesh_path = Path(str(self.gen_3D_path).replace("align_mesh_image", "3D_gen")) / "white_mesh_remesh.obj"
        self.condition_image_path = self.gen_3D_path / "image.png"
        self.camera_path = self.gen_3D_path / "camera.json"
        self.depth_path = self.gen_3D_path / "depth.png"

    def get_cond_image(self, target_size=None):
        img = Image.open(self.condition_image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.BICUBIC)
        img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) float32 [0, 1]
        img = img.transpose(2, 0, 1)  # (3, H, W)
        return img

    def get_cond_mask(self):
        # mask is the alpha channel of condition image
        img = Image.open(self.condition_image_path)
        if img.mode == "RGBA":
            mask = np.array(img.getchannel('A'))
            mask = mask > 0  # (H, W) bool
            return mask
        else:
            # if no alpha channel, return all True mask
            width, height = img.size
            mask = np.ones((height, width), dtype=bool)
            return mask

    def get_cond_depth(self):
        return get_depth(str(self.depth_path))
    
    def get_cond_intrinsic(self):
        with open(self.camera_path, "r") as f:
            camera_data = yaml.safe_load(f)
        return np.array(camera_data["K"])
    
    def get_cond_extrinsic(self):
        with open(self.camera_path, "r") as f:
            camera_data = yaml.safe_load(f)
        w2c = np.array(camera_data["blw2cvc"])
        return w2c
    
    def get_mesh_path(self):
        return str(self.mesh_path)
    
    def save_aligned_pose(self, gen2obj):
        self.gen2obj = gen2obj

    def get_aligned_pose(self):
        return self.gen2obj



def load_and_preprocess_images_square_ZED(image_path_list, args, target_size=1024, out_dir=None):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.
        out_dir (str | Path, optional): If provided, saves/loads cached tensors in this directory.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    cache_path = None
    image_path_list = [str(p) for p in image_path_list]
    instance_id = args.instance_id
    if out_dir is not None:
        cache_dir = Path(out_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"preprocessed_{target_size}_{instance_id}.pt"
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu")
                if (
                    cached.get("target_size") == target_size
                    and cached.get("instance_id") == instance_id
                    and cached.get("image_paths") == image_path_list
                ):
                    print(f"[load_and_preprocess_images_square] Loaded cache from {cache_path}")
                    return (
                        cached["images"],
                        cached["original_coords"],
                        cached["masks"],
                        cached["depths"],
                    )
            except Exception as e:
                print(f"[load_and_preprocess_images_square] Failed to load cache {cache_path}: {e}")

    images = []
    masks = []
    depths = []
    original_coords = []  # Renamed from position_info to be more descriptive

    def _pil_to_chw_float_tensor(pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        width, height = pil_img.size
        data = torch.frombuffer(bytearray(pil_img.tobytes()), dtype=torch.uint8)
        data = data.view(height, width, 3).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
        return data

    def _pil_l_to_1hw_float_tensor(pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")
        width, height = pil_img.size
        data = torch.frombuffer(bytearray(pil_img.tobytes()), dtype=torch.uint8)
        data = data.view(height, width).to(dtype=torch.float32).div_(255.0)
        return data.unsqueeze(0)
    for image_path in tqdm(image_path_list, desc="Loading and preprocessing images", total=len(image_path_list)):
        # Open image
        img = Image.open(image_path)
        mask_path = Path(image_path.replace('images', 'masks'))
        depth_path = Path(image_path.replace('images', 'depth_fs'))

        if img.mode == "RGBA":
            mask = np.array(img.getchannel('A'))
        elif mask_path.exists():
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            mask[mask != instance_id] = 0      
        else:
            mask = np.ones((img.size[1], img.size[0]))
        mask = mask > 0  # (H, W) bool

        # If there's an alpha channel, blend onto black background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img)
            # Convert to RGB
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
            img_np = np.array(img, dtype=np.uint8)
            img_np = img_np * mask[..., None].astype(np.uint8)
            img = Image.fromarray(img_np, mode="RGB")
        
        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append([float(x1), float(y1), float(x2), float(y2), float(width), float(height)])

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Create a square mask aligned with the preprocessed image
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        square_mask_img = Image.new("L", (max_dim, max_dim), 0)
        square_mask_img.paste(mask_img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        square_mask_img = square_mask_img.resize((target_size, target_size), Image.Resampling.NEAREST)
        square_mask_tensor = _pil_l_to_1hw_float_tensor(square_mask_img) > 0.5

        # If a depth map exists, use it to further refine the mask (drop pixels with invalid depth).
        if depth_path.exists():
            depth_np = get_depth(str(depth_path))  # (H, W) float
            depth_np = depth_np * mask.astype(depth_np.dtype)

            depth_img = Image.fromarray(depth_np.astype(np.float32), mode="F")
            square_depth_img = Image.new("F", (max_dim, max_dim), 0.0)
            square_depth_img.paste(depth_img, (left, top))
            square_depth_img = square_depth_img.resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )
            depth_square = torch.frombuffer(bytearray(square_depth_img.tobytes()), dtype=torch.float32)
            depth_square = depth_square.view(target_size, target_size)
            depth_square = erode_depth_map_torch(depth_square, structure_size=2, d_thresh=0.01, frac_req=0.6)
            depth_square = erode_depth_map_torch(depth_square, structure_size=2, d_thresh=0.01, frac_req=0.6)

            square_mask_tensor = square_mask_tensor & (depth_square > 0.0).unsqueeze(0)

            depths.append(depth_square)

        # Convert to tensor
        images.append(_pil_to_chw_float_tensor(square_img))
        masks.append(square_mask_tensor.to(dtype=torch.float32))

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.tensor(original_coords, dtype=torch.float32)
    masks = torch.stack(masks)
    depths = torch.stack(depths)

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)
            masks = masks.unsqueeze(0)

    if cache_path is not None:
        torch.save(
            {
                "images": images.cpu(),
                "original_coords": original_coords.cpu(),
                "masks": masks.cpu(),
                "depths": depths.cpu(),
                "image_paths": image_path_list,
                "target_size": target_size,
                "instance_id": instance_id,
            },
            cache_path,
        )
        print(f"[load_and_preprocess_images_square] Saved cache to {cache_path}")

    return images, original_coords, masks, depths


def load_and_preprocess_images_square_HO3D(image_path_list, args, target_size=1024, out_dir=None):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.
        out_dir (str | Path, optional): If provided, saves/loads cached tensors in this directory.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    cache_path = None
    image_path_list = [str(p) for p in image_path_list]
    instance_id = args.instance_id
    if out_dir is not None:
        cache_dir = Path(out_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"preprocessed_{target_size}_{instance_id}.pt"
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu")
                if (
                    cached.get("target_size") == target_size
                    and cached.get("instance_id") == instance_id
                    and cached.get("image_paths") == image_path_list
                ):
                    print(f"[load_and_preprocess_images_square] Loaded cache from {cache_path}")
                    return (
                        cached["images"],
                        cached["original_coords"],
                        cached["masks"],
                        cached["depths"],
                    )
            except Exception as e:
                print(f"[load_and_preprocess_images_square] Failed to load cache {cache_path}: {e}")

    images = []
    masks = []
    depths = []
    original_coords = []  # Renamed from position_info to be more descriptive

    def _pil_to_chw_float_tensor(pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        width, height = pil_img.size
        data = torch.frombuffer(bytearray(pil_img.tobytes()), dtype=torch.uint8)
        data = data.view(height, width, 3).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
        return data

    def _pil_l_to_1hw_float_tensor(pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")
        width, height = pil_img.size
        data = torch.frombuffer(bytearray(pil_img.tobytes()), dtype=torch.uint8)
        data = data.view(height, width).to(dtype=torch.float32).div_(255.0)
        return data.unsqueeze(0)

    for image_path in tqdm(image_path_list, desc="Loading and preprocessing images", total=len(image_path_list)):
        # Open image
        img = Image.open(image_path)
        seq_id = Path(image_path).parents[1].name
        img_index = int(Path(image_path).stem.split('_')[-1])
        mask_path = Path(image_path).parents[3] / "masks_XMem" / seq_id / f"{img_index:05d}.png"
        depth_path = Path(image_path.replace('rgb', 'depth').replace('.jpg', '.png'))

        if img.mode == "RGBA":
            mask = np.array(img.getchannel('A'))
        elif mask_path.exists():
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            # mask[mask != instance_id] = 0      
        else:
            mask = np.ones((img.size[1], img.size[0]))
        mask = mask > 0  # (H, W) bool

        # If there's an alpha channel, blend onto black background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img)
            # Convert to RGB
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
            img_np = np.array(img, dtype=np.uint8)
            img_np = img_np * mask[..., None].astype(np.uint8)
            img = Image.fromarray(img_np, mode="RGB")
        
        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append([float(x1), float(y1), float(x2), float(y2), float(width), float(height)])

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Create a square mask aligned with the preprocessed image
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        square_mask_img = Image.new("L", (max_dim, max_dim), 0)
        square_mask_img.paste(mask_img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        square_mask_img = square_mask_img.resize((target_size, target_size), Image.Resampling.NEAREST)
        square_mask_tensor = _pil_l_to_1hw_float_tensor(square_mask_img) > 0.5

        # If a depth map exists, use it to further refine the mask (drop pixels with invalid depth).
        if depth_path.exists():
            depth_np = get_depth(str(depth_path))  # (H, W) float
            depth_np = depth_np * mask.astype(depth_np.dtype)

            depth_img = Image.fromarray(depth_np.astype(np.float32), mode="F")
            square_depth_img = Image.new("F", (max_dim, max_dim), 0.0)
            square_depth_img.paste(depth_img, (left, top))
            square_depth_img = square_depth_img.resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )
            depth_square = torch.frombuffer(bytearray(square_depth_img.tobytes()), dtype=torch.float32)
            depth_square = depth_square.view(target_size, target_size)
            depth_square = erode_depth_map_torch(depth_square, structure_size=2, d_thresh=0.01, frac_req=0.6)
            depth_square = erode_depth_map_torch(depth_square, structure_size=2, d_thresh=0.01, frac_req=0.6)

            square_mask_tensor = square_mask_tensor & (depth_square > 0.0).unsqueeze(0)

            depths.append(depth_square)

        # Convert to tensor
        images.append(_pil_to_chw_float_tensor(square_img))
        masks.append(square_mask_tensor.to(dtype=torch.float32))

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.tensor(original_coords, dtype=torch.float32)
    masks = torch.stack(masks)
    depths = torch.stack(depths)

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)
            masks = masks.unsqueeze(0)

    if cache_path is not None:
        torch.save(
            {
                "images": images.cpu(),
                "original_coords": original_coords.cpu(),
                "masks": masks.cpu(),
                "depths": depths.cpu(),
                "image_paths": image_path_list,
                "target_size": target_size,
                "instance_id": instance_id,
            },
            cache_path,
        )
        print(f"[load_and_preprocess_images_square] Saved cache to {cache_path}")

    return images, original_coords, masks, depths

def load_and_preprocess_images_square(image_path_list, args, target_size=1024, out_dir=None):
    if args.dataset_type == "ZED":
        return load_and_preprocess_images_square_ZED(image_path_list, args, target_size, out_dir)
    elif args.dataset_type == "HO3D":
        return load_and_preprocess_images_square_HO3D(image_path_list, args, target_size, out_dir)

    return None, None, None, None

def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto black background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (0, 0, 0, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
