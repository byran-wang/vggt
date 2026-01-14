# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Data loading and preprocessing functions for the COLMAP pipeline.
"""

import os
import glob
from pathlib import Path

import numpy as np
import torch

from vggt.utils.load_fn import load_and_preprocess_images_square, load_intrinsics, GEN_3D

from .geometry_utils import adjust_intrinsic_for_new_image_size
from .visualization_io import save_input_data


def get_image_list_ZED(args):
    """Get image list for ZED dataset format.

    Args:
        args: Arguments with scene_dir, min_frame_num, max_frame_num, frame_interval

    Returns:
        Tuple of (image_dir, image_path_list)
    """
    image_dir = Path(os.path.join(args.scene_dir, "images"))
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    image_path_list = sorted(image_path_list)
    image_path_list = image_path_list[args.min_frame_num:args.max_frame_num:args.frame_interval]

    return image_dir, image_path_list


def get_image_list_HO3D(args):
    """Get image list for HO3D dataset format.

    Args:
        args: Arguments with scene_dir, min_frame_num, max_frame_num, frame_interval

    Returns:
        Tuple of (image_dir, image_path_list)
    """
    image_dir = Path(os.path.join(args.scene_dir, "rgb"))
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    image_path_list = [path for path in image_path_list if path.endswith(".jpg") or path.endswith(".png")]
    image_path_list = sorted(image_path_list)
    image_path_list = image_path_list[args.min_frame_num:args.max_frame_num:args.frame_interval]

    return image_dir, image_path_list


def get_image_list(args):
    """Get image list based on dataset type.

    Args:
        args: Arguments with dataset_type and scene parameters

    Returns:
        Tuple of (image_dir, image_path_list)
    """
    if args.dataset_type == "ZED":
        return get_image_list_ZED(args)
    elif args.dataset_type == "HO3D":
        return get_image_list_HO3D(args)

    return None, []


def save_intrinsics(intrinsic, filepath):
    """Save intrinsic matrix to file.

    Args:
        intrinsic: Camera intrinsic matrix (3x3)
        filepath: Output file path
    """
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0, 2], intrinsic[1, 2]
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.savetxt(filepath, K, fmt="%.8f")


def load_images_and_intrinsics(args, device):
    """Load images, depth prior, masks, and camera intrinsics.

    Args:
        args: Arguments with scene parameters
        device: Target device (cuda/cpu)

    Returns:
        Tuple containing:
            - image_dir: Directory containing images
            - image_path_list: List of image file paths
            - base_image_path_list: List of base image filenames
            - images: Preprocessed image tensors
            - original_coords: Original image coordinates
            - image_masks: Image mask tensors
            - depth_prior: Depth prior maps
            - intrinsic: Camera intrinsic matrix
            - depth_conf: Depth confidence maps
            - vggt_fixed_resolution: VGGT model resolution (518)
            - img_load_resolution: Image loading resolution
            - gen_3d: Generated 3D model object
    """
    from PIL import Image

    image_dir, image_path_list = get_image_list(args)
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # check the frame index range
    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    print(f"Processing images in {image_dir} with the list  {base_image_path_list}")

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518

    img_load_resolution = Image.open(image_path_list[0]).size[0]

    images, original_coords, image_masks, depth_prior = load_and_preprocess_images_square(
        image_path_list,
        args,
        target_size=img_load_resolution,
        out_dir=f"{args.output_dir}/data_processed",
    )
    gen_3d = GEN_3D(f"{args.scene_dir}/align_mesh_image/{args.cond_index_raw:04d}")
    save_input_data(images, image_masks, depth_prior, gen_3d, image_path_list, f"{args.output_dir}/results/")

    images = images.to(device)
    original_coords = original_coords.to(device)
    image_masks = image_masks.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    intrinsic = load_intrinsics(os.path.join(args.scene_dir, "meta", f"{args.cond_index_raw:04d}.pkl"))
    intrinsic = adjust_intrinsic_for_new_image_size(intrinsic, original_coords, frame_idx=args.cond_index)

    depth_conf = np.ones_like(depth_prior)
    return (
        image_dir,
        image_path_list,
        base_image_path_list,
        images,
        original_coords,
        image_masks,
        depth_prior,
        intrinsic,
        depth_conf,
        vggt_fixed_resolution,
        img_load_resolution,
        gen_3d,
    )
