import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images_square_HO3D

# DINO normalization constants
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# DINO Ranking
# -----------------------------------------------------------------------------

def generate_rank_by_dino(
    images, masks, image_size=336, model_name="dinov2_vitb14_reg", device="cuda"
):
    """
    Generate a ranking of frames using DINO ViT features.

    Frames are ranked by their total similarity to all other frames,
    with the most representative (common) frame ranked first.
    Only features within the mask are used, discarding background.

    Args:
        images: Tensor of shape (S, 3, H, W) with values in range [0, 1]
        masks: Tensor of shape (S, 1, H, W) or (S, H, W) with object masks
        image_size: Size to resize images to before processing
        model_name: Name of the DINO model to use
        device: Device to run the model on

    Returns:
        List of frame indices ranked by their representativeness (most common first)
    """
    num_frames = images.shape[0]

    # Resize images and masks to the target size
    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)

    # Prepare masks
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)  # (S, H, W) -> (S, 1, H, W)
    masks = F.interpolate(masks.float(), (image_size, image_size), mode="nearest")
    masks = masks > 0.5  # (S, 1, H, W)

    # Load DINO model
    dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
    dino_v2_model.eval()
    dino_v2_model = dino_v2_model.to(device)

    # Get patch size from model (typically 14 for ViT-B/14)
    patch_size = dino_v2_model.patch_size if hasattr(dino_v2_model, 'patch_size') else 14
    num_patches_per_side = image_size // patch_size

    # Resize masks to patch grid size
    patch_masks = F.interpolate(
        masks.float(), (num_patches_per_side, num_patches_per_side), mode="nearest"
    )  # (S, 1, H_p, W_p)
    patch_masks = patch_masks.squeeze(1).view(num_frames, -1) > 0.5  # (S, num_patches)

    # Normalize images using ResNet normalization
    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    images_resnet_norm = (images - resnet_mean) / resnet_std

    with torch.no_grad():
        frame_feat = dino_v2_model(images_resnet_norm, is_training=True)

    # Use patch tokens for masked feature extraction
    patch_tokens = frame_feat["x_norm_patchtokens"]  # (S, num_patches, feat_dim)

    # Compute masked mean features for each frame
    masked_features = []
    for i in range(num_frames):
        mask_i = patch_masks[i]  # (num_patches,)
        if mask_i.sum() > 0:
            # Average features within the mask
            masked_feat = patch_tokens[i, mask_i].mean(dim=0)
        else:
            # Fallback to all patches if mask is empty
            masked_feat = patch_tokens[i].mean(dim=0)
        masked_features.append(masked_feat)

    masked_features = torch.stack(masked_features)  # (S, feat_dim)
    masked_features = F.normalize(masked_features, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.mm(masked_features, masked_features.transpose(0, 1))

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)

    # Rank frames by similarity sum (descending - most common first)
    ranked_indices = torch.argsort(similarity_sum, descending=True).tolist()

    # Clean up to free memory
    del patch_tokens, masked_features, similarity_matrix, similarity_sum
    del dino_v2_model
    torch.cuda.empty_cache()

    return ranked_indices


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def get_image_paths(scene_dir, min_frame, max_frame, interval):
    """Load and filter image paths from the scene directory."""
    image_dir = Path(scene_dir) / "rgb"
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    return image_dir, [str(p) for p in image_paths[min_frame:max_frame:interval]]


# -----------------------------------------------------------------------------
# Tensor/Array Conversion
# -----------------------------------------------------------------------------

def to_numpy(tensor):
    """Convert tensor to numpy array if needed."""
    return tensor.numpy() if torch.is_tensor(tensor) else tensor


def squeeze_to_2d(arr):
    """Squeeze a 3D array to 2D by removing singleton dimension."""
    if arr.ndim == 3:
        return arr.squeeze(-1) if arr.shape[-1] == 1 else arr.squeeze(0)
    return arr


# -----------------------------------------------------------------------------
# Image Processing
# -----------------------------------------------------------------------------

def normalize_image_to_uint8(image):
    """Convert image to HWC uint8 format."""
    image = to_numpy(image)

    # Convert CHW to HWC if needed
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))

    # Normalize to 0-255
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def create_alpha_channel(mask, target_shape):
    """Create alpha channel from mask, resizing if needed."""
    alpha = (mask.squeeze() * 255).astype(np.uint8)

    if alpha.shape[:2] != target_shape[:2]:
        alpha_img = Image.fromarray(alpha)
        alpha_img = alpha_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        alpha = np.array(alpha_img)

    return alpha


def create_rgba_image(image, mask):
    """Convert image and mask to RGBA format."""
    image = normalize_image_to_uint8(image)
    alpha = create_alpha_channel(mask, image.shape)

    if image.shape[-1] == 3:
        return np.concatenate([image, alpha[..., None]], axis=-1)
    else:
        image[..., 3] = alpha
        return image


# -----------------------------------------------------------------------------
# Frame Processing
# -----------------------------------------------------------------------------

def save_frame(idx, original_frame_idx, images, masks, out_dir):
    """Save a single frame's RGBA image.

    Args:
        idx: Index in the images array
        original_frame_idx: Original frame index for output naming
        images: Preprocessed images tensor
        masks: Preprocessed masks tensor
        out_dir: Output directory path
    """
    frame_out_dir = out_dir / f"{original_frame_idx:04d}"
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare mask
    mask = squeeze_to_2d(to_numpy(masks[idx])) > 0.5

    # Save RGBA image
    rgba = create_rgba_image(images[idx], mask)
    Image.fromarray(rgba).save(frame_out_dir / "image.png")


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_condition_ids(out_dir, ranked_indices):
    """Save frame indices ranked by DINO similarity to condition_id.txt."""
    with open(out_dir / "condition_id.txt", "w") as f:
        for rank, frame_idx in enumerate(ranked_indices):
            f.write(f"{frame_idx:04d} {rank}\n")

    print(f"\nSaved condition IDs to {out_dir / 'condition_id.txt'}")
    print(f"Top 5 frames by DINO similarity: {[f'{idx:04d}' for idx in ranked_indices[:5]]}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    # Load image paths
    image_dir, image_path_list = get_image_paths(
        args.scene_dir, args.min_frame_num, args.max_frame_num, args.frame_interval
    )
    print(f"Processing {len(image_path_list)} images from {image_dir}")

    # Preprocess images and masks
    images, _, masks, _ = load_and_preprocess_images_square_HO3D(
        image_path_list,
        args,
        target_size=1024,
        out_dir=None,
    )

    # Rank frames by DINO similarity (most representative first)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ranking frames using DINO features on {device}...")

    ranked_indices = generate_rank_by_dino(
        images.to(device),
        masks.to(device),
        device=device,
    )
    print(f"Ranked {len(ranked_indices)} frames by DINO similarity")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save images for each frame
    for idx in range(len(images)):
        original_frame_idx = idx * args.frame_interval
        save_frame(idx, original_frame_idx, images, masks, out_dir)
        # print(f"Saved frame {original_frame_idx:04d}")

    # Save sorted condition IDs
    # Convert ranked indices to original frame indices
    ranked_original_indices = [idx * args.frame_interval for idx in ranked_indices]
    save_condition_ids(out_dir, ranked_original_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get condition IDs by sorting frames by DINO similarity (most common first)"
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True,
        help="Directory containing scene data (rgb/, meta/)"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for images and condition IDs"
    )
    parser.add_argument(
        "--instance_id", type=int, default=0,
        help="Instance ID for image preprocessing"
    )
    parser.add_argument(
        "--min_frame_num", type=int, default=0,
        help="Minimum frame number to process"
    )
    parser.add_argument(
        "--max_frame_num", type=int, default=-1,
        help="Maximum frame number to process (-1 for all)"
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1,
        help="Frame interval for processing"
    )
    args = parser.parse_args()

    main(args)
