from glob import glob
from PIL import Image
import numpy as np
import os
import numpy as np
import os.path as op
from tqdm import tqdm
import sys
import cv2

sys.path = ['../code'] + sys.path


def hand_object_mask_gen(args):
    seq_name = args.seq_name
    out_name = args.out_name
    rgba_format = args.rgba_format
    print(f"Processing {seq_name}")


    # rgb image with only object pixels
    rgb_ps = sorted(glob(f"data/{seq_name}/processed/images/*.png"))
    mask_ps = sorted(glob(f"data/{seq_name}/processed/masks/*.png"))
    assert len(rgb_ps) == len(mask_ps), f"len(rgb_ps) = {len(rgb_ps)}, len(mask_ps) = {len(mask_ps)}"
    print(f"Processing {seq_name} to data/{seq_name}/processed/{out_name}")
    
    from src.utils.const import SEGM_IDS
    for rgb_p, object_mask_p in zip(rgb_ps, mask_ps):
        mask_obj_hand = cv2.imread(object_mask_p, cv2.IMREAD_GRAYSCALE)
        
        mask_object = mask_obj_hand == SEGM_IDS["object"]    
        mask_hand = (mask_obj_hand == SEGM_IDS["right"]) | (mask_obj_hand == SEGM_IDS["left"])
        # save object mask
        out_p = rgb_p.replace("/images/", f"/masks_object/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(mask_object).save(out_p)
        print(f"Saved {out_p}")

        # save hand mask
        out_p = rgb_p.replace("/images/", f"/masks_hand/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(mask_hand).save(out_p)
        print(f"Saved {out_p}")

        # Create a kernel for dilation
        kernel = np.ones((5,5), np.uint8)

        # Dilate the hand mask
        mask_hand = cv2.dilate(mask_hand.astype(np.uint8), kernel, iterations=1)

        merged_mask = np.logical_or(mask_object, mask_hand).astype(np.uint8) * 255

        _, binary = cv2.threshold(merged_mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(mask_obj_hand)
        cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Save contour mask
        out_p = rgb_p.replace("/images/", f"/{out_name}/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(contour_mask).save(out_p)
        print(f"Saved {out_p}")

        # Load RGB image
        rgb = np.array(Image.open(rgb_p))

        # Create overlay by applying contour mask with opacity
        overlay = rgb.copy()
        overlay[contour_mask == 255] = [255, 0, 0]  # Red overlay for the contour

        # Blend original image with overlay
        alpha = 0.5  # Opacity of overlay
        blended = cv2.addWeighted(rgb, 1-alpha, overlay, alpha, 0)

        # Save overlay image
        overlay_out_p = rgb_p.replace("/images/", f"/{out_name}_overlay/")
        os.makedirs(os.path.dirname(overlay_out_p), exist_ok=True)
        Image.fromarray(blended).save(overlay_out_p)
        print(f"Saved {overlay_out_p}")

    # Create separate overlay for hand and object visualization
    for rgb_p, object_mask_p in zip(rgb_ps, mask_ps):
        # Load masks
        mask_obj_hand = cv2.imread(object_mask_p, cv2.IMREAD_GRAYSCALE)
        mask_object = (mask_obj_hand == SEGM_IDS["object"]).astype(np.uint8) * 255
        mask_hand = ((mask_obj_hand == SEGM_IDS["right"]) | (mask_obj_hand == SEGM_IDS["left"])).astype(np.uint8) * 255
        mask_bg = (mask_obj_hand == SEGM_IDS["bg"]).astype(np.uint8) * 255

        # Load RGB image
        rgb = np.array(Image.open(rgb_p))
        
        # Create overlays for hand and object separately
        hand_overlay = rgb.copy()
        object_overlay = rgb.copy()
        bg_overlay = rgb.copy()
        
        hand_overlay[mask_hand == 255] = [0, 0, 0]
        hand_overlay[mask_hand == 0] = [255, 255, 255]
        # Apply red color to object mask
        object_overlay[mask_object == 255] = [255, 255, 255]
        object_overlay[mask_object == 0] = [0, 0, 0]
        
        bg_overlay[mask_bg == 255] = [255, 255, 255]
        bg_overlay[mask_bg == 0] = [0, 0, 0]
        # Show hand_overlay, bg_overlay, and object_overlay
        # Save hand overlay
        hand_overlay_out_p = rgb_p.replace("/images/", "/hand_overlay/")
        os.makedirs(os.path.dirname(hand_overlay_out_p), exist_ok=True)
        Image.fromarray(hand_overlay).save(hand_overlay_out_p)
        print(f"Saved {hand_overlay_out_p}")
        

        # Save object overlay
        object_overlay_out_p = rgb_p.replace("/images/", "/object_overlay/")
        os.makedirs(os.path.dirname(object_overlay_out_p), exist_ok=True)
        Image.fromarray(object_overlay).save(object_overlay_out_p)
        print(f"Saved {object_overlay_out_p}")

        # Save background overlay
        bg_overlay_out_p = rgb_p.replace("/images/", "/bg_overlay/")
        os.makedirs(os.path.dirname(bg_overlay_out_p), exist_ok=True)
        Image.fromarray(bg_overlay).save(bg_overlay_out_p)
        print(f"Saved {bg_overlay_out_p}")
        # Blend original with hand overlay
        alpha = 0.4  # Opacity
        temp_blend = cv2.addWeighted(rgb, 1-alpha, hand_overlay, alpha, 0)
        # alpha = 0.8  # Opacity
        # temp_blend = cv2.addWeighted(temp_blend, 1-alpha, object_overlay, alpha, 0)        
        # alpha = 0.8  # Opacity
        # temp_blend = cv2.addWeighted(temp_blend, 1-alpha, bg_overlay, alpha, 0)


        # Save visualization
        vis_out_p = rgb_p.replace("/images/", "/masks_overlay/")
        os.makedirs(os.path.dirname(vis_out_p), exist_ok=True)
        Image.fromarray(temp_blend).save(vis_out_p)
        print(f"Saved {vis_out_p}")
    print('Done!')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--rgba_format", action="store_true", help="generate object images with rgba format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    hand_object_mask_gen(args)
