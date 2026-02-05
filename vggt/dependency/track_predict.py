# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .vggsfm_utils import *


def predict_tracks(
    images,
    image_masks=None,
    conf=None,
    points_3d=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
    query_frame_indexes = [0]
):
    """
    Predict tracks for the given images and masks.

    TODO: support non-square images
    TODO: support masks


    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    Args:
        images: Tensor of shape [S, 3, H, W] containing the input images.
        conf: Tensor of shape [S, 1, H, W] containing the confidence scores. Default is None.
        points_3d: Tensor containing 3D points. Default is None.
        masks: Optional tensor of shape [S, 1, H, W] containing masks. Default is None.
        max_query_pts: Maximum number of query points. Default is 2048.
        query_frame_num: Number of query frames to use. Default is 5.
        keypoint_extractor: Method for keypoint extraction. Default is "aliked+sp".
        max_points_num: Maximum number of points to process at once. Default is 163840.
        fine_tracking: Whether to use fine tracking. Default is True.
        complete_non_vis: Whether to augment non-visible frames. Default is True.

    Returns:
        pred_tracks: Numpy array containing the predicted tracks.
        pred_vis_scores: Numpy array containing the visibility scores for the tracks.
        pred_confs: Numpy array containing the confidence scores for the tracks.
        pred_points_3d: Numpy array containing the 3D points for the tracks.
        pred_colors: Numpy array containing the point colors for the tracks. (0, 255)
    """

    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    if query_frame_num > 1:
        # Find query frames using DINO ranking
        query_frame_indexes_gen = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

        # Remove duplicates: filter out frames already in the input query_frame_indexes
        query_frame_indexes_gen = [idx for idx in query_frame_indexes_gen if idx not in query_frame_indexes]

        # Put input query_frame_indexes at the front, followed by generated ones
        query_frame_indexes = list(query_frame_indexes) + query_frame_indexes_gen

    # TODO: add the functionality to handle the masks
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
            image_masks,
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_colors.append(pred_color)
        if pred_point_3d is not None:
            pred_points_3d.append(pred_point_3d)
        if pred_conf is not None:
            pred_confs.append(pred_conf)

    if complete_non_vis:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            min_vis=500,
            non_vis_thresh=0.1,
            device=device,
            image_masks=image_masks,
        )

    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    # from vggt.utils.visual_track import visualize_tracks_on_images
    # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")
    torch.cuda.empty_cache()
    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
    image_masks=None,
):
    """
    Process a single query frame for track prediction.

    Args:
        query_index: Index of the query frame
        images: Tensor of shape [S, 3, H, W] containing the input images
        conf: Confidence tensor
        points_3d: 3D points tensor
        fmaps_for_tracker: Feature maps for the tracker
        keypoint_extractors: Initialized feature extractors
        tracker: VGG-SFM tracker
        max_points_num: Maximum number of points to process at once
        fine_tracking: Whether to use fine tracking
        device: Device to use for computation
        image_masks: Tensor of shape [S, 1, H, W] containing the masks

    Returns:
        pred_track: Predicted tracks
        pred_vis: Visibility scores for the tracks
        pred_conf: Confidence scores for the tracks
        pred_point_3d: 3D points for the tracks
        pred_color: Point colors for the tracks (0, 255)
    """
    frame_num, _, height, width = images.shape

    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    # Extract the color at the keypoint locations
    query_points_long = query_points.squeeze(0).round().long()
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    # # plot query points on the image
    # import matplotlib.pyplot as plt
    # plt.imshow(query_image.permute(1, 2, 0).cpu().numpy())
    # # Remove batch dimension and plot points with their actual colors
    # points_to_plot = query_points[0].cpu().numpy()  # Shape: [1930, 2]
    # # Normalize colors from [0, 255] to [0, 1] for matplotlib
    # colors_normalized = pred_color / 255.0
    # plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='red', s=10)
    # plt.savefig(f"query_points_{query_index}.png")
    # plt.close()    
    # Query the confidence and points_3d at the keypoint locations
    if (conf is not None) and (points_3d is not None):
        assert height == width
        assert conf.shape[-2] == conf.shape[-1]
        assert conf.shape[:3] == points_3d.shape[:3]
        scale = conf.shape[-1] / width

        query_points_scaled = (query_points.squeeze(0) * scale).round().long()
        query_points_scaled = query_points_scaled.cpu().numpy()

        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        # # plot query conf and pred_conf in the same figure
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 4, 1)
        # plt.imshow(conf[query_index])
        # plt.subplot(1, 4, 2)
        # pred_conf_plot = np.zeros((conf[0].shape[0], conf[0].shape[1]))
        # pred_conf_plot[query_points_scaled[:, 1], query_points_scaled[:, 0]] = pred_conf
        # plt.imshow(pred_conf_plot)
        # plt.subplot(1, 4, 3)
        # plt.imshow(image_masks[query_index][0].cpu().numpy())

        if image_masks is not None:
            image_masks = image_masks.cpu().numpy()
            query_points_raw = query_points.squeeze(0).round().long().cpu().numpy()
            pred_conf = pred_conf * image_masks[query_index][0][query_points_raw[:, 1], query_points_raw[:, 0]]

            # # plot the pred_conf
            # pred_conf_plot = np.zeros((height, width))
            # pred_conf_plot[query_points_raw[:, 1], query_points_raw[:, 0]] = pred_conf
            # plt.subplot(1, 4, 4)
            # plt.imshow(pred_conf_plot)
            # plt.show()
        
        # heuristic to remove low confidence points
        # should I export this as an input parameter?
        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]  # Make sure shape is compatible
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]  # add batch dimension
    fmaps_feed = fmaps_feed[None]  # add batch dimension

    all_points_num = images_feed.shape[1] * query_points.shape[1]

    # Don't need to be scared, this is just chunking to make GPU happy
    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

    pred_track = pred_track.squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _augment_non_visible_frames(
    pred_tracks: list,  # ← running list of np.ndarrays
    pred_vis_scores: list,  # ← running list of np.ndarrays
    pred_confs: list,  # ← running list of np.ndarrays for confidence scores
    pred_points_3d: list,  # ← running list of np.ndarrays for 3D points
    pred_colors: list,  # ← running list of np.ndarrays for colors
    images: torch.Tensor,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    *,
    min_vis: int = 500,
    non_vis_thresh: float = 0.1,
    device: torch.device = None,
    image_masks=None,
):
    """
    Augment tracking for frames with insufficient visibility.

    Args:
        pred_tracks: List of numpy arrays containing predicted tracks.
        pred_vis_scores: List of numpy arrays containing visibility scores.
        pred_confs: List of numpy arrays containing confidence scores.
        pred_points_3d: List of numpy arrays containing 3D points.
        pred_colors: List of numpy arrays containing point colors.
        images: Tensor of shape [S, 3, H, W] containing the input images.
        conf: Tensor of shape [S, 1, H, W] containing confidence scores
        points_3d: Tensor containing 3D points
        fmaps_for_tracker: Feature maps for the tracker
        keypoint_extractors: Initialized feature extractors
        tracker: VGG-SFM tracker
        max_points_num: Maximum number of points to process at once
        fine_tracking: Whether to use fine tracking
        min_vis: Minimum visibility threshold
        non_vis_thresh: Non-visibility threshold
        device: Device to use for computation

    Returns:
        Updated pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, and pred_colors lists.
    """
    last_query = -1
    final_trial = False
    cur_extractors = keypoint_extractors  # may be replaced on the final trial

    while True:
        # Visibility per frame
        vis_array = np.concatenate(pred_vis_scores, axis=1)

        # Count frames with sufficient visibility using numpy
        sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
        non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()

        if len(non_vis_frames) == 0:
            break

        print("Processing non visible frames:", non_vis_frames)

        # Decide the frames & extractor for this round
        if non_vis_frames[0] == last_query:
            # Same frame failed twice - final "all-in" attempt
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_vis_frames  # blast them all at once
        else:
            query_frame_list = [non_vis_frames[0]]  # Process one at a time

        last_query = non_vis_frames[0]

        # Run the tracker for every selected frame
        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
                image_masks,
            )
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_colors.append(new_color)
            if new_conf is not None:
                pred_confs.append(new_conf)
            if new_point_3d is not None:
                pred_points_3d.append(new_point_3d)

        if final_trial:
            break  # Stop after final attempt

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors
