import os
import h5py
import numpy as np
import sys
sys.path.append("third_party/Hierarchical-Localization/")
from hloc.match_features import WorkQueue, writer_fn, names_to_pair, FeaturePairsDataset
from functools import partial
import torch
from tqdm import tqdm

def prepare_pairs(
        image_list,
        pairs_file,
    ):
    image_names = [os.path.basename(image_path) for image_path in image_list]
    pairs = []
    for i in tqdm(range(len(image_names)), desc="preparing pairs"):
        for j in range(i + 1, len(image_names)):
            pairs.append((image_names[i], image_names[j]))
    with open(str(pairs_file), "w") as f:
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

def prepare_features(
        tracks,
        track_masks,
        image_list,
        image_size, 
        feats_file,
        as_half: bool = True,
):
    assert len(tracks) == len(image_list)
    image_names = [os.path.basename(image_path) for image_path in image_list]
    if as_half:
        tracks = [track.astype(np.float16) for track in tracks]

    with h5py.File(str(feats_file), "a", libver="latest") as fd:
        for name, track, track_mask in tqdm(zip(image_names, tracks, track_masks), desc="preparing features"):
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            grp.create_dataset("keypoints", data=track)
            grp.create_dataset("track_mask", data=track_mask)
            grp.create_dataset("image_size", data=image_size)

    print(f"Finished exporting features to {feats_file}")
    
def prepare_matches(
        tracks,
        pairs_file,
        feats_file,
        matches_file
    ):

    # get the pairs
    pairs = []
    with open(str(pairs_file), "r") as f:
        for line in f:
            pairs.append(line.strip().split())

    assert len(pairs) > 0, "No pairs found"

    dataset = FeaturePairsDataset(pairs, feats_file, feats_file)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=matches_file), 5)        
    # load the feats file
    for idx, data in enumerate(tqdm(loader, desc="preparing matches")):
        pred = {}
        track_mask0 = data["track_mask0"] # [B, N]
        track_mask1 = data["track_mask1"] # [B, N]
        matches = track_mask0 * track_mask1
        # get the index of matches where matches > 0
        matches_idx = torch.where(matches > 0)
        pred["matches0"] = torch.ones_like(matches).long() * -1
        pred["matches0"][matches>0] = matches_idx[1]
        pred['matching_scores0'] = torch.zeros_like(matches)
        pred['matching_scores0'][matches>0] = 1

        pred["matches1"] = pred["matches0"]
        pred["matching_scores1"] = pred["matching_scores0"]

        pair = names_to_pair(*pairs[idx])       
        writer_queue.put((pair, pred))
    writer_queue.join()

    print("Finished exporting matches.")

