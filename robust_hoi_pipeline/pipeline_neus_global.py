import argparse
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.pipeline_joint_opt import (
    load_preprocessed_data,
    _stack_intrinsics,
    _filter_depth_by_object_bbox,
)
from robust_hoi_pipeline.pipeline_neus_init import (
    _load_joint_opt_image_info,
    _select_registered_frame_subset,
)
from robust_hoi_pipeline.neus_integration import (
    prepare_neus_data,
    run_neus_training,
)
from utils_simba.eval_vis import load_mesh_as_trimesh
from utils_simba.logger import get_logger
from utils_simba.render import nvdiffrast_render

logger = get_logger(__name__)


def _find_latest_neus_mesh(search_dir: Path) -> Path:
    if not search_dir.exists():
        raise FileNotFoundError(
            f"NeuS mesh search directory not found: {search_dir}. "
            f"Run pipeline_joint_opt first to produce a NeuS mesh."
        )
    obj_files = sorted(search_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime)
    if not obj_files:
        raise FileNotFoundError(
            f"No .obj mesh found under {search_dir}. "
            f"Run pipeline_joint_opt first to produce a NeuS mesh."
        )
    return obj_files[-1]


def _render_object_mask(mesh, K: np.ndarray, o2c: np.ndarray, H: int, W: int, glctx) -> np.ndarray:
    ob_in_cvcams = torch.as_tensor(o2c, device="cuda", dtype=torch.float)[None]
    _, depth, _ = nvdiffrast_render(
        K=K, H=H, W=W,
        ob_in_cvcams=ob_in_cvcams,
        glctx=glctx,
        mesh=mesh,
        output_size=(H, W),
    )
    return (depth[0].detach().cpu().numpy() > 0)


def _filter_depth_keyframes(depths_kf, masks_kf, K_kf, o2c_kf, bbox_half_extent: float):
    bbox_min = [-bbox_half_extent] * 3
    bbox_max = [bbox_half_extent] * 3
    image_info_work = {
        "depth_priors": depths_kf,
        "intrinsics": np.asarray(K_kf, dtype=np.float32),
        "image_masks": masks_kf,
    }
    for kf_idx in range(len(depths_kf)):
        _filter_depth_by_object_bbox(
            image_info_work, kf_idx,
            bbox_min=bbox_min, bbox_max=bbox_max,
            extrinsic=o2c_kf[kf_idx],
        )


def _filter_keyframes_by_mask_iou(
    mesh, K_kf, o2c_kf, masks_kf, masks_hand_kf, H, W, threshold: float
):
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

    n_kf = len(masks_kf)
    keep = np.zeros(n_kf, dtype=bool)
    ious = np.zeros(n_kf, dtype=np.float32)

    for i in range(n_kf):
        pred_obj = _render_object_mask(mesh, K_kf[i], o2c_kf[i], H, W, glctx)

        gt_obj = (np.asarray(masks_kf[i]).squeeze() > 0)
        if masks_hand_kf is not None and masks_hand_kf[i] is not None:
            hand = (np.asarray(masks_hand_kf[i]).squeeze() > 0)
            gt_full = gt_obj | hand
            pred_full = pred_obj | hand
        else:
            gt_full = gt_obj
            pred_full = pred_obj

        inter = np.logical_and(gt_full, pred_full).sum()
        union = np.logical_or(gt_full, pred_full).sum()
        iou = float(inter) / max(int(union), 1)
        ious[i] = iou
        keep[i] = iou >= threshold

    logger.info(
        f"[neus_global] mask IoU stats: min={ious.min():.3f}, "
        f"max={ious.max():.3f}, mean={ious.mean():.3f}, threshold={threshold}"
    )
    return keep, ious


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    result_dir = Path(args.result_dir)
    cond_idx = args.cond_index

    sam3d_root_dir = data_dir / "SAM3D_aligned_post_process" / f"{cond_idx:04d}"
    data_preprocess_dir = data_dir / "pipeline_preprocess"
    joint_opt_dir = result_dir / "pipeline_joint_opt"
    neus_mesh_search_dir = joint_opt_dir / "neus_training"

    mesh_path = _find_latest_neus_mesh(neus_mesh_search_dir)
    logger.info(f"Using NeuS mesh from joint-opt: {mesh_path}")
    mesh = load_mesh_as_trimesh(mesh_path)

    logger.info("Loading latest image info from pipeline_joint_opt...")
    image_info, last_register_idx, image_info_path = _load_joint_opt_image_info(joint_opt_dir)
    logger.info(f"Loaded image info from {image_info_path} (register_idx={last_register_idx:04d})")

    frame_indices, selected_local_indices, keyframe_local_indices = _select_registered_frame_subset(
        image_info, joint_opt_dir, args.max_registered_frames,
    )
    if keyframe_local_indices.size == 0:
        raise RuntimeError("No keyframes found in latest image_info.")

    preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)

    if "c2o" not in image_info:
        raise KeyError("Latest image_info is missing 'c2o'.")
    c2o_all = np.asarray(image_info["c2o"], dtype=np.float32)[selected_local_indices]
    o2c_all = np.linalg.inv(c2o_all).astype(np.float32)
    o2c_keyframes = o2c_all[keyframe_local_indices]

    if "intrinsics" in image_info:
        intrinsics_all = np.asarray(image_info["intrinsics"], dtype=np.float32)[selected_local_indices]
    else:
        intrinsics_all = _stack_intrinsics(preprocessed_data["intrinsics"])
    K_keyframes = intrinsics_all[keyframe_local_indices]

    images_kf = [preprocessed_data["images"][i] for i in keyframe_local_indices]
    masks_kf = [preprocessed_data["masks_obj"][i] for i in keyframe_local_indices]
    masks_hand_kf = (
        [preprocessed_data["masks_hand"][i] for i in keyframe_local_indices]
        if "masks_hand" in preprocessed_data else None
    )
    depths_kf = [preprocessed_data["depths"][i] for i in keyframe_local_indices]

    n_kf = len(images_kf)
    H, W = images_kf[0].shape[:2]
    logger.info(f"[neus_global] {n_kf} keyframes loaded ({H}x{W})")

    logger.info(
        f"[neus_global] filtering depth by bbox half-extent {args.bbox_half_extent}"
    )
    _filter_depth_keyframes(depths_kf, masks_kf, K_keyframes, o2c_keyframes,
                            args.bbox_half_extent)

    logger.info("[neus_global] computing mask IoU per keyframe")
    keep, ious = _filter_keyframes_by_mask_iou(
        mesh, K_keyframes, o2c_keyframes, masks_kf, masks_hand_kf, H, W,
        args.mask_iou_threshold,
    )
    n_kept = int(keep.sum())
    logger.info(
        f"[neus_global] kept {n_kept}/{n_kf} keyframes after IoU filter "
        f"(threshold={args.mask_iou_threshold})"
    )
    if n_kept == 0:
        raise RuntimeError(
            f"No keyframes survived IoU filtering at threshold "
            f"{args.mask_iou_threshold}. Lower --mask_iou_threshold or check the "
            f"NeuS init mesh quality."
        )

    images_kf = [images_kf[i] for i in range(n_kf) if keep[i]]
    masks_kf = [masks_kf[i] for i in range(n_kf) if keep[i]]
    if masks_hand_kf is not None:
        masks_hand_kf = [masks_hand_kf[i] for i in range(n_kf) if keep[i]]
    depths_kf = [depths_kf[i] for i in range(n_kf) if keep[i]]
    K_keyframes = K_keyframes[keep]
    o2c_keyframes = o2c_keyframes[keep]

    neus_data_dir = out_dir / "neus_data"

    prepare_neus_data(
        images=images_kf,
        masks=masks_kf,
        depths=depths_kf,
        extrinsics_o2c=o2c_keyframes,
        intrinsics=K_keyframes,
        neus_data_dir=neus_data_dir,
        masks_hand=masks_hand_kf,
    )

    neus_ckpt, neus_mesh, _ = run_neus_training(
        neus_data_dir,
        config_path="configs/neus-pipeline.yaml",
        max_steps=args.max_steps,
        checkpoint_path=None,
        output_dir=out_dir / "neus_training",
        sam3d_root_dir=sam3d_root_dir,
        robust_hoi_weight=args.robust_hoi_weight,
        sam3d_weight=args.sam3d_weight,
        export_only=args.export_only,
    )

    logger.info(f"NeuS global complete. Checkpoint: {neus_ckpt}, mesh: {neus_mesh}")


def parse_args():
    parser = argparse.ArgumentParser(description="NeuS global refinement with bbox + IoU filtering")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/SM2/)")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing latest pipeline_joint_opt and pipeline_neus_init results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--robust_hoi_weight", type=float, default=1.0)
    parser.add_argument("--sam3d_weight", type=float, default=0.0)
    parser.add_argument("--max_registered_frames", type=int, default=-1)
    parser.add_argument("--export_only", action="store_true", default=False)
    parser.add_argument("--mask_iou_threshold", type=float, default=0.5,
                        help="Drop keyframes whose mask IoU is below this threshold")
    parser.add_argument("--bbox_half_extent", type=float, default=0.55,
                        help="Object-space bbox half-extent for depth filtering")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
