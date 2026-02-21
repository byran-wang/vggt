import argparse
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from robust_hoi_pipeline.frame_management import (
    load_register_indices,
    check_frame_invalid,
    check_reprojection_error,
    _refine_frame_pose_3d,
)
from robust_hoi_pipeline.optimization import register_new_frame_by_PnP
from robust_hoi_pipeline.pipeline_joint_opt import (
    TeeStream,
    load_preprocessed_data,
    lift_tracks_to_3d,
    mask_track_for_outliers,
    print_image_info_stats,
    save_results,
    _build_default_joint_opt_args,
    _stack_intrinsics,
)


def main(args):
    log_file = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        data_dir = Path(args.data_dir)
        out_dir = Path(args.output_dir)
        cond_idx = args.cond_index

        results_dir = out_dir / "pipeline_joint_opt"
        data_preprocess_dir = data_dir / "pipeline_preprocess"

        log_dir = results_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "log_reg_remaining.txt"
        log_file = open(log_path, "a", buffering=1)
        sys.stdout = TeeStream(orig_stdout, log_file)
        sys.stderr = TeeStream(orig_stderr, log_file)
        print(f"[logging] Writing console output to {log_path}")

        # Load the last image_info from pipeline_joint_opt
        register_indices = load_register_indices(results_dir)
        last_register_idx = register_indices[-1]
        image_info = np.load(
            results_dir / f"{last_register_idx:04d}" / "image_info.npy",
            allow_pickle=True,
        ).item()

        # Load preprocessed data
        frame_indices = image_info["frame_indices"]
        preprocessed_data = load_preprocessed_data(data_preprocess_dir, frame_indices)

        # Build image_info_work (same structure as register_remaining_frames)
        intrinsics = _stack_intrinsics(preprocessed_data["intrinsics"])
        c2o = image_info.get("c2o")
        if c2o is None:
            c2o = np.tile(np.eye(4, dtype=np.float32), (len(frame_indices), 1, 1))
        extrinsics = np.linalg.inv(c2o).astype(np.float32)

        points_3d_global = image_info["points_3d"].astype(np.float32)

        args_joint = _build_default_joint_opt_args(out_dir, cond_idx)
        invalid_cnt = {
            "insufficient_pixel": 0,
            "3d_3d_corr": 0,
            "reproj_err": 0,
        }

        image_info_work = {
            "frame_indices": frame_indices,
            "pred_tracks": image_info["tracks"],
            "track_mask": image_info["tracks_mask"],
            "points_3d": points_3d_global,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth_priors": preprocessed_data["depths"],
            "images": preprocessed_data["images"],
            "image_masks": preprocessed_data.get("masks_obj"),
            "keyframe": np.array(image_info["keyframe"], dtype=bool),
            "registered": np.array(image_info["register"], dtype=bool),
            "invalid": np.array(image_info["invalid"], dtype=bool),
        }

        # Find invalid frames to attempt re-registration
        invalid_indices = np.where(image_info_work["invalid"])[0]
        print(f"[reg_remaining] Found {len(invalid_indices)} invalid frames to re-register")
        print_image_info_stats(image_info_work, invalid_cnt)

        for frame_idx in invalid_indices:
            print("+" * 50)
            print(f"Attempting to re-register invalid frame: {frame_indices[frame_idx]} (local idx {frame_idx})")

            # TODO 1: check_frame_invalid -> if fails, keep invalid, print stats, continue
            if check_frame_invalid(
                image_info_work,
                frame_idx,
                min_inlier_per_frame=args_joint.min_inlier_per_frame,
                min_depth_pixels=args_joint.min_depth_pixels,
            ):
                print(f"[reg_remaining] Frame {frame_idx} still invalid: insufficient inliers/depth pixels")
                invalid_cnt["insufficient_pixel"] += 1
                print_image_info_stats(image_info_work, invalid_cnt)
                continue

            # TODO 2: Temporarily unmark invalid, run PnP + mask outliers + refine + check reproj
            image_info_work["invalid"][frame_idx] = False

            register_new_frame_by_PnP(image_info_work, frame_idx, args_joint)
            mask_track_for_outliers(image_info_work, frame_idx, args_joint.pnp_reproj_thresh)

            if not _refine_frame_pose_3d(image_info_work, frame_idx, args_joint):
                image_info_work["invalid"][frame_idx] = True
                print(f"[reg_remaining] Frame {frame_idx} marked as invalid: 3D-3D refinement failure")
                invalid_cnt["3d_3d_corr"] += 1
                print_image_info_stats(image_info_work, invalid_cnt)
                continue

            if check_reprojection_error(image_info_work, frame_idx, args_joint):
                image_info_work["invalid"][frame_idx] = True
                print(f"[reg_remaining] Frame {frame_idx} marked as invalid: large reprojection error")
                invalid_cnt["reproj_err"] += 1
                print_image_info_stats(image_info_work, invalid_cnt)
                continue

            # TODO 3: Mark as registered
            image_info_work["registered"][frame_idx] = True
            print(f"[reg_remaining] Successfully re-registered frame {frame_indices[frame_idx]}")

            # TODO 4: Lift tracks to 3D for newly registered frame, merge new 3D points
            frame_c2o = np.linalg.inv(image_info_work["extrinsics"][frame_idx]).astype(np.float32)
            new_points = lift_tracks_to_3d(
                image_info_work["pred_tracks"][frame_idx:frame_idx + 1],
                image_info_work["track_mask"][frame_idx:frame_idx + 1],
                [preprocessed_data["depths"][frame_idx]],
                [image_info_work["intrinsics"][frame_idx]],
                frame_c2o,
            )[0]  # (N, 3)

            # Merge: fill NaN positions in global points_3d with newly lifted points
            nan_mask = ~np.isfinite(image_info_work["points_3d"]).all(axis=-1)
            new_valid = np.isfinite(new_points).all(axis=-1)
            fill_mask = nan_mask & new_valid
            if fill_mask.any():
                image_info_work["points_3d"][fill_mask] = new_points[fill_mask]
                print(f"[reg_remaining] Lifted {fill_mask.sum()} new 3D points from frame {frame_indices[frame_idx]}")

            # TODO 5: Print stats
            print_image_info_stats(image_info_work, invalid_cnt)

        # TODO 6: Copy results back to image_info and save
        image_info["register"] = image_info_work["registered"].tolist()
        image_info["invalid"] = image_info_work["invalid"].tolist()
        image_info["keyframe"] = image_info_work["keyframe"].tolist()
        image_info["c2o"] = np.linalg.inv(image_info_work["extrinsics"]).astype(np.float32)
        image_info["points_3d"] = image_info_work["points_3d"].astype(np.float32)

        print("=" * 50)
        print("[reg_remaining] Final stats:")
        print_image_info_stats(image_info_work, invalid_cnt)

        save_results(
            image_info=image_info,
            register_idx=last_register_idx,
            preprocessed_data=preprocessed_data,
            results_dir=results_dir,
        )
        print("[reg_remaining] Done.")

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-register invalid frames from pipeline_joint_opt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/MC1)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--cond_index", type=int, default=0,
                        help="Condition frame index (keyframe with known SAM3D pose)")

    args = parser.parse_args()
    main(args)
