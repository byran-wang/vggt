"""NeuS integration for the joint optimization pipeline.

Provides functions to prepare data, run NeuS training as a subprocess,
and save the resulting mesh after each keyframe.
"""

import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# Depth encoding scale (matches utils_simba.depth)
_DEPTH_SCALE = 0.00012498664727900177


def _save_depth_png(depth: np.ndarray, path: str) -> None:
    """Save depth as 24-bit encoded PNG (3-channel uint8), matching utils_simba format."""
    import cv2

    depth_scale_inv = 1.0 / _DEPTH_SCALE
    max_depth = (2**24) * _DEPTH_SCALE
    depth = depth.clip(0, max_depth)

    depth_scaled = (depth * depth_scale_inv).astype(np.uint32)
    encoded = np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    encoded[..., 2] = np.bitwise_and(depth_scaled, 0xFF)
    encoded[..., 1] = np.bitwise_and(np.right_shift(depth_scaled, 8), 0xFF)
    encoded[..., 0] = np.bitwise_and(np.right_shift(depth_scaled, 16), 0xFF)
    cv2.imwrite(path, encoded)


def prepare_neus_data(
    keyframe_indices: List[int],
    images: List[np.ndarray],
    masks: List[np.ndarray],
    depths: List[np.ndarray],
    extrinsics_o2c: np.ndarray,
    intrinsics: np.ndarray,
    neus_data_dir: Path,
) -> None:
    """Write keyframe data to disk in the format expected by ObjDataProvider / robust_hoi dataset.

    Creates the following structure under neus_data_dir:
        images/0000.png, 0001.png, ...
        masks/0000.png, 0001.png, ...
        depth_prior/0000.png, 0001.png, ...
        key_frame_idx.txt
        0000/results.pkl  (contains intrinsics, extrinsics, keyframe flags)

    Args:
        keyframe_indices: Local indices of keyframes (0-based within the keyframe list).
        images: List of (H, W, 3) uint8 RGB images for each keyframe.
        masks: List of (H, W) uint8 object masks for each keyframe.
        depths: List of (H, W) float32 depth maps for each keyframe.
        extrinsics_o2c: (N_kf, 4, 4) object-to-camera (w2c) matrices.
        intrinsics: (N_kf, 3, 3) camera intrinsic matrices.
        neus_data_dir: Output directory for NeuS data.
    """
    neus_data_dir = Path(neus_data_dir)

    # Create subdirectories
    img_dir = neus_data_dir / "images"
    mask_dir = neus_data_dir / "masks"
    depth_dir = neus_data_dir / "depth_prior"
    for d in [img_dir, mask_dir, depth_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n_kf = len(images)
    assert len(masks) == n_kf
    assert len(depths) == n_kf
    assert extrinsics_o2c.shape[0] == n_kf
    assert intrinsics.shape[0] == n_kf

    # Write images, masks, depths
    for i in range(n_kf):
        fname = f"{i:04d}.png"

        # RGB image
        if images[i] is not None:
            Image.fromarray(images[i]).save(str(img_dir / fname))

        # Object mask
        if masks[i] is not None:
            mask = masks[i]
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            Image.fromarray(mask).save(str(mask_dir / fname))

        # Depth (24-bit encoded PNG)
        if depths[i] is not None:
            _save_depth_png(depths[i], str(depth_dir / fname))

    # Write key_frame_idx.txt (all frames are keyframes in this context)
    kf_idx_path = neus_data_dir / "key_frame_idx.txt"
    with open(kf_idx_path, "w") as f:
        for i in range(n_kf):
            f.write(f"{i}\n")

    # Build and save results.pkl in a "0000" subdirectory
    # The ObjDataProvider reads the last step's results.pkl which contains
    # intrinsics, extrinsics (w2c, 3x4), and keyframe flags for all frames.
    results_dir = neus_data_dir / "0000"
    results_dir.mkdir(parents=True, exist_ok=True)

    # extrinsics_o2c is (N_kf, 4, 4), store as (N_kf, 3, 4)
    extrinsics_3x4 = extrinsics_o2c[:, :3, :4].astype(np.float32)

    results_pkl = {
        "intrinsics": intrinsics.astype(np.float32),  # (N_kf, 3, 3)
        "extrinsics": extrinsics_3x4,  # (N_kf, 3, 4) w2c
        "keyframe": [True] * n_kf,
    }
    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results_pkl, f)

    print(f"[NeuS] Prepared data for {n_kf} keyframes in {neus_data_dir}")


def run_neus_training(
    neus_data_dir: Path,
    config_path: str,
    max_steps: int,
    checkpoint_path: Optional[str],
    output_dir: Path,
    sam3d_root_dir: Optional[Path] = None,
    gpu_id: int = 0,
) -> Tuple[Optional[str], Optional[str]]:
    """Run NeuS training as a subprocess.

    Args:
        neus_data_dir: Path to prepared NeuS data directory.
        config_path: Path to NeuS YAML config (relative to project root or absolute).
        max_steps: Absolute max training steps (PyTorch Lightning convention).
        checkpoint_path: Path to checkpoint for resuming, or None for fresh training.
        output_dir: Directory for NeuS experiment outputs (checkpoints, meshes).
        sam3d_root_dir: Path to SAM3D renders directory (for mixed dataset).
        gpu_id: GPU device ID to use.

    Returns:
        Tuple of (latest_checkpoint_path, exported_mesh_path), either may be None
        if not found after training.
    """
    neus_data_dir = Path(neus_data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the project root (where launch.py lives)
    project_root = Path(__file__).resolve().parent.parent
    launch_script = project_root / "third_party" / "instant-nsr-pl" / "launch.py"

    if not launch_script.exists():
        raise FileNotFoundError(f"NeuS launch script not found: {launch_script}")

    # Resolve config path
    config_resolved = Path(config_path)
    if not config_resolved.is_absolute():
        config_resolved = project_root / config_path
    if not config_resolved.exists():
        raise FileNotFoundError(f"NeuS config not found: {config_resolved}")

    cmd = [
        "/home/simba/miniconda3/envs/vggsfm_tmp/bin/python", str(launch_script),
        "--config", str(config_resolved),
        "--train",
        "--gpu", str(gpu_id),
        "--exp_dir", str(output_dir),
        f"dataset.root_dir={neus_data_dir}",
        f"trainer.max_steps={max_steps}",
        f"export.export_vertex_color=false",
        f"checkpoint.every_n_train_steps={max_steps}",
    ]

    if sam3d_root_dir is not None and Path(sam3d_root_dir).exists():
        cmd.append(f"dataset.sam3d_root_dir={sam3d_root_dir}")

    

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        cmd.extend(["--resume", str(checkpoint_path)])

    print(f"[NeuS] Running training: max_steps={max_steps}, resume={checkpoint_path is not None}")
    print(f"[NeuS] Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["CC"] = "gcc-11"
    env["CXX"] = "g++-11"
    env["CUDAHOSTCXX"] = "g++-11"

    result = subprocess.run(
        cmd,
        cwd=str(project_root / "third_party" / "instant-nsr-pl"),
        capture_output=False,
        env=env,
    )

    if result.returncode != 0:
        print(f"[NeuS] Training subprocess returned non-zero exit code: {result.returncode}")

    # Scan output directory for latest checkpoint and exported mesh
    latest_ckpt = _find_latest_checkpoint(output_dir)
    latest_mesh = _find_latest_mesh(output_dir)

    print(f"[NeuS] Latest checkpoint: {latest_ckpt}")
    print(f"[NeuS] Exported mesh: {latest_mesh}")

    return latest_ckpt, latest_mesh


def _find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the most recently modified .ckpt file under output_dir."""
    ckpt_files = list(output_dir.rglob("*.ckpt"))
    if not ckpt_files:
        return None
    return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))


def _find_latest_mesh(output_dir: Path) -> Optional[str]:
    """Find the most recently modified .obj mesh file under output_dir."""
    mesh_files = list(output_dir.rglob("*.obj"))
    if not mesh_files:
        return None
    return str(max(mesh_files, key=lambda p: p.stat().st_mtime))


def save_neus_mesh(mesh_path: Optional[str], target_dir: Path) -> None:
    """Copy the exported NeuS mesh to the target results directory.

    Args:
        mesh_path: Path to the exported .obj mesh, or None if not available.
        target_dir: Destination directory (e.g., output/SM2/pipeline_joint_opt/0018/).
    """
    if mesh_path is None or not Path(mesh_path).exists():
        print(f"[NeuS] No mesh to save (path={mesh_path})")
        return

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "neus_mesh.obj"
    shutil.copy2(mesh_path, dest)
    print(f"[NeuS] Saved mesh to {dest}")
