"""NeuS integration for the joint optimization pipeline.

Provides functions to prepare data, run NeuS training in-process with model
caching, and save the resulting mesh after each keyframe.
"""

import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level cache: keeps the NeuS system (model) in GPU memory between
# calls to run_neus_training(), avoiding subprocess startup and redundant
# checkpoint loading on every keyframe.
# ---------------------------------------------------------------------------
_neus_cache: Dict = {}


def _ensure_neus_imports() -> None:
    """Add the NeuS third-party directory to sys.path (idempotent)."""
    neus_root = str(Path(__file__).resolve().parent.parent / "third_party" / "instant-nsr-pl")
    if neus_root not in sys.path:
        sys.path.insert(0, neus_root)


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
    robust_hoi_weight: float = 1.0,
    sam3d_weight: float = 0.3,
) -> Tuple[Optional[str], Optional[str]]:
    """Run NeuS training in-process with system caching.

    On the first call the NeuS system (model) is created and kept in GPU
    memory.  Subsequent calls reuse the cached system, so only the dataset
    and trainer are rebuilt.  The checkpoint is still written to disk for
    correct PyTorch-Lightning training-state restoration (step count,
    optimizer, LR scheduler), but the heavy subprocess startup / reimport /
    CUDA-init overhead is eliminated.

    Args:
        neus_data_dir: Path to prepared NeuS data directory.
        config_path: Path to NeuS YAML config (relative to project root or absolute).
        max_steps: Absolute max training steps (PyTorch Lightning convention).
        checkpoint_path: Path to checkpoint for resuming, or None for fresh training.
        output_dir: Directory for NeuS experiment outputs (checkpoints, meshes).
        sam3d_root_dir: Path to SAM3D renders directory (for mixed dataset).
        gpu_id: GPU device ID to use.

    Returns:
        Tuple of (latest_checkpoint_path, exported_mesh_path), either may be
        None if not found after training.
    """
    _ensure_neus_imports()

    neus_data_dir = Path(neus_data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent

    # Resolve config path
    config_resolved = Path(config_path)
    if not config_resolved.is_absolute():
        config_resolved = project_root / config_path
    if not config_resolved.exists():
        raise FileNotFoundError(f"NeuS config not found: {config_resolved}")

    # ------------------------------------------------------------------
    # Lazy imports – Python caches them after the first call
    # ------------------------------------------------------------------
    import datasets as neus_datasets          # noqa: E402 (from instant-nsr-pl)
    import systems as neus_systems            # noqa: E402
    import pytorch_lightning as pl            # noqa: E402
    from pytorch_lightning import Trainer     # noqa: E402
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor  # noqa: E402
    from pytorch_lightning.loggers import CSVLogger  # noqa: E402
    from utils.callbacks import CustomProgressBar    # noqa: E402
    from utils.misc import load_config               # noqa: E402

    # ------------------------------------------------------------------
    # Determine which checkpoint to resume from:
    #   cached (from previous in-process run) > provided (from caller)
    # ------------------------------------------------------------------
    resume_ckpt = _neus_cache.get("ckpt_path") or checkpoint_path
    if resume_ckpt and not Path(resume_ckpt).exists():
        resume_ckpt = None

    # ------------------------------------------------------------------
    # First call: build config & create system
    # ------------------------------------------------------------------
    if "system" not in _neus_cache:
        os.environ["CC"] = "gcc-11"
        os.environ["CXX"] = "g++-11"
        os.environ["CUDAHOSTCXX"] = "g++-11"

        cli_args = [
            f"dataset.root_dir={neus_data_dir}",
            f"trainer.max_steps={max_steps}",
            f"export.export_vertex_color=false",
            f"checkpoint.every_n_train_steps={max_steps}",
            f"dataset.robust_hoi_weight={robust_hoi_weight}",
            f"dataset.sam3d_weight={sam3d_weight}",
        ]
        if sam3d_root_dir and Path(sam3d_root_dir).exists():
            cli_args.append(f"dataset.sam3d_root_dir={sam3d_root_dir}")

        config = load_config(str(config_resolved), cli_args=cli_args)

        from datetime import datetime
        config.trial_name = config.get("trial_name") or (
            config.tag + datetime.now().strftime("@%Y%m%d-%H%M%S")
        )
        config.exp_dir = str(output_dir)
        config.save_dir = os.path.join(str(output_dir), config.trial_name, "save")
        config.ckpt_dir = os.path.join(str(output_dir), config.trial_name, "ckpt")
        config.code_dir = os.path.join(str(output_dir), config.trial_name, "code")
        config.config_dir = os.path.join(str(output_dir), config.trial_name, "config")
        config.cmd_args = {}

        system = neus_systems.make(config.system.name, config)

        _neus_cache["config"] = config
        _neus_cache["system"] = system
        print(f"[NeuS] Created in-process system (first call)")
    else:
        # --------------------------------------------------------------
        # Subsequent calls: reuse cached system, update mutable config
        # --------------------------------------------------------------
        config = _neus_cache["config"]
        system = _neus_cache["system"]

        config.dataset.root_dir = str(neus_data_dir)
        config.trainer.max_steps = max_steps
        config.checkpoint.every_n_train_steps = max_steps
        config.dataset.robust_hoi_weight = robust_hoi_weight
        config.dataset.sam3d_weight = sam3d_weight
        if sam3d_root_dir and Path(sam3d_root_dir).exists():
            config.dataset.sam3d_root_dir = str(sam3d_root_dir)
        print(f"[NeuS] Reusing cached system")

    # ------------------------------------------------------------------
    # Fresh data module + trainer for each call
    # ------------------------------------------------------------------
    pl.seed_everything(config.get("seed", 42))
    dm = neus_datasets.make(config.dataset.name, config.dataset)

    callbacks = [
        ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
        LearningRateMonitor(logging_interval="step"),
        CustomProgressBar(refresh_rate=1),
    ]
    loggers = [CSVLogger(config.exp_dir, name=config.trial_name, version="csv_logs")]

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        strategy="auto",
        **config.trainer,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"[NeuS] Training: max_steps={max_steps}, resume={'yes' if resume_ckpt else 'no'}")
    if resume_ckpt:
        trainer.fit(system, datamodule=dm, ckpt_path=resume_ckpt)
    else:
        trainer.fit(system, datamodule=dm)

    # Export mesh
    system.cuda()
    system.export()

    # ------------------------------------------------------------------
    # Update cache
    # ------------------------------------------------------------------
    latest_ckpt = _find_latest_checkpoint(output_dir)
    latest_mesh = _find_latest_mesh(output_dir)
    _neus_cache["ckpt_path"] = latest_ckpt

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
