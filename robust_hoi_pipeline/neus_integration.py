"""NeuS integration for the joint optimization pipeline.

Provides functions to prepare data, run NeuS training in-process with model
caching, and save the resulting mesh after each keyframe.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Module-level cache: keeps the NeuS system (model) in GPU memory between
# calls to run_neus_training(), avoiding subprocess startup and redundant
# checkpoint loading on every keyframe.
# ---------------------------------------------------------------------------
_neus_cache: Dict = {}

# Module-level in-memory cache keyed by absolute neus_data_dir path.
# Populated by prepare_neus_data(); consumed by the NeuS dataset loader in
# third_party/instant-nsr-pl/datasets/robust_hoi.py via get_neus_in_memory_data().
# Using in-memory arrays avoids encoding images/masks/depths to PNG and decoding
# them back on every keyframe, which dominated prepare_neus_data cost.
_neus_inmem_cache: Dict[str, Dict] = {}


def _ensure_neus_imports() -> None:
    """Add the NeuS third-party directory to sys.path (idempotent)."""
    neus_root = str(Path(__file__).resolve().parent.parent / "third_party" / "instant-nsr-pl")
    if neus_root not in sys.path:
        sys.path.insert(0, neus_root)


def reset_neus_cache() -> None:
    """Drop the cached in-process NeuS system and resume checkpoint."""
    _neus_cache.clear()


def _inmem_key(neus_data_dir) -> str:
    return str(Path(neus_data_dir).resolve())


def get_neus_in_memory_data(neus_data_dir) -> Optional[Dict]:
    """Return keyframe data previously stashed by prepare_neus_data, or None."""
    return _neus_inmem_cache.get(_inmem_key(neus_data_dir))


def clear_neus_in_memory_data(neus_data_dir=None) -> None:
    """Drop cached keyframe arrays. Clears everything if neus_data_dir is None."""
    if neus_data_dir is None:
        _neus_inmem_cache.clear()
    else:
        _neus_inmem_cache.pop(_inmem_key(neus_data_dir), None)



def prepare_neus_data(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    depths: List[np.ndarray],
    extrinsics_o2c: np.ndarray,
    intrinsics: np.ndarray,
    neus_data_dir: Path,
    masks_hand: Optional[List[np.ndarray]] = None,
) -> None:
    """Stash keyframe data in an in-memory cache for the NeuS dataset loader.

    The cached entry — keyed by the absolute ``neus_data_dir`` path — is read by
    ``RobustHOIDatasetBase.setup`` (see third_party/instant-nsr-pl/datasets/
    robust_hoi.py), bypassing PNG encode/decode and disk I/O entirely.

    Args:
        images: List of (H, W, 3) uint8 RGB images for each keyframe.
        masks: List of (H, W) uint8 object masks for each keyframe.
        depths: List of (H, W) float32 depth maps for each keyframe.
        extrinsics_o2c: (N_kf, 4, 4) object-to-camera (w2c) matrices.
        intrinsics: (N_kf, 3, 3) camera intrinsic matrices.
        neus_data_dir: Identity key used by run_neus_training; also created on
            disk so downstream code that inspects the directory still works.
        masks_hand: Optional list of (H, W) uint8 hand masks for each keyframe.
    """
    neus_data_dir = Path(neus_data_dir)
    neus_data_dir.mkdir(parents=True, exist_ok=True)

    n_kf = len(images)
    assert len(masks) == n_kf
    assert len(depths) == n_kf
    assert extrinsics_o2c.shape[0] == n_kf
    assert intrinsics.shape[0] == n_kf

    def _squeeze_mask(m):
        if m is None:
            return None
        return m[:, :, 0] if m.ndim == 3 else m

    _neus_inmem_cache[_inmem_key(neus_data_dir)] = {
        "images": list(images),
        "masks": [_squeeze_mask(m) for m in masks],
        "masks_hand": (
            [_squeeze_mask(m) for m in masks_hand] if masks_hand is not None else None
        ),
        "depths": list(depths),
        "intrinsics": intrinsics.astype(np.float32),  # (N_kf, 3, 3)
        "extrinsics_o2c": extrinsics_o2c[:, :3, :4].astype(np.float32),  # (N_kf, 3, 4) w2c
        "n_kf": n_kf,
    }

    print(f"[NeuS] Cached {n_kf} keyframes in-memory for {neus_data_dir}")


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
    export_only: bool = False,
    mc_resolution: Optional[int] = None,
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
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger  # noqa: E402
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
            f"export.export_vertex_color=true",
            f"checkpoint.every_n_train_steps={max_steps}",
            f"dataset.robust_hoi_weight={robust_hoi_weight}",
            f"dataset.sam3d_weight={sam3d_weight}",
        ]
        if sam3d_root_dir and Path(sam3d_root_dir).exists() and sam3d_weight > 0.0:
            cli_args.append(f"dataset.sam3d_root_dir={sam3d_root_dir}")
        if mc_resolution is not None:
            cli_args.append(f"model.geometry.isosurface.resolution={int(mc_resolution)}")

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

        # Save resolved config alongside checkpoint for later use
        os.makedirs(config.config_dir, exist_ok=True)
        from omegaconf import OmegaConf
        OmegaConf.save(config, os.path.join(config.config_dir, "resolved_config.yaml"))

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
        if sam3d_root_dir and Path(sam3d_root_dir).exists() and sam3d_weight > 0.0:
            config.dataset.sam3d_root_dir = str(sam3d_root_dir)
        if mc_resolution is not None:
            config.model.geometry.isosurface.resolution = int(mc_resolution)
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
    loggers = [
        CSVLogger(config.exp_dir, name=config.trial_name, version="csv_logs"),
        TensorBoardLogger(config.exp_dir, name=config.trial_name, version="tb_logs"),
    ]

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        strategy="auto",
        **config.trainer,
    )

    # ------------------------------------------------------------------
    # Train or export only
    # ------------------------------------------------------------------
    if export_only:
        # Skip training, just load checkpoint and export mesh
        resume_ckpt = _find_latest_checkpoint(output_dir)
        if resume_ckpt is None:
            raise FileNotFoundError(f"[NeuS] No checkpoint found in {output_dir} for export_only mode")
        import torch
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        system.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[NeuS] Export only: loaded checkpoint {resume_ckpt}")
        system.trainer = trainer  # attach trainer so system.print() / system.export() works
        system.cuda()
        system.export()
    else:
        print(f"[NeuS] Training: max_steps={max_steps}, resume={'yes' if resume_ckpt else 'no'}")
        if resume_ckpt:
            trainer.fit(system, datamodule=dm, ckpt_path=resume_ckpt)
        else:
            trainer.fit(system, datamodule=dm)

        # Export mesh
        system.cuda()
        try:
            system.export()
        except (IndexError, RuntimeError) as e:
            print(f"[NeuS] Mesh export failed (SDF may not have converged): {e}")

    # ------------------------------------------------------------------
    # Update cache
    # ------------------------------------------------------------------
    latest_ckpt = _find_latest_checkpoint(output_dir)
    latest_mesh = _find_latest_mesh(output_dir)
    _prune_stale_trial_dirs(output_dir, [latest_ckpt, latest_mesh])
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


def _prune_stale_trial_dirs(output_dir: Path, keep_paths) -> None:
    """Delete trial folders under output_dir other than the ones producing the latest artifacts.

    Each trial lives at ``output_dir/<trial_name>/joint_opt/{ckpt|save|...}`` (see
    run_neus_training), so the trial folder is the grandparent of any .ckpt or
    .obj file. Older trials accumulate when tags or timestamps change across
    runs; keep only the trial(s) that produced the latest checkpoint/mesh.
    """
    output_dir = Path(output_dir).resolve()
    keep_dirs = set()
    for p in keep_paths:
        if p is None:
            continue
        trial_dir = Path(p).resolve().parent.parent.parent  # <trial>/joint_opt/{ckpt|save}/<file>
        if trial_dir == output_dir:
            keep_dirs.add(trial_dir)
    if not keep_dirs:
        return
    for child in output_dir.parent.iterdir():
        if not child.is_dir():
            continue
        if child.resolve() in keep_dirs:
            continue
        # For each stale sibling, preserve its 'save/' subdirectory (exported
        # meshes) and delete every other file/subdirectory inside.
        try:
            for entry in (child / "joint_opt").iterdir():
                if entry.is_dir() and entry.name == "save":
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                print(f"[NeuS] Deleted stale entry: {entry}")
        except OSError as exc:
            print(f"[NeuS] Failed to clean {child}: {exc}")


def _get_checkpoint_global_step(checkpoint_path: Optional[str]) -> Optional[int]:
    """Read the training step from a PyTorch Lightning checkpoint."""
    if checkpoint_path is None:
        return None

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return None

    try:
        import torch

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        global_step = checkpoint.get("global_step")
        if global_step is not None:
            return int(global_step)
    except Exception:
        pass

    match = re.search(r"step=(\d+)", ckpt_path.name)
    if match is not None:
        return int(match.group(1))
    return None


def load_latest_neus_artifacts(sam3d_root_dir: Path) -> Tuple[str, Optional[str], Optional[int]]:
    """Load the latest NeuS checkpoint, exported mesh, and checkpoint step.

    Args:
        sam3d_root_dir: Root directory for one SAM3D sample, e.g. ``.../0001``.

    Returns:
        A tuple of ``(checkpoint_path, mesh_path, checkpoint_global_step)``.

    Raises:
        FileNotFoundError: If no NeuS checkpoint is found under the expected
            ``neus/neus_training/joint_opt`` directory.
    """
    neus_training_dir = Path(sam3d_root_dir) / "neus" / "neus_training" / "joint_opt"
    neus_ckpt = _find_latest_checkpoint(neus_training_dir / "ckpt")
    neus_init_mesh = _find_latest_mesh(neus_training_dir / "save")

    if neus_ckpt is None:
        raise FileNotFoundError(
            f"No NeuS checkpoint found in {neus_training_dir}. "
            "Please run hoi_pipeline_data_preprocess_sam3d_neus.py first to generate the checkpoint."
        )

    ckpt_global_step = _get_checkpoint_global_step(neus_ckpt)
    return neus_ckpt, neus_init_mesh, ckpt_global_step


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
