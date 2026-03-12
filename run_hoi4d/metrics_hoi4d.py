#!/usr/bin/env python3
"""
Evaluate BundleSDF pose results on HOI4D dataset.

Output columns: Sequences | ADD(%) | ADD-S(%) | CD(cm) | F5(%)
Rows sorted alphabetically, final row MEAN.

Usage:
  python run_hoi4d/metrics_hoi4d.py \
    --out_base output_hoi4d \
    --data_base /home/simba-1/Documents/dataset/HOI4D_processed
"""

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh

# ── Path setup so that eval_modules can resolve 'common.metrics' ──────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "FoundationPose"))

from vggt.utils.eval_modules import (   # noqa: E402
    add_err,
    adi_err,
    compute_auc,
    calculate_chamfer_f_scores,
)

# ── HOI4D category mapping ─────────────────────────────────────────────────────
CAT_TO_CADNAME = {
    "C1": "ToyCar", "C2": "Mug",    "C5":  "Bottle",
    "C7": "Bowl",   "C12": "Kettle", "C13": "Knife",
}


# ── HOI4D helpers ──────────────────────────────────────────────────────────────

def parse_hoi4d_sequence(sequence: str):
    """'ZY..._H2_C7_N41_...' → (cat_name='Bowl', obj_key='Bowl041')"""
    parts    = sequence.split("_")
    cat      = next((p for p in parts if re.match(r"^C\d+$", p)), None)
    n_id     = next((p for p in parts if re.match(r"^N\d+$", p)), None)
    if cat is None or n_id is None:
        return None, None
    cat_name = CAT_TO_CADNAME.get(cat, cat)
    obj_key  = f"{cat_name}{int(n_id[1:]):03d}"
    return cat_name, obj_key


def load_model_pts(data_base: Path, sequence: str, n_sample: int = 3000) -> Optional[np.ndarray]:
    """Load and randomly sample canonical model points (metres)."""
    _, obj_key = parse_hoi4d_sequence(sequence)
    if obj_key is None:
        return None
    mesh_path = data_base / "models" / obj_key / "textured_simple.obj"
    if not mesh_path.exists():
        print(f"  [WARN] CAD model not found: {mesh_path}")
        return None
    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        loaded = trimesh.util.concatenate(meshes) if meshes else None
    if not isinstance(loaded, trimesh.Trimesh):
        return None
    verts = np.asarray(loaded.vertices, dtype=np.float64)
    if len(verts) > n_sample:
        idx   = np.random.RandomState(42).choice(len(verts), n_sample, replace=False)
        verts = verts[idx]
    return verts


def build_gt_pose(obj_trans, obj_rot) -> np.ndarray:
    """Reconstruct 4×4 GT pose from objTrans (3,) and objRot Rodrigues (3,)."""
    T = np.eye(4, dtype=np.float64)
    R, _ = cv2.Rodrigues(np.asarray(obj_rot, dtype=np.float64))
    T[:3, :3] = R
    T[:3, 3]  = np.asarray(obj_trans, dtype=np.float64)
    return T


def load_gt_poses(seq_dir: Path) -> dict:
    """Load GT poses from meta/*.pkl. Returns {stem: 4×4 ndarray}."""
    meta_dir = seq_dir / "meta"
    if not meta_dir.exists():
        return {}
    gt = {}
    for pkl_path in sorted(meta_dir.glob("*.pkl")):
        stem = pkl_path.stem
        try:
            with open(pkl_path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            continue
        obj_trans = meta.get("objTrans")
        obj_rot   = meta.get("objRot")
        if obj_trans is None or obj_rot is None:
            continue
        try:
            gt[stem] = build_gt_pose(obj_trans, obj_rot)
        except Exception:
            pass
    return gt


def load_pred_poses(pose_dir: Path) -> dict:
    """Load predicted poses from ob_in_cam/*.txt. Returns {stem: 4×4 ndarray}."""
    pred = {}
    for f in sorted(pose_dir.glob("*.txt")):
        stem = f.stem
        try:
            raw = np.loadtxt(f).astype(np.float64)
            if raw.shape == (4, 4):
                pose = raw
            elif raw.size == 12:
                pose = np.eye(4, dtype=np.float64)
                pose[:3] = raw.reshape(3, 4)
            else:
                continue
            pred[stem] = pose
        except Exception:
            pass
    return pred


def match_frames(pred_map: dict, gt_map: dict):
    """Match frames by stem; fall back to integer normalisation."""
    common = sorted(set(pred_map) & set(gt_map))
    if common:
        return common, pred_map, gt_map

    def int_str(s):
        return str(int(s)) if s.isdigit() else s

    pred_int = {int_str(k): v for k, v in pred_map.items() if k.isdigit()}
    gt_int   = {int_str(k): v for k, v in gt_map.items()   if k.isdigit()}
    common   = sorted(set(pred_int) & set(gt_int), key=lambda x: int(x))
    return common, pred_int, gt_int


# ── Per-sequence evaluation ────────────────────────────────────────────────────

def eval_sequence(seq_name: str, out_base: Path, data_base: Path) -> Optional[dict]:
    out_dir  = out_base / seq_name
    pose_dir = out_dir / "ob_in_cam"
    if not pose_dir.exists():
        print(f"[SKIP] {seq_name}: ob_in_cam not found")
        return None

    seq_dir = data_base / "train" / seq_name
    if not seq_dir.exists():
        print(f"[SKIP] {seq_name}: GT seq dir not found ({seq_dir})")
        return None

    model_pts = load_model_pts(data_base, seq_name)
    if model_pts is None:
        print(f"[SKIP] {seq_name}: CAD model not found")
        return None

    pred_map = load_pred_poses(pose_dir)
    gt_map   = load_gt_poses(seq_dir)
    if not pred_map:
        print(f"[SKIP] {seq_name}: no predicted poses")
        return None
    if not gt_map:
        print(f"[SKIP] {seq_name}: no GT poses with valid annotations")
        return None

    common, pred_map, gt_map = match_frames(pred_map, gt_map)
    if not common:
        print(f"[SKIP] {seq_name}: no matching frame IDs between pred and GT")
        return None

    pred_poses = [pred_map[k] for k in common]
    gt_poses   = [gt_map[k]   for k in common]

    # Align predicted trajectory to GT using the first common frame
    try:
        align_tf = np.linalg.inv(pred_poses[0]) @ gt_poses[0]
    except np.linalg.LinAlgError:
        print(f"[SKIP] {seq_name}: singular pred pose at frame 0")
        return None
    pred_aligned = [p @ align_tf for p in pred_poses]

    add_vals  = []
    adds_vals = []
    cd_vals   = []
    f5_vals   = []

    for pred_p, gt_p in zip(pred_aligned, gt_poses):
        add_vals.append(add_err(pred_p, gt_p, model_pts))
        adds_vals.append(adi_err(pred_p, gt_p, model_pts))
        # Transform canonical pts to camera space for Chamfer
        v_pred = (pred_p[:3, :3] @ model_pts.T).T + pred_p[:3, 3]
        v_gt   = (gt_p[:3, :3]   @ model_pts.T).T + gt_p[:3, 3]
        cd, f5, _ = calculate_chamfer_f_scores(v_pred, v_gt)
        cd_vals.append(cd)
        f5_vals.append(f5 * 100.0)

    add_arr  = np.array(add_vals,  dtype=np.float64)
    adds_arr = np.array(adds_vals, dtype=np.float64)

    return {
        "add_auc":  compute_auc(add_arr)  * 100.0,
        "adds_auc": compute_auc(adds_arr) * 100.0,
        "cd_cm":    float(np.nanmean(cd_vals)),
        "f5":       float(np.nanmean(f5_vals)),
        "n_frames": len(common),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HOI4D BundleSDF evaluation")
    p.add_argument(
        "--out_base", type=str,
        default=str(_PROJECT_ROOT / "output_hoi4d"),
        help="BundleSDF output_hoi4d directory",
    )
    p.add_argument(
        "--data_base", type=str,
        default="/home/simba-1/Documents/dataset/HOI4D_processed",
        help="HOI4D_processed dataset directory",
    )
    p.add_argument(
        "--seqs", type=str, default="",
        help="Comma-separated sequence names; empty = auto-scan out_base",
    )
    p.add_argument(
        "--csv", type=str, default="",
        help="Path to save CSV output; default: <out_base>/hoi4d_metrics.csv",
    )
    return p.parse_args()


def main():
    args      = parse_args()
    out_base  = Path(args.out_base)
    data_base = Path(args.data_base)

    if args.seqs:
        sequences = [s.strip() for s in args.seqs.split(",") if s.strip()]
    else:
        if not out_base.exists():
            print(f"out_base not found: {out_base}")
            sys.exit(1)
        sequences = sorted(
            d.name for d in out_base.iterdir()
            if d.is_dir() and (d / "ob_in_cam").exists()
        )

    if not sequences:
        print(f"No sequences found under {out_base}")
        sys.exit(1)

    print(f"Evaluating {len(sequences)} sequences")
    print(f"  out_base:  {out_base}")
    print(f"  data_base: {data_base}\n")

    rows  = []
    skips = []
    for seq in sequences:
        result = eval_sequence(seq, out_base, data_base)
        if result is None:
            skips.append(seq)
        else:
            rows.append((seq, result))
            print(
                f"  {seq}: "
                f"ADD={result['add_auc']:.2f}%  "
                f"ADD-S={result['adds_auc']:.2f}%  "
                f"CD={result['cd_cm']:.3f}cm  "
                f"F5={result['f5']:.2f}%  "
                f"(n={result['n_frames']})"
            )

    if not rows:
        print("\nNo sequences evaluated successfully.")
        sys.exit(1)

    rows.sort(key=lambda x: x[0])

    add_mean  = float(np.mean([r["add_auc"]  for _, r in rows]))
    adds_mean = float(np.mean([r["adds_auc"] for _, r in rows]))
    cd_mean   = float(np.mean([r["cd_cm"]    for _, r in rows]))
    f5_mean   = float(np.mean([r["f5"]       for _, r in rows]))

    col_w  = max(len(s) for s, _ in rows + [("Sequences", None)]) + 2
    header = (
        f"{'Sequences':<{col_w}}"
        f"{'ADD(%)':>10}  {'ADD-S(%)':>10}  {'CD(cm)':>10}  {'F5(%)':>10}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)
    for seq, r in rows:
        print(
            f"{seq:<{col_w}}"
            f"{r['add_auc']:>10.2f}  "
            f"{r['adds_auc']:>10.2f}  "
            f"{r['cd_cm']:>10.3f}  "
            f"{r['f5']:>10.2f}"
        )
    print(sep)
    print(
        f"{'MEAN':<{col_w}}"
        f"{add_mean:>10.2f}  "
        f"{adds_mean:>10.2f}  "
        f"{cd_mean:>10.3f}  "
        f"{f5_mean:>10.2f}"
    )
    print(sep)

    if skips:
        print(f"\nSkipped ({len(skips)}): {', '.join(skips)}")

    # Write CSV
    csv_path = Path(args.csv) if args.csv else out_base / "hoi4d_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Sequences", "ADD(%)", "ADD-S(%)", "CD(cm)", "F5(%)"])
        for i, (seq, r) in enumerate(rows, 1):
            writer.writerow([
                i,
                seq,
                f"{r['add_auc']:.2f}",
                f"{r['adds_auc']:.2f}",
                f"{r['cd_cm']:.3f}",
                f"{r['f5']:.2f}",
            ])
        writer.writerow([
            "",
            "MEAN",
            f"{add_mean:.2f}",
            f"{adds_mean:.2f}",
            f"{cd_mean:.3f}",
            f"{f5_mean:.2f}",
        ])
    print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
