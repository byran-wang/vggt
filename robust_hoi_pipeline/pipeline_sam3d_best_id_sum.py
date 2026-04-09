import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "utils_simba"))

from utils_simba.logger import get_logger

logger = get_logger(__name__)


def _load_best_id(scene_dir):
    """Load best_id.txt for a scene. Returns str or None."""
    path = scene_dir / "SAM3D_align_filter" / "best_id.txt"
    if not path.exists():
        return None
    return path.read_text().strip()


def _load_score(scene_dir, best_id):
    """Load alignment score for a specific frame from scores.json. Returns float or None."""
    path = scene_dir / "SAM3D_aligned_pts" / "scores.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        scores = json.load(f)
    return scores.get(best_id)


def _load_coverage(scene_dir, best_id):
    """Load face coverage info for a specific frame. Returns (coverage_str, faces_str) or (None, None)."""
    path = scene_dir / "SAM3D_align_filter" / "frame_list_faces_coverage.txt"
    if not path.exists():
        return None, None
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == best_id:
                return parts[1], parts[2]
    return None, None


def main(args):
    dataset_dir = Path(args.dataset_dir)
    sequences = [s.strip() for s in args.sequence_list.split(",") if s.strip()]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"{'Sequence':<12} {'BestID':>6} {'Score':>10} {'Coverage':>8} {'Faces'}"
    sep = "-" * 70
    lines = [header, sep]
    logger.info(header)
    logger.info(sep)

    for seq in sequences:
        scene_dir = dataset_dir / seq
        best_id = _load_best_id(scene_dir)
        if best_id is None:
            logger.warning(f"{seq}: best_id.txt not found, skipping")
            continue

        score = _load_score(scene_dir, best_id)
        coverage, faces = _load_coverage(scene_dir, best_id)

        score_str = f"{score:.6f}" if score is not None else "N/A"
        coverage_str = coverage if coverage else "N/A"
        faces_str = faces if faces else "N/A"

        row = f"{seq:<12} {best_id:>6} {score_str:>10} {coverage_str:>8} {faces_str}"
        lines.append(row)
        logger.info(row)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved summary ({len(sequences)} sequences) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize best frame id across all sequences")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root dataset directory containing sequence subdirectories")
    parser.add_argument("--sequence_list", type=str, required=True,
                        help="Comma-separated list of sequence names")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to write the summary table")
    args = parser.parse_args()
    main(args)
