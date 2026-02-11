import json
import numpy as np
import trimesh
from pathlib import Path


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    HY_dir = out_dir / "pipeline_hunyuan_omni"
    algin_SAM3D_with_HY_dir = out_dir / "pipeline_algin_SAM3D_with_HY"

    result_dir = out_dir / "pipeline_HY_to_SAM3D"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load the HY mesh
    hy_obj_path = HY_dir / "white_mesh_remesh.obj"
    hy_mesh = trimesh.load(str(hy_obj_path), process=False)
    print(f"Loaded Hunyuan mesh: {len(hy_mesh.vertices)} vertices, {len(hy_mesh.faces)} faces")

    # Load the SAM3D-to-Hunyuan transformation
    transform_path = algin_SAM3D_with_HY_dir / "SAM3D_aligned_with_HY.json"
    with open(transform_path, "r") as f:
        transform_params = json.load(f)
    transform_4x4 = np.array(transform_params["transform_4x4"], dtype=np.float64)
    print(f"Loaded transform from {transform_path}")
    print(f"  scale={transform_params['scale']}, rotation_y_deg={transform_params['rotation_y_deg']}, "
          f"translation={transform_params['translation']}")

    # Transform HY mesh to SAM3D coordinate system (inverse of SAM3D-to-Hunyuan)
    inv_transform = np.linalg.inv(transform_4x4)
    hy_verts = np.array(hy_mesh.vertices, dtype=np.float64)
    sam3d_verts = (inv_transform[:3, :3] @ hy_verts.T).T + inv_transform[:3, 3]

    transformed_mesh = hy_mesh.copy()
    transformed_mesh.vertices = sam3d_verts.astype(np.float32)

    # Save the transformed HY mesh
    out_path = result_dir / "HY_omni_in_sam3d.obj"
    transformed_mesh.export(str(out_path))
    print(f"Saved transformed Hunyuan mesh to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Transform Hunyuan mesh to SAM3D coordinate system")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input data directory (e.g., HO3D_v3/train/GSF13)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    args = parser.parse_args()
    main(args)
