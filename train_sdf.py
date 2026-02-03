import argparse
import os
import sys
import subprocess
from pathlib import Path

from viewer.viewer_step import ObjDataProvider


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDF network from posed images")
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Path to result folder with posed images",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/sdf_train", help="Output directory"
    )
    parser.add_argument("--num_iters", type=int, default=20000, help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples per ray")
    parser.add_argument(
        "--mesh_resolution", type=int, default=128, help="Mesh extraction resolution"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--test_only", action="store_true", help="Run test only (skip training)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument(
        "--sdf_init",
        type=str,
        default="sphere",
        choices=["sphere", "mesh"],
        help="SDF initialization method: 'sphere' (default) or 'mesh' (use gen3d mesh)",
    )
    return parser.parse_args()


def main(args):
    # Load object dataset from ObjDataProvider
    obj_data = ObjDataProvider(Path(args.result_folder))
    seq_name = obj_data.get_seq_name()

    # Setup paths
    instant_nsr_dir = Path(__file__).parent / "third_party" / "instant-nsr-pl"
    config_path = instant_nsr_dir / "configs" / "neus-robust-hoi.yaml"
    output_dir = Path(args.output_dir) / seq_name

    # Build command for instant-nsr-pl launch.py
    launch_script = instant_nsr_dir / "launch.py"
    # Determine init mesh path
    if args.sdf_init == "mesh":
        init_mesh_path = obj_data.gen3d_aligned["gen3d_mesh"]
        if init_mesh_path.exists():
            init_mesh_path = str(init_mesh_path)
            print(f"Using gen3d_mesh (object coords): {init_mesh_path}")
        else:  
            print(
                f"Warning: gen3d_mesh not found at {init_mesh_path}. Please provide a valid init_mesh_path or use 'sphere' initialization."
            )
            return



    # Common arguments
    cmd = [
        sys.executable,
        str(launch_script),
        "--config", str(config_path),
        "--gpu", args.gpu,
        f"dataset.root_dir={args.result_folder}",
        f"model.geometry.sdf_init={args.sdf_init}",  # dataset.sdf_init uses interpolation from this
        # f"trainer.max_steps={args.num_iters}",
        # f"model.train_num_rays={args.batch_size}",
        # f"model.num_samples_per_ray={args.n_samples}",
        # f"model.geometry.isosurface.resolution={args.mesh_resolution}",
        # f"system.optimizer.args.lr={args.lr}",
        f"--exp_dir={output_dir}",
    ]

    # Add init_mesh_path if using mesh initialization
    if args.sdf_init == "mesh" and init_mesh_path:
        cmd.append(f"model.geometry.init_mesh_path={init_mesh_path}")

    if args.test_only:
        # Run test only (evaluate and save mesh)
        cmd.append("--test")
        if args.resume:
            cmd.extend(["--resume", args.resume])
        else:
            # Find latest checkpoint
            ckpt_dir = output_dir / "neus-robust-hoi" / "ckpt"
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("*.ckpt"))
                if ckpts:
                    cmd.extend(["--resume", str(ckpts[-1])])
    else:
        # Train mode
        cmd.append("--train")
        if args.resume:
            cmd.extend(["--resume", args.resume])

    print(f"Running: {' '.join(cmd)}")

    # Change to instant-nsr-pl directory and run
    env = os.environ.copy()
    env["PYTHONPATH"] = str(instant_nsr_dir) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=str(instant_nsr_dir),
        env=env,
    )

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"Training completed. Results saved to {output_dir}")



if __name__ == "__main__":
    args = parse_args()
    main(args)
