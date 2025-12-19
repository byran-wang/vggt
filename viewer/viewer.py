import argparse
import os
from typing import Optional, Type

import rerun as rr  # @manual
from tqdm import tqdm

from visulizer import Visualizer
from data_provider import DataProvider
from reconstruct_provider import ReconstructProvider
from meshlab_vis import MeshLabVis
import re
from pathlib import Path
import numpy as np
import trimesh

import sys
sys.path.append("../third_party/utils_simba")
from utils_simba.rerun import Visualizer
from utils_simba.vis import quaternion_to_rotation_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="path to data sequence",
        required=True,
    )
    parser.add_argument(
        "--reconstruction_folder",
        type=str,
        help="path to reconstruction folder",
        required=True,
    )
    parser.add_argument(
        "--show_on_mesh_lab",
        action="store_true",
        help="show the mesh on mesh lab",
    )

    parser.add_argument("--jpeg_quality", type=int, default=30, help=argparse.SUPPRESS)

    # If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument(
        "--rrd_output_path", type=str, default=None, help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--world_coordinate", type=str, default="object", help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--image_plane_distance", type=float, default=0.05, help=argparse.SUPPRESS
    )


    return parser.parse_args()

def execute_rerun(
    sequence_folder: str,
    reconstruction_folder: str,
    rrd_output_path: Optional[str],
    jpeg_quality: int,
    timestamps_slice: Type[slice],
    world_coordinate: str = "object",
    image_plane_distance: float = 1.0,
):
    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(reconstruction_folder):
        raise RuntimeError(
            f"Reconstruction folder {reconstruction_folder} does not exist"
        )

    reconstruct_provider = ReconstructProvider(
        reconstruct_folder=Path(reconstruction_folder),
    )


    data_provider = DataProvider(
        sequence_folder=sequence_folder,
    )
    viewer_name = reconstruct_provider.get_test_name() + "_" + reconstruct_provider.get_scene_name() + "_" + reconstruct_provider.get_opti_type()
    visualizer = Visualizer(viewer_name)   

    # Loop over the timestamps of the sequence and visualize corresponding data
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    images = reconstruct_provider.get_image()
    points3D = reconstruct_provider.get_point3D()
    cameras = reconstruct_provider.get_camera()
    point3D_with_conf = reconstruct_provider.get_point3D_with_conf()
    aligned_mesh_path = reconstruct_provider.get_aligned_3D_gen()

    points = np.array([point.xyz for point in points3D.values()])
    point_colors = np.array([point.rgb for point in points3D.values()])
    point_errors = np.array([point.error for point in points3D.values()])
    visualizer.log_points("points", points, point_colors, point_errors, static=True)

    if point3D_with_conf is not None:
        visualizer.log_points("points_with_conf", point3D_with_conf["points"], point3D_with_conf["conf"], static=True)
    if aligned_mesh_path is not None:
        visualizer.log_mesh("aligned_mesh", str(aligned_mesh_path), static=True)


    for i, image in enumerate(sorted(images.values(), key=lambda im: im.name)):  # type: ignore[no-any-return]
        image_file, frame_idx = data_provider.get_image_file(image.name)

        if image_file is None:
            continue
        rr.set_time_sequence("frame", frame_idx)
        visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]        

        visible_xys = image.xys[visible]

        quat_xyzw, tvec = reconstruct_provider.get_image_pose(image)
        R = quaternion_to_rotation_matrix(quat_xyzw)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = tvec
        visualizer.log_cam_pose(f"camera/image_{i}", c2w, static=True)

        camera = cameras[image.camera_id]

        if camera.model != "SIMPLE_PINHOLE":
            raise ValueError(f"Unsupported camera model: {camera.model}")

        visualizer.log_calibration(
            f"camera/image_{i}",
            resolution=[camera.width, camera.height],
            intrins=np.array(
                [
                    [camera.params[0], 0, camera.params[1]],
                    [0, camera.params[0], camera.params[2]],
                    [0, 0, 1],
                ]
            ),
            image_plane_distance=image_plane_distance,
            static=True,
        )
        visualizer.log_image(                            
            f"camera/image_{i}",
            image_file,
            static=True
        )
        
        rr.log(f"camera/image_{i}/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))

def main():
    args = parse_args()
    print(f"args provided: {args}")

    execute_rerun(
        sequence_folder=args.sequence_folder,
        reconstruction_folder=args.reconstruction_folder,
        rrd_output_path=args.rrd_output_path,
        jpeg_quality=args.jpeg_quality,
        timestamps_slice=slice(None, None, None),
        world_coordinate=args.world_coordinate,
        image_plane_distance=args.image_plane_distance,
    )



if __name__ == "__main__":
    main()
