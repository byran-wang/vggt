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
    show_on_mesh_lab: bool = False,
    world_coordinate: str = "object",
    image_plane_distance: float = 1.0,
):
    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(reconstruction_folder):
        raise RuntimeError(
            f"Reconstruction folder {reconstruction_folder} does not exist"
        )


    #
    # Initialize hot3d data provider
    #

    reconstruct_provider = ReconstructProvider(
        reconstruct_folder=Path(reconstruction_folder),
    )

    if show_on_mesh_lab:
        MeshLabVis(reconstruct_provider).show()
        return

    data_provider = DataProvider(
        sequence_folder=sequence_folder,
    )        
    #
    # Initialize the rerun hot3d visualizer interface
    #
    rr_visualizer = Visualizer(data_provider, reconstruct_provider, jpeg_quality, rrd_output_path, world_coordinate=world_coordinate)
    # Log static assets (aka Timeless assets)

    
    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    images = reconstruct_provider.get_image()
    points3D = reconstruct_provider.get_point3D()
    cameras = reconstruct_provider.get_camera()
    point3D_with_conf = reconstruct_provider.get_point3D_with_conf()
    aligned_mesh_path = reconstruct_provider.get_aligned_3D_gen()
    # points = [point.xyz for point in visible_xyzs]
    # point_colors = [point.rgb for point in visible_xyzs]
    # point_errors = [point.error for point in visible_xyzs]
    points = np.array([point.xyz for point in points3D.values()])
    point_colors = np.array([point.rgb for point in points3D.values()])
    point_errors = np.array([point.error for point in points3D.values()])
    rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors), static=True)
    if point3D_with_conf is not None:
        rr.log("points_with_conf", rr.Points3D(point3D_with_conf["points"], colors=point3D_with_conf["conf"]), static=True)
    if aligned_mesh_path is not None:
        mesh = trimesh.load(str(aligned_mesh_path), force="mesh")
        mesh_colors = None
        if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors):
            mesh_colors = mesh.visual.vertex_colors[:, :3]
        rr.log(
            "aligned_mesh",
            rr.Points3D(np.asarray(mesh.vertices), colors=mesh_colors if mesh_colors is not None else [180, 180, 180]),
            static=True,
        )
    rr.log("camera", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward

    for i, image in enumerate(sorted(images.values(), key=lambda im: im.name)):  # type: ignore[no-any-return]
        image_file, frame_idx = data_provider.get_image_file(image.name)

        if image_file is None:
            continue
        rr.set_time_sequence("frame", frame_idx)
        visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]        
        visible_ids = image.point3D_ids[visible]

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]
    

        # points = [point.xyz for point in visible_xyzs]
        # point_colors = [point.rgb for point in visible_xyzs]
        # point_errors = [point.error for point in visible_xyzs]

        # rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))
        quat_xyzw, tvec = reconstruct_provider.get_image_pose(image)
        rr.log(
            "camera/image", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True)
        )
        rr.log(
            f"camera/image_{i}", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True), static=True
        )        
        

        # Log camera intrinsics
        camera = cameras[image.camera_id]
        if camera.model == "SIMPLE_PINHOLE":
            rr.log(
                f"camera/image_{i}",
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=[camera.params[0], camera.params[0]],
                    principal_point=camera.params[1:3],
                    image_plane_distance=image_plane_distance,
                ),
                static=True
            )
            rr.log(
                f"camera/image",
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=[camera.params[0], camera.params[0]],
                    principal_point=camera.params[1:3],
                    image_plane_distance=image_plane_distance,
                ),
                static=False
            )                              
        else:
            raise ValueError(f"Unsupported camera model: {camera.model}")

        rr.log(f"camera/image_{i}", rr.ImageEncoded(path=image_file), static=True)
        rr.log(f"camera/image", rr.ImageEncoded(path=image_file), static=False)
        # rr.log("camera/image", rr.Image(image_file))
        
        rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))
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
        show_on_mesh_lab=args.show_on_mesh_lab,
        world_coordinate=args.world_coordinate,
        image_plane_distance=args.image_plane_distance,
    )



if __name__ == "__main__":
    main()
