# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import os
import numpy as np
import rerun as rr  # @manual
import rerun.blueprint as rrb
from data_provider import DataProvider
from reconstruct_provider import ReconstructProvider
import trimesh
import cv2
import time
import sys
sys.path = ["../"] + sys.path

class Visualizer:
    def __init__(
        self,
        data_provider: DataProvider,
        reconstruct_provider: ReconstructProvider,
        jpeg_quality: int = 75,
        rrd_output_path: Optional[str] = None,
        world_coordinate: str = "object",   
    ) -> None:
        self._data_provider = data_provider
        self._reconstruct_provider = reconstruct_provider
        # To be parametrized later
        self._jpeg_quality = jpeg_quality

        #
        # Prepare the rerun rerun log configuration
        #
        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D", origin="/"),
                rrb.Spatial2DView(name="Camera", origin="/camera/image"),
                column_shares=[10, 5],
            ),
            # rrb.Vertical(
            rrb.Horizontal(
                rrb.Grid(
                    *[rrb.Spatial2DView(name=f"image_{i}", origin=f"/camera/image_{i}") for i in range(40)],
                    grid_columns=10,
                ),
            ),

            # ),            
            # column_shares=[5, 10],                                                      
            # ),
            # rrb.Horizontal(
            #     rrb.Spatial2DView(name="Camera", origin="/camera/image"),
            #     rrb.TimeSeriesView(origin="/plot"),
            # ),
            row_shares=[3, 3],
        )
        
        viewer_name = self._reconstruct_provider.get_test_name() + "_" + self._reconstruct_provider.get_scene_name() + "_" + self._reconstruct_provider.get_opti_type()
        rr.init(viewer_name, spawn=True)
        rr.send_blueprint(blueprint)

        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self.world_coordinate = world_coordinate
        self.world_transform = np.eye(4)  
        
    def log_static_assets(
        self,
    ) -> None:
        """
        Log all static assets (aka Timeless assets)
        - assets that are immutable (but can still move if attached to a 3D Pose)
        """
        # o2c_4x4 = self._reconstruct_provider.get_predict_pose(0)
        if self.world_coordinate == "object":
            obj_mesh_path = self._reconstruct_provider.get_obj_model_latest_file()
            if obj_mesh_path != None:
                self.log_point_cloud("pc/nerf_final", trimesh.load(obj_mesh_path).vertices, color=gray_color, radii=0.001, static=True)


    def log_dynamic_assets(
        self,
        frame_index: int,
        pc_downsample_ratio: float = 1,     
        log_image: bool = True,        
        log_pc_all_depth: bool = False,
        log_pc_obj_depth: bool = True,
        log_pc_hand_depth: bool = False,
        log_pc_obj_coarse: bool = False,
        log_pc_obj_ba: bool = False,
        log_normal_gradient_statistics: bool = False,
        log_hand_fit_intrinsic: bool = False,
        log_hand_fit_trans: bool = False,
        log_hand_fit_rot: bool = False,
        log_hand_fit_pose: bool = False,
        log_hand_fit_all: bool = False,
        log_obj_nerf: bool = True,
        log_camera: bool = True,
        log_normal_map: bool = False,
        log_feature_match: bool = True,
    ) -> None:
        """
        Log dynamic assets:
        I.e assets that are moving, such as:
        - camera poses
        - Image
        - point cloud
        """
        color_mask_overlay = self._data_provider.get_color_mask_overlay(frame_index)
        o2c_4x4 = self._reconstruct_provider.get_final_pose(frame_index)
        c2o_4x4 = np.linalg.inv(o2c_4x4)
        intrins = self._data_provider.get_intrinsics()
        self.c2o_4x4 = c2o_4x4
        self.pc_downsample_ratio = pc_downsample_ratio
        if self.world_coordinate == "camera":
            self.world_transform = o2c_4x4
        nearest_index, obj_mesh_path = self._reconstruct_provider.get_obj_model_nearest_file(frame_index)
        
 

    @staticmethod
    def log_image(label: str, image, jpeg_quality: int = 75, static=False) -> None:
        rr.log(label, rr.Image(image).compress(jpeg_quality=jpeg_quality), static=static)

    @staticmethod
    def log_text(label: str, text: str, static=False) -> None:
        rr.log(label, rr.TextLog(text), static=static)


    @staticmethod
    def log_image_view_coordinates(label: str, static=True) -> None:
        # Log image view coordinates
        rr.log(
            label,
            rr.ViewCoordinates.RDF,
            static=static
        )

    @staticmethod
    def log_scalar(label: str, value: float, static=False) -> None:
        rr.log(label, rr.Scalar(value), static=static)

 