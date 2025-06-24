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


# 3D assets
# - object_uid

# 3D transform
# Aria Device to Optitrack

# Generic idea around the DataProvider is that is allow to initialize the data reading
# and offer a generic interface to retrieve timestamp data by TYPE (Image, Object, Hand, etc.)

from read_write_model import read_model
from pathlib import Path
class ReconstructProvider:
    """
    High Level interface to retrieve and use data from the hot3d dataset
    """

    def __init__(
        self,
        reconstruct_folder: str,
    ) -> None:
        """
        INIT_DOC_STRING
        """
        # Will read all required metadata
        # Hands
        # Objects
        # Device type, ...
        print("Reading sparse COLMAP reconstruction")
        self.reconstruct_folder = Path(reconstruct_folder)
        self.opti_type = self.reconstruct_folder.name        
        self.scene = self.reconstruct_folder.parent.name
        self.test_name = self.reconstruct_folder.parent.parent.name   
        
        self.cameras, self.images, self.points3D = read_model(reconstruct_folder, ext=".bin")

    def get_test_name(self):
        return self.test_name
    
    def get_scene_name(self):
        return self.scene
    
    def get_opti_type(self):
        return self.opti_type

    def get_camera(self):
        return self.cameras
    
    def get_image(self):
        return self.images
    
    def get_point3D(self):
        return self.points3D
    
    def get_image_pose(self, image):
        # pose from object to camera
        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        tvec = image.tvec
        return quat_xyzw, tvec



