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

import os
import numpy as np
import re

# 3D assets
# - object_uid

# 3D transform
# Aria Device to Optitrack

# Generic idea around the DataProvider is that is allow to initialize the data reading
# and offer a generic interface to retrieve timestamp data by TYPE (Image, Object, Hand, etc.)


class DataProvider:
    """
    High Level interface to retrieve and use data from the hot3d dataset
    """

    def __init__(
        self,
        sequence_folder: str,
    ) -> None:
        """
        INIT_DOC_STRING
        """
        # Will read all required metadata
        # Hands
        # Objects
        # Device type, ...

        if os.path.exists(sequence_folder):
            self.sequence_folder = sequence_folder
        else:
            raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")

        self.sequence_folder = sequence_folder     

    def get_image_file(self, image_name: str):
        image_file = os.path.join(self.sequence_folder, "images", image_name)        
        if not os.path.exists(image_file):
            return None, None
        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image_name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))        
        return image_file, frame_idx

    

        
