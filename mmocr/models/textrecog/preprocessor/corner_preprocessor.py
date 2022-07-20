# Modified from https://github.com/clovaai/deep-text-recognition-benchmark
#
# Licensed under the Apache License, Version 2.0 (the "License");s
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from mmocr.models.builder import PREPROCESSOR
from .base_preprocessor import BasePreprocessor


@PREPROCESSOR.register_module()
class CornerPreprocessor(BasePreprocessor):

    def __init__(self,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.maxCorners = 200
        self.qualityLevel = 0.01
        self.minDistance = 3

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Corner map with size :math:`(N, 1, H, W)`.
        """
        device = batch_img.device
        img_np = batch_img.cpu().numpy()
        batch_corner_map = torch.Tensor()
        for i in range(img_np.shape[0]):
            
            sin_img = img_np[i].transpose(1,2,0) * 255
            gray_img = cv2.cvtColor(sin_img, cv2.COLOR_BGR2GRAY)
            gray_img = np.float32(gray_img)
            img_bg = np.zeros(gray_img.shape, dtype="uint8")
            corners = cv2.goodFeaturesToTrack(gray_img, self.maxCorners, self.qualityLevel, self.minDistance)
            try:
                corners = np.int0(corners)
                for corner in corners:
                    x,y = corner.ravel()
                    # print(x,y)
                    img_bg[y,x] = 1
            except TypeError:
                print('No corner detected!')
            # print("-------------------")

            
            corner_mask = torch.tensor(img_bg).unsqueeze(0).unsqueeze(0)
            corner_mask = corner_mask.to(torch.float32)
            batch_corner_map = torch.cat([batch_corner_map, corner_mask], dim=0)

        batch_corner_map = batch_corner_map.to(device)

        return batch_corner_map