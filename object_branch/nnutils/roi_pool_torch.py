'''
ROI pooling layer implemented by PyTorch native function.
'''
import torch
import torch.nn as nn
from torchvision.ops import roi_pool
import numpy as np


class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        outputs = roi_pool(features, rois, (self.pooled_height, self.pooled_width), self.spatial_scale)
        return outputs
