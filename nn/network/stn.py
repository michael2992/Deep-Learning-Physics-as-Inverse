import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialTransformer(nn.Module):
    def __init__(self, out_size):
        super(SpatialTransformer, self).__init__()
        self.out_height, self.out_width = out_size

    def forward(self, U, theta):
        num_batch, num_channels, height, width = U.size()

        # Convert theta from [num_batch, 6] to [num_batch, 2, 3]
        theta = theta.view(-1, 2, 3)

        # Generate grid
        grid = F.affine_grid(theta, U.size(), align_corners=False)

        # Sample
        output = F.grid_sample(U, grid, align_corners=False)

        return output

class BatchSpatialTransformer(nn.Module):
    def __init__(self, out_size):
        super(BatchSpatialTransformer, self).__init__()
        self.spatial_transformer = SpatialTransformer(out_size)

    def forward(self, U, thetas):
        num_batch, num_transforms, _ = thetas.size()
        _, num_channels, height, width = U.size()

        # Repeat U for each transformation
        U = U.repeat(1, num_transforms, 1, 1)
        U = U.view(num_batch * num_transforms, num_channels, height, width)

        # Apply transformations
        output = self.spatial_transformer(U, thetas.view(-1, 2, 3))

        return output
