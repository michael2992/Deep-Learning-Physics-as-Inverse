import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialTransformer(nn.Module):
    def __init__(self, out_size):
        super(SpatialTransformer, self).__init__()
        self.out_height, self.out_width = out_size

    def forward(self, U, theta):
        num_batch = U.shape[0]
        num_channels = U.shape[1]  # Assuming U has shape [N, C, H, W]
        
        # if theta.shape != (num_batch, 2, 3):
        #     # Assuming theta might be flat [N, 6] or incorrectly shaped, reshape it
        #     theta = theta.view(-1, 2, 3)
        # Correcting the affine_grid call:
        # The size parameter should describe the output tensor size as [N, C, H, W]
        output_size = [num_batch, num_channels, self.out_height, self.out_width]
        print(theta.shape, output_size)
        
        # Generating the affine grid
        grid = F.affine_grid(theta, output_size, align_corners=False)
        
        # Applying the grid to the input tensor U
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
