import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
npz_file = np.load('Deep-Learning-Physics-as-Inverse/data/datasets/bouncing/color_bounce_vx8_vy8_sl12_r2.npz')

# Access the arrays
train_x = torch.tensor(npz_file["train_x"], dtype=torch.float32)

# Reshape train_x to have shape (5000, 12, 32, 32, 3)
train_x = train_x.reshape(5000, 12, 32, 32, 3)

# Iterate over samples in the training data
for i in range(train_x.shape[0]):
    print(f"Sample {i+1}:")
    # Iterate over frames in the sample
    for j in range(train_x.shape[1]):
        frame = train_x[i, j]  # Select data for the jth frame of the ith sample
        # Plot the tensor data for the jth frame
        plt.figure(figsize=(4, 4))
        plt.imshow(frame)  # Permute dimensions for correct visualization
        plt.title(f'Frame {j+1}')
        plt.axis('off')
        plt.show()
