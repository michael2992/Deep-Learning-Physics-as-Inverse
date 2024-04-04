import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms.functional import center_crop
import torch

""" Useful subnetwork components """
# comment: In a PyTorch tensor with shape (N, C, H, W), dim=1 targets the C (channels) dimension.
# In a TensorFlow tensor with shape (N, H, W, C), axis=-1 targets the C (channels) dimension.

class UNet(Module):
    def __init__(self, input, base_channels, out_channels, depth=4): # For shallow Unet, set depth = 3
        super().__init__()

        in_channels = input.shape[1] 
        input_size = input.shape[2]
        self.depth = depth  # Depth of the UNet
        self.maxpool = MaxPool2d(kernel_size=2, stride=2) 

        # Encoder
        self.Encoder_layers = ModuleList([])
        for i in range(depth):
            if i == 0:
                self.Encoder_layers.append(Block(in_channels, base_channels))
            else:
                self.Encoder_layers.append(Block(base_channels * (2**(i-1)), base_channels * (2**i)))

        # Decoder
        self.Decoder_layers = ModuleList([])
        self.UpConv_layers = []
        for i in range(depth-1, 0, -1):
            self.Decoder_layers.append(Block(base_channels*(2**i), base_channels*(2**(i-1))))
            self.UpConv_layers.append(ConvTranspose2d(base_channels*(2**i), base_channels*(2**(i-1)), kernel_size=2, stride=2))

        padding_out = input_size - (2**(depth-1)) * (input_size // 2**(depth-1)) # Padding so the output mask has the same size as the input image, works for kernel=stride=1 
        self.output = Conv2d(base_channels, out_channels, kernel_size=1, padding=padding_out) 

    def forward(self, x):
        # Encoder
        layers_to_copy = [] # Stores layers that need to be cropped and concatenated during decoder part

        for i in range(self.depth):
            x = self.Encoder_layers[i].forward(x)
            layers_to_copy.append(x)
            if i < self.depth-1:
                x = self.maxpool(x)
        
        layers_to_copy.pop() # Remove last layer because we don't need it
        #print("Shape after encoding: {}".format(x.shape))

        # Decoder
        for i in range(self.depth-1):
            x = self.UpConv_layers[i](x)
            x_to_copy = layers_to_copy[-1]  # The last layer added to the list is the first to be concatenated during decoding
            x_to_copy = center_crop(x_to_copy, x.shape[2]) # Crop so it can be concatenated to the current tensor, should have the same HxW as current x after cropping
            x = torch.cat([x, x_to_copy], dim=1)
            x = self.Decoder_layers[i].forward(x)
            layers_to_copy.pop() 
   
        output = self.output(x)
        return output
    
class Block(Module):
    """A block that consists of two convolutional layers and one ReLU activation layer."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

        self.layers = [self.conv1, self.conv2]


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    
"""def unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = torch.nn.Conv2d(h, base_channels, 3, padding="same")
    h = nn.Sequential(h, nn.ReLU())
    h1 = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.MaxPool2d(h1, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    h2 = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.MaxPool2d(h2, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="same")
    h3 = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.MaxPool2d(h3, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*8, 3, activation=torch.nn.ReLU(), padding="same")
    h4 = torch.nn.Conv2d(h, base_channels*8, 3, activation=torch.nn.ReLU(), padding="same")
    if upsamp:
        h = torch.nn.functional.interpolate(h4, size=h3.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="same")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*4, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h3], dim=-1)
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="same")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h2.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="same")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*2, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h2], dim=-1)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h1.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="same")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h1], dim=-1)
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="same")

    h = torch.nn.Conv2d(h, out_channels, 1, activation=None, padding="same")
    return h


def shallow_unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    print("h: {}".format(h.shape))
    print("base channels: {}".format(base_channels))
    h = nn.Conv2d(h.shape[3], base_channels, 3, padding="same")
    h = nn.Sequential(h, nn.ReLU())
    print("h after first steps: {}".format(h.shape))
    h1 = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.MaxPool2d(h1, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    h2 = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.MaxPool2d(h2, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="same")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h2.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="same")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*2, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h2], dim=-1)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="same")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h1.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="same")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h1], dim=-1)
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="same")
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="same")

    h = torch.nn.Conv2d(h, out_channels, 1, activation=None, padding="same")
    return h"""

# Replace with a class
"""def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = torch.ones([1,10])
    var = torch.nn.Linear(var, 200, activation=torch.nn.Tanh())
    var = torch.nn.Linear(var, np.prod(shape), activation=None)
    var = var.view(shape)
    return var"""

class VariableNetwork(Module):
    """Produces a variable from a vector of 1's and improves learning speed of contents and masks."""
    def __init__(self, shape):
        super().__init__()
        var = torch.ones([1,10])
        self.layer1 = nn.Linear(var, 200)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(200, np.prod(shape))

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        output = self.output(x)
        return output
