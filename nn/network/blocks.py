import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Useful subnetwork components """
# comment: In a PyTorch tensor with shape (N, C, H, W), dim=1 targets the C (channels) dimension.
# In a TensorFlow tensor with shape (N, H, W, C), axis=-1 targets the C (channels) dimension.

def unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = conv_layer(h, base_channels, 3)
    h1 = conv_layer(h, base_channels, 3)
    h = max_layer(h1)
    h = conv_layer(h, base_channels*2, 3)
    h2 = conv_layer(h, base_channels*2, 3)
    h = max_layer(h2)
    h = conv_layer(h, base_channels*4, 3)
    h3 = conv_layer(h, base_channels*4, 3)
    h = max_layer(h3)
    h = conv_layer(h, base_channels*8, 3)
    h4 = conv_layer(h, base_channels*8, 3)
    if upsamp:
        h = torch.nn.functional.interpolate(h4, size=h3.shape[2:])
        h = conv_layer(h, base_channels*2, 3)
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*4, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h3], dim=-1)
    h = conv_layer(h, base_channels*4, 3)
    h = conv_layer(h, base_channels*4, 3)
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h2.shape[2:])
        h = conv_layer(h, base_channels*2, 3)
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*2, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h2], dim=-1)
    h = conv_layer(h, base_channels*2, 3)
    h = conv_layer(h, base_channels*2, 3)
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h1.shape[2:])
        h = conv_layer(h, base_channels*2, 3)
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels, 3, 2, activation=None, padding="same")
    h = torch.cat([h, h1], dim=-1)
    h = conv_layer(h, base_channels, 3)
    h = conv_layer(h, base_channels, 3)

    h = conv_layer(h, out_channels, 1)
    return h

def conv_layer(inp, out_channels, kernel_size, padding='same', activation=True):
    conv = nn.Conv2d(inp.size(1), out_channels, kernel_size, padding=padding)
    h = conv(inp)
    if activation:
        h = F.relu(h)
    return h

def max_layer(h):
    print("MaxPool input shape",h.shape)
    maxPool_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    h1 = maxPool_layer(h)
    print("h: {}".format(h1.shape))
    return h1


def shallow_unet(inp, base_channels, out_channels, upsamp=True):
    h = inp.permute(0, 3, 1, 2)
    print("h_shape", h.shape)
    h = conv_layer(h, base_channels, 3)
    h1 = conv_layer(h, base_channels, 3)
    h = F.max_pool2d(h1, 2, 2)
    h = conv_layer(h, base_channels*2, 3)
    h2 = conv_layer(h, base_channels*2, 3)
    h = F.max_pool2d(h2, 2, 2)
    h = conv_layer(h, base_channels*4, 3)
    h = conv_layer(h, base_channels*4, 3)
    print("h_shape", h.shape)
    if upsamp:
        h = F.interpolate(h, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h = conv_layer(h, base_channels*2, 3, activation=False)
    else:
        h = nn.ConvTranspose2d(h.size(1), base_channels*2, 3, stride=2, padding=1)(h)
    print("h_shape", h.shape)
    h = torch.cat([h, h2], dim=1)
    h = conv_layer(h, base_channels*2, 3)
    h = conv_layer(h, base_channels*2, 3)
    print("h_shape", h.shape)
    if upsamp:
        h = F.interpolate(h, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h = conv_layer(h, base_channels*2, 3, activation=False)
    else:
        h = nn.ConvTranspose2d(h.size(1), base_channels, 3, stride=2, padding=1)(h)
    print("h_shape", h.shape)
    h = torch.cat([h, h1], dim=1)
    print("h_shape", h.shape)
    h = conv_layer(h, base_channels, 3)
    h = conv_layer(h, base_channels, 3)

    h = nn.Conv2d(h.size(1), out_channels, 1, padding='same')(h)
    print("h_shape", h.shape)
    return h


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = torch.ones([1,10])
    var = torch.nn.Linear(var, 200, activation=torch.nn.Tanh())
    var = torch.nn.Linear(var, np.prod(shape), activation=None)
    var = var.view(shape)
    return var
