import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Useful subnetwork components """


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

def conv_layer(h, base_channels, out_channels):
    print(type(h))
    conv_layer = nn.Conv2d(in_channels=h.size(1), out_channels=base_channels, kernel_size=3, padding='same')#check h.size
    h = conv_layer(h)
    h = F.relu(h)
    return h

def max_layer(h):
    print("MaxPool input shape",h.shape)
    maxPool_layer = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
    h1 = maxPool_layer(h)
    print("h: {}".format(h1.shape))
    return h1


def shallow_unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    print("h: {}".format(h.shape))
    print("base channels: {}".format(base_channels))
    h = conv_layer(h, base_channels, 3)
    # print("h after first steps: {}".format(h.shape))
    h1 = conv_layer(h, base_channels, 3)
    h = max_layer(h1)
    h = conv_layer(h, base_channels*2, 3)
    h2 = conv_layer(h, base_channels*2, 3)
    h = max_layer(h2)
    h = conv_layer(h, base_channels*4, 3)
    h = conv_layer(h, base_channels*4, 3)
    if upsamp:
        print("UPPPh: {}".format(h.shape))
        h = torch.nn.functional.interpolate(h, size=h2.shape[1:3])
        h = conv_layer(h, base_channels*2, 3)
        print("UPPPh: {}".format(h.shape))
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*2, 3, 2, activation=None, padding="same")
    print("h: {}".format(h.shape))
    print("h2: {}".format(h2.shape))
    h = torch.cat([h, h2], dim=-1)
    h = conv_layer(h, base_channels*2, 3)
    h = conv_layer(h, base_channels*2, 3)
    if upsamp:
        # print("UPPPh: {}".format(h.shape))
        h = torch.nn.functional.interpolate(h, size=h1.shape[1:3])
        h = conv_layer(h, base_channels*2, 3)
        # print("UPPPh: {}".format(h.shape))
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels, 3, 2, padding="same")
    print("h: {}".format(h.shape))
    print("h1: {}".format(h1.shape))
    h = torch.cat([h, h1], dim=-1)
    h = conv_layer(h, base_channels, 3)
    h = conv_layer(h, base_channels, 3)

    h = nn.Conv2d(h.size(-1), out_channels, 1, padding="same")

    return h


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = torch.ones([1,10])
    var = torch.nn.Linear(var, 200, activation=torch.nn.Tanh())
    var = torch.nn.Linear(var, np.prod(shape), activation=None)
    var = var.view(shape)
    return var
