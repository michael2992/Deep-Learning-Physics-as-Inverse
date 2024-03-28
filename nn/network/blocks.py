import numpy as np
import torch
import torch.nn as nn

""" Useful subnetwork components """


def unet(inp, base_channels, out_channels, upsamp=True):
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
    return h


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = torch.ones([1,10])
    var = torch.nn.Linear(var, 200, activation=torch.nn.Tanh())
    var = torch.nn.Linear(var, np.prod(shape), activation=None)
    var = var.view(shape)
    return var
